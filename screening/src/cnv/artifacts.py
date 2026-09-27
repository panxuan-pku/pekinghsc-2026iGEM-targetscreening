"""Small, file-based lifecycle and provenance record for CNV workflow stages."""
import copy
import hashlib
import json
import time
import uuid
from pathlib import Path

from .infercnv import InsufficientDataError


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StageRun:
    """Archive owned outputs, record an execution, and verify consumed artifacts.

    Not a transaction or concurrent-writer lock. Historical files are moved,
    never deleted. Records are immutable; run_<stage>.json is the latest status.
    """
    def __init__(self, cfg, stage, outputs, parameters):
        self.outdir = Path(cfg.get("output", {}).get("dir", "outputs/cnv"))
        self.outputs = [Path(p) for p in outputs]
        self.stage = stage
        self.audit = self.outdir / "audit" / f"run_{stage}.json"
        self.data = {"schema_version": 1, "execution_id": uuid.uuid4().hex,
                     "stage": stage, "project": cfg.get("project"), "status": "running",
                     "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "config": copy.deepcopy(cfg), "parameters": copy.deepcopy(parameters),
                     "inputs": {}, "upstream": {}, "outputs": {}}
        self.history = self.outdir / "history" / self.data["execution_id"]
        # Reject aliases before moving anything, including a source used as an output.
        sources = [Path(v) for k, v in cfg.get("annotation", {}).items()
                   if k in ("gtf", "gene_order_tsv") and v]
        if stage == "prepare_order":
            sources = [Path(cfg["annotation"]["gtf"])]
        for sample in cfg.get("samples", []):
            if sample.get("path"):
                sources.extend(Path(sample["path"]) / (n + suffix)
                               for n in ("matrix.mtx", "barcodes.tsv", "genes.tsv", "features.tsv")
                               for suffix in ("", ".gz"))
        if cfg.get("_config_path"):
            sources.append(Path(cfg["_config_path"]))
        self.sources = {p.resolve() for p in sources}
        resolved = [p.resolve() for p in self.outputs]
        if len(set(resolved)) != len(resolved) or self.sources.intersection(resolved):
            raise ValueError("stage outputs overlap inputs or each other; use separate paths")

    def _archive(self, paths, kind):
        existing = [p for p in paths if p.exists() or p.is_symlink()]
        for path in existing:
            if path.is_dir() and not path.is_symlink():
                raise ValueError(f"expected an output file, not directory: {path}")
        for path in existing:
            target = self.history / kind / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                raise ValueError(f"archive destination already exists: {target}")
            path.rename(target)

    def __enter__(self):
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.audit.parent.mkdir(parents=True, exist_ok=True)
        self._archive(self.outputs + [self.audit], "previous")
        self._save()
        return self

    def _save(self, final=False):
        text = json.dumps(self.data, indent=2, default=str)
        temporary = self.audit.with_suffix(".tmp")
        temporary.write_text(text)
        temporary.replace(self.audit)
        if final:
            record = self.audit.parent / "records" / f"{self.data['execution_id']}.json"
            record.parent.mkdir(exist_ok=True)
            with record.open("x") as f:
                f.write(text)

    def input(self, name, path, producer=None, parameters=None):
        path = Path(path)
        entry = {"path": str(path.resolve()), "sha256": sha256_file(path)}
        if name in self.data["inputs"] and self.data["inputs"][name] != entry:
            raise ValueError(f"ambiguous input name: {name}")
        self.data["inputs"][name] = entry
        if producer:
            parent = self.upstream(producer, parameters)
            artifact = parent.get("outputs", {}).get(path.name)
            if artifact != entry:
                raise ValueError(f"upstream path/checksum mismatch: {path}; rerun {producer}")
        return entry

    def upstream(self, producer, parameters=None):
        audit = self.audit.parent / f"run_{producer}.json"
        if not audit.is_file():
            raise ValueError(f"missing upstream audit: {audit}; rerun {producer}")
        parent = json.loads(audit.read_text())
        if (parent.get("schema_version") != 1 or parent.get("stage") != producer
                or parent.get("status") != "success" or not parent.get("execution_id")):
            raise ValueError(f"unverified or unsuccessful upstream: {audit}; rerun {producer}")
        if producer != "prepare_order" and parent.get("project") != self.data["project"]:
            raise ValueError(f"upstream project mismatch: {audit}")
        if parameters is not None and parent.get("parameters") != parameters:
            raise ValueError(f"upstream parameters mismatch: {audit}; rerun {producer}")
        record = self.audit.parent / "records" / f"{parent['execution_id']}.json"
        if not record.is_file() or json.loads(record.read_text()) != parent:
            raise ValueError(f"upstream execution record mismatch: {audit}")
        self.data["upstream"][producer] = {"execution_id": parent["execution_id"],
                                           "record": str(record.resolve()),
                                           "sha256": sha256_file(record)}
        return parent

    def skip_output(self, path):
        """A failed optional writer may have left a partial file; keep it as history."""
        self._archive([Path(path)], "incomplete")

    def invalidate(self, stage, outputs):
        """Do not leave downstream results current while rerunning their producer."""
        audit = self.audit.parent / f"run_{stage}.json"
        if not any(p.exists() or p.is_symlink() for p in list(outputs) + [audit]):
            return
        if any(p.resolve() in self.sources for p in outputs):
            raise ValueError("downstream outputs overlap input files; use separate paths")
        self._archive(list(outputs) + [audit], f"invalidated_{stage}")
        audit.write_text(json.dumps({"stage": stage, "status": "invalidated",
                                     "invalidated_by": self.data["execution_id"],
                                     "reason": f"{self.stage} is being rerun"}, indent=2))
        self.data.setdefault("invalidated_stages", []).append(stage)

    def __exit__(self, exc_type, exc, traceback):
        error = exc
        if error is None:
            try:
                for entry in self.data["inputs"].values():
                    if sha256_file(entry["path"]) != entry["sha256"]:
                        raise ValueError(f"input changed during execution: {entry['path']}")
                for stage, parent in self.data["upstream"].items():
                    latest = json.loads((self.audit.parent / f"run_{stage}.json").read_text())
                    if latest.get("execution_id") != parent["execution_id"]:
                        raise ValueError(f"upstream changed during execution: {stage}")
                self.data["outputs"] = {p.name: {"path": str(p.resolve()), "sha256": sha256_file(p)}
                                        for p in self.outputs if p.is_file()}
            except Exception as failure:
                error = failure
        self.data["status"] = "failed" if error else "success"
        if error:
            self.data["validation_status"] = "INSUFFICIENT_DATA" if isinstance(error, InsufficientDataError) else "ERROR"
            self.data["error"] = f"{type(error).__name__}: {error}"
            self.data["outputs"] = {}
            self._archive(self.outputs, "incomplete")
        self._save(final=True)
        if error is not None and exc is None:
            raise error
        return False
