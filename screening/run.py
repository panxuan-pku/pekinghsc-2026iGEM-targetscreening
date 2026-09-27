#!/usr/bin/env python3
"""One merged pipeline: deletion interval or gene list → ranking → design annotations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from screening.download import download, sha256
from screening.prepare_input import load_prepared_input
from screening.report import INPUT_TYPES, output_names, report, write_summary

ROOT = Path(__file__).resolve().parents[1]

from screening.src.screening import CONFIG, add_impc, load_config, merge_core, rank_candidates, select_modules
from screening.src.screening_inputs import build_candidates, transcript_design
from screening.src.opentargets import annotate_snapshot, fetch_snapshot


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_references(directory):
    lock = json.loads((directory / "references.json").read_text())
    if lock.get("status") != "complete":
        raise ValueError("reference preparation is incomplete")
    for name, expected in load_config()["references"].items():
        recorded = lock["files"][name]
        if any(recorded[k] != expected[k] for k in ("url", "version")):
            raise ValueError(f"reference version/source mismatch: {name}")
        if sha256(directory / name) != recorded["sha256"]:
            raise ValueError(f"reference checksum mismatch: {name}")
    return lock


def prepare(directory):
    print("\n=== Reference preparation ===", flush=True)
    print(f"[INFO] Directory: {directory}", flush=True)
    if directory.exists():
        # A failed download can resume only if its completed files are unchanged.
        lock = json.loads((directory / "references.json").read_text())
    else:
        directory.mkdir(parents=True)
        lock = {"created_utc": now(), "status": "preparing", "files": {}}
        write_json(directory / "references.json", lock)
    sources = load_config()["references"]
    for index, (name, source) in enumerate(sources.items(), 1):
        if name in lock["files"]:
            record = lock["files"][name]
            if any(record[k] != source[k] for k in ("url", "version")) or sha256(directory / name) != record["sha256"]:
                raise ValueError(f"existing reference changed; use a new directory: {name}")
            print(f"[OK] {index}/{len(sources)} {name}: checksum verified; reusing local file", flush=True)
            continue
        print(f"[INFO] {index}/{len(sources)} Downloading {name} ({source['version']})", flush=True)
        lock["files"][name] = {**download(source["url"], directory / name, progress=True), "version": source["version"]}
        write_json(directory / "references.json", lock)
        print(f"[OK] {name}: downloaded and checksum recorded", flush=True)
    lock["status"] = "complete"
    write_json(directory / "references.json", lock)
    verify_references(directory)
    print(f"\n[OK] Reference preparation complete ({len(sources)}/{len(sources)}). Screening has not run.")
    print(f"  References: {directory.resolve()}")
    print(f"  Reference manifest: {directory / 'references.json'}")


def code_provenance():
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    paths = [Path(__file__).resolve(), ROOT / "screening/download.py",
             ROOT / "screening/prepare_input.py", ROOT / "screening/report.py", CONFIG]
    paths += sorted((ROOT / "screening/src").rglob("*.py"))
    return {"git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "file_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in paths},
            "python": sys.version, "packages": {p: importlib.metadata.version(p) for p in ("pandas", "numpy", "PyYAML", "requests")}}


def run(args):
    config = load_config()
    if args.module_budget <= 0 or args.module_size <= 0:
        raise ValueError("module budget and size must be positive")
    prepared = load_prepared_input(args.input_file) if args.input_file else None
    if prepared:
        args.interval = prepared.get("interval")
        args.genes = prepared.get("genes")
        if args.input_source == "user_supplied; source_not_verified":
            args.input_source = prepared["source"]
    references = args.references.resolve()
    print("\n=== Candidate prioritization ===", flush=True)
    print(f"[INFO] Verifying references: {references}", flush=True)
    lock = verify_references(references)
    print("[OK] Reference checksums verified", flush=True)
    genes = args.genes
    if args.gene_list:
        genes = [s.strip() for s in args.gene_list.read_text().splitlines() if s.strip()]
    input_type = "interval" if args.interval else "gene_list"
    names = output_names(input_type)
    output = (args.output if args.output is not None else
              ROOT / "workspace/screening/results" / f"{input_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}").resolve()
    print(f"[INFO] Input type: {INPUT_TYPES[input_type]} ({input_type})", flush=True)
    print(f"[INFO] Output directory: {output}", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "started_utc": now(), "disease_id": args.disease_id,
                "phenotype_ids": args.phenotype_id, "reference_directory": str(references),
                "references": lock, "code": code_provenance(),
                "outputs": names,
                "input": {"input_type": input_type, "interval": args.interval, "genes": genes, "source": args.input_source,
                          "genome_build": "GRCh38", "coordinate_system": "1-based-inclusive"},
                "module_budget_bp": args.module_budget, "assumed_module_bp": args.module_size,
                "optional_inputs": {}}
    if prepared:
        manifest["input"]["preparation"] = prepared
    for label, path in (("input_file", args.input_file), ("gene_list", args.gene_list),
                        ("impc_evidence", args.impc_evidence), ("ot_snapshot", args.ot_snapshot)):
        if path:
            manifest["optional_inputs"][label] = {"path": str(path.resolve()), "sha256": sha256(path)}
    write_json(output / names["manifest.json"], manifest)
    (output / names["config.yaml"]).write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    try:
        print("[INFO] 1/4 Resolving candidate IDs, positions and transcripts", flush=True)
        frame = build_candidates(references / "annotation.gtf.gz", references / "hgnc.tsv", interval=args.interval, genes=genes)
        frame = transcript_design(references / "annotation.gtf.gz", frame)
        frame.to_csv(output / names["candidates.csv"], index=False, encoding="utf-8-sig")
        print(f"[OK] Identified {len(frame)} candidate genes", flush=True)
        print("[INFO] 2/4 Integrating versioned evidence", flush=True)
        frame = merge_core(frame, references)
        if args.impc_evidence:
            frame = add_impc(frame, args.impc_evidence)
        print("[INFO] 3/4 Querying or replaying OT annotations (does not change ranking)", flush=True)
        snapshot = json.loads(args.ot_snapshot.read_text()) if args.ot_snapshot else fetch_snapshot(
            frame.gene_id.tolist(), args.disease_id, args.phenotype_id, config["ot_release"])
        write_json(output / names["ot_snapshot.json"], snapshot)
        frame = annotate_snapshot(frame, snapshot, args.disease_id, args.phenotype_id, config["ot_release"])
        print("[INFO] 4/4 Ranking and applying the 30 bp design threshold and module budget", flush=True)
        ranked, _ = rank_candidates(frame)
        ranked = select_modules(ranked, args.module_budget, args.module_size)
        if len(ranked) != len(frame) or set(ranked.gene_id) != set(frame.gene_id):
            raise ValueError("ranker did not preserve the candidate set")
        ranked.to_csv(output / names["ranked.csv"], index=False, encoding="utf-8-sig")
        write_summary(output, ranked, input_type)
        manifest.update(status="completed_with_annotation_gaps" if ranked.ot_status.eq("query_failed").any() else "complete",
                        completed_utc=now(), candidate_gene_ids=sorted(frame.gene_id),
                        ot_status_counts=ranked.ot_status.value_counts().to_dict(),
                        output_sha256={p.name: sha256(p) for p in (output / names["summary.csv"], output / names["ranked.csv"], output / names["candidates.csv"], output / names["ot_snapshot.json"], output / names["config.yaml"])})
        report(output, ranked, manifest)
        write_json(output / names["manifest.json"], manifest)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc), ended_utc=now())
        write_json(output / names["manifest.json"], manifest)
        raise
    if manifest["status"] == "complete":
        print("\n[OK] Screening complete", flush=True)
    else:
        gaps = int(ranked.ot_status.eq("query_failed").sum())
        print(f"\n[WARN] Ranking generated; OT queries failed for {gaps} candidates. Review the report.", flush=True)
    print(f"  Candidates: {len(ranked)}; selected for design: {int(ranked.selected.sum())} (not experimentally validated)")
    print(f"  Input type: {INPUT_TYPES[input_type]} ({input_type})")
    print(f"  Output directory: {output}")
    print(f"  Run status: {manifest['status']}")
    print(f"  HTML report: {output / names['report.html']}")
    print(f"  Summary table: {output / names['summary.csv']}")
    print(f"  Full evidence: {output / names['ranked.csv']}")
    print(f"  Run manifest: {output / names['manifest.json']}")


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="download and lock the reference bundle")
    prep.add_argument("--references", type=Path, default=ROOT / "workspace/screening/references",
                      help="reference directory (default: repository/workspace/screening/references)")
    screen = sub.add_parser("run", help="rank a supplied interval or gene list")
    screen.add_argument("--references", type=Path, default=ROOT / "workspace/screening/references",
                        help="same directory used by prepare (default: repository/workspace/screening/references)")
    inputs = screen.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--interval", help="GRCh38, 1-based inclusive, e.g. chr4:start-end; no commas")
    inputs.add_argument("--genes", nargs="+", help="HGNC IDs, Ensembl gene IDs or symbols")
    inputs.add_argument("--gene-list", type=Path, help="one gene per line, no header")
    inputs.add_argument("--input-file", type=Path, help="JSON from screening.prepare_input (genes or interval)")
    screen.add_argument("--input-source", default="user_supplied; source_not_verified")
    screen.add_argument("--disease-id", default=config["disease_id"])
    screen.add_argument("--phenotype-id", action="append", default=[])
    screen.add_argument("--ot-snapshot", type=Path)
    screen.add_argument("--impc-evidence", type=Path)
    screen.add_argument("--module-budget", type=int, default=config["module_budget_bp"])
    screen.add_argument("--module-size", type=int, default=config["assumed_module_bp"])
    screen.add_argument("--output", type=Path, help="new output directory; default: workspace/screening/results/<input_type>_<timestamp>")
    args = parser.parse_args()
    try:
        prepare(args.references.resolve()) if args.command == "prepare" else run(args)
    except Exception as exc:
        parser.exit(1, f"[FAIL] {'Reference preparation stopped' if args.command == 'prepare' else 'Screening stopped'}: {exc}\n")


if __name__ == "__main__":
    main()
