#!/usr/bin/env python3
"""CLI orchestrator: scRNA-seq → CNV intervals → candidate gene list for the L1 pipeline.

Subcommands (run from the pipeline root, screening/):

  stage-samples   link raw GEO files into per-sample 10x dirs
  prepare-order   GTF → gene order table (data/gencode_gene_order.tsv)
  infercnv        load samples → QC → normalize → infercnvpy → window signals + h5ad
  call-segments   window signals → deletion segment tables (per resolution)
  extract-genes   segments + known clinical intervals → candidates.csv + expression QC + report
  all             prepare-order → infercnv → call-segments → extract-genes

Outputs land under config['output']['dir'] (default outputs/cnv/).
"""
import argparse
import gzip
import json
import sys
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .expression_qc import interval_gene_qc
from .artifacts import StageRun, sha256_file
from .gene_order import genes_in_interval, parse_gtf_genes
from .infercnv import validate_sample_specs
from .segments import SEGMENT_COLUMNS, call_deletion_segments, filter_segments, segment_overlap_frac


# ------------------------------------------------------------------ helpers
def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a mapping")
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def _outdir(cfg):
    d = Path(cfg.get("output", {}).get("dir", "outputs/cnv"))
    (d / "audit").mkdir(parents=True, exist_ok=True)
    return d


def _resolutions(cfg):
    resolutions = cfg.get("infercnv", {}).get(
        "resolutions", [{"name": "standard", "window_size": 100, "step": 10}])
    names = [r["name"] for r in resolutions]
    if not names or len(set(names)) != len(names) or any(
            not isinstance(n, str) or not n or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in n)
            for n in names):
        raise ValueError("resolution names must be unique nonempty filename-safe names")
    return resolutions


def _stage_parameters(cfg, stage):
    ann = cfg.get("annotation", {})
    if stage == "prepare_order":
        return {"genome_build": ann.get("genome_build"), "coordinate_system": "1-based-inclusive",
                "gene_types": ["protein_coding"]}
    if stage == "infercnv":
        cnv, qc = cfg.get("infercnv", {}), cfg.get("qc", {})
        return {"samples": [{"id": s["id"], "group": s["group"], "path": str(Path(s["path"]).resolve())}
                            for s in cfg.get("samples", [])],
                "genome_build": ann.get("genome_build"),
                "gene_order": str(Path(ann["gene_order_tsv"]).resolve()),
                "qc": {k: qc.get(k, v) for k, v in {"min_genes_per_cell": 200,
                       "max_mito_pct": 20.0, "downsample_per_sample": None, "seed": 0}.items()},
                "reference_group": cnv.get("reference_group", "reference"),
                "patient_group": cnv.get("patient_group", "patient"),
                "n_jobs": cnv.get("n_jobs"),
                "resolutions": _resolutions(cfg),
                "exclude_chromosomes": cnv.get("exclude_chromosomes", ["chrX", "chrY"])}
    if stage == "call_segments":
        seg = cfg.get("segments", {})
        return {"z_threshold": seg.get("z_threshold", 1.5), "min_windows": seg.get("min_windows", 3),
                "max_segment_mb": seg.get("max_segment_mb") or None, "resolutions": _resolutions(cfg)}
    extract = cfg.get("extract", {})
    return {"mode": extract.get("mode", "cnv"),
            "include_auto_in_candidates": extract.get("include_auto_in_candidates", False),
            "known_intervals": cfg.get("known_intervals", {}) or {},
            "genome_build": ann.get("genome_build"), "coordinate_system": "1-based-inclusive",
            "min_overlap_frac": cfg.get("validation", {}).get("min_overlap_frac", 0.5),
            "patient_group": cfg.get("infercnv", {}).get("patient_group", "patient"),
            "reference_group": cfg.get("infercnv", {}).get("reference_group", "reference")}


CANDIDATE_COLUMNS = ["gene_symbol", "source", "deletion_id", "chrom", "start", "end"]
PROVENANCE_COLUMNS = CANDIDATE_COLUMNS + ["gene_id", "strand", "genome_build",
                                         "interval_chrom", "interval_start", "interval_end"]
QC_COLUMNS = ["gene_name", "found", "mean_counts_patient", "mean_counts_ref",
              "pct_expr_patient", "pct_expr_ref", "log2fc_patient_vs_ref", "chrom", "start", "end"]
VALIDATION_COLUMNS = ["interval", "n_genes", "n_genes_in_expression", "mean_log2fc_interval_genes",
                      "auto_overlap_frac", "AC-CNV-1 (auto recovers known interval)", "markers",
                      "markers_in_expression", "AC-CNV-2 (markers detected in data)", "reason"]
EXTRACT_OUTPUTS = ["candidates.csv", "candidates_all.csv", "auto_segment_genes.csv",
                   "candidate_provenance.csv", "interval_gene_expression_qc.csv",
                   "validation_known_intervals.csv", "cnv_report.md"]


def _audited_stage(stage):
    def decorate(function):
        @wraps(function)
        def execute(cfg):
            if stage == "infercnv":
                validate_sample_specs(cfg.get("samples"))
            outdir = Path(cfg.get("output", {}).get("dir", "outputs/cnv"))
            if stage == "prepare_order":
                outputs = [Path(cfg["annotation"]["gene_order_tsv"])]
            elif stage == "infercnv":
                outputs = [outdir / name for name in ["cnv_input.h5ad", "gene_position_mapping.csv", "heatmap_patient.png"] +
                           [f"window_signal_{r['name']}.tsv" for r in _resolutions(cfg)]]
            elif stage == "call_segments":
                outputs = [outdir / name for name in ["segments_all.csv"] +
                           [f"segments_{r['name']}.csv" for r in _resolutions(cfg)]]
            else:
                outputs = [outdir / name for name in EXTRACT_OUTPUTS]
            if stage in ("infercnv", "call_segments"):
                audit = outdir / "audit" / f"run_{stage}.json"
                previous = json.loads(audit.read_text()) if audit.is_file() else {}
                prefix, suffix = ("window_signal_", ".tsv") if stage == "infercnv" else ("segments_", ".csv")
                outputs += [outdir / n for n in previous.get("outputs", {})
                            if Path(n).name == n and n.startswith(prefix) and n.endswith(suffix)
                            and outdir / n not in outputs]
            with StageRun(cfg, stage, outputs, _stage_parameters(cfg, stage)) as run:
                if cfg.get("_config_path"):
                    run.input("config_file", cfg["_config_path"])
                if stage in ("prepare_order", "infercnv", "call_segments"):
                    downstream = {"prepare_order": ["infercnv", "call_segments", "extract_genes"],
                                  "infercnv": ["call_segments", "extract_genes"],
                                  "call_segments": ["extract_genes"]}[stage]
                    for child in downstream:
                        audit = outdir / "audit" / f"run_{child}.json"
                        old = json.loads(audit.read_text()) if audit.is_file() else {}
                        # Independent interval-only output does not depend on CNV.
                        if (child == "extract_genes" and stage != "prepare_order"
                                and old.get("parameters", {}).get("mode") == "intervals-only"):
                            continue
                        if child == "extract_genes":
                            names = EXTRACT_OUTPUTS
                        else:
                            names = (["cnv_input.h5ad", "gene_position_mapping.csv", "heatmap_patient.png"] if child == "infercnv"
                                     else ["segments_all.csv"])
                            prefix, suffix = ("window_signal_", ".tsv") if child == "infercnv" else ("segments_", ".csv")
                            names += [f"{prefix}{r['name']}{suffix}" for r in _resolutions(cfg)]
                            names += [n for n in old.get("outputs", {})
                                      if Path(n).name == n and n.startswith(prefix) and n.endswith(suffix)]
                        # A nested prepare-order has no completed current caller output yet.
                        if old.get("status") == "running":
                            continue
                        run.invalidate(child, [outdir / n for n in sorted(set(names))])
                return function(cfg, run)
        return execute
    return decorate


# ------------------------------------------------------------------ stages
def cmd_stage_samples(cfg):
    """Link raw GEO per-sample files (GSMxxx_SAMPLE_matrix.mtx.gz ...) into
    per-sample 10x dirs, checking every sample before writing any files."""
    validate_sample_specs(cfg.get("samples"))
    plans = []
    destinations = set()
    for s in cfg["samples"]:
        raw_dir = Path(s["raw_dir"])
        prefix = s["raw_prefix"]
        dest = Path(s["path"])
        destinations.add(dest.resolve())
        for directory in (dest, *dest.parents):
            if (directory.exists() or directory.is_symlink()) and not directory.is_dir():
                raise ValueError(f"sample directory conflicts with existing path: {directory}")
        links = []
        for src_name, dst_name in [
            (f"{prefix}_matrix.mtx.gz", "matrix.mtx.gz"),
            (f"{prefix}_barcodes.tsv.gz", "barcodes.tsv.gz"),
            (f"{prefix}_genes.tsv.gz", "genes.tsv.gz"),
        ]:
            src, dst = raw_dir / src_name, dest / dst_name
            if not src.is_file():
                raise FileNotFoundError(f"missing raw file or not a regular file: {src}")
            if dst.exists() or dst.is_symlink():
                if not dst.is_symlink() or dst.resolve() != src.resolve():
                    raise ValueError(f"sample file conflict: {dst}; expected a link to {src.resolve()}; "
                                     "use a new sample path or inspect the existing file; nothing overwritten")
            else:
                links.append((src.resolve(), dst))
        genes = raw_dir / f"{prefix}_genes.tsv.gz"
        features, link_features = _expected_features(genes)
        feat = dest / "features.tsv.gz"
        create_features = not (feat.exists() or feat.is_symlink())
        if not create_features:
            if feat.is_symlink():
                matches = link_features and feat.resolve() == genes.resolve()
            else:
                matches = feat.is_file() and _read_gene_text(feat) == features
            if not matches:
                raise ValueError(f"features conflict: {feat}; does not match {genes}; "
                                 "use a new sample path or inspect the existing file; nothing overwritten")
        if create_features and link_features:
            links.append((genes.resolve(), feat))
        plans.append((dest, links, features if create_features and not link_features else None))

    output_paths = {directory / name for directory in destinations
                    for name in ("matrix.mtx.gz", "barcodes.tsv.gz", "genes.tsv.gz", "features.tsv.gz")}
    for directory in destinations:
        for parent in (directory, *directory.parents):
            if parent in output_paths:
                raise ValueError(f"sample directory overlaps a staged file: {parent}; use separate sample paths")

    # Known source/target conflicts must fail above, before even creating directories.
    for dest, links, features in plans:
        dest.mkdir(parents=True, exist_ok=True)
        for src, dst in links:
            dst.symlink_to(src)
        if features is not None:
            with gzip.open(dest / "features.tsv.gz", "xt") as f:
                f.write(features)
    staged = [str(plan[0]) for plan in plans]
    print(f"staged {len(staged)} samples: {staged}")


def _read_gene_text(path):
    try:
        with gzip.open(path, "rt") as f:
            return f.read()
    except (OSError, EOFError, UnicodeError) as exc:
        raise ValueError(f"cannot read gene/features file {path}: {exc}") from exc


def _expected_features(genes):
    """Return expected features text and whether the raw genes can be linked."""
    text = _read_gene_text(genes)
    rows = text.splitlines()
    widths = [len(row.split("\t")) for row in rows]
    if not rows or min(widths) < 2 or len(set(widths)) != 1:
        raise ValueError(f"invalid genes table: {genes}; expected nonempty rows with consistent >=2 columns")
    if widths[0] >= 3:
        return text, True
    return "".join(row + "\tGene Expression\n" for row in rows), False


@_audited_stage("prepare_order")
def cmd_prepare_order(cfg, run):
    ann = cfg["annotation"]
    run.input("gtf", ann["gtf"])
    genes = parse_gtf_genes(ann["gtf"])
    out = Path(ann["gene_order_tsv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    genes.to_csv(out, sep="\t", index=False)
    print(f"wrote {out}: {len(genes)} genes ({genes['chrom'].nunique()} chromosomes)")
    return genes


def _load_gene_order(cfg, run):
    tsv = Path(cfg["annotation"]["gene_order_tsv"])
    if not tsv.exists():
        print(f"gene order table missing at {tsv}; building from GTF")
        cmd_prepare_order(cfg)
    audit = run.audit.parent / "run_prepare_order.json"
    producer = None
    if audit.is_file():
        previous = json.loads(audit.read_text())
        entry = previous.get("outputs", {}).get(tsv.name, {})
        if entry.get("path") == str(tsv.resolve()):
            producer = "prepare_order"
    run.input("gene_order", tsv, producer,
              _stage_parameters(cfg, "prepare_order") if producer else None)
    run.data["gene_order_origin"] = "prepare_order" if producer else "supplied_table"
    return pd.read_csv(tsv, sep="\t")


@_audited_stage("infercnv")
def cmd_infercnv(cfg, run):
    from . import infercnv as ic  # deferred: needs scanpy/infercnvpy

    outdir = _outdir(cfg)
    genes = _load_gene_order(cfg, run)
    qc = cfg.get("qc", {})
    cnv_cfg = cfg.get("infercnv", {})

    ic.validate_groups([s["group"] for s in cfg["samples"]],
                       cnv_cfg.get("patient_group", "patient"), cnv_cfg.get("reference_group", "reference"))
    for s in cfg["samples"]:
        for path in ic.sample_input_paths(s["path"]):
            run.input(f"{s['id']}/{path.name}", path)

    adata = ic.load_samples(cfg["samples"],
                            min_genes=qc.get("min_genes_per_cell", 200),
                            max_mito_pct=qc.get("max_mito_pct", 20.0),
                            downsample=qc.get("downsample_per_sample"),
                            seed=qc.get("seed", 0))
    ic.validate_groups(adata.obs["group"], cnv_cfg.get("patient_group", "patient"),
                       cnv_cfg.get("reference_group", "reference"))
    print(f"loaded {adata.n_obs} cells × {adata.n_vars} genes "
          f"({dict(adata.obs['group'].value_counts())})")
    adata = ic.normalize_for_cnv(adata)
    adata = ic.order_by_position(adata, genes)
    print(f"positioned genes: {adata.n_vars}")
    mapping = adata.uns["gene_position_mapping"]
    mapping.to_csv(outdir / "gene_position_mapping.csv", index=False)
    run.data["gene_position_mapping"] = mapping["status"].value_counts().to_dict()
    unmatched = mapping[mapping["status"] != "positioned"]
    if not unmatched.empty:
        print(f"WARN: {len(unmatched)} input genes lack positions in the supplied annotation; "
              f"see {outdir / 'gene_position_mapping.csv'}", file=sys.stderr)

    input_path = outdir / "cnv_input.h5ad"
    adata.write_h5ad(input_path)

    resolutions = cnv_cfg.get("resolutions",
                              [{"name": "standard", "window_size": 100, "step": 10}])
    exclude = cnv_cfg.get("exclude_chromosomes", ["chrX", "chrY"])
    for res in resolutions:
        ic.run_infercnv(adata, reference_key="group",
                        reference_cat=cnv_cfg.get("reference_group", "reference"),
                        window_size=res["window_size"], step=res["step"],
                        exclude_chromosomes=exclude, n_jobs=cnv_cfg.get("n_jobs"))
        tab = ic.window_table(adata, res["window_size"], res["step"],
                              group_col="group",
                              patient_cat=cnv_cfg.get("patient_group", "patient"),
                              ref_cat=cnv_cfg.get("reference_group", "reference"),
                              exclude_chromosomes=exclude)
        sig_path = outdir / f"window_signal_{res['name']}.tsv"
        tab.to_csv(sig_path, sep="\t", index=False)
        print(f"[{res['name']}] {len(tab)} windows → {sig_path} "
              f"(patient mean |score| {tab['score'].abs().mean():.4f}, "
              f"ref mean {tab['ref_score'].abs().mean():.4f})")

    run.data["heatmap"] = _save_heatmap(adata, cfg, outdir)
    if run.data["heatmap"]["status"] != "success":
        run.skip_output(outdir / "heatmap_patient.png")
    run.data.update({"n_cells": adata.n_obs, "n_positioned_genes": adata.n_vars,
                     "h5ad": str(input_path)})


def _save_heatmap(adata, cfg, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import infercnvpy as cnv
        cnv_cfg = cfg.get("infercnv", {})
        pat = cnv_cfg.get("patient_group", "patient")
        sub = adata[adata.obs["group"] == pat]
        cnv.pl.chromosome_heatmap(sub, groupby="sample", show=False)
        path = outdir / "heatmap_patient.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"heatmap → {path}")
        return {"status": "success"}
    except Exception as e:  # heatmap is evidence, not a gate
        print(f"WARN: heatmap skipped ({e})", file=sys.stderr)
        return {"status": "failed", "reason": str(e)}


@_audited_stage("call_segments")
def cmd_call_segments(cfg, run):
    outdir = _outdir(cfg)
    seg_cfg = cfg.get("segments", {})
    z = seg_cfg.get("z_threshold", 1.5)
    min_w = seg_cfg.get("min_windows", 3)
    max_bp = (seg_cfg.get("max_segment_mb") or 0) * 1_000_000 or None
    resolutions = cfg.get("infercnv", {}).get(
        "resolutions", [{"name": "standard", "window_size": 100, "step": 10}])

    frames = []
    for res in resolutions:
        run.input(f"window_signal_{res['name']}.tsv", outdir / f"window_signal_{res['name']}.tsv",
                  "infercnv", _stage_parameters(cfg, "infercnv"))
        sig = pd.read_csv(outdir / f"window_signal_{res['name']}.tsv", sep="\t")
        segs = call_deletion_segments(sig[["chr", "start", "end", "score"]],
                                      z_threshold=z, min_windows=min_w)
        n_raw = len(segs)
        segs = filter_segments(segs, max_bp)
        segs["resolution"] = res["name"]
        segs.to_csv(outdir / f"segments_{res['name']}.csv", index=False)
        frames.append(segs)
        print(f"[{res['name']}] {len(segs)} deletion segments "
              f"(z<=-{z}, min {min_w} windows; {n_raw - len(segs)} dropped by size cap)")
    all_segs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    all_segs.to_csv(outdir / "segments_all.csv", index=False)
    run.data["n_segments"] = int(len(all_segs))
    return all_segs


def _check_known_intervals(cfg):
    ann = cfg["annotation"]
    if not ann.get("genome_build"):
        raise ValueError("interval extraction requires annotation.genome_build")
    if ann.get("coordinate_system", "1-based-inclusive") != "1-based-inclusive":
        raise ValueError("annotation must use 1-based-inclusive coordinates")
    known = cfg.get("known_intervals", {}) or {}
    if not isinstance(known, dict):
        raise ValueError("known_intervals must be a mapping")
    for name, iv in known.items():
        if not isinstance(iv, dict):
            raise ValueError(f"invalid interval {name}: expected a mapping")
        start, end = iv.get("start"), iv.get("end")
        if (type(start) is not int or type(end) is not int or start < 1 or end < start
                or str(iv.get("chr", "")) not in
                [str(i) for i in range(1, 23)] + [f"chr{i}" for i in range(1, 23)] + ["X", "Y", "chrX", "chrY"]):
            raise ValueError(f"invalid interval {name}: expected chr1-22/X/Y and integer 1 <= start <= end")
        if iv.get("genome_build") != ann["genome_build"]:
            raise ValueError(f"interval {name}: genome_build must match annotation ({ann['genome_build']})")
        if iv.get("coordinate_system", "1-based-inclusive") != "1-based-inclusive":
            raise ValueError(f"interval {name}: only 1-based-inclusive coordinates are supported; convert BED first")


def _unique_candidates(origins, by_id=False):
    """Collapse verified identities, retaining every extraction record as JSON."""
    identities = origins[["gene_symbol", "gene_id"]].dropna().drop_duplicates()
    ambiguous = identities[identities.gene_id.duplicated(keep=False)]
    if not ambiguous.empty:
        raise ValueError(f"conflicting gene identity: {ambiguous.to_dict('records')}")
    identity_cols = ["gene_id", "chrom", "start", "end", "strand", "genome_build"]
    key = "gene_id" if by_id else "gene_symbol"
    if by_id and (origins.gene_id.isna().any() or origins.gene_id.astype(str).str.strip().eq("").any()):
        raise ValueError("incomplete gene identity in exploratory candidates: gene_id required")
    repeated = origins[key].duplicated().any()
    rows = []
    columns = CANDIDATE_COLUMNS + (["gene_id"] if by_id else [])
    for symbol, group in origins.groupby(key, sort=False, dropna=False):
        identity = group[identity_cols].drop_duplicates()
        if (by_id or len(group) > 1) and (len(identity) != 1 or identity.isna().any().any()
                               or any(not str(v).strip() for v in identity.iloc[0])):
            raise ValueError(f"conflicting or incomplete gene identity for {symbol}: "
                             f"{identity.to_dict('records')}")
        row = group.iloc[0][columns].to_dict()
        if repeated:
            # A merged candidate must not pretend to have only its first origin.
            if len(group[["source", "deletion_id"]].drop_duplicates()) > 1:
                row["deletion_id"] = None
            if group.source.nunique() > 1:
                row["source"] = "multiple"
            ordered = group.sort_values(["source", "deletion_id", "interval_chrom", "interval_start", "interval_end"])
            records = ordered.astype(object).where(pd.notna(ordered), None).to_dict("records")
            row["candidate_provenance"] = json.dumps(records, ensure_ascii=False, allow_nan=False)
        rows.append(row)
    return pd.DataFrame(rows, columns=columns + (["candidate_provenance"] if repeated else []))


def _check_missing_cnv_output(run, producer):
    """A missing file after a recorded execution is not an unexecuted check."""
    from .infercnv import InsufficientDataError
    audit = run.outdir / "audit" / f"run_{producer}.json"
    if audit.exists():
        record = json.loads(audit.read_text())
        message = f"missing upstream output from {producer}: {record.get('error', record.get('status'))}; rerun {producer}"
        if record.get("validation_status") == "INSUFFICIENT_DATA":
            raise InsufficientDataError(message)
        raise ValueError(message)


@_audited_stage("extract_genes")
def cmd_extract_genes(cfg, run):
    outdir = _outdir(cfg)
    mode = cfg.get("extract", {}).get("mode", "cnv")
    if mode not in ("cnv", "intervals-only"):
        raise ValueError("extract.mode must be cnv or intervals-only")
    direct = mode == "intervals-only"
    if direct or cfg.get("known_intervals"):
        _check_known_intervals(cfg)
    genes = _load_gene_order(cfg, run)
    cnv_cfg = cfg.get("infercnv", {})
    known = cfg.get("known_intervals", {}) or {}
    include_auto = cfg.get("extract", {}).get("include_auto_in_candidates", False)
    if direct and include_auto:
        raise ValueError("intervals-only cannot include auto CNV candidates")

    seg_path = outdir / "segments_all.csv"
    auto = pd.DataFrame(columns=SEGMENT_COLUMNS + ["resolution"])
    windows = pd.DataFrame(columns=["chr", "start", "end"])
    if not direct and not seg_path.exists():
        _check_missing_cnv_output(run, "call_segments")
    if not direct and seg_path.exists():
        run.input("segments_all.csv", seg_path, "call_segments", _stage_parameters(cfg, "call_segments"))
        segments_record = json.loads(Path(run.data["upstream"]["call_segments"]["record"]).read_text())
        infer_record = run.upstream("infercnv", _stage_parameters(cfg, "infercnv"))
        if segments_record.get("upstream", {}).get("infercnv", {}).get("execution_id") != infer_record["execution_id"]:
            raise ValueError("upstream mismatch: segments belong to another infercnv execution; rerun call-segments")
        if infer_record.get("inputs", {}).get("gene_order") != run.data["inputs"]["gene_order"]:
            raise ValueError("upstream gene-order checksum mismatch; rerun infercnv")
        auto = pd.read_csv(seg_path)
        for name, entry in segments_record.get("inputs", {}).items():
            if name.startswith("window_signal_"):
                run.input(f"validation/{name}", entry["path"], "infercnv", _stage_parameters(cfg, "infercnv"))
                windows = pd.concat([windows, pd.read_csv(entry["path"], sep="\t")], ignore_index=True)

    # ---- interval genes (known + auto, tracked separately) ------------------
    interval_genes, known_rows, auto_rows = {}, [], []
    for iv_id, iv in known.items():
        g = genes_in_interval(genes, iv["chr"], iv["start"], iv["end"])
        interval_genes[("known_interval", iv_id)] = g
        for _, r in g.iterrows():
            known_rows.append({"gene_symbol": r["gene_name"], "source": "known_interval",
                               "deletion_id": iv_id, "chrom": r["chrom"],
                               "start": r["start"], "end": r["end"],
                               "gene_id": r.get("gene_id"), "strand": r.get("strand"),
                               "genome_build": cfg["annotation"].get("genome_build"),
                               "interval_chrom": iv["chr"], "interval_start": iv["start"],
                               "interval_end": iv["end"]})
    for i, s in auto.iterrows():
        iv_id = f"auto_{s['resolution']}_{i}"
        g = genes_in_interval(genes, s["chr"], s["start"], s["end"])
        interval_genes[("auto_segment", iv_id)] = g
        for _, r in g.iterrows():
            auto_rows.append({"gene_symbol": r["gene_name"], "source": "auto_segment",
                              "deletion_id": iv_id, "chrom": r["chrom"],
                              "start": r["start"], "end": r["end"],
                              "gene_id": r.get("gene_id"), "strand": r.get("strand"),
                              "genome_build": cfg["annotation"].get("genome_build"),
                              "interval_chrom": s["chr"], "interval_start": s["start"],
                              "interval_end": s["end"]})

    known_origins = pd.DataFrame(known_rows, columns=PROVENANCE_COLUMNS)
    auto_origins = pd.DataFrame(auto_rows, columns=PROVENANCE_COLUMNS)
    origins = pd.concat([known_origins, auto_origins], ignore_index=True)
    # candidates.csv = the L1-ready input; auto segments are exploratory by default
    cand = _unique_candidates(origins if include_auto else known_origins)
    cand_all = _unique_candidates(origins, by_id=True)
    auto_cand = _unique_candidates(auto_origins, by_id=True)
    origins["in_candidates"] = (origins.source == "known_interval") | include_auto
    origins.to_csv(outdir / "candidate_provenance.csv", index=False)
    auto_cand.to_csv(outdir / "auto_segment_genes.csv", index=False)
    cand.to_csv(outdir / "candidates.csv", index=False)
    cand_all.to_csv(outdir / "candidates_all.csv", index=False)
    print(f"candidates.csv: {cand['gene_symbol'].nunique() if not cand.empty else 0} genes "
          f"(known intervals only={not include_auto}); "
          f"candidates_all.csv: {len(cand_all)} gene IDs")

    # ---- expression QC -------------------------------------------------------
    qc_df = pd.DataFrame(columns=QC_COLUMNS)
    all_iv_genes = (pd.concat(interval_genes.values()).drop_duplicates()
                    if interval_genes else pd.DataFrame())
    h5ad = outdir / "cnv_input.h5ad"
    if not direct and not h5ad.exists():
        # A completed inference may intentionally supply windows only (external producer).
        # Failed/invalidated inference must still fail extraction even without a file.
        audit = outdir / "audit" / "run_infercnv.json"
        if audit.exists():
            record = json.loads(audit.read_text())
            if record.get("status") != "success" or "cnv_input.h5ad" in record.get("outputs", {}):
                _check_missing_cnv_output(run, "infercnv")
    run.data["expression_qc"] = {"status": "not_evaluated", "reason": (
        "intervals-only: CNV and expression QC not requested" if direct else "no interval genes or no h5ad")}
    if not direct and not all_iv_genes.empty and h5ad.exists():
        run.input("cnv_input.h5ad", h5ad, "infercnv", _stage_parameters(cfg, "infercnv"))
        parent = json.loads(Path(run.data["upstream"]["infercnv"]["record"]).read_text())
        if parent.get("inputs", {}).get("gene_order") != run.data["inputs"]["gene_order"]:
            raise ValueError("upstream gene-order checksum mismatch; rerun infercnv")
        import anndata
        adata = anndata.read_h5ad(h5ad)
        qc_df = interval_gene_qc(adata, all_iv_genes, group_col="group",
                                 patient_cat=cnv_cfg.get("patient_group", "patient"),
                                 ref_cat=cnv_cfg.get("reference_group", "reference"),
                                 layer="counts")
        run.data["expression_qc"] = {"status": "success"}
    qc_df.to_csv(outdir / "interval_gene_expression_qc.csv", index=False)

    # ---- validation vs known intervals ----------------------------------------
    val_cfg = cfg.get("validation", {})
    min_ov = val_cfg.get("min_overlap_frac", 0.5)
    validations = []
    for iv_id, iv in known.items():
        g = interval_genes.get(("known_interval", iv_id), pd.DataFrame())
        overlaps = [segment_overlap_frac(s["chr"], s["start"], s["end"],
                                         iv["chr"], iv["start"], iv["end"])
                    for _, s in auto.iterrows()]
        best_ov = max(overlaps) if overlaps else 0.0
        markers = iv.get("markers", [])
        iv_qc = (qc_df[qc_df["gene_id"].isin(g["gene_id"])]
                 if "gene_id" in qc_df and "gene_id" in g else qc_df)
        found = set(iv_qc.loc[iv_qc["found"] == True, "gene_name"]) if not iv_qc.empty else set()
        iv_genes_found = set(g["gene_name"]) & found
        iv_fc = (iv_qc[iv_qc["gene_name"].isin(iv_genes_found)]["log2fc_patient_vs_ref"].mean()
                 if not iv_qc.empty and iv_genes_found else float("nan"))
        reasons = []
        cnv_status, expression_status = "NOT_EVALUATED", "NOT_EVALUATED"
        if direct:
            reasons.append("intervals-only: CNV and expression QC not requested")
        else:
            if not seg_path.exists():
                reasons.append("CNV: call-segments not executed")
            elif not any(segment_overlap_frac(s["chr"], s["start"], s["end"],
                                              iv["chr"], iv["start"], iv["end"]) > 0
                         for _, s in windows.iterrows()):
                cnv_status = "INSUFFICIENT_DATA"
                reasons.append("CNV: no inferred windows overlap this interval")
            else:
                cnv_status = "PASS" if best_ov >= min_ov else "FAIL"
            if not h5ad.exists():
                reasons.append("expression: cnv_input.h5ad unavailable; QC not executed")
            elif not markers or not iv_genes_found:
                expression_status = "INSUFFICIENT_DATA"
                reasons.append("expression: no configured markers or no interval genes in expression data")
            else:
                expression_status = "PASS" if all(m in found for m in markers) else "FAIL"
        validations.append({
            "interval": iv_id,
            "n_genes": int(len(g)),
            "n_genes_in_expression": np.nan if expression_status == "NOT_EVALUATED" else int(len(iv_genes_found)),
            "mean_log2fc_interval_genes": float(iv_fc),
            "auto_overlap_frac": float(best_ov) if cnv_status in ("PASS", "FAIL") else np.nan,
            "AC-CNV-1 (auto recovers known interval)": cnv_status,
            "markers": ",".join(markers),
            "markers_in_expression": ",".join(m for m in markers if m in found),
            "AC-CNV-2 (markers detected in data)": expression_status,
            "reason": "; ".join(reasons),
        })
    val = pd.DataFrame(validations, columns=VALIDATION_COLUMNS)
    val.to_csv(outdir / "validation_known_intervals.csv", index=False)

    _write_report(cfg, outdir, cand, cand_all, qc_df, auto, val, include_auto,
                  known_genes=set(pd.concat([g for k, g in interval_genes.items()
                                             if k[0] == "known_interval"])[
                                                 "gene_id" if "gene_id" in qc_df else "gene_name"]) if known else set())
    run.data.update({
        "n_candidates": int(cand["gene_symbol"].nunique()) if not cand.empty else 0,
        "validation": val.astype(object).where(pd.notna(val), None).to_dict("records")})
    return cand, qc_df, val


# ------------------------------------------------------------------ report
def _write_report(cfg, outdir, cand, cand_all, qc_df, auto, val, include_auto,
                  known_genes=None):
    lines = ["# scRNA-seq to deletion-interval genes: workflow report", ""]
    direct = cfg.get("extract", {}).get("mode", "cnv") == "intervals-only"
    lines.append(f"Extraction mode: {'intervals-only (CNV/expression validation not run)' if direct else 'cnv'}")
    lines.append(f"Project: {cfg.get('project')} · "
                 f"Samples: {', '.join(s['id'] + '(' + s['group'] + ')' for s in cfg.get('samples', [])) or 'Not supplied (samples not required for interval extraction)'}")
    lines.append("")
    lines.append("## Acceptance criteria (AC)")
    lines.append("")
    if val.empty:
        lines.append("No known_intervals configured; validation skipped.")
    elif direct:
        lines.append("| interval | Interval gene count | CNV/expression validation | Reason |")
        lines.append("|---|---|---|---|")
        for _, r in val.iterrows():
            lines.append(f"| {r['interval']} | {r['n_genes']} | NOT_EVALUATED | Interval input only; CNV/expression validation not requested |")
    else:
        lines.append("| interval | Interval gene count | Detected gene count | Mean interval-gene log2FC | Automatic discovery coverage | AC-CNV-1 | AC-CNV-2 | Reason |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in val.iterrows():
            lines.append(f"| {r['interval']} | {r['n_genes']} | {r['n_genes_in_expression']} | "
                         f"{r['mean_log2fc_interval_genes']:.2f} | "
                         f"{r['auto_overlap_frac']:.0%} | "
                         f"{r['AC-CNV-1 (auto recovers known interval)']} | "
                         f"{r['AC-CNV-2 (markers detected in data)']} | {r['reason']} |")
        lines.append("")
        lines.append("> Mean interval-gene log2FC near zero neither excludes a deletion nor establishes dosage compensation. "
                     "Sequencing depth, relative normalization, cell composition and sample differences can affect results. "
                     "Deletions require independent DNA evidence; compensation mechanisms and protein levels require separate validation.")
        if (val["AC-CNV-1 (auto recovers known interval)"] == "FAIL").any():
            lines.append("")
            lines.append("> **AC-CNV-1 FAIL (expected negative result)**: expression-based CNV inference did not automatically recover the known microdeletion. "
                         "Germline heterozygous deletions of 1–3 Mb are at the resolution limit of inferCNV; automatic segments are exploratory candidates. "
                         "Generation of candidates.csv does not depend on them.")
    lines.append("")
    lines.append("## Automatically discovered deletion segments (exploratory; require orthogonal validation)")
    lines.append("")
    if auto.empty:
        lines.append("Not run; intervals-only does not read automatic segments." if direct else "None")
    else:
        lines.append("| chr | start | end | Window count | mean_z | resolution |")
        lines.append("|---|---|---|---|---|---|")
        for _, s in auto.iterrows():
            lines.append(f"| {s['chr']} | {int(s['start'])} | {int(s['end'])} | "
                         f"{s.get('n_windows', '')} | {s.get('mean_z', float('nan')):.2f} | "
                         f"{s['resolution']} |")
    lines.append("")
    lines.append("## Candidate genes (for the L1 pipeline)")
    lines.append("")
    lines.append(f"- `candidates.csv` (**L1 pipeline input**): "
                 f"{cand['gene_symbol'].nunique() if not cand.empty else 0} genes"
                 f"{' (including genes from automatic segments)' if include_auto else ' (known clinical intervals only; recommended)'}")
    lines.append(f"- `candidates_all.csv` (union of known intervals and automatic segments): "
                 f"{len(cand_all)} gene IDs; distinct IDs sharing a symbol are retained separately")
    lines.append(f"- `auto_segment_genes.csv` (automatic segments only; exploratory)")
    lines.append("- `candidate_provenance.csv`: per-extraction source, gene identity, genome build and interval coordinates; "
                 "`in_candidates` records inclusion in L1 input.")
    lines.append("- Each gene enters the candidate set once. For repeated extractions, the `candidate_provenance` JSON field "
                 "retains all sources through ranking. For multiple sources, `deletion_id` is empty and mixed `source` is `multiple`.")
    lines.append("")
    lines.append("Run the L1 pipeline:")
    lines.append("```bash")
    lines.append(f"cp {outdir}/candidates.csv input/candidates.csv")
    lines.append("python -m screening.src.normalize --input input/candidates.csv "
                 "--out outputs_repro/normalized.csv --hgnc-alias data/hgnc_aliases.tsv")
    lines.append("```")
    lines.append("")
    lines.append("## Interval-gene expression QC (E6 check; known clinical intervals only)")
    lines.append("")
    qc_show = (qc_df[qc_df["gene_id" if "gene_id" in qc_df else "gene_name"].isin(known_genes)] if known_genes else qc_df) \
        if not qc_df.empty else qc_df
    if qc_show.empty:
        lines.append("NOT_EVALUATED: intervals-only does not read expression data." if direct else "Skipped: no h5ad or interval genes.")
    else:
        sub = qc_show.sort_values("log2fc_patient_vs_ref").head(10)
        lines.append("Ten interval genes with the lowest patient-versus-control log2FC:")
        lines.append("")
        lines.append("| gene | chr | mean_counts_patient | mean_counts_ref | "
                     "pct_expr_patient | pct_expr_ref | log2FC |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            lines.append(f"| {r['gene_name']} | {r.get('chrom', '')} | "
                         f"{r['mean_counts_patient']:.3f} | {r['mean_counts_ref']:.3f} | "
                         f"{r['pct_expr_patient']:.1%} | {r['pct_expr_ref']:.1%} | "
                         f"{r['log2fc_patient_vs_ref']:.2f} |")
        silent = qc_show[(qc_show["found"] == True) & (qc_show["pct_expr_patient"] < 0.05)]
        if not silent.empty:
            lines.append("")
            lines.append(f"⚠️ Interval genes expressed in <5% of patient cells (E6 risk; "
                         f"unsupported by the L3 perturbation tool): {', '.join(silent['gene_name'])}")
    lines.append("")
    lines.append("## Method notes")
    lines.append("")
    lines.append("- Expression-based CNV inference (infercnvpy) is a **fallback**. Germline heterozygous microdeletions (1–3 Mb) are "
                 "at its resolution limit; automatic segments require orthogonal confirmation (CNV arrays/WGS/MLPA).")
    lines.append("- Gene extraction from known clinical intervals **does not depend** on expression. AC-CNV-1 failure does not prevent candidates.csv generation.")
    lines.append("- Expression QC uses raw counts with a pseudocount of 0.5 for log2FC.")
    lines.append("")
    lines.append("Audit: `audit/run_extract_genes.json` (input/output checksums, parameters and execution status)")
    if not direct:
        lines.append("Check heatmap status in the infercnv audit; an existing file alone does not establish successful generation.")
    (outdir / "cnv_report.md").write_text("\n".join(lines))
    print(f"report → {outdir / 'cnv_report.md'}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(prog="cnv_workflow")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stage-samples", "prepare-order", "infercnv",
                 "call-segments", "extract-genes", "all"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "extract-genes":
            p.add_argument("--mode", choices=("cnv", "intervals-only"),
                           help="explicitly skip CNV/expression inputs with intervals-only")
    args = ap.parse_args()
    try:
        cfg = load_config(args.config)
        if getattr(args, "mode", None):
            cfg.setdefault("extract", {})["mode"] = args.mode
        if args.cmd == "stage-samples":
            cmd_stage_samples(cfg)
        elif args.cmd == "prepare-order":
            cmd_prepare_order(cfg)
        elif args.cmd == "infercnv":
            cmd_infercnv(cfg)
        elif args.cmd == "call-segments":
            cmd_call_segments(cfg)
        elif args.cmd == "extract-genes":
            cmd_extract_genes(cfg)
        elif args.cmd == "all":
            if cfg.get("extract", {}).get("mode") == "intervals-only":
                raise ValueError("intervals-only: use extract-genes instead of all; no CNV steps are needed")
            cmd_prepare_order(cfg)
            cmd_infercnv(cfg)
            cmd_call_segments(cfg)
            cmd_extract_genes(cfg)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        ap.error(f"{args.cmd}: {exc}")


if __name__ == "__main__":
    main()
