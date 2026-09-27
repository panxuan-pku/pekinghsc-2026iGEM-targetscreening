#!/usr/bin/env python3
"""
Merge evidence from ClinGen, gnomAD, and HPA into evidence.parquet.
Also computes checksums for data files into audit dir.
"""
import argparse
import pandas as pd
import hashlib
from pathlib import Path
import json
from datetime import datetime, timezone
from screening.src.normalize import load_alias_map
from screening.src.output_paths import validate_output_paths
from screening.src.phenotype_panels import load_phenotype_panels, tissue_column, tissue_coverage, describe_panels

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DATA_FILES = (
    "hgnc_aliases.tsv",
    "clinGen_gene_curation_list_GRCh38.tsv",
    "gnomad_constraint.tsv",
    "rna_tissue_consensus.tsv",
)

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def join_on_hgnc(base, evidence, name, symbol_map=None, skiprows=0):
    base = base.copy()
    base["hgnc_id"] = base["hgnc_id"].astype("string").str.strip()
    if name == "clinGen":
        with open(evidence) as f:
            skiprows = next((i for i, line in enumerate(f)
                             if line.split("\t", 1)[0].strip().lower() == "#gene symbol"), None)
        if skiprows is None:
            raise ValueError(f"{name}: {evidence}: missing required column: gene symbol (#Gene Symbol header)")
    ev = pd.read_csv(evidence, sep="\t", dtype=str, skiprows=skiprows)
    ev.columns = [c.strip().lower().lstrip('#') for c in ev.columns]
    # Drop duplicate columns (keep first) — ClinGen has repeated PMID headers
    ev = ev.loc[:, ~ev.columns.duplicated()]
    if name == "gnomad_constraint" and "gene" in ev.columns:
        ev = ev.rename(columns={"gene": "gene_symbol"})
    required = {
        "clinGen": ["gene symbol", "haploinsufficiency score", "triplosensitivity score"],
        "gnomad_constraint": ["gene_symbol", "pli", "oe_lof_upper"],
    }.get(name, [])
    missing = [col for col in required if col not in ev.columns]
    if missing:
        raise ValueError(f"{name}: {evidence}: missing required columns: {', '.join(missing)}")
    # gnomAD resolves its symbol column below; gene_id is optional metadata.
    if name != "gnomad_constraint" and "hgnc_id" not in ev.columns:
        for alt in ["hgnc id", "hgncid", "gene_id", "hgncid", "gene symbol", "gene"]:
            if alt in ev.columns:
                ev = ev.rename(columns={alt: "hgnc_id"})
                break
    if name != "gnomad_constraint" and "hgnc_id" not in ev.columns:
        raise ValueError(f"{name}: {evidence}: missing required column: hgnc_id")
    # Map gene symbols to HGNC IDs if needed
    if symbol_map is not None and name == "clinGen":
        ev["hgnc_id"] = ev["hgnc_id"].str.strip().str.upper().map(symbol_map)
        matched = ev["hgnc_id"].notna().sum()
        print(f"{name}: mapped {matched}/{len(ev)} symbols to HGNC ID")
        ev = ev[ev["hgnc_id"].notna()]
        # ClinGen uses scores 0,1,2,3,30,40
        # v2.1: preserve original score for annotation, create clean scoring column
        for col in ["haploinsufficiency score", "triplosensitivity score"]:
            if col in ev.columns:
                ev[col.replace(" score", "_raw_text")] = ev[col].copy()
                ev[col] = pd.to_numeric(ev[col], errors="coerce")
                # Preserve raw score before cleaning
                ev[col.replace(" score", "_raw")] = ev[col].copy()
                # 30="recessive disease association" -> neutral for haploinsufficiency
                # 40="dosage sensitivity unlikely": opposing evidence, not uncurated.
                # NULL=not curated -> neutral (v2.1: don't penalize unstudied genes)
                ev.loc[ev[col].isin([30, 40]) | ev[col].isna(), col] = pd.NA
                raw = ev[col.replace(" score", "_raw")]
                if not raw.dropna().isin([0, 1, 2, 3, 30, 40, -1]).all():
                    raise ValueError(f"ClinGen: unexpected category in {col}")
                ev[col.replace(" score", "_status")] = raw.map({
                    0: "no_evidence", 1: "little_evidence", 2: "emerging_evidence",
                    3: "sufficient_evidence", 30: "recessive_association",
                    40: "dosage_unlikely", -1: "not_evaluated",
                }).fillna("not_evaluated")
                ev.loc[~raw.isin([0, 1, 2, 3]), col] = pd.NA
    # gnomAD uses gene symbols directly; match on hgnc_id after symbol resolution
    # (fix: input symbols like NSD2 never matched gnomAD's historical symbol WHSC1,
    #  silently dropping the strongest constraint evidence for renamed genes)
    if name == "gnomad_constraint":
        # Keep only pLI and LOEUF columns
        ev = ev[["gene_symbol", "pli", "oe_lof_upper"]]
        # Resolve symbols (incl. prev_symbol/alias) to HGNC ID, then merge on hgnc_id
        ev["gene_symbol"] = ev["gene_symbol"].str.strip().str.upper()
        if symbol_map is not None:
            ev["hgnc_id"] = ev["gene_symbol"].map(symbol_map)
            matched = ev["hgnc_id"].notna().sum()
            print(f"gnomAD: resolved {matched}/{len(ev)} symbols to HGNC ID")
            ev = ev[ev["hgnc_id"].notna()].drop_duplicates("hgnc_id")
        ev.columns = ["gene_symbol", "gnomad_pLI", "gnomad_LOEUF"] + (["hgnc_id"] if "hgnc_id" in ev.columns else [])
        # gnomAD uses 'NA' for missing values
        ev = ev.replace('NA', pd.NA)
        # Also handle empty strings
        ev = ev.replace('', pd.NA)
        if "hgnc_id" in ev.columns:
            merged = base.merge(ev[["hgnc_id", "gnomad_pLI", "gnomad_LOEUF"]], on="hgnc_id", how="left", validate="many_to_one")
        else:
            # fallback: legacy exact-symbol match
            merged = base.merge(ev, left_on="input_symbol", right_on="gene_symbol", how="left")
        matched = merged["gnomad_pLI"].notna().sum()
        print(f"gnomAD: matched {matched}/{len(base)} candidates")
        return merged
    # Rename evidence columns to standard names
    col_rename = {
        "haploinsufficiency score": "clinGen_hi_score",
        "triplosensitivity score": "clinGen_triplo_score",
        "pli": "gnomad_pLI",
        "oe_lof_upper": "gnomad_LOEUF",  # gnomAD v2.1.1 uses oe_lof_upper as LOEUF
        "rna_expression_score": "hpa_expr",
    }
    ev = ev.rename(columns={k: v for k, v in col_rename.items() if k in ev.columns})
    cols = ["hgnc_id"] + [c for c in col_rename.values() if c in ev.columns]
    if len(cols) == 1:
        raise ValueError(f"{name}: {evidence}: no supported evidence columns; expected one of {list(col_rename.values())}")
    if name == "clinGen":
        annotations = [c for c in ev.columns if c.startswith(("haploinsufficiency", "triplosensitivity"))
                       or c == "date last evaluated"]
        renamed = {c: "clinGen_" + c.replace(" ", "_") for c in annotations}
        ev = ev.rename(columns=renamed)
        cols += list(renamed.values())
    ev["hgnc_id"] = ev["hgnc_id"].astype("string").str.strip()
    result = base.merge(ev[cols].dropna(subset=["hgnc_id"]).drop_duplicates(),
                        on="hgnc_id", how="left", validate="many_to_one")
    if name == "clinGen":
        for col in ("clinGen_haploinsufficiency_status", "clinGen_triplosensitivity_status"):
            result[col] = result[col].fillna("not_curated")
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normalized", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data",
                    help="evidence input directory (default: <pipeline>/data)")
    args = ap.parse_args()

    src_dir = args.data_dir.resolve()
    audit_dir = Path(args.audit)
    try:
        validate_output_paths(
            [args.normalized, args.config, *[src_dir / name for name in REQUIRED_DATA_FILES]],
            [args.out, audit_dir / "data_checksums.txt", audit_dir / "run.json"],
        )
    except (ValueError, OSError, RuntimeError) as exc:
        ap.error(str(exc))

    norm = pd.read_csv(args.normalized)
    if "hgnc_id" not in norm.columns:
        raise ValueError("normalized.csv must contain hgnc_id")

    base = norm.copy()  # retain recessive and resolution annotations
    base["hgnc_id"] = base["hgnc_id"].astype("string").str.strip().replace("", pd.NA)
    if base["hgnc_id"].dropna().duplicated().any():
        raise ValueError("normalized: duplicate HGNC IDs; resolve duplicate candidates first")

    # Load config for HPA organs
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    try:
        panels = load_phenotype_panels(cfg)
    except ValueError as exc:
        ap.error(str(exc))
    
    missing = [name for name in REQUIRED_DATA_FILES if not (src_dir / name).is_file()]
    if missing:
        ap.error(f"missing required evidence files in {src_dir}: {', '.join(missing)}")
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Build symbol→HGNC ID map for ClinGen (which uses gene symbols, not HGNC IDs)
    symbol_map = {s.upper(): hid for s, hid in load_alias_map(src_dir / "hgnc_aliases.tsv").items()}
    
    checks = {"hgnc_aliases.tsv": sha256_file(src_dir / "hgnc_aliases.tsv")}
    for src_name, f, skip in [
        ("clinGen", "clinGen_gene_curation_list_GRCh38.tsv", 5),
        ("gnomad_constraint", "gnomad_constraint.tsv", 0),
    ]:
        p = src_dir / f
        checks[f] = sha256_file(p)
        try:
            joined = join_on_hgnc(base, p, src_name, symbol_map=symbol_map, skiprows=skip)
        except ValueError as exc:
            ap.error(str(exc))
        base = joined
    
    # HPA needs special handling: wide format (Gene, Tissue, nTPM) → pivot to hpa_{organ}_expr
    p = src_dir / "rna_tissue_consensus.tsv"
    checks[p.name] = sha256_file(p)
    hpa = pd.read_csv(p, sep="\t", dtype=str)
    hpa.columns = [c.strip().lower() for c in hpa.columns]
    missing = [c for c in ("gene name", "tissue", "ntpm") if c not in hpa.columns]
    if missing:
        ap.error(f"hpa: {p}: missing required columns: {', '.join(missing)}")
    # Map gene symbols to HGNC ID
    hpa["hgnc_id"] = hpa["gene name"].str.strip().str.upper().map(symbol_map)
    hpa["ntpm"] = pd.to_numeric(hpa["ntpm"], errors="raise")
    hpa = hpa[hpa["hgnc_id"].notna()]
    # Pivot to wide format
    hpa_wide = hpa.pivot_table(index="hgnc_id", columns="tissue", values="ntpm", aggfunc="max")
    requested = list(dict.fromkeys(t for panel in panels for t in panel["tissues"]))
    raw_tissues = hpa_wide[[t for t in requested if t in hpa_wide.columns]].rename(columns=tissue_column)
    organs = cfg.get("hpa_filter", {}).get("organs", ["liver", "brain", "kidney", "gastrointestinal"])
    groups = cfg.get("hpa_filter", {}).get("organ_tissues", {})
    organ_coverage = {organ: tissue_coverage(groups.get(organ, [organ]), hpa_wide.columns) for organ in organs}
    for organ, coverage in organ_coverage.items():
        if coverage["missing_tissues"]:
            print(f"WARNING hpa organ {organ}: {coverage['coverage_status']}; missing usable tissues: {coverage['missing_tissues']}")
    for organ, tissues in cfg.get("hpa_filter", {}).get("organ_tissues", {}).items():
        available = [t for t in tissues if t in hpa_wide.columns]
        if available:
            hpa_wide[organ] = hpa_wide[available].max(axis=1)
    hpa_wide.columns = [f"hpa_{c.lower().replace(' ', '_')}_expr" for c in hpa_wide.columns]
    hpa_wide = hpa_wide.reset_index()
    # Keep only organs we care about
    organs = cfg.get("hpa_filter", {}).get("organs", ["liver", "brain", "kidney", "gastrointestinal"])
    organ_cols = [f"hpa_{o}_expr" for o in organs]
    keep_cols = ["hgnc_id"] + [c for c in organ_cols if c in hpa_wide.columns]
    base = base.merge(hpa_wide[keep_cols], on="hgnc_id", how="left", validate="many_to_one")
    if panels:
        base = base.merge(raw_tissues.reset_index(), on="hgnc_id", how="left", validate="many_to_one")
    panel_coverage = describe_panels(base, panels)
    for panel in panel_coverage:
        if panel["missing_tissues"]:
            print(f"WARNING hpa panel {panel['id']}: {panel['coverage_status']}; missing usable tissues: {panel['missing_tissues']}")
    print(f"hpa: joined {hpa_wide['hgnc_id'].notna().sum()} genes, organs: {[c for c in keep_cols if c != 'hgnc_id']}")
    
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(args.out, index=False)

    (audit_dir / "data_checksums.txt").write_text(
        "\n".join(f"{f}  {h}" for f, h in checks.items())
    )
    (audit_dir / "run.json").write_text(json.dumps({
        "stage": "merge_evidence", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": sha256_file(__file__),
        "config": cfg, "config_sha256": sha256_file(args.config),
        "normalized_sha256": sha256_file(args.normalized), "data_checksums": checks,
        "data_directory": str(src_dir), "rows": len(base),
        "output_sha256": sha256_file(args.out),
        "coverage": base.notna().mean().to_dict(),
        "hpa_group_aggregation": "maximum nTPM; expression annotation, not safety assessment",
        "hpa_coverage": {"organs": organ_coverage, "phenotype_panels": panel_coverage},
    }, ensure_ascii=False, indent=2))
    print(f"wrote {args.out} with {len(base)} rows")

if __name__ == "__main__":
    main()
