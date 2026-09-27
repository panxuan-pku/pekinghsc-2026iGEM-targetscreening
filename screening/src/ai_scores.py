#!/usr/bin/env python3
"""Convert official precomputed DeepLOF scores into the L1 pipeline's ai_scores.csv.

Source: LaPolice & Huang (2023), BMC Bioinformatics 24:347.
Official scores: "Additional file 3 / Data File 2 — Scores from the nonlinear
DeepLOF model" (figshare 24159504, CC BY 4.0), mirrored to data/DeepLOF_scores.csv.
MD5 (official): 4f7a9c6520b840db677963cfe82d3e34

Output contract (docs/api/ai_scores.md): outputs/ai_scores.csv with columns
  hgnc_id, DeepLOF_score
read by src/consensus_v2.py via `--ai-scores`.

Usage (from screening/, after conda activate virtual-screening):
  python -m screening.src.ai_scores deeplof \
      --scores data/DeepLOF_scores.csv \
      --aliases data/hgnc_aliases.tsv \
      --out outputs/ai_scores.csv
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
from screening.src.normalize import load_alias_map

OFFICIAL_MD5 = "4f7a9c6520b840db677963cfe82d3e34"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_symbol_map(alias_path):
    return {s.upper(): hid for s, hid in load_alias_map(alias_path).items()}


def build_ensembl_map(alias_path):
    aliases = pd.read_csv(alias_path, sep=None, engine="python", dtype=str)
    aliases.columns = [c.strip() for c in aliases.columns]
    required = {"hgnc_id", "ensembl_gene_id"}
    if not required <= set(aliases.columns):
        raise ValueError(f"HGNC aliases require columns {sorted(required)}")
    ids = {}
    ambiguous = set()
    for ensembl, hgnc_id in zip(aliases["ensembl_gene_id"], aliases["hgnc_id"]):
        ensembl = ensembl.strip() if pd.notna(ensembl) else ""
        hgnc_id = hgnc_id.strip() if pd.notna(hgnc_id) else ""
        if not ensembl or not hgnc_id:
            continue
        if ensembl in ids and ids[ensembl] != hgnc_id:
            ambiguous.add(ensembl)
        else:
            ids[ensembl] = hgnc_id
    for ensembl in ambiguous:
        ids.pop(ensembl, None)
    return ids, ambiguous


def cmd_deeplof(args):
    scores = pd.read_csv(args.scores)
    required = {"ensembl", "gene_symbol", "DeepLOF_score"}
    if not required <= set(scores.columns):
        raise ValueError(f"scores file must have columns {sorted(required)}, "
                         f"got {scores.columns.tolist()}")
    if scores.empty:
        raise ValueError("DeepLOF scores input is empty")
    values = pd.to_numeric(scores["DeepLOF_score"], errors="coerce")
    valid = values.between(0, 1) & (values.dtype.kind != "b")  # rejects NaN/Inf and booleans
    if not valid.all():
        rows = (scores.index[~valid] + 1).tolist()
        raise ValueError("DeepLOF_score must contain finite numbers in [0, 1]; "
                         f"invalid data rows (1-based, first 10): {rows[:10]}")
    scores["DeepLOF_score"] = values

    md5 = hashlib.md5(Path(args.scores).read_bytes()).hexdigest()
    if md5 != OFFICIAL_MD5:
        print(f"WARN: md5 {md5} != official {OFFICIAL_MD5}; proceed anyway", file=sys.stderr)

    ensembl_map, ambiguous = build_ensembl_map(args.aliases)
    symbol_map = build_symbol_map(args.aliases)
    out = scores[["ensembl", "gene_symbol", "DeepLOF_score"]].copy()
    out["ensembl"] = out["ensembl"].astype("string").str.strip()
    out["gene_symbol_upper"] = out["gene_symbol"].str.strip().str.upper()
    out["ensembl_hgnc_id"] = out["ensembl"].map(ensembl_map)
    out["symbol_hgnc_id"] = out["gene_symbol_upper"].map(symbol_map)
    disagreement = (out["ensembl_hgnc_id"].notna() & out["symbol_hgnc_id"].notna()
                    & (out["ensembl_hgnc_id"] != out["symbol_hgnc_id"]))
    out["hgnc_id"] = out["ensembl_hgnc_id"].mask(disagreement)
    unmapped = out["ensembl_hgnc_id"].isna()
    mapped = out["hgnc_id"].notna().sum()
    if not mapped:
        raise ValueError("DeepLOF: no Ensembl IDs mapped to HGNC; check aliases and headers")
    print(f"mapped {mapped}/{len(out)} genes to HGNC ID "
          f"({mapped / len(out):.1%})")
    print(f"excluded {int(unmapped.sum())} unmatched Ensembl IDs and "
          f"{int(disagreement.sum())} Ensembl/symbol disagreements; see audit")
    n_scores = out.groupby("hgnc_id")["DeepLOF_score"].nunique()
    conflicts = n_scores[n_scores > 1].index
    if len(conflicts):
        details = scores.loc[out["hgnc_id"].isin(conflicts),
                             ["ensembl", "gene_symbol", "DeepLOF_score"]].copy()
        details.insert(0, "data_row", details.index + 1)
        details["hgnc_id"] = out.loc[details.index, "hgnc_id"]
        raise ValueError(f"DeepLOF: conflicting scores for {len(conflicts)} HGNC IDs; "
                         "resolve input sources before rerunning; no score selected. "
                         "First 10 records:\n" + details.sort_values("hgnc_id", kind="stable").head(10).to_string(
                             index=False, float_format=str))
    out_source = out
    out = out[["hgnc_id", "DeepLOF_score"]].dropna(subset=["hgnc_id"])
    n_duplicates = int(out.duplicated("hgnc_id").sum())
    out = out.drop_duplicates("hgnc_id")

    out_path = Path(args.out)

    # score distribution
    q = out["DeepLOF_score"].quantile([0.1, 0.5, 0.9])
    print(f"DeepLOF score distribution: median {q[0.5]:.3f}, "
          f"p10 {q[0.1]:.3f}, p90 {q[0.9]:.3f}")

    audit = {
        "source": "figshare 24159504 (Additional file 3, Data File 2)",
        "md5_official": OFFICIAL_MD5,
        "md5_downloaded": md5,
        "aliases_sha256": sha256_file(args.aliases),
        "scores_sha256": sha256_file(args.scores),
        "n_genes_total": len(scores),
        "n_genes_mapped": len(out),
        "mapping_rate": round(mapped / len(scores), 4),
        "mapping_policy": "unique Ensembl-to-HGNC only; no symbol fallback; discordant symbols excluded",
        "n_unmapped_ensembl": int(unmapped.sum()),
        "unmapped_ensembl": [
            {"data_row": int(i + 1), "ensembl": str(row["ensembl"]),
             "gene_symbol": str(row["gene_symbol"]),
             "symbol_hgnc_id": row["symbol_hgnc_id"] if pd.notna(row["symbol_hgnc_id"]) else None,
             "reason": "ambiguous HGNC crosswalk" if row["ensembl"] in ambiguous else "not in HGNC crosswalk"}
            for i, row in out_source.loc[unmapped].iterrows()
        ],
        "n_symbol_disagreements": int(disagreement.sum()),
        "symbol_disagreements": [
            {"data_row": int(i + 1), "ensembl": str(row["ensembl"]),
             "gene_symbol": str(row["gene_symbol"]),
             "ensembl_hgnc_id": row["ensembl_hgnc_id"],
             "symbol_hgnc_id": row["symbol_hgnc_id"]}
            for i, row in out_source.loc[disagreement].iterrows()
        ],
        "duplicate_policy": "identical_scores_only; conflicting_scores_rejected",
        "n_identical_duplicates_removed": n_duplicates,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    out_path.parent.joinpath("audit").mkdir(parents=True, exist_ok=True)
    Path(out_path.parent / "audit" / f"ai_scores_deeplof_{time.strftime('%Y%m%d')}.json").write_text(
        json.dumps(audit, indent=2))
    print(f"wrote {out_path} ({len(out)} genes); audit saved")
    return out


def main():
    ap = argparse.ArgumentParser(prog="ai_scores")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("deeplof")
    p.add_argument("--scores", required=True)
    p.add_argument("--aliases", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "deeplof":
        try:
            cmd_deeplof(args)
        except (OSError, ValueError) as exc:
            ap.error(f"DeepLOF input ({args.scores}, {args.aliases}): {exc}")


if __name__ == "__main__":
    main()
