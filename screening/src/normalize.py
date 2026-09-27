#!/usr/bin/env python3
"""
Normalize candidate gene symbols to HGNC IDs.

Reads input candidates.csv (required column 'gene_symbol'),
builds alias map from data/hgnc_aliases.tsv, and writes to the path specified by --out
with columns: hgnc_id, input_symbol, status, plus source, deletion_id,
chrom, start, end, candidate_provenance when present in the input.
"""
import argparse
from pathlib import Path
import pandas as pd
import sys
import re
from screening.src.output_paths import validate_output_paths

EMPTY_CANDIDATES_ERROR = "candidate input is empty; check input or upstream filtering results"

def load_alias_map(path):
    df = pd.read_csv(path, sep=None, engine="python", dtype=str)  # sniff , or \t
    df.columns = [c.strip() for c in df.columns]  # data files carry padded headers
    if not {"hgnc_id", "symbol"} <= set(df.columns):
        raise ValueError("HGNC aliases require hgnc_id/symbol columns; check format and hydrate Git LFS pointers")
    df["hgnc_id"] = df["hgnc_id"].str.strip()
    df = df.dropna(subset=["hgnc_id", "symbol"])
    m = dict(zip(df["symbol"].str.strip().str.lower(), df["hgnc_id"]))
    aliases = {}
    for _, row in df.iterrows():
        for col in ("prev_symbol", "alias_symbol"):
            if pd.isna(row.get(col)):
                continue
            value = str(row[col]).strip()
            # Padded TSV fields may retain the enclosing quotes after parsing.
            if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            for alias in re.split(r"[,;|]", value):
                a = alias.strip().lower()
                if a:
                    aliases.setdefault(a, set()).add(row["hgnc_id"])
    # Approved symbols win; ambiguous historical aliases remain unresolved.
    for alias, ids in aliases.items():
        if alias not in m and len(ids) == 1:
            m[alias] = next(iter(ids))
    return m

def resolve(df, alias_map):
    if df.empty:
        raise ValueError(EMPTY_CANDIDATES_ERROR)
    out = []
    for s in df["gene_symbol"]:
        st = str(s).strip() if pd.notna(s) else ""
        hid = alias_map.get(st.lower())
        status = "ok" if hid is not None else "unresolved"
        out.append({"hgnc_id": hid, "input_symbol": st, "status": status})
    normalized = pd.DataFrame(out)
    for col in ("source", "deletion_id", "chrom", "start", "end", "candidate_provenance"):
        if col in df.columns:
            normalized[col] = df[col].reset_index(drop=True)
    return normalized

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hgnc-alias", required=True)
    args = ap.parse_args()

    try:
        validate_output_paths([args.input, args.hgnc_alias], [args.out])
    except (ValueError, OSError, RuntimeError) as exc:
        ap.error(str(exc))

    try:
        df = pd.read_csv(args.input)
    except pd.errors.EmptyDataError:
        ap.error(EMPTY_CANDIDATES_ERROR)
    if df.empty:
        ap.error(EMPTY_CANDIDATES_ERROR)
    if "gene_symbol" not in df.columns:
        df.columns = [c.strip() for c in df.columns]
        if len(df.columns) != 1:
            print("ERROR: candidates.csv must contain exactly one column 'gene_symbol'", file=sys.stderr)
            sys.exit(2)
        df = df.rename(columns={df.columns[0]: "gene_symbol"})

    alias_map = load_alias_map(args.hgnc_alias)
    norm = resolve(df, alias_map)

    unresolved = (norm["status"] == "unresolved").mean()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    norm.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(norm)} genes, unresolved {unresolved:.1%}")

    if unresolved > 0.1:
        print("WARN: >10% candidate genes unresolved — check input symbols vs HGNC", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
