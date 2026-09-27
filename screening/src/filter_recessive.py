#!/usr/bin/env python3
"""Annotate reported recessive inheritance, not absence of dominant effects.

gnomAD pLI/LOEUF measure depletion of predicted loss-of-function variation,
not inheritance. Default scoring is neutral; downweighting is opt-in heuristic.
"""
import argparse
from pathlib import Path
import pandas as pd
from screening.src.normalize import load_alias_map


def load_omim_recessive(path):
    """Read named genemap2 columns; return autosomal recessive symbols.

    Mixed dominant/recessive rows are not tagged. X-linked inheritance needs
    sex/ploidy context and is not treated as autosomal recessive here.
    """
    with open(path) as f:
        header_row = next((i for i, line in enumerate(f)
                           if "\tPhenotypes\t" in line and "Approved Gene Symbol" in line), None)
    if header_row is None:
        raise ValueError("OMIM genemap2: missing named Phenotypes/Approved Gene Symbol columns")
    df = pd.read_csv(path, sep="\t", skiprows=header_row, dtype=str)
    df.columns = df.columns.str.lstrip("#").str.strip()
    recessive = set()
    for _, row in df.iterrows():
        inheritance = str(row["Phenotypes"]).lower()
        if "autosomal recessive" in inheritance and "dominant" not in inheritance:
            symbols = row.get("Approved Gene Symbol")
            if pd.isna(symbols):
                symbols = row.get("Gene Symbols", "")
            if pd.notna(symbols):
                recessive.update(s.strip().upper() for s in symbols.split(",") if s.strip())
    return recessive


def load_clinvar_recessive(path):
    """Read an explicitly prepared inheritance table, not variant_summary.txt."""
    if path is None:
        return set()
    df = pd.read_csv(path, sep="\t", dtype=str)
    if not {"ModeOfInheritance", "GeneSymbol"} <= set(df.columns):
        raise ValueError("ClinVar input must have GeneSymbol/ModeOfInheritance columns")
    symbols = df["GeneSymbol"].str.strip().str.upper()
    inheritance = df["ModeOfInheritance"].fillna("").str.strip().str.lower()
    dominant = set(symbols[inheritance.str.contains("dominant")].dropna())
    recessive = set(symbols[inheritance.eq("autosomal recessive")].dropna())
    return recessive - dominant


def build_symbol_map(aliases_path):
    return {s.upper(): {hid} for s, hid in load_alias_map(aliases_path).items()}


def filter_recessive(candidates_df, recessive_symbols, symbol_map, source="OMIM"):
    """False means no matching record, not proof of non-recessive inheritance."""
    ids = set()
    for sym in recessive_symbols:
        ids.update(symbol_map.get(sym, set()))
    candidates = candidates_df.copy()
    candidates["hgnc_id"] = candidates["hgnc_id"].astype("string").str.strip()
    candidates["recessive"] = candidates["hgnc_id"].isin(ids)
    candidates["recessive_source"] = candidates["recessive"].map({True: source, False: ""})
    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normalized", required=True)
    ap.add_argument("--hgnc-alias", required=True)
    ap.add_argument("--omim", default=None, help="OMIM genemap2.txt with named header")
    ap.add_argument("--clinvar", default=None, help="prepared GeneSymbol/ModeOfInheritance TSV")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = pd.read_csv(args.normalized)
    sym_map = build_symbol_map(args.hgnc_alias)
    sources = {}
    if args.omim:
        sources["OMIM"] = load_omim_recessive(args.omim)
    if args.clinvar:
        sources["ClinVar"] = load_clinvar_recessive(args.clinvar)
    result["recessive"] = False
    result["recessive_source"] = ""
    result["recessive_assessment"] = "records_checked" if sources else "not_assessed"
    for name, symbols in sources.items():
        tagged = filter_recessive(result, symbols, sym_map, source=name)
        mask = tagged["recessive"]
        result["recessive"] |= mask
        result.loc[mask, "recessive_source"] += name + ";"
    result["recessive_source"] = result["recessive_source"].str.rstrip(";")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {result['recessive'].sum()} recessive annotations")


if __name__ == "__main__":
    main()
