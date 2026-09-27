#!/usr/bin/env python3
"""Gene ordering + interval→gene mapping from a GENCODE GTF.

Pure pandas; no scanpy dependency, so it is cheap to unit-test.
Coordinates are 1-based inclusive (GTF convention).
"""
import gzip

import pandas as pd

CHROM_ORDER = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
_CHROM_RANK = {c: i for i, c in enumerate(CHROM_ORDER)}


def parse_gtf_genes(gtf_path, gene_types=("protein_coding",)):
    """Parse a GTF (plain or .gz) → DataFrame[gene_name, gene_id, chrom, start, end, strand].

    One row per gene, main chromosomes only, sorted by (chrom, start).
    `gene_types=None` keeps every gene.
    """
    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    rows = []
    with opener(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            chrom = parts[0]
            if chrom not in _CHROM_RANK:
                continue
            attrs = {}
            for kv in parts[8].strip().rstrip(";").split(";"):
                kv = kv.strip()
                if not kv or " " not in kv:
                    continue
                k, v = kv.split(" ", 1)
                attrs[k] = v.strip('"')
            gtype = attrs.get("gene_type") or attrs.get("gene_biotype", "")
            if gene_types and gtype not in gene_types:
                continue
            rows.append({
                "gene_name": attrs.get("gene_name", attrs.get("gene_id", "")),
                "gene_id": attrs.get("gene_id", "").split(".")[0],
                "chrom": chrom,
                "start": int(parts[3]),
                "end": int(parts[4]),
                "strand": parts[6],
            })
    df = pd.DataFrame(rows, columns=["gene_name", "gene_id", "chrom", "start", "end", "strand"])
    if not df.empty:
        df["_rank"] = df["chrom"].map(_CHROM_RANK)
        df = df.sort_values(["_rank", "start"]).drop(columns="_rank").reset_index(drop=True)
    return df


def genes_in_interval(genes, chrom, start, end):
    """Genes overlapping [start, end] (1-based inclusive) on `chrom`."""
    chrom = str(chrom)
    if not chrom.startswith("chr"):
        chrom = f"chr{chrom}"
    sel = genes[(genes["chrom"] == chrom) & (genes["end"] >= start) & (genes["start"] <= end)]
    return sel.sort_values("start").reset_index(drop=True)
