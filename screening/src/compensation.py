#!/usr/bin/env python3
"""mRNA compensation status analysis (v2.1 — 质询 #0).

Given scRNA-seq data with patient/control groups, computes per-gene
compensation ratio and classification.

Legacy compensation labels describe relative expression, not mechanisms.
A ratio near one does not establish compensation, protein deficiency or
SINEUP suitability. Cell composition and donor effects remain confounders.
"""
import argparse
import sys
import numpy as np
import pandas as pd

COMPENSATION_THRESHOLDS = {
    "full": (0.95, 1.05),
    "partial_high": (0.85, 0.95),
    "partial_low": (0.70, 0.85),
    "uncompensated": (0.40, 0.70),
    "unreliable": None,  # low-expression genes
}

def compute_compensation(adata, genes_of_interest, group_col="condition",
                         patient_cat="WS", ref_cat="CTRL",
                         layer="counts",
                         min_detection_pct=0.01, library_size_col=None,
                         full_gene_matrix=False):
    """For each gene, compute patient/control expression ratio and classify.

    Returns DataFrame: gene, ratio, compensation_class, mean_pat, mean_ref,
    pct_pat, pct_ref, n_pat, n_ref
    """
    if len(genes_of_interest) == 0:
        raise ValueError("candidate gene list is empty; check the input candidate table")
    if group_col not in adata.obs.columns:
        raise ValueError(f"adata.obs lacks {group_col!r}")
    groups = adata.obs[group_col].astype(str)
    pat_mask = (groups == str(patient_cat)).to_numpy()
    ref_mask = (groups == str(ref_cat)).to_numpy()
    if pat_mask.sum() == 0 or ref_mask.sum() == 0:
        raise ValueError(f"No cells for {patient_cat!r} or {ref_cat!r}")

    X = adata.layers[layer] if layer else adata.X
    data = X.data if hasattr(X, "tocsr") else np.asarray(X)
    if not np.isfinite(data).all() or (data < 0).any() or not np.allclose(data, np.rint(data)):
        raise ValueError("expression input must contain non-negative raw counts, not log/transformed values")
    if library_size_col:
        library_sizes = pd.to_numeric(adata.obs[library_size_col], errors="raise").to_numpy()
    elif full_gene_matrix:
        library_sizes = np.asarray(X.sum(axis=1)).ravel()
    else:
        raise ValueError("provide full-gene library_size_col or explicitly confirm full_gene_matrix; HVG subsets are not valid denominators")
    if not np.isfinite(library_sizes).all() or (library_sizes <= 0).any():
        raise ValueError("zero-library cells must be removed before relative expression analysis")
    if (library_sizes + 1e-6 < np.asarray(X.sum(axis=1)).ravel()).any():
        raise ValueError("full-gene library sizes cannot be smaller than matrix row sums")
    var_idx = pd.Index(adata.var_names.astype(str))
    if not var_idx.is_unique:
        raise ValueError("duplicate gene names in expression matrix")
    if str(patient_cat) == str(ref_cat):
        raise ValueError("patient and reference must be different groups")

    rows = []
    in_data = [g for g in genes_of_interest if g in var_idx]
    missing = [g for g in genes_of_interest if g not in var_idx]
    if missing:
        print(f"WARN: {len(missing)} genes not in adata: {missing[:5]}...",
              file=sys.stderr)
    if not in_data:
        raise ValueError("no candidate genes match expression var_names; check gene symbols and input sources")

    for gene in in_data:
        gi = var_idx.get_loc(gene)
        col = X[:, gi]
        vals = col.toarray().ravel() if hasattr(col, "toarray") else np.asarray(col).ravel()
        vals = vals / library_sizes * 10000  # all genes, not just interval genes
        pv, rv = vals[pat_mask], vals[ref_mask]
        mp, mr = float(pv.mean()), float(rv.mean())
        pp = float((pv > 0).mean())
        rp = float((rv > 0).mean())
        ratio = mp / mr if mr > 0 else np.nan

        classification = "unreliable"
        if pp < min_detection_pct and rp < min_detection_pct:
            classification = "unreliable"
        elif not np.isnan(ratio):
            for cls, (lo, hi) in {
                k: v for k, v in COMPENSATION_THRESHOLDS.items() if v is not None
            }.items():
                if lo <= ratio <= hi:
                    classification = cls
                    break
            else:
                if ratio > 1.05:
                    classification = "overcompensated"
                elif ratio < 0.40:
                    classification = "strongly_down"

        rows.append({
            "gene": gene,
            "compensation_ratio": round(ratio, 4) if not np.isnan(ratio) else np.nan,
            "compensation_class": classification,
            "expression_basis": "mean_library_normalized_counts_per_10000",
            "mechanism_status": "not_established",
            "mean_patient": round(mp, 4),
            "mean_reference": round(mr, 4),
            "pct_patient": round(pp, 4),
            "pct_reference": round(rp, 4),
            "n_patient": int(pat_mask.sum()),
            "n_reference": int(ref_mask.sum()),
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--genes", required=True, help="CSV of gene_symbols (one column)")
    ap.add_argument("--group-col", default="condition")
    ap.add_argument("--patient", default="WS")
    ap.add_argument("--reference", default="CTRL")
    ap.add_argument("--library-size-col", help="obs column of full-gene raw library sizes, saved before gene subsetting")
    ap.add_argument("--full-gene-matrix", action="store_true", help="explicitly confirm input counts include all measured genes, not HVGs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        import scanpy as sc
    except ImportError:
        print("FATAL: scanpy not installed", file=sys.stderr)
        sys.exit(3)

    try:
        genes_df = pd.read_csv(args.genes)
        gene_col = "gene_symbol" if "gene_symbol" in genes_df.columns else genes_df.columns[0]
        genes = genes_df[gene_col].dropna().astype(str).tolist()
        if not genes:
            raise ValueError("candidate gene list is empty; check the input candidate table")
        adata = sc.read_h5ad(args.h5ad)
        result = compute_compensation(adata, genes,
                                      group_col=args.group_col,
                                      patient_cat=args.patient,
                                      ref_cat=args.reference,
                                      library_size_col=args.library_size_col,
                                      full_gene_matrix=args.full_gene_matrix)
    except pd.errors.EmptyDataError:
        ap.error(f"{args.genes}: candidate gene list is empty")
    except (OSError, ValueError, KeyError) as exc:
        ap.error(f"compensation input ({args.genes}, {args.h5ad}): {exc}")
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print("WARNING: legacy labels describe relative RNA only; compensation mechanism and protein status are not established")
    print(f"wrote {args.out}: {len(result)} genes assessed")

    # summary
    counts = result["compensation_class"].value_counts()
    for cls in ["full", "partial_high", "partial_low", "uncompensated",
                "overcompensated", "strongly_down", "unreliable"]:
        n = counts.get(cls, 0)
        labels = {
            "full": "fully compensated (ratio 0.95-1.05)",
            "partial_high": "partial: high (0.85-0.95)",
            "partial_low": "partial: low (0.70-0.85)",
            "uncompensated": "uncompensated (0.40-0.70)",
            "overcompensated": "overcompensated (>1.05)",
            "strongly_down": "strongly downregulated (<0.40)",
            "unreliable": "unreliable (low expression)",
        }
        if n > 0:
            print(f"  {cls}: {n} genes ({labels[cls]})")


if __name__ == "__main__":
    main()
