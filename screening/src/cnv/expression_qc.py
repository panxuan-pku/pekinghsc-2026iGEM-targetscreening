#!/usr/bin/env python3
"""Per-gene expression QC for genes inside a deletion interval.

Answers the E6-type question before anyone trusts the interval: are these
genes even expressed in this dataset, in patient vs reference cells?
Uses RAW counts (adata.layers[layer], or .X when layer is None).
"""
import numpy as np
import pandas as pd

from .infercnv import gene_match_keys, validate_groups


def interval_gene_qc(adata, genes_df, group_col, patient_cat, ref_cat, layer="counts"):
    """Per-gene expression stats in patient vs reference cells.

    genes_df: DataFrame with at least a `gene_name` column (order preserved).
    Returns DataFrame: gene_name, found, mean_counts_patient, mean_counts_ref,
    pct_expr_patient, pct_expr_ref, log2fc_patient_vs_ref (+ chrom/start/end
    if present in genes_df).
    """
    if group_col not in adata.obs.columns:
        raise ValueError(f"adata.obs lacks group column {group_col!r}")
    validate_groups(adata.obs[group_col], patient_cat, ref_cat)
    groups = adata.obs[group_col].astype(str)
    pat = (groups == str(patient_cat)).to_numpy()
    ref = (groups == str(ref_cat)).to_numpy()
    if pat.sum() == 0 or ref.sum() == 0:
        raise ValueError(f"{group_col!r} lacks cells for {patient_cat!r} or {ref_cat!r}")

    X = adata.layers[layer] if layer else adata.X
    var_index, key = gene_match_keys(adata, genes_df)
    gnames = genes_df["gene_name"].astype(str).tolist()
    locs = var_index.get_indexer(genes_df[key])

    rows = []
    for gname, gi in zip(gnames, locs):
        if gi < 0:
            rows.append({"gene_name": gname, "found": False,
                         "mean_counts_patient": np.nan, "mean_counts_ref": np.nan,
                         "pct_expr_patient": np.nan, "pct_expr_ref": np.nan,
                         "log2fc_patient_vs_ref": np.nan})
            continue
        col = X[:, gi]
        vals = col.toarray().ravel() if hasattr(col, "toarray") else np.asarray(col).ravel()
        pv, rv = vals[pat], vals[ref]
        mp, mr = float(pv.mean()), float(rv.mean())
        rows.append({"gene_name": gname, "found": True,
                     "mean_counts_patient": mp, "mean_counts_ref": mr,
                     "pct_expr_patient": float((pv > 0).mean()),
                     "pct_expr_ref": float((rv > 0).mean()),
                     "log2fc_patient_vs_ref": float(np.log2((mp + 0.5) / (mr + 0.5)))})
    out = pd.DataFrame(rows)
    if key == "gene_id":
        out["gene_id"] = genes_df[key].to_numpy()
        symbols = (adata.var["gene_symbols"].to_numpy() if "gene_symbols" in adata.var
                   else adata.var_names.to_numpy())
        out["gene_symbol"] = [symbols[i] if i >= 0 else "" for i in locs]
    for c in ("chrom", "start", "end"):
        if c in genes_df.columns:
            out[c] = genes_df[c].to_numpy()
    return out
