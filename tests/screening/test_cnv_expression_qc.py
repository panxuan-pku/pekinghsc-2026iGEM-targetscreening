"""Tests for interval gene expression QC (needs anndata)."""
import numpy as np
import pandas as pd
import anndata
import pytest

from screening.src.cnv.expression_qc import interval_gene_qc


def _adata():
    # 4 patient cells, 4 reference cells, 3 genes; G1 halved in patient
    X = np.array([[2, 4, 0], [2, 4, 0], [2, 4, 1], [2, 4, 0],     # patient
                  [4, 4, 0], [4, 4, 0], [4, 4, 2], [4, 4, 0]], dtype=float)  # ref
    ad = anndata.AnnData(X=X)
    ad.var_names = ["G1", "G2", "G3"]
    ad.obs["group"] = ["patient"] * 4 + ["reference"] * 4
    ad.layers["counts"] = ad.X.copy()
    return ad


def test_qc_stats_exact():
    genes = pd.DataFrame({"gene_name": ["G1", "G2", "MISSING"],
                          "chrom": ["chr7"] * 3, "start": [1, 2, 3], "end": [9, 9, 9]})
    qc = interval_gene_qc(_adata(), genes, "group", "patient", "reference")
    g1 = qc[qc["gene_name"] == "G1"].iloc[0]
    assert g1["mean_counts_patient"] == 2.0
    assert g1["mean_counts_ref"] == 4.0
    assert g1["pct_expr_patient"] == 1.0
    assert g1["log2fc_patient_vs_ref"] < 0          # depleted in patient
    missing = qc[qc["gene_name"] == "MISSING"].iloc[0]
    assert missing["found"] == False
    assert np.isnan(missing["mean_counts_patient"])


def test_bad_group_raises():
    genes = pd.DataFrame({"gene_name": ["G1"]})
    with pytest.raises(ValueError):
        interval_gene_qc(_adata(), genes, "group", "patient", "nonexistent")
