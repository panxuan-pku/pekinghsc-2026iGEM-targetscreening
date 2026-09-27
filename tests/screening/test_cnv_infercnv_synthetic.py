"""End-to-end synthetic test of the infercnv step (needs scanpy + infercnvpy)."""
import numpy as np
import pandas as pd
import scanpy
import infercnvpy
import anndata

from screening.src.cnv.infercnv import (normalize_for_cnv, order_by_position,
                              run_infercnv, window_table)
from screening.src.cnv.segments import call_deletion_segments


def _synthetic(seed=0):
    """3 chromosomes × 200 genes; chr2 genes 50..99 have halved counts in patient cells."""
    rng = np.random.default_rng(seed)
    n_pat, n_ref = 120, 120
    base = rng.poisson(5, size=(n_pat + n_ref, 600)).astype(float)
    deleted = np.arange(200 + 50, 200 + 100)          # chr2, gene indices 250..299
    base[:n_pat][:, deleted] = rng.poisson(2.5, size=(n_pat, 50))
    ad = anndata.AnnData(X=base)
    ad.var_names = [f"G{i}" for i in range(600)]
    ad.obs["group"] = ["patient"] * n_pat + ["reference"] * n_ref
    ad.layers["counts"] = ad.X.copy()
    genes = pd.DataFrame({
        "gene_name": ad.var_names,
        "gene_id": [f"ENSG{i}" for i in range(600)],
        "chrom": ["chr1"] * 200 + ["chr2"] * 200 + ["chr3"] * 200,
        "start": [i * 10000 for i in range(200)] * 3,
        "end": [i * 10000 + 9000 for i in range(200)] * 3,
        "strand": ["+"] * 600,
    })
    return ad, genes


def test_infercnv_recovers_planted_deletion():
    ad, genes = _synthetic()
    ad = normalize_for_cnv(ad)
    ad = order_by_position(ad, genes)
    run_infercnv(ad, window_size=50, step=10)
    tab = window_table(ad, window_size=50, step=10)
    segs = call_deletion_segments(tab[["chr", "start", "end", "score"]],
                                  z_threshold=1.5, min_windows=2)
    assert not segs.empty
    hit = segs[(segs["chr"] == "chr2") & (segs["start"] < 60 * 10000) &
               (segs["end"] > 40 * 10000)]
    assert len(hit) >= 1, f"planted chr2 deletion not recovered; got {segs.to_dict('records')}"


def test_window_table_count_matches_x_cnv():
    ad, genes = _synthetic()
    ad = normalize_for_cnv(ad)
    ad = order_by_position(ad, genes)
    run_infercnv(ad, window_size=50, step=10)
    tab = window_table(ad, window_size=50, step=10)
    # 3 chromosomes × ((200-50)//10 + 1) = 3 × 16 windows
    assert len(tab) == 48
    assert len(tab) == ad.obsm["X_cnv"].shape[1]
