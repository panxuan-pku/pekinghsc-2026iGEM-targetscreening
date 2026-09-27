"""Gene IDs, rather than shared symbols, determine count and position lookup."""
import json

import anndata
import numpy as np
import pandas as pd
import pytest

from screening.src.cnv import infercnv as ic
from screening.src.cnv.expression_qc import interval_gene_qc
from .test_cnv_input_contract import _sample, _write


def _adata():
    ad = anndata.AnnData(np.array([[3, 7, 1], [6, 2, 4]], dtype=np.float32),
                        var=pd.DataFrame({"gene_ids": ["ID_A", "ID_B", "ID_C"],
                                          "gene_symbols": ["SHARED", "SHARED", "OTHER"]},
                                         index=["ID_A", "ID_B", "ID_C"]))
    ad.obs["group"] = ["patient", "reference"]
    ad.layers["counts"] = ad.X.copy()
    return ad


def _genes():
    # Annotation order and names differ from input; only the IDs are authoritative.
    return pd.DataFrame({"gene_id": ["ID_B", "ID_A"], "gene_name": ["RENAMED", "SHARED"],
                         "chrom": ["chr1", "chr1"], "start": [10, 30], "end": [20, 40]})


@pytest.mark.parametrize("legacy", [False, True])
def test_shared_symbols_load_all_ids_and_counts_without_suffixes(tmp_path, legacy):
    samples = [_sample(tmp_path, name, group, legacy) for name, group in
               [("P", "patient"), ("R", "reference")]]
    for sample in samples:
        name = "genes.tsv" if legacy else "features.tsv.gz"
        rows = "ID1\tSHARED\nID2\tSHARED\n"
        _write(tmp_path / sample["id"] / name,
               rows if legacy else rows.replace("\n", "\tGene Expression\n"))
    ad = ic.load_samples(samples, min_genes=1, max_mito_pct=100)
    assert ad.var_names.tolist() == ["ID1", "ID2"]
    assert ad.var.gene_symbols.tolist() == ["SHARED", "SHARED"]
    assert ad.var.gene_ids.tolist() == ["ID1", "ID2"]
    np.testing.assert_array_equal(ad.layers["counts"].toarray(), [[9, 3], [4, 7]] * 2)
    assert [json.loads(n) for n in ad.obs_names] == [[s, b] for s in ["P", "R"] for b in ["CELL1", "CELL2"]]
    path = tmp_path / "roundtrip.h5ad"
    ad.write_h5ad(path)
    reread = anndata.read_h5ad(path)
    pd.testing.assert_frame_equal(ad.var, reread.var, check_categorical=False)
    np.testing.assert_array_equal(reread.layers["counts"].toarray(), ad.layers["counts"].toarray())


def test_positioning_uses_ids_and_audits_every_input_row():
    ad = _adata()
    before = ad.layers["counts"].copy()
    positioned = ic.order_by_position(ad, _genes())
    assert positioned.var_names.tolist() == ["ID_B", "ID_A"]
    assert positioned.var.gene_symbols.tolist() == ["SHARED", "SHARED"]
    assert positioned.var.start.tolist() == [10, 30]
    np.testing.assert_array_equal(positioned.layers["counts"], before[:, [1, 0]])
    np.testing.assert_array_equal(ad.layers["counts"], before)
    audit = positioned.uns["gene_position_mapping"]
    assert audit.gene_id.tolist() == ["ID_A", "ID_B", "ID_C"]
    assert audit.gene_symbol.tolist() == ["SHARED", "SHARED", "OTHER"]
    assert audit.status.tolist() == ["positioned", "positioned", "not_in_supplied_annotation"]
    assert audit.annotation_symbol.tolist() == ["SHARED", "RENAMED", ""]


def test_same_id_at_two_positions_is_an_explicit_error():
    genes = _genes()
    genes.loc[1, "gene_id"] = "ID_B"
    with pytest.raises(ValueError, match="ambiguous.*ID_B"):
        ic.order_by_position(_adata(), genes)


def test_id_input_never_falls_back_to_symbol_only_annotation():
    with pytest.raises(ValueError, match="gene_id"):
        ic.order_by_position(_adata(), _genes().drop(columns="gene_id"))


@pytest.mark.parametrize("problem", ["duplicate", "missing"])
def test_invalid_matrix_ids_are_rejected_by_positioning(problem):
    ad = _adata()
    ad.var.loc["ID_B", "gene_ids"] = "ID_A" if problem == "duplicate" else ""
    with pytest.raises(ValueError, match="gene ID"):
        ic.order_by_position(ad, _genes())


def test_qc_selects_each_id_even_when_symbols_are_shared_or_renamed():
    genes = _genes()
    qc = interval_gene_qc(_adata(), genes, "group", "patient", "reference")
    assert qc.gene_id.tolist() == ["ID_B", "ID_A"]
    assert qc.gene_name.tolist() == ["RENAMED", "SHARED"]
    assert qc.gene_symbol.tolist() == ["SHARED", "SHARED"]
    assert qc.found.tolist() == [True, True]
    assert qc.mean_counts_patient.tolist() == [7, 3]
    assert qc.mean_counts_ref.tolist() == [2, 6]


def test_qc_does_not_substitute_a_matching_symbol_for_an_absent_id():
    genes = _genes().iloc[[1]].copy()
    genes["gene_id"] = "ABSENT"
    qc = interval_gene_qc(_adata(), genes, "group", "patient", "reference")
    assert qc.found.tolist() == [False]
    assert qc.mean_counts_patient.isna().all()


def test_window_table_retains_symbols_and_separate_ids():
    ad = ic.order_by_position(_adata(), _genes())
    ad.obsm["X_cnv"] = np.array([[-0.2], [0.0]])
    table = ic.window_table(ad, window_size=2, step=1)
    assert table.genes.tolist() == ["SHARED,SHARED"]
    assert table.gene_ids.tolist() == ["ID_B,ID_A"]
    assert table.score.tolist() == [-0.2]


def test_mitochondrial_qc_still_uses_original_symbols(tmp_path):
    sample = _sample(tmp_path)
    _write(tmp_path / "P/features.tsv.gz", "ID1\tMT-TEST\tGene Expression\nID2\tGENE2\tGene Expression\n")
    # CELL1 has 75% mitochondrial counts; CELL2 has 4/11, so only CELL2 remains.
    ad = ic.load_samples([sample], min_genes=1, max_mito_pct=50)
    assert ad.obs.barcode.tolist() == ["CELL2"]
    np.testing.assert_array_equal(ad.layers["counts"].toarray(), [[4, 7]])


def test_qc_rejects_conflicting_annotation_for_one_id():
    genes = _genes()
    genes.loc[1, 'gene_id'] = 'ID_B'
    with pytest.raises(ValueError, match='ambiguous.*ID_B'):
        interval_gene_qc(_adata(), genes, 'group', 'patient', 'reference')


@pytest.mark.parametrize('column,value', [('chrom', 'chrUnknown'), ('start', np.nan),
                                         ('end', 0), ('start', 10.5)])
def test_positioning_reports_invalid_coordinates_with_id(column, value):
    genes = _genes()
    if column in ('start', 'end'):
        genes[column] = genes[column].astype(float)
    genes.loc[0, column] = value
    with pytest.raises(ValueError, match='position.*ID_B'):
        ic.order_by_position(_adata(), genes)
