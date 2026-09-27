"""Raw 10x counts must retain unambiguous row/column identities."""
import gzip
import json

import numpy as np
import pytest

from screening.src.cnv import infercnv as ic


def _write(path, text):
    if path.suffix == ".gz":
        with gzip.open(path, "wt") as stream:
            stream.write(text)
    else:
        path.write_text(text)


def _sample(root, name="P", group="patient", legacy=False):
    path = root / name
    path.mkdir()
    suffix = "" if legacy else ".gz"
    _write(path / ("matrix.mtx" + suffix),
           "%%MatrixMarket matrix coordinate real general\n2 2 4\n1 1 9\n2 1 3\n1 2 4\n2 2 7\n")
    _write(path / ("barcodes.tsv" + suffix), "CELL1\nCELL2\n")
    features = "ID1\tGENE1\nID2\tGENE2\n"
    _write(path / (("genes.tsv" if legacy else "features.tsv") + suffix),
           features if legacy else features.replace("\n", "\tGene Expression\n"))
    return {"id": name, "group": group, "path": str(path)}


@pytest.mark.parametrize("problem", ["nan", "inf", "negative", "fraction", "duplicate_id",
                                    "empty_id", "empty_symbol", "duplicate_barcode",
                                    "empty_barcode", "dimensions", "corrupt_matrix", "corrupt_barcodes",
                                    "corrupt_features", "feature_type", "no_genes", "no_cells"])
def test_bad_raw_input_has_sample_specific_error(tmp_path, problem):
    sample = _sample(tmp_path)
    path = tmp_path / "P"
    if problem in ("nan", "inf", "negative", "fraction"):
        value = {"nan": "nan", "inf": "inf", "negative": "-1", "fraction": "0.5"}[problem]
        _write(path / "matrix.mtx.gz",
               f"%%MatrixMarket matrix coordinate real general\n2 2 1\n1 1 {value}\n")
    elif problem.startswith("corrupt_"):
        name = {"corrupt_matrix": "matrix.mtx.gz", "corrupt_barcodes": "barcodes.tsv.gz",
                "corrupt_features": "features.tsv.gz"}[problem]
        (path / name).write_bytes(b"not gzip")
    elif problem in ("duplicate_barcode", "empty_barcode"):
        _write(path / "barcodes.tsv.gz", "CELL1\n" + ("CELL1\n" if problem == "duplicate_barcode" else "\n"))
    elif problem in ("dimensions", "no_genes", "no_cells"):
        shape = {"dimensions": "3 2", "no_genes": "0 2", "no_cells": "2 0"}[problem]
        _write(path / "matrix.mtx.gz", f"%%MatrixMarket matrix coordinate real general\n{shape} 0\n")
    else:
        rows = {"duplicate_id": "ID1\tGENE1\nID1\tGENE2\n",
                "empty_id": "\tGENE1\nID2\tGENE2\n",
                "empty_symbol": "ID1\t\nID2\tGENE2\n",
                "feature_type": "ID1\tGENE1\nID2\tGENE2\n"}[problem]
        rows = rows.replace("\n", "\tGene Expression\n")
        if problem == "feature_type":
            rows = rows.replace("Gene Expression", "Antibody Capture", 1)
        _write(path / "features.tsv.gz", rows)
    with pytest.raises(ValueError, match="sample P:"):
        ic.load_samples([sample], min_genes=1, max_mito_pct=100)


@pytest.mark.parametrize("problem", ["gene_order", "gene_set", "symbol_mapping"])
def test_mismatched_samples_are_not_silently_aligned_or_zero_filled(tmp_path, problem):
    samples = [_sample(tmp_path, "P"), _sample(tmp_path, "R", "reference")]
    rows = {"gene_order": "ID2\tGENE2\nID1\tGENE1\n",
            "gene_set": "ID1\tGENE1\nID3\tGENE3\n",
            "symbol_mapping": "ID1\tGENE2\nID2\tGENE1\n"}[problem]
    _write(tmp_path / "R" / "features.tsv.gz", rows.replace("\n", "\tGene Expression\n"))
    with pytest.raises(ValueError, match="sample R:.*gene.*(order|identity)"):
        ic.load_samples(samples, min_genes=1, max_mito_pct=100)


@pytest.mark.parametrize("legacy", [False, True])
def test_multisample_counts_identity_and_order_are_exact(tmp_path, legacy):
    samples = [_sample(tmp_path, name, "patient" if name.startswith("P") else "reference", legacy)
               for name in ("P1", "P2", "R1", "R2")]
    ad = ic.load_samples(samples, min_genes=1, max_mito_pct=100)
    assert ad.obs_names.is_unique
    assert ad.obs["sample"].tolist() == [s["id"] for s in samples for _ in range(2)]
    assert ad.obs["barcode"].tolist() == ["CELL1", "CELL2"] * 4
    assert [json.loads(n) for n in ad.obs_names] == [[s["id"], b] for s in samples for b in ("CELL1", "CELL2")]
    assert ad.var_names.tolist() == ["ID1", "ID2"]
    assert ad.var.gene_symbols.tolist() == ["GENE1", "GENE2"]
    assert ad.var["gene_ids"].tolist() == ["ID1", "ID2"]
    np.testing.assert_array_equal(ad.layers["counts"].toarray(), [[9, 3], [4, 7]] * 4)


def test_qc_cannot_silently_remove_a_sample(tmp_path):
    samples = [_sample(tmp_path, "P"), _sample(tmp_path, "R", "reference")]
    _write(tmp_path / "R" / "matrix.mtx.gz",
           "%%MatrixMarket matrix coordinate real general\n2 2 1\n1 1 1\n")
    with pytest.raises(ValueError, match="sample R:.*(QC|cells)"):
        ic.load_samples(samples, min_genes=2, max_mito_pct=100)


@pytest.mark.parametrize("groups,patient,reference", [(["patient"], "patient", "reference"),
                         (["reference"], "patient", "reference"),
                         (["patient", "reference"], "patient", "patient"),
                         (["patient", None], "patient", "reference")])
def test_required_groups_are_explicit(groups, patient, reference):
    with pytest.raises(ValueError, match="group"):
        ic.validate_groups(groups, patient, reference)


@pytest.mark.parametrize("legacy", [False, True])
def test_sample_counts_cannot_be_used_as_stage_output(tmp_path, legacy):
    from pathlib import Path
    from screening.src.cnv.artifacts import StageRun
    sample = _sample(tmp_path, legacy=legacy)
    matrix = Path(sample["path"]) / ("matrix.mtx" if legacy else "matrix.mtx.gz")
    before = matrix.read_bytes()
    with pytest.raises(ValueError, match="overlap"):
        StageRun({"samples": [sample], "output": {"dir": str(tmp_path / "out")}},
                 "infercnv", [matrix], {})
    assert matrix.read_bytes() == before


def test_positioning_rejects_zero_usable_genes(tmp_path):
    import pandas as pd
    ad = ic.load_samples([_sample(tmp_path)], min_genes=1, max_mito_pct=100)
    genes = pd.DataFrame({"gene_id": ["ID_OTHER"], "gene_name": ["OTHER"], "chrom": ["chr1"], "start": [1], "end": [2]})
    with pytest.raises(ValueError, match="no.*genes"):
        ic.order_by_position(ad, genes)


def test_ambiguous_annotation_is_not_resolved_by_first_row(tmp_path):
    import pandas as pd
    ad = ic.load_samples([_sample(tmp_path)], min_genes=1, max_mito_pct=100)
    genes = pd.DataFrame({"gene_id": ["ID1", "ID1"], "gene_name": ["GENE1", "GENE1"], "chrom": ["chr1", "chr2"],
                          "start": [1, 1], "end": [2, 2]})
    with pytest.raises(ValueError, match="ambiguous.*ID1"):
        ic.order_by_position(ad, genes)


def test_no_windows_is_data_insufficiency(tmp_path):
    import pandas as pd
    ad = ic.load_samples([_sample(tmp_path)], min_genes=1, max_mito_pct=100)
    genes = pd.DataFrame({"gene_id": ["ID1", "ID2"], "gene_name": ["GENE1", "GENE2"], "chrom": ["chr1"] * 2,
                          "start": [1, 3], "end": [2, 4]})
    ad = ic.order_by_position(ad, genes)
    with pytest.raises(ValueError, match="(insufficient|no).*window"):
        ic.run_infercnv(ad, window_size=100)
