"""Unavailable evidence must not be reported as a biological negative result."""
import numpy as np
import pandas as pd
import pytest

from screening.src.cnv import workflow as w
from screening.src.cnv.artifacts import StageRun
from screening.src.cnv.infercnv import InsufficientDataError
from .test_cnv_workflow import cfg, _out, _audit, _seed_infercnv

AC1 = "AC-CNV-1 (auto recovers known interval)"
AC2 = "AC-CNV-2 (markers detected in data)"


def test_missing_cnv_evidence_is_not_evaluated(cfg):
    cfg["extract"]["mode"] = "cnv"
    _, _, val = w.cmd_extract_genes(cfg)
    assert val[AC1].tolist() == ["NOT_EVALUATED"]
    assert val[AC2].tolist() == ["NOT_EVALUATED"]
    assert pd.isna(val.auto_overlap_frac.iloc[0])
    assert "FAIL" not in (_out(cfg) / "cnv_report.md").read_text()


@pytest.mark.parametrize("error,category", [(OSError("broken input"), "ERROR"),
                                           (InsufficientDataError("no cells after QC"), "INSUFFICIENT_DATA")])
def test_failed_inference_is_explicit_and_cannot_look_unexecuted(cfg, error, category):
    cfg["extract"]["mode"] = "cnv"
    with pytest.raises(type(error)):
        with StageRun(cfg, "infercnv", [], w._stage_parameters(cfg, "infercnv")):
            raise error
    assert _audit(cfg, "infercnv")["validation_status"] == category
    with pytest.raises((ValueError, InsufficientDataError), match="upstream"):
        w.cmd_extract_genes(cfg)
    assert _audit(cfg)["validation_status"] == category
    assert not (_out(cfg) / "candidates.csv").exists()


def test_completed_empty_segments_are_actual_fail(cfg):
    cfg["extract"]["mode"] = "cnv"
    _seed_infercnv(cfg)
    w.cmd_call_segments(cfg)
    _, _, val = w.cmd_extract_genes(cfg)
    assert val[AC1].tolist() == ["FAIL"]
    assert val.auto_overlap_frac.tolist() == [0.]
    assert val[AC2].tolist() == ["NOT_EVALUATED"]


def test_interval_outside_available_windows_is_insufficient(cfg):
    cfg["extract"]["mode"] = "cnv"
    _seed_infercnv(cfg)
    w.cmd_call_segments(cfg)
    cfg["known_intervals"]["del1"]["chr"] = "chr2"
    _, _, val = w.cmd_extract_genes(cfg)
    assert val[AC1].tolist() == ["INSUFFICIENT_DATA"]
    assert pd.isna(val.auto_overlap_frac.iloc[0])


def test_deleted_recorded_expression_is_error_not_unexecuted(cfg):
    cfg["extract"]["mode"] = "cnv"
    h5ad = _out(cfg) / "cnv_input.h5ad"
    with StageRun(cfg, "infercnv", [h5ad], w._stage_parameters(cfg, "infercnv")) as run:
        run.input("gene_order", cfg["annotation"]["gene_order_tsv"])
        h5ad.write_bytes(b"previously produced expression data")
    h5ad.unlink()
    with pytest.raises(ValueError, match="missing upstream"):
        w.cmd_extract_genes(cfg)
    assert _audit(cfg)["validation_status"] == "ERROR"
    assert not (_out(cfg) / "candidates.csv").exists()


@pytest.mark.parametrize("chrom", ["1", "chr1", 1])
def test_overlap_accepts_same_chromosome_labels(chrom):
    from screening.src.cnv.segments import segment_overlap_frac
    assert segment_overlap_frac("chr1", 100, 300, chrom, 100, 300) == 1.


@pytest.mark.parametrize("markers,expected", [([], "INSUFFICIENT_DATA"), (["A"], "PASS"), (["B"], "FAIL")])
def test_marker_check_requires_requested_markers_and_expression(cfg, markers, expected):
    import anndata
    cfg["extract"]["mode"] = "cnv"
    cfg["known_intervals"]["del1"]["markers"] = markers
    h5ad = _out(cfg) / "cnv_input.h5ad"
    ad = anndata.AnnData(np.array([[1.], [2.]]),
                        obs=pd.DataFrame({"group": ["patient", "reference"]}, index=["p", "r"]),
                        var=pd.DataFrame(index=["A"]))
    ad.layers["counts"] = ad.X.copy()
    with StageRun(cfg, "infercnv", [h5ad], w._stage_parameters(cfg, "infercnv")) as run:
        run.input("gene_order", cfg["annotation"]["gene_order_tsv"])
        ad.write_h5ad(h5ad)
    _, _, val = w.cmd_extract_genes(cfg)
    assert val[AC2].tolist() == [expected]
