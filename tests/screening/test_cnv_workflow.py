"""Known-interval entry, output lifecycle and stage provenance regressions."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from screening.src.cnv import workflow as w


@pytest.fixture
def cfg(tmp_path):
    order = tmp_path / "order.tsv"
    pd.DataFrame([{"gene_name": "A", "gene_id": "ID1", "chrom": "chr1",
                   "start": 100, "end": 200, "strand": "+"},
                  {"gene_name": "B", "gene_id": "ID2", "chrom": "chr1",
                   "start": 300, "end": 400, "strand": "+"}]).to_csv(order, sep="\t", index=False)
    return {"project": "TEST", "annotation": {"gene_order_tsv": str(order), "genome_build": "GRCh38"},
            "known_intervals": {"del1": {"chr": "1", "start": 200, "end": 300, "genome_build": "GRCh38"}},
            "extract": {"mode": "intervals-only"}, "output": {"dir": str(tmp_path / "out")}}


def _out(cfg):
    return Path(cfg["output"]["dir"])


def _audit(cfg, stage="extract_genes"):
    return json.loads((_out(cfg) / "audit" / f"run_{stage}.json").read_text())


def test_intervals_only_without_samples_ignores_cnv_artifacts(cfg):
    out = _out(cfg)
    out.mkdir()
    for name in ("segments_all.csv", "cnv_input.h5ad", "heatmap_patient.png"):
        (out / name).write_bytes(b"unrelated old artifact")
    cand, qc, val = w.cmd_extract_genes(cfg)
    assert cand.gene_symbol.tolist() == ["A", "B"]
    assert cand.deletion_id.tolist() == ["del1", "del1"]
    assert qc.empty
    assert val["AC-CNV-1 (auto recovers known interval)"].tolist() == ["NOT_EVALUATED"]
    assert val["AC-CNV-2 (markers detected in data)"].tolist() == ["NOT_EVALUATED"]
    report = (out / "cnv_report.md").read_text()
    assert "intervals-only" in report and "NOT_EVALUATED" in report
    assert "FAIL" not in report and "heatmap_patient.png" not in report
    assert (out / "cnv_input.h5ad").read_bytes() == b"unrelated old artifact"
    assert list(_audit(cfg)["inputs"]) == ["gene_order"]
    assert _audit(cfg)["validation"][0]["auto_overlap_frac"] is None


@pytest.mark.parametrize("change", [{"gene_id": "OTHER"}, {"start": 110}, {"end": 210},
                                    {"strand": "-"}, {"gene_id": ""}, {"gene_name": "ALIAS_A"}])
def test_duplicate_gene_identity_conflict_fails_and_archives_sources(cfg, change):
    w.cmd_extract_genes(cfg)
    previous = _out(cfg) / "candidate_provenance.csv"
    old = previous.read_bytes() if previous.exists() else b"missing provenance"
    order = Path(cfg["annotation"]["gene_order_tsv"])
    genes = pd.read_csv(order, sep="\t")
    duplicate = genes.iloc[0].to_dict()
    duplicate.update(change)
    pd.concat([genes, pd.DataFrame([duplicate])], ignore_index=True).to_csv(order, sep="\t", index=False)
    with pytest.raises(ValueError, match="gene identity.*A"):
        w.cmd_extract_genes(cfg)
    assert _audit(cfg)["status"] == "failed"
    assert not (_out(cfg) / "candidates.csv").exists()
    assert not (_out(cfg) / "candidate_provenance.csv").exists()
    assert any(p.read_bytes() == old for p in (_out(cfg) / "history").rglob("candidate_provenance.csv"))


def test_identical_annotation_duplicates_and_empty_provenance(cfg):
    order = Path(cfg["annotation"]["gene_order_tsv"])
    genes = pd.read_csv(order, sep="\t")
    pd.concat([genes, genes.iloc[[0]]], ignore_index=True).to_csv(order, sep="\t", index=False)
    cand, _, _ = w.cmd_extract_genes(cfg)
    assert cand.gene_symbol.tolist() == ["A", "B"]
    detail = pd.read_csv(_out(cfg) / "candidate_provenance.csv")
    assert len(detail) == 3
    cfg["known_intervals"] = {}
    w.cmd_extract_genes(cfg)
    empty = pd.read_csv(_out(cfg) / "candidate_provenance.csv")
    assert empty.empty and empty.columns.tolist() == detail.columns.tolist()


@pytest.mark.parametrize("include_auto", [False, True])
def test_known_and_auto_overlap_preserves_separate_sources(cfg, include_auto):
    from screening.src.cnv.artifacts import StageRun
    cfg["extract"].update(mode="cnv", include_auto_in_candidates=include_auto)
    cfg["known_intervals"] = {"auto_standard_0": cfg["known_intervals"]["del1"]}
    _seed_infercnv(cfg)
    seg = _out(cfg) / "segments_all.csv"
    with StageRun(cfg, "call_segments", [seg], w._stage_parameters(cfg, "call_segments")) as run:
        run.upstream("infercnv", w._stage_parameters(cfg, "infercnv"))
        pd.DataFrame([{"chr": "chr1", "start": 100, "end": 210, "resolution": "standard"}]).to_csv(seg, index=False)
    cand, _, val = w.cmd_extract_genes(cfg)
    assert cand.gene_symbol.tolist() == ["A", "B"]
    assert val.n_genes.tolist() == [2]
    detail = pd.read_csv(_out(cfg) / "candidate_provenance.csv")
    assert len(detail) == 3
    assert detail.loc[detail.source == "auto_segment", "in_candidates"].tolist() == [include_auto]
    all_cand = pd.read_csv(_out(cfg) / "candidates_all.csv")
    origins = json.loads(all_cand.set_index("gene_symbol").loc["A", "candidate_provenance"])
    assert {r["source"] for r in origins} == {"known_interval", "auto_segment"}
    if include_auto:
        assert cand.set_index("gene_symbol").loc["A", "source"] == "multiple"
    else:
        assert "candidate_provenance" not in cand.columns


@pytest.mark.parametrize("empty_intervals", [False, True])
def test_empty_results_have_headers_and_old_results_are_archived(cfg, empty_intervals):
    w.cmd_extract_genes(cfg)
    old = (_out(cfg) / "candidates.csv").read_bytes()
    if empty_intervals:
        cfg["known_intervals"] = {}
    else:
        cfg["known_intervals"]["del1"].update(start=500, end=600)
    w.cmd_extract_genes(cfg)
    for name in ("candidates.csv", "candidates_all.csv", "auto_segment_genes.csv"):
        table = pd.read_csv(_out(cfg) / name)
        assert table.empty
        assert table.columns.tolist() == w.CANDIDATE_COLUMNS + ([] if name == "candidates.csv" else ["gene_id"])
    assert pd.read_csv(_out(cfg) / "interval_gene_expression_qc.csv").empty
    assert any(p.read_bytes() == old for p in (_out(cfg) / "history").rglob("candidates.csv"))
    assert _audit(cfg)["status"] == "success"


@pytest.mark.parametrize("change", [{"start": 0}, {"start": 400, "end": 300},
                                    {"start": 1.5}, {"genome_build": "GRCh37"},
                                    {"coordinate_system": "0-based-half-open"}])
def test_invalid_interval_fails_explicitly(cfg, change):
    cfg["known_intervals"]["del1"].update(change)
    with pytest.raises(ValueError):
        w.cmd_extract_genes(cfg)
    assert not (_out(cfg) / "candidates.csv").exists()
    assert _audit(cfg)["status"] == "failed"


def test_failed_repeat_does_not_leave_success_outputs(cfg, monkeypatch):
    w.cmd_extract_genes(cfg)
    old_id = _audit(cfg)["execution_id"]
    def fail(*args, **kwargs):
        raise RuntimeError("report write failed")
    monkeypatch.setattr(w, "_write_report", fail)
    with pytest.raises(RuntimeError, match="report write failed"):
        w.cmd_extract_genes(cfg)
    assert not (_out(cfg) / "candidates.csv").exists()
    assert _audit(cfg)["status"] == "failed"
    assert _audit(cfg)["outputs"] == {}
    assert (_out(cfg) / "audit" / "records" / f"{old_id}.json").exists()
    assert list((_out(cfg) / "history").rglob("candidates.csv"))


def test_audit_records_real_input_output_and_effective_parameters(cfg):
    w.cmd_extract_genes(cfg)
    audit = _audit(cfg)
    assert audit["inputs"]["gene_order"]["sha256"] == w.sha256_file(cfg["annotation"]["gene_order_tsv"])
    assert audit["outputs"]["candidates.csv"]["sha256"] == w.sha256_file(_out(cfg) / "candidates.csv")
    assert audit["parameters"]["mode"] == "intervals-only"
    assert audit["parameters"]["known_intervals"] == cfg["known_intervals"]
    assert audit["parameters"]["coordinate_system"] == "1-based-inclusive"
    assert audit["upstream"] == {}
    w.cmd_extract_genes(cfg)
    assert _audit(cfg)["execution_id"] != audit["execution_id"]


def test_cli_intervals_only_and_invalid_input(cfg, tmp_path):
    path = tmp_path / "config.yaml"
    cfg["extract"].pop("mode")
    path.write_text(yaml.safe_dump(cfg))
    command = [sys.executable, "-m", "screening.src.cnv.workflow", "extract-genes", "--config", str(path),
               "--mode", "intervals-only"]
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cfg["known_intervals"]["del1"]["start"] = 0
    path.write_text(yaml.safe_dump(cfg))
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    assert result.returncode == 2 and "del1" in result.stderr
    assert "Traceback" not in result.stderr


def _seed_infercnv(cfg):
    """Synthetic upstream producer, without running inference or changing its algorithm."""
    from screening.src.cnv.artifacts import StageRun
    out = _out(cfg)
    signal = out / "window_signal_standard.tsv"
    with StageRun(cfg, "infercnv", [signal], w._stage_parameters(cfg, "infercnv")) as run:
        run.input("gene_order", cfg["annotation"]["gene_order_tsv"])
        pd.DataFrame({"chr": ["chr1"] * 4, "start": [100, 200, 300, 400],
                      "end": [199, 299, 399, 499], "score": [0., 0., 0., 0.]}).to_csv(signal, sep="\t", index=False)
    return signal


def test_obsolete_resolution_output_is_archived_on_rerun(cfg):
    from screening.src.cnv.artifacts import StageRun
    obsolete = _out(cfg) / "segments_old.csv"
    with StageRun(cfg, "call_segments", [obsolete], {}) as run:
        obsolete.write_text("old resolution\n")
    _seed_infercnv(cfg)
    w.cmd_call_segments(cfg)
    assert not obsolete.exists()
    assert any(p.read_text() == "old resolution\n"
               for p in (_out(cfg) / "history").rglob("segments_old.csv"))
    assert obsolete.name not in _audit(cfg, "call_segments")["outputs"]


def test_segment_lineage_allows_downstream_changes_but_rejects_modified_input(cfg):
    cfg["extract"]["mode"] = "cnv"
    signal = _seed_infercnv(cfg)
    parent = _audit(cfg, "infercnv")
    cfg["segments"] = {"z_threshold": 2.5}
    w.cmd_call_segments(cfg)
    audit = _audit(cfg, "call_segments")
    assert audit["upstream"]["infercnv"]["execution_id"] == parent["execution_id"]
    assert audit["inputs"]["window_signal_standard.tsv"]["sha256"] == w.sha256_file(signal)
    assert audit["parameters"]["z_threshold"] == 2.5
    signal.write_text(signal.read_text() + "chr1\t500\t599\t9\n")
    with pytest.raises(ValueError, match="checksum"):
        w.cmd_call_segments(cfg)
    assert _audit(cfg, "call_segments")["status"] == "failed"


@pytest.mark.parametrize("change", ["project", "infercnv", "legacy"])
def test_upstream_identity_parameters_and_legacy_audits_rejected(cfg, change):
    _seed_infercnv(cfg)
    if change == "project":
        cfg["project"] = "OTHER_STUDY"
    elif change == "infercnv":
        cfg["infercnv"] = {"reference_group": "other"}
    else:
        (_out(cfg) / "audit" / "run_infercnv.json").write_text('{"stage": "infercnv"}')
    with pytest.raises(ValueError, match="upstream"):
        w.cmd_call_segments(cfg)


def test_cnv_without_execution_is_not_evaluated(cfg):
    cfg["extract"]["mode"] = "cnv"
    cfg["samples"] = []
    _, _, val = w.cmd_extract_genes(cfg)
    assert val["AC-CNV-1 (auto recovers known interval)"].tolist() == ["NOT_EVALUATED"]
    assert val["AC-CNV-2 (markers detected in data)"].tolist() == ["NOT_EVALUATED"]


@pytest.mark.parametrize("change", ["new_inference", "gene_order"])
def test_extract_rejects_mixed_upstream_versions(cfg, change):
    cfg["extract"]["mode"] = "cnv"
    _seed_infercnv(cfg)
    w.cmd_call_segments(cfg)
    if change == "new_inference":
        _seed_infercnv(cfg)
    else:
        order = Path(cfg["annotation"]["gene_order_tsv"])
        order.write_text(order.read_text().replace("ID1", "CHANGED"))
    with pytest.raises(ValueError, match="upstream"):
        w.cmd_extract_genes(cfg)
    assert _audit(cfg)["status"] == "failed"


def test_valid_cnv_segment_chain_and_downstream_interval_change(cfg):
    cfg["extract"]["mode"] = "cnv"
    _seed_infercnv(cfg)
    w.cmd_call_segments(cfg)
    cfg["known_intervals"]["del1"].update(start=300, end=400)
    cand, _, _ = w.cmd_extract_genes(cfg)
    assert cand.gene_symbol.tolist() == ["B"]
    assert set(_audit(cfg)["upstream"]) == {"call_segments", "infercnv"}


def test_prepare_order_has_provenance_and_changed_table_rejected(cfg, tmp_path):
    gtf = tmp_path / "genes.gtf"
    gtf.write_text('chr1\ttest\tgene\t100\t200\t.\t+\t.\tgene_id "ID1"; gene_name "A"; gene_type "protein_coding";\n')
    cfg["annotation"]["gtf"] = str(gtf)
    w.cmd_prepare_order(cfg)
    w.cmd_extract_genes(cfg)
    assert "prepare_order" in _audit(cfg)["upstream"]
    order = Path(cfg["annotation"]["gene_order_tsv"])
    order.write_text(order.read_text().replace("ID1", "OTHER"))
    with pytest.raises(ValueError, match="checksum"):
        w.cmd_extract_genes(cfg)


def test_missing_order_built_from_gtf_without_cnv(cfg, tmp_path):
    gtf = tmp_path / "genes.gtf"
    gtf.write_text('chr1\ttest\tgene\t100\t200\t.\t+\t.\tgene_id "ID1"; gene_name "A"; gene_type "protein_coding";\n')
    cfg["annotation"].update(gtf=str(gtf), gene_order_tsv=str(tmp_path / "new-order.tsv"))
    cand, _, _ = w.cmd_extract_genes(cfg)
    assert cand.gene_symbol.tolist() == ["A"]
    assert _audit(cfg, "prepare_order")["inputs"]["gtf"]["sha256"] == w.sha256_file(gtf)


def test_overlapping_input_output_is_rejected_without_moving_input(cfg):
    out = _out(cfg)
    out.mkdir()
    source = out / "candidates.csv"
    source.write_bytes(Path(cfg["annotation"]["gene_order_tsv"]).read_bytes())
    before = source.read_bytes()
    cfg["annotation"]["gene_order_tsv"] = str(source)
    with pytest.raises(ValueError, match="overlap"):
        w.cmd_extract_genes(cfg)
    assert source.read_bytes() == before


def test_input_mutation_during_run_fails_and_archives_new_outputs(cfg, monkeypatch):
    original = w._write_report
    def mutate(*args, **kwargs):
        original(*args, **kwargs)
        path = Path(cfg["annotation"]["gene_order_tsv"])
        path.write_text(path.read_text().replace("ID1", "OTHER"))
    monkeypatch.setattr(w, "_write_report", mutate)
    with pytest.raises(ValueError, match="input changed"):
        w.cmd_extract_genes(cfg)
    assert not (_out(cfg) / "candidates.csv").exists()
    assert _audit(cfg)["status"] == "failed"


def test_intervals_only_rejects_auto_candidates(cfg):
    cfg["extract"]["include_auto_in_candidates"] = True
    with pytest.raises(ValueError, match="cannot include auto"):
        w.cmd_extract_genes(cfg)


def test_optional_heatmap_failure_keeps_main_outputs_and_archives_old_and_partial(cfg, monkeypatch):
    from screening.src.cnv import infercnv as ic
    from types import SimpleNamespace

    cfg["samples"] = []
    for sample_id, group in [("S1", "patient"), ("S2", "reference")]:
        sample = _out(cfg).parent / sample_id
        sample.mkdir()
        for name in ("matrix.mtx.gz", "barcodes.tsv.gz", "features.tsv.gz"):
            (sample / name).write_bytes(b"synthetic input for mocked inference")
        cfg["samples"].append({"id": sample_id, "group": group, "path": str(sample)})
    adata = SimpleNamespace(n_obs=2, n_vars=2, obs=pd.DataFrame({"group": ["patient", "reference"]}),
                            uns={"gene_position_mapping": pd.DataFrame({"status": ["positioned"] * 2})},
                            write_h5ad=lambda path: path.write_bytes(b"synthetic h5ad"))
    monkeypatch.setattr(ic, "load_samples", lambda *a, **k: adata)
    monkeypatch.setattr(ic, "normalize_for_cnv", lambda data: data)
    monkeypatch.setattr(ic, "order_by_position", lambda data, genes: data)
    monkeypatch.setattr(ic, "run_infercnv", lambda *a, **k: None)
    monkeypatch.setattr(ic, "window_table", lambda *a, **k: pd.DataFrame(
        {"chr": ["chr1"], "start": [100], "end": [200], "score": [0.], "ref_score": [0.]}))
    def good_heatmap(data, config, out):
        (out / "heatmap_patient.png").write_bytes(b"previous successful image")
        return {"status": "success"}
    monkeypatch.setattr(w, "_save_heatmap", good_heatmap)
    w.cmd_infercnv(cfg)
    def bad_heatmap(data, config, out):
        (out / "heatmap_patient.png").write_bytes(b"incomplete image")
        return {"status": "failed", "reason": "test renderer failed"}
    monkeypatch.setattr(w, "_save_heatmap", bad_heatmap)
    w.cmd_infercnv(cfg)
    audit = _audit(cfg, "infercnv")
    assert audit["status"] == "success"
    assert audit["heatmap"]["status"] == "failed"
    assert "heatmap_patient.png" not in audit["outputs"]
    assert not (_out(cfg) / "heatmap_patient.png").exists()
    assert (_out(cfg) / "cnv_input.h5ad").exists()
    assert "gene_position_mapping.csv" in audit["outputs"]
    assert len(list((_out(cfg) / "history").rglob("gene_position_mapping.csv"))) == 1
    assert {p.read_bytes() for p in (_out(cfg) / "history").rglob("heatmap_patient.png")} == {
        b"previous successful image", b"incomplete image"}


def test_cnv_qc_failure_is_not_silently_treated_as_empty(cfg, monkeypatch):
    import anndata
    from screening.src.cnv.artifacts import StageRun
    cfg["extract"]["mode"] = "cnv"
    h5ad = _out(cfg) / "cnv_input.h5ad"
    with StageRun(cfg, "infercnv", [h5ad], w._stage_parameters(cfg, "infercnv")) as run:
        run.input("gene_order", cfg["annotation"]["gene_order_tsv"])
        h5ad.write_bytes(b"corrupt expression data")
    def fail(*args, **kwargs):
        raise OSError("cannot read expression data")
    monkeypatch.setattr(anndata, "read_h5ad", fail)
    with pytest.raises(OSError, match="cannot read expression"):
        w.cmd_extract_genes(cfg)
    assert _audit(cfg)["status"] == "failed"
    assert not (_out(cfg) / "candidates.csv").exists()


def test_failed_upstream_repeat_invalidates_downstream_candidates(cfg):
    cfg["extract"]["mode"] = "cnv"
    signal = _seed_infercnv(cfg)
    w.cmd_call_segments(cfg)
    w.cmd_extract_genes(cfg)
    assert (_out(cfg) / "candidates.csv").exists()
    signal.write_text("changed source")
    with pytest.raises(ValueError, match="checksum"):
        w.cmd_call_segments(cfg)
    assert not (_out(cfg) / "candidates.csv").exists()
    assert _audit(cfg)["status"] == "invalidated"
    assert list((_out(cfg) / "history").rglob("candidates.csv"))


def test_cnv_rerun_does_not_invalidate_independent_interval_results(cfg):
    w.cmd_extract_genes(cfg)
    before = (_out(cfg) / "candidates.csv").read_bytes()
    execution = _audit(cfg)["execution_id"]
    _seed_infercnv(cfg)
    w.cmd_call_segments(cfg)
    assert (_out(cfg) / "candidates.csv").read_bytes() == before
    assert _audit(cfg)["execution_id"] == execution
