"""Synthetic contracts for the merged pipeline; no scientific results or network."""
import json
from pathlib import Path

import pandas as pd
import pytest

from screening.src.screening_inputs import build_candidates, transcript_design
from screening.src.screening import join_constraint, select_modules, rank_candidates
from screening.src.opentargets import annotate_snapshot, fetch_snapshot


@pytest.fixture
def inputs(tmp_path):
    hgnc = tmp_path / "hgnc.tsv"
    hgnc.write_text("hgnc_id\tsymbol\tstatus\tensembl_gene_id\talias_symbol\n"
                    "HGNC:1\tA\tApproved\tENSG00000000001\tOLD_A\n"
                    "HGNC:2\tB\tApproved\tENSG00000000002\t\n")
    gtf = tmp_path / "annotation.gtf"
    lines = []
    for gid, symbol, strand, blocks in [
        ("ENSG00000000001", "A", "+", [(100, 199), (300, 329)]),
        ("ENSG00000000002", "B", "-", [(500, 528), (600, 699)]),
    ]:
        attrs = f'gene_id "{gid}.1"; gene_name "{symbol}"; gene_type "protein_coding";'
        def row(feature, start, end, extra=""):
            return f'chr4\ttest\t{feature}\t{start}\t{end}\t.\t{strand}\t.\t{attrs} {extra}\n'
        tx = f'transcript_id "ENST{gid[4:]}.1"; tag "MANE_Select";'
        lines.extend([row("gene", blocks[0][0], blocks[-1][1]),
                      row("transcript", blocks[0][0], blocks[-1][1], tx)])
        lines.extend(row("exon", s, e, tx) for s, e in blocks)
        # Coding ends include stop_codon: 30 bp UTR on +, 29 bp on -.
        coding = blocks[0] if strand == "+" else blocks[1]
        lines.append(row("CDS", coding[0] + (3 if strand == "-" else 0),
                         coding[1] - (3 if strand == "+" else 0), tx))
        stop = (coding[1]-2, coding[1]) if strand == "+" else (coding[0], coding[0]+2)
        lines.append(row("stop_codon", *stop, tx))
    gtf.write_text("".join(lines))
    return gtf, hgnc


def test_two_inputs_have_identical_identities_and_keep_original_labels(inputs):
    gtf, hgnc = inputs
    interval = build_candidates(gtf, hgnc, interval="chr4:100-699")
    listed = build_candidates(gtf, hgnc, genes=["OLD_A", "HGNC:2"])
    assert interval.gene_id.tolist() == listed.gene_id.tolist()
    assert interval.hgnc_id.tolist() == listed.hgnc_id.tolist()
    assert listed.input_symbol.tolist() == ["OLD_A", "HGNC:2"]
    assert interval.interval_overlap.tolist() == ["full", "full"]
    assert build_candidates(gtf, hgnc, interval="chr4:150-600").interval_overlap.tolist() == ["partial", "partial"]


@pytest.mark.parametrize("genes", [["UNKNOWN"], ["A", "OLD_A"], []])
def test_bad_or_duplicate_identity_is_reported(inputs, genes):
    with pytest.raises(ValueError):
        build_candidates(*inputs, genes=genes)


@pytest.mark.parametrize("interval", ["chr4:0-3", "chr4:9-2", "chr4:1-20trailing"])
def test_bad_coordinates_are_rejected(inputs, interval):
    with pytest.raises(ValueError):
        build_candidates(*inputs, interval=interval)


def test_utr_uses_spliced_exons_on_both_strands_and_includes_stop(inputs):
    candidates = build_candidates(*inputs, genes=["A", "B"])
    designs = transcript_design(inputs[0], candidates)
    assert designs.utr3_bp.tolist() == [30, 29]
    assert designs.transcript_id.str.endswith(".1").all()
    assert designs.transcript_choice.tolist() == ["MANE_Select", "MANE_Select"]


def test_constraint_joins_identifiers_selects_mane_and_rejects_ambiguity(inputs, tmp_path):
    candidates = build_candidates(*inputs, genes=["A", "B"])
    path = tmp_path / "gnomad.tsv"
    path.write_text("gene_id\ttranscript\tmane_select\tcanonical\tlof.pLI\tlof.oe_ci.upper\tconstraint_flags\tgene_flags\n"
                    "ENSG00000000001\tT1\ttrue\ttrue\t0.9\t0.2\t[]\t[]\n"
                    "ENSG00000000001\tT2\tfalse\tfalse\t0.1\t0.9\t[]\t[]\n"
                    "ENSG00000000002\tT3\ttrue\ttrue\t0.8\t0.3\t[\"no_exp_lof\"]\t[]\n")
    joined = join_constraint(candidates, path)
    assert joined.gnomad_LOEUF.iloc[0] == 0.2
    assert pd.isna(joined.gnomad_LOEUF.iloc[1])
    assert joined.gnomad_LOEUF_raw.iloc[1] == 0.3
    path.write_text(path.read_text().replace("T2\tfalse\tfalse", "T2\ttrue\ttrue"))
    with pytest.raises(ValueError, match="ambiguous"):
        join_constraint(candidates, path)


def evidence(inputs):
    frame = transcript_design(inputs[0], build_candidates(*inputs, genes=["A", "B"]))
    frame["clinGen_hi_score"] = [3.0, float("nan")]
    frame["clinGen_haploinsufficiency_raw"] = [3, 30]
    frame["gnomad_LOEUF"] = [0.1, 0.3]
    frame["gnomad_pLI"] = [0.9, 0.7]
    return frame


def test_only_one_ranking_ot_and_mouse_notes_cannot_change_it(inputs):
    frame = evidence(inputs)
    before, _ = rank_candidates(frame)
    frame["ot_disease_score"] = [0.01, 0.99]
    frame["impc_viability"] = ["viable", "lethal"]
    after, _ = rank_candidates(frame)
    assert before.consensus_score.tolist() == after.consensus_score.tolist()
    selected = select_modules(after, capacity=300, module_size=300)
    assert selected.selected.tolist() == [True, False]
    assert "IMPC" in selected.evidence_notes.iloc[0]
    assert "ClinGen" in selected.evidence_notes.iloc[1]
    assert selected.selection_reason.iloc[1] == "below_project_utr3_minimum"
    frame["utr3_bp"] = [30, 30]
    assert select_modules(rank_candidates(frame)[0], capacity=600, module_size=300).selected.all()


def test_ot_zero_no_record_error_are_distinct_and_gene_identity_checked():
    frame = pd.DataFrame({"gene_id": ["G1", "G2", "G3"]})
    snapshot = {"release": "26.09", "disease_id": "MONDO_0008684", "phenotype_ids": [],
                "records": {
                    "G1": {"status": "ok", "associations": [{"disease": {"id": "MONDO_0008684", "name": "WHS"}, "score": 0, "datasourceScores": []}]},
                    "G2": {"status": "ok", "associations": []},
                    "G3": {"status": "query_failed", "error": "timeout"}}}
    out = annotate_snapshot(frame, snapshot, "MONDO_0008684", [], "26.09")
    assert out.ot_status.tolist() == ["available", "no_record", "query_failed"]
    assert out.ot_disease_score.iloc[0] == 0
    assert out.ot_disease_score.iloc[1:].isna().all()
    with pytest.raises(ValueError):
        annotate_snapshot(frame, snapshot, "MONDO_DIFFERENT", [], "26.09")
    with pytest.raises(ValueError):
        annotate_snapshot(frame, snapshot, "MONDO_0008684", [], "25.12")


def test_ot_fetch_filters_exact_disease_and_checks_release():
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"data": {"meta": {"dataVersion": {"year": "26", "month": "09", "iteration": None}},
                    "disease": {"id": "MONDO_0008684", "name": "WHS"},
                    "target": {"id": "G1", "associations": {"count": 0, "rows": []},
                               "background": {"rows": []}}}}
    calls = []
    def post(url, **kwargs):
        calls.append(kwargs["json"])
        return Response()
    snap = fetch_snapshot(["G1"], "MONDO_0008684", [], "26.09", post=post)
    assert snap["records"]["G1"]["status"] == "ok"
    assert calls[0]["variables"]["diseases"] == ["MONDO_0008684"]
    assert "enableIndirect: false" in calls[0]["query"]
    assert fetch_snapshot(["G1"], "MONDO_0008684", [], "25.12", post=post)["records"]["G1"]["status"] == "query_failed"


def test_cli_two_inputs_replay_and_reference_tampering(inputs, tmp_path, monkeypatch):
    import gzip
    import hashlib
    import subprocess
    import sys
    import zipfile
    from screening.src.screening import load_config

    root = Path(__file__).resolve().parents[2]
    refs = tmp_path / "references"
    refs.mkdir()
    (refs / "annotation.gtf.gz").write_bytes(gzip.compress(inputs[0].read_bytes()))
    (refs / "hgnc.tsv").write_bytes(inputs[1].read_bytes())
    (refs / "clingen.tsv").write_text("#Gene Symbol\tHaploinsufficiency Score\tTriplosensitivity Score\nA\t3\t0\nB\t30\t0\n")
    gnomad = ("gene_id\ttranscript\tmane_select\tcanonical\tlof.pLI\tlof.oe_ci.upper\tconstraint_flags\tgene_flags\n"
              "ENSG00000000001\tT1\ttrue\ttrue\t0.9\t0.2\t[]\t[]\n"
              "ENSG00000000002\tT2\ttrue\ttrue\t0.8\t0.3\t[]\t[]\n")
    (refs / "gnomad.tsv.gz").write_bytes(gzip.compress(gnomad.encode()))
    with zipfile.ZipFile(refs / "hpa.zip", "w") as archive:
        archive.writestr("rna_tissue_consensus.tsv", "Gene\tTissue\tnTPM\nENSG00000000001\tbrain\t5.0\n")
    lock = {"status": "complete", "files": {name: {**source, "sha256": hashlib.sha256((refs / name).read_bytes()).hexdigest()}
                                            for name, source in load_config()["references"].items()}}
    (refs / "references.json").write_text(json.dumps(lock))
    snap = tmp_path / "ot.json"
    snap.write_text(json.dumps({"release": "26.09", "disease_id": "MONDO_0008684", "phenotype_ids": [],
                               "records": {gid: {"status": "ok", "associations": []}
                                           for gid in ["ENSG00000000001", "ENSG00000000002"]}}))
    common = [sys.executable, str(root / "screening/run.py"), "run", "--references", str(refs), "--ot-snapshot", str(snap)]
    prepared_inputs = []
    for name, kind, file_format, text, options in [
        ("prepared_genes", "genes", "tsv", "Gene stable ID\nOLD_A\nHGNC:2\n", ["--gene-column", "Gene stable ID"]),
        ("prepared_interval", "interval", "bed", "chr4\t99\t699\n", ["--genome-build", "GRCh38"]),
    ]:
        raw, prepared = tmp_path / (name + ".txt"), tmp_path / (name + ".json")
        raw.write_text(text)
        converted = subprocess.run([sys.executable, str(root / "screening/prepare_input.py"), kind,
                                    "--input", str(raw), "--format", file_format, "--source", "synthetic download",
                                    "--output", str(prepared), *options], capture_output=True, text=True)
        assert converted.returncode == 0, converted.stderr
        prepared_inputs.append((name, ["--input-file", str(prepared)]))
    gene_list = tmp_path / "genes.txt"
    gene_list.write_text("OLD_A\nHGNC:2\n")
    outputs = []
    for name, cli in [("interval", ["--interval", "chr4:100-699"]), ("list", ["--genes", "OLD_A", "HGNC:2"]),
                      ("list_file", ["--gene-list", str(gene_list)]), *prepared_inputs]:
        output = tmp_path / name
        input_type = "interval" if "interval" in name else "gene_list"
        result = subprocess.run([*common, *cli, "--output", str(output)], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "[OK] Screening complete" in result.stdout
        import re
        assert not re.search(r"[\u4e00-\u9fff]", result.stdout)
        for generated in output.iterdir():
            assert not re.search(r"[\u4e00-\u9fff]", generated.read_text(encoding="utf-8-sig")), generated.name
        assert str(output / f"{input_type}_report.html") in result.stdout
        outputs.append(pd.read_csv(output / f"{input_type}_ranked.csv"))
        summary = pd.read_csv(output / f"{input_type}_summary.csv")
        assert len(summary.columns) == 16
        pd.testing.assert_frame_equal(summary, outputs[-1][summary.columns])
        manifest = json.loads((output / f"{input_type}_manifest.json").read_text())
        assert manifest["status"] == "complete"
        assert manifest["input"]["input_type"] == input_type
        assert f" ({input_type})" in result.stdout
        assert set(p.name for p in output.iterdir()) == {
            f"{input_type}_{name}" for name in (
                "report.html", "summary.csv", "ranked.csv", "candidates.csv",
                "manifest.json", "config.yaml", "ot_snapshot.json",
            )
        }
        from html.parser import HTMLParser

        class Links(HTMLParser):
            def __init__(self):
                super().__init__()
                self.local = []

            def handle_starttag(self, tag, attrs):
                href = dict(attrs).get("href", "")
                if tag == "a" and href and ":" not in href:
                    self.local.append(href)

        links = Links()
        links.feed((output / f"{input_type}_report.html").read_text())
        assert links.local
        assert all((output / name).is_file() for name in links.local)
        assert all(name.startswith(input_type + "_") for name in links.local)
        assert all((output / name).is_file() for name in manifest["outputs"].values())
        assert all((output / f"{input_type}_{name}.csv").read_bytes().startswith(b"\xef\xbb\xbf")
                   for name in ("summary", "ranked", "candidates"))
        assert manifest["code"]["file_sha256"]
        assert manifest["output_sha256"][f"{input_type}_summary.csv"] == hashlib.sha256((output / f"{input_type}_summary.csv").read_bytes()).hexdigest()
        if name.startswith("prepared_"):
            assert manifest["input"]["source"] == "synthetic download"
            assert manifest["input"]["preparation"]["original_file"]["sha256"]
            assert manifest["optional_inputs"]["input_file"]["sha256"]
        assert (output / f"{input_type}_report.html").is_file()
    for actual in outputs[1:]:
        assert outputs[0].gene_id.tolist() == actual.gene_id.tolist()
        assert outputs[0].consensus_score.tolist() == actual.consensus_score.tolist()
    assert outputs[0].selected.tolist() == [True, False]
    # Default directories identify the input type, including prepared JSON inputs.
    import importlib.util
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("screening_cli_output_names", root / "screening/run.py")
    cli_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli_module)
    monkeypatch.setattr(cli_module, "ROOT", tmp_path)
    monkeypatch.setattr(cli_module, "code_provenance", lambda: {})
    for input_type, arguments in [("interval", ["--input-file", str(tmp_path / "prepared_interval.json")]),
                                  ("gene_list", ["--gene-list", str(gene_list)])]:
        monkeypatch.setattr(sys, "argv", [str(root / "screening/run.py"), *common[2:], *arguments])
        cli_module.main()
        directories = list((tmp_path / "workspace/screening/results").glob(f"{input_type}_*"))
        assert len(directories) == 1
        assert (directories[0] / f"{input_type}_manifest.json").is_file()
    # Incomplete optional annotation must not be reported as complete success.
    snapshot = json.loads(snap.read_text())
    snapshot["records"]["ENSG00000000001"] = {"status": "query_failed", "error": "synthetic outage"}
    snap.write_text(json.dumps(snapshot))
    gaps = subprocess.run([*common, "--interval", "chr4:100-699", "--output", str(tmp_path / "gaps")],
                          capture_output=True, text=True)
    assert gaps.returncode == 0, gaps.stderr
    assert "[WARN] Ranking generated" in gaps.stdout and "[OK] Screening complete" not in gaps.stdout
    assert "completed_with_annotation_gaps" in gaps.stdout
    assert pd.read_csv(tmp_path / "gaps/interval_ranked.csv").consensus_score.tolist() == outputs[0].consensus_score.tolist()
    before = (tmp_path / "list/gene_list_ranked.csv").read_bytes()
    repeated = subprocess.run([*common, "--genes", "A", "--output", str(tmp_path / "list")], capture_output=True)
    assert repeated.returncode != 0 and (tmp_path / "list/gene_list_ranked.csv").read_bytes() == before
    failed = subprocess.run([*common, "--genes", "UNKNOWN_GENE", "--output", str(tmp_path / "failed")],
                            capture_output=True, text=True)
    assert failed.returncode != 0
    failed_manifest = json.loads((tmp_path / "failed/gene_list_manifest.json").read_text())
    assert failed_manifest["status"] == "failed"
    assert failed_manifest["input"]["input_type"] == "gene_list"
    assert not (tmp_path / "failed/gene_list_report.html").exists()
    (refs / "hgnc.tsv").write_text("tampered")
    bad = subprocess.run([*common, "--genes", "A", "--output", str(tmp_path / "bad")], capture_output=True, text=True)
    assert bad.returncode != 0 and "checksum mismatch" in bad.stderr
    assert not (tmp_path / "bad").exists()


def test_missing_stop_codon_keeps_rankable_gene_but_requires_design_review(inputs):
    gtf, hgnc = inputs
    gtf.write_text("".join(line for line in gtf.read_text().splitlines(True)
                           if '\tstop_codon\t' not in line))
    frame = transcript_design(gtf, build_candidates(gtf, hgnc, genes=["A"]))
    assert len(frame) == 1 and frame.utr3_status.item() == "incomplete_coding_annotation"
    assert not select_modules(frame).selected.item()
    assert select_modules(frame).selection_reason.item() == "unresolved_transcript_or_utr"


def test_impc_requires_provenance_and_never_removes_candidates(inputs, tmp_path):
    from screening.src.screening import add_impc
    frame = evidence(inputs)
    frame["impc_viability"] = "not_supplied"
    path = tmp_path / "impc.tsv"
    path.write_text("gene_id\tmouse_gene_id\timpc_viability\tsource\trelease\tmapping_source\n"
                    "ENSG00000000001\tMGI:1\tviable\tcurated_source\tv1\torthology_source\n")
    out = add_impc(frame, path)
    assert out.gene_id.tolist() == frame.gene_id.tolist()
    assert out.impc_viability.tolist() == ["viable", "not_supplied"]
    path.write_text(path.read_text().replace("orthology_source", ""))
    with pytest.raises(ValueError, match="nonempty"):
        add_impc(frame, path)


def test_reference_default_is_same_for_prepare_and_run(monkeypatch, tmp_path):
    import importlib.util
    import sys

    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("screening_cli_defaults", root / "screening/run.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    observed = []
    monkeypatch.setattr(cli, "prepare", lambda directory: observed.append(directory))
    monkeypatch.setattr(cli, "run", lambda args: observed.append(args.references.resolve()))
    monkeypatch.chdir(tmp_path)
    for arguments in (["prepare"], ["run", "--genes", "NSD2"]):
        monkeypatch.setattr(sys, "argv", ["run_screening.py", *arguments])
        cli.main()
    assert observed == [root / "workspace/screening/references"] * 2
    for arguments in (["prepare"], ["run", "--genes", "NSD2"]):
        monkeypatch.setattr(sys, "argv", ["run_screening.py", *arguments, "--references", "custom"])
        cli.main()
    assert observed[2:] == [tmp_path / "custom"] * 2
