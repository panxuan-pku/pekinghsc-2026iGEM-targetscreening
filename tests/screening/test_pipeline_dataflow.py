"""Portable CLI acceptance: synthetic intervals -> normalize -> merge -> rank.

No repository data, historical output, network or clinical assertions are used.
The public test runner retains tmp_path inputs, outputs and per-command logs.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import yaml

PIPELINE = Path(__file__).resolve().parents[2]
CONTEXT = ["source", "deletion_id", "chrom", "start", "end"]


@pytest.fixture
def study(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "hgnc_aliases.tsv").write_text(
        "hgnc_id\tsymbol\talias_symbol\nHGNC:1\tTEST_A\tOLD_A\nHGNC:2\tTEST_B\tOLD_B\n")
    (sources / "clinGen_gene_curation_list_GRCh38.tsv").write_text(
        "#synthetic test only\n#Gene Symbol\tHaploinsufficiency Score\tTriplosensitivity Score\n"
        "TEST_A\t3\t0\nTEST_B\t2\t0\n")
    (sources / "gnomad_constraint.tsv").write_text(
        "gene\tpli\toe_lof_upper\nTEST_A\t0.9\t0.1\nTEST_B\t0.8\t0.2\n")
    (sources / "rna_tissue_consensus.tsv").write_text(
        "Gene name\tTissue\tnTPM\nTEST_A\tliver\t1\nTEST_B\tliver\t2\n")
    order = tmp_path / "gene_order.tsv"
    order.write_text("gene_name\tgene_id\tchrom\tstart\tend\tstrand\n"
                     "TEST_A\tID1\tchr1\t100\t200\t+\nTEST_B\tID2\tchr1\t300\t400\t+\n")
    # Reuse the real public config, overriding only test-specific controls/context.
    config = yaml.safe_load((PIPELINE / "screening/config/pipeline.yaml").read_text())
    config["consensus_v2"]["tuning"].update(enabled=False, positive_controls=[])
    config["experiments"] = {"enabled": False}
    config["phenotype_panels"] = []
    config["hpa_filter"] = {"organs": ["liver"], "organ_tissues": {"liver": ["liver"]}}
    config.update(project="SYNTHETIC_A", samples=[],
                  annotation={"gene_order_tsv": str(order), "genome_build": "GRCh38"},
                  known_intervals={"synthetic_a": {"chr": "chr1", "start": 100, "end": 400,
                                                   "genome_build": "GRCh38"}},
                  extract={"mode": "intervals-only"}, output={"dir": str(tmp_path / "cnv")})
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return tmp_path, sources, path


def cli(study, module, args, label, expected=0):
    root, _, _ = study
    command = [sys.executable, "-m", module, *map(str, args)]
    result = subprocess.run(command, cwd=PIPELINE, text=True, capture_output=True, timeout=120)
    (root / f"{label}.log").write_text(
        f"command: {command!r}\nexit: {result.returncode}\n{result.stdout}\n{result.stderr}")
    assert result.returncode == expected, f"{label}: {result.stdout}\n{result.stderr}"
    return result


def extract(study, label="extract"):
    cli(study, "screening.src.cnv.workflow", ["extract-genes", "--config", study[2]], label)


def normalize(study, expected=0):
    root, sources, _ = study
    return cli(study, "screening.src.normalize", ["--input", root / "cnv/candidates.csv", "--out",
               root / "normalized.csv", "--hgnc-alias", sources / "hgnc_aliases.tsv"], "normalize", expected)


def merge(study, expected=0):
    root, sources, config = study
    return cli(study, "screening.src.merge_evidence", ["--normalized", root / "normalized.csv", "--config", config,
               "--data-dir", sources, "--out", root / "evidence.parquet", "--audit", root / "audit"], "merge", expected)


def rank(study):
    root, _, config = study
    cli(study, "screening.src.consensus_v2", ["--evidence", root / "evidence.parquet", "--config", config,
        "--mode", "rank", "--sensitivity", "--out", root / "ranked.csv", "--report", root / "report.md"], "rank")


def test_complete_cli_dataflow_and_study_isolation(study):
    root, sources, config_path = study
    snapshots = {}
    for suffix, start, expected in [("a", 100, ["TEST_A", "TEST_B"]), ("b", 300, ["TEST_B"])]:
        current = root / f"study_{suffix}"
        current.mkdir()
        config = yaml.safe_load(config_path.read_text())
        config["project"] = f"SYNTHETIC_{suffix.upper()}"
        config["known_intervals"] = {f"synthetic_{suffix}": {
            "chr": "chr1", "start": start, "end": 400, "genome_build": "GRCh38"}}
        config["output"]["dir"] = str(current / "cnv")
        cfg = current / "config.yaml"
        cfg.write_text(yaml.safe_dump(config))
        case = current, sources, cfg
        extract(case)
        normalize(case)
        merge(case)
        rank(case)
        candidate = pd.read_csv(current / "cnv/candidates.csv")
        assert candidate.gene_symbol.tolist() == expected
        baseline = candidate.set_index("gene_symbol")[CONTEXT]
        for frame in [pd.read_csv(current / "normalized.csv"), pd.read_parquet(current / "evidence.parquet"),
                      pd.read_csv(current / "ranked.csv")]:
            pd.testing.assert_frame_equal(frame.set_index("input_symbol").loc[expected, CONTEXT],
                                          baseline, check_names=False)
        audit = json.loads((current / "cnv/audit/run_extract_genes.json").read_text())
        assert audit["project"] == config["project"] and audit["status"] == "success"
        assert audit["outputs"]["candidates.csv"]["sha256"] == hashlib.sha256(
            (current / "cnv/candidates.csv").read_bytes()).hexdigest()
        assert set(audit["inputs"]) == {"config_file", "gene_order"}
        merge_audit = json.loads((current / "audit/run.json").read_text())
        for name, digest in merge_audit["data_checksums"].items():
            assert digest == hashlib.sha256((sources / name).read_bytes()).hexdigest()
        rank_audit = json.loads((current / "ranked.csv.audit.json").read_text())
        assert rank_audit["results"]["mode"] == "rank"
        assert rank_audit["results"]["controls_total"] == 0
        for path in (current / "evidence.parquet", cfg):
            assert rank_audit["input_checksums"][str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert rank_audit["output_sha256"] == hashlib.sha256((current / "ranked.csv").read_bytes()).hexdigest()
        assert (current / "report.md").stat().st_size > 0
        repeated = {current / name: (current / name).read_bytes()
                    for name in ("normalized.csv", "evidence.parquet", "ranked.csv", "report.md")}
        extract(case)
        normalize(case)
        merge(case)
        rank(case)
        assert all(p.read_bytes() == content for p, content in repeated.items())
        snapshots[current / "ranked.csv"] = (current / "ranked.csv").read_bytes()
    assert all(p.read_bytes() == content for p, content in snapshots.items())
    ids = [json.loads(p.read_text())["execution_id"] for p in root.glob("study_*/cnv/audit/run_extract_genes.json")]
    assert len(set(ids)) == 2


def test_empty_interval_output_stops_at_normalize(study):
    root, _, path = study
    config = yaml.safe_load(path.read_text())
    config["known_intervals"] = {}
    path.write_text(yaml.safe_dump(config))
    extract(study)
    assert pd.read_csv(root / "cnv/candidates.csv").empty
    assert "candidate input is empty" in normalize(study, expected=2).stderr
    assert not (root / "normalized.csv").exists()
    assert not (root / "evidence.parquet").exists()


def test_overlapping_intervals_rank_once_with_complete_provenance(study):
    root, _, path = study
    extract(study)
    normalize(study)
    merge(study)
    rank(study)
    baseline = pd.read_csv(root / "ranked.csv")
    config = yaml.safe_load(path.read_text())
    config["known_intervals"]['second,"interval'] = {
        "chr": "chr1", "start": 150, "end": 250, "genome_build": "GRCh38"}
    path.write_text(yaml.safe_dump(config))
    extract(study)
    candidates = pd.read_csv(root / "cnv/candidates.csv")
    assert candidates.gene_symbol.tolist() == ["TEST_A", "TEST_B"]
    assert pd.isna(candidates.loc[0, "deletion_id"])
    origins = json.loads(candidates.loc[0, "candidate_provenance"])
    assert {r["deletion_id"] for r in origins} == {"synthetic_a", 'second,"interval'}
    assert {tuple(r[k] for k in ("interval_chrom", "interval_start", "interval_end"))
            for r in origins} == {("chr1", 100, 400), ("chr1", 150, 250)}
    assert all(r["gene_id"] == "ID1" and r["genome_build"] == "GRCh38" for r in origins)
    normalize(study)
    merge(study)
    rank(study)
    ranked = pd.read_csv(root / "ranked.csv")
    pd.testing.assert_frame_equal(ranked.drop(columns=["deletion_id", "candidate_provenance"]),
                                  baseline.drop(columns="deletion_id"), check_exact=True)
    for frame in (pd.read_csv(root / "normalized.csv"), pd.read_parquet(root / "evidence.parquet"), ranked):
        row = frame.set_index("input_symbol").loc["TEST_A"]
        assert json.loads(row.candidate_provenance) == origins
    detail = pd.read_csv(root / "cnv/candidate_provenance.csv")
    assert len(detail) == 3 and detail.in_candidates.all()
    audit = json.loads((root / "cnv/audit/run_extract_genes.json").read_text())
    assert audit["outputs"]["candidate_provenance.csv"]["sha256"] == hashlib.sha256(
        (root / "cnv/candidate_provenance.csv").read_bytes()).hexdigest()


@pytest.mark.parametrize("problem", ["missing", "malformed"])
def test_invalid_evidence_stops_before_merge_output(study, problem):
    root, sources, _ = study
    extract(study)
    normalize(study)
    path = sources / "gnomad_constraint.tsv"
    if problem == "missing":
        path.rename(sources / "gnomad_constraint.saved.tsv")
    else:
        path.write_text("gene\tunrelated\nTEST_A\t1\n")
    assert "gnomad_constraint.tsv" in merge(study, expected=2).stderr
    assert not (root / "evidence.parquet").exists()
    assert not (root / "ranked.csv").exists()


@pytest.mark.parametrize("case", [
    "normalize_input", "normalize_aliases", "normalize_symlink", "normalize_hardlink",
    "merge_input", "merge_source", "merge_audit", "merge_audit_parent",
    "rank_input", "rank_report", "rank_audit", "rank_report_directory", "rank_report_parent",
    "rank_report_symlink", "rank_report_hardlink", "rank_output_parent", "rank_controls",
])
def test_output_conflicts_fail_before_any_write(study, case):
    root, sources, config = study
    candidate = root / "candidates.csv"
    candidate.write_text("gene_symbol\nTEST_A\nTEST_B\n")
    normalized = root / "normalized.csv"
    frame = pd.DataFrame({"hgnc_id": ["HGNC:1", "HGNC:2"],
                          "input_symbol": ["TEST_A", "TEST_B"], "status": ["ok", "ok"],
                          "clinGen_hi_score": [3, 2], "gnomad_pLI": [0.9, 0.8],
                          "gnomad_LOEUF": [0.1, 0.2]})
    frame[["hgnc_id", "input_symbol", "status"]].to_csv(normalized, index=False)
    evidence = root / "evidence.parquet"
    frame.to_parquet(evidence, index=False)
    output = root / "result.csv"
    output.write_text("previous result\n")
    audit = root / "audit"
    audit.mkdir()
    (audit / "run.json").write_text('{"previous": true}\n')
    report = root / "report.md"
    report.write_text("previous report\n")
    aliases = sources / "hgnc_aliases.tsv"
    rank_audit = root / "rank.audit.json"
    rank_audit.write_text('{"previous": true}\n')
    if case.startswith("normalize"):
        module = "screening.src.normalize"
        output = aliases if case == "normalize_aliases" else candidate
        if case in ("normalize_symlink", "normalize_hardlink"):
            output = root / "linked.csv"
            if case == "normalize_symlink":
                output.symlink_to(candidate)
            else:
                output.hardlink_to(candidate)
        args = ["--input", candidate, "--hgnc-alias", aliases, "--out", output]
    elif case.startswith("merge"):
        module = "screening.src.merge_evidence"
        output = {"merge_input": normalized, "merge_source": aliases,
                  "merge_audit": audit / "run.json", "merge_audit_parent": audit}[case]
        args = ["--normalized", normalized, "--config", config, "--data-dir", sources,
                "--out", output, "--audit", audit]
    else:
        module = "screening.src.consensus_v2"
        if case == "rank_input":
            report = config
        elif case == "rank_report":
            report = output
        elif case == "rank_audit":
            rank_audit = report
        elif case == "rank_report_directory":
            report = audit
        elif case == "rank_report_parent":
            report = normalized / "report.md"
        elif case in ("rank_report_symlink", "rank_report_hardlink"):
            report.unlink()
            if case == "rank_report_symlink":
                report.symlink_to(output)
            else:
                report.hardlink_to(output)
        elif case == "rank_output_parent":
            output = root / "new-result"
            report = output / "report.md"
        args = ["--evidence", evidence, "--config", config, "--out", output,
                "--report", report, "--audit", rank_audit, "--mode", "rank"]
        if case == "rank_controls":
            report.write_text("TEST_A\n")
            args.extend(["--controls", report])
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = cli(study, module, args, "conflict", expected=2)
    assert "output path conflict" in result.stderr
    assert all(p.read_bytes() == content for p, content in before.items())
    assert {p for p in root.rglob("*") if p.is_file()} == set(before) | {root / "conflict.log"}
