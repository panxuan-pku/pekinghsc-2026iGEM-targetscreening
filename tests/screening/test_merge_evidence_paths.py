"""Regression tests for data-source resolution and missing-source failures."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from screening.src.consensus_v2 import load_v2_config, run_consensus
from screening.src.merge_evidence import join_on_hgnc
from screening.src.normalize import load_alias_map, resolve


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def data_dir(tmp_path):
    """Small portable sources: the worktree may contain HGNC LFS pointers."""
    path = tmp_path / "sources"
    path.mkdir()
    (path / "hgnc_aliases.tsv").write_text(" hgnc_id \t symbol \t prev_symbol \n HGNC:12766 \t NSD2 \t WHSC1 \n")
    (path / "clinGen_gene_curation_list_GRCh38.tsv").write_text(
        "#test snapshot\n#Gene Symbol\tHaploinsufficiency Score\tTriplosensitivity Score\tDate Last Evaluated\n"
        "WHSC1\t3\t0\t2026-09-19\n")
    (path / "gnomad_constraint.tsv").write_text("gene\tgene_id\tpli\toe_lof_upper\nWHSC1\tENSG1\t0.99\t0.1\n")
    (path / "rna_tissue_consensus.tsv").write_text(
        "Gene name\tTissue\tnTPM\nNSD2\tcerebral cortex\t20\nNSD2\tcerebellum\t3\nNSD2\tliver\t2\n")
    return path


@pytest.mark.parametrize("case", ["partial", "zero", "empty", "unmapped"])
def test_phenotype_merge_preserves_ranking_and_records_coverage(tmp_path, data_dir, case):
    source = data_dir / "rna_tissue_consensus.tsv"
    if case == "zero":
        source.write_text(source.read_text().replace("\t20\n", "\t0\n"))
    elif case == "empty":
        source.write_text("Gene name\tTissue\tnTPM\n")
    elif case == "unmapped":
        source.write_text(source.read_text().replace("NSD2", "UNMAPPED"))
    normalized = tmp_path / "normalized.csv"
    pd.DataFrame({"hgnc_id": ["HGNC:12766", None], "input_symbol": ["NSD2", "UNKNOWN"]}).to_csv(normalized, index=False)
    config = yaml.safe_load((ROOT / "screening/config/pipeline.yaml").read_text())
    frames = []
    for enabled in (False, True):
        config["phenotype_panels"] = ([
            {"id": "a", "disease": "Synthetic", "phenotype": "A", "tissues": ["cerebral cortex", "absent"]},
            {"id": "b", "disease": "Synthetic", "phenotype": "B", "tissues": ["kidney"]},
        ] if enabled else [])
        cfg = tmp_path / f"config-{enabled}.yaml"
        cfg.write_text(yaml.safe_dump(config))
        output = tmp_path / f"evidence-{enabled}.parquet"
        audit = tmp_path / f"audit-{enabled}"
        command = _command(normalized, output, audit, data_dir)
        command[command.index("--config") + 1] = str(cfg)
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        frames.append(pd.read_parquet(output))
        if enabled:
            metadata = json.loads((audit / "run.json").read_text())["hpa_coverage"]
            a, b = metadata["phenotype_panels"]
            assert a["coverage_status"] == ("partial" if case in ("partial", "zero") else "unavailable")
            assert b["coverage_status"] == "unavailable"
            assert "kidney" in metadata["organs"]
            assert "WARNING hpa panel a" in result.stdout
            if case in ("partial", "zero"):
                col = "hpa_tissue::cerebral cortex::ntpm"
                assert frames[-1].loc[0, col] == (0 if case == "zero" else 20)
                assert pd.isna(frames[-1].loc[1, col])
    pd.testing.assert_frame_equal(frames[0], frames[1][frames[0].columns], check_exact=True)
    baseline, _ = run_consensus(frames[0], load_v2_config(config))
    ranked, _ = run_consensus(frames[1], load_v2_config(config))
    pd.testing.assert_frame_equal(baseline, ranked[baseline.columns], check_exact=True)


@pytest.mark.parametrize("source,columns,missing", [
    ("clinGen", ["Gene Symbol", "Haploinsufficiency Score", "Triplosensitivity Score"], [column])
    for column in ["Gene Symbol", "Haploinsufficiency Score", "Triplosensitivity Score"]
] + [
    ("gnomad_constraint", ["gene", "pli", "oe_lof_upper"], [column])
    for column in ["gene", "pli", "oe_lof_upper"]
] + [
    ("clinGen", ["Gene Symbol", "Haploinsufficiency Score", "Triplosensitivity Score"],
     ["Haploinsufficiency Score", "Triplosensitivity Score"]),
    ("gnomad_constraint", ["gene", "pli", "oe_lof_upper"], ["pli", "oe_lof_upper"]),
])
def test_source_schema_reports_file_and_missing_columns(tmp_path, source, columns, missing):
    path = tmp_path / "broken.tsv"
    remaining = [column for column in columns if column not in missing]
    prefix = "#snapshot\n#" if source == "clinGen" else ""
    path.write_text(prefix + "\t".join(remaining) + "\n" + "\t".join(["G"] * len(remaining)) + "\n")
    base = pd.DataFrame({"hgnc_id": ["HGNC:1"], "input_symbol": ["G"]})
    with pytest.raises(ValueError) as exc:
        join_on_hgnc(base, path, source, {"G": "HGNC:1"})
    message = str(exc.value).lower()
    assert source.lower() in message
    assert str(path).lower() in message
    assert "missing required column" in message
    for column in missing:
        assert column.lower() in message


def test_join_rejects_identifier_only_evidence(tmp_path):
    path = tmp_path / "identifier_only.tsv"
    path.write_text("hgnc_id\tunexpected\nHGNC:1\tvalue\n")
    with pytest.raises(ValueError, match="evidence column") as exc:
        join_on_hgnc(pd.DataFrame({"hgnc_id": ["HGNC:1"]}), path, "probe")
    assert str(path) in str(exc.value)


@pytest.mark.parametrize("source,header,values,output_columns", [
    ("clinGen", "#Gene Symbol\tHaploinsufficiency Score\tTriplosensitivity Score",
     "3\t0", ["clinGen_hi_score", "clinGen_triplo_score"]),
    ("gnomad_constraint", "gene\tpli\toe_lof_upper", "0.99\t0.1", ["gnomad_pLI", "gnomad_LOEUF"]),
    ("gnomad_constraint", "gene_symbol\tpli\toe_lof_upper", "0.99\t0.1", ["gnomad_pLI", "gnomad_LOEUF"]),
])
@pytest.mark.parametrize("case", ["normal", "missing_values", "unmapped"])
def test_schema_allows_optional_metadata_and_missing_gene_evidence(tmp_path, source, header, values, output_columns, case):
    path = tmp_path / "minimal.tsv"
    symbol = "UNKNOWN" if case == "unmapped" else "G"
    values = "\t" if case == "missing_values" else values
    path.write_text(header + "\n" + symbol + "\t" + values + "\n")
    base = pd.DataFrame({"hgnc_id": ["HGNC:1", None], "input_symbol": ["G", "UNRESOLVED"]})
    result = join_on_hgnc(base, path, source, {"G": "HGNC:1"})
    assert len(result) == 2
    assert result["input_symbol"].tolist() == ["G", "UNRESOLVED"]
    assert result.loc[1, output_columns].isna().all()
    if case == "normal":
        assert result.loc[0, output_columns].notna().all()
    else:
        assert result[output_columns].isna().all().all()
    if source == "clinGen":
        assert result.loc[1, "clinGen_haploinsufficiency_status"] == "not_curated"
        if case == "missing_values":
            assert result.loc[0, "clinGen_haploinsufficiency_status"] == "not_evaluated"


@pytest.mark.parametrize("filename,old,new,missing", [
    ("clinGen_gene_curation_list_GRCh38.tsv", "Haploinsufficiency Score", "unexpected", "haploinsufficiency score"),
    ("gnomad_constraint.tsv", "oe_lof_upper", "unexpected", "oe_lof_upper"),
    ("rna_tissue_consensus.tsv", "Gene name", "unexpected", "gene name"),
    ("rna_tissue_consensus.tsv", "Tissue", "unexpected", "tissue"),
    ("rna_tissue_consensus.tsv", "nTPM", "unexpected", "ntpm"),
])
@pytest.mark.parametrize("existing_output", [False, True])
def test_bad_schema_cli_preserves_outputs(tmp_path, data_dir, filename, old, new, missing, existing_output):
    source = data_dir / filename
    source.write_text(source.read_text().replace(old, new))
    normalized = tmp_path / "normalized.csv"
    pd.DataFrame({"hgnc_id": ["HGNC:12766"], "input_symbol": ["NSD2"]}).to_csv(normalized, index=False)
    output = tmp_path / "output" / "evidence.parquet"
    audit = tmp_path / "audit"
    paths = [output, audit / "run.json", audit / "data_checksums.txt"]
    if existing_output:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"previous successful output")
    result = subprocess.run(_command(normalized, output, audit, data_dir), cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert str(source) in result.stderr
    assert "missing required column" in result.stderr
    assert missing in result.stderr
    for path in paths:
        if existing_output:
            assert path.read_bytes() == b"previous successful output"
        else:
            assert not path.exists()


def _command(normalized, output, audit, data_dir):
    return [
        sys.executable, "-m", "screening.src.merge_evidence",
        "--normalized", str(normalized),
        "--config", str(ROOT / "screening" / "config" / "pipeline.yaml"),
        "--out", str(output),
        "--audit", str(audit),
        "--data-dir", str(data_dir),
    ]


def test_nested_audit_does_not_change_data_source(tmp_path, data_dir):
    normalized = tmp_path / "normalized.csv"
    pd.DataFrame([{"hgnc_id": "HGNC:12766", "input_symbol": "NSD2"}]).to_csv(normalized, index=False)
    output = tmp_path / "nested" / "evidence.parquet"
    audit = tmp_path / "nested" / "deep" / "audit"
    output.parent.mkdir()
    result = subprocess.run(
        _command(normalized, output, audit, data_dir),
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    frame = pd.read_parquet(output)
    assert len(frame) == 1
    assert frame.loc[0, "input_symbol"] == "NSD2"
    assert pd.notna(frame.loc[0, "gnomad_pLI"])
    assert "clinGen_hi_score" in frame.columns
    assert (audit / "data_checksums.txt").exists()
    assert (audit / "run.json").exists()
    assert frame.loc[0, "hpa_brain_expr"] == 20  # numeric maximum, not lexical
    assert frame.loc[0, "clinGen_haploinsufficiency_raw"] == 3


def test_missing_sources_fail_before_output(tmp_path):
    normalized = tmp_path / "normalized.csv"
    pd.DataFrame([{"hgnc_id": "HGNC:12766", "input_symbol": "NSD2"}]).to_csv(normalized, index=False)
    output = tmp_path / "evidence.parquet"
    audit = tmp_path / "audit"
    result = subprocess.run(
        _command(normalized, output, audit, tmp_path / "empty"),
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "missing required evidence files" in result.stderr
    assert not output.exists()
    assert not audit.exists()


def test_audit_checksums_use_filenames_across_data_directories(tmp_path, data_dir):
    normalized = tmp_path / "normalized.csv"
    pd.DataFrame({"hgnc_id": ["HGNC:12766"], "input_symbol": ["NSD2"]}).to_csv(normalized, index=False)
    relocated = tmp_path / "relocated sources"
    shutil.copytree(data_dir, relocated)
    filenames = ["hgnc_aliases.tsv", "clinGen_gene_curation_list_GRCh38.tsv",
                 "gnomad_constraint.tsv", "rna_tissue_consensus.tsv"]
    expected = {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest() for name in filenames}
    frames = []
    for index, sources in enumerate([data_dir, relocated]):
        output = tmp_path / f"evidence_{index}.parquet"
        audit = tmp_path / f"audit_{index}"
        result = subprocess.run(_command(normalized, output, audit, sources),
                                cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        metadata = json.loads((audit / "run.json").read_text())
        checksums = dict(line.split("  ", 1) for line in (audit / "data_checksums.txt").read_text().splitlines())
        assert metadata["data_checksums"] == expected
        assert checksums == expected
        assert metadata["data_directory"] == str(sources.resolve())
        assert metadata["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
        frames.append(pd.read_parquet(output))
    pd.testing.assert_frame_equal(frames[0], frames[1], check_exact=True)


def test_candidate_context_reaches_ranking_without_changing_scores(tmp_path, data_dir):
    candidates = pd.DataFrame({
        "gene_symbol": ["ENSG00000289346", "NSD2"],
        "source": ["known_interval", "known_interval"],
        "deletion_id": ["williams_7q11.23", "wolf_hirschhorn_4p16.3"],
        "chrom": ["chr7", "chr4"],
        "start": [74796150, 100],
        "end": [74890585, 200],
    })
    context = ["source", "deletion_id", "chrom", "start", "end"]
    normalized = resolve(candidates, load_alias_map(data_dir / "hgnc_aliases.tsv"))
    cfg = load_v2_config(yaml.safe_load((ROOT / "screening" / "config" / "pipeline.yaml").read_text()))
    results = {}
    for name, frame in (("legacy", normalized[["hgnc_id", "input_symbol", "status"]]),
                        ("context", normalized)):
        norm_path = tmp_path / f"{name}.csv"
        evidence_path = tmp_path / f"{name}.parquet"
        frame.to_csv(norm_path, index=False)
        result = subprocess.run(
            _command(norm_path, evidence_path, tmp_path / name, data_dir),
            cwd=ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        evidence = pd.read_parquet(evidence_path)
        ranked, _ = run_consensus(evidence, cfg, mode="rank")
        results[name] = (evidence, ranked)

    expected = candidates.rename(columns={"gene_symbol": "input_symbol"}).set_index("input_symbol")[context]
    for legacy, current in zip(results["legacy"], results["context"]):
        pd.testing.assert_frame_equal(
            current.set_index("input_symbol").loc[expected.index, context], expected,
        )
        pd.testing.assert_frame_equal(current.drop(columns=context), legacy, check_exact=True)
        unresolved = current.loc[current["input_symbol"] == "ENSG00000289346"].iloc[0]
        assert unresolved["status"] == "unresolved"
        assert pd.isna(unresolved["hgnc_id"])
