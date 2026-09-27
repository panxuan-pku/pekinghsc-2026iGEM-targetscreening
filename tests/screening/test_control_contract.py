"""Explicit control-gene input and CLI output protection."""
from pathlib import Path
import json
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from screening.src.consensus_v2 import load_v2_config, run_consensus


@pytest.fixture
def case():
    config = {"consensus_v2": {
        "evidence": {"gnomad_pLI": {"weight": 2}},
        "hpa_penalty": {"organs": []},
        "tuning": {"enabled": False, "positive_controls": []},
    }}
    evidence = pd.DataFrame({"hgnc_id": ["HGNC:1", "HGNC:2"],
                             "input_symbol": ["TEST_A", "TEST_B"], "gnomad_pLI": [0.9, 0.8]})
    return config, evidence


@pytest.mark.parametrize("controls", ["NSD2", None, 17, {"NSD2": True}, [None], [17], [" "]])
def test_invalid_controls_fail_before_outputs(case, tmp_path, controls):
    config, evidence = case
    config["consensus_v2"]["tuning"]["positive_controls"] = controls
    with pytest.raises(ValueError, match="positive_controls.*list of non-empty strings"):
        run_consensus(evidence, load_v2_config(config))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config))
    ev = tmp_path / "evidence.parquet"
    evidence.to_parquet(ev, index=False)
    output, report, audit = [tmp_path / name for name in ("rank.csv", "report.md", "rank.csv.audit.json")]
    for path in (output, report, audit):
        path.write_text("previous result\n")
    command = [sys.executable, "-m", "screening.src.consensus_v2", "--config", str(cfg),
               "--evidence", str(ev), "--out", str(output), "--report", str(report)]
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    (tmp_path / "cli.log").write_text(f"{command!r}\nexit={result.returncode}\n{result.stdout}\n{result.stderr}")
    assert result.returncode == 2 and "positive_controls" in result.stderr
    assert all(path.read_text() == "previous result\n" for path in (output, report, audit))


def test_control_file_overrides_config_and_controls_do_not_change_untuned_scores(case, tmp_path):
    config, evidence = case
    baseline, empty_meta = run_consensus(evidence, load_v2_config(config))
    assert empty_meta["controls_list"] == []
    config["consensus_v2"]["tuning"]["positive_controls"] = [" test_a ", "TEST_A", "test_b"]
    ranked, meta = run_consensus(evidence, load_v2_config(config))
    assert meta["controls_list"] == ["TEST_A", "TEST_B"]
    pd.testing.assert_frame_equal(ranked, baseline, check_exact=True)
    config["consensus_v2"]["tuning"]["positive_controls"] = "ignored malformed config"
    control_file = tmp_path / "controls.txt"
    control_file.write_text("test_b\n\nTEST_B\n")
    ranked, meta = run_consensus(evidence, load_v2_config(config), controls_path=control_file)
    assert meta["controls_list"] == ["TEST_B"]
    pd.testing.assert_frame_equal(ranked, baseline, check_exact=True)
    control_file.write_text("")
    _, meta = run_consensus(evidence, load_v2_config(config), controls_path=control_file)
    assert meta["controls_list"] == []


@pytest.mark.parametrize("missing", ["tuning", "positive_controls"])
@pytest.mark.parametrize("existing", [False, True])
def test_missing_controls_require_explicit_input_before_outputs(case, tmp_path, missing, existing):
    config, evidence = case
    if missing == "tuning":
        del config["consensus_v2"]["tuning"]
    else:
        del config["consensus_v2"]["tuning"]["positive_controls"]
    with pytest.raises(ValueError, match="positive_controls"):
        run_consensus(evidence, load_v2_config(config))
    cfg, ev = tmp_path / "config.yaml", tmp_path / "evidence.parquet"
    cfg.write_text(yaml.safe_dump(config))
    evidence.to_parquet(ev, index=False)
    output, report, audit = [tmp_path / name for name in ("rank.csv", "report.md", "rank.csv.audit.json")]
    if existing:
        for path in (output, report, audit):
            path.write_text("previous result\n")
    command = [sys.executable, "-m", "screening.src.consensus_v2", "--config", str(cfg),
               "--evidence", str(ev), "--out", str(output), "--report", str(report)]
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    assert result.returncode == 2 and "positive_controls" in result.stderr
    for path in (output, report, audit):
        if existing:
            assert path.read_text() == "previous result\n"
        else:
            assert not path.exists()
    controls = tmp_path / "controls.txt"
    controls.write_text("TEST_B\n")
    ranked, meta = run_consensus(evidence, load_v2_config(config), controls_path=controls)
    assert len(ranked) == 2 and meta["controls_list"] == ["TEST_B"]
    result = subprocess.run(command + ["--controls", str(controls)],
                            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert all(path.is_file() for path in (output, report, audit))


@pytest.mark.parametrize("mode", ["rank", "validate", "full", "exploratory"])
@pytest.mark.parametrize("source", ["config", "file"])
def test_explicit_empty_controls_are_reported_as_not_evaluated(case, tmp_path, mode, source):
    config, evidence = case
    extra = []
    if source == "file":
        del config["consensus_v2"]["tuning"]["positive_controls"]
        controls = tmp_path / "controls.txt"
        controls.write_text("")
        extra = ["--controls", str(controls)]
    cfg, ev = tmp_path / "config.yaml", tmp_path / "evidence.parquet"
    cfg.write_text(yaml.safe_dump(config))
    evidence.to_parquet(ev, index=False)
    output, report = tmp_path / "rank.csv", tmp_path / "report.md"
    result = subprocess.run([sys.executable, "-m", "screening.src.consensus_v2", "--config", str(cfg),
                             "--evidence", str(ev), "--out", str(output), "--report", str(report),
                             "--mode", mode, *extra],
                            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "No positive controls configured; this check was not evaluated" in report.read_text()
    assert "Positive-control top-10 hit rate" not in report.read_text()
    results = json.loads((tmp_path / "rank.csv.audit.json").read_text())["results"]
    assert results["controls_list"] == [] and results["controls_check"] == []
    if mode == "validate":
        assert results["validation_pass"] is None
