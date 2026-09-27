"""CR-043: callers must resolve overlapping evidence sources explicitly."""
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import yaml


@pytest.mark.parametrize("source", ["ai", "compensation", "both"])
def test_conflicting_sources_fail_before_outputs(tmp_path, source):
    frame = pd.DataFrame({"hgnc_id": ["HGNC:1", "HGNC:2"],
                          "input_symbol": ["A", "B"], "gnomad_pLI": [0.9, 0.2]})
    if source == "ai":
        frame["DeepLOF_score"] = [0.1, 0.2]
        extra = pd.DataFrame({"hgnc_id": frame.hgnc_id, "DeepLOF_score": [0.8, 0.9]})
        flag, conflict = "--ai-scores", "DeepLOF_score"
    else:
        extra = pd.DataFrame({"gene": ["A", "B"], "compensation_class": ["none", "full"]})
        flag, conflict = "--compensation", "compensation_class"
        if source == "compensation":
            frame[conflict] = ["old", "old"]
    args, outputs = command(tmp_path, frame)
    if source == "both":
        ai = tmp_path / "ai.csv"
        pd.DataFrame({"hgnc_id": frame.hgnc_id, conflict: ["old", "old"]}).to_csv(ai, index=False)
        args += ["--ai-scores", str(ai)]
    path = tmp_path / "extra.csv"
    extra.to_csv(path, index=False)
    args += [flag, str(path)]
    for existing in (False, True):
        if existing:
            for output in outputs:
                output.write_text("previous\n")
        result = subprocess.run(args, cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
        assert result.returncode == 2
        assert conflict in result.stderr and str(path) in result.stderr
        assert "choose" in result.stderr and "Traceback" not in result.stderr
        for output in outputs:
            assert output.read_text() == "previous\n" if existing else not output.exists()


def command(tmp_path, frame):
    evidence, config = tmp_path / "evidence.parquet", tmp_path / "config.yaml"
    frame.to_parquet(evidence)
    config.write_text(yaml.safe_dump({"consensus_v2": {
        "tuning": {"positive_controls": []},
        "evidence": {"gnomad_pLI": {"weight": 2}}, "hpa_penalty": {"organs": []}}}))
    out, report = tmp_path / "ranked.csv", tmp_path / "report.md"
    return ([sys.executable, "-m", "screening.src.consensus_v2", "--evidence", str(evidence),
             "--config", str(config), "--out", str(out), "--report", str(report)],
            [out, report, tmp_path / "ranked.csv.audit.json"])


def test_nonconflicting_sources_preserve_columns(tmp_path):
    frame = pd.DataFrame({"hgnc_id": ["HGNC:1", "HGNC:2"], "input_symbol": ["A", "B"],
                          "gnomad_pLI": [0.9, 0.2]})
    args, outputs = command(tmp_path, frame)
    ai, comp = tmp_path / "ai.csv", tmp_path / "comp.csv"
    pd.DataFrame({"hgnc_id": frame.hgnc_id, "DeepLOF_score": [0.1, 0.9]}).to_csv(ai, index=False)
    pd.DataFrame({"gene": ["A", "B"], "compensation_class": ["none", "full"]}).to_csv(comp, index=False)
    result = subprocess.run(args + ["--ai-scores", str(ai), "--compensation", str(comp)],
                            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    actual = pd.read_csv(outputs[0]).set_index("hgnc_id")
    assert actual.loc["HGNC:1", "DeepLOF_score"] == 0.1
    assert actual.loc["HGNC:2", "compensation_class"] == "full"
    assert not any(c.endswith(("_x", "_y")) for c in actual.columns)
