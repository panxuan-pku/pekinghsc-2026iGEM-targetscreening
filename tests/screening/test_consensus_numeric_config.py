"""CR-047: invalid numeric configuration must fail before writing results."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from screening.src.consensus_v2 import load_v2_config, run_consensus


def config():
    return {"consensus_v2": {
        "evidence": {"gnomad_pLI": {"weight": 2}},
        "hpa_penalty": {"organs": []},
        "tuning": {"positive_controls": []},
    }}


def evidence():
    return pd.DataFrame({"hgnc_id": ["HGNC:1", "HGNC:2", "HGNC:3"],
                         "input_symbol": ["A", "B", "C"],
                         "gnomad_pLI": [0.1, 0.5, 0.9]})


FIELDS = [
    ("evidence", "gnomad_pLI", "weight"),
    ("evidence", "gnomad_pLI", "ordinal_scale"),
    ("evidence", "gnomad_pLI", "low_score_penalty"),
    ("hpa_penalty", "expr_threshold"),
    ("hpa_penalty", "per_organ_weight"),
    ("hpa_penalty", "cap"),
    ("tuning", "target_top_n"),
    ("recessive", "gnomad_weight_multiplier"),
    ("validation", "max_missing_rate"),
]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("path", FIELDS)
@pytest.mark.parametrize("entry", ["load", "run"])
def test_nonfinite_config_rejected_with_field_path(value, path, entry):
    cfg = load_v2_config(config())
    target = cfg
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = value
    with pytest.raises(ValueError, match="finite") as error:
        if entry == "run":
            run_consensus(evidence(), cfg)
        else:
            load_v2_config({"consensus_v2": cfg, "recessive": cfg["recessive"],
                            "validation": cfg["validation"]})
    assert ".".join(path) in str(error.value)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_grid_rejected_even_when_tuning_disabled(value):
    cfg = config()
    cfg["consensus_v2"]["tuning"] = {"enabled": False, "weight_grid": [0, value]}
    with pytest.raises(ValueError, match=r"tuning.weight_grid\[1\].*finite"):
        load_v2_config(cfg)


def test_inherited_weight_rejected_but_finite_override_accepted():
    cfg = {"consensus": {"base_weights": {"gnomad_pLI": np.nan}}}
    with pytest.raises(ValueError, match="evidence.gnomad_pLI.weight.*finite"):
        load_v2_config(cfg)
    cfg.update(config())
    assert load_v2_config(cfg)["evidence"]["gnomad_pLI"]["weight"] == 2


@pytest.mark.parametrize("borda", [False, True])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_cli_rejects_bad_weight_without_replacing_outputs(tmp_path, borda, value):
    cfg = config()
    cfg["consensus_v2"]["evidence"]["gnomad_pLI"]["weight"] = value
    cfg["consensus_v2"]["borda"] = {"enabled": borda}
    _cli_failure(tmp_path, cfg, evidence(), "evidence.gnomad_pLI.weight")


@pytest.mark.parametrize("kind", ["consensus", "borda"])
def test_cli_rejects_finite_weight_overflow_before_writing(tmp_path, kind):
    cfg = config()
    cfg["consensus_v2"]["evidence"]["gnomad_pLI"]["weight"] = 1e308
    df = evidence()
    if kind == "consensus":
        cfg["consensus_v2"]["evidence"]["other"] = {"weight": 1e308}
        cfg["consensus_v2"]["borda"] = {"enabled": False}
        df["other"] = df["gnomad_pLI"]
    _cli_failure(tmp_path, cfg, df, f"non-finite {kind}")


def _cli_failure(tmp_path, cfg, df, message):
    source, config_path = tmp_path / "evidence.parquet", tmp_path / "config.yaml"
    df.to_parquet(source)
    config_path.write_text(yaml.safe_dump(cfg))
    out, report = tmp_path / "ranked.csv", tmp_path / "report.md"
    audit = tmp_path / "ranked.csv.audit.json"
    command = [sys.executable, "-m", "screening.src.consensus_v2", "--evidence", str(source),
               "--config", str(config_path), "--out", str(out), "--report", str(report)]
    for existing in (False, True):
        if existing:
            for path in (out, report, audit):
                path.write_text("previous result\n")
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True)
        assert result.returncode == 2, result.stderr
        assert message in result.stderr
        assert "wrote " not in result.stdout
        assert "Traceback" not in result.stderr
        for path in (out, report, audit):
            if existing:
                assert path.read_text() == "previous result\n"
            else:
                assert not path.exists()


def test_valid_scores_missing_evidence_and_negative_penalty_unchanged():
    cfg = load_v2_config(config())
    cfg["hpa_penalty"] = {"organs": ["brain"], "expr_threshold": 1,
                          "per_organ_weight": -2, "cap": -4}
    df = evidence()
    df.loc[1, "gnomad_pLI"] = np.nan
    df["hpa_brain_expr"] = [0, 0, 2]
    out, _ = run_consensus(df, cfg, sensitivity=True)
    assert out["hgnc_id"].tolist() == ["HGNC:1", "HGNC:2", "HGNC:3"]
    assert out["consensus_score"].tolist() == [0, 0, 0]
    assert out["evidence_status"].tolist() == ["available", "no_usable_evidence", "available"]


def test_tuning_and_sensitivity_also_reject_overflow():
    cfg = load_v2_config(config())
    cfg["evidence"]["other"] = {"weight": 2}
    cfg["tuning"].update(enabled=True, positive_controls=["A", "B", "C"],
                         weight_grid=[1e308])
    df = evidence()
    df["other"] = df["gnomad_pLI"]
    with pytest.raises(ValueError, match="non-finite consensus"):
        run_consensus(df, cfg, sensitivity=True)
