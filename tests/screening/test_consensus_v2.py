"""Tests for consensus v2 scoring."""
import numpy as np
import pandas as pd

from screening.src.consensus_v2 import (
    _default_config,
    build_scoring_matrix,
    borda_points,
    hpa_penalty_term,
    load_v2_config,
    loo_tune,
    normalize_series,
    run_consensus,
)

TEST_CONTROLS = ["TBX1", "ELN", "KCTD13", "RAI1", "GTF2I"]

CFG2 = load_v2_config({
    "consensus_v2": {"tuning": {"positive_controls": TEST_CONTROLS}},
    "consensus": {"base_weights": {
        "clinGen_hi_score": 3, "gnomad_LOEUF": 2, "gnomad_pLI": 2,
        "DeepLOF_score": 1, "negative_hpa_organ": -2}},
    "hpa_filter": {"organs": ["liver", "brain"], "high_expression_log": 1.0},
})


def test_rank_normalization_direction():
    s = pd.Series([0.1, 0.5, 0.9])
    hi = normalize_series(s, "rank", "higher_better")
    lo = normalize_series(s, "rank", "lower_better")
    assert hi.iloc[2] > hi.iloc[0]
    assert lo.iloc[0] > lo.iloc[2]           # lower raw value wins for lower_better
    assert abs(hi.iloc[0] + lo.iloc[0] - 1.0) < 1e-9


def test_missing_is_neutral_not_penalized():
    df = pd.DataFrame({
        "hgnc_id": ["HGNC:1", "HGNC:2"],
        "input_symbol": ["A", "B"],
        "gnomad_pLI": [0.99, np.nan],        # B has no data
    })
    cfg = dict(CFG2)
    cfg["evidence"] = {"gnomad_pLI": {"direction": "higher_better", "weight": 2}}
    cfg["hpa_penalty"] = {"organs": [], "expr_threshold": 1.0, "per_organ_weight": -2, "cap": -4}
    cfg["borda"] = {"enabled": False}
    out, _ = run_consensus(df, cfg)
    b = out[out["input_symbol"] == "B"].iloc[0]
    assert b["consensus_score"] == 0.0       # missing -> 0, not negative


def test_hpa_penalty_capped():
    df = pd.DataFrame({
        "hgnc_id": ["HGNC:1"],
        "input_symbol": ["A"],
        "hpa_liver_expr": [5.0], "hpa_brain_expr": [5.0],
        "hpa_kidney_expr": [5.0], "hpa_gastrointestinal_expr": [5.0],
    })
    term = hpa_penalty_term(df, CFG2["hpa_penalty"])  # 4 organs above threshold
    assert term[0] == -4.0                   # capped, not 4 * -2 = -8


def test_borda_missing_gets_median_points():
    normed = {"x": pd.Series([1.0, 0.5, np.nan])}
    pts = borda_points(normed, ["x"], [1.0], n=3, missing="median")
    assert pts[0] == 2.0                     # best gets n-1
    assert pts[2] == 1.0                     # missing gets median, not worst


def test_v1_config_fallback_derives_evidence():
    cfg2 = _default_config({"consensus": {"base_weights": {
        "clinGen_hi_score": 3, "gnomad_LOEUF": 2, "negative_hpa_organ": -2}}})
    assert cfg2["evidence"]["gnomad_LOEUF"]["direction"] == "lower_better"
    assert cfg2["evidence"]["clinGen_hi_score"]["direction"] == "ordinal"
    assert cfg2["hpa_penalty"]["per_organ_weight"] == -2
    assert "negative_hpa_organ" not in cfg2["evidence"]


def test_loo_tune_recovers_separable_signal():
    # Controls are the only genes with high pLI AND low LOEUF -> recall@N should be 1.0
    rows = []
    for i in range(40):
        rows.append({"hgnc_id": f"HGNC:{i}", "input_symbol": f"GENE{i}",
                     "gnomad_pLI": 0.1, "gnomad_LOEUF": 0.9})
    for i, c in enumerate(TEST_CONTROLS):
        rows.append({"hgnc_id": f"HGNC:C{i}", "input_symbol": c,
                     "gnomad_pLI": 0.99, "gnomad_LOEUF": 0.1})
    df = pd.DataFrame(rows)
    cfg = dict(CFG2)
    cfg["evidence"] = {"gnomad_pLI": {"direction": "higher_better", "weight": 2},
                       "gnomad_LOEUF": {"direction": "lower_better", "weight": 2}}
    cfg["hpa_penalty"] = {"organs": [], "expr_threshold": 1.0, "per_organ_weight": -2, "cap": -4}
    cfg["borda"] = {"enabled": False}
    A, cols, _, _ = build_scoring_matrix(df, cfg["evidence"], "rank")
    res = loo_tune(df, A, cols, [2.0, 2.0], cfg["tuning"], np.zeros(len(df)), "input_symbol")
    assert res["loo_hit_rate"] == 1.0
    assert res["final_recall"] == 1.0


def test_output_columns_and_sorting():
    df = pd.DataFrame({
        "hgnc_id": ["HGNC:1", "HGNC:2", "HGNC:3"],
        "input_symbol": ["TBX1", "X", "Y"],
        "clinGen_hi_score": [3, np.nan, 0],
        "gnomad_pLI": [0.99, 0.1, 0.2],
        "gnomad_LOEUF": [0.1, 0.9, 0.8],
    })
    out, meta = run_consensus(df, CFG2)
    assert out.iloc[0]["input_symbol"] == "TBX1"
    assert "borda_rank" in out.columns
    assert any(c.startswith("contrib_") for c in out.columns)
    assert 0.0 <= meta["coverage"]["clinGen_hi_score"] <= 1.0
