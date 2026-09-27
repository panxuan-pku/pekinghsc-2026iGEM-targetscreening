"""Phenotype panels are annotations, never an independent ranking."""
import copy

import pandas as pd
import pytest

from screening.src.consensus_v2 import load_v2_config, render_report, run_consensus


PANELS = [
    {"id": "group_a", "disease": "Synthetic disease", "phenotype": "Phenotype A",
     "tissues": ["cerebral cortex", "cerebellum"]},
    {"id": "group_b", "disease": "Synthetic disease", "phenotype": "Phenotype B",
     "tissues": ["kidney"]},
]


def configuration(panels):
    return load_v2_config({
        "consensus_v2": {"evidence": {"gnomad_pLI": {"weight": 1, "direction": "higher_better"}},
                         "tuning": {"positive_controls": []},
                         "hpa_penalty": {"per_organ_weight": 0, "cap": 0}},
        "phenotype_panels": panels,
    })


def test_panels_preserve_scores_and_show_all_candidates_and_missingness():
    frame = pd.DataFrame({"hgnc_id": ["HGNC:1", "HGNC:2", None],
                          "input_symbol": ["A", "B", "UNKNOWN"],
                          "gnomad_pLI": [0.9, 0.2, None],
                          "hpa_tissue::cerebral cortex::ntpm": [0.0, 20.0, None]})
    baseline, base_meta = run_consensus(frame, configuration([]), sensitivity=True)
    out, meta = run_consensus(frame, configuration(PANELS), sensitivity=True)
    pd.testing.assert_frame_equal(out, baseline, check_exact=True)
    assert meta["sensitivity"] == base_meta["sensitivity"]
    a, b = meta["phenotype_panels"]
    assert a["coverage_status"] == "partial"
    assert a["missing_tissues"] == ["cerebellum"]
    assert a["genes_with_expression"] == 2  # zero is measured, not missing
    assert b["coverage_status"] == "unavailable"
    report = render_report(out, meta, top_n=1)
    panel_text = report.split("## Phenotype evidence panels", 1)[1]
    assert "Phenotype A" in panel_text and "Phenotype B" in panel_text
    assert "UNKNOWN" in panel_text and " B " in panel_text
    assert "No data" in panel_text and "Incomplete coverage" in panel_text
    assert "not a phenotype-specific ranking" in panel_text
    assert "## Phenotype evidence panels" not in render_report(baseline, base_meta)


@pytest.mark.parametrize("panels", [None, {}, "brain", ["brain"],
    [{"id": "a", "disease": "D", "phenotype": "P", "tissues": []}],
    [{"id": "a", "disease": "D", "phenotype": "P", "tissues": "brain"}],
    [{"id": "a", "disease": "D", "phenotype": "P", "tissues": ["brain", "brain"]}],
    [PANELS[0], PANELS[0]],
    [{"id": "a", "disease": "", "phenotype": "P", "tissues": ["brain"]}],
])
def test_bad_panel_configuration_fails_clearly(panels):
    with pytest.raises(ValueError, match="phenotype_panels"):
        configuration(panels)


def test_complete_panel_and_legacy_evidence():
    frame = pd.DataFrame({"hgnc_id": ["HGNC:1"], "input_symbol": ["A"], "gnomad_pLI": [0.9]})
    out, meta = run_consensus(frame, configuration(PANELS[:1]))
    assert meta["phenotype_panels"][0]["coverage_status"] == "unavailable"
    assert "no available tissue columns" in render_report(out, meta)
    frame["hpa_tissue::cerebral cortex::ntpm"] = 0.0
    frame["hpa_tissue::cerebellum::ntpm"] = 2.0
    _, meta = run_consensus(frame, configuration(PANELS[:1]))
    assert meta["phenotype_panels"][0]["coverage_status"] == "complete"
    changed = copy.deepcopy(PANELS[:1])
    changed[0]["phenotype"] = "A | B\nC"
    out, meta = run_consensus(frame, configuration(changed))
    assert "A \\| B C" in render_report(out, meta)
