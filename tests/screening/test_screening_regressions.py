"""Regression cases from the 2026-09-19 screening audit."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from screening.src.ai_scores import build_symbol_map
from screening.src.normalize import load_alias_map, resolve
from screening.src.merge_evidence import join_on_hgnc
from screening.src.filter_recessive import load_omim_recessive, load_clinvar_recessive
from screening.src.consensus_v2 import load_v2_config, normalize_series, run_consensus, render_report

ROOT = Path(__file__).resolve().parents[2]


def config():
    return load_v2_config(yaml.safe_load((ROOT / "screening/config/pipeline.yaml").read_text()))


def test_padded_ids_headers_and_alias_ambiguity(tmp_path):
    path = tmp_path / "aliases.tsv"
    path.write_text(" hgnc_id \t symbol \t prev_symbol \t alias_symbol \n"
                    " HGNC:1 \t NEW \t OLD;OLDER \t SHARED,OTHER \n"
                    "HGNC:2\tOTHER\tOLDER2\tSHARED\n")
    m = load_alias_map(path)
    assert m["old"] == m["older"] == "HGNC:1"
    assert m["other"] == "HGNC:2"  # approved symbol beats alias
    assert "shared" not in m
    assert build_symbol_map(path)["NEW"] == "HGNC:1"
    assert resolve(pd.DataFrame({"gene_symbol": [" NEW ", None]}), m).status.tolist() == ["ok", "unresolved"]


@pytest.mark.parametrize("method", ["raw", "zscore", "rank"])
@pytest.mark.parametrize("direction", ["higher_better", "lower_better"])
def test_constant_column_keeps_missing(method, direction):
    values = normalize_series(pd.Series([0.9, 0.9, np.nan]), method, direction)
    assert pd.isna(values.iloc[2])
    assert values.iloc[0] == values.iloc[1]


def test_clingen_categories_survive_join_and_score(tmp_path):
    path = tmp_path / "clingen.tsv"
    path.write_text("#snapshot\n#Gene Symbol\tHaploinsufficiency Score\tTriplosensitivity Score\t"
                    "Haploinsufficiency PMID1\tDate Last Evaluated\n" +
                    "".join(f"G{i}\t{v}\t0\t1234\t2026-09-19\n" for i, v in enumerate([0, 1, 2, 3, 30, 40, -1])))
    df = pd.DataFrame({"hgnc_id": [f"HGNC:{i}" for i in range(8)],
                       "input_symbol": [f"G{i}" for i in range(8)], "gnomad_pLI": [0.8] * 8,
                       "recessive": [False] * 8})
    out = join_on_hgnc(df, path, "clinGen", dict(zip(df.input_symbol, df.hgnc_id)))
    assert out.loc[4, "clinGen_haploinsufficiency_raw"] == 30
    assert out.loc[5, "clinGen_haploinsufficiency_status"] == "dosage_unlikely"
    assert out.loc[6, "clinGen_haploinsufficiency_status"] == "not_evaluated"
    assert out.loc[7, "clinGen_haploinsufficiency_status"] == "not_curated"
    assert out.loc[0, "clinGen_haploinsufficiency_pmid1"] == "1234"
    scored, meta = run_consensus(out, config())
    by_symbol = scored.set_index("input_symbol")
    assert by_symbol.loc[[f"G{i}" for i in range(8)], "contrib_clinGen_hi_score"].tolist() == [0, 1, 2, 3, 0, 0, 0, 0]
    assert bool(by_symbol.loc["G5", "dosage_conflict"])
    assert "dosage_unlikely" in render_report(scored, meta)


def test_recessive_only_applied_once_and_before_sensitivity():
    df = pd.DataFrame({"hgnc_id": ["HGNC:1", "HGNC:2"], "input_symbol": ["A", "B"],
                       "gnomad_pLI": [0.9, 0.9], "recessive": ["True", "False"]})
    cfg = config()
    out, _ = run_consensus(df, cfg)
    assert out.consensus_score.tolist() == [1, 1]  # default is annotation only
    cfg["recessive"]["gnomad_weight_multiplier"] = 0.5
    out, meta = run_consensus(df, cfg, sensitivity=True)
    scores = out.set_index("input_symbol").contrib_gnomad_pLI
    assert scores["A"] == 0.5 and scores["B"] == 1
    assert meta["recessive_applied"]
    assert meta["sensitivity"]["max_rank_shift"] == 0  # both positive perturbations preserve B>A


def test_tied_loo_uses_same_ranks_as_output():
    df = pd.DataFrame({"hgnc_id": [f"HGNC:{i:02d}" for i in range(13)],
                       "input_symbol": [f"G{i}" for i in range(13)], "gnomad_pLI": [1.] * 13})
    cfg = config()
    cfg["tuning"].update({"positive_controls": ["G10", "G11", "G12"], "enabled": True})
    out, meta = run_consensus(df, cfg)
    assert meta["controls_list"] == ["G10", "G11", "G12"]
    assert meta["tuning"]["loo_hit_rate"] == meta["tuning"]["final_recall"] == 0
    ranks = out.set_index("input_symbol")["rank"]
    for fold in meta["tuning"]["folds"]:
        assert fold["held_out_rank"] == ranks[fold["held_out"]]


def test_duplicate_and_empty_evidence_fail_closed():
    df = pd.DataFrame({"hgnc_id": ["HGNC:1", " HGNC:1 "], "gnomad_pLI": [0.9, 0.2]})
    with pytest.raises(ValueError, match="duplicate HGNC"):
        run_consensus(df, config())
    with pytest.raises(ValueError, match="without usable evidence"):
        run_consensus(pd.DataFrame({"hgnc_id": ["HGNC:1"]}), config())


def test_inheritance_parser_uses_phenotype_not_symbol_column(tmp_path):
    path = tmp_path / "genemap2.txt"
    path.write_text("# generated header\n# Chromosome\tGene Symbols\tApproved Gene Symbol\tPhenotypes\tMouse Gene Symbol/ID\n"
                    "chr1\tOLD\tNEW\tDisorder, 123 (3), Autosomal recessive\t\n"
                    "chr1\tMIXED\tMIXED\tAutosomal recessive; Autosomal dominant\t\n"
                    "chrX\tXL\tXL\tX-linked recessive\t\n")
    assert load_omim_recessive(path) == {"NEW"}
    path.write_text("GeneSymbol\tModeOfInheritance\nA\tAutosomal recessive\nB\tAutosomal recessive\nB\tAutosomal dominant\n")
    assert load_clinvar_recessive(path) == {"A"}



def test_cli_config_controls_audit_and_failure_modes(tmp_path):
    cfg = yaml.safe_load((ROOT / "screening/config/pipeline.yaml").read_text())
    cfg["pipeline_mode"] = "exploratory"
    cfg["consensus_v2"]["tuning"]["positive_controls"] = ["A", "ABSENT"]
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    ev = tmp_path / "evidence.parquet"
    pd.DataFrame({"hgnc_id": [" HGNC:1 "], "input_symbol": ["A"],
                  "gnomad_pLI": [0.9]}).to_parquet(ev)
    ai = tmp_path / "ai.csv"
    ai.write_text("hgnc_id,DeepLOF_score\nHGNC:1,0.99\n")
    out, report = tmp_path / "out.csv", tmp_path / "report.md"
    cmd = [sys.executable, "-m", "screening.src.consensus_v2", "--evidence", str(ev), "--ai-scores", str(ai),
           "--config", str(cfg_path), "--out", str(out), "--report", str(report)]
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert pd.read_csv(out).DeepLOF_score.iloc[0] == 0.99
    audit = json.loads(Path(str(out) + ".audit.json").read_text())
    assert audit["results"]["mode"] == "exploratory"
    assert audit["results"]["controls_total"] == 2
    assert audit["results"]["controls_present_n"] == 1
    assert "Positive-control check" in report.read_text()
    assert audit["results"]["experiments"] == []
    # Opposing dosage evidence is visible and prevents an automatic pass,
    # even when population constraint is high.
    pd.DataFrame({"hgnc_id": ["HGNC:1"], "input_symbol": ["A"],
                  "gnomad_pLI": [0.99], "clinGen_haploinsufficiency_raw": [40],
                  "clinGen_hi_score": [np.nan]}).to_parquet(ev)
    validation = subprocess.run(cmd + ["--mode", "validate"], cwd=ROOT, text=True, capture_output=True)
    assert validation.returncode == 0, validation.stderr
    audit = json.loads(Path(str(out) + ".audit.json").read_text())
    assert audit["results"]["validation_pass"] is False
    assert audit["results"]["experiments"] == []
    ai.write_text("hgnc_id,DeepLOF_score\nHGNC:1,0.99\nHGNC:1,0.1\n")
    bad = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    assert bad.returncode != 0


def test_validate_does_not_claim_therapy_suitability(tmp_path):
    # Rendering an older metadata structure must still not make therapeutic claims.
    df = pd.DataFrame({"hgnc_id": ["HGNC:1"], "input_symbol": ["A"], "gnomad_pLI": [0.9]})
    out, meta = run_consensus(df, config(), mode="validate")
    meta.update({"validation_panel": [{"gene": "A", "rank": 1, "consensus_score": 1,
                                        "gnomad_pLI": 0.9, "target_ok": True}], "validation_pass": True})
    report = render_report(out, meta)
    assert "Meets SINEUP targeting conditions" not in report
    assert "Preliminary genetic support" in report
