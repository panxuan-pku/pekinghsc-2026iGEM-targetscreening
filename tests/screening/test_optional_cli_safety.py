"""Optional CLI input failures must not publish misleading outputs."""
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from screening.src.ai_scores import cmd_deeplof
from screening.src.compensation import compute_compensation

ROOT = Path(__file__).resolve().parents[2]


def run_cli(module, *args):
    return subprocess.run([sys.executable, "-m", f"screening.src.{module}", *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True, timeout=60)


@pytest.fixture
def counts():
    data = ad.AnnData(np.array([[10, 90], [20, 180]]),
                     obs=pd.DataFrame({"condition": ["CTRL", "WS"]}, index=["c1", "c2"]),
                     var=pd.DataFrame(index=["A", "B"]))
    data.layers["counts"] = data.X.copy()
    return data


@pytest.mark.parametrize("genes, message", [([], "candidate gene list is empty"),
                                           (["MISSING"], "no candidate genes match")])
def test_compensation_rejects_zero_matches(counts, genes, message):
    with pytest.raises(ValueError, match=message):
        compute_compensation(counts, genes, full_gene_matrix=True)


@pytest.mark.parametrize("content", ["", "gene_symbol\n", "gene_symbol\nMISSING\n"])
@pytest.mark.parametrize("existing", [False, True])
def test_compensation_cli_preserves_output(tmp_path, counts, content, existing):
    h5ad, genes, out = tmp_path / "counts.h5ad", tmp_path / "genes.csv", tmp_path / "result/out.csv"
    counts.write_h5ad(h5ad)
    genes.write_text(content)
    if existing:
        out.parent.mkdir()
        out.write_text("previous result\n")
    result = run_cli("compensation", "--h5ad", h5ad, "--genes", genes,
                     "--full-gene-matrix", "--out", out)
    assert result.returncode == 2, result.stderr
    assert str(genes) in result.stderr
    assert "empty" in result.stderr or "no candidate genes match" in result.stderr
    assert "Traceback" not in result.stderr
    assert "wrote" not in result.stdout
    if existing:
        assert out.read_text() == "previous result\n"
    else:
        assert not out.parent.exists()


def test_compensation_cli_partial_match_keeps_calculation(tmp_path, counts):
    h5ad, genes, out = tmp_path / "counts.h5ad", tmp_path / "genes.csv", tmp_path / "result/out.csv"
    counts.write_h5ad(h5ad)
    genes.write_text("gene_symbol\nA\nMISSING\n")
    result = run_cli("compensation", "--h5ad", h5ad, "--genes", genes,
                     "--full-gene-matrix", "--out", out)
    assert result.returncode == 0, result.stderr
    assert "MISSING" in result.stderr
    actual = pd.read_csv(out)
    expected = compute_compensation(counts, ["A"], full_gene_matrix=True)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.compensation_ratio.tolist() == [1.0]
    assert actual.compensation_class.tolist() == ["full"]


@pytest.fixture
def ai_args(tmp_path):
    aliases = tmp_path / "aliases.tsv"
    aliases.write_text("hgnc_id\tsymbol\tprev_symbol\tensembl_gene_id\n"
                       "HGNC:1\tA\tOLD\tE1\nHGNC:1\tA\tOLD\tE2\n"
                       "HGNC:2\tB\t\tE3\n")
    return SimpleNamespace(scores=tmp_path / "scores.csv", aliases=aliases,
                           out=tmp_path / "result/ai.csv")


def audit_path(args):
    return args.out.parent / "audit" / f"ai_scores_deeplof_{time.strftime('%Y%m%d')}.json"


@pytest.mark.parametrize("value", ["bad", "NaN", "inf", "-inf", "1e300", "-0.1", "1.1", "True", ""])
def test_ai_invalid_scores_checked_before_any_write(ai_args, value):
    ai_args.scores.write_text(f"ensembl,gene_symbol,DeepLOF_score\nE1,A,{value}\nE2,B,0.9\n")
    with pytest.raises(ValueError, match="DeepLOF_score.*finite.*0.*1"):
        cmd_deeplof(ai_args)
    assert not ai_args.out.parent.exists()


@pytest.mark.parametrize("value", ["True", "False"])
def test_ai_boolean_column_is_not_a_score(ai_args, value):
    ai_args.scores.write_text(f"ensembl,gene_symbol,DeepLOF_score\nE1,A,{value}\n")
    with pytest.raises(ValueError, match="DeepLOF_score"):
        cmd_deeplof(ai_args)
    assert not ai_args.out.parent.exists()


@pytest.mark.parametrize("bad_input, message", [
    ("ensembl,gene_symbol,DeepLOF_score\nE1,A,bad\n", "DeepLOF_score"),
    ("ensembl,gene_symbol,DeepLOF_score\n", "empty"),
    ("ensembl,gene_symbol,DeepLOF_score\nE1,A,0.1\nE2,OLD,0.9\n", "conflicting"),
])
@pytest.mark.parametrize("existing", [False, True])
def test_ai_cli_failure_protects_result_and_audit(ai_args, bad_input, message, existing):
    ai_args.scores.write_text(bad_input)
    audit = audit_path(ai_args)
    if existing:
        audit.parent.mkdir(parents=True)
        ai_args.out.write_text("previous scores\n")
        audit.write_text("previous audit\n")
    result = run_cli("ai_scores", "deeplof", "--scores", ai_args.scores,
                     "--aliases", ai_args.aliases, "--out", ai_args.out)
    assert result.returncode == 2, result.stderr
    assert str(ai_args.scores) in result.stderr
    assert message in result.stderr
    assert "Traceback" not in result.stderr
    assert "wrote" not in result.stdout
    if existing:
        assert ai_args.out.read_text() == "previous scores\n"
        assert audit.read_text() == "previous audit\n"
    else:
        assert not ai_args.out.parent.exists()


@pytest.mark.parametrize("reverse", [False, True])
def test_ai_conflicts_report_identity_and_scores_regardless_of_order(ai_args, reverse):
    rows = ["E1,A,0.1", "E2,OLD,0.9"]
    if reverse:
        rows.reverse()
    ai_args.scores.write_text("ensembl,gene_symbol,DeepLOF_score\n" + "\n".join(rows))
    with pytest.raises(ValueError, match="conflicting") as error:
        cmd_deeplof(ai_args)
    for detail in ("HGNC:1", "A", "OLD", "0.1", "0.9"):
        assert detail in str(error.value)
    assert not ai_args.out.parent.exists()


def test_ai_valid_cli_keeps_identical_duplicates_and_unmapped_policy(ai_args):
    ai_args.scores.write_text("ensembl,gene_symbol,DeepLOF_score\n"
                             "E1,A,0\nE2,OLD,0.0\nE3,B,1\nE4,A,0.5\n")
    result = run_cli("ai_scores", "deeplof", "--scores", ai_args.scores,
                     "--aliases", ai_args.aliases, "--out", ai_args.out)
    assert result.returncode == 0, result.stderr
    actual = pd.read_csv(ai_args.out)
    assert actual.hgnc_id.tolist() == ["HGNC:1", "HGNC:2"]
    assert actual.DeepLOF_score.tolist() == [0.0, 1.0]
    audit = json.loads(audit_path(ai_args).read_text())
    assert audit["n_genes_total"] == 4 and audit["n_genes_mapped"] == 2
    assert audit["mapping_rate"] == 0.75
    assert audit["duplicate_policy"] == "identical_scores_only; conflicting_scores_rejected"
    assert audit["n_identical_duplicates_removed"] == 1
    assert audit["n_unmapped_ensembl"] == 1
    assert audit["unmapped_ensembl"][0]["ensembl"] == "E4"


def test_ai_discordant_symbol_and_ensembl_is_excluded_and_audited(ai_args):
    ai_args.scores.write_text("ensembl,gene_symbol,DeepLOF_score\n"
                             "E1,B,0.2\nE3,B,0.8\n")
    cmd_deeplof(ai_args)
    actual = pd.read_csv(ai_args.out)
    assert actual.hgnc_id.tolist() == ["HGNC:2"]
    assert actual.DeepLOF_score.tolist() == [0.8]
    audit = json.loads(audit_path(ai_args).read_text())
    assert audit["n_symbol_disagreements"] == 1
    assert audit["symbol_disagreements"][0]["ensembl_hgnc_id"] == "HGNC:1"
    assert audit["symbol_disagreements"][0]["symbol_hgnc_id"] == "HGNC:2"


def test_ai_aliases_require_ensembl_column(ai_args):
    ai_args.aliases.write_text("hgnc_id\tsymbol\nHGNC:1\tA\n")
    ai_args.scores.write_text("ensembl,gene_symbol,DeepLOF_score\nE1,A,0.2\n")
    with pytest.raises(ValueError, match="ensembl_gene_id"):
        cmd_deeplof(ai_args)
    assert not ai_args.out.parent.exists()
