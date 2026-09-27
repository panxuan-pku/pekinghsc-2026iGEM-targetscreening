"""Tests for normalize.py"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from screening.src.normalize import load_alias_map, resolve


@pytest.mark.parametrize("separator", ["|", ";", ","])
@pytest.mark.parametrize("padding", ["", " "])
def test_quoted_alias_fields(tmp_path, separator, padding):
    alias = tmp_path / "aliases.tsv"
    names = separator.join(["ACF", "ASP", "APOBEC1CF"])
    alias.write_text(
        " hgnc_id \t symbol \t prev_symbol \t alias_symbol \n"
        f' HGNC:24086 \t A1CF \t{padding}"OLD1{separator}OLD2"{padding}\t{padding}"{names}"{padding}\n'
    )
    m = load_alias_map(alias)
    for name in ["a1cf", "acf", "asp", "apobec1cf", "old1", "old2"]:
        assert m[name] == "HGNC:24086"
    assert not any('"' in name for name in m)


def test_quoted_aliases_preserve_approved_priority_and_ambiguity(tmp_path):
    alias = tmp_path / "aliases.tsv"
    alias.write_text(
        "hgnc_id\tsymbol\talias_symbol\n"
        'HGNC:1\tNEW\t "SHARED|OTHER|UNIQUE" \n'
        'HGNC:2\tOTHER\t "SHARED" \n'
    )
    m = load_alias_map(alias)
    assert m["other"] == "HGNC:2"
    assert m["unique"] == "HGNC:1"
    assert "shared" not in m

def test_resolve_ok_and_unresolved(tmp_path):
    alias = tmp_path / "aliases.tsv"
    alias.write_text("hgnc_id,symbol,prev_symbol\nHGNC:1,KCTD13,\nHGNC:2,TBX1,\n")
    m = load_alias_map(str(alias))
    df = pd.DataFrame({"gene_symbol": ["KCTD13", "TBX1", "NOT_A_GENE"]})
    out = resolve(df, m)
    assert len(out) == 3
    assert out.iloc[0]["hgnc_id"] == "HGNC:1"
    assert out.iloc[2]["status"] == "unresolved"
    assert list(out.columns) == ["hgnc_id", "input_symbol", "status"]


@pytest.mark.parametrize("columns", [
    ["source"],
    ["source", "deletion_id", "chrom", "start", "end"],
])
def test_resolve_preserves_candidate_context(columns):
    df = pd.DataFrame({
        "gene_symbol": ["NSD2", "ENSG00000289346"],
        "source": ["known_interval", None],
        "deletion_id": ["wolf_hirschhorn_4p16.3", "williams_7q11.23"],
        "chrom": ["chr4", "chr7"],
        "start": [100, 74796150],
        "end": [200, 74890585],
    }, index=[8, 3])[["gene_symbol"] + columns]
    # Unrecognized columns must not override resolution or scoring fields.
    df["status"] = "override"
    df["consensus_score"] = 999
    out = resolve(df, {"nsd2": "HGNC:12766"})
    assert list(out.columns) == ["hgnc_id", "input_symbol", "status"] + columns
    pd.testing.assert_frame_equal(out[columns], df[columns].reset_index(drop=True))
    assert out["input_symbol"].tolist() == ["NSD2", "ENSG00000289346"]
    assert out["status"].tolist() == ["ok", "unresolved"]
    assert pd.isna(out.loc[1, "hgnc_id"])


@pytest.mark.parametrize("columns", [
    ["gene_symbol"],
    ["gene_symbol", "source", "deletion_id", "chrom", "start", "end"],
])
def test_resolve_rejects_empty_candidates(columns):
    with pytest.raises(ValueError, match="candidate input is empty"):
        resolve(pd.DataFrame(columns=columns), {})


@pytest.mark.parametrize("content", [
    "", "\n \n", "gene_symbol\n", "gene_symbol,source,deletion_id,chrom,start,end\n",
])
@pytest.mark.parametrize("existing_output", [False, True])
def test_cli_empty_candidates_fail_before_loading_aliases_or_writing(tmp_path, content, existing_output):
    candidates = tmp_path / "candidates.csv"
    candidates.write_text(content)
    output = tmp_path / "output" / "normalized.csv"
    if existing_output:
        output.parent.mkdir()
        output.write_bytes(b"existing result\n")
    result = subprocess.run([
        sys.executable, "-m", "screening.src.normalize", "--input", str(candidates),
        "--out", str(output), "--hgnc-alias", str(tmp_path / "missing_aliases.tsv"),
    ], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    assert result.returncode == 2
    assert "candidate input is empty" in result.stderr
    assert "check input or upstream filtering results" in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""
    if existing_output:
        assert output.read_bytes() == b"existing result\n"
    else:
        assert not output.parent.exists()


def test_rows_with_blank_symbols_remain_unresolved():
    out = resolve(pd.DataFrame({"gene_symbol": [None, " "]}), {})
    assert len(out) == 2
    assert out["status"].tolist() == ["unresolved", "unresolved"]


def test_cli_nonempty_candidates_still_succeed(tmp_path):
    candidates = tmp_path / "candidates.csv"
    candidates.write_text("gene_symbol,source\nNSD2,known_interval\n")
    aliases = tmp_path / "aliases.tsv"
    aliases.write_text("hgnc_id\tsymbol\nHGNC:12766\tNSD2\n")
    output = tmp_path / "output" / "normalized.csv"
    result = subprocess.run([
        sys.executable, "-m", "screening.src.normalize", "--input", str(candidates),
        "--out", str(output), "--hgnc-alias", str(aliases),
    ], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert pd.read_csv(output).to_dict("records") == [{
        "hgnc_id": "HGNC:12766", "input_symbol": "NSD2", "status": "ok", "source": "known_interval",
    }]
