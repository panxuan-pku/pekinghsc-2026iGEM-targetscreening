"""Sample staging must never silently combine old files with new inputs."""
import gzip
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from screening.src.cnv.workflow import cmd_stage_samples


def _gzip(path, text):
    with gzip.open(path, "wt") as f:
        f.write(text)


def _sample(tmp_path, name, columns=2):
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    genes = "ID1\tGENE1\nID2\tGENE2\n"
    if columns == 3:
        genes = genes.replace("\n", "\tGene Expression\n")
    for suffix, text in {
        "matrix.mtx.gz": "%%MatrixMarket matrix coordinate integer general\n2 1 2\n1 1 9\n2 1 3\n",
        "barcodes.tsv.gz": f"{name}_CELL\n",
        "genes.tsv.gz": genes,
    }.items():
        _gzip(raw / f"{name}_{suffix}", text)
    return {"id": name, "group": "patient", "raw_dir": str(raw),
            "raw_prefix": name, "path": str(tmp_path / name)}


def _snapshot(path):
    return {p.name: (p.is_symlink(), os.readlink(p) if p.is_symlink() else p.read_bytes(),
                     p.lstat().st_mtime_ns) for p in path.iterdir()}


@pytest.mark.parametrize("columns", [2, 3])
def test_first_run_and_repeat_preserve_inputs_and_outputs(tmp_path, columns):
    samples = [_sample(tmp_path, name, columns) for name in ("A", "B")]
    original = _snapshot(tmp_path / "raw")
    cmd_stage_samples({"samples": samples})
    for sample in samples:
        dest = Path(sample["path"])
        for name in ("matrix.mtx.gz", "barcodes.tsv.gz", "genes.tsv.gz"):
            assert (dest / name).is_symlink()
            assert (dest / name).resolve() == (tmp_path / "raw" / f"{sample['id']}_{name}").resolve()
        with gzip.open(dest / "features.tsv.gz", "rt") as f:
            assert f.read() == "ID1\tGENE1\tGene Expression\nID2\tGENE2\tGene Expression\n"
        assert (dest / "features.tsv.gz").is_symlink() == (columns == 3)
    before = [_snapshot(Path(s["path"])) for s in samples]
    cmd_stage_samples({"samples": samples})
    assert before == [_snapshot(Path(s["path"])) for s in samples]
    assert original == _snapshot(tmp_path / "raw")


@pytest.mark.parametrize("name", ["matrix.mtx.gz", "barcodes.tsv.gz", "genes.tsv.gz"])
@pytest.mark.parametrize("kind", ["regular", "wrong_link", "broken_link", "directory"])
def test_raw_conflict_in_last_sample_prevents_all_writes(tmp_path, name, kind):
    samples = [_sample(tmp_path, n) for n in ("A", "B")]
    dest = Path(samples[1]["path"])
    dest.mkdir()
    target = dest / name
    if kind == "regular":
        target.write_bytes(b"old input")
    elif kind == "wrong_link":
        target.symlink_to(tmp_path / "raw" / f"A_{name}")
    elif kind == "broken_link":
        target.symlink_to(tmp_path / "missing")
    else:
        target.mkdir()
    original = target.lstat()
    with pytest.raises(ValueError, match=name):
        cmd_stage_samples({"samples": samples})
    assert not Path(samples[0]["path"]).exists()
    assert list(dest.iterdir()) == [target]
    assert target.lstat() == original


@pytest.mark.parametrize("kind", ["stale", "extra", "reordered", "corrupt", "wrong_link", "broken_link", "directory"])
def test_features_conflict_prevents_all_writes(tmp_path, kind):
    samples = [_sample(tmp_path, n) for n in ("A", "B")]
    dest = Path(samples[1]["path"])
    dest.mkdir()
    target = dest / "features.tsv.gz"
    if kind in ("stale", "extra", "reordered"):
        rows = ["ID1\tGENE1\tGene Expression\n", "ID2\tGENE2\tGene Expression\n"]
        text = {"stale": rows[0], "extra": "".join(rows + rows[:1]),
                "reordered": "".join(reversed(rows))}[kind]
        _gzip(target, text)
    elif kind == "corrupt":
        target.write_bytes(b"not gzip")
    elif kind == "wrong_link":
        target.symlink_to(tmp_path / "raw" / "B_genes.tsv.gz")
    elif kind == "broken_link":
        target.symlink_to(tmp_path / "missing")
    else:
        target.mkdir()
    original = target.lstat()
    with pytest.raises(ValueError, match="features.tsv.gz"):
        cmd_stage_samples({"samples": samples})
    assert not Path(samples[0]["path"]).exists()
    assert list(dest.iterdir()) == [target]
    assert target.lstat() == original


@pytest.mark.parametrize("kind", ["missing_matrix", "corrupt_genes", "empty_genes", "uneven_genes",
                                 "blocked_parent", "shared_dest", "nested_file_dest"])
def test_invalid_last_sample_prevents_first_sample_creation(tmp_path, kind):
    samples = [_sample(tmp_path, n) for n in ("A", "B")]
    if kind == "missing_matrix":
        (tmp_path / "raw" / "B_matrix.mtx.gz").unlink()
    elif kind == "corrupt_genes":
        (tmp_path / "raw" / "B_genes.tsv.gz").write_bytes(b"not gzip")
    elif kind == "empty_genes":
        _gzip(tmp_path / "raw" / "B_genes.tsv.gz", "")
    elif kind == "uneven_genes":
        _gzip(tmp_path / "raw" / "B_genes.tsv.gz", "ID1\tGENE1\nID2\n")
    elif kind == "blocked_parent":
        (tmp_path / "blocked").write_text("keep")
        samples[1]["path"] = str(tmp_path / "blocked" / "B")
    elif kind == "shared_dest":
        samples[1]["path"] = str(tmp_path / "A" / ".." / "A")
    else:
        samples[1]["path"] = str(tmp_path / "A" / "matrix.mtx.gz" / "B")
    with pytest.raises((ValueError, OSError)):
        cmd_stage_samples({"samples": samples})
    assert not Path(samples[0]["path"]).exists()


def test_matching_relative_links_and_missing_features_are_supported(tmp_path):
    sample = _sample(tmp_path, "A", 3)
    dest = Path(sample["path"])
    dest.mkdir()
    for name in ("matrix.mtx.gz", "barcodes.tsv.gz", "genes.tsv.gz"):
        (dest / name).symlink_to(f"../raw/A_{name}")
    cmd_stage_samples({"samples": [sample]})
    assert (dest / "features.tsv.gz").resolve() == (tmp_path / "raw" / "A_genes.tsv.gz").resolve()
    assert os.readlink(dest / "matrix.mtx.gz") == "../raw/A_matrix.mtx.gz"


@pytest.mark.parametrize("columns", [2, 3])
def test_matching_regular_features_are_reused(tmp_path, columns):
    sample = _sample(tmp_path, "A", columns)
    dest = Path(sample["path"])
    dest.mkdir()
    feat = dest / "features.tsv.gz"
    _gzip(feat, "ID1\tGENE1\tGene Expression\nID2\tGENE2\tGene Expression\n")
    before = _snapshot(dest)[feat.name]
    cmd_stage_samples({"samples": [sample]})
    assert _snapshot(dest)[feat.name] == before


def test_same_content_features_link_to_wrong_source_is_rejected(tmp_path):
    sample = _sample(tmp_path, "A", 3)
    _sample(tmp_path, "B", 3)
    dest = Path(sample["path"])
    dest.mkdir()
    feat = dest / "features.tsv.gz"
    feat.symlink_to(tmp_path / "raw" / "B_genes.tsv.gz")
    before = _snapshot(dest)
    with pytest.raises(ValueError, match="features.tsv.gz"):
        cmd_stage_samples({"samples": [sample]})
    assert _snapshot(dest) == before


@pytest.mark.parametrize("columns", [2, 3])
def test_staged_sample_is_readable_by_scanpy(tmp_path, columns):
    import scanpy as sc

    sample = _sample(tmp_path, "A", columns)
    cmd_stage_samples({"samples": [sample]})
    adata = sc.read_10x_mtx(sample["path"])
    assert adata.obs_names.tolist() == ["A_CELL"]
    assert adata.var_names.tolist() == ["GENE1", "GENE2"]
    assert adata.X.toarray().tolist() == [[9, 3]]


def test_changed_genes_reject_old_generated_features(tmp_path):
    sample = _sample(tmp_path, "A")
    cmd_stage_samples({"samples": [sample]})
    _gzip(tmp_path / "raw" / "A_genes.tsv.gz", "ID2\tGENE2\nID1\tGENE1\n")
    before = (Path(sample["path"]) / "features.tsv.gz").read_bytes()
    with pytest.raises(ValueError, match="features.tsv.gz"):
        cmd_stage_samples({"samples": [sample]})
    assert (Path(sample["path"]) / "features.tsv.gz").read_bytes() == before


def test_cli_conflict_is_actionable_and_nonzero(tmp_path):
    sample = _sample(tmp_path, "A")
    dest = Path(sample["path"])
    dest.mkdir()
    (dest / "matrix.mtx.gz").write_bytes(b"old matrix")
    config = tmp_path / "cnv.yaml"
    config.write_text(yaml.safe_dump({"samples": [sample]}))
    result = subprocess.run([sys.executable, "-m", "screening.src.cnv.workflow", "stage-samples",
                             "--config", str(config)], cwd=Path(__file__).resolve().parents[2],
                            text=True, capture_output=True)
    assert result.returncode == 2
    assert str(dest / "matrix.mtx.gz") in result.stderr
    assert "Traceback" not in result.stderr
    assert "staged" not in result.stdout
    assert (dest / "matrix.mtx.gz").read_bytes() == b"old matrix"
