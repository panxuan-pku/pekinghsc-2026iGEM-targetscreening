"""Sample identity is checked before staging or loading any sample."""
from pathlib import Path

import pytest

from screening.src.cnv import infercnv as ic
from screening.src.cnv import workflow as w
from .test_cnv_staging import _sample


@pytest.mark.parametrize("problem", ["duplicate_id", "duplicate_path", "symlink_path", "empty_id",
                                    "padded_id", "missing_group", "numeric_group", "empty_path", "empty_samples"])
@pytest.mark.parametrize("entry", ["stage", "load", "infercnv"])
def test_invalid_sample_identity_is_rejected_before_io(tmp_path, monkeypatch, problem, entry):
    monkeypatch.chdir(tmp_path)
    samples = [_sample(tmp_path, name) for name in ("A", "B")]
    if problem == "duplicate_id":
        samples[1]["id"] = "A"
    elif problem == "duplicate_path":
        samples[1]["path"] = str(tmp_path / "A" / ".." / "A")
    elif problem == "symlink_path":
        (tmp_path / "alias").symlink_to(tmp_path / "A", target_is_directory=True)
        samples[1]["path"] = str(tmp_path / "alias")
    elif problem == "empty_id":
        samples[1]["id"] = ""
    elif problem == "padded_id":
        samples[1]["id"] = " A "
    elif problem == "missing_group":
        samples[1].pop("group")
    elif problem == "numeric_group":
        samples[1]["group"] = 17
    elif problem == "empty_path":
        samples[1]["path"] = ""
    else:
        samples = []
    def unexpected(*args, **kwargs):
        pytest.fail("sample validation must precede reading and importing runtime dependencies")
    monkeypatch.setattr(ic, "_import_deps", unexpected)
    monkeypatch.setattr(w, "_load_gene_order", unexpected)
    with pytest.raises(ValueError, match="sample"):
        if entry == "stage":
            w.cmd_stage_samples({"samples": samples})
        elif entry == "load":
            ic.load_samples(samples)
        else:
            w.cmd_infercnv({"samples": samples, "output": {"dir": str(tmp_path / "out")}})
    assert not (tmp_path / "A").exists()
    assert not (tmp_path / "B").exists()
    assert not (tmp_path / "out").exists()


def test_real_multisample_loading_preserves_labels_and_counts(tmp_path):
    samples = [_sample(tmp_path, name) for name in ("P1", "P2", "R1", "R2")]
    for sample in samples:
        sample["group"] = "patient" if sample["id"].startswith("P") else "reference"
    w.cmd_stage_samples({"samples": samples})
    adata = ic.load_samples(samples, min_genes=1, max_mito_pct=100)
    assert adata.obs["sample"].tolist() == [s["id"] for s in samples]
    assert adata.obs["group"].tolist() == [s["group"] for s in samples]
    assert adata.n_obs == 4 and adata.n_vars == 2
    assert adata.layers["counts"].toarray().tolist() == [[9, 3]] * 4
    assert all(Path(s["path"]).is_dir() for s in samples)


def test_invalid_samples_leave_previous_infercnv_outputs_untouched(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    names = ("cnv_input.h5ad", "window_signal_default.tsv", "segments_all.csv", "candidates.csv")
    for name in names:
        (output / name).write_text("previous result\n")
    with pytest.raises(ValueError, match="samples"):
        w.cmd_infercnv({"samples": [], "output": {"dir": str(output)}})
    assert sorted(p.name for p in output.iterdir()) == sorted(names)
    assert all((output / name).read_text() == "previous result\n" for name in names)
