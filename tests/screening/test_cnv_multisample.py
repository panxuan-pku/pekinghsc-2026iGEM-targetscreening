"""Exercise real matrix reading, inference, reruns and separate study outputs."""
import gzip
import json

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.io import mmwrite
from scipy.sparse import csr_matrix

from screening.src.cnv import workflow as w
from screening.src.cnv.infercnv import InsufficientDataError
from .test_cnv_infercnv_synthetic import _synthetic
from .test_cnv_input_contract import _sample
from .test_cnv_workflow import cfg, _audit, _out


def test_two_studies_four_samples_and_rerun_keep_counts_and_identity(tmp_path):
    preserved = {}
    for study in range(2):
        root = tmp_path / f"study_{study}"
        root.mkdir()
        ad, genes = _synthetic(seed=study)
        genes[["start", "end"]] += 1
        order = root / "order.tsv"
        genes.to_csv(order, sep="\t", index=False)
        samples = []
        symbols = genes.gene_name.tolist()
        symbols[:2] = ["SHARED", "SHARED"]
        for i, name in enumerate(["P1", "P2", "R1", "R2"]):
            path = root / name
            path.mkdir()
            with gzip.open(path / "matrix.mtx.gz", "wb") as stream:
                mmwrite(stream, csr_matrix(ad.layers["counts"][i * 60:(i + 1) * 60].T))
            with gzip.open(path / "features.tsv.gz", "wt") as stream:
                stream.write("".join(f"{gene_id}\t{symbol}\tGene Expression\n"
                                     for gene_id, symbol in zip(genes.gene_id, symbols)))
            with gzip.open(path / "barcodes.tsv.gz", "wt") as stream:
                stream.write("".join(f"CELL{j}\n" for j in range(60)))
            samples.append({"id": name, "group": "patient" if i < 2 else "reference", "path": str(path)})
        config = {"project": f"study_{study}", "samples": samples,
                  "annotation": {"gene_order_tsv": str(order), "genome_build": "GRCh38"},
                  "infercnv": {"resolutions": [{"name": "test", "window_size": 50, "step": 10}]},
                  "known_intervals": {"deletion": {"chr": "chr2", "start": 500001, "end": 1000000,
                                                  "genome_build": "GRCh38", "markers": ["G250"]}},
                  "extract": {"mode": "cnv"}, "output": {"dir": str(root / "out")}}
        for repeat in range(2):
            w.cmd_infercnv(config)
            w.cmd_call_segments(config)
            _, _, validation = w.cmd_extract_genes(config)
            loaded = anndata.read_h5ad(root / "out/cnv_input.h5ad")
            assert loaded.obs_names.is_unique and loaded.shape == (240, 600)
            assert [json.loads(n) for n in loaded.obs_names] == [
                [s["id"], f"CELL{j}"] for s in samples for j in range(60)]
            assert loaded.obs["group"].tolist() == [s["group"] for s in samples for _ in range(60)]
            np.testing.assert_array_equal(loaded.layers["counts"].toarray(), ad.layers["counts"])
            assert loaded.var["gene_ids"].tolist() == genes.gene_id.tolist()
            assert loaded.var_names.tolist() == genes.gene_id.tolist()
            assert loaded.var.gene_symbols.tolist() == symbols
            mapping = pd.read_csv(root / "out/gene_position_mapping.csv")
            assert mapping.gene_symbol.tolist() == symbols
            assert mapping.annotation_symbol.tolist() == genes.gene_name.tolist()
            assert validation["AC-CNV-2 (markers detected in data)"].tolist() == ["PASS"]
            assert validation["AC-CNV-1 (auto recovers known interval)"].iloc[0] in ("PASS", "FAIL")
            paths = [root / "out" / name for name in ("gene_position_mapping.csv", "window_signal_test.tsv", "segments_all.csv",
                     "candidates.csv", "candidate_provenance.csv", "interval_gene_expression_qc.csv",
                     "validation_known_intervals.csv", "cnv_report.md")]
            if repeat == 0:
                first = {p: p.read_bytes() for p in paths}
            else:
                assert all(p.read_bytes() == data for p, data in first.items())
            assert all(p.read_bytes() == data for p, data in preserved.items())
        preserved.update(first)


def test_workflow_missing_reference_stops_before_loading(cfg, tmp_path, monkeypatch):
    from screening.src.cnv import infercnv as ic
    cfg["samples"] = [_sample(tmp_path)]
    def forbidden(*args, **kwargs):
        pytest.fail("matrix loading must not run without a reference group")
    monkeypatch.setattr(ic, "load_samples", forbidden)
    with pytest.raises(InsufficientDataError, match="reference"):
        w.cmd_infercnv(cfg)
    assert _audit(cfg, "infercnv")["validation_status"] == "INSUFFICIENT_DATA"
    assert not (_out(cfg) / "cnv_input.h5ad").exists()
