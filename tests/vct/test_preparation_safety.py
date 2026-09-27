"""Offline preparation contracts; never run a legacy data-rebuilding import."""
import ast
import contextlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = (Path(__file__).resolve().parents[2] / "vct")
sys.path.insert(0, str(ROOT / "src"))


def preparation_module():
    tree = ast.parse((ROOT / "src/prepare_data.py").read_text())
    if not any(isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)
               for node in tree.body):
        raise AssertionError("prepare_data must be import-safe before testing; no data rebuild allowed")
    return importlib.import_module("prepare_data")


class PBMCPreparationTests(unittest.TestCase):
    def setUp(self):
        self.prepare = preparation_module()
        self.raw = ad.AnnData(sparse.csr_matrix([[1., 0.], [2., 3.], [0., 1.]]),
                             obs=pd.DataFrame(index=["c1", "c2", "c3"]),
                             var=pd.DataFrame(index=["A", "B"]))
        self.meta = pd.DataFrame({"cell": ["c3", "c1", "c2"],
                                  "cluster": ["2", "0", "1"],
                                  "cell_type": ["Type3", "Type1", "Type2"]})
        self.emb = np.arange(6, dtype=float).reshape(3, 2)

    def save(self, output, emb=None):
        self.prepare.save_run(output, self.raw, self.raw.copy(),
                              self.emb if emb is None else emb, self.meta,
                              self.emb, "approved-labels.csv")

    def test_labels_align_by_cell_id_not_row_position(self):
        result = self.prepare.validate_metadata(self.meta, self.raw.obs_names)
        self.assertEqual(result["cell_type"].tolist(), ["Type1", "Type2", "Type3"])
        self.assertEqual(result["cluster"].tolist(), ["0", "1", "2"])
        self.assertEqual(self.meta["cell"].tolist(), ["c3", "c1", "c2"])
        no_clusters = self.prepare.validate_metadata(self.meta.drop(columns="cluster"), self.raw.obs_names)
        self.assertEqual(list(no_clusters.columns), ["cell", "cell_type"])

    def test_invalid_annotations_rejected(self):
        invalid = [self.meta.iloc[:0], self.meta.drop(columns="cell_type"),
                   self.meta.assign(cell=["c1", "c1", "c3"]),
                   self.meta.assign(cell_type=["", "Type1", "Type2"]),
                   self.meta.assign(cell=[None, "c1", "c2"]),
                   self.meta.assign(cluster=[" ", "0", "1"])]
        for meta in invalid:
            with self.subTest(meta=meta.to_dict()), self.assertRaises(ValueError):
                self.prepare.validate_metadata(meta)
        for cells in (["c1", "c2"], ["c1", "c2", "other"], ["c1", "c1", "c3"]):
            with self.subTest(cells=cells), self.assertRaises(ValueError):
                self.prepare.validate_metadata(self.meta, cells)

    def test_complete_bundle_and_existing_output_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            self.save(output)
            self.assertEqual(len(list(output.iterdir())), 7)
            np.testing.assert_array_equal(sparse.load_npz(output / "expr_aligned.npz").toarray(), self.raw.X.toarray())
            self.assertEqual(pd.read_csv(output / "meta.csv")["cell"].tolist(), ["c1", "c2", "c3"])
            self.assertEqual((output / "gene_order.txt").read_text(), "A\nB\n")
            info = json.loads((output / "pbmc_info.json").read_text())
            self.assertEqual(info["metadata_source"], str(Path("approved-labels.csv").resolve()))
            self.assertEqual(info["purpose"], "PBMC tool feasibility test")
            self.assertEqual(ad.read_h5ad(output / "pbmc3k_raw.h5ad").obs_names.tolist(), ["c1", "c2", "c3"])
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaisesRegex(ValueError, "--output"):
                self.save(output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_invalid_arrays_and_late_write_failure_not_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            for emb in (self.emb[:2], np.full((3, 2), np.nan), np.full((3, 2), np.inf)):
                with self.subTest(shape=emb.shape), self.assertRaises(ValueError):
                    self.save(output, emb=emb)
                self.assertFalse(output.exists())
            with patch.object(pd.DataFrame, "to_csv", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    self.save(output)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_missing_metadata_cli_fails_without_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            result = subprocess.run([sys.executable, str(ROOT / "src/prepare_data.py"),
                                     "--output", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("--metadata", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertNotIn("1/5", result.stdout)
            self.assertFalse(output.exists())

    def test_mismatched_cells_stop_before_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels, output = Path(tmp) / "labels.csv", Path(tmp) / "run"
            self.meta.assign(cell=["c1", "c2", "unknown"]).to_csv(labels, index=False)
            self.raw.write_h5ad(Path(tmp) / "raw.h5ad")
            modules = {"scanpy": SimpleNamespace(settings=SimpleNamespace(), read_h5ad=lambda path: self.raw.copy(), datasets=SimpleNamespace(pbmc3k=lambda: self.raw.copy()))}
            with patch.dict(sys.modules, modules), \
                    patch.object(sys, "argv", ["prepare_data", "--input", str(Path(tmp) / "raw.h5ad"), "--metadata", str(labels), "--output", str(output)]), \
                    patch.object(self.prepare, "save_run") as save:
                with self.assertRaisesRegex(ValueError, "不匹配"):
                    self.prepare.main()
                save.assert_not_called()
            self.assertFalse(output.exists())

    def test_existing_output_cli_fails_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = Path(tmp) / "keep.txt"
            sentinel.write_text("unchanged")
            result = subprocess.run([sys.executable, str(ROOT / "src/prepare_data.py"),
                                     "--metadata", str(Path(tmp) / "missing.csv"), "--output", tmp],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("--output", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(sentinel.read_text(), "unchanged")

    def test_invalid_metadata_cli_fails_before_data_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels, output = Path(tmp) / "labels.csv", Path(tmp) / "run"
            self.meta.drop(columns="cell_type").to_csv(labels, index=False)
            result = subprocess.run([sys.executable, str(ROOT / "src/prepare_data.py"),
                                     "--metadata", str(labels), "--output", str(output)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("cell_type", result.stderr)
            self.assertNotIn("1/5", result.stdout)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse(output.exists())

    def test_cli_dataflow_with_stubbed_model_preserves_labels_and_raw(self):
        class Model:
            def eval(self):
                return self

            def __call__(self, x):
                return SimpleNamespace(numpy=lambda: x)

        def normalize(adata):
            adata.X = adata.X * 2  # Makes accidental mutation of the raw file visible.

        modules = {
            "scanpy": SimpleNamespace(settings=SimpleNamespace(), read_h5ad=lambda path: self.raw.copy(), datasets=SimpleNamespace(pbmc3k=lambda: self.raw.copy())),
            "torch": SimpleNamespace(no_grad=contextlib.nullcontext, tensor=lambda x: x),
            "SIGnature.models.scimilarity": SimpleNamespace(SCimilarityWrapper=lambda **kw:
                                                            SimpleNamespace(model=Model(), gene_order=["A", "B"])),
            "SIGnature.utils": SimpleNamespace(align_dataset=lambda data, *args, **kw: data,
                                               lognorm_counts=normalize),
            "umap": SimpleNamespace(UMAP=lambda **kw: SimpleNamespace(
                fit=lambda emb: SimpleNamespace(embedding_=emb))),
        }
        with tempfile.TemporaryDirectory() as tmp:
            labels, output = Path(tmp) / "labels.csv", Path(tmp) / "run"
            self.meta.to_csv(labels, index=False)
            self.raw.write_h5ad(Path(tmp) / "raw.h5ad")
            with patch.dict(sys.modules, modules), patch.object(sys, "path", sys.path.copy()), \
                    patch.object(sys, "argv", ["prepare_data", "--input", str(Path(tmp) / "raw.h5ad"), "--metadata", str(labels), "--output", str(output)]):
                self.prepare.main()
            raw = ad.read_h5ad(output / "pbmc3k_raw.h5ad")
            np.testing.assert_array_equal(raw.X.toarray(), self.raw.X.toarray())
            np.testing.assert_array_equal(sparse.load_npz(output / "expr_aligned.npz").toarray(), self.raw.X.toarray() * 2)
            self.assertEqual(pd.read_csv(output / "meta.csv")["cell_type"].tolist(), ["Type1", "Type2", "Type3"])


class GeneMaximumTests(unittest.TestCase):
    def run_maximum_step(self, script, x, directory):
        # Execute only the real max/save block: importing these legacy scripts
        # would run expensive reconstruction and overwrite their data directory.
        nodes = ast.parse((ROOT / "src" / script).read_text()).body
        start = next(i for i, node in enumerate(nodes) if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "maxv" for t in node.targets))
        end = next(i for i in range(start + 1, len(nodes)) if isinstance(nodes[i], ast.Expr)
                   and isinstance(nodes[i].value, ast.Call)
                   and ast.unparse(nodes[i].value.func) == "np.save")
        exec(compile(ast.Module(body=nodes[start:end + 1], type_ignores=[]), script, "exec"),
             {"X": x, "np": np, "os": os, "DATA": directory})

    def test_numeric_vector_without_full_matrix_densification(self):
        x = sparse.csr_matrix([[0., -2., 3., 0.], [0., 0., 1., 4.]])
        for dataset in ("ms", "ws"):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as tmp:
                with patch.object(sparse.csr_matrix, "toarray", side_effect=AssertionError("full matrix")):
                    self.run_maximum_step(f"prepare_{dataset}_web.py", x, tmp)
                values = np.load(Path(tmp) / f"{dataset}_gene_max.npy", allow_pickle=False)
                self.assertEqual(values.shape, (4,))
                self.assertNotEqual(values.dtype, object)
                np.testing.assert_array_equal(values, [0., 0., 3., 4.])

    def test_nonfinite_maximum_rejected_before_saving(self):
        for dataset in ("ms", "ws"):
            for value in (np.nan, np.inf):
                with self.subTest(dataset=dataset, value=value), tempfile.TemporaryDirectory() as tmp:
                    with self.assertRaises(ValueError):
                        self.run_maximum_step(f"prepare_{dataset}_web.py", sparse.csr_matrix([[value, 1.]]), tmp)
                    self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
