"""Offline core regressions; no model download, server or scientific data needed."""
from pathlib import Path
import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = (Path(__file__).resolve().parents[2] / "vct")
sys.path.insert(0, str(ROOT / "src"))
from cipher_engine import CipherEngine
from linear_baseline import LinearBaseline


class EngineSafetyTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[1., 2.], [2., 4.], [3., 6.]])

    def test_subset_unknown_gene_rejected(self):
        engine = CipherEngine()
        engine.fit(self.x, ["A", "B"])
        for genes in (["MISSING", "B"], ["A", "MISSING"], ["MISSING"]):
            with self.subTest(genes=genes), self.assertRaisesRegex(ValueError, "MISSING"):
                engine.predict({"A": 0}, gene_list=genes)

    def test_known_subset_names_and_values_match_full_delta(self):
        engine = CipherEngine()
        engine.fit(self.x, ["A", "B"])
        expected = engine.predict({"A": 0})
        result = engine.predict({"A": 0}, gene_list=["B", "A"])
        self.assertEqual(result["delta_subset"], {"B": expected["delta"][1], "A": expected["delta"][0]})
        np.testing.assert_array_equal(result["delta"], expected["delta"])

    def test_invalid_fit_preserves_prior_model(self):
        invalid = [(np.empty((0, 2)), ["A", "B"]), (self.x, ["A"]),
                   (self.x, ["A", "A"]), (self.x, ["", "B"]),
                   (self.x.ravel(), ["A", "B"]), (np.empty((3, 0)), []),
                   (np.array([[np.nan, 1], [2, 3]]), ["A", "B"]),
                   (np.array([[np.inf, 1], [2, 3]]), ["A", "B"])]
        for cls in (CipherEngine, LinearBaseline):
            for x, genes in invalid:
                with self.subTest(engine=cls.__name__, shape=x.shape, genes=genes):
                    fresh = cls()
                    with self.assertRaises(ValueError):
                        fresh.fit(x, genes)
                    self.assertFalse(fresh._fitted)
                    engine = cls()
                    engine.fit(self.x, ["A", "B"])
                    before = engine._ctrl_mean.copy()
                    with self.assertRaises(ValueError):
                        engine.fit(x, genes)
                    np.testing.assert_array_equal(engine._ctrl_mean, before)
                    self.assertEqual(engine._gene_list, ["A", "B"])

    def test_sample_minimum_is_specific_to_engine(self):
        with self.assertRaises(ValueError):
            CipherEngine().fit(self.x[:1], ["A", "B"])
        baseline = LinearBaseline()
        baseline.fit(sparse.csr_matrix(self.x[:1]), ["A", "B"])
        np.testing.assert_array_equal(baseline._ctrl_mean, [1, 2])

    def test_stats_dimensions_and_finite_values(self):
        baseline = LinearBaseline()
        baseline.fit_stats(np.array([1., 2.]), ["A", "B"])
        for mean, genes in [([1], ["A", "B"]), ([np.nan, 2], ["A", "B"]),
                            ([1, 2], ["A", "A"]), ([[1, 2]], ["A", "B"]), ([], [])]:
            with self.subTest(mean=mean, genes=genes), self.assertRaises(ValueError):
                baseline.fit_stats(np.asarray(mean), genes)
            np.testing.assert_array_equal(baseline._ctrl_mean, [1, 2])

    def test_sparse_baseline_does_not_densify(self):
        x = sparse.csr_matrix(self.x)
        with patch.object(sparse.csr_matrix, "toarray", side_effect=AssertionError("dense")):
            baseline = LinearBaseline()
            baseline.fit(x, ["A", "B"])
        np.testing.assert_array_equal(baseline._ctrl_mean, self.x.mean(axis=0))


class ResourceSafetyTests(unittest.TestCase):
    def test_cli_resource_errors_do_not_create_outputs(self):
        import viz_perturbation as viz
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result"
            for error in (ValueError("estimated memory exceeds budget"), MemoryError()):
                stderr = io.StringIO()
                with self.subTest(error=type(error).__name__), \
                        patch.object(sys, "argv", ["viz", "--gene", "A", "--output", str(output)]), \
                        patch.object(viz, "load_data", side_effect=error), contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as stopped:
                        viz.main()
                    self.assertEqual(stopped.exception.code, 2)
                    self.assertFalse(output.exists())
                    self.assertIn("insufficient memory" if isinstance(error, MemoryError)
                                  else "estimated memory exceeds budget", stderr.getvalue())

    def test_large_shape_rejected_before_loading_expression(self):
        import viz_perturbation as viz
        with tempfile.TemporaryDirectory() as tmp:
            # Only the sparse archive shape is needed to reject the allocation.
            np.savez(Path(tmp) / "ws_expr.npz", shape=np.array([96969, 28231]))
            with patch.object(viz, "DATA_DIR", tmp), patch.object(viz.sp, "load_npz") as load:
                with self.assertRaisesRegex(ValueError, "GiB"):
                    viz.load_data("ws")
                load.assert_not_called()

    def test_small_matrix_and_invalid_budget(self):
        import viz_perturbation as viz
        with tempfile.TemporaryDirectory() as tmp:
            x = sparse.csr_matrix([[1., 2.], [3., 4.]])
            sparse.save_npz(Path(tmp) / "expr_aligned.npz", x)
            pd.DataFrame({"cell_type": ["A", "B"]}).to_csv(Path(tmp) / "meta.csv", index=False)
            (Path(tmp) / "gene_order.txt").write_text("A\nB\n")
            with patch.object(viz, "DATA_DIR", tmp):
                actual, _, genes = viz.load_data("pbmc")
                np.testing.assert_array_equal(actual, x.toarray())
                self.assertEqual(genes, ["A", "B"])
                for budget in (0, -1, np.nan, np.inf):
                    with self.subTest(budget=budget), self.assertRaises(ValueError):
                        viz.load_data("pbmc", max_memory_gib=budget)


class AttributionOutputTests(unittest.TestCase):
    def setUp(self):
        import compute_ms_attribution as ms
        self.ms = ms
        self.x = np.array([[1., 9.], [3., 11.], [0., 0.], [2., 2.]])
        self.groups = np.array(["MS", "MS", "normal", "normal"])
        self.table = ms.differential_table(self.x, self.x, self.groups, ["A", "B"])
        self.meta = pd.DataFrame({"cell": ["c1", "c2", "c3", "c4"], "group": self.groups})

    def save(self, output):
        self.ms.save_run(output, self.x, np.ones((4, 2)), self.meta, self.table, ["A", "B"])

    def test_effect_sort_and_formula(self):
        self.assertEqual(self.table["gene"].tolist(), ["B", "A"])
        np.testing.assert_allclose(self.table["effect_size"], np.array([9., 1.]) / (1 + 1e-6))
        with self.assertRaisesRegex(ValueError, "MS.*normal"):
            self.ms.differential_table(self.x, self.x, np.array(["MS"] * 4), ["A", "B"])

    def test_complete_bundle_preserves_array_gene_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            self.save(output)
            self.assertEqual(len(list(output.iterdir())), 5)
            np.testing.assert_array_equal(np.load(output / "ms_attribution_fp16.npy"), self.x)
            self.assertEqual((output / "gene_order.txt").read_text(), "A\nB\n")
            self.assertEqual(pd.read_csv(output / "ms_differential_attribution.csv")["gene"].tolist(), ["B", "A"])
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaisesRegex(ValueError, "--output"):
                self.save(output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_late_write_failure_leaves_no_partial_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            with patch.object(pd.DataFrame, "to_csv", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    self.save(output)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_invalid_results_are_not_published(self):
        for bad in (np.ones((3, 2)), np.full((4, 2), np.nan), np.full((4, 2), np.inf)):
            with self.subTest(shape=bad.shape), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "run"
                with self.assertRaises(ValueError):
                    self.ms.save_run(output, bad, np.ones((4, 2)), self.meta, self.table, ["A", "B"])
                self.assertFalse(output.exists())

    def test_cli_refuses_existing_directory_before_loading_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = Path(tmp) / "keep.txt"
            sentinel.write_text("unchanged")
            result = subprocess.run([sys.executable, str(ROOT / "src/compute_ms_attribution.py"),
                                     "--output", tmp], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("--output", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertNotIn("1/6", result.stdout)
            self.assertEqual(sentinel.read_text(), "unchanged")


if __name__ == "__main__":
    unittest.main()
