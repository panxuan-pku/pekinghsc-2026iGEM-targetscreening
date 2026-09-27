"""Local SIGnature wrapper contracts, using a tiny deterministic encoder."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
import torch

ROOT = (Path(__file__).resolve().parents[2] / "vct")
sys.path.insert(0, str(ROOT / "src"))
import perturb
import attribute


class SignatureContractTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.model.weight.copy_(torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
        wrapper = SimpleNamespace(model=self.model, gene_order=["A", "B", "C"])
        with patch.object(perturb, "SCimilarityWrapper", return_value=wrapper), \
                patch.object(attribute, "SCimilarityWrapper", return_value=wrapper):
            self.engine = perturb.PerturbEngine()
            self.attr = attribute.AttributionEngine()
        self.x = np.array([1., 2., 3.], dtype=np.float32)

    def test_single_cell_inputs_fail_before_model(self):
        invalid = [self.x[None, :], self.x[:2], np.array([]), [1., 2., 3.],
                   sparse.csr_matrix(self.x[None, :]), np.array([1., np.nan, 3.]),
                   np.array([1., np.inf, 3.]), np.array([1., -np.inf, 3.]),
                   np.array([1e300, 2., 3.]), np.array([True, False, True]),
                   self.x.astype(complex), self.x.astype(str), self.x.astype(object)]
        for x in invalid:
            for call in (lambda: self.engine.perturb(x, "A", 0.),
                         lambda: self.engine.perturb_multi(x, {"A": 0.}),
                         lambda: self.attr.attribute(x), lambda: self.attr.top_genes(x)):
                with self.subTest(value=repr(x)), patch.object(self.model, "forward") as forward:
                    with self.assertRaises(ValueError):
                        call()
                    forward.assert_not_called()

    def test_batch_inputs_fail_before_model(self):
        for x in (self.x, np.zeros((2, 2)), np.zeros((0, 3)), np.zeros((2, 1, 3)),
                  np.full((2, 3), np.nan), np.full((2, 3), 1e300),
                  sparse.csr_matrix(self.x[None, :]), [[1., 2., 3.]]):
            for call in (lambda: self.engine.embed(x),
                         lambda: self.engine.batch_perturb(x, "A", 0.)):
                with self.subTest(value=repr(x)), patch.object(self.model, "forward") as forward:
                    with self.assertRaises(ValueError):
                        call()
                    forward.assert_not_called()

    def test_unknown_gene_errors_are_consistent(self):
        for call in (lambda: self.engine.perturb(self.x, "MISSING", 0.),
                     lambda: self.engine.perturb_multi(self.x, {"A": 0., "MISSING": 0.}),
                     lambda: self.engine.batch_perturb(self.x[None, :], "MISSING", 0.)):
            with patch.object(self.model, "forward") as forward:
                with self.assertRaisesRegex(ValueError, "MISSING"):
                    call()
                forward.assert_not_called()

    def test_invalid_replacements_fail_before_model(self):
        for value in (np.nan, np.inf, -np.inf, 1e300, "0", None, True, 1j, [0.], np.array([0.])):
            for call in (lambda: self.engine.perturb(self.x, "A", value),
                         lambda: self.engine.perturb_multi(self.x, {"A": value}),
                         lambda: self.engine.batch_perturb(self.x[None, :], "A", value)):
                with self.subTest(value=repr(value)), patch.object(self.model, "forward") as forward:
                    with self.assertRaises(ValueError):
                        call()
                    forward.assert_not_called()

    def test_valid_embedding_and_perturbations_preserve_inputs(self):
        expected = self.x @ self.model.weight.detach().numpy().T
        np.testing.assert_allclose(self.engine.embed(self.x[None, :]), expected[None, :])
        changed = self.x.copy()
        changed[0] = 0.5
        result = self.engine.perturb(self.x, "A", 0.5)
        np.testing.assert_allclose(result["emb_old"], expected)
        np.testing.assert_allclose(result["emb_new"], changed @ self.model.weight.detach().numpy().T)
        self.assertAlmostEqual(result["displacement"], np.linalg.norm(result["delta"]), places=6)
        multi = self.engine.perturb_multi(self.x, {"A": 0.5})
        np.testing.assert_array_equal(result["emb_new"], multi["emb_new"])
        np.testing.assert_array_equal(self.engine.batch_perturb(self.x[None, :], "A", 0.5)[0], result["emb_new"])
        np.testing.assert_array_equal(self.x, [1., 2., 3.])
        integer = np.array([1, 2, 3])
        np.testing.assert_array_equal(self.engine.perturb(integer, "A", 0.5)["emb_new"], result["emb_new"])
        np.testing.assert_array_equal(integer, [1, 2, 3])

    def test_valid_attribution_and_batch_chunking(self):
        # Linear encoder + zero baseline has an exact IG result.
        np.testing.assert_allclose(self.attr.attribute(self.x), self.x * [5., 7., 9.], rtol=1e-5)
        self.assertEqual([row[0] for row in self.attr.top_genes(self.x, k=2)], ["C", "B"])
        x = np.tile(self.x, (513, 1))
        np.testing.assert_allclose(self.engine.embed(x), x @ self.model.weight.detach().numpy().T)


if __name__ == "__main__":
    unittest.main()
