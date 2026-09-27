"""Read-only historical evidence: no models, CellOracle or inference."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = (Path(__file__).resolve().parents[2] / "vct")


class HistoryTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("history_panel", ROOT / "src/history_panel.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_real_archive_is_recomputed_without_changing_sources(self):
        if not (self.module.REPO / self.module.ARTIFACTS["ws_result"]).is_file():
            self.skipTest("optional local Williams archive not downloaded")
        path = self.module.artifact_path("ws_result")
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        body = self.module.history_summary()
        self.assertTrue(body["read_only"])
        self.assertTrue(body["dataset_independent"])
        ws = body["results"]["ws_result"]
        self.assertEqual(ws["status"], "available")
        self.assertEqual(ws["values"]["n_genes"], 3016)
        self.assertAlmostEqual(ws["values"]["ko_spearman"], -0.04560393386)
        self.assertAlmostEqual(ws["values"]["oe_spearman"], 0.07187827359)
        self.assertEqual(body["artifacts"]["ws_result"]["sha256"], before)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
        json.dumps(body, allow_nan=False)

    def test_missing_files_leave_readable_partial_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            body = self.module.history_summary(Path(tmp))
            self.assertTrue(all(a["status"] == "missing" for a in body["artifacts"].values()))
            self.assertTrue(all(r["status"] == "missing" for r in body["results"].values()))

    def test_invalid_result_is_not_reported_as_zero_or_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / self.module.ARTIFACTS["ws_result"]
            path.parent.mkdir(parents=True)
            cases = [dict(genes=np.array(["A", "B"]), shift_ko=[1, 2], shift_oe=[2, 1], real_de=[1, 2]),
                     dict(genes=np.array(["A", "A"]), shift_ko=[1, 2], shift_oe=[2, 1], real_de=[1, 2]),
                     dict(genes=np.array(["A", "B"]), shift_ko=[1, float("nan")], shift_oe=[2, 1], real_de=[1, 2]),
                     dict(genes=np.array(["A", "B"]), shift_ko=[1], shift_oe=[2, 1], real_de=[1, 2]),
                     dict(genes=np.array(["A", "B"], dtype=object), shift_ko=[1, 2], shift_oe=[2, 1], real_de=[1, 2]),
                     dict(genes=np.array(["A", "B"]), shift_ko=[1, 1], shift_oe=[2, 1], real_de=[1, 2])]
            for i, payload in enumerate(cases):
                np.savez(path, **payload)
                result = self.module.history_summary(root)["results"]["ws_result"]
                self.assertEqual(result["status"], "available" if i == 0 else "invalid")
                if i:
                    self.assertNotIn("values", result)
            path.write_bytes(b"broken archive")
            self.assertEqual(self.module.history_summary(root)["results"]["ws_result"]["status"], "invalid")

    def test_bad_json_is_isolated_and_historical_pass_is_not_reinterpreted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / self.module.ARTIFACTS["encoder_result"]
            path.parent.mkdir(parents=True)
            for content in ("{", '{"n_b_cells": NaN}', "[]"):
                path.write_text(content)
                body = self.module.history_summary(root)
                self.assertEqual(body["results"]["encoder_result"]["status"], "invalid")
                self.assertEqual(body["results"]["ws_result"]["status"], "missing")
            path.write_text(json.dumps({"n_b_cells": 2, "n_random_sets": 3,
                                        "AC1_direction": {"marker_away": 0.1, "pass": True},
                                        "AC2_magnitude": {"marker_disp": 0.2, "pass": False}}))
            body = self.module.history_summary(root)
            self.assertTrue(body["results"]["encoder_result"]["values"]["AC1_direction"]["pass"])
            self.assertFalse(body["results"]["encoder_result"]["values"]["AC2_magnitude"]["pass"])

    def test_only_allowlisted_regular_files_inside_repository_are_served(self):
        for key in ("../README.md", "/etc/passwd", "data/ms_expr.npz", "missing"):
            with self.assertRaises(FileNotFoundError):
                self.module.artifact_path(key)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            path = root / self.module.ARTIFACTS["ws_result"]
            path.parent.mkdir(parents=True)
            outside = Path(tmp) / "outside"
            outside.write_text("private")
            path.symlink_to(outside)
            with self.assertRaises(FileNotFoundError):
                self.module.artifact_path("ws_result", root)


if __name__ == "__main__":
    unittest.main()
