"""Exploration output semantics, without loading real data or models."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = (Path(__file__).resolve().parents[2] / "vct")
SCRIPTS = ("src/validate.py", "src/tier0_validate.py", "analysis/predictive_test.py",
           "analysis/validate_attribution_vs_perturb.py")


class ExplorationReportTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("exploration_report", ROOT / "src/exploration_report.py")
        self.report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.report)

    def test_negative_historical_criterion_is_a_completed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            self.report.write_exploration_report(path, {"AC1_direction": {"pass": False, "p_value": 0.9}})
            body = json.loads(path.read_text())
            self.assertFalse(body["AC1_direction"]["pass"])
            self.assertEqual(body["report_type"], "exploration")
            self.assertEqual(body["completion_status"], "completed")
            self.assertIn("不代表", body["interpretation"])

    def test_invalid_results_do_not_overwrite_previous_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text("previous")
            for results in ({}, {"metric": float("nan")}, {"metric": float("inf")}, {"metric": object()}):
                with self.subTest(results=results), self.assertRaises((ValueError, TypeError)):
                    self.report.write_exploration_report(path, results)
                self.assertEqual(path.read_text(), "previous")

    def test_cli_exit_distinguishes_negative_result_from_output_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            for results, output, success in (("{'pass': False}", Path(tmp) / "ok.json", True),
                                              ("{'r': float('nan')}", Path(tmp) / "bad.json", False),
                                              ("{'pass': False}", Path(tmp) / "missing/r.json", False)):
                code = (f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); "
                        "from exploration_report import write_exploration_report; "
                        f"write_exploration_report({str(output)!r}, {results})")
                run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
                self.assertEqual(run.returncode == 0, success, run.stderr)

    def test_all_four_scripts_identify_exploration_and_publish_report(self):
        for relative in SCRIPTS:
            tree = ast.parse((ROOT / relative).read_text())
            with self.subTest(script=relative):
                self.assertIn("探索", ast.get_docstring(tree))
                calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name)]
                self.assertEqual(sum(node.func.id == "write_exploration_report" for node in calls), 1)
                self.assertTrue(any(node.func.id == "print" and any(isinstance(arg, ast.Name)
                                    and arg.id == "EXPLORATION_NOTICE" for arg in node.args) for node in calls))

    def test_actual_report_calls_with_synthetic_metrics(self):
        # Execute only each script's final report call: no data/model imports or
        # scientific computation. This verifies its real payload/output wiring.
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "data").mkdir()
            namespace = dict(os=os, DATA=str(Path(tmp) / "data"), VCT_ROOT=tmp, DATA_DIR=str(Path(tmp) / "data"),
                             write_exploration_report=self.report.write_exploration_report,
                             res={"AC1_direction": {"pass": False, "p_value": 0.9}},
                             result={"r": -0.3}, df=SimpleNamespace(to_dict=lambda **kw: [{"gene": "A"}]),
                             direction_rows=[{"gene": "A", "cosine": -0.2}], imp=[1, 2],
                             r_attr=-0.2, p_attr=0.8, r_expr=0.1, p_expr=0.9, r_partial=-0.1,
                             results={"A": {"r_abs": -0.3, "rank_de": None}})
            for relative in SCRIPTS:
                tree = ast.parse((ROOT / relative).read_text())
                call = next(node for node in tree.body if isinstance(node, ast.Expr)
                            and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                            and node.value.func.id == "write_exploration_report")
                with self.subTest(script=relative):
                    exec(compile(ast.Module(body=[call], type_ignores=[]), relative, "exec"), namespace)
            reports = list((Path(tmp) / "data").glob("*.json"))
            self.assertEqual(len(reports), 4)
            for path in reports:
                self.assertEqual(json.loads(path.read_text())["report_type"], "exploration")


if __name__ == "__main__":
    unittest.main()
