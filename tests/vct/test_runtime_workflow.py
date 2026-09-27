"""Offline regressions for launcher identity and online-test assertions."""
import ast
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

ROOT = (Path(__file__).resolve().parents[2] / "vct")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RuntimeWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.launcher = load("web_launcher", ROOT / "src/web_launcher.py")
        self.api = load("online_checks", ROOT.parent / "tests/vct/tests_api.py")
        self.health = dict(service="VirtualCellTool", ready=True, pid=1234, root=str(ROOT.resolve()))

    def test_probe_accepts_only_this_service_and_workspace(self):
        for update in ({}, {"service": "other"}, {"root": "/other"}, {"pid": True}, {"ready": False}):
            body = dict(self.health, **update)
            response = Mock(status=200)
            response.read.return_value = json.dumps(body).encode()
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            with self.subTest(update=update), patch.object(self.launcher, "urlopen", return_value=response):
                if update:
                    with self.assertRaises(RuntimeError):
                        self.launcher.probe(8377, ROOT)
                else:
                    self.assertEqual(self.launcher.probe(8377, ROOT), body)

    def test_probe_rejects_http_errors_and_malformed_body(self):
        for code in (404, 500):
            with patch.object(self.launcher, "urlopen", side_effect=HTTPError("url", code, "error", {}, None)):
                with self.assertRaisesRegex(RuntimeError, "端口"):
                    self.launcher.probe(8377, ROOT)
        with patch.object(self.launcher, "urlopen", return_value=io.BytesIO(b"not JSON")):
            with self.assertRaises(RuntimeError):
                self.launcher.probe(8377, ROOT)

    def test_refused_connection_is_not_ready(self):
        with patch.object(self.launcher, "urlopen", side_effect=URLError(ConnectionRefusedError())):
            self.assertIsNone(self.launcher.probe(8377, ROOT))

    def test_existing_service_reused_without_starting_process(self):
        with patch.object(self.launcher, "probe", return_value=self.health), patch.object(self.launcher.subprocess, "Popen") as start:
            self.assertEqual(self.launcher.launch(8377, ROOT), self.health)
            start.assert_not_called()

    def test_conflict_does_not_start_or_kill_any_process(self):
        with patch.object(self.launcher, "probe", side_effect=RuntimeError("端口冲突")), patch.object(self.launcher.subprocess, "Popen") as start:
            with self.assertRaises(RuntimeError):
                self.launcher.launch(8377, ROOT)
            start.assert_not_called()

    def test_new_process_identity_and_pid_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            health = dict(self.health, root=str(root))
            child = Mock(pid=1234)
            child.poll.return_value = None
            with patch.object(self.launcher, "probe", side_effect=[None, None, health]), patch.object(self.launcher.subprocess, "Popen", return_value=child) as start, patch.object(self.launcher.time, "sleep"):
                self.assertEqual(self.launcher.launch(8399, root), health)
            self.assertEqual(start.call_args.kwargs["env"]["VCT_PORT"], "8399")
            record = json.loads((root.parent / "workspace/vct/runtime/server-8399.json").read_text())
            self.assertEqual(record["pid"], child.pid)
            self.assertEqual(record["port"], 8399)
            child.terminate.assert_not_called()

    def test_failed_or_wrong_process_never_reported_ready(self):
        for state in ("exited", "wrong_pid", "timeout"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as directory:
                child = Mock(pid=99)
                child.poll.return_value = 1 if state == "exited" else None
                responses = [None, self.health] if state == "wrong_pid" else [None]
                with patch.object(self.launcher, "probe", side_effect=responses), patch.object(self.launcher.subprocess, "Popen", return_value=child):
                    with self.assertRaises(RuntimeError):
                        self.launcher.launch(8377, Path(directory), timeout=0 if state == "timeout" else 120)
                self.assertFalse((Path(directory).parent / "workspace/vct/runtime/server-8377.json").exists())
                if state != "exited":
                    child.terminate.assert_called_once()

    def test_port_validation_precedes_startup(self):
        for port in (0, -1, 65536):
            with patch.object(self.launcher, "probe") as probe:
                with self.assertRaises(ValueError):
                    self.launcher.launch(port, ROOT)
                probe.assert_not_called()

    def test_online_http_expected_status_is_enforced(self):
        error = HTTPError("url", 422, "bad", {}, io.BytesIO(b'{"detail":"invalid cell"}'))
        with patch.object(self.api, "urlopen", side_effect=error):
            self.assertIn("detail", self.api.get("/api/attribute", expected_status=422, cell=-1))
        error = HTTPError("url", 500, "bad", {}, io.BytesIO(b'{"error":"crash"}'))
        with patch.object(self.api, "urlopen", side_effect=error):
            with self.assertRaises(AssertionError):
                self.api.get("/api/attribute", expected_status=422, cell=-1)

    def test_online_contract_detects_wrong_source_lengths_and_values(self):
        data = dict(ds_key="ms", dataset="MS", n_cells=3, umap=[[0, 1], [2, 3], [4, 5]], cell_types=["A", "B", "B"])
        expr = dict(dataset="ms", gene="G", values=[1., 2., 3.], mean=2., max=3., by_group=[dict(group="B", n_cells=2)])
        def responses(route, **query):
            return {"/api/data": data, "/api/gene_expr_map": expr,
                    "/api/cell_gene_expr": dict(gene="G", value=expr["values"][query.get("cell", 0)])}[route]
        with patch.object(self.api, "get", side_effect=responses):
            self.api.check_dataset("ms", "G", 3)
            for target, key, bad in ((data, "ds_key", "pbmc"), (data, "umap", [[1, 2]]), (data, "cell_types", ["A"]), (expr, "dataset", "pbmc"), (expr, "values", [1.]), (expr, "values", [1., float("nan"), 3.]), (expr, "mean", 99.)):
                old = copy.deepcopy(target[key])
                target[key] = bad
                with self.subTest(key=key, bad=bad), self.assertRaises(AssertionError):
                    self.api.check_dataset("ms", "G", 3)
                target[key] = old

    def test_e2e_driver_does_not_bypass_user_events_or_hard_sleep(self):
        source = (ROOT.parent / "tests/vct/tests_e2e_playwright.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotEqual(node.func.attr, "wait_for_timeout")
                if node.func.attr in {"evaluate", "wait_for_function"}:
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            self.assertNotRegex(arg.value, r"\b(selectCell|pickGene)\s*\(")
        with patch.dict("os.environ", {"VCT_PORT": "8399"}):
            e2e = load("browser_checks", ROOT.parent / "tests/vct/tests_e2e_playwright.py")
        self.assertEqual(e2e.URL, "http://127.0.0.1:8399/")
        self.assertIn("pg.mouse.click", source)
        self.assertIn("vct_failure.png", source)


if __name__ == "__main__":
    unittest.main()
