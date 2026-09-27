"""Offline API contracts and lazy-loading regressions, without app startup.

Compile the actual route/helper functions, replacing only model/data IO. ASGI
requests exercise FastAPI validation without httpx, a server or model downloads.
"""
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Literal
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import scipy.sparse as sp
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

SOURCE = (Path(__file__).resolve().parents[2] / "vct") / "web/app.py"
sys.path.insert(0, str(SOURCE.parents[1] / "src"))
from cipher_engine import CipherEngine
from linear_baseline import LinearBaseline
from history_panel import ARTIFACTS, artifact_path, history_summary


class WebSafetyTests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.ns = dict(app=self.app, Query=Query, HTTPException=HTTPException,
                       JSONResponse=JSONResponse, FileResponse=FileResponse,
                       artifact_path=artifact_path, history_summary=history_summary,
                       Literal=Literal, np=np, pd=pd,
                       sp=sp, os=os, time=time, threading=threading,
                       _inference_lock=threading.Lock(), ROOT=str(SOURCE.parents[1]))
        tree = ast.parse(SOURCE.read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        exec(compile(tree, str(SOURCE), "exec"), self.ns)
        self.genes = ["A", "B", "C", "D"]
        self.x = sp.csr_matrix([[1., 2, 3, 4], [11, 12, 13, 14], [31, 32, 33, 34]])
        self.meta = pd.DataFrame({"cell_type": ["T", "B", "NK"]})
        self.data = dict(X=self.x, emb=np.zeros((3, 2)), meta=self.meta,
                         umap_coords=np.zeros((3, 2)), CANDIDATES=self.genes)
        self.ce = SimpleNamespace(_fitted=True, _gene_idx=dict(zip(self.genes, range(4))),
                                  _gene_list=self.genes, _ctrl_mean=np.array([2., 3, 4, 5]),
                                  predict=Mock(return_value=dict(delta=np.array([2., 0, 0, 0]),
                                                                 delta_top=[("A", 2.)])))
        self.attr = SimpleNamespace(top_genes=Mock(return_value=[("A", 1., 2.)]))
        datasets = {key: dict(emb="emb.npy", expr="expr.npz", meta="meta.csv",
                             umap_npy="umap.npy", title=key) for key in ("pbmc", "ms", "ws")}
        self.ns.update(DATASETS=datasets, DATA="/synthetic-no-files", gene_order=self.genes,
                       perturb_eng=self.ce, attr_eng=self.attr,
                       _DS_CACHE={"pbmc": self.data}, _CIPHER_FITTED={"pbmc"},
                       _CIPHER_ENGINES={"pbmc": self.ce}, _BASELINE_ENGINES={"pbmc": self.ce},
                       _CIPHER_HVG_IDX={}, _DS_LOCKS={key: threading.Lock() for key in datasets})
        self.ns["perturb_eng"] = SimpleNamespace(gene_idx=self.ce._gene_idx,
                                                 perturb=Mock(), perturb_multi=Mock())

    def request(self, path, **params):
        async def run():
            messages = []
            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}
            async def send(message):
                messages.append(message)
            scope = dict(type="http", asgi={"version": "3.0"}, http_version="1.1", method="GET",
                         scheme="http", path="/api/" + path, root_path="",
                         query_string=urlencode(params).encode(), headers=[],
                         client=("127.0.0.1", 1), server=("127.0.0.1", 80))
            await self.app(scope, receive, send)
            status = next(m["status"] for m in messages if m["type"] == "http.response.start")
            body = b"".join(m.get("body", b"") for m in messages)
            return status, json.loads(body)
        return asyncio.run(run())

    def test_health_identity_without_model_inference(self):
        status, body = self.request("health")
        self.assertEqual(status, 200)
        self.assertEqual(body, dict(service="VirtualCellTool", ready=True,
                                    root=str(SOURCE.parents[1].resolve()), pid=os.getpid()))
        self.attr.top_genes.assert_not_called()

    def test_history_is_dataset_independent_read_only_and_allowlisted(self):
        load = Mock(side_effect=AssertionError("history must not load live datasets"))
        with patch.dict(self.ns, {"_load_dataset": load}):
            bodies = [self.request("history", ds=ds) for ds in ("ms", "ws", "unknown")]
        self.assertTrue(all(status == 200 for status, body in bodies))
        self.assertEqual(bodies[0][1], bodies[1][1])
        self.assertEqual(bodies[0][1], bodies[2][1])
        self.assertTrue(bodies[0][1]["read_only"])
        self.assertEqual(self.request("history/artifacts/not-allowed")[0], 404)
        # Exercise real allowlist resolution against synthetic local artifacts.
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp)
            result = archive / ARTIFACTS["gears_result"]
            result.parent.mkdir(parents=True)
            result.write_text(json.dumps({"target": "synthetic-gene"}))
            figure = archive / ARTIFACTS["gears_figure"]
            figure.write_bytes(b"synthetic image header")
            with patch.dict(self.ns, {"artifact_path": lambda key: artifact_path(key, archive)}):
                self.assertEqual(self.request("history/artifacts/gears_result")[1]["target"], "synthetic-gene")
                response = self.ns["history_artifact"]("gears_figure")
                self.assertEqual(response.media_type, "image/png")
                self.assertEqual(response.headers["x-content-type-options"], "nosniff")
                self.assertEqual(self.request("history/artifacts/not-allowed")[0], 404)
            report_path = archive / ARTIFACTS["ws_report"]
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("<p>Archive fixture</p>")
            with patch.dict(self.ns, {"artifact_path": lambda key: artifact_path(key, archive)}):
                report = self.ns["history_artifact"]("ws_report")
                self.assertEqual(report.media_type, "text/html")
                self.assertEqual(report.headers["x-content-type-options"], "nosniff")
                self.assertIn("Archive fixture", Path(report.path).read_text())
        load.assert_not_called()
        self.attr.top_genes.assert_not_called()
        self.ns["perturb_eng"].perturb.assert_not_called()

    def test_online_contract_checker_against_actual_asgi_routes(self):
        spec = importlib.util.spec_from_file_location("online_checks", SOURCE.parents[2] / "tests/vct/tests_api.py")
        checks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checks)
        def get(route, *, expected_status=200, **query):
            status, body = self.request(route.removeprefix("/api/"), **query)
            self.assertEqual(status, expected_status)
            return body
        with patch.object(checks, "get", side_effect=get):
            checks.check_dataset("pbmc", "A", 3)
        for route, query, status in [
            ("data", {"ds": "unknown"}, 400),
            ("attribute", {"cell": -1}, 422),
            ("perturb_cipher", {"gene": "A", "direction": "invalid"}, 422),
            ("perturb_baseline", {"gene": "A", "method": "invalid"}, 422),
        ]:
            self.assertEqual(self.request(route, **query)[0], status)

    def test_invalid_cells_rejected_before_inference(self):
        routes = ("attribute", "attribution_map", "cell_gene_expr", "perturb", "perturb_multi")
        for route in routes:
            for cell in (-1, 3, 100000):
                with self.subTest(route=route, cell=cell):
                    status, body = self.request(route, ds="pbmc", cell=cell, gene="A", genes="A", value=0)
                    self.assertEqual(status, 422)
                    self.assertIn("cell", str(body))
        self.attr.top_genes.assert_not_called()
        self.ns["perturb_eng"].perturb.assert_not_called()
        self.ns["perturb_eng"].perturb_multi.assert_not_called()

    def test_valid_cell_boundaries_and_default_dataset(self):
        for cell, expected in ((0, 1.), (2, 31.)):
            status, body = self.request("cell_gene_expr", cell=cell, gene="A")
            self.assertEqual((status, body["value"]), (200, expected))
        self.assertEqual(self.request("data")[1]["ds_key"], "pbmc")

    def test_result_limits_rejected_before_load_or_inference(self):
        load = Mock(side_effect=AssertionError("invalid input loaded data"))
        with patch.dict(self.ns, {"_load_dataset": load}):
            for route, key, high in (("attribute", "k", 100), ("attribution_map", "k", 100),
                                     ("search_genes", "limit", 200)):
                for value in (-1, 0, high + 1):
                    with self.subTest(route=route, value=value):
                        status, body = self.request(route, cell=0, q="", **{key: value})
                        self.assertEqual(status, 422)
                        self.assertIn(key, str(body))
        load.assert_not_called()
        self.attr.top_genes.assert_not_called()

    def test_valid_result_limit_boundaries(self):
        for route in ("attribute", "attribution_map"):
            for k in (1, 100):
                self.assertEqual(self.request(route, cell=0, k=k)[0], 200)
                self.assertEqual(self.attr.top_genes.call_args.kwargs["k"], k)
        for limit in (1, 200):
            status, body = self.request("search_genes", q="", limit=limit)
            self.assertEqual(status, 200)
            self.assertEqual(len(body["matches"]), min(limit, 4))

    def test_unknown_dataset_rejected_consistently(self):
        routes = ("data", "attribute", "attribution_map", "perturb", "perturb_multi",
                  "cell_gene_expr", "gene_expr", "gene_expr_map", "cipher_status",
                  "gene_in_cipher", "perturb_cipher", "perturb_baseline", "perturb_compare",
                  "celltype_response", "perturb_summary")
        with patch.dict(self.ns, {"_read_dataset": Mock(side_effect=AssertionError("read"))}):
            for route in routes:
                with self.subTest(route=route):
                    status, body = self.request(route, ds="bogus", cell=0, gene="A", genes="A", value=0)
                    self.assertEqual(status, 400)
                    self.assertIn("bogus", str(body))

    def test_invalid_modes_rejected_before_loading(self):
        load = Mock(side_effect=AssertionError("load"))
        with patch.dict(self.ns, {"_load_dataset": load}):
            for route in ("perturb_cipher", "perturb_compare", "celltype_response", "perturb_summary"):
                with self.subTest(route=route):
                    status, body = self.request(route, gene="A", direction="typo")
                    self.assertEqual(status, 422)
                    self.assertIn("direction", str(body))
            status, body = self.request("perturb_baseline", gene="A", method="typo")
            self.assertEqual(status, 422)
            self.assertIn("method", str(body))
        load.assert_not_called()

    def test_valid_modes_keep_target_values(self):
        for direction, target in (("ko", 0.), ("oe", 4.)):
            for route in ("perturb_cipher", "perturb_compare", "celltype_response", "perturb_summary"):
                with self.subTest(route=route, direction=direction), \
                        patch.dict(self.ns, {"_celltype_rows": lambda *args: []}):
                    self.assertEqual(self.request(route, gene="A", direction=direction)[0], 200)
                    self.ce.predict.assert_called_with({"A": target})
        self.assertEqual(self.request("perturb_cipher", gene="A", direction="oe", target_value=7)[0], 200)
        self.ce.predict.assert_called_with({"A": 7.})
        self.ce.predict_ctrl_mean = Mock(return_value=dict(method="ctrl_mean", description="test"))
        self.ce.predict_additive = Mock(return_value=dict(method="additive", description="test",
                                                         delta=np.zeros(4), delta_top=[]))
        for method, expected in (("ctrl_mean", {"ctrl_mean"}), ("additive", {"additive"}),
                                 ("both", {"ctrl_mean", "additive"})):
            status, body = self.request("perturb_baseline", gene="A", method=method)
            self.assertEqual(status, 200)
            self.assertEqual(set(body["results"]), expected)

    def mock_reads(self):
        self.enterContext(patch.dict(sys.modules, {"umap": SimpleNamespace()}))
        self.enterContext(patch.object(np, "load", return_value=np.zeros((3, 2))))
        load = self.enterContext(patch.object(sp, "load_npz", return_value=self.x))
        self.enterContext(patch.object(pd, "read_csv", return_value=self.meta))
        self.enterContext(patch("os.path.exists", return_value=True))
        return load

    def test_failed_initialization_can_retry_without_partial_state(self):
        self.mock_reads()
        fit = Mock(side_effect=[RuntimeError("transient"), (self.ce, self.ce, np.arange(4))])
        with patch.dict(self.ns, {"_fit_cipher_cached": fit}):
            ce, error = self.ns["_require_cipher"]("ms")
            self.assertIsNone(ce)
            self.assertEqual(error.status_code, 500)
            for key in ("_DS_CACHE", "_CIPHER_FITTED", "_CIPHER_ENGINES", "_BASELINE_ENGINES", "_CIPHER_HVG_IDX"):
                self.assertNotIn("ms", self.ns[key])
            ce, error = self.ns["_require_cipher"]("ms")
            self.assertIsNone(error)
            self.assertIs(ce, self.ce)
            self.assertEqual(fit.call_count, 2)

    def test_concurrent_first_load_is_once_and_other_datasets_stay_intact(self):
        load = self.mock_reads()
        def slow_load(*args):
            time.sleep(.03)  # Give simultaneous callers time to enter initialization.
            return self.x
        load.side_effect = slow_load
        fit = Mock(return_value=(self.ce, self.ce, np.arange(4)))
        with patch.dict(self.ns, {"_fit_cipher_cached": fit}), ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(self.ns["_load_dataset"], ["ms"] * 6))
        self.assertEqual(load.call_count, 1)
        self.assertEqual(fit.call_count, 1)
        self.assertTrue(all(result is results[0] for result in results))
        self.assertIs(self.ns["_DS_CACHE"]["pbmc"], self.data)
        self.assertIn("ms", self.ns["_CIPHER_FITTED"])

    def test_different_datasets_can_initialize_independently(self):
        self.mock_reads()
        gate = threading.Barrier(2)
        def fit(*args):
            gate.wait(timeout=3)
            return self.ce, self.ce, np.arange(4)
        with patch.dict(self.ns, {"_fit_cipher_cached": fit}), ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(self.ns["_load_dataset"], ["ms", "ws"]))
        self.assertTrue({"ms", "ws"} <= self.ns["_CIPHER_FITTED"])

    def test_real_small_fit_and_disk_cache_return_same_predictions(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
                self.ns, DATA=tmp, CipherEngine=CipherEngine, LinearBaseline=LinearBaseline):
            first = self.ns["_fit_cipher_cached"]("ms", self.x, self.genes)
            with patch.object(CipherEngine, "fit", side_effect=AssertionError("refit")):
                cached = self.ns["_fit_cipher_cached"]("ms", self.x, self.genes)
            for before, after in zip(first[:2], cached[:2]):
                np.testing.assert_array_equal(before._ctrl_mean, after._ctrl_mean)
            np.testing.assert_array_equal(first[0].predict({"A": 0})["delta"],
                                          cached[0].predict({"A": 0})["delta"])
            np.testing.assert_array_equal(first[2], cached[2])
            # Fitting alone must not publish half of a dataset's runtime state.
            self.assertNotIn("ms", self.ns["_CIPHER_FITTED"])

    def test_disk_write_failure_still_publishes_complete_runtime_state(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
                self.ns, DATA=tmp, CipherEngine=CipherEngine, LinearBaseline=LinearBaseline,
                _read_dataset=lambda key: {**self.data, "gene_var": None}), \
                patch.object(np, "savez_compressed", side_effect=OSError("read-only cache")):
            loaded = self.ns["_load_dataset"]("ms")
            self.assertIs(loaded, self.ns["_DS_CACHE"]["ms"])
            self.assertTrue(self.ns["_CIPHER_ENGINES"]["ms"]._fitted)
            self.assertTrue(self.ns["_BASELINE_ENGINES"]["ms"]._fitted)
            self.assertIn("ms", self.ns["_CIPHER_FITTED"])


if __name__ == "__main__":
    unittest.main()
