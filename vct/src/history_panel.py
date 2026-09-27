"""Small, read-only archive for the web panel; never load models or patient matrices."""
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from scipy.stats import spearmanr

import os

REPO = Path(os.environ.get("VCT_WORKSPACE", Path(__file__).resolve().parents[2] / "workspace/vct"))
VCT = ""
# Fixed public artifacts only, not a general repository file browser.
ARTIFACTS = {
    "ws_result": "history/williams/results.npz",
    "ws_script": "history/williams/celloracle_run.py",
    "ws_report": "history/williams/index.html",
    "tbx_script": "history/22q11/celloracle_run.py",
    "encoder_result": VCT + "data/validation_results.json",
    "ms_figure": VCT + "data/ms_analysis_report.png",
    "ms_table": VCT + "data/ms_differential_attribution.csv",
    "gears_result": VCT + "data/v2_closed_loop.json",
    "gears_figure": VCT + "data/v2_closed_loop_report.png",
    "tier0_result": VCT + "data/tier0_validation.json",
    "method_review": VCT + "history/DESIGN_REVIEW_ITERATIVE.md",
    "cipher_figure": VCT + "outputs/figures/MS4A1_ko_baseline.svg",
    "cipher_result": VCT + "outputs/figures/MS4A1_ko_summary.json",
    "architecture": VCT + "history/ARCHITECTURE_v3.md",
}


def artifact_path(key, repo=REPO):
    root = Path(repo).resolve()
    path = (root / ARTIFACTS.get(key, "")).resolve()
    if key not in ARTIFACTS or not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError("历史材料不存在或不在公开清单中")
    return path


def _ws_summary(path):
    with np.load(path, allow_pickle=False) as archive:
        genes = archive["genes"]
        if (genes.ndim != 1 or len(genes) < 2 or genes.dtype.kind not in "US"
                or len(set(genes)) != len(genes) or any(not str(g).strip() for g in genes)):
            raise ValueError("基因名为空、重复或格式错误")
        vectors = [archive[k] for k in ("shift_ko", "shift_oe", "real_de")]
        for vector in vectors:
            if (vector.shape != genes.shape or vector.dtype.kind not in "fiu"
                    or not np.isfinite(vector).all() or np.ptp(vector) == 0):
                raise ValueError("向量必须等长、有限且非恒定，不能伪造相关系数")
        ko, oe, de = vectors
        return dict(n_genes=len(genes), ko_spearman=float(spearmanr(ko, de).statistic),
                    oe_spearman=float(spearmanr(oe, de).statistic))


def _json_summary(path, key):
    body = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError("历史报告不是对象")
    if key == "encoder_result":
        values = {k: body[k] for k in ("n_b_cells", "n_random_sets", "AC1_direction", "AC2_magnitude")}
        numbers = [values["n_b_cells"], values["n_random_sets"],
                   values["AC1_direction"]["marker_away"], values["AC2_magnitude"]["marker_disp"]]
    elif key == "gears_result":
        values = {k: body[k] for k in ("target", "pearson_all", "n_real_kd_cells")}
        numbers = [values["pearson_all"], values["n_real_kd_cells"]]
    else:
        values = body["metrics"]
        numbers = [values["n_valid_genes"], values["spearman_attr_effect"]["r"],
                   values["spearman_attr_effect"]["p"]]
    if any(isinstance(n, bool) or not isinstance(n, (int, float)) or not np.isfinite(n) for n in numbers):
        raise ValueError("历史指标不是有限数值")
    return values


def history_summary(repo=REPO):
    artifacts, results = {}, {}
    for key, relative in ARTIFACTS.items():
        entry = {"source": relative, "status": "missing"}
        artifacts[key] = entry
        try:
            path = artifact_path(key, repo)
            entry.update(status="available", url=f"/api/history/artifacts/{key}",
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        except FileNotFoundError:
            pass
        except OSError:
            entry["status"] = "unreadable"
    for key in ("ws_result", "encoder_result", "gears_result", "tier0_result"):
        result = {"status": artifacts[key]["status"]}
        results[key] = result
        if result["status"] != "available":
            continue
        try:
            path = artifact_path(key, repo)
            values = _ws_summary(path) if key == "ws_result" else _json_summary(path, key)
            json.dumps(values, allow_nan=False)
            result["values"] = values
        except (OSError, ValueError, TypeError, KeyError, BadZipFile):
            result.update(status="invalid", message="材料读取或校验失败；未用零值代替，请核对原始文件。")
    return dict(read_only=True, dataset_independent=True, artifacts=artifacts, results=results)
