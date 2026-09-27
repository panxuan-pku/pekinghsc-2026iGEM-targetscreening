"""VirtualCellTool v3.3 — 虚拟细胞交互式网页 demo

FastAPI 后端，单服务承载 3 个数据集（?ds=pbmc|ms|ws，懒加载 + 进程内缓存）。

端点分组：
  数据集/浏览   /api/data  /api/search_genes  /api/gene_expr
  归因(v1)      /api/attribute  /api/attribution_map
  反事实(v1)    /api/perturb  /api/perturb_multi  /api/cell_gene_expr
  因果(v2)      /api/gene_coverage  /api/perturb_causal        ← GEARS，负对照
  线性响应(v3)  /api/cipher_status  /api/gene_in_cipher
                /api/perturb_cipher  /api/perturb_summary
                /api/celltype_response
  基线对照(v3)  /api/perturb_baseline  /api/perturb_compare
  表达分布(v3.3)/api/gene_expr_map

版本锚点（git tag）：v1.0-scimilarity · v2.0-gears · v3.1-fixed · v3.2-zoom · v3.3-viz
自有工作（见 ATTRIBUTIONS.md）；交接说明见 docs/HANDOFF_v3.md
"""
import os, sys, time
import threading
from typing import Literal
_MPL_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "workspace", "vct", "cache", "matplotlib")
os.makedirs(_MPL_CACHE, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPL_CACHE)
import numpy as np
import pandas as pd
import scipy.sparse as sp
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from paths import DATA_DIR
from perturb import PerturbEngine
from attribute import AttributionEngine
from gears_engine import get_gears_engine
from cipher_engine import CipherEngine, get_cipher_engine
from linear_baseline import LinearBaseline, get_baseline
from history_panel import artifact_path, history_summary

DATA = DATA_DIR
# group_label：各数据集 meta.cell_type 列的真实语义。
# ws/ms 的该列装的是「病/健分组」(WS vs CTRL) 而非细胞类型，
# 前端若一律写“细胞类型响应”会误导读者，故在此显式声明。
DATASETS = {
    "pbmc": {"emb": "embeddings.npy", "expr": "expr_aligned.npz", "meta": "meta.csv",
              "umap_npy": "umap.npy", "title": "PBMC 2700", "group_label": "细胞类型"},
    "ms":   {"emb": "ms_embeddings.npy", "expr": "ms_expr.npz", "meta": "ms_web_meta.csv",
              "umap_npy": "ms_umap.npy", "title": "MS 少突胶质细胞", "group_label": "病健分组"},
    "ws":   {"emb": "ws_embeddings.npy", "expr": "ws_expr.npz", "meta": "ws_web_meta.csv",
              "umap_npy": "ws_umap.npy", "title": "Williams 脑类器官", "group_label": "病健分组"},
}

app = FastAPI(title="VirtualCellTool")

# 推理锁：PyTorch 模型在 CPU 上并发推理可能导致线程爆炸或结果竞争
_inference_lock = threading.Lock()
from fastapi.staticfiles import StaticFiles
app.mount("/web", StaticFiles(directory=os.path.join(ROOT, "web")), name="web")

# 引擎与模型是数据集无关的（SCimilarity 全局共享），只加载一次
print("加载 SCimilarity 引擎 ...")
perturb_eng = PerturbEngine()
attr_eng = AttributionEngine()
gene_order = perturb_eng.wrapper.gene_order

# v3: CIPHER + 线性基线引擎（按数据集独立实例——每种数据有自己的协方差矩阵，
# 绝不能共用一个全局单例，否则切数据集时后加载的会覆盖先加载的，结果错乱）
print("初始化 CIPHER + 线性基线引擎（按数据集惰性拟合）...")
_CIPHER_ENGINES = {}   # ds_key -> CipherEngine
_BASELINE_ENGINES = {} # ds_key -> LinearBaseline
_CIPHER_FITTED = set() # 已拟合的数据集 key 集合
_CIPHER_HVG_IDX = {}   # ds_key → 全基因索引中 HVG 的位置数组


def _cipher_for(ds: str):
    """取该数据集的 CIPHER 引擎（未拟合则 None）。"""
    return _CIPHER_ENGINES.get(ds)


def _baseline_for(ds: str):
    return _BASELINE_ENGINES.get(ds)

# ---------- 数据集懒加载缓存 ----------
_DS_CACHE = {}   # ds_key -> dict(emb, X, meta, umap_coords, centroids, CANDIDATES)
_DS_LOCKS = {key: threading.Lock() for key in DATASETS}


def _validate_dataset(key: str):
    if key not in DATASETS:
        raise HTTPException(status_code=400, detail=f"未知数据集 {key}；可选：{', '.join(DATASETS)}")


def _load_dataset(key: str) -> dict:
    _validate_dataset(key)
    # 同一数据集只初始化一次；失败不发布状态，下次请求可重新尝试。
    with _DS_LOCKS[key]:
        if key not in _DS_CACHE:
            d = _read_dataset(key)
            ce, bl, hvg_idx = _fit_cipher_cached(key, d["X"], gene_order, d["gene_var"])
            _CIPHER_ENGINES[key] = ce
            _BASELINE_ENGINES[key] = bl
            _CIPHER_HVG_IDX[key] = hvg_idx
            _DS_CACHE[key] = d
            _CIPHER_FITTED.add(key)
            print(f"[dataset] {DATASETS[key]['title']} 数据与模型就绪")
        return _DS_CACHE[key]


def _read_dataset(key: str) -> dict:
    f = DATASETS[key]
    print(f"[dataset] 加载 {f['title']} ...")
    t0 = time.time()
    emb = np.load(os.path.join(DATA, f["emb"]))
    # 关键：保持 CSR 稀疏，不再 .toarray()。
    # ws 数据集稠密化 = 96969×28231 float64 ≈ 21.9GB，必然打爆内存/进入 swap。
    X = sp.load_npz(os.path.join(DATA, f["expr"])).tocsr().astype(np.float32)
    meta = pd.read_csv(os.path.join(DATA, f["meta"]))
    if key == "pbmc":
        import json
        info_path = os.path.join(DATA, "pbmc_info.json")
        if os.path.isfile(info_path):
            with open(info_path, encoding="utf-8") as info_file:
                info = json.load(info_file)
            if info.get("label_policy", "").startswith("computed groups"):
                f["group_label"] = "计算分组（未经细胞类型注释）"
    import umap
    # 优先加载缓存的 UMAP，否则重新计算
    umap_cache = os.path.join(DATA, f.get("umap_npy", ""))
    if os.path.exists(umap_cache):
        print(f"[dataset] 加载缓存的 UMAP: {umap_cache}")
        umap_coords = np.load(umap_cache)
        # UMAP 坐标反查：k 近邻距离加权插值（不是 k=1 最近邻！）
        # k=1 会把坐标「吸附」到某个已有细胞上——只要扰动后的嵌入最近邻没换人，
        # 输出坐标就和扰动前完全相同，UMAP 上永远看不到位移（实测 0.3 量级的
        # 位移仍然 Δ=0）。改成 kNN 反距离加权后坐标连续可微，小扰动也能看出移动。
        class _FakeReducer:
            def __init__(self, coords, emb_ref, k=15, power=2.0):
                self._coords = coords
                self._emb = emb_ref
                self._k = min(k, len(emb_ref))
                self._power = power
            def _tree_(self):
                from scipy.spatial import cKDTree
                if not hasattr(self, "_tree_obj"):
                    self._tree_obj = cKDTree(self._emb)
                return self._tree_obj
            def transform(self, X):
                single = (X.ndim == 1)
                Q = X.reshape(1, -1) if single else X
                dist, idx = self._tree_().query(Q, k=self._k)
                dist = np.atleast_2d(dist); idx = np.atleast_2d(idx)
                out = np.empty((Q.shape[0], self._coords.shape[1]), dtype=float)
                for r in range(Q.shape[0]):
                    d, i = dist[r], idx[r]
                    if d[0] < 1e-12:          # 命中已有细胞本身
                        out[r] = self._coords[i[0]]
                        continue
                    w = 1.0 / np.power(d, self._power)
                    w /= w.sum()
                    out[r] = (self._coords[i] * w[:, None]).sum(axis=0)
                return out[0] if single else out
            @property
            def embedding_(self): return self._coords
        reducer = _FakeReducer(umap_coords, emb)
    else:
        # ws 用 v2 紧致参数（与 prepare_ws_umap_v2 一致），其余用默认
        reducer = (umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
                   if key == "ws" else
                   umap.UMAP(n_neighbors=15, min_dist=0.5, random_state=42)).fit(emb)
        umap_coords = reducer.embedding_
        np.save(umap_cache, umap_coords)
        print(f"[dataset] UMAP 已缓存: {umap_cache}")
    centroids = {ct: emb[(meta.cell_type == ct).values].mean(axis=0)
                 for ct in meta.cell_type.unique()}
    # 稀疏矩阵的逐列方差：E[x²] - E[x]²（不稠密化）
    gene_var = _sparse_col_var(X)
    candidate_idx = np.argsort(-gene_var)[:2000]
    CANDIDATES = sorted([gene_order[i] for i in candidate_idx])
    d = {"emb": emb, "X": X, "meta": meta, "reducer": reducer,
         "umap_coords": umap_coords, "centroids": centroids,
         "CANDIDATES": CANDIDATES, "gene_var": gene_var}
    print(f"[dataset] {f['title']} 数据读取完成 ({time.time()-t0:.0f}s, {emb.shape[0]} 细胞)")
    return d


def _sparse_col_var(X) -> np.ndarray:
    """CSR 稀疏矩阵的逐列方差，不稠密化。"""
    n = X.shape[0]
    mean = np.asarray(X.mean(axis=0)).ravel()
    mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
    return np.maximum(mean_sq - mean ** 2, 0.0) * (n / max(n - 1, 1))


def _row(X, i) -> np.ndarray:
    """取第 i 个细胞的稠密表达向量（28231,）。"""
    if not 0 <= i < X.shape[0]:
        raise HTTPException(status_code=422, detail=f"cell 必须在 0～{X.shape[0] - 1} 之间")
    if sp.issparse(X):
        return np.asarray(X[i].todense()).ravel().astype(np.float32)
    return X[i]

def _get_ds(ds: str) -> dict:
    return _load_dataset(ds)


def _fit_cipher_cached(key: str, X, gene_order, gene_var=None):
    """用 Top 5000 高变基因拟合 CIPHER，缓存协方差矩阵到磁盘。
    避免每次启动都算 28231×28231 协方差（~3GB，~2-3 分钟）。

    X 可以是 CSR 稀疏矩阵（ws 数据集稠密化会占 22GB，必须保持稀疏）。
    """
    cache_path = os.path.join(DATA, f"cipher_{key}.npz")
    ce = CipherEngine()
    bl = LinearBaseline()

    if os.path.exists(cache_path):
        print(f"[cipher] 加载缓存 {cache_path} ...")
        try:
            c = np.load(cache_path, allow_pickle=True)
            if "hvg_idx" not in c:
                raise KeyError("缓存缺少 hvg_idx（旧版格式），重新拟合")
            ce._gene_list = [str(g) for g in c["gene_list"]]
            ce._gene_idx = {g: i for i, g in enumerate(ce._gene_list)}
            ce._ctrl_mean = c["ctrl_mean"]
            ce._Sigma = c["Sigma"]
            ce._fitted = True
            hvg_idx = np.asarray(c["hvg_idx"])
            bl.fit_stats(np.asarray(c["full_mean"]), list(gene_order))
            print(f"[cipher] 缓存加载完成: {len(ce._gene_list)} 基因")
            return ce, bl, hvg_idx
        except Exception as e:
            print(f"[cipher] 缓存加载失败: {e}，重新拟合")

    # 选 Top 5000 高变基因（方差最大）
    n_hvg = min(5000, X.shape[1])
    if gene_var is None:
        gene_var = _sparse_col_var(X) if sp.issparse(X) else X.var(axis=0)
    hvg_idx = np.sort(np.argsort(-gene_var)[:n_hvg])   # 排序保证列切片高效且可复现
    X_hvg = X[:, hvg_idx]
    if sp.issparse(X_hvg):
        X_hvg = np.asarray(X_hvg.todense(), dtype=np.float32)   # 5000 列稠密化仅 ~2GB→ws 也可接受
    genes_hvg = [gene_order[i] for i in hvg_idx]
    print(f"[cipher] 拟合 {key} ({X_hvg.shape[0]} 细胞 × {n_hvg} HVG)...")
    t0 = time.time()
    ce.fit(X_hvg, genes_hvg)
    full_mean = (np.asarray(X.mean(axis=0)).ravel() if sp.issparse(X) else X.mean(axis=0))
    bl.fit_stats(full_mean, list(gene_order))
    print(f"[cipher] 拟合完成 {time.time()-t0:.0f}s")
    del X_hvg

    # 保存缓存（★ 必须含 hvg_idx + full_mean，否则下次加载又会全量重算）
    try:
        np.savez_compressed(cache_path,
                            gene_list=np.array(ce._gene_list, dtype=object),
                            ctrl_mean=ce._ctrl_mean,
                            Sigma=ce._Sigma,
                            hvg_idx=hvg_idx,
                            full_mean=full_mean)
        print(f"[cipher] 缓存已保存: {cache_path}")
    except Exception as e:
        print(f"[cipher] 缓存保存失败: {e}")
    return ce, bl, hvg_idx

# 启动时预载默认数据集（ws），其余懒加载
# 默认数据集：pbmc（2700 细胞，秒级就绪）。ws/ms 懒加载，用户切换时才加载。
# 旧版默认 ws（96969 细胞）导致启动要几分钟且页面一直白屏。
_DEFAULT = os.environ.get("VCT_DATASET", "pbmc")
_D = _load_dataset(_DEFAULT if _DEFAULT in DATASETS else "pbmc")

def _zoom_window(old_xy, new_xy, pad_ratio: float = 6.0, min_span: float = 0.35):
    """算出一个刚好框住「扰动前→后」两点的局部视窗，供前端放大显示。

    单基因扰动在全图尺度上只有零点几个 UMAP 单位，全景视图里根本看不出来；
    返回一个以位移中点为中心、按位移长度自适应的方框，前端据此做局部放大。
    """
    o = np.asarray(old_xy, dtype=float)
    n = np.asarray(new_xy, dtype=float)
    cx, cy = (o + n) / 2.0
    span = max(float(np.linalg.norm(n - o)) * pad_ratio, min_span)
    return {
        "x": [round(cx - span, 4), round(cx + span, 4)],
        "y": [round(cy - span, 4), round(cy + span, 4)],
        "span": round(span, 4),
    }


def _local_neighbors(d: dict, old_xy, new_xy, radius_mult: float = 6.0, max_pts: int = 900):
    """取局部视窗内的背景细胞（坐标+类型），供前端在放大视图里画上下文。

    放大后如果只有孤零零两个点，用户无法判断细胞移动是否跨越了群体边界，
    所以要把邻域内的真实细胞一起送过去当参照系。
    """
    coords = d["umap_coords"]
    o = np.asarray(old_xy, dtype=float)
    n = np.asarray(new_xy, dtype=float)
    c = (o + n) / 2.0
    r = max(float(np.linalg.norm(n - o)) * radius_mult, 0.35)
    m = (np.abs(coords[:, 0] - c[0]) <= r) & (np.abs(coords[:, 1] - c[1]) <= r)
    idx = np.flatnonzero(m)
    if idx.size > max_pts:                      # 太密就均匀抽样，控制传输量
        idx = idx[np.linspace(0, idx.size - 1, max_pts).astype(int)]
    types = d["meta"]["cell_type"].to_numpy()
    return {
        "x": [round(float(v), 4) for v in coords[idx, 0]],
        "y": [round(float(v), 4) for v in coords[idx, 1]],
        "t": [str(types[i]) for i in idx],
    }


def classify(d: dict, e: np.ndarray) -> tuple:
    dd = {ct: float(np.linalg.norm(e - c)) for ct, c in d["centroids"].items()}
    return min(dd, key=dd.get), dd

print(f"就绪: 默认数据集 {_D['emb'].shape[0]} 细胞（其余数据集懒加载）")

@app.get("/api/history")
def history_archive():
    """Dataset-independent saved evidence; no inference or historical rerun."""
    return history_summary()


@app.get("/api/history/artifacts/{key}")
def history_artifact(key: str):
    try:
        path = artifact_path(key)
    except (OSError, ValueError):
        raise HTTPException(status_code=404, detail="历史材料不存在或不在公开清单中")
    media = {".png": "image/png", ".svg": "image/svg+xml", ".json": "application/json",
             ".md": "text/plain", ".py": "text/plain", ".csv": "text/plain",
             ".html": "text/html"}.get(path.suffix)
    return FileResponse(path, media_type=media or "application/octet-stream",
                        headers={"X-Content-Type-Options": "nosniff"})


@app.get("/api/health")
def health():
    # Served only after startup has loaded the default dataset/model.
    return {"service": "VirtualCellTool", "ready": True,
            "root": os.path.realpath(ROOT), "pid": os.getpid()}


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "web", "index.html"),
                        headers={"Cache-Control": "no-store"})

@app.get("/api/data")
def data(ds: str = Query("pbmc")):
    d = _get_ds(ds)
    resp = {
        "n_cells": int(d["emb"].shape[0]),
        "umap": d["umap_coords"].round(3).tolist(),
        "cell_types": d["meta"].cell_type.tolist(),
        "genes": d["CANDIDATES"],
        "dataset": DATASETS[ds]["title"],
        "group_label": DATASETS[ds].get("group_label", "细胞类型"),
        "ds_key": ds,
    }
    if "cluster" in d["meta"].columns:
        resp["clusters"] = d["meta"].cluster.astype(str).tolist()
    return resp

@app.get("/api/attribute")
def attribute(ds: str = Query("pbmc"), cell: int = Query(..., ge=0), k: int = Query(10, ge=1, le=100)):
    d = _get_ds(ds)
    t0 = time.time()
    with _inference_lock:
        top = attr_eng.top_genes(_row(d["X"], cell), k=k)
    return {"cell": cell, "cell_type": d["meta"].cell_type.iloc[cell],
            "top_genes": [{"gene": g, "attribution": round(a, 4), "expr": round(e, 3)} for g, a, e in top],
            "elapsed": round(time.time() - t0, 2)}

@app.get("/api/perturb")
def perturb(ds: str = Query("pbmc"), cell: int = Query(..., ge=0), gene: str = Query(...), value: float = Query(...)):
    d = _get_ds(ds)
    if gene not in perturb_eng.gene_idx:
        return JSONResponse({"error": f"基因 '{gene}' 不在模型基因空间内"}, status_code=400)
    t0 = time.time()
    with _inference_lock:
        r = perturb_eng.perturb(_row(d["X"], cell), gene, value)
    old_xy = d["reducer"].transform(r["emb_old"][None, :])[0]
    new_xy = d["reducer"].transform(r["emb_new"][None, :])[0]
    # UMAP 坐标由最近邻查表得到（见 _FakeReducer）：单基因扰动的嵌入位移通常
    # 只有 ~0.01，最近邻往往仍是同一个细胞 → 两个坐标完全相同，箭头长度为 0。
    # 明确告诉前端「UMAP 上不可见」，避免画一个零长度箭头误导用户。
    umap_moved = bool(np.linalg.norm(np.asarray(old_xy) - np.asarray(new_xy)) > 1e-9)
    return {
        "gene": gene, "old_value": round(r["old_value"], 3), "new_value": value,
        "old_xy": [round(float(v), 4) for v in old_xy],
        "new_xy": [round(float(v), 4) for v in new_xy],
        "displacement": round(r["displacement"], 4),
        "umap_moved": umap_moved,
        "umap_shift": round(float(np.linalg.norm(np.asarray(old_xy) - np.asarray(new_xy))), 4),
        "zoom": _zoom_window(old_xy, new_xy),
        "neighbors": _local_neighbors(d, old_xy, new_xy),
        "elapsed": round(time.time() - t0, 2),
    }

@app.get("/api/perturb_multi")
def perturb_multi(ds: str = Query("pbmc"), cell: int = Query(..., ge=0), genes: str = Query(...), value: float = Query(0.0)):
    """多基因联合扰动：genes 为逗号分隔的基因列表，统一设为 value（默认0=敲低）。
    返回扰动前后最近质心分类（细胞类型重判）。"""
    d = _get_ds(ds)
    t0 = time.time()
    gene_list = [g.strip() for g in genes.split(",") if g.strip()]
    with _inference_lock:
        r = perturb_eng.perturb_multi(_row(d["X"], cell), {g: value for g in gene_list})
    old_xy = d["reducer"].transform(r["emb_old"][None, :])[0]
    new_xy = d["reducer"].transform(r["emb_new"][None, :])[0]
    old_type, old_dist = classify(d, r["emb_old"])
    new_type, new_dist = classify(d, r["emb_new"])
    umap_moved = bool(np.linalg.norm(np.asarray(old_xy) - np.asarray(new_xy)) > 1e-9)
    return {
        "genes": gene_list, "value": value,
        "umap_moved": umap_moved,
        "umap_shift": round(float(np.linalg.norm(np.asarray(old_xy) - np.asarray(new_xy))), 4),
        "zoom": _zoom_window(old_xy, new_xy),
        "neighbors": _local_neighbors(d, old_xy, new_xy),
        "old_xy": [round(float(v), 4) for v in old_xy],
        "new_xy": [round(float(v), 4) for v in new_xy],
        "displacement": round(r["displacement"], 4),
        "declared_type": d["meta"].cell_type.iloc[cell],
        "old_nearest": old_type, "new_nearest": new_type,
        "changed": old_type != new_type,
        "old_dist": {k: round(v, 3) for k, v in old_dist.items()},
        "new_dist": {k: round(v, 3) for k, v in new_dist.items()},
        "elapsed": round(time.time() - t0, 2),
    }

@app.get("/api/search_genes")
def search_genes(q: str = Query(...), limit: int = Query(50, ge=1, le=200)):
    """在全部 28231 个基因空间里搜索（不限于候选列表）"""
    qu = q.upper()
    starts = [g for g in gene_order if g.upper().startswith(qu)]
    contains = [g for g in gene_order if qu in g.upper() and not g.upper().startswith(qu)]
    return {"matches": (starts + contains)[:limit]}

@app.get("/api/cell_gene_expr")
def cell_gene_expr(ds: str = Query("pbmc"), cell: int = Query(..., ge=0), gene: str = Query(...)):
    d = _get_ds(ds)
    if gene not in perturb_eng.gene_idx:
        return JSONResponse({"error": "gene not found"}, status_code=404)
    return {"gene": gene, "value": round(float(_row(d["X"], cell)[perturb_eng.gene_idx[gene]]), 2)}

@app.get("/api/gene_expr")
def gene_expr(ds: str = Query("pbmc"), gene: str = Query(...)):
    d = _get_ds(ds)
    if gene not in perturb_eng.gene_idx:
        return JSONResponse({"error": "gene not found"}, status_code=404)
    col = d["X"][:, perturb_eng.gene_idx[gene]]
    col = np.asarray(col.todense()).ravel() if sp.issparse(col) else np.asarray(col).ravel()
    return {"gene": gene, "max": round(float(col.max()), 2), "mean": round(float(col.mean()), 3)}

# ---------- v2: GEARS 因果模式 ----------
@app.get("/api/gene_coverage")
def gene_coverage(gene: str = Query(...)):
    """查询基因是否在 GEARS 训练图内（决定前端因果按钮是否可用）。覆盖表走缓存，秒级。"""
    try:
        eng = get_gears_engine()
        if not eng._pert_genes:
            eng._ensure_loaded()
        return {"gene": gene, "covered": eng.gene_covered(gene),
                "engine": "gears-causal (Norman Perturb-seq)"}
    except Exception as e:
        return JSONResponse({"error": f"GEARS 引擎加载失败: {str(e)[:120]}"}, status_code=500)

@app.get("/api/perturb_causal")
def perturb_causal(gene: str = Query(...)):
    """GEARS 因果预测：单基因敲低的全转录组预测（分钟级）。仅限训练图内基因。"""
    try:
        eng = get_gears_engine()
        r = eng.predict_single(gene)
        if "error" in r:
            return JSONResponse(r, status_code=400)
        return r
    except Exception as e:
        return JSONResponse({"error": f"因果预测失败: {str(e)[:120]}"}, status_code=500)

# ---------- v3: CIPHER 线性响应 + 线性基线 ----------
# 说明：CIPHER / 基线引擎按数据集独立实例（_CIPHER_ENGINES[ds]），
# 端点统一用 _require_cipher(ds) 取引擎——它会在数据集尚未加载时自动触发加载，
# 避免旧版"必须先手动访问 /api/data 才能用 CIPHER"的 503 陷阱。


def _require_cipher(ds: str):
    """取（必要时加载）该数据集的 CIPHER 引擎。返回 (engine, error_response)。"""
    _validate_dataset(ds)
    if ds not in _CIPHER_FITTED:
        try:
            _load_dataset(ds)     # 懒加载 + 拟合（进程内只发生一次）
        except Exception as e:
            return None, JSONResponse(
                {"error": f"数据集 {ds} 加载失败: {str(e)[:160]}"}, status_code=500)
    ce = _CIPHER_ENGINES.get(ds)
    if ce is None or not ce._fitted:
        return None, JSONResponse({"error": f"CIPHER 在数据集 {ds} 上未拟合"}, status_code=503)
    return ce, None


def _gene_not_in_space(ce, gene: str, ds: str):
    """基因不在 HVG 空间时，给出可操作的提示（附近似候选）。"""
    if gene in ce._gene_idx:
        return None
    up = gene.upper()
    near = [g for g in ce._gene_list if g.upper().startswith(up[:3])][:8] if len(up) >= 3 else []
    return JSONResponse({
        "error": f"基因 '{gene}' 不在 CIPHER 的 Top5000 高变基因集内（数据集 {ds}）",
        "reason": "CIPHER 只在高变基因上拟合协方差；低变异基因的协方差估计不可靠，故不提供预测。",
        "suggestions": near,
        "hint": "改用 attribution Top 列表里的基因，或换一个高变基因。",
    }, status_code=400)


def _celltype_rows(ds: str, ce, delta: np.ndarray, gene: str):
    """各细胞类型对该扰动的响应强度（Augur 风格排序）。稀疏安全。

    度量：该类型「特异富集」基因上的 |Δ| 加权均值
        w_ct[g] = max(0, expr_ct[g] - expr_all[g])   # 该类型相对全局的富集量
        response = Σ w_ct[g]·|Δ[g]| / Σ w_ct[g]

    为什么不用旧公式（该类型 Top500 高表达基因的 |Δ| 均值）：
    各细胞类型的最高表达基因几乎都是核糖体/管家基因、彼此高度重叠，
    导致所有类型得分几乎相同（实测 MS4A1 敲低时 B 细胞只排第 3，
    与第 1 名差距 <5%），图表没有判别力。改用特异富集加权后，
    扰动会正确命中携带该基因共表达模块的细胞类型。
    """
    d = _get_ds(ds)
    meta, X = d["meta"], d["X"]
    hvg_idx = _CIPHER_HVG_IDX.get(ds)
    gi_full = perturb_eng.gene_idx.get(gene)
    abs_delta = np.abs(delta)

    # 全局（所有细胞）在 HVG 空间的平均表达，作为特异性基准。
    # 注意：用「各类型均值的均值」而非全体细胞均值——否则最大的细胞群
    # （PBMC 里 T 细胞占 45%）会把基准拉向自己，导致它的特异基因权重被压扁，
    # 在任何扰动下都排不到前面（实测 CD3D 敲低时 T 细胞只排第 4）。
    cache_key = f"_hvg_ct_mean_{ds}"
    cached = d.get(cache_key)
    if cached is None:
        per_ct = {}
        for ct in meta["cell_type"].unique():
            m = (meta["cell_type"] == ct).values
            if int(m.sum()) < 5:
                continue
            sub_h = X[m][:, hvg_idx] if hvg_idx is not None else X[m]
            per_ct[str(ct)] = np.asarray(sub_h.mean(axis=0)).ravel()
        global_mean = np.mean(np.stack(list(per_ct.values())), axis=0)
        cached = (per_ct, global_mean)
        d[cache_key] = cached
    per_ct, global_mean = cached

    rows = []
    for ct_name, ct_expr_hvg in per_ct.items():
        ct_mask = (meta["cell_type"] == ct_name).values
        n_cells = int(ct_mask.sum())

        # 该类型特异富集的基因（相对各类型平均），作为特征向量
        w = np.maximum(ct_expr_hvg - global_mean, 0.0)
        nw, nd = np.linalg.norm(w), np.linalg.norm(abs_delta)
        if nw > 1e-9 and nd > 1e-9:
            # 余弦相似度：该类型的特征谱与扰动效应谱的对齐程度。
            # 同时消除两侧的幅度影响 —— 单纯用加权均值会让本身高表达的类型
            # （单核细胞 LYZ/S100A9）在任何扰动下都拿最高分。
            response = float((w * abs_delta).sum() / (nw * nd))
        else:
            response = 0.0

        # 该类型里贡献最大的 3 个基因（可解释性：为什么这个类型响应高）
        top_contrib_idx = np.argsort(-(w * abs_delta))[:3]
        drivers = [ce._gene_list[i] for i in top_contrib_idx if w[i] > 0]

        if gi_full is not None:
            col = X[ct_mask][:, gi_full]
            target_expr = float(np.asarray(col.mean()).ravel()[0]) if sp.issparse(col) else float(np.mean(col))
        else:
            target_expr = 0.0

        rows.append({
            "cell_type": ct_name,
            "n_cells": n_cells,
            "target_expr": round(target_expr, 3),
            "response": round(response, 6),
            "mean_abs_delta": round(response, 6),   # 兼容旧字段名
            "drivers": drivers,
        })
    rows.sort(key=lambda r: r["response"], reverse=True)
    return rows


@app.get("/api/cipher_status")
def cipher_status(ds: str = Query("pbmc")):
    """CIPHER 引擎在该数据集上的状态。"""
    _validate_dataset(ds)
    fitted = ds in _CIPHER_FITTED
    ce = _CIPHER_ENGINES.get(ds)
    return {
        "fitted": fitted,
        "loading": (ds not in _CIPHER_FITTED),
        "dataset": ds,
        "n_genes": len(ce._gene_list) if (ce and ce._fitted) else 0,
        "method": "CIPHER (线性响应)",
        "data_requirement": "仅需未扰动对照细胞",
        "note": "CIPHER 基于统计物理涨落-耗散定理，用对照细胞的基因-基因协方差预测任意基因扰动后的全转录组响应。支持敲低和激活两个方向。",
    }


@app.get("/api/gene_in_cipher")
def gene_in_cipher(ds: str = Query("pbmc"), gene: str = Query(...)):
    """查询某基因是否可被 CIPHER 预测（前端用来决定按钮可用性）。"""
    _validate_dataset(ds)
    ce = _CIPHER_ENGINES.get(ds)
    if ds not in _CIPHER_FITTED or ce is None or not ce._fitted:
        return {"gene": gene, "covered": False, "ready": False}
    return {"gene": gene, "covered": gene in ce._gene_idx, "ready": True,
            "ctrl_expr": round(float(ce._ctrl_mean[ce._gene_idx[gene]]), 3) if gene in ce._gene_idx else None}


@app.get("/api/perturb_cipher")
def perturb_cipher(
    ds: str = Query("pbmc"),
    gene: str = Query(...),
    direction: Literal["ko", "oe"] = Query("ko"),
    target_value: float = Query(None),
):
    """CIPHER 线性响应预测。
    direction: 'ko' (敲低→0) 或 'oe' (过表达→ctrl×2，或指定 target_value)。
    """
    ce, err = _require_cipher(ds)
    if err:
        return err
    err = _gene_not_in_space(ce, gene, ds)
    if err:
        return err

    gi = ce._gene_idx[gene]
    ctrl_expr = float(ce._ctrl_mean[gi])
    if target_value is None:
        target_value = 0.0 if direction == "ko" else ctrl_expr * 2.0

    t0 = time.time()
    r = ce.predict({gene: target_value})
    elapsed = time.time() - t0

    delta = r["delta"]
    top_delta = r["delta_top"][:15]
    return {
        "method": "CIPHER (线性响应)",
        "gene": gene,
        "direction": direction,
        "dataset": ds,
        "ctrl_expr": ctrl_expr,
        "target_value": target_value,
        "delta_abs_max": round(float(np.abs(delta).max()), 5),
        "delta_mean": round(float(np.abs(delta).mean()), 6),
        "n_sig_genes": int((np.abs(delta) > 0.001).sum()),
        "top_up": [(g, round(d, 5)) for g, d in reversed(top_delta) if d > 0][:8],
        "top_down": [(g, round(d, 5)) for g, d in top_delta if d < 0][:8],
        "elapsed": round(elapsed, 3),
        "note": "CIPHER 基于对照细胞协方差 + 线性响应理论（仅需未扰动数据）。结果用于 L3 假设生成，非因果验证。",
    }


@app.get("/api/perturb_baseline")
def perturb_baseline(
    ds: str = Query("pbmc"),
    gene: str = Query(...),
    method: Literal["ctrl_mean", "additive", "both"] = Query("both"),
):
    """线性基线预测（Nature Methods 2025 强制对照）。"""
    ce, err = _require_cipher(ds)
    if err:
        return err
    bl = _BASELINE_ENGINES.get(ds)
    if bl is None or gene not in bl._gene_idx:
        return JSONResponse({"error": f"基因 '{gene}' 不在模型基因空间内"}, status_code=400)

    results = {}
    if method in ("ctrl_mean", "both"):
        r = bl.predict_ctrl_mean()
        results["ctrl_mean"] = {
            "method": r["method"],
            "description": r["description"],
            "delta_abs_max": 0.0,
            "note": "这是 Nature Methods 2025 基准中追平/反超多数深度学习模型的最简基线。",
        }
    if method in ("additive", "both"):
        r = bl.predict_additive({gene: 0.0})
        results["additive"] = {
            "method": r["method"],
            "description": r["description"],
            "delta_abs_max": round(float(np.abs(r["delta"]).max()), 5),
            "top_changed": r["delta_top"][:5],
            "note": "仅目标基因自身变化（单基因简化）。真实 Perturb-seq 中应由实测 LFC 替代。",
        }
    return {
        "gene": gene, "dataset": ds, "results": results,
        "citation": "Ahlmann-Eltze et al., Nature Methods 2025. Deep-learning perturbation prediction does not yet outperform simple linear baselines.",
    }


@app.get("/api/gene_expr_map")
def gene_expr_map(ds: str = Query("pbmc"), gene: str = Query(...)):
    """该基因在每个细胞上的表达值，用于把 UMAP 按表达量着色。

    这是回答「这个基因到底在哪些细胞里表达」最直观的方式——
    比任何统计量都好使：一眼就能看出它是某个群体的特异标志物，
    还是全体细胞都在表达的管家基因。
    """
    d = _get_ds(ds)
    if gene not in perturb_eng.gene_idx:
        return JSONResponse({"error": f"基因 '{gene}' 不在基因空间内"}, status_code=400)
    gi = perturb_eng.gene_idx[gene]
    col = d["X"][:, gi]
    vals = np.asarray(col.todense()).ravel() if sp.issparse(col) else np.asarray(col).ravel()

    # 按分组统计，供前端在图注里说明「哪个群体表达最高」
    meta = d["meta"]
    by_group = []
    for ct in meta["cell_type"].unique():
        m = (meta["cell_type"] == ct).values
        if int(m.sum()) < 5:
            continue
        sub = vals[m]
        by_group.append({
            "group": str(ct),
            "n_cells": int(m.sum()),
            "mean": round(float(sub.mean()), 3),
            "pct_expressing": round(float((sub > 0).mean() * 100), 1),
        })
    by_group.sort(key=lambda r: r["mean"], reverse=True)

    return {
        "gene": gene,
        "dataset": ds,
        "group_label": DATASETS[ds].get("group_label", "细胞类型"),
        "values": [round(float(v), 3) for v in vals],
        "max": round(float(vals.max()), 3),
        "mean": round(float(vals.mean()), 4),
        "pct_expressing": round(float((vals > 0).mean() * 100), 1),
        "by_group": by_group,
    }


@app.get("/api/attribution_map")
def attribution_map(ds: str = Query("pbmc"), cell: int = Query(..., ge=0), k: int = Query(12, ge=1, le=100)):
    """单个细胞的归因结果 + 每个基因在该细胞/全体中的表达对比。

    侧栏的小表格放不下横向对比，做成图后能同时看到：
    归因强度（模型认为多重要）与实际表达量（这个细胞里有多少）。
    """
    d = _get_ds(ds)
    with _inference_lock:
        top = attr_eng.top_genes(_row(d["X"], cell), k=k)
    rows = []
    for g, a, e in top:
        gi = perturb_eng.gene_idx.get(g)
        if gi is not None:
            col = d["X"][:, gi]
            colv = np.asarray(col.todense()).ravel() if sp.issparse(col) else np.asarray(col).ravel()
            pop_mean = float(colv.mean())
        else:
            pop_mean = 0.0
        rows.append({
            "gene": g,
            "attribution": round(float(a), 4),
            "expr": round(float(e), 3),
            "pop_mean": round(pop_mean, 3),
            "enrichment": round(float(e) - pop_mean, 3),   # 相对全体的富集量
        })
    return {
        "cell": cell,
        "cell_type": str(d["meta"].cell_type.iloc[cell]),
        "dataset": ds,
        "rows": rows,
    }


# ---------- v3 可视化数据端点 ----------
@app.get("/api/perturb_compare")
def perturb_compare(
    ds: str = Query("pbmc"),
    gene: str = Query(...),
    direction: Literal["ko", "oe"] = Query("ko"),
):
    """CIPHER vs 线性基线 对比：同一基因、并排展示。"""
    ce, err = _require_cipher(ds)
    if err:
        return err
    err = _gene_not_in_space(ce, gene, ds)
    if err:
        return err

    gi = ce._gene_idx[gene]
    ctrl_expr = float(ce._ctrl_mean[gi])
    target = 0.0 if direction == "ko" else ctrl_expr * 2.0
    r = ce.predict({gene: target})
    delta = r["delta"]

    top_idx = np.argsort(-np.abs(delta))[:30]
    rows = []
    for i in top_idx:
        g = ce._gene_list[i]
        rows.append({
            "gene": g,
            "delta_cipher": round(float(delta[i]), 5),
            "ctrl_expr": round(float(ce._ctrl_mean[i]), 3),
            "baseline_ctrl_mean": round(float(ce._ctrl_mean[i]), 3),
            "baseline_additive": round(
                float(ce._ctrl_mean[i]) + (0 if g != gene else (target - ctrl_expr)), 3),
            "is_target": g == gene,
        })

    return {
        "gene": gene, "direction": direction, "dataset": ds,
        "ctrl_expr": ctrl_expr, "target": target, "rows": rows,
        "summary": {
            "n_sig_up": int((delta > 0.001).sum()),
            "n_sig_down": int((delta < -0.001).sum()),
            "delta_abs_max": round(float(np.abs(delta).max()), 5),
        },
    }


@app.get("/api/celltype_response")
def celltype_response(
    ds: str = Query("pbmc"),
    gene: str = Query(...),
    direction: Literal["ko", "oe"] = Query("ko"),
):
    """细胞类型响应排序（Augur 风格）。"""
    ce, err = _require_cipher(ds)
    if err:
        return err
    err = _gene_not_in_space(ce, gene, ds)
    if err:
        return err

    gi = ce._gene_idx[gene]
    ctrl_expr = float(ce._ctrl_mean[gi])
    target = 0.0 if direction == "ko" else ctrl_expr * 2.0
    r = ce.predict({gene: target})
    return {
        "gene": gene, "direction": direction, "dataset": ds,
        "cell_types": _celltype_rows(ds, ce, r["delta"], gene),
        "note": "mean_abs_delta = CIPHER 预测 |Δ| 在该类型 Top500 高表达基因上的均值。值越大 = 该细胞类型对该基因扰动越敏感。",
    }


@app.get("/api/perturb_summary")
def perturb_summary(
    ds: str = Query("pbmc"),
    gene: str = Query(...),
    direction: Literal["ko", "oe"] = Query("ko"),
):
    """扰动预测综合摘要：CIPHER 热图 + 细胞类型响应 一次性返回（前端主力端点）。"""
    ce, err = _require_cipher(ds)
    if err:
        return err
    err = _gene_not_in_space(ce, gene, ds)
    if err:
        return err

    gi = ce._gene_idx[gene]
    ctrl_expr = float(ce._ctrl_mean[gi])
    target = 0.0 if direction == "ko" else ctrl_expr * 2.0

    t0 = time.time()
    r = ce.predict({gene: target})
    delta = r["delta"]

    top_idx = np.argsort(-np.abs(delta))[:30]
    heatmap_rows = [{
        "gene": ce._gene_list[i],
        "delta": round(float(delta[i]), 5),
        "ctrl": round(float(ce._ctrl_mean[i]), 3),
        "is_target": ce._gene_list[i] == gene,
    } for i in top_idx]

    ct_rows = _celltype_rows(ds, ce, delta, gene)
    elapsed = round(time.time() - t0, 3)

    return {
        "gene": gene, "direction": direction, "dataset": ds,
        "group_label": DATASETS[ds].get("group_label", "细胞类型"),
        "ctrl_expr": ctrl_expr, "target": target, "elapsed": elapsed,
        "heatmap": heatmap_rows,
        "cell_types": ct_rows,
        "summary": {
            "n_sig_up": int((delta > 0.001).sum()),
            "n_sig_down": int((delta < -0.001).sum()),
            "delta_abs_max": round(float(np.abs(delta).max()), 5),
            "baseline_note": "control-mean 基线预测 |Δ|=0（扰动无效应）；CIPHER 预测了真实的基因协方差结构",
        },
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("VCT_PORT", "8377"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
