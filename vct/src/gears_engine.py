"""v2: GEARS 因果扰动引擎 — 用 Perturb-seq 训练的预测模型替代/补充反事实扰动
引擎定位（见 docs/IGEM_ARCHITECTURE.md 循环 5）：
  - 反事实引擎(perturb.py, v1)：任意基因、秒级、模型内反事实 → 交互演示用
  - 因果引擎(本文件, v2)：仅限 GEARS 训练图内的基因、分钟级、真实扰动数据训练 → 深度分析用
覆盖性检查来自 E11 实验：微缺失病主效基因不在 Norman 训练图 → 前端必须先查再启用。
"""
import os, sys, pickle
import numpy as np

from paths import VCT_ROOT, GEARS_DATA_DIR, GEARS_CKPT_DIR, DATA_DIR

# pandas 2.x 兼容补丁（GEARS 内部依赖 Series.nonzero，见 load_gears.py 踩坑记录）
import pandas as pd
if not hasattr(pd.Series, "nonzero"):
    pd.Series.nonzero = lambda self: np.flatnonzero(self.values)

ROOT = VCT_ROOT
DATA_DIR = GEARS_DATA_DIR
CKPT = GEARS_CKPT_DIR


class GearsEngine:
    """GEARS 因果预测引擎（Norman 预训练权重）。
    惰性加载：GEARS + PertData 约占 2GB 内存、加载 ~30s，只在首次因果预测时初始化。
    """

    def __init__(self):
        self._model = None
        self._pert_genes = None  # Norman 训练图覆盖的扰动基因集合

    # ---------- 惰性加载 ----------
    def _ensure_loaded(self):
        if self._model is not None:
            return
        import scipy.sparse as sp
        from gears import PertData, GEARS

        print("[gears_engine] 首次加载 GEARS 模型（约 30s）...")
        pert_data = PertData(DATA_DIR, default_pert_graph=False)
        pert_data.load(data_name="norman")
        pert_data.prepare_split(split="simulation", seed=1)
        pert_data.get_dataloader(batch_size=32, test_batch_size=128)
        # scipy 兼容修复（load_gears.py 同款）：视图实体化 + 稀疏转稠密
        pert_data.adata = pert_data.adata.to_memory()
        if sp.issparse(pert_data.adata.X):
            pert_data.adata.X = pert_data.adata.X.toarray().astype(np.float32)

        model = GEARS(pert_data, device="cpu")
        model.load_pretrained(CKPT)
        self._model = model
        # 训练图覆盖的扰动基因 = adata 里出现过的 perturbation 基因
        self._pert_genes = self._load_pert_genes()
        print(f"[gears_engine] 就绪：训练图覆盖 {len(self._pert_genes)} 个扰动基因")

    def _load_pert_genes(self) -> set:
        """GEARS 训练图的真实扰动基因覆盖。
        优先读缓存 data/gears_pert_list.pkl；缓存缺失时从已加载模型动态提取。
        注意（E11 教训）：essential_all_data_pert_genes.pkl 是 9976 基因的跨数据集总表，
        不等于模型可预测的 Norman 训练图。权威来源 = model.pert_list。
        """
        cache = os.path.join(DATA_DIR, "gears_pert_list.pkl")
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                genes = pickle.load(f)
            if genes:
                print(f"[gears_engine] 训练图覆盖 {len(genes)} 基因（来自缓存 gears_pert_list.pkl）")
                return genes
        # 缓存缺失 → 从已加载的模型提取
        if self._model is not None:
            genes = set(self._model.pert_list)
            print(f"[gears_engine] 训练图覆盖 {len(genes)} 基因（从模型 pert_list 提取）")
            # 写回缓存，避免下次再走全模型加载
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            with open(cache, "wb") as f:
                pickle.dump(genes, f)
            return genes
        # 模型也未加载 → 返回空，gene_covered 全 False（安全降级）
        print("[gears_engine] 模型未加载且无缓存，因果模式待 _ensure_loaded() 初始化")
        return set()

    # ---------- 公共接口 ----------
    def gene_covered(self, gene: str) -> bool:
        """基因是否在 GEARS 训练图内（决定前端是否亮因果按钮）。"""
        return gene in self._pert_genes if self._pert_genes else False

    def coverage_set(self) -> set:
        self._ensure_loaded()
        return self._pert_genes

    def predict_single(self, gene: str) -> dict:
        """单基因敲低的因果预测（GEARS predict 全转录组）。返回轻量摘要。"""
        self._ensure_loaded()
        if not self.gene_covered(gene):
            return {"error": f"基因 {gene} 不在 GEARS 训练图内（Norman/K562 Perturb-seq 覆盖范围），无因果预测可用。请使用反事实模式。"}
        pred = self._model.predict([[gene]])
        key = list(pred.keys())[0]
        delta = np.asarray(pred[key]).ravel()  # 预测的 post-perturb 表达（DE 空间）
        order = np.argsort(-np.abs(delta))[:20]
        gene_names = np.asarray(self._model.gene_list)  # GEARS 内置 var 顺序基因名（5045 训练图基因）
        top = [(str(gene_names[i]), round(float(delta[i]), 4)) for i in order]
        return {
            "engine": "gears-causal",
            "gene": gene,
            "n_genes_predicted": int(delta.shape[0]),
            "delta_mean": round(float(delta.mean()), 5),
            "delta_abs_max": round(float(np.abs(delta).max()), 4),
            "top_up": [t for t in top if t[1] > 0][:10],
            "top_down": [t for t in top if t[1] < 0][:10],
            "note": "GEARS 因果预测基于 Norman (K562) Perturb-seq 训练；对非造血系细胞类型为域外推断，解读需谨慎。",
        }


_engine = None

def get_gears_engine() -> GearsEngine:
    global _engine
    if _engine is None:
        _engine = GearsEngine()
    return _engine


if __name__ == "__main__":
    eng = get_gears_engine()
    print("CEBPA 在训练图内:", eng.gene_covered("CEBPA"))
    eng._ensure_loaded()
    print("GTF2I 在训练图内:", eng.gene_covered("GTF2I"), "（应为 False，E11 结论）")
    r = eng.predict_single("CEBPA")
    print("CEBPA 预测:", {k: r[k] for k in ("engine", "n_genes_predicted", "delta_abs_max")})
    print("top_down[:5]:", r["top_down"][:5])
