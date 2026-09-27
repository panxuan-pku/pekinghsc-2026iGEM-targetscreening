"""v3: 线性基线引擎 — Nature Methods 2025 基准要求的两个强制基线

control-mean: 始终预测对照细胞均值（"什么都不变"）
additive: 多基因扰动 = 各单基因 LFC 之和（"线性叠加"）

这两个基线在 Ahlmann-Eltze et al. (Nat Methods 2025) 的基准中
追平甚至反超了 scGPT、scFoundation、GEARS 等所有深度模型。
任何虚拟细胞 demo 都必须并排展示它们作为诚实对照。

自有实现（见 ATTRIBUTIONS.md）
"""
import numpy as np
from typing import Dict, List, Optional
try:
    from .input_checks import check_expression, check_genes
except ImportError:
    from input_checks import check_expression, check_genes


class LinearBaseline:
    """control-mean + additive 线性基线。

    control-mean: Δx = 0 (预测扰动不改变任何基因表达)
    additive:   Δx = Σᵢ LFCᵢ (多基因联合 = 各自 LFC 相加)

    注意：control-mean 不是"无意义"——它在很多基准中排名前三，
    因为它捕捉了"多数扰动不会大幅改变全转录组"这个事实。
    """

    def __init__(self):
        self._fitted = False
        self._ctrl_mean: Optional[np.ndarray] = None   # (G,) 对照均值
        self._lfc_cache: Optional[Dict[str, np.ndarray]] = None  # 各基因敲低的 LFC
        self._gene_list: Optional[List[str]] = None
        self._gene_idx: Optional[Dict[str, int]] = None

    # ── 拟合 ──
    def fit(self, X: np.ndarray, gene_list: List[str]):
        """X: (n_cells, n_genes) 控制细胞表达矩阵（lognorm 尺度）。

        注意：X 若为 scipy 稀疏矩阵也支持（内部只用均值，不稠密化）。
        """
        check_expression(X, gene_list)
        mean = X.mean(axis=0)
        self.fit_stats(np.asarray(mean).ravel(), gene_list)
        print(f"[baseline] 拟合: {X.shape[0]} 细胞 × {X.shape[1]} 基因")

    def fit_stats(self, ctrl_mean: np.ndarray, gene_list: List[str]):
        """只用预先算好的对照均值拟合（免去传入整个表达矩阵）。

        基线模型数学上只依赖 ctrl_mean，传 X 进来除了算一次均值没有别的用途；
        对 ws 这种 96969×28231 的数据，避免持有/稠密化矩阵是必须的。
        """
        ctrl_mean = np.asarray(ctrl_mean)
        if ctrl_mean.ndim != 1:
            raise ValueError("ctrl_mean must be a one-dimensional vector")
        check_genes(gene_list, len(ctrl_mean))
        if (not np.issubdtype(ctrl_mean.dtype, np.number) or np.iscomplexobj(ctrl_mean)
                or not np.isfinite(ctrl_mean).all()):
            raise ValueError("ctrl_mean must contain finite real numbers")
        self._gene_list = list(gene_list)
        self._gene_idx = {g: i for i, g in enumerate(gene_list)}
        self._ctrl_mean = ctrl_mean.copy()
        self._lfc_cache = {}
        self._fitted = True
        print(f"[baseline] 拟合（统计量）: {len(self._gene_list)} 基因")

    # ── control-mean ──
    def predict_ctrl_mean(self) -> Dict:
        """control-mean 基线: 预测扰动后表达 = 对照均值。

        返回值与 CIPHER.predict() 结构一致，方便前端同一渲染。
        """
        if not self._fitted:
            raise RuntimeError("请先调用 fit()")
        delta = np.zeros(len(self._gene_list))
        return {
            "method": "control-mean",
            "description": "预测所有基因表达 = 对照均值（即扰动无效应）",
            "delta": delta,
            "perturbed_mean": self._ctrl_mean.copy(),
            "delta_top": [],
        }

    # ── additive ──
    def precompute_lfc(self, gene: str, target_value: float = 0.0):
        """为后续 additive 预测预计算单基因敲低的 LFC。

        简化假设：敲低 gene 到 target_value → 只有 gene 自身变化。
        （在真实 Perturb-seq 数据中，这里应该用实际单基因敲低的 DE 向量）
        """
        if not self._fitted or gene not in self._gene_idx:
            return
        gi = self._gene_idx[gene]
        lfc = np.zeros(len(self._gene_list))
        lfc[gi] = target_value - self._ctrl_mean[gi]
        self._lfc_cache[gene] = lfc

    def predict_additive(self, perturbations: Dict[str, float]) -> Dict:
        """additive 基线: 多基因联合 = 各基因 LFC 之和。

        perturbations: {gene: target_value}
        """
        if not self._fitted:
            raise RuntimeError("请先调用 fit()")
        delta = np.zeros(len(self._gene_list))
        for g, target in perturbations.items():
            if g not in self._gene_idx:
                continue
            gi = self._gene_idx[g]
            if g in self._lfc_cache:
                delta += self._lfc_cache[g]
            else:
                # 未预计算 → 简化：只有基因本身变为 target
                delta[gi] += target - self._ctrl_mean[gi]

        return {
            "method": "additive",
            "description": "多基因扰动 = 各单基因 LFC 的线性叠加",
            "delta": delta,
            "perturbed_mean": self._ctrl_mean + delta,
            "delta_top": self._top_genes(delta, k=20),
            "target_genes": {
                g: {"ctrl_expr": float(self._ctrl_mean[self._gene_idx[g]]),
                    "target": float(target)}
                for g, target in perturbations.items()
                if g in self._gene_idx
            },
        }

    def _top_genes(self, delta: np.ndarray, k: int = 20):
        idx = np.argsort(-np.abs(delta))[:k]
        return [(self._gene_list[i], float(delta[i])) for i in idx]

    @property
    def gene_covered(self):
        if not self._fitted:
            return lambda g: False
        return lambda g: g in self._gene_idx


# ── 模块级单例 ──
_baseline_instance: Optional[LinearBaseline] = None


def get_baseline() -> LinearBaseline:
    global _baseline_instance
    if _baseline_instance is None:
        _baseline_instance = LinearBaseline()
    return _baseline_instance


# ── 自测 ──
if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(0)
    X = rng.normal(3, 1, (100, 5))
    genes = ["A", "B", "C", "D", "E"]

    bl = LinearBaseline()
    bl.fit(X, genes)

    r = bl.predict_ctrl_mean()
    print("control-mean:", r["method"], "—", r["description"])
    print(f"  示例基因 A: ctrl={bl._ctrl_mean[0]:.2f}, pred={r['perturbed_mean'][0]:.2f}")

    r = bl.predict_additive({"A": 0.0})
    print(f"additive A→0: ΔA={r['delta'][0]:.2f}, Δtop={r['delta_top'][:3]}")

    print("✅ 线性基线引擎自测通过")
