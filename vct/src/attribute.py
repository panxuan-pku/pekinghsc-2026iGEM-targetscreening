"""M4: Attribution 模块 — 用 Captum IG 计算每个基因对 embedding 的贡献
基于 SIGnature 模型包装层思路（Captum），自有封装（见 ATTRIBUTIONS.md）
"""
import os
import numpy as np
import torch
from captum.attr import IntegratedGradients

from paths import SIG_DIR, MODEL_DIR
from input_checks import check_model_input

import sys
sys.path.insert(0, os.path.join(SIG_DIR, "src"))
from SIGnature.models.scimilarity import SCimilarityWrapper


class AttributionEngine:
    """对细胞表达向量算 IG attribution：哪个基因对细胞身份最重要。"""

    def __init__(self):
        self.wrapper = SCimilarityWrapper(model_path=MODEL_DIR)
        self.wrapper.model.eval()
        self.gene_order = self.wrapper.gene_order

    def _sum_output(self, x):
        """多维 embedding → 标量（求和），供梯度归因（SIGnature 同款做法）。"""
        return self.wrapper.model(x).sum(dim=1)

    def attribute(self, x: np.ndarray, n_steps: int = 32) -> np.ndarray:
        """x: (28231,) → attribution (28231,)，与表达向量同形。"""
        x = check_model_input(x, len(self.gene_order))
        t = torch.tensor(x[None, :], dtype=torch.float32, requires_grad=True)
        baseline = torch.zeros_like(t)
        ig = IntegratedGradients(self._sum_output)
        attr = ig.attribute(t, baselines=baseline, n_steps=n_steps)
        return attr.detach().numpy()[0]

    def top_genes(self, x: np.ndarray, k: int = 20) -> list:
        attr = self.attribute(x)
        idx = np.argsort(-np.abs(attr))[:k]
        return [(self.gene_order[i], float(attr[i]), float(x[i])) for i in idx]


if __name__ == "__main__":
    eng = AttributionEngine()
    x = np.random.rand(28231).astype(np.float32) * 3
    top = eng.top_genes(x, k=5)
    for g, a, e in top:
        print(f"  {g}: attribution={a:+.3f}, expr={e:.2f}")
    print("✅ M4 Attribution 模块可用")
