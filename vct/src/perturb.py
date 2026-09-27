"""M3: 核心扰动引擎 — 修改基因表达 → 重算 embedding → 位移分析
自有工作（见 ATTRIBUTIONS.md）
"""
import os
import numpy as np
import torch

from paths import SIG_DIR, MODEL_DIR
from input_checks import check_model_input, check_perturbation_value

import sys
sys.path.insert(0, os.path.join(SIG_DIR, "src"))
from SIGnature.models.scimilarity import SCimilarityWrapper


class PerturbEngine:
    """虚拟细胞核心：对表达向量做反事实扰动，观察 embedding 迁移。"""

    def __init__(self):
        self.wrapper = SCimilarityWrapper(model_path=MODEL_DIR)
        self.wrapper.model.eval()
        self.gene_idx = {g: i for i, g in enumerate(self.wrapper.gene_order)}

    def _gene_index(self, gene):
        if not isinstance(gene, str) or gene not in self.gene_idx:
            raise ValueError(f"基因 {gene} 不在模型的 {len(self.gene_idx)} 基因空间内")
        return self.gene_idx[gene]

    def embed(self, X: np.ndarray) -> np.ndarray:
        """(n, 28231) lognorm 表达 → (n, 128) embedding"""
        X = check_model_input(X, len(self.wrapper.gene_order), ndim=2)
        self.wrapper.model.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32)
            outs = [self.wrapper.model(t[i:i+512]).numpy() for i in range(0, len(t), 512)]
        return np.vstack(outs)

    def perturb(self, x: np.ndarray, gene: str, new_value: float) -> dict:
        """对单个细胞向量 x (28231,) 把 gene 设为 new_value（lognorm 尺度），返回位移信息。"""
        x = check_model_input(x, len(self.wrapper.gene_order))
        i = self._gene_index(gene)
        new_value = check_perturbation_value(new_value)
        x_new = x.copy()
        old_value = float(x[i])
        x_new[i] = new_value
        e_old = self.embed(x[None, :])[0]
        e_new = self.embed(x_new[None, :])[0]
        delta = e_new - e_old
        return {
            "gene": gene,
            "old_value": old_value,
            "new_value": float(new_value),
            "emb_old": e_old,
            "emb_new": e_new,
            "delta": delta,
            "displacement": float(np.linalg.norm(delta)),
        }

    def perturb_multi(self, x: np.ndarray, gene_values: dict) -> dict:
        """多基因联合扰动：gene_values = {gene: new_value}，返回位移信息。"""
        x = check_model_input(x, len(self.wrapper.gene_order))
        x_new = x.copy()
        applied = {}
        for g, v in gene_values.items():
            i = self._gene_index(g)
            v = check_perturbation_value(v)
            applied[g] = (float(x[i]), v)
            x_new[i] = v
        e_old = self.embed(x[None, :])[0]
        e_new = self.embed(x_new[None, :])[0]
        delta = e_new - e_old
        return {
            "applied": applied,
            "emb_old": e_old,
            "emb_new": e_new,
            "delta": delta,
            "displacement": float(np.linalg.norm(delta)),
        }

    def batch_perturb(self, X: np.ndarray, gene: str, new_value: float) -> np.ndarray:
        """对一批细胞做同样扰动，返回 (n,128) 新 embedding。"""
        X = check_model_input(X, len(self.wrapper.gene_order), ndim=2)
        i = self._gene_index(gene)
        new_value = check_perturbation_value(new_value)
        X_new = X.copy()
        X_new[:, i] = new_value
        return self.embed(X_new)


if __name__ == "__main__":
    # 快速自测：随机向量扰动
    eng = PerturbEngine()
    x = np.random.rand(28231).astype(np.float32)
    r = eng.perturb(x, "MS4A1", 0.0)
    print(f"自测: 敲低 MS4A1 → 位移 {r['displacement']:.4f}")
    print("✅ M3 扰动引擎可用")
