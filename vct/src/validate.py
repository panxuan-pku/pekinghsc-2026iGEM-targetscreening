"""M5: AC1/AC2 历史探索报告（不是科学有效性验收）
AC1 方向性: 敲低 B 细胞标志基因 MS4A1 后，B 细胞 embedding 应远离 B 细胞质心（与随机基因对比）
AC2 区分度: 标志基因扰动的位移幅度应显著大于随机基因
置换检验 p 值。结果写入 data/validation_results.json
"""
import os
from paths import DATA_DIR
import numpy as np
import pandas as pd
import scipy.sparse as sp
from perturb import PerturbEngine
from exploration_report import EXPLORATION_NOTICE, write_exploration_report

print(EXPLORATION_NOTICE)

DATA = DATA_DIR
emb = np.load(os.path.join(DATA, "embeddings.npy"))
X = sp.load_npz(os.path.join(DATA, "expr_aligned.npz")).toarray()
meta = pd.read_csv(os.path.join(DATA, "meta.csv"))

eng = PerturbEngine()
rng = np.random.default_rng(42)

# 选 B 细胞亚群
b_mask = (meta.cell_type == "B cell").values
b_idx = np.where(b_mask)[0]
print(f"B 细胞数: {len(b_idx)}")
if len(b_idx) < 10:
    raise SystemExit("B 细胞太少，换细胞类型")

centroid_b = emb[b_idx].mean(axis=0)

def knockdown_stats(gene_list, label):
    """敲低基因组 → 每个 B 细胞的位移 & 远离质心程度"""
    disps, away = [], []
    for g in gene_list:
        if g not in eng.gene_idx:
            continue
        emb_new = eng.batch_perturb(X[b_idx], g, 0.0)
        delta = emb_new - emb[b_idx]
        # 位移幅度
        d = np.linalg.norm(delta, axis=1)
        # 远离质心: 扰动后到质心距离 - 扰动前到质心距离
        away.append(np.linalg.norm(emb_new - centroid_b, axis=1) - np.linalg.norm(emb[b_idx] - centroid_b, axis=1))
        disps.append(d)
    return np.mean(disps), np.mean(away)

# 真实标志基因组 vs 随机基因组（控制基因表达水平相近: 选在 B 细胞中有表达的随机基因）
MARKER = ["MS4A1", "CD79A", "CD79B"]
expressed = np.where(X[b_idx].mean(axis=0) > 0.1)[0]
random_sets = [list(rng.choice(expressed, size=3, replace=False)) for _ in range(200)]
random_genes = [[eng.wrapper.gene_order[i] for i in s] for s in random_sets]

disp_m, away_m = knockdown_stats(MARKER, "marker")
rand_disp = np.array([knockdown_stats(gs, "rand")[0] for gs in random_genes])
rand_away = np.array([knockdown_stats(gs, "rand")[1] for gs in random_genes])

p_disp = (rand_disp >= disp_m).mean()
p_away = (rand_away >= away_m).mean()

print(f"\n=== AC2 位移幅度 ===")
print(f"标志基因敲低平均位移: {disp_m:.4f} | 随机基因: {rand_disp.mean():.4f} ± {rand_disp.std():.4f} | p={p_disp:.4f}")
print(f"=== AC1 方向性（远离B质心） ===")
print(f"标志基因敲低: {away_m:+.4f} | 随机基因: {rand_away.mean():+.4f} ± {rand_away.std():.4f} | p={p_away:.4f}")

res = {
    "AC1_direction": {"marker_away": float(away_m), "random_away_mean": float(rand_away.mean()),
                      "random_away_std": float(rand_away.std()), "p_value": float(p_away),
                      "pass": bool(p_away < 0.05 and away_m > 0)},
    "AC2_magnitude": {"marker_disp": float(disp_m), "random_disp_mean": float(rand_disp.mean()),
                      "random_disp_std": float(rand_disp.std()), "p_value": float(p_disp),
                      "pass": bool(p_disp < 0.05 and disp_m > rand_disp.mean())},
    "n_b_cells": int(len(b_idx)), "markers": MARKER, "n_random_sets": 200,
}
# Retain historical pass fields and formulas for comparison, not an exit-code gate.
write_exploration_report(os.path.join(DATA, "validation_results.json"), res)
