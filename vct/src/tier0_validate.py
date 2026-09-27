"""Tier 0 探索报告：SCimilarity 归因 vs 真实扰动效应（历史分析）

探索真实单基因敲低细胞中"归因排序 ≈ 因果重要性"的假设，不作为科学验收，不用 GEARS。
GEARS 的职责见 docs/DESIGN_REVIEW_ITERATIVE.md（Tier 2 扩展层）。

产出 data/tier0_validation.json:
  - n_valid_genes
  - spearman_attr_effect / spearman_expr_effect / spearman_random_effect (+p)
  - recall@10/20/50, AUROC(top10% 效应为正类)
  - 效应 top 基因表（sanity check）
"""
import os, sys, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import torch
from scipy.stats import spearmanr
from captum.attr import IntegratedGradients

from paths import SIG_DIR, MODEL_DIR, VCT_ROOT, GEARS_DATA_DIR, DATA_DIR
from exploration_report import EXPLORATION_NOTICE, write_exploration_report

print(EXPLORATION_NOTICE)

sys.path.insert(0, os.path.join(SIG_DIR, "src"))
from SIGnature.models.scimilarity import SCimilarityWrapper
from SIGnature.utils import align_dataset, lognorm_counts

DATA = DATA_DIR
N_STEPS = 16
BATCH = 64


def perturbed_gene(cond):
    parts = str(cond).split("+")
    non_ctrl = [p for p in parts if p != "ctrl"]
    return non_ctrl[0] if len(non_ctrl) == 1 else None


print("1/6 加载 Norman 数据（raw counts）...")
a = ad.read_h5ad(os.path.join(GEARS_DATA_DIR, "norman/perturb_processed.h5ad"))
raw = a.layers["counts"]
gene_names = np.array(a.var["gene_name"].values)
cond = a.obs["condition"].values.astype(str)

ctrl_mask = cond == "ctrl"
pgene = np.array([perturbed_gene(c) for c in cond], dtype=object)
single_mask = np.array([x is not None for x in pgene])
keep = ctrl_mask | single_mask
raw_sub = raw[keep]  # 兼容 dense(ndarray) / sparse
cond_sub = cond[keep]
pgene_sub = pgene[keep]
print(f"   ctrl 细胞 {int(ctrl_mask.sum())} | 单基因敲低细胞 {int(single_mask.sum())} | 丢弃双基因 {int((~ctrl_mask & ~single_mask).sum())}")

print("2/6 对齐 SCimilarity 28231 基因空间 + lognorm ...")
adata = ad.AnnData(X=raw_sub.copy(), var=pd.DataFrame(index=gene_names))
adata.var_names_make_unique()
wrapper = SCimilarityWrapper(model_path=MODEL_DIR)
overlap = int(sum(adata.var_names.isin(wrapper.gene_order)))
print(f"   基因重合: {overlap}/{len(gene_names)}")
adata = align_dataset(adata, wrapper.gene_order, gene_overlap_threshold=500)
adata.layers["counts"] = adata.X.copy()
lognorm_counts(adata)
X = adata.X.tocsr()
print(f"   aligned: {adata.n_obs} x {adata.n_vars}")

print("3/6 计算 embedding（全部子集细胞）...")
model = wrapper.model.eval()
embs = []
with torch.no_grad():
    for i in range(0, X.shape[0], 512):
        xb = X[i:i + 512].toarray().astype(np.float32)
        embs.append(model(torch.tensor(xb)).numpy())
emb = np.vstack(embs)
print(f"   embedding: {emb.shape}")

print("4/6 对 ctrl 细胞做 IG 归因，累积 mean|attr| ...")
ctrl_idx = np.where(cond_sub == "ctrl")[0]
Xc = X[ctrl_idx]
sum_abs_attr = np.zeros(X.shape[1], dtype=np.float64)


def sum_out(x):
    return model(x).sum(dim=1)


ig = IntegratedGradients(sum_out)
t0 = time.time()
for i in range(0, len(ctrl_idx), BATCH):
    xb = torch.tensor(Xc[i:i + BATCH].toarray().astype(np.float32), requires_grad=True)
    attr = ig.attribute(xb, baselines=torch.zeros_like(xb), n_steps=N_STEPS).detach().numpy()
    sum_abs_attr += np.abs(attr).sum(axis=0)
    if (i // BATCH) % 15 == 0:
        print(f"   {i + BATCH}/{len(ctrl_idx)} ({time.time() - t0:.0f}s)")
mean_abs_attr = sum_abs_attr / len(ctrl_idx)
expr_ctrl = np.asarray(Xc.mean(axis=0)).ravel()
print(f"   归因完成 {time.time() - t0:.0f}s")

print("5/6 计算每个单基因敲低的真实效应 ...")
ctrl_centroid = emb[ctrl_idx].mean(axis=0)
genes = sorted({g for g in pgene_sub if g is not None})
rows = []
for g in genes:
    if g not in wrapper.gene_order:
        continue
    gi = wrapper.gene_order.index(g)
    idx = np.where(pgene_sub == g)[0]
    effect = float(np.linalg.norm(emb[idx].mean(axis=0) - ctrl_centroid))
    rows.append({"gene": g, "n_cells": int(len(idx)),
                 "attr": float(mean_abs_attr[gi]), "expr": float(expr_ctrl[gi]),
                 "effect": effect})
df = pd.DataFrame(rows).sort_values("effect", ascending=False).reset_index(drop=True)
print(f"   有效单基因扰动: {len(df)} (映射进 SCimilarity 且有敲低细胞)")

print("6/6 正向 + 反向检验 ...")
r_attr = spearmanr(df["attr"], df["effect"])
r_expr = spearmanr(df["expr"], df["effect"])
rng = np.random.default_rng(0)
rand_rs = [spearmanr(rng.permutation(df["attr"].values), df["effect"].values).statistic for _ in range(200)]
r_rand_mean = float(np.mean(rand_rs))
r_rand_std = float(np.std(rand_rs))


def recall(k):
    top_eff = set(df.head(k)["gene"])
    top_attr = set(df.nlargest(k, "attr")["gene"])
    return len(top_eff & top_attr) / k


rec = {f"recall@{k}": recall(k) for k in [10, 20, 50]}
n_pos = max(1, int(len(df) * 0.1))
pos = set(df.head(n_pos)["gene"])
y = df["gene"].isin(pos).astype(int).values
attr_vals = df["attr"].values
pos_attr = attr_vals[y == 1]
neg_attr = attr_vals[y == 0]
auc_sum = 0.0
for p in pos_attr:
    auc_sum += float((neg_attr < p).sum()) + 0.5 * float((neg_attr == p).sum())
auroc = auc_sum / (len(pos_attr) * len(neg_attr))

result = {
    "cell_type": "A549",
    "n_valid_genes": int(len(df)),
    "spearman_attr_effect": {"r": float(r_attr.statistic), "p": float(r_attr.pvalue)},
    "spearman_expr_effect": {"r": float(r_expr.statistic), "p": float(r_expr.pvalue)},
    "spearman_random_effect": {"mean": r_rand_mean, "std": r_rand_std},
    **rec,
    "auroc_top10_effect": float(auroc),
}
print("\n========== Tier 0 探索结果 ==========")
for k, v in result.items():
    if isinstance(v, dict):
        print(f"  {k}: " + " ".join(f"{kk}={vv:.4g}" if isinstance(vv, float) else f"{kk}={vv}" for kk, vv in v.items()))
    else:
        print(f"  {k}: {v}")
print("\n  Top10 真实效应基因（sanity check）:")
print(df.head(10).to_string(index=False))
write_exploration_report(os.path.join(DATA, "tier0_validation.json"),
                         {"metrics": result, "table": df.to_dict(orient="records")})
