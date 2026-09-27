"""历史探索：梯度位移和归因相关性报告，不是自动化科学验收。"""
import os, sys
import numpy as np
import anndata as ad
import torch
from scipy.stats import spearmanr, rankdata

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from paths import VCT_ROOT, SIG_DIR, MODEL_DIR, GEARS_DATA_DIR, DATA_DIR
from exploration_report import EXPLORATION_NOTICE, write_exploration_report

print(EXPLORATION_NOTICE)

sys.path.insert(0, os.path.join(SIG_DIR, "src"))
from SIGnature.models.scimilarity import SCimilarityWrapper
from captum.attr import IntegratedGradients

rng = np.random.default_rng(0)

print("加载 Norman 数据 ...")
adata = ad.read_h5ad(os.path.join(GEARS_DATA_DIR, "norman", "perturb_processed.h5ad"))
obs = adata.obs
gene_names = np.array(adata.var["gene_name"].astype(str))
X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
ctrl_mask = (obs["condition"] == "ctrl").values
print(f"总细胞 {X.shape[0]}, 基因 {X.shape[1]}, ctrl 细胞 {ctrl_mask.sum()}")

wrapper = SCimilarityWrapper(model_path=os.path.join(SIG_DIR, "model_files/model_files/scimilarity"))
wrapper.model.eval()
gidx = {g: i for i, g in enumerate(wrapper.gene_order)}
norman_to_scim = [gidx[g] if g in gidx else -1 for g in gene_names]
G = len(wrapper.gene_order)

def to_scim_batch(vecs):
    n = vecs.shape[0]
    out = np.zeros((n, G), dtype=np.float32)
    for j, k in enumerate(norman_to_scim):
        if k >= 0:
            out[:, k] = vecs[:, j]
    return out

def embed(xs):
    with torch.no_grad():
        t = torch.tensor(xs, dtype=torch.float32)
        outs = [wrapper.model(t[i:i+256]).numpy() for i in range(0, len(t), 256)]
    return np.vstack(outs)

sum_out = lambda x: wrapper.model(x).sum(dim=1)
ig = IntegratedGradients(sum_out)

def attr_batch(xs):
    t = torch.tensor(xs, dtype=torch.float32, requires_grad=True)
    return ig.attribute(t, baselines=torch.zeros_like(t), n_steps=16).detach().numpy()

def jacobian_col(xs, g_col):  # (n,G)->(n,128) = d z / d x_G
    t = torch.tensor(xs, dtype=torch.float32, requires_grad=True)
    z = wrapper.model(t)
    n = z.shape[0]
    J = torch.zeros(n, 128, dtype=torch.float32)
    for j in range(128):
        gr = torch.autograd.grad(z[:, j].sum(), t, retain_graph=True)[0]
        J[:, j] = gr[:, g_col]
    return J.detach().numpy()

# 固定 ctrl 采样（预测侧只用它）
ctrl_idx = np.where(ctrl_mask)[0]
ctrl_idx = rng.choice(ctrl_idx, size=80, replace=False)
ctrl_X = X[ctrl_idx]
ctrl_Xs = to_scim_batch(ctrl_X)
ctrl_embed = embed(ctrl_Xs)
print("ctrl 归因中 ...")
ctrl_attr = attr_batch(ctrl_Xs[:50])          # (50,G)
ctrl_attr_5045 = np.zeros((ctrl_attr.shape[0], len(gene_names)))
for j, k in enumerate(norman_to_scim):
    if k >= 0:
        ctrl_attr_5045[:, j] = ctrl_attr[:, k]
mean_abs_attr = np.abs(ctrl_attr_5045).mean(0)   # (5045,) 重要性（ctrl 状态）

# ============ 枚举单基因扰动 ============
single_genes = set()
for c in obs["condition"].unique():
    s = str(c)
    if "ctrl" not in s:
        continue
    gs = [p for p in s.split("+") if p != "ctrl"]
    if len(gs) == 1:
        single_genes.add(gs[0])
single_genes = sorted(g for g in single_genes if g in set(gene_names))
print(f"单基因扰动条件数: {len(single_genes)}")

def gene_de(g):
    conds = [c for c in obs["condition"].unique() if str(c) in (f"{g}+ctrl", f"ctrl+{g}")]
    treat_mask = obs["condition"].isin(conds).values
    treat = X[treat_mask]
    return treat.mean(0) - ctrl_X.mean(0), int(treat_mask.sum())  # (5045,)

# ============ 测试A：一阶反事实预测 embedding 位移（方向） ============
print("\n===== 测试A：只用 ctrl 的梯度预测扰动 embedding 位移方向 =====")
print(f"{'基因':8s} | {'cosine(预测,真实)':>18s} | {'随机null p值':>12s} | {'位移幅度比 预测/真实':>16s}")
testA_genes = ["KLF1", "CEBPA", "BAK1", "ETS2", "CEBPE"]
cosines = {}
direction_rows = []
for g in testA_genes:
    if g not in set(gene_names):
        print(f"{g:8s} 不在数据中"); continue
    gi = int(np.where(gene_names == g)[0][0])
    g_col = norman_to_scim[gi]
    if g_col < 0:
        print(f"{g:8s} 不在 SCimilarity 空间"); continue
    # 预测侧：ctrl 梯度
    J = jacobian_col(ctrl_Xs[:20], g_col)         # (20,128)
    xG = ctrl_Xs[:20, g_col]                      # (20,) ctrl 中 G 的表达
    pred = -(xG[:, None] * J)                     # (20,128) 敲低到0的一阶位移
    pred_mean = pred.mean(0)
    # 答案侧：held-out 扰动细胞
    de, n_treat = gene_de(g)
    treat_mask = obs["condition"].isin([c for c in obs["condition"].unique()
                                        if str(c) in (f"{g}+ctrl", f"ctrl+{g}")]).values
    treat_embed = embed(to_scim_batch(X[treat_mask][:150]))
    actual = treat_embed.mean(0) - ctrl_embed.mean(0)   # (128,)
    cos = float(pred_mean @ actual / (np.linalg.norm(pred_mean) * np.linalg.norm(actual) + 1e-12))
    # 随机 null：1000 个随机 128 方向
    null = rng.normal(size=(2000, 128))
    null_cos = (null @ actual) / (np.linalg.norm(null, axis=1) * np.linalg.norm(actual) + 1e-12)
    p = float((null_cos >= cos).mean())
    mag_ratio = float(np.linalg.norm(pred_mean) / (np.linalg.norm(actual) + 1e-12))
    cosines[g] = cos
    direction_rows.append(dict(gene=g, cosine=cos, null_p=p, magnitude_ratio=mag_ratio, n_treat=n_treat))
    print(f"{g:8s} | {cos:+18.4f} | {p:12.4f} | {mag_ratio:16.4f}  (n_treat={n_treat})")
print("方向相似性、null p 值和幅度比仅作为探索指标，不据此确认预测能力或级联机制。")

# ============ 测试B：ctrl 重要性(归因) 预测该基因的扰动下游影响 ============
print("\n===== 测试B：ctrl 状态基因重要性 能否预测其敲低的下游影响 =====")
rows = []
for g in single_genes:
    gi = int(np.where(gene_names == g)[0][0])
    de, n_treat = gene_de(g)
    if n_treat < 10:
        continue
    impact = float(np.abs(np.delete(de, gi)).sum())   # 敲低 G 后除 G 外全转录组 |DE| 之和
    imp = float(mean_abs_attr[gi])                     # 预测：G 在 ctrl 的重要性
    expr = float(ctrl_X[:, gi].mean())                 # baseline：G 的 ctrl 平均表达
    rows.append((g, imp, expr, impact, n_treat))
rows = np.array([(r[1], r[2], r[3]) for r in rows], dtype=float)
imp, expr, impact = rows[:, 0], rows[:, 1], rows[:, 2]

def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    bx = np.polyfit(rz, rx, 1); res_x = rx - (bx[0]*rz + bx[1])
    by = np.polyfit(rz, ry, 1); res_y = ry - (by[0]*rz + by[1])
    return spearmanr(res_x, res_y)[0]

r_attr, p_attr = spearmanr(imp, impact)
r_expr, p_expr = spearmanr(expr, impact)
r_partial = partial_spearman(imp, impact, expr)
print(f"归因重要性 vs 下游影响: Spearman r={r_attr:+.3f} (p={p_attr:.3g}), n={len(imp)}")
print(f"表达量      vs 下游影响: Spearman r={r_expr:+.3f} (p={p_expr:.3g})")
print(f"归因(控表达后) vs 影响: partial Spearman r={r_partial:+.3f}")
print("相关性及控制表达后的相关性是探索证据，不等于因果或泛化预测能力验收。")
write_exploration_report(os.path.join(DATA_DIR, "predictive_exploration.json"), {
    "direction_results": direction_rows,
    "impact_metrics": {"n_genes": len(imp), "r_attr": float(r_attr), "p_attr": float(p_attr),
                       "r_expr": float(r_expr), "p_expr": float(p_expr), "r_partial": float(r_partial)},
})
