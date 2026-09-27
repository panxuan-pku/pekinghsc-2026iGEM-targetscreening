"""历史探索：差分归因与实测差异表达的比较报告，不是科学验收。"""
import os, sys
import numpy as np
import anndata as ad
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from paths import VCT_ROOT, SIG_DIR, MODEL_DIR, GEARS_DATA_DIR, DATA_DIR
from exploration_report import EXPLORATION_NOTICE, write_exploration_report

print(EXPLORATION_NOTICE)

sys.path.insert(0, os.path.join(SIG_DIR, "src"))
from SIGnature.models.scimilarity import SCimilarityWrapper
from captum.attr import IntegratedGradients

print("加载 Norman 数据 ...")
adata = ad.read_h5ad(os.path.join(GEARS_DATA_DIR, "norman", "perturb_processed.h5ad"))
obs = adata.obs
gene_names = np.array(adata.var["gene_name"].astype(str))
X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
ctrl_mask = (obs["condition"] == "ctrl").values
print(f"总细胞 {X.shape[0]}, 基因 {X.shape[1]}, ctrl 细胞 {ctrl_mask.sum()}")

print("加载 SCimilarity ...")
wrapper = SCimilarityWrapper(model_path=os.path.join(SIG_DIR, "model_files/model_files/scimilarity"))
wrapper.model.eval()
gidx = {g: i for i, g in enumerate(wrapper.gene_order)}  # 28231 基因

# 5045 -> 28231 映射
norman_to_scim = [gidx[g] if g in gidx else -1 for g in gene_names]
hit = sum(1 for x in norman_to_scim if x >= 0)
print(f"Norman 5045 基因中 {hit} 个在 SCimilarity 28231 空间")

def to_scim(vec5045):
    x = np.zeros(len(wrapper.gene_order), dtype=np.float32)
    for j, v in enumerate(vec5045):
        k = norman_to_scim[j]
        if k >= 0:
            x[k] = v
    return x

def embed(xs):  # (n,28231)->(n,128)
    with torch.no_grad():
        t = torch.tensor(xs, dtype=torch.float32)
        outs = [wrapper.model(t[i:i+256]).numpy() for i in range(0, len(t), 256)]
    return np.vstack(outs)

sum_out = lambda x: wrapper.model(x).sum(dim=1)
ig = IntegratedGradients(sum_out)

def attr_batch(xs):  # (n,28231)->(n,28231) attribution
    t = torch.tensor(xs, dtype=torch.float32, requires_grad=True)
    return ig.attribute(t, baselines=torch.zeros_like(t), n_steps=16).detach().numpy()

def real_de(treat_mask, n_sample=200):
    """真实差异表达: mean(treat)-mean(ctrl), 全基因"""
    treat = X[treat_mask][:n_sample]
    ctrl = X[ctrl_mask][:n_sample]
    return treat.mean(0) - ctrl.mean(0)   # (5045,)

def diff_attr(treat_mask, n_sample=25):
    """差分归因: mean(attr(treat))-mean(attr(ctrl)), 映射回 5045"""
    idx_t = np.where(treat_mask)[0][:n_sample]
    idx_c = np.where(ctrl_mask)[0][:n_sample]
    a_t = attr_batch(np.stack([to_scim(X[i]) for i in idx_t]))   # (n,28231)
    a_c = attr_batch(np.stack([to_scim(X[i]) for i in idx_c]))
    dattr = a_t.mean(0) - a_c.mean(0)                            # (28231,)
    # 映射回 5045
    back = np.array([dattr[k] if k >= 0 else np.nan for k in norman_to_scim])
    return back

def compare(gene, treat_mask):
    de = real_de(treat_mask)
    dattr = diff_attr(treat_mask)
    # 用非 NaN 基因比较
    valid = ~np.isnan(dattr)
    de_v, da_v = de[valid], dattr[valid]
    # spearman on |值|
    from scipy.stats import spearmanr
    r_signed, _ = spearmanr(de_v, da_v)
    r_abs, _ = spearmanr(np.abs(de_v), np.abs(da_v))
    # top-50 overlap
    k = 50
    top_de = set(np.argsort(-np.abs(de_v))[:k])
    top_da = set(np.argsort(-np.abs(da_v))[:k])
    jacc = len(top_de & top_da) / len(top_de | top_da)
    # 扰动基因本身 G 的排名
    g_pos = np.where(gene_names == gene)[0]
    rank_de = rank_da = None
    if len(g_pos):
        gi = g_pos[0]
        rank_de = int((np.abs(de_v) >= np.abs(de[gi])).sum())
        rank_da = int((np.abs(da_v) >= np.abs(dattr[gi])).sum())
    return dict(r_signed=r_signed, r_abs=r_abs, jacc50=jacc, rank_de=rank_de, rank_da=rank_da,
                n_treat=int(treat_mask.sum()))

# 待验证的单基因扰动
genes = ["KLF1", "CEBPA", "BAK1", "ETS2", "CEBPE"]
results = {}
print("\n{'基因': {'真实DE的|差分归因|Spearman': r, 'top50 Jaccard': j, 'G在DE排名': 1-based, 'G在Δattr排名': 1-based}}")
for g in genes:
    conds = [c for c in obs["condition"].unique() if str(c) in (f"{g}+ctrl", f"ctrl+{g}")]
    if not conds:
        print(f"{g:8s} 无单基因扰动条件"); continue
    treat_mask = obs["condition"].isin(conds).values
    r = compare(g, treat_mask)
    results[g] = r
    print(f"{g:8s} n={r['n_treat']:5d} | r(|DE|,|Δattr|)={r['r_abs']:+.3f} | r(DE,Δattr)={r['r_signed']:+.3f} | "
          f"Jaccard50={r['jacc50']:.3f} | G在DE排名 {r['rank_de']} | G在Δattr排名 {r['rank_da']}")
write_exploration_report(os.path.join(DATA_DIR, "attribution_vs_perturb_exploration.json"), results)
