"""M15+M16: v1→v2 完整闭环（一次跑完）
流程: GEARS 因果预测 → 与真实扰动细胞对比验证 → 映射到 SCimilarity 空间 →
      重新 embedding + attribution → 对比扰动前后变化
自有工作（见 ATTRIBUTIONS.md）
"""
import os, sys, json
import numpy as np
import pandas as pd
import scipy.sparse as sp

if not hasattr(pd.Series, "nonzero"):
    pd.Series.nonzero = lambda self: np.flatnonzero(self.values)

from gears import PertData, GEARS

from paths import VCT_ROOT, SIG_DIR, MODEL_DIR, GEARS_DATA_DIR, GEARS_CKPT_DIR, DATA_DIR

sys.path.insert(0, os.path.join(SIG_DIR, "src"))
from SIGnature.models.scimilarity import SCimilarityWrapper

TARGET = "CEBPA"  # 髓系分化关键转录因子，Norman 中的经典单基因敲低

print("① 加载 GEARS + Norman 数据 ...")
pert_data = PertData(GEARS_DATA_DIR, default_pert_graph=False)
pert_data.load(data_name="norman")
pert_data.prepare_split(split="simulation", seed=1)
pert_data.get_dataloader(batch_size=32, test_batch_size=128)
pert_data.adata = pert_data.adata.to_memory()
if sp.issparse(pert_data.adata.X):
    pert_data.adata.X = pert_data.adata.X.toarray().astype(np.float32)
gm = GEARS(pert_data, device="cpu")
gm.load_pretrained(GEARS_CKPT_DIR)
print("   模型与数据就绪")

print(f"② GEARS 预测: {TARGET} 敲低 ...")
pred = gm.predict([[TARGET]])
pred_vec = np.asarray(pred[TARGET])          # 预测的扰动后表达谱 (5045,)
ctrl_vec = gm.ctrl_expression.cpu().numpy()  # 对照平均表达 (5045,)

print("③ 验证: 预测 vs Norman 中真实敲低细胞 ...")
gene_names = pert_data.adata.var["gene_name"].values
adata = pert_data.adata
real_mask = adata.obs.condition.values == TARGET
if real_mask.sum() == 0:
    # 找包含 CEBPA 的条件
    cands = [c for c in adata.obs.condition.unique() if TARGET in str(c)]
    print("   含 CEBPA 的条件:", cands)
    real_mask = adata.obs.condition.values == cands[0]
real_vec = adata.X[real_mask].mean(axis=0)
r_all = np.corrcoef(pred_vec, real_vec)[0,1]
# top 差异基因上的相关性（更严格的指标）
de_idx = np.argsort(-np.abs(real_vec - ctrl_vec))[:50]
r_de = np.corrcoef(pred_vec[de_idx], real_vec[de_idx])[0,1]
print(f"   真实敲低细胞数: {real_mask.sum()} | 全基因 pearson r = {r_all:.3f} | Top50 DE 基因 r = {r_de:.3f}")

print("④ 映射到 SCimilarity 28231 基因空间并重新 embedding ...")
wrapper = SCimilarityWrapper(model_path=MODEL_DIR)
wrapper.model.eval()
gidx = {g: i for i, g in enumerate(wrapper.gene_order)}
def to_scim(vec5045):
    x = np.zeros(len(wrapper.gene_order), dtype=np.float32)
    hit = 0
    for j, g in enumerate(gene_names):
        if g in gidx:
            x[gidx[g]] = vec5045[j]; hit += 1
    return x, hit
x_ctrl, h1 = to_scim(ctrl_vec)
x_pred, h2 = to_scim(pred_vec)
print(f"   基因映射命中率: {h2}/{len(gene_names)}")
import torch
with torch.no_grad():
    e_ctrl = wrapper.model(torch.tensor(x_ctrl[None,:])).numpy()[0]
    e_pred = wrapper.model(torch.tensor(x_pred[None,:])).numpy()[0]
disp = float(np.linalg.norm(e_pred - e_ctrl))
print(f"   embedding 位移（对照 → 预测敲低）: {disp:.4f}")

print("⑤ 对预测结果重新归因（IG）...")
from captum.attr import IntegratedGradients
def sum_out(x): return wrapper.model(x).sum(dim=1)
ig = IntegratedGradients(sum_out)
def attr_of(x):
    t = torch.tensor(x[None,:], dtype=torch.float32, requires_grad=True)
    return ig.attribute(t, baselines=torch.zeros_like(t), n_steps=16).detach().numpy()[0]
a_ctrl = attr_of(x_ctrl)
a_pred = attr_of(x_pred)
delta_attr = a_pred - a_ctrl
top_change = np.argsort(-np.abs(delta_attr))[:15]
rows = [(wrapper.gene_order[i], float(a_ctrl[i]), float(a_pred[i]), float(delta_attr[i])) for i in top_change]
print("   扰动后 attribution 变化 Top15:")
for g, ac, ap, d in rows:
    print(f"   {g:12s} ctrl={ac:+.4f} → pred={ap:+.4f} (Δ{d:+.4f})")

json.dump({
    "target": TARGET, "n_real_kd_cells": int(real_mask.sum()),
    "pearson_all": float(r_all), "pearson_top50DE": float(r_de),
    "gene_map_hit": f"{h2}/{len(gene_names)}",
    "embedding_displacement": disp,
    "top_attr_change": [{"gene": g, "ctrl": ac, "pred": ap, "delta": d} for g, ac, ap, d in rows],
}, open(os.path.join(DATA_DIR, "v2_closed_loop.json"), "w"), indent=2, ensure_ascii=False)
np.savez(os.path.join(DATA_DIR, "v2_vectors.npz"),
         pred_vec=pred_vec, real_vec=real_vec, ctrl_vec=ctrl_vec,
         e_ctrl=e_ctrl, e_pred=e_pred, a_ctrl=a_ctrl, a_pred=a_pred)
print("✅ M15+M16 完成，结果存 data/v2_closed_loop.json")
