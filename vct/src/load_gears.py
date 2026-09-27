"""M14: 加载 GEARS 预训练模型并验证（Norman Perturb-seq）
数据/权重来源见 ATTRIBUTIONS.md
"""
import os, sys, pickle
import numpy as np
import pandas as pd

# pandas 2.x 移除了 Series.nonzero()，scipy 稀疏矩阵布尔索引依赖它 → 兼容补丁
if not hasattr(pd.Series, "nonzero"):
    pd.Series.nonzero = lambda self: np.flatnonzero(self.values)

from gears import PertData, GEARS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from paths import VCT_ROOT, GEARS_DATA_DIR, GEARS_CKPT_DIR, DATA_DIR

ROOT = VCT_ROOT
DATA_DIR = GEARS_DATA_DIR
CKPT = GEARS_CKPT_DIR

print("1/4 PertData 初始化（使用本地 gene2go）...")
pert_data = PertData(DATA_DIR, default_pert_graph=False)
pert_data.load(data_name="norman")
print("   adata:", pert_data.adata.shape)

print("2/4 划分 + dataloader ...")
pert_data.prepare_split(split="simulation", seed=1)
pert_data.get_dataloader(batch_size=32, test_batch_size=128)

# 兼容修复：GEARS 内部对稀疏矩阵做布尔行索引与新版 scipy 不兼容 → 转稠密 float32
# 注意：adata 此时是视图（view），需先实体化否则赋值不生效
import scipy.sparse as sp
pert_data.adata = pert_data.adata.to_memory()
if sp.issparse(pert_data.adata.X):
    pert_data.adata.X = pert_data.adata.X.toarray().astype(np.float32)
    print("   X 已转稠密:", type(pert_data.adata.X), pert_data.adata.X.shape)

print("3/4 加载预训练权重（matthewshu/gears-norman）...")
gears_model = GEARS(pert_data, device="cpu")
gears_model.load_pretrained(CKPT)
print("   模型加载成功")

print("4/4 快速预测测试：敲低 CEBPA ...")
pred = gears_model.predict([["CEBPA"]])
for k, v in pred.items():
    v = np.asarray(v)
    print(f"   {k}: shape={v.shape}, mean={v.mean():.3f}")
np.save(os.path.join(DATA_DIR, "gears_test_pred.npy"),
        np.asarray(pred[list(pred.keys())[0]]))
print("✅ M14 完成")
