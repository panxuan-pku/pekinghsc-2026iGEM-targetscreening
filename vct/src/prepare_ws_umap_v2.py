"""方案 A+C：重算 Williams UMAP（n_neighbors=30, min_dist=0.1 更分明的簇）+ 导出 leiden cluster 标签
覆盖: ws_umap.npy / ws_web_meta.csv（增加 cluster 列）
"""
import os, sys
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from paths import DATA_DIR

DATA = DATA_DIR
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", required=True, help="Williams AnnData input path; see docs/vct.md")
WS_H5AD = parser.parse_args().input
os.makedirs(DATA, exist_ok=True)

print("1/3 加载 h5ad 取标签 ...")
adata = sc.read_h5ad(WS_H5AD, backed="r")
groups = adata.obs["condition"].astype(str).values
clusters = adata.obs["cluster"].astype(str).values
n = adata.n_obs

print("2/3 加载已有嵌入，重算 UMAP（n_neighbors=30, min_dist=0.1）...")
emb = np.load(os.path.join(DATA, "ws_embeddings.npy"))
assert emb.shape[0] == n, f"嵌入数 {emb.shape[0]} != 细胞数 {n}"
import umap
xy = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(emb)
np.save(os.path.join(DATA, "ws_umap.npy"), xy)
print("   ws_umap.npy 已覆盖:", xy.shape, "| x std", round(xy[:,0].std(),2), "y std", round(xy[:,1].std(),2))

print("3/3 meta.csv 加 cluster 列 ...")
meta = pd.DataFrame({"cell_type": groups, "cluster": clusters})
meta.to_csv(os.path.join(DATA, "ws_web_meta.csv"), index=False)
print("   cluster 分布:", dict(pd.Series(clusters).value_counts().sort_index()))
print("✅ 完成")
