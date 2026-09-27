"""为网页工具准备 Williams 数据：SCimilarity 编码 + 对齐表达矩阵(CSR) + 标签
产物: ws_expr.npz / ws_embeddings.npy / ws_web_meta.csv / ws_gene_max.npy / ws_umap.npy
"""
import os, sys
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from paths import SIG_DIR, MODEL_DIR, DATA_DIR

sys.path.insert(0, os.path.join(SIG_DIR, "src"))
from SIGnature.models.scimilarity import SCimilarityWrapper
from SIGnature.utils import align_dataset, lognorm_counts

DATA = DATA_DIR
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", required=True, help="Williams AnnData input path; see docs/vct.md")
WS_H5AD = parser.parse_args().input
os.makedirs(DATA, exist_ok=True)

print("1/4 加载 Williams processed.h5ad + SCimilarity 对齐 ...")
adata = sc.read_h5ad(WS_H5AD)
adata.var_names_make_unique()
wrapper = SCimilarityWrapper(model_path=MODEL_DIR)
adata = align_dataset(adata, wrapper.gene_order, gene_overlap_threshold=500)
adata.layers["counts"] = adata.X.copy()
lognorm_counts(adata)
X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
sp.save_npz(os.path.join(DATA, "ws_expr.npz"), sp.csr_matrix(X))
print("   ws_expr.npz 已存:", X.shape)

print("2/4 SCimilarity 编码 + 分组标签 ...")
wrapper.model.eval()
import torch
emb = []
with torch.no_grad() if False else torch.inference_mode():
    for i in range(0, X.shape[0], 2000):
        batch = torch.tensor(np.asarray(X[i:i+2000].todense()), dtype=torch.float32)
        emb.append(wrapper.model(batch).numpy())
emb = np.vstack(emb).astype(np.float32)
np.save(os.path.join(DATA, "ws_embeddings.npy"), emb)
print("   ws_embeddings.npy 已存:", emb.shape)

groups = adata.obs["condition"].astype(str).values
meta = pd.DataFrame({"cell_type": groups})
meta.to_csv(os.path.join(DATA, "ws_web_meta.csv"), index=False)
print("   标签:", groups.tolist().count("WS"), "WS /", groups.tolist().count("CTRL"), "CTRL")

print("3/4 UMAP ...")
import umap
xy = umap.UMAP(n_neighbors=15, min_dist=0.5, random_state=42).fit_transform(emb)
np.save(os.path.join(DATA, "ws_umap.npy"), xy)

print("4/4 各基因最大表达（滑块范围）...")
maxv = X.max(axis=0).toarray().ravel()
if maxv.shape != (X.shape[1],) or not np.isfinite(maxv).all():
    raise ValueError("gene maxima must be a finite numeric vector matching the gene count")
np.save(os.path.join(DATA, "ws_gene_max.npy"), maxv, allow_pickle=False)
print("✅ 完成")
