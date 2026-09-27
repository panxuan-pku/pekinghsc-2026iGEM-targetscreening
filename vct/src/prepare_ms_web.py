"""为网页工具准备 MS 数据：对齐表达矩阵(CSR) + UMAP 坐标"""
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = DATA_DIR

print("1/4 加载 + 对齐 MS 数据 ...")
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", required=True, help="MS oligodendrocyte AnnData input")
args = parser.parse_args()
os.makedirs(DATA, exist_ok=True)
adata = sc.read_h5ad(args.input)
adata.var_names = adata.var["feature_name"].astype(str)
adata.var_names_make_unique()
wrapper = SCimilarityWrapper(model_path=MODEL_DIR)
adata = align_dataset(adata, wrapper.gene_order, gene_overlap_threshold=500)
adata.layers["counts"] = adata.X.copy()
lognorm_counts(adata)
X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
sp.save_npz(os.path.join(DATA, "ms_expr.npz"), sp.csr_matrix(X))
print("   ms_expr.npz 已存:", X.shape)

print("2/4 分组标签 ...")
groups = adata.obs["disease"].astype(str).map(
    lambda s: "MS" if "sclerosis" in s.lower() else "normal")
meta = pd.DataFrame({"cell_type": groups.values})
meta.to_csv(os.path.join(DATA, "ms_web_meta.csv"), index=False)
print("   ", groups.value_counts().to_dict())

print("3/4 UMAP ...")
emb = np.load(os.path.join(DATA, "ms_embeddings.npy"))
import umap
xy = umap.UMAP(n_neighbors=15, min_dist=0.5, random_state=42).fit_transform(emb)
np.save(os.path.join(DATA, "ms_umap.npy"), xy)

print("4/4 各基因最大表达（滑块范围）...")
maxv = X.max(axis=0).toarray().ravel()
if maxv.shape != (X.shape[1],) or not np.isfinite(maxv).all():
    raise ValueError("gene maxima must be a finite numeric vector matching the gene count")
np.save(os.path.join(DATA, "ms_gene_max.npy"), maxv, allow_pickle=False)
print("✅ 完成")
