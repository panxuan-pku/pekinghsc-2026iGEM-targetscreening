"""M1+M2: PBMC 3k 工具可行性测试数据 → SCimilarity embedding → UMAP。
显式选择已确认注释或计算分组；完整产物写入新的运行目录。
"""
import argparse
import hashlib
from paths import OUTPUT_DIR
from datetime import datetime
import json
import os, sys, time
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    from .input_checks import check_expression
except ImportError:
    from input_checks import check_expression


def validate_metadata(meta, cell_ids=None):
    """Validate supplied annotations and, when available, align by cell ID."""
    required = {"cell", "cell_type"}
    missing = required - set(meta.columns)
    if missing or meta.empty:
        raise ValueError(f"注释表需有非空 cell、cell_type 列；缺少: {sorted(missing)}")
    columns = ["cell"] + (["cluster"] if "cluster" in meta else []) + ["cell_type"]
    meta = meta[columns].copy().astype("string")
    for column in columns:
        if meta[column].isna().any() or meta[column].str.strip().eq("").any():
            raise ValueError(f"注释表 {column} 含空值")
    if meta["cell"].duplicated().any():
        raise ValueError("注释表 cell ID 重复，不能确定一一对应关系")
    if cell_ids is not None:
        cells = pd.Index(cell_ids)
        if (cells.empty or cells.hasnans or not cells.is_unique
                or any(not isinstance(cell, str) or not cell.strip() for cell in cells)):
            raise ValueError("表达矩阵的 cell ID 必须非空且唯一")
        indexed = meta.set_index("cell")
        missing = cells.difference(indexed.index)
        extra = indexed.index.difference(cells)
        if len(missing) or len(extra):
            raise ValueError(f"注释与表达细胞不匹配：缺少 {len(missing)} 个 {list(missing[:5])}；"
                             f"多余 {len(extra)} 个 {list(extra[:5])}")
        meta = indexed.loc[cells].reset_index(names="cell")
    return meta


def save_run(output, raw, adata, emb, meta, umap_coords, metadata_source, *, label_policy="supplied annotations; no automatic clustering or cell-type assignment", provenance=None):
    """Validate and publish one complete PBMC bundle without replacing data/."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError(f"输出目录已存在，请指定新的 --output: {output}")
    meta = validate_metadata(meta, adata.obs_names)
    validate_metadata(meta, raw.obs_names)
    genes = list(adata.var_names)
    check_expression(adata.X, genes, min_cells=2)
    if (emb.ndim != 2 or emb.shape[0] != adata.n_obs or emb.shape[1] == 0
            or umap_coords.shape != (adata.n_obs, 2)):
        raise ValueError("表达、embedding、UMAP 的细胞数量或维度不一致")
    if not np.isfinite(emb).all() or not np.isfinite(umap_coords).all():
        raise ValueError("embedding/UMAP 含 NaN/Inf，未发布结果")

    info = {
        "n_cells": int(adata.n_obs), "n_genes": int(adata.n_vars),
        "embedding_dim": int(emb.shape[1]),
        "cell_types": meta.cell_type.unique().tolist(),
        "purpose": "PBMC tool feasibility test",
        "metadata_source": str(Path(metadata_source).resolve()) if metadata_source else None,
        "label_policy": label_policy,
        "provenance": provenance or {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pbmc-preparation-", dir=output.parent) as tmp:
        stage = Path(tmp)
        raw.write_h5ad(stage / "pbmc3k_raw.h5ad")
        sp.save_npz(stage / "expr_aligned.npz", sp.csr_matrix(adata.X))
        np.save(stage / "embeddings.npy", emb, allow_pickle=False)
        np.save(stage / "umap.npy", umap_coords, allow_pickle=False)
        meta.to_csv(stage / "meta.csv", index=False)
        (stage / "gene_order.txt").write_text("\n".join(genes) + "\n", encoding="utf-8")
        (stage / "pbmc_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        if output.exists() or output.is_symlink():
            raise ValueError(f"输出目录已存在，请指定新的 --output: {output}")
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    labels = parser.add_mutually_exclusive_group(required=True)
    labels.add_argument("--metadata", type=Path,
                        help="已确认来源的 CSV：cell、cell_type 必需，cluster 可选；不自动沿用历史标签")
    labels.add_argument("--computed-groups", type=int, metavar="N",
                        help="Group fresh PBMC embeddings with KMeans; labels are NOT confirmed cell types")
    parser.add_argument("--input", type=Path, help="Optional raw PBMC h5ad; otherwise download the public PBMC3k dataset")
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_DIR) /
                        ("pbmc_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")),
                        help="新的运行目录，不覆盖现有目录或网页 data/")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error(f"输出目录已存在，请指定新的 --output: {args.output}")
    if args.computed_groups is not None and not 2 <= args.computed_groups <= 30:
        parser.error("--computed-groups must be between 2 and 30")
    meta = validate_metadata(pd.read_csv(args.metadata, dtype=str, keep_default_na=False)) if args.metadata else None

    import scanpy as sc
    print("1/5 获取 PBMC 3k 测试数据 ...")
    from paths import WORKSPACE
    sc.settings.datasetdir = Path(WORKSPACE) / "downloads"
    sc.settings.datasetdir.mkdir(parents=True, exist_ok=True)
    raw = sc.read_h5ad(args.input) if args.input else sc.datasets.pbmc3k()
    if args.computed_groups and args.computed_groups >= raw.n_obs:
        parser.error("--computed-groups must be smaller than the number of cells")
    if meta is not None:
        meta = validate_metadata(meta, raw.obs_names)
    print(f"   细胞 {raw.n_obs}, 基因 {raw.n_vars}；注释按 cell ID 对齐")

    # Only load the model after annotation/ID validation. Keep raw counts intact.
    import torch
    from paths import SIG_DIR, MODEL_DIR
    sys.path.insert(0, os.path.join(SIG_DIR, "src"))
    from SIGnature.models.scimilarity import SCimilarityWrapper
    from SIGnature.utils import align_dataset, lognorm_counts

    print("2/5 对齐 SCimilarity 基因空间 ...")
    wrapper = SCimilarityWrapper(model_path=MODEL_DIR)
    overlap = int(sum(raw.var_names.isin(wrapper.gene_order)))
    print(f"   基因重合: {overlap}/{len(raw.var_names)}")
    adata = align_dataset(raw.copy(), wrapper.gene_order, gene_overlap_threshold=500)
    if meta is not None:
        meta = validate_metadata(meta, adata.obs_names)
    adata.layers["counts"] = adata.X.copy()
    lognorm_counts(adata)
    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)

    print("3/5 计算 embedding (128 维) ...")
    model = wrapper.model.eval()
    embs = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, X.shape[0], 512):
            xb = X[i:i + 512].toarray().astype(np.float32)
            embs.append(model(torch.tensor(xb)).numpy())
    emb = np.vstack(embs)
    print(f"   {emb.shape} ({time.time() - t0:.0f}s)")

    label_policy = "supplied annotations; no automatic clustering or cell-type assignment"
    if args.computed_groups:
        from sklearn.cluster import KMeans
        groups = KMeans(n_clusters=args.computed_groups, random_state=42, n_init=10).fit_predict(emb)
        meta = pd.DataFrame({"cell": adata.obs_names, "cluster": groups.astype(str),
                             "cell_type": [f"Computed group {i + 1}" for i in groups]})
        label_policy = f"computed groups, not confirmed cell types; KMeans(k={args.computed_groups}, random_state=42, n_init=10) on SCimilarity embeddings"
    print("4/5 计算 UMAP ...")
    import umap
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.5, random_state=42).fit(emb)
    umap_coords = reducer.embedding_
    print(f"   UMAP: {umap_coords.shape}")

    print("5/5 验证并发布完整测试数据 ...")
    raw_path = args.input or sc.settings.datasetdir / "pbmc3k_raw.h5ad"
    provenance = {"input": str(raw_path.resolve()), "input_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                  "source": str(args.input) if args.input else "https://falexwolf.de/data/pbmc3k_raw.h5ad",
                  "model_dir": MODEL_DIR, "embedding": "SCimilarity", "umap_random_state": 42}
    save_run(args.output, raw, adata, emb, meta, umap_coords, args.metadata,
             label_policy=label_policy, provenance=provenance)
    print(f"✅ 数据管线完成: {args.output}（未替换网页 data/）")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, MemoryError) as exc:
        print(f"PBMC preparation failed: {str(exc) or 'insufficient memory'}", file=sys.stderr)
        sys.exit(2)
