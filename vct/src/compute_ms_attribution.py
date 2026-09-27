"""M9+M10: MS 少突胶质细胞数据集 → 真实 attribution 矩阵 → 差异归因
病变(MS) vs 健康(normal) 同细胞类型对比，输出候选靶基因清单。
自有工作（见 ATTRIBUTIONS.md）
"""
import argparse
from paths import DATA_DIR, OUTPUT_DIR
import os, sys, time
from datetime import datetime
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = DATA_DIR
N_STEPS = 16   # IG 积分步数（32→16 提速一倍，精度损失小；DBTL 记录）
BATCH = 64


def differential_table(X, attr_all, groups, gene_order):
    """Keep the existing differential calculation; sort by its actual column."""
    if not {"MS", "normal"}.issubset(set(groups)):
        raise ValueError("差异归因需要同时存在 MS 和 normal 样本")
    ms_mask = groups == "MS"
    nm_mask = groups == "normal"
    A = attr_all.astype(np.float32)
    ms_mean = A[ms_mask].mean(axis=0)
    nm_mean = A[nm_mask].mean(axis=0)
    # 效应量：均值差 / 合并标准差（Cohen's d 风格）
    pooled_std = np.sqrt((A[ms_mask].var(axis=0) + A[nm_mask].var(axis=0)) / 2) + 1e-6
    diff = ms_mean - nm_mean
    effect = diff / pooled_std

    return pd.DataFrame({
        "gene": gene_order,
        "attr_MS": ms_mean, "attr_normal": nm_mean,
        "diff": diff, "effect_size": effect,
        "expr_MS": X[ms_mask].mean(axis=0), "expr_normal": X[nm_mask].mean(axis=0),
    }).sort_values("effect_size", key=abs, ascending=False)


def save_run(output, attr_all, emb, meta, table, gene_order):
    """Publish a complete new bundle, never overwrite the current Web data."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError(f"输出目录已存在，请指定新的 --output: {output}")
    if (attr_all.ndim != 2 or emb.ndim != 2 or len(emb) != len(meta)
            or attr_all.shape != (len(meta), len(gene_order))
            or len(set(gene_order)) != len(gene_order)
            or len(table) != len(gene_order)
            or set(table["gene"]) != set(gene_order)):
        raise ValueError("归因、embedding、样本和基因顺序不一致")
    for values in (attr_all, emb, table.select_dtypes(include="number").to_numpy()):
        for i in range(0, len(values), 1024):
            if not np.isfinite(values[i:i+1024]).all():
                raise ValueError("输出包含 NaN/Inf，未发布结果")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ms-attribution-", dir=output.parent) as tmp:
        stage = Path(tmp)
        np.save(stage / "ms_attribution_fp16.npy", attr_all)
        np.save(stage / "ms_embeddings.npy", emb)
        meta.to_csv(stage / "ms_meta.csv", index=False)
        table.to_csv(stage / "ms_differential_attribution.csv", index=False)
        (stage / "gene_order.txt").write_text("\n".join(gene_order) + "\n", encoding="utf-8")
        if output.exists() or output.is_symlink():
            raise ValueError(f"输出目录已存在，请指定新的 --output: {output}")
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_DIR) /
                        ("ms_attribution_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")),
                        help="新的运行目录；不会覆盖现有目录或 data/")
    parser.add_argument("--input", type=Path, required=True, help="MS oligodendrocyte AnnData input")
    args = parser.parse_args()
    output = args.output
    if output.exists() or output.is_symlink():
        parser.error(f"输出目录已存在，请指定新的 --output: {output}")

    import scanpy as sc
    import scipy.sparse as sp
    import torch
    from captum.attr import IntegratedGradients
    from paths import SIG_DIR, MODEL_DIR
    sys.path.insert(0, os.path.join(SIG_DIR, "src"))
    from SIGnature.models.scimilarity import SCimilarityWrapper
    from SIGnature.utils import align_dataset, lognorm_counts

    print("1/6 加载 MS 数据集...")
    adata = sc.read_h5ad(args.input)
    # CELLxGENE h5ad 的 var_names 是 Ensembl ID，映射为基因符号（SCimilarity 用符号）
    adata.var_names = adata.var["feature_name"].astype(str)
    adata.var_names_make_unique()
    print(f"   {adata.n_obs} 细胞 x {adata.n_vars} 基因")
    print("   obs 列:", list(adata.obs.columns)[:15])
    # 找疾病标签列
    dcol = None
    for cand in ["disease", "Disease", "diagnosis", "condition", "group"]:
        if cand in adata.obs.columns:
            dcol = cand; break
    if dcol is None:
        # 打印所有列的唯一值帮助定位
        for c in adata.obs.columns:
            u = adata.obs[c].unique()
            if len(u) <= 8:
                print(f"   {c}: {list(u)}")
        raise SystemExit("未找到疾病列，请检查上面输出")
    adata.obs["group"] = adata.obs[dcol].astype(str).map(
        lambda s: "MS" if "sclerosis" in s.lower() else ("normal" if "normal" in s.lower() or "control" in s.lower() else s))
    print("   分组:", adata.obs["group"].value_counts().to_dict())

    for column in ("Lesion", "donor_id"):
        if column not in adata.obs:
            raise ValueError(f"缺少样本字段: {column}")
    if not {"MS", "normal"}.issubset(set(adata.obs["group"])):
        raise ValueError("差异归因需要同时存在 MS 和 normal 样本")

    print("2/6 对齐基因空间 + lognorm...")
    wrapper = SCimilarityWrapper(model_path=MODEL_DIR)
    overlap = sum(adata.var_names.isin(wrapper.gene_order))
    print(f"   基因重合: {overlap}")
    adata = align_dataset(adata, wrapper.gene_order, gene_overlap_threshold=500)
    # CELLxGENE h5ad 的 X 通常是原始 counts
    adata.layers["counts"] = adata.X.copy()
    lognorm_counts(adata)
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X, dtype=np.float32)
    groups = adata.obs["group"].values

    print("3/6 计算 embedding（全细胞）...")
    model = wrapper.model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            embs.append(model(torch.tensor(X[i:i+512], dtype=torch.float32)).numpy())
    emb = np.vstack(embs)

    print(f"4/6 计算 attribution 矩阵（IG, n_steps={N_STEPS}, batch={BATCH}）...")
    def sum_output(x):
        return model(x).sum(dim=1)
    ig = IntegratedGradients(sum_output)
    attr_all = np.zeros((len(X), X.shape[1]), dtype=np.float16)  # fp16 省内存
    t0 = time.time()
    for i in range(0, len(X), BATCH):
        xb = torch.tensor(X[i:i+BATCH], dtype=torch.float32, requires_grad=True)
        bl = torch.zeros_like(xb)
        attr_all[i:i+BATCH] = ig.attribute(xb, baselines=bl, n_steps=N_STEPS).detach().numpy().astype(np.float16)
        if (i // BATCH) % 20 == 0:
            done = i + BATCH
            rate = done / (time.time() - t0)
            eta = (len(X) - done) / rate / 60
            print(f"   {done}/{len(X)} 细胞 ({rate:.0f}/s, ETA {eta:.1f} min)")
    meta = pd.DataFrame({"cell": adata.obs_names, "group": groups,
                  "lesion": adata.obs["Lesion"].astype(str).values,
                  "donor": adata.obs["donor_id"].astype(str).values})

    print("5/6 差异归因：MS vs normal ...")
    df = differential_table(X, attr_all, groups, wrapper.gene_order)
    save_run(output, attr_all, emb, meta, df, wrapper.gene_order)

    print("6/6 结果预览（|效应量| Top 20）:")
    print(df.head(20)[["gene","attr_MS","attr_normal","diff","effect_size","expr_MS","expr_normal"]].to_string(index=False))
    print(f"\n✅ M9+M10 完成: {output}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, MemoryError) as exc:
        print(f"MS attribution failed: {str(exc) or 'insufficient memory'}", file=sys.stderr)
        sys.exit(2)
