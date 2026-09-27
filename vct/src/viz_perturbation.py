"""CLI 可视化脚本：CIPHER + 基线 → 出版物级 SVG/PNG
用法: python vct/src/viz_perturbation.py --gene GTF2I --direction oe --output outputs/
产出: perturbation_heatmap.svg, celltype_response.svg, baseline_comparison.svg
"""
import os, sys, argparse, json
import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(__file__))
from paths import DATA_DIR, OUTPUT_DIR
from cipher_engine import CipherEngine
from linear_baseline import LinearBaseline


plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "figure.dpi": 150, "savefig.dpi": 300,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})
# 尝试使用系统中文友好字体，回退 DejaVu Sans
for _font in ["Arial", "Helvetica", "DejaVu Sans"]:
    try:
        plt.rcParams["font.sans-serif"] = [_font]
        break
    except Exception:
        pass

STYLE = {"up": "#2E7D32", "down": "#1565C0", "target": "#CC0000", "baseline": "#999999"}


def load_data(dataset: str = "ws", max_memory_gib=2.0):
    """加载表达矩阵和基因列表。"""
    if dataset == "ws":
        expression, metadata = "ws_expr.npz", "ws_web_meta.csv"
    elif dataset == "pbmc":
        expression, metadata = "expr_aligned.npz", "meta.csv"
    else:
        raise ValueError(f"未知数据集: {dataset}")

    if not np.isfinite(max_memory_gib) or max_memory_gib <= 0:
        raise ValueError("--max-memory-gib must be positive and finite")
    path = os.path.join(DATA_DIR, expression)
    # Read only the small shape member, before decompressing expression values.
    with np.load(path, allow_pickle=False) as archive:
        n_cells, n_genes = map(int, archive["shape"])
    if n_cells < 2 or n_genes < 1:
        raise ValueError("expression requires at least two cells and one gene")
    # Budget for full/selected/centered matrices and covariance temporaries,
    # conservatively at float64. This is an estimate, not an OS memory limit.
    estimated_gib = (4 * n_cells * n_genes + 5 * n_genes * n_genes) * 8 / 1024**3
    if estimated_gib > max_memory_gib:
        raise ValueError(f"estimated dense CIPHER working memory {estimated_gib:.2f} GiB "
                         f"exceeds budget {max_memory_gib:g} GiB ({n_cells} cells x {n_genes} genes). "
                         "No expression matrix was loaded. Use the Web HVG workflow or a suitable machine; "
                         "raise --max-memory-gib only after checking available RAM.")
    X = sp.load_npz(path).toarray()
    meta = pd.read_csv(os.path.join(DATA_DIR, metadata))

    # 基因顺序：从数据维度推断，或从 gene_order.txt 读取
    go_path = os.path.join(DATA_DIR, "gene_order.txt")
    if os.path.exists(go_path):
        with open(go_path) as f:
            gene_list = f.read().strip().split("\n")
    else:
        gene_list = [f"G{i}" for i in range(X.shape[1])]

    return X, meta, gene_list


def plot_heatmap(delta, gene_list, target_gene, direction, output_path):
    """扰动响应热图：Top30 |Δ| 基因的水平条形图。"""
    top_idx = np.argsort(-np.abs(delta))[:30]
    genes = [gene_list[i] for i in top_idx]
    values = [delta[i] for i in top_idx]
    colors = [STYLE["target"] if g == target_gene else (STYLE["up"] if v > 0 else STYLE["down"])
              for g, v in zip(genes, values)]

    fig, ax = plt.subplots(figsize=(8, 7))
    y_pos = range(len(genes))
    ax.barh(y_pos, values, color=colors, height=0.7, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(genes, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=1)

    direction_label = "敲低→0" if direction == "ko" else "过表达→2×"
    ax.set_xlabel("预测 Δx (lognorm 尺度)")
    ax.set_title(f"CIPHER 预测: {target_gene} {direction_label}\nTop30 |Δ| 基因")
    ax.tick_params(axis="x", labelsize=8)

    # 图例
    from matplotlib.patches import Patch
    legend = [
        Patch(color=STYLE["up"], label="上调"),
        Patch(color=STYLE["down"], label="下调"),
        Patch(color=STYLE["target"], label="目标基因"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")


def plot_celltype_response(ct_rows, target_gene, output_path):
    """细胞类型响应排序条形图。"""
    fig, ax = plt.subplots(figsize=(8, max(4, len(ct_rows) * 0.35)))
    y_pos = range(len(ct_rows))
    labels = [f"{r['cell_type']} (n={r['n_cells']})" for r in ct_rows]
    values = [r["response"] * 1000 for r in ct_rows]  # ×1000 便于阅读
    colors = ["#E65100" if r["target_expr"] > 0.5 else "#78909C" for r in ct_rows]

    ax.barh(y_pos, values, color=colors, height=0.6, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("mean|Δ| × 10³ (Top500 高表达基因)")
    ax.set_title(f"细胞类型对 {target_gene} 扰动的预测响应排序")
    ax.tick_params(axis="x", labelsize=8)

    from matplotlib.patches import Patch
    legend = [
        Patch(color="#E65100", label=f"{target_gene} 表达 > 0.5"),
        Patch(color="#78909C", label=f"{target_gene} 低表达"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")


def plot_baseline_comparison(cipher_delta, baseline_delta, gene_list, target_gene, output_path):
    """CIPHER vs 基线叠加图。"""
    top_idx = np.argsort(-np.abs(cipher_delta))[:25]
    genes = [gene_list[i] for i in top_idx]
    c_vals = [cipher_delta[i] for i in top_idx]
    b_vals = [baseline_delta[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = range(len(genes))
    width = 0.35

    ax.barh([y + width / 2 for y in y_pos], c_vals, width,
            color=[STYLE["target"] if g == target_gene else STYLE["up"] if v > 0 else STYLE["down"]
                   for g, v in zip(genes, c_vals)],
            label="CIPHER Δx", edgecolor="white", linewidth=0.3)
    ax.barh([y - width / 2 for y in y_pos], b_vals, width,
            color=STYLE["baseline"], alpha=0.7,
            label="control-mean (=0)", edgecolor="white", linewidth=0.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(genes, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_xlabel("Δx (lognorm 尺度)")
    ax.set_title(f"CIPHER vs 线性基线 — {target_gene} 敲低")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="CIPHER 扰动预测可视化")
    parser.add_argument("--gene", required=True, help="目标基因名")
    parser.add_argument("--direction", default="ko", choices=["ko", "oe"], help="ko=敲低 oe=过表达")
    parser.add_argument("--dataset", default="ws", help="数据集: ws|pbmc")
    parser.add_argument("--output", default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--max-memory-gib", type=float, default=2.0,
                        help="dense CIPHER estimated working-memory budget in GiB (default: 2)")
    args = parser.parse_args()

    print(f"加载 {args.dataset} 数据 ...")
    try:
        X, meta, gene_list = load_data(args.dataset, args.max_memory_gib)
    except (ValueError, MemoryError) as exc:
        parser.error(f"cannot load safely: {str(exc) or 'insufficient memory; reduce the workload'}")

    # 取 CTRL 细胞 (ws) 或全部 (pbmc)
    if args.dataset == "ws" and "cell_type" in meta.columns:
        ctrl_mask = (meta["cell_type"] == "CTRL").values
        X_fit = X[ctrl_mask]
        print(f"  CTRL 细胞: {ctrl_mask.sum()}/{X.shape[0]}")
    else:
        X_fit = X

    print(f"拟合 CIPHER ({X_fit.shape[0]} 细胞 × {X_fit.shape[1]} 基因) ...")
    eng = CipherEngine(reg_strength=0.1)
    try:
        eng.fit(X_fit, gene_list)
    except (ValueError, MemoryError) as exc:
        parser.error(f"cannot fit safely: {str(exc) or 'insufficient memory; reduce the workload'}")

    if args.gene not in eng._gene_idx:
        print(f"错误: {args.gene} 不在基因空间内")
        sys.exit(1)

    gi = eng._gene_idx[args.gene]
    ctrl_expr = float(eng._ctrl_mean[gi])
    target = 0.0 if args.direction == "ko" else ctrl_expr * 2.0

    print(f"预测 {args.gene}: {ctrl_expr:.2f} → {target:.2f} ...")
    r = eng.predict({args.gene: target})
    delta = r["delta"]

    # 基线
    bl = LinearBaseline()
    bl.fit(X_fit, gene_list)
    base = bl.predict_additive({args.gene: target})
    base_delta = base["delta"]

    # 细胞类型响应
    cell_types = meta["cell_type"].unique()
    ct_rows = []
    for ct in cell_types:
        ct_mask = (meta["cell_type"] == ct).values
        n_cells = int(ct_mask.sum())
        if n_cells < 5:
            continue
        ct_expr = X[ct_mask].mean(axis=0)
        top_expr_idx = np.argsort(-ct_expr)[:500]
        response = float(np.abs(delta[top_expr_idx]).mean())
        ct_rows.append({
            "cell_type": str(ct), "n_cells": n_cells,
            "target_expr": float(ct_expr[gi]), "response": response,
        })
    ct_rows.sort(key=lambda r: r["response"], reverse=True)

    # 绘图
    os.makedirs(args.output, exist_ok=True)
    prefix = f"{args.gene}_{args.direction}"
    print(f"\n生成图表 → {args.output}")
    plot_heatmap(delta, gene_list, args.gene, args.direction,
                 os.path.join(args.output, f"{prefix}_heatmap.svg"))
    plot_celltype_response(ct_rows, args.gene,
                           os.path.join(args.output, f"{prefix}_celltype.svg"))
    plot_baseline_comparison(delta, base_delta, gene_list, args.gene,
                             os.path.join(args.output, f"{prefix}_baseline.svg"))

    # 保存数据 JSON
    summary = {
        "gene": args.gene, "direction": args.direction,
        "ctrl_expr": ctrl_expr, "target": target,
        "n_sig_up": int((delta > 0.001).sum()),
        "n_sig_down": int((delta < -0.001).sum()),
        "cell_type_response": ct_rows,
    }
    json_path = os.path.join(args.output, f"{prefix}_summary.json")
    json.dump(summary, open(json_path, "w"), indent=2, ensure_ascii=False)
    print(f"  → {json_path}")
    print("✅ 完成")


if __name__ == "__main__":
    try:
        main()
    except MemoryError:
        sys.exit("Insufficient memory for CIPHER visualization; reduce the workload or use a suitable machine.")
