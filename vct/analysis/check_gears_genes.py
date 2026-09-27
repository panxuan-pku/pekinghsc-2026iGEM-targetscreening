import os, pickle
import numpy as np
import anndata as ad

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from paths import GEARS_DATA_DIR

adata = ad.read_h5ad(os.path.join(GEARS_DATA_DIR, "norman", "perturb_processed.h5ad"))

# 图节点 = 数据集里所有有表达的基因
gene_names = set(adata.var["gene_name"].astype(str).tolist())
print(f"Norman 数据集基因数(图节点): {len(gene_names)}")

# 直接扰动过的基因
with open(os.path.join(GEARS_DATA_DIR, "essential_all_data_pert_genes.pkl"), "rb") as f:
    pert_genes = set(pickle.load(f))
print(f"直接扰动过的基因数(pert set): {len(pert_genes)}")

# 微缺失病目标基因
diseases = {
    "22q11.2DS": ["TBX1","COMT","PRODH","DGCR8","HIRA","UFD1L","SEPT5","CLTCL1","RTN4R","GNB1L"],
    "Angelman(15q11)": ["UBE3A","GABRB3","GABRA5","SNRPN"],
    "Williams(7q11.23)": ["GTF2I","GTF2IRD1","ELN","LIMK1","BAZ1B","CLIP2","STX1A"],
    "Smith-Magenis(17p11.2)": ["RAI1","TNFRSF13B","LLGL1"],
    "15q13.3": ["CHRNA7","OTUD7A","KLF13","TRPM1"],
    "Cri-du-chat(5p15)": ["CTNND2","TERT","SEMA5A"],
    "Wolf-Hirschhorn(4p16.3)": ["NSD2","LETM1","WHSC1","FGFR3"],
    "1p36": ["SKI","PRDM16","KCNAB2","GABRD"],
}

print("\n{'病': '基因': 在图?(可扰动) / 在pert集?(直接训练过)'}")
for d, genes in diseases.items():
    for g in genes:
        in_graph = g in gene_names
        in_pert = g in pert_genes
        tag = "✅ 图内" if in_graph else "❌ 图外"
        pt = "·直接扰动过" if in_pert else ""
        print(f"{d:22s} {g:10s} {tag}{pt}")

# 汇总：每个病有多少基因在图内
print("\n=== 汇总 ===")
for d, genes in diseases.items():
    n_in = sum(1 for g in genes if g in gene_names)
    print(f"{d:22s} {n_in}/{len(genes)} 在图内")
