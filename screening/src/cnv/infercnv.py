#!/usr/bin/env python3
"""Load 10x samples, normalize, order by genomic position, run infercnvpy,
and build the per-window CNV signal table consumed by segments.py.

The unified pipeline environment includes infercnvpy and scanpy. This module
raises a clear setup error if that environment is incomplete.
"""
from pathlib import Path
import gzip
import json

import numpy as np
import pandas as pd

from .gene_order import CHROM_ORDER


class InsufficientDataError(ValueError):
    """Required observations are unavailable; this is not a negative CNV result."""


def validate_groups(groups, patient_cat="patient", ref_cat="reference"):
    if (not isinstance(patient_cat, str) or not patient_cat.strip()
            or not isinstance(ref_cat, str) or not ref_cat.strip() or patient_cat == ref_cat):
        raise ValueError("patient/reference group names must be nonempty and distinct")
    values = pd.Series(groups)
    if values.isna().any() or not values.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("missing or invalid cell group labels")
    missing = [name for name in (patient_cat, ref_cat) if not (values == name).any()]
    if missing:
        raise InsufficientDataError(f"insufficient data: missing required group(s): {missing}")


def validate_sample_specs(sample_specs):
    """Check sample identities before reading inputs or creating outputs."""
    if not isinstance(sample_specs, list) or not sample_specs:
        raise ValueError("samples must be a non-empty list")
    ids, paths = set(), set()
    for i, spec in enumerate(sample_specs):
        if not isinstance(spec, dict):
            raise ValueError(f"sample {i}: expected a mapping")
        for key in ("id", "group"):
            value = spec.get(key)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"sample {i}: {key} must be a non-empty string without surrounding whitespace")
        if spec["id"] in ids:
            raise ValueError(f"duplicate sample id: {spec['id']}")
        ids.add(spec["id"])
        path = spec.get("path")
        if not isinstance(path, (str, Path)) or not str(path).strip():
            raise ValueError(f"sample {spec['id']}: path must be a non-empty path")
        resolved = Path(path).resolve()
        if resolved in paths:
            raise ValueError(f"shared sample destination: {path}; use separate sample paths")
        paths.add(resolved)


def _import_deps():
    try:
        import anndata  # noqa: F401
        import infercnvpy as cnv
        import scanpy as sc
    except ImportError as e:
        raise ImportError(
            "scanpy/infercnvpy missing from the pipeline environment. "
            "From the repository root, run 'conda activate virtual-screening' and "
            "'python -m pip install -r screening/requirements-cnv.txt', then retry."
        ) from e
    return sc, cnv


def sample_input_paths(path):
    """Resolve one complete 10x layout without guessing between alternatives."""
    path = Path(path)
    matrices = [path / name for name in ("matrix.mtx", "matrix.mtx.gz")
                if (path / name).exists()]
    if len(matrices) != 1:
        raise ValueError(f"{path}: expected exactly one matrix.mtx or matrix.mtx.gz")
    suffix = ".gz" if matrices[0].suffix == ".gz" else ""
    features = path / ("features.tsv" + suffix)
    if not features.exists():
        features = path / ("genes.tsv" + suffix)
    return matrices[0], features, path / ("barcodes.tsv" + suffix)


def _read_counts(path):
    """Validate identities and raw counts before Scanpy QC can discard rows."""
    from anndata import AnnData
    from scipy.io import mmread
    from scipy.sparse import issparse

    matrix, features, barcodes = sample_input_paths(path)
    def read_rows(file):
        opener = gzip.open if file.suffix == ".gz" else open
        with opener(file, "rt") as stream:
            return [line.rstrip("\r\n").split("\t") for line in stream]
    genes, cells = read_rows(features), read_rows(barcodes)
    if not genes or any(len(row) not in (2, 3) for row in genes):
        raise ValueError(f"{features}: expected nonempty gene ID/symbol rows")
    if any(len(row) == 3 and row[2] != "Gene Expression" for row in genes):
        raise ValueError(f"{features}: only Gene Expression features are supported; no rows are silently removed")
    if not cells or any(len(row) != 1 for row in cells):
        raise ValueError(f"{barcodes}: expected one barcode per row")
    for label, values in (("gene ID", [r[0] for r in genes]),
                          ("gene symbol", [r[1] for r in genes]),
                          ("barcode", [r[0] for r in cells])):
        if any(not value.strip() or value != value.strip() for value in values):
            raise ValueError(f"missing or whitespace-padded {label}")
        index = pd.Index(values)
        if label != "gene symbol" and not index.is_unique:
            raise ValueError(f"duplicate {label}: {index[index.duplicated()][0]!r}; resolve identity before loading")
    X = mmread(matrix)
    if X.shape != (len(genes), len(cells)):
        raise ValueError(f"matrix dimensions {X.shape} do not match genes/barcodes {(len(genes), len(cells))}")
    values = X.data if issparse(X) else np.asarray(X)
    if (np.iscomplexobj(values) or not np.isfinite(values).all()
            or (values < 0).any() or (values != np.floor(values)).any()):
        raise ValueError("matrix must contain finite nonnegative integer raw counts")
    if not np.equal(values, values.astype(np.float32)).all():
        raise ValueError("raw counts exceed exact float32 representation used by Scanpy")
    # Match the existing Scanpy input dtype for valid data.
    X = X.T.astype(np.float32)
    if issparse(X):
        entries = X.nnz
        X = X.tocsr()
        if X.nnz != entries:
            raise ValueError("duplicate matrix coordinates; counts must be unambiguous")
    return AnnData(X=X, obs=pd.DataFrame(index=[r[0] for r in cells]),
                   var=pd.DataFrame({"gene_ids": [r[0] for r in genes],
                                     "gene_symbols": [r[1] for r in genes]}, index=[r[0] for r in genes]))


def load_samples(sample_specs, min_genes=200, max_mito_pct=20.0,
                 downsample=None, seed=0):
    """Read per-sample 10x mtx dirs → one AnnData with obs[sample|group] and layers['counts'].

    sample_specs: [{id, group, path}] — path must contain matrix.mtx[.gz],
    barcodes.tsv[.gz], genes.tsv[.gz] (standard 10x layout).
    """
    validate_sample_specs(sample_specs)
    sc, _ = _import_deps()
    import anndata

    rng = np.random.default_rng(seed)
    adatas = []
    for spec in sample_specs:
        try:
            ad = _read_counts(spec["path"])
        except (OSError, EOFError, ValueError) as exc:
            raise ValueError(f"sample {spec['id']}: {exc}") from exc
        if adatas and not ad.var.equals(adatas[0].var[["gene_ids", "gene_symbols"]]):
            raise ValueError(f"sample {spec['id']}: gene identity/order differs from sample {sample_specs[0]['id']}; "
                             "provide matching feature lists; no reordering or zero filling is performed")
        ad.obs["barcode"] = ad.obs_names.to_numpy()
        ad.obs_names = [json.dumps([spec["id"], barcode], ensure_ascii=False) for barcode in ad.obs_names]
        ad.obs["sample"] = spec["id"]
        ad.obs["group"] = spec["group"]
        sc.pp.filter_cells(ad, min_genes=min_genes)
        if ad.n_obs == 0:
            raise InsufficientDataError(f"sample {spec['id']}: no cells remain after min_genes QC")
        ad.var["mt"] = ad.var["gene_symbols"].str.upper().str.startswith("MT-")
        sc.pp.calculate_qc_metrics(ad, qc_vars=["mt"], inplace=True, percent_top=None)
        ad = ad[ad.obs["pct_counts_mt"] < max_mito_pct].copy()
        if ad.n_obs == 0:
            raise InsufficientDataError(f"sample {spec['id']}: no cells remain after mitochondrial QC")
        if downsample and ad.n_obs > downsample:
            keep = rng.choice(ad.n_obs, size=downsample, replace=False)
            ad = ad[np.sort(keep)].copy()
        ad.layers["counts"] = ad.X.copy()
        adatas.append(ad)
    adata = anndata.concat(adatas, join="inner", merge="same")
    return adata


def normalize_for_cnv(adata, target_sum=1e4):
    """Library-size normalize + log1p into .X (raw counts stay in layers['counts'])."""
    sc, _ = _import_deps()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


def gene_match_keys(adata, genes_df):
    """Use exact IDs when available; support only unambiguous legacy symbols."""
    key = "gene_id" if "gene_ids" in adata.var else "gene_name"
    if key not in genes_df:
        raise ValueError(f"annotation requires {key}; gene IDs cannot fall back to symbols")
    names = pd.Index(adata.var["gene_ids"] if key == "gene_id" else adata.var_names)
    label = "gene ID" if key == "gene_id" else "gene symbol"
    for values in (names, pd.Index(genes_df[key])):
        if any(not isinstance(v, str) or not v.strip() or v != v.strip() for v in values):
            raise ValueError(f"missing or invalid {label}")
    if not names.is_unique:
        raise ValueError(f"duplicate {label}: {names[names.duplicated()][0]}")
    matched = genes_df[genes_df[key].isin(names)].drop_duplicates()
    ambiguous = matched[key][matched[key].duplicated()]
    if not ambiguous.empty:
        raise ValueError(f"ambiguous gene annotation for {ambiguous.iloc[0]}")
    return names, key


def order_by_position(adata, genes_df):
    """Restrict to genes with genomic positions and sort by (chrom, start).

    Sets adata.var['chromosome'|'start'|'end'] as infercnvpy expects.
    genes_df: output of gene_order.parse_gtf_genes.
    """
    names, key = gene_match_keys(adata, genes_df)
    matched = genes_df[genes_df[key].isin(names)].drop_duplicates()
    coordinates = matched[["start", "end"]].apply(pd.to_numeric, errors="coerce")
    valid = (np.isfinite(coordinates).all(axis=1)
             & (coordinates == np.floor(coordinates)).all(axis=1)
             & (coordinates["start"] >= 0) & (coordinates["end"] >= coordinates["start"])
             & matched["chrom"].isin(CHROM_ORDER))
    if not valid.all():
        raise ValueError(f"invalid genomic position for {matched.loc[~valid, key].iloc[0]}")
    matched = matched.assign(start=coordinates["start"].astype(int), end=coordinates["end"].astype(int))
    pos = (matched
           .set_index(key)[["chrom", "start", "end"]])
    keep = names.isin(pos.index)
    sub = adata[:, keep].copy()
    if sub.n_vars == 0:
        raise InsufficientDataError("no genes match the genomic annotation")
    sub.var["chromosome"] = pos.loc[names[keep], "chrom"].to_numpy()
    sub.var["start"] = pos.loc[names[keep], "start"].to_numpy()
    sub.var["end"] = pos.loc[names[keep], "end"].to_numpy()
    audit = pd.DataFrame({key: names.to_numpy(),
                          "gene_symbol": adata.var["gene_symbols"].to_numpy()
                          if "gene_symbols" in adata.var else adata.var_names.to_numpy(),
                          "status": np.where(keep, "positioned", "not_in_supplied_annotation")})
    audit["annotation_symbol"] = matched.set_index(key, drop=False)["gene_name"].reindex(names).fillna("").to_numpy()
    for column in ("chrom", "start", "end"):
        audit[column] = pos[column].reindex(names).to_numpy()
    audit["chrom"] = audit["chrom"].fillna("")
    sub.uns["gene_position_mapping"] = audit
    rank = {c: i for i, c in enumerate(CHROM_ORDER)}
    order = (sub.var.assign(_r=sub.var["chromosome"].map(rank))
             .sort_values(["_r", "start"]).index)
    sub = sub[:, order].copy()
    return sub


def run_infercnv(adata, reference_key="group", reference_cat="reference",
                 window_size=100, step=10, exclude_chromosomes=("chrX", "chrY"), n_jobs=None):
    """Run infercnvpy; result lands in adata.obsm['X_cnv'] (cells × windows).

    chrX/chrY are excluded by default (sex-chromosome dosage dominates the
    signal otherwise) — MUST match window_table()'s exclusion."""
    usable = adata.var.loc[~adata.var["chromosome"].isin(exclude_chromosomes or ()), "chromosome"]
    if usable.value_counts().max() < window_size or usable.empty:
        raise InsufficientDataError(f"insufficient genes for any CNV window of size {window_size}")
    _, cnv = _import_deps()
    cnv.tl.infercnv(adata, reference_key=reference_key,
                    reference_cat=[reference_cat] if isinstance(reference_cat, str) else reference_cat,
                    window_size=window_size, step=step, n_jobs=n_jobs,
                    exclude_chromosomes=list(exclude_chromosomes) if exclude_chromosomes else None)
    return adata


def window_table(adata, window_size, step, group_col="group",
                 patient_cat="patient", ref_cat="reference",
                 exclude_chromosomes=("chrX", "chrY")):
    """Reconstruct window→gene mapping and aggregate the CNV signal.

    Returns DataFrame[chr, start, end, genes, score, ref_score] with one row
    per smoothed window; `score` = mean X_cnv over patient cells, `ref_score`
    = mean over reference cells (sanity check, should hover near 0).

    The window layout must match infercnvpy's definition: per chromosome,
    window i covers ordered genes [i*step, i*step + window_size). A hard
    assertion guards against silent mismatch if infercnvpy changes this.
    """
    if "X_cnv" not in adata.obsm:
        raise ValueError("adata.obsm['X_cnv'] missing — run run_infercnv first")
    X = adata.obsm["X_cnv"]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    if X.shape[1] == 0:
        raise InsufficientDataError("no CNV windows available")
    if not np.isfinite(X).all():
        raise ValueError("CNV window signals must be finite")
    validate_groups(adata.obs[group_col], patient_cat, ref_cat)
    var = adata.var

    rows, w = [], 0
    excluded = set(exclude_chromosomes or ())
    for chrom in [c for c in CHROM_ORDER
                  if (var["chromosome"] == c).any() and c not in excluded]:
        sub = var[var["chromosome"] == chrom]
        n_genes = len(sub)
        n_win = (n_genes - window_size) // step + 1
        for i in range(max(n_win, 0)):
            g = sub.iloc[i * step: i * step + window_size]
            rows.append({"chr": chrom, "start": int(g["start"].min()),
                         "end": int(g["end"].max()),
                         "genes": ",".join(map(str, g["gene_symbols"] if "gene_symbols" in g else g.index)),
                         "win_idx": w + i})
            if "gene_ids" in g:
                rows[-1]["gene_ids"] = ",".join(g["gene_ids"])
        w += max(n_win, 0)
    if w != X.shape[1]:
        raise RuntimeError(
            f"reconstructed {w} windows but X_cnv has {X.shape[1]} — infercnvpy "
            f"window layout differs from (i*step, window_size) or exclude_chromosomes "
            f"mismatch between run_infercnv() and window_table()"
        )

    tab = pd.DataFrame(rows).set_index("win_idx").sort_index()
    groups = adata.obs[group_col].astype(str)
    pat = (groups == str(patient_cat)).to_numpy()
    ref = (groups == str(ref_cat)).to_numpy()
    if pat.sum() == 0 or ref.sum() == 0:
        raise ValueError(f"{group_col!r} lacks {patient_cat!r} or {ref_cat!r} cells")
    tab["score"] = X[pat].mean(axis=0)
    tab["ref_score"] = X[ref].mean(axis=0)
    return tab.reset_index(drop=True)
