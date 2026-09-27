"""Small shared input contracts for local engines and model wrappers."""
import numpy as np
from scipy import sparse


def _model_values(values, name):
    if values.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    limit = np.finfo(np.float32).max
    if not np.isfinite(values).all() or (values > limit).any() or (values < -limit).any():
        raise ValueError(f"{name} must contain finite values within float32 range")
    return values.astype(np.float32, copy=False)


def check_model_input(x, n_genes, ndim=1):
    """Require a dense, model-aligned vector/batch; never guess axes or densify."""
    expected = f"({n_genes},)" if ndim == 1 else f"(n, {n_genes}), n >= 1"
    if not isinstance(x, np.ndarray) or x.ndim != ndim or x.shape[-1] != n_genes or not x.size:
        raise ValueError(f"expression must be a dense numpy array of shape {expected}; "
                         f"got {getattr(x, 'shape', type(x).__name__)}")
    return _model_values(x, "expression")


def check_perturbation_value(value):
    value = np.asarray(value)
    if value.ndim != 0:
        raise ValueError("new_value must be a real numeric scalar")
    return float(_model_values(value, "new_value"))


def check_genes(genes, size):
    if len(genes) != size or not size:
        raise ValueError("gene_list must match a non-empty gene dimension")
    if any(not isinstance(g, str) or not g.strip() for g in genes):
        raise ValueError("gene_list must contain non-empty names")
    if len(set(genes)) != size:
        raise ValueError("gene_list must contain unique names")


def check_expression(X, genes, min_cells=1):
    if getattr(X, "ndim", None) != 2 or X.shape[0] < min_cells:
        raise ValueError(f"expression must be 2D with at least {min_cells} cells")
    check_genes(genes, X.shape[1])
    values = X.data if sparse.issparse(X) else np.asarray(X)
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError("expression must contain real numeric values")
    if not np.isfinite(values).all():
        raise ValueError("expression must contain only finite values")
