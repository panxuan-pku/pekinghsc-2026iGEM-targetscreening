"""Single-cell expression regression cases; requires optional CNV dependencies."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from screening.src.compensation import compute_compensation


@pytest.mark.parametrize("sparse", [False, True])
def test_expression_depth_is_not_compensation(sparse):
    x = np.array([[10, 90], [20, 180]])
    if sparse:
        x = csr_matrix(x)
    adata = SimpleNamespace(layers={"counts": x}, var_names=pd.Index(["A", "B"]),
                            obs=pd.DataFrame({"condition": ["CTRL", "WS"]}))
    with pytest.raises(ValueError, match="HVG subsets"):
        compute_compensation(adata, ["A"])
    out = compute_compensation(adata, ["A"], full_gene_matrix=True)
    assert out.compensation_ratio.iloc[0] == 1  # raw depth ratio would be 2
    assert out.mechanism_status.iloc[0] == "not_established"
    adata.layers["counts"] = np.log1p(np.array([[10, 90], [20, 180]]))
    with pytest.raises(ValueError, match="raw counts"):
        compute_compensation(adata, ["A"], full_gene_matrix=True)

