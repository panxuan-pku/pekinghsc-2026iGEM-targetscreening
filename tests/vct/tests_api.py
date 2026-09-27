"""Smoke-test every public API route against a running local VirtualCellTool."""
import argparse
import json
import math
import os
from collections import Counter
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

BASE = f"http://127.0.0.1:{os.environ.get('VCT_PORT', '8377')}"


def get(route, *, expected_status=200, **query):
    url = BASE + route + ("?" + urlencode(query) if query else "")
    try:
        response = urlopen(url, timeout=240)
    except HTTPError as exc:
        response = exc
    with response:
        body = response.read()
        assert response.status == expected_status, (url, response.status, expected_status, body[:300])
    return json.loads(body) if route.startswith("/api/") else body.decode()


def check_dataset(ds, gene, expected_cells):
    data = get("/api/data", ds=ds)
    n = data["n_cells"]
    assert data["ds_key"] == ds and n == expected_cells, (ds, data["ds_key"], n)
    assert len(data["umap"]) == len(data["cell_types"]) == n, ds
    assert all(len(xy) == 2 and all(math.isfinite(v) for v in xy) for xy in data["umap"]), ds
    if "clusters" in data:
        assert len(data["clusters"]) == n, ds
    expression = get("/api/gene_expr_map", ds=ds, gene=gene)
    assert expression["dataset"] == ds and expression["gene"] == gene, ds
    values = expression["values"]
    assert len(values) == n and all(math.isfinite(v) for v in values), ds
    # API rounds individual values to 3 decimals; statistics are rounded separately.
    assert abs(sum(values) / n - expression["mean"]) <= 0.001, ds
    assert abs(max(values) - expression["max"]) <= 0.001, ds
    groups = Counter(data["cell_types"])
    for row in expression["by_group"]:
        assert row["group"] in groups and row["n_cells"] == groups[row["group"]], (ds, row)
    for cell in (0, n - 1):
        single = get("/api/cell_gene_expr", ds=ds, cell=cell, gene=gene)
        assert single["gene"] == gene and abs(single["value"] - values[cell]) <= 0.006, (ds, cell)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-gears", action="store_true", help="skip the two GEARS routes when Norman data is not downloaded")
    parser.add_argument("--datasets", nargs="+", choices=("pbmc", "ms", "ws"), default=["pbmc", "ms", "ws"],
                        help="Prepared datasets to verify; the core route checks also require PBMC")
    args = parser.parse_args()
    assert "VirtualCellTool" in get("/")
    health = get("/api/health")
    assert health["service"] == "VirtualCellTool" and health["ready"] is True, health
    for route, query, status in [
        ("/api/data", {"ds": "unknown"}, 400),
        ("/api/attribute", {"cell": -1}, 422),
        ("/api/perturb_cipher", {"gene": "MS4A1", "direction": "invalid"}, 422),
        ("/api/perturb_baseline", {"gene": "MS4A1", "method": "invalid"}, 422),
    ]:
        assert "detail" in get(route, expected_status=status, **query), route
    # Capture BEFORE loading MS/WS, so this actually checks cross-dataset isolation.
    before = get("/api/perturb_cipher", ds="pbmc", gene="MS4A1", direction="ko")
    datasets = {"pbmc": (2700, "MS4A1"), "ms": (17799, "MOBP"), "ws": (96969, "GTF2I")}
    for ds in args.datasets:
        n_cells, gene = datasets[ds]
        check_dataset(ds, gene, n_cells)
        response = get("/api/perturb_cipher", ds=ds, gene=gene, direction="ko")
        assert response["dataset"] == ds and response["gene"] == gene, (ds, gene)
        assert get("/api/cipher_status", ds=ds)["dataset"] == ds, ds
        assert response["delta_abs_max"] > 0, (ds, gene)
        print(f"{ds}: {n_cells} cells, {gene} |Δ|max={response['delta_abs_max']}")

    checks = [
        ("/api/attribute", {"cell": 1, "k": 5}, "top_genes"),
        ("/api/perturb", {"cell": 1, "gene": "MS4A1", "value": 0}, "displacement"),
        ("/api/perturb_multi", {"cell": 1, "genes": "MS4A1,CD79A", "value": 0}, "displacement"),
        ("/api/search_genes", {"q": "MS4A1"}, "matches"),
        ("/api/cell_gene_expr", {"cell": 1, "gene": "MS4A1"}, "value"),
        ("/api/gene_expr", {"gene": "MS4A1"}, "mean"),
        ("/api/gene_coverage", {"gene": "CEBPA"}, "covered"),
        ("/api/perturb_causal", {"gene": "CEBPA"}, "n_genes_predicted"),
        ("/api/cipher_status", {}, "fitted"),
        ("/api/gene_in_cipher", {"gene": "MS4A1"}, "covered"),
        ("/api/perturb_cipher", {"gene": "MS4A1"}, "delta_abs_max"),
        ("/api/perturb_baseline", {"gene": "MS4A1"}, "results"),
        ("/api/gene_expr_map", {"gene": "MS4A1"}, "values"),
        ("/api/attribution_map", {"cell": 1, "k": 5}, "rows"),
        ("/api/perturb_compare", {"gene": "MS4A1"}, "rows"),
        ("/api/celltype_response", {"gene": "MS4A1"}, "cell_types"),
        ("/api/perturb_summary", {"gene": "MS4A1"}, "heatmap"),
    ]
    if args.skip_gears:
        checks = [item for item in checks if item[0] not in {"/api/gene_coverage", "/api/perturb_causal"}]
    for route, query, key in checks:
        result = get(route, **query)
        assert key in result, (route, result)
    after = get("/api/perturb_cipher", ds="pbmc", gene="MS4A1", direction="ko")
    for key in ("ctrl_expr", "target_value", "delta_abs_max", "n_sig_genes", "top_up", "top_down"):
        assert before[key] == after[key], f"PBMC CIPHER {key} changed after loading MS/WS"
    print(f"{len(checks) + 2}/19 API routes and home page passed; 4 error contracts and {len(args.datasets)} dataset contracts passed")
    if args.skip_gears:
        print("SKIPPED: gene_coverage / perturb_causal (GEARS); not a full API pass")


if __name__ == "__main__":
    main()
