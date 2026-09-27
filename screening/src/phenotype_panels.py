"""Configured phenotype evidence views; no scoring or candidate filtering."""
import pandas as pd


def load_phenotype_panels(cfg):
    panels = cfg.get("phenotype_panels", [])
    if not isinstance(panels, list):
        raise ValueError("phenotype_panels must be a list")
    seen = set()
    for panel in panels:
        if not isinstance(panel, dict):
            raise ValueError("phenotype_panels entries must be mappings")
        for key in ("id", "disease", "phenotype"):
            if not isinstance(panel.get(key), str) or not panel[key].strip():
                raise ValueError(f"phenotype_panels: {key} must be a nonempty string")
        tissues = panel.get("tissues")
        if (not isinstance(tissues, list) or not tissues
                or any(not isinstance(t, str) or not t.strip() for t in tissues)):
            raise ValueError("phenotype_panels: tissues must be a nonempty list of names")
        if len(set(tissues)) != len(tissues) or panel["id"] in seen:
            raise ValueError("phenotype_panels: duplicate tissue names or panel IDs")
        seen.add(panel["id"])
    return panels


def tissue_column(tissue):
    return f"hpa_tissue::{tissue}::ntpm"


def tissue_coverage(tissues, available):
    found = [t for t in tissues if t in available]
    missing = [t for t in tissues if t not in available]
    return {"available_tissues": found, "missing_tissues": missing,
            "coverage_status": "complete" if not missing else "partial" if found else "unavailable"}


def describe_panels(frame, panels):
    summaries = []
    for panel in panels:
        available = [t for t in panel["tissues"] if tissue_column(t) in frame.columns]
        columns = [tissue_column(t) for t in available]
        values = frame[columns].apply(pd.to_numeric, errors="coerce")
        summaries.append({**panel, **tissue_coverage(panel["tissues"], available),
                          "genes_with_expression": int(values.notna().any(axis=1).sum()),
                          "candidate_count": len(frame)})
    return summaries


def render_phenotype_panels(frame, panels):
    if not panels:
        return []

    def show(value):
        if pd.isna(value) or value == "":
            return "No data"
        return " ".join(str(value).splitlines()).replace("|", "\\|")

    lines = ["", "## Phenotype evidence panels", "",
             "All candidates follow the overall ranking; this is not a phenotype-specific ranking and does not change scores, weights or candidates.",
             "Phenotype-to-tissue mappings come from the configuration. Expression provides tissue context, not evidence of phenotype causality or safety.",
             "Zero means measured zero expression; missing data is not zero. Coverage describes available tissue columns; per-gene missingness is listed separately."]
    evidence = [("clinGen_hi_score", "ClinGen HI"),
                ("clinGen_haploinsufficiency_status", "ClinGen HI status"),
                ("gnomad_pLI", "pLI"), ("gnomad_LOEUF", "LOEUF")]
    evidence = [(col, label) for col, label in evidence if col in frame.columns]
    for panel in panels:
        status = {"complete": "Complete tissue coverage", "partial": "Incomplete coverage",
                  "unavailable": "Not evaluated: no available tissue columns"}[panel["coverage_status"]]
        lines += ["", f"### {show(panel['disease'])} / {show(panel['phenotype'])} ({show(panel['id'])})", "",
                  f"{status}; candidates with expression records: {panel['genes_with_expression']}/{len(frame)}.",
                  "Missing tissue columns: " + (", ".join(show(t) for t in panel["missing_tissues"]) or "None") + "."]
        if panel["missing_tissues"]:
            lines.append("Check HPA source data and merge configuration. Rebuild older evidence tables with the same panel configuration; missing data does not establish absence of tissue expression.")
        headers = ["Overall rank (reference)", "Gene", "HGNC", "Overall score (reference)"]
        headers += [label for _, label in evidence]
        headers += [show(t) + " nTPM" for t in panel["tissues"]] + ["Per-gene tissue coverage"]
        lines += ["", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in frame.iterrows():
            values = [pd.to_numeric(row.get(tissue_column(t)), errors="coerce") for t in panel["tissues"]]
            n = sum(pd.notna(v) for v in values)
            cells = [row.get("rank"), row.get("input_symbol"), row.get("hgnc_id"), row.get("consensus_score")]
            cells += [row.get(col) for col, _ in evidence] + values
            cells += [f"{n}/{len(values)}" + (" (no data)" if n == 0 else " (incomplete coverage)" if n < len(values) else "")]
            lines.append("| " + " | ".join(show(v) for v in cells) + " |")
    return lines
