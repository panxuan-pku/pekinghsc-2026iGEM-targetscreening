"""Merged evidence and design stages; reuse the existing consensus ranker."""
import json
from pathlib import Path
import zipfile

import pandas as pd
import yaml

from screening.src.consensus_v2 import run_consensus
from screening.src.merge_evidence import join_on_hgnc
from screening.src.normalize import load_alias_map

CONFIG = Path(__file__).resolve().parents[1] / "config/screening.yaml"


def load_config():
    return yaml.safe_load(CONFIG.read_text())


def join_constraint(candidates, path):
    required = {"gene_id", "transcript", "mane_select", "canonical", "lof.pLI", "lof.oe_ci.upper", "constraint_flags", "gene_flags"}
    columns = pd.read_csv(path, sep="\t", nrows=0).columns
    if not required <= set(columns):
        raise ValueError("gnomAD v4.1.1 missing columns: " + ", ".join(sorted(required - set(columns))))
    chunks = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, usecols=sorted(required), chunksize=50000):
        chunk["gene_id"] = chunk.gene_id.str.split(".").str[0]
        chunks.append(chunk[chunk.gene_id.isin(candidates.gene_id)])
    ev = pd.concat(chunks, ignore_index=True)
    rows = []
    for gid, group in ev.groupby("gene_id"):
        chosen = group[group.mane_select.str.lower().isin(["true", "1"])]
        choice = "MANE_Select"
        if chosen.empty:
            chosen = group[group.canonical.str.lower().isin(["true", "1"])]
            choice = "canonical"
        if len(chosen) > 1:
            raise ValueError(f"ambiguous gnomAD constraint transcript: {gid}")
        if chosen.empty:
            continue
        row = chosen.iloc[0]
        flags = {k: row[k] for k in ("constraint_flags", "gene_flags")}
        flagged = any(v.strip() not in ("", "[]", "{}", "NA", "null") for v in flags.values())
        def numeric(value):
            if value in ("", "NA", "NaN"):
                return float("nan")
            number = float(value)
            if not 0 <= number < float("inf"):
                raise ValueError(f"invalid gnomAD metric for {gid}: {value}")
            return number
        pli, loeuf = numeric(row["lof.pLI"]), numeric(row["lof.oe_ci.upper"])
        if pli > 1:
            raise ValueError(f"invalid pLI for {gid}")
        rows.append({"gene_id": gid, "gnomad_transcript_id": row.transcript,
                     "gnomad_transcript_choice": choice, "gnomad_flags": json.dumps(flags),
                     "gnomad_pLI_raw": pli, "gnomad_LOEUF_raw": loeuf,
                     "gnomad_pLI": float("nan") if flagged else pli,
                     "gnomad_LOEUF": float("nan") if flagged else loeuf,
                     "gnomad_status": "quality_flagged_not_scored" if flagged else "available"})
    columns = ["gene_id", "gnomad_transcript_id", "gnomad_transcript_choice", "gnomad_flags",
               "gnomad_pLI_raw", "gnomad_LOEUF_raw", "gnomad_pLI", "gnomad_LOEUF", "gnomad_status"]
    result = candidates.merge(pd.DataFrame(rows, columns=columns), on="gene_id", how="left", validate="one_to_one")
    result["gnomad_status"] = result.gnomad_status.fillna("no_selected_transcript_record")
    return result


def merge_core(candidates, references):
    aliases = {k.upper(): v for k, v in load_alias_map(references / "hgnc.tsv").items()}
    frame = join_on_hgnc(candidates, references / "clingen.tsv", "clinGen", symbol_map=aliases)
    frame = join_constraint(frame, references / "gnomad.tsv.gz")
    with zipfile.ZipFile(references / "hpa.zip") as archive:
        with archive.open("rna_tissue_consensus.tsv") as handle:
            hpa = pd.read_csv(handle, sep="\t", dtype=str)
    required = {"Gene", "Tissue", "nTPM"}
    if not required <= set(hpa.columns):
        raise ValueError("HPA requires Gene/Tissue/nTPM")
    hpa = hpa[hpa.Gene.isin(frame.gene_id)].copy()
    hpa["nTPM"] = pd.to_numeric(hpa["nTPM"], errors="raise")
    by_gene = {gid: json.dumps(group.groupby("Tissue")["nTPM"].max().to_dict(), ensure_ascii=False)
               for gid, group in hpa.groupby("Gene")}
    frame["hpa_tissue_ntpm"] = frame.gene_id.map(by_gene).fillna("{}")
    frame["impc_viability"] = "not_supplied"
    return frame


def add_impc(frame, path):
    """Accept curated orthology-aware evidence, not capitalization-based mouse lookup."""
    ev = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"gene_id", "mouse_gene_id", "impc_viability", "source", "release", "mapping_source"}
    if not required <= set(ev.columns) or ev[list(required)].eq("").any().any():
        raise ValueError("IMPC annotations require nonempty gene_id/mouse_gene_id/impc_viability/source/release/mapping_source")
    if ev.gene_id.duplicated().any() or not set(ev.impc_viability) <= {"viable", "subviable", "lethal", "unknown"}:
        raise ValueError("IMPC requires one unambiguous record per gene and a supported viability status")
    ev = ev[list(sorted(required))].rename(columns={k: "impc_" + k for k in ("mouse_gene_id", "source", "release", "mapping_source")})
    out = frame.drop(columns=["impc_viability"]).merge(ev, on="gene_id", how="left", validate="one_to_one")
    out["impc_viability"] = out.impc_viability.fillna("not_supplied")
    return out


def rank_candidates(frame):
    return run_consensus(frame, load_config()["consensus_v2"], mode="rank")


def select_modules(ranked, capacity=4400, module_size=300):
    if capacity <= 0 or module_size <= 0:
        raise ValueError("module budget and size must be positive")
    out = ranked.copy()
    minimum = load_config()["project_min_utr3_bp"]
    out["project_min_utr3_bp"] = minimum
    out["module_budget_bp"], out["assumed_module_bp"] = capacity, module_size
    selected, reasons, notes = [], [], []
    count = 0
    for row in out.to_dict("records"):
        note = []
        if row.get("clinGen_haploinsufficiency_raw") == 30:
            note.append("ClinGen recessive association: annotation only; neither automatic exclusion nor evidence of no dosage effect.")
        if row.get("clinGen_haploinsufficiency_raw") == 40:
            note.append("ClinGen dosage sensitivity unlikely: conflicting evidence requires manual review.")
        if row.get("impc_viability") == "viable":
            note.append("IMPC mouse viability does not establish absence of other phenotypes or effects of a human deletion.")
        if row.get("interval_overlap") == "partial":
            note.append("Partial interval overlap: review functional impact and the status of the remaining copy.")
        if row.get("evidence_status") == "no_usable_evidence":
            note.append("No usable core evidence; a low score is not negative evidence.")
        if row.get("utr3_status") != "ok" or pd.isna(row.get("utr3_bp")):
            reason = "unresolved_transcript_or_utr"
        elif row["utr3_bp"] < minimum:
            reason = "below_project_utr3_minimum"
        elif count >= capacity // module_size:
            reason = "module_budget_limit"
        else:
            reason = "selected_within_module_budget"
            count += 1
        selected.append(reason == "selected_within_module_budget")
        reasons.append(reason)
        notes.append(" ".join(note))
    out["selected"], out["selection_reason"], out["evidence_notes"] = selected, reasons, notes
    out["selection_caveat"] = "Fixed module-budget estimate; full construct length, packaging, target-sequence specificity and therapeutic effects remain unvalidated."
    return out
