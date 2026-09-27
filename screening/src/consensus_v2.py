#!/usr/bin/env python3
"""
Consensus v2 — normalized weighted scoring + Borda cross-check + optional
leave-one-out positive-control weight tuning.

Current behavior (the legacy v1 script was retired on 2026-09-25):
1. Every continuous evidence is normalized (rank | zscore | raw) BEFORE
   weighting, so weights are comparable across differently-scaled sources
   (v1 used threshold-based weighted votes).
2. HPA organ penalty reads per_organ_weight from config (v1 hard-coded -2)
   and is capped (hpa_penalty.cap) so multi-organ high expression cannot
   sink a gene without bound.
3. Missing evidence is neutral: 0 contribution in the additive score and
   median points in Borda (config: borda.missing), never a hidden penalty.
4. Optional Borda rank aggregation is reported next to the additive rank;
   their Spearman agreement is a consistency diagnostic, not biological validation.
5. Optional leave-one-out tuning of evidence weights against known driver
   genes (positive controls), maximizing recall@top-N with a simplicity
   tie-break (closest to default weights).

Expected evidence schema (see 工程化指导书 §3.4):
  hgnc_id, input_symbol,
  clinGen_hi_score (0-3), gnomad_pLI (0-1), gnomad_LOEUF (>=0, lower=constrained),
  hpa_{organ}_expr (nTPM), optional AI scores
  (DeepLOF_score / DosaCNV_score / DeepGenePrior_score, 0-1).

Current ranking CLI options include:
  --controls PATH   txt with one positive-control symbol per line (overrides config)
  --tune            force-enable leave-one-out weight tuning
"""
import argparse
import itertools
from screening.src.phenotype_panels import load_phenotype_panels, describe_panels, render_phenotype_panels
from pathlib import Path
import sys
import json
from datetime import datetime, timezone
from screening.src.merge_evidence import sha256_file
from screening.src.output_paths import validate_output_paths

import numpy as np
import pandas as pd
import yaml

HPA_DEFAULT_ORGANS = ["liver", "brain", "kidney", "gastrointestinal"]


# ---------------------------------------------------------------- config
def _default_config(cfg):
    """Derive v2 scoring defaults from the v1 pipeline.yaml blocks."""
    v1w = cfg.get("consensus", {}).get("base_weights", {}) or {}
    hpa = cfg.get("hpa_filter", {}) or {}
    evidence = {}
    for col, w in v1w.items():
        if col == "negative_hpa_organ":
            continue
        if col == "clinGen_hi_score":
            direction = "ordinal"
        elif col == "gnomad_LOEUF":
            direction = "lower_better"
        else:
            direction = "higher_better"
        evidence[col] = {"direction": direction, "weight": float(w)}
        if col == "clinGen_hi_score":
            evidence[col]["ordinal_scale"] = 3
    return {
        "normalize": "raw",
        "evidence": evidence,
        "hpa_penalty": {
            "organs": list(hpa.get("organs", HPA_DEFAULT_ORGANS)),
            "expr_threshold": float(hpa.get("high_expression_log", 1.0)),
            "per_organ_weight": float(v1w.get("negative_hpa_organ", -2)),
            "cap": -4.0,
        },
        "borda": {"enabled": True, "missing": "median"},
        "tuning": {
            "enabled": False,
            "weight_grid": [0, 1, 2, 3],
            "target_top_n": 10,
        },
    }


def load_v2_config(cfg):
    """Merge config['consensus_v2'] over v1-derived defaults."""
    base = _default_config(cfg)
    user = cfg.get("consensus_v2") or {}
    for key in ("normalize",):
        if key in user:
            base[key] = user[key]
    for col, spec in (user.get("evidence") or {}).items():
        merged = dict(base["evidence"].get(col, {"direction": "higher_better", "weight": 1.0}))
        merged.update(spec or {})
        if "weight" in merged:
            merged["weight"] = float(merged["weight"])
        base["evidence"][col] = merged
    for section in ("hpa_penalty", "borda", "tuning"):
        if section in user and isinstance(user[section], dict):
            base[section].update(user[section])
    for section in ("recessive", "validation", "warnings", "experiments", "sensitivity"):
        base[section] = dict(cfg.get(section) or {})
    base["pipeline_mode"] = cfg.get("pipeline_mode", "auto")
    base["phenotype_panels"] = load_phenotype_panels(cfg)
    validate_numeric_config(base)
    return base


def validate_numeric_config(cfg2):
    """Reject non-finite scoring parameters without changing their semantics."""
    def finite(value, path):
        try:
            valid = np.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError(f"{path} must be a finite number; got {value!r}")

    for col, spec in cfg2.get("evidence", {}).items():
        for key in ("weight", "ordinal_scale", "low_score_penalty"):
            if key in spec and not (key == "ordinal_scale" and spec[key] is None):
                finite(spec[key], f"evidence.{col}.{key}")
    for section, keys in (
        ("hpa_penalty", ("expr_threshold", "per_organ_weight", "cap")),
        ("tuning", ("target_top_n",)),
        ("recessive", ("gnomad_weight_multiplier",)),
        ("validation", ("max_missing_rate",)),
    ):
        for key in keys:
            if key in cfg2.get(section, {}):
                finite(cfg2[section][key], f"{section}.{key}")
    for i, value in enumerate(cfg2.get("tuning", {}).get("weight_grid", [])):
        finite(value, f"tuning.weight_grid[{i}]")


# -------------------------------------------------------- normalization
def normalize_series(s, method, direction, ordinal_scale=None):
    """Map an evidence column to [0,1] (NaN preserved); 1.0 = most driver-like."""
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()
    if valid.empty:
        return s.astype(float)
    if ordinal_scale:
        return (s.clip(lower=0, upper=ordinal_scale) / float(ordinal_scale)).astype(float)
    if method == "rank":
        v = s.rank(pct=True, method="average")
    elif method == "zscore":
        mu, sd = float(valid.mean()), float(valid.std(ddof=0))
        z = (s - mu) / (sd if sd > 0 else 1.0)
        lo, hi = float(z.min()), float(z.max())
        v = (z - lo) / (hi - lo) if hi > lo else s.where(s.isna(), 0.5)
    elif method == "raw":
        lo, hi = float(valid.min()), float(valid.max())
        v = (s - lo) / (hi - lo) if hi > lo else s.where(s.isna(), 0.5)
    else:
        raise ValueError(f"unknown normalize method: {method!r} (rank|zscore|raw)")
    if direction == "lower_better":
        v = 1.0 - v
    return v.astype(float)


def build_scoring_matrix(df, ev_cfg, method):
    """Return (A, cols, coverage, normed).

    A: (n_genes x n_evidences) adjusted contribution matrix with NaN -> 0,
       where ordinal low-score penalties are folded in as negative indicators
       (scaled by spec['low_score_penalty'], default 0 for non-ordinal evidence).
    normed: {col: normalized Series with NaN preserved} for Borda.
    """
    normed = {}
    coverage = {}
    cols, mats = [], []
    for col, spec in ev_cfg.items():
        if col not in df.columns:
            coverage[col] = 0.0
            continue
        scale = spec.get("ordinal_scale")
        raw = pd.to_numeric(df[col], errors="coerce")
        if col in ("clinGen_hi_score", "clinGen_triplo_score"):
            raw = raw.where(raw.isin([0, 1, 2, 3]))
            scale = 3  # special category codes are never ordinal scores
            if spec.get("low_score_penalty", 0):
                raise ValueError("ClinGen 0/1 are not negative evidence; set low_score_penalty=0")
        v = normalize_series(raw, method, spec.get("direction", "higher_better"), scale)
        cov = float(v.notna().mean())
        coverage[col] = cov
        if cov == 0.0:
            continue
        adj = v.fillna(0.0).to_numpy(dtype=float)
        low_pen = float(spec.get("low_score_penalty", 0.0) or 0.0)
        if low_pen and scale:
            raw = pd.to_numeric(df[col], errors="coerce")
            indicator = ((raw <= 1.0) & raw.notna()).to_numpy(dtype=float)
            adj = adj - low_pen * indicator
        normed[col] = v
        cols.append(col)
        mats.append(adj)
    A = np.column_stack(mats) if mats else np.zeros((len(df), 0))
    return A, cols, coverage, normed


# ------------------------------------------------------------- scoring
def hpa_penalty_term(df, hpa_cfg):
    """Per-gene negative term: per_organ_weight per organ above threshold, capped."""
    organs = hpa_cfg.get("organs", HPA_DEFAULT_ORGANS)
    thr = float(hpa_cfg.get("expr_threshold", 1.0))
    per = float(hpa_cfg.get("per_organ_weight", -2.0))
    cap = float(hpa_cfg.get("cap", -4.0))
    term = np.zeros(len(df))
    for organ in organs:
        col = f"hpa_{organ}_expr"
        if col not in df.columns:
            continue
        expr = pd.to_numeric(df[col], errors="coerce")
        term += per * (expr > thr).fillna(False).to_numpy(dtype=float)
    return np.maximum(term, cap)  # cap is negative: never penalize more than cap


def score_all(A, weights, hpa_term):
    with np.errstate(over="ignore", invalid="ignore"):
        scores = A @ np.asarray(weights, dtype=float) + hpa_term
    if not np.isfinite(scores).all():
        raise ValueError("non-finite consensus scores; check numeric configuration and evidence scale")
    return scores


def ordinal_ranks(scores, ids):
    """One deterministic tie policy for output, tuning and sensitivity."""
    order = np.lexsort((pd.Series(ids).fillna("~").astype(str).to_numpy(), -np.asarray(scores)))
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def borda_points(normed, cols, weights, n, missing="median"):
    """Weighted Borda: best gene gets (n-1) points per evidence, scaled by weight.

    Missing values get median points (neutral) or 0, per `missing`.
    """
    pts = np.zeros(n)
    for col, w in zip(cols, weights):
        s = normed[col]
        ranks = s.rank(ascending=False, method="average")  # best -> 1
        p = (n - ranks).to_numpy(dtype=float)
        fill = (n - 1) / 2.0 if missing == "median" else 0.0
        p = np.where(np.isnan(p), fill, p)
        with np.errstate(over="ignore", invalid="ignore"):
            pts += float(w) * p
    if not np.isfinite(pts).all():
        raise ValueError("non-finite borda scores; check evidence weights")
    return pts


# -------------------------------------------------------------- tuning
def _validated_controls(controls):
    if (not isinstance(controls, list)
            or any(not isinstance(c, str) or not c.strip() for c in controls)):
        raise ValueError("tuning.positive_controls must be an explicit list of non-empty strings; "
                         "use [] for no controls or provide --controls PATH")
    return list(dict.fromkeys(c.strip().upper() for c in controls))


def loo_tune(df, A, cols, default_w, tune_cfg, hpa_term, symbol_col):
    """Leave-one-out weight tuning against positive controls.

    For each held-out control: grid-search weights maximizing recall@N on the
    remaining controls (tie-break: closest L1 distance to default weights),
    then evaluate the held-out control's rank. Final weights are re-selected
    on ALL controls with the same rule.
    """
    grid = [float(x) for x in tune_cfg.get("weight_grid", [0, 1, 2, 3])]
    top_n = int(tune_cfg.get("target_top_n", 10))
    controls = _validated_controls(tune_cfg.get("positive_controls"))
    syms = df[symbol_col].astype(str).str.strip().str.upper()
    sym_set = set(syms)
    present = [c for c in controls if c.upper() in sym_set]
    missing_ctrl = [c for c in controls if c.upper() not in sym_set]
    idx_of = {c: int(np.flatnonzero(syms.to_numpy() == c.upper())[0]) for c in present}
    result = {"controls_requested": controls, "controls_present": present,
              "controls_missing": missing_ctrl, "folds": [], "final_weights": None}

    if len(present) < 3 or not cols:
        result["note"] = "need >=3 positive controls present in candidates; tuning skipped"
        return result

    combos = [w for w in itertools.product(grid, repeat=len(cols)) if any(x > 0 for x in w)]
    if not combos:
        raise ValueError("weight_grid must contain a positive weight")
    default_w = list(default_w)

    def recall(weights, control_idxs):
        scores = score_all(A, weights, hpa_term)
        top = set(np.flatnonzero(ordinal_ranks(scores, df["hgnc_id"]) <= top_n))
        return sum(1 for i in control_idxs if i in top) / len(control_idxs)

    def best_weights(train_idxs):
        scored = [(recall(w, train_idxs), w) for w in combos]
        best_r = max(r for r, _ in scored)
        tied = [w for r, w in scored if r == best_r]
        tied.sort(key=lambda w: sum(abs(a - b) for a, b in zip(w, default_w)))
        return list(tied[0]), best_r

    for held in present:
        train = [idx_of[c] for c in present if c != held]
        w_star, r_train = best_weights(train)
        scores = score_all(A, w_star, hpa_term)
        held_idx = idx_of[held]
        rank = int(ordinal_ranks(scores, df["hgnc_id"])[held_idx])
        result["folds"].append({"held_out": held, "train_recall": r_train,
                                "held_out_rank": rank, "held_out_in_topN": rank <= top_n,
                                "weights": w_star})
    w_final, r_all = best_weights([idx_of[c] for c in present])
    result["final_weights"] = w_final
    result["final_recall"] = r_all
    result["loo_hit_rate"] = float(np.mean([f["held_out_in_topN"] for f in result["folds"]]))
    return result


# ---------------------------------------------------------------- run
def run_consensus(df, cfg2, controls_path=None, tune_override=False, sensitivity=False,
                 mode="auto"):
    """Score + rank. Returns (out_df, meta).
    v2.1: sensitivity=True enables weight sensitivity analysis.
    v2.2: mode selects behaviours (validate|rank|full|exploratory|auto).
    """
    validate_numeric_config(cfg2)
    df = df.copy().reset_index(drop=True)
    df["hgnc_id"] = df["hgnc_id"].astype("string").str.strip().replace("", pd.NA)
    if df.empty:
        raise ValueError("no candidate genes")
    if df["hgnc_id"].dropna().duplicated().any():
        raise ValueError("duplicate HGNC IDs in candidates")
    ev_cfg = cfg2["evidence"]
    method = cfg2.get("normalize", "rank")
    symbol_col = "input_symbol" if "input_symbol" in df.columns else (
        "gene_symbol" if "gene_symbol" in df.columns else "hgnc_id")

    A, cols, coverage, normed = build_scoring_matrix(df, ev_cfg, method)
    active_cols = [c for c in cols if float(ev_cfg[c].get("weight", 0)) > 0]
    supported = pd.DataFrame({c: normed[c].notna() for c in active_cols}, index=df.index).any(axis=1)
    supported &= df["hgnc_id"].notna()
    missing_rate = float((~supported).mean())
    if missing_rate > float(cfg2.get("validation", {}).get("max_missing_rate", 0.5)):
        raise ValueError(f"candidates without usable evidence: {missing_rate:.1%}; exceeds max_missing_rate")
    # Apply the optional inheritance heuristic once, BEFORE every scoring path.
    recessive_applied = False
    factor = float(cfg2.get("recessive", {}).get("gnomad_weight_multiplier", 1.0))
    if not 0 <= factor <= 1:
        raise ValueError("gnomad_weight_multiplier must be in [0,1]")
    if "recessive" in df.columns:
        flags = df["recessive"].fillna(False).astype(str).str.strip().str.lower()
        if not flags.isin(["true", "false", "1", "0"]).all():
            raise ValueError("recessive flags must be boolean, not arbitrary strings")
        r_mask = flags.isin(["true", "1"]).to_numpy()
        for j, c in enumerate(cols):
            if c.startswith("gnomad_"):
                A[r_mask, j] *= factor
                normed[c] = normed[c].where(~r_mask, normed[c] * factor)
        recessive_applied = bool(r_mask.any() and factor != 1)
    hpa_term = hpa_penalty_term(df, cfg2["hpa_penalty"])
    default_w = [float(ev_cfg[c].get("weight", 0.0)) for c in cols]

    tune_cfg = dict(cfg2.get("tuning", {}))
    controls_list = tune_cfg.get("positive_controls")
    if controls_path:
        with open(controls_path) as f:
            controls_list = [ln.strip() for ln in f if ln.strip()]
    controls_list = _validated_controls(controls_list)
    tune_cfg["positive_controls"] = controls_list
    do_tune = bool(tune_cfg.get("enabled")) or tune_override
    tuning = None
    weights = default_w
    if do_tune:
        tuning = loo_tune(df, A, cols, default_w, tune_cfg, hpa_term, symbol_col)
        if tuning.get("final_weights"):
            weights = tuning["final_weights"]

    scores = score_all(A, weights, hpa_term)
    out = df.copy()
    out["evidence_status"] = np.where(supported, "available", "no_usable_evidence")
    raw_hi = pd.to_numeric(out.get("clinGen_haploinsufficiency_raw", out.get("clinGen_hi_score", pd.Series(np.nan, index=out.index))), errors="coerce")
    out["dosage_conflict"] = raw_hi.eq(40)
    out["consensus_score"] = scores
    for j, col in enumerate(cols):
        out[f"contrib_{col}"] = A[:, j] * weights[j]
    out["hpa_penalty"] = hpa_term

    # ---- v2.1: weight sensitivity (perturb each weight ±1) ----
    sensitivity_result = None
    if sensitivity and cols:
        sens = {"perturbations": []}
        baseline_ranks = None
        for ci, col in enumerate(cols):
            for delta in [-1, +1]:
                w_pert = list(weights)
                w_pert[ci] = max(0, w_pert[ci] + delta)
                if w_pert == list(weights):
                    continue
                s_pert = score_all(A, w_pert, hpa_term)
                ranks_pert = pd.Series(ordinal_ranks(s_pert, df["hgnc_id"]))
                if baseline_ranks is None:
                    baseline_ranks = pd.Series(ordinal_ranks(scores, df["hgnc_id"]))
                rank_shift = (ranks_pert - baseline_ranks).abs().max()
                top3_baseline = set(baseline_ranks.nsmallest(3).index)
                top3_pert = set(ranks_pert.nsmallest(3).index)
                top3_change = len(top3_baseline.symmetric_difference(top3_pert)) // 2
                sens["perturbations"].append({
                    "evidence": col, "delta": delta, "weight_perturbed": w_pert[ci],
                    "max_rank_shift": int(rank_shift), "top3_genes_changed": int(top3_change),
                })
        max_shift = max(p["max_rank_shift"] for p in sens["perturbations"]) if sens["perturbations"] else 0
        n_perturb = len(sens["perturbations"])
        n_top3_changes = sum(1 for p in sens["perturbations"] if p["top3_genes_changed"] > 0)
        sensitivity_result = {
            "perturbations": sens["perturbations"],
            "max_rank_shift": max_shift,
            "n_perturbations_tested": n_perturb,
            "n_causing_top3_change": n_top3_changes,
            "robustness": "high" if n_top3_changes == 0
            else ("moderate" if n_top3_changes <= n_perturb // 2 else "low"),
        }

    meta = {"method": method, "cols": cols, "coverage": coverage,
            "weights": dict(zip(cols, weights)), "default_weights": dict(zip(cols, default_w)),
            "tuning": tuning, "hpa_cfg": cfg2["hpa_penalty"],
            "recessive_applied": recessive_applied,
            "recessive_multiplier": factor, "no_evidence_rate": missing_rate,
            "sensitivity": sensitivity_result,
            "mode": mode,
            "controls_list": controls_list,
            "symbol_col": symbol_col}

    borda_cfg = cfg2.get("borda", {})
    if borda_cfg.get("enabled", True) and cols:
        out["borda_score"] = borda_points(normed, cols, weights, len(df),
                                          missing=borda_cfg.get("missing", "median"))
        out["borda_rank"] = out["borda_score"].rank(ascending=False, method="min").astype(int)
        meta["spearman"] = (float(out["consensus_score"].rank().corr(out["borda_score"].rank()))
                            if out["consensus_score"].nunique() > 1 and out["borda_score"].nunique() > 1 else None)
    else:
        meta["spearman"] = None

    out = out.sort_values(["consensus_score", "hgnc_id"], ascending=[False, True],
                          kind="stable").reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    meta["phenotype_panels"] = describe_panels(out, cfg2.get("phenotype_panels", []))
    return out, meta


# ------------------------------------------------------------- report
def render_report(out, meta, top_n=10):
    def show(value):
        return "—" if pd.isna(value) or value == "" else str(value)

    mode = meta.get("mode", "auto")
    mode_label = {
        "validate": "Mode A · Validation report (known driver genes)",
        "rank": "Mode B · Ranking with positive-control checks",
        "full": "Mode C · Ranking with expression status",
        "exploratory": "Mode D · Exploratory prioritization",
        "auto": "auto",
    }.get(mode, mode)
    lines = ["# Microdeletion candidate prioritization — consensus ranking (v2)", ""]
    lines.append(f"Pipeline mode: **{mode_label}** · Normalization: **{meta['method']}** · {len(out)} genes")
    lines.append("")
    lines.append("## Weights used")
    lines.append("")
    lines.append("| evidence | weight (default) | weight (used) | coverage |")
    lines.append("|---|---|---|---|")
    for col in meta["cols"]:
        lines.append(f"| {col} | {meta['default_weights'][col]:g} | "
                     f"{meta['weights'][col]:g} | {meta['coverage'][col]:.0%} |")
    missing_cols = [c for c, cov in meta["coverage"].items() if cov == 0.0]
    if missing_cols:
        lines.append("")
        lines.append(f"⚠️ evidence columns absent or all-missing (0 contribution): "
                     f"{', '.join(missing_cols)}")
    lines.append("")
    lines.append(f"HPA penalty: per-organ {meta['hpa_cfg'].get('per_organ_weight')} above "
                 f"expr>{meta['hpa_cfg'].get('expr_threshold')}, capped at {meta['hpa_cfg'].get('cap')}")
    lines.append("")

    # ---- v2.1: recessive penalty note ----
    if meta.get("recessive_applied"):
        lines.append(f"⚠️ Recessive annotation: gnomAD contributions multiplied by {meta['recessive_multiplier']:g} "
                     "only for tagged genes. This is an unvalidated, opt-in heuristic.")
        lines.append("")
    lines.append("Ranks indicate candidate priority, not causality or SINEUP feasibility. HI=40 is evidence against dosage sensitivity and requires review; it differs from not evaluated.")
    lines.append("")
    if "clinGen_haploinsufficiency_status" in out.columns:
        lines += ["## Raw ClinGen evidence", "", "| gene | HI raw | status | evaluated |", "|---|---|---|---|"]
        for _, r in out.iterrows():
            lines.append(f"| {r.get('input_symbol', '')} | {show(r.get('clinGen_haploinsufficiency_raw'))} | "
                         f"{r['clinGen_haploinsufficiency_status']} | {show(r.get('clinGen_date_last_evaluated'))} |")
        lines.append("")

    # ---- v2.1: compensation status ----
    if "compensation_class" in out.columns:
        lines.append("## Relative mRNA expression (observational; does not establish compensation mechanisms)")
        lines.append("")
        lines.append("| rank | gene | patient/reference ratio | legacy class | interpretation |")
        lines.append("|---|---|---|---|---|")
        for _, r in out.head(top_n).iterrows():
            c = r.get("compensation_class", "unreliable")
            ratio = r.get("compensation_ratio", "")
            ratio_str = f"{ratio:.3f}" if isinstance(ratio, float) and not (isinstance(ratio, float) and np.isnan(ratio)) else "N/A"
            sineup = "Protein status and SINEUP feasibility unknown"
            lines.append(f"| {int(r['rank'])} | {r.get('input_symbol', r.get('gene', ''))} | {ratio_str} | {c} | {sineup} |")
        lines.append("")

    # ---- v2.2: positive-control check (Mode B/C; informative for all) ----
    cc = meta.get("controls_check")
    if cc:
        lines.append("## Positive-control check (ranks of known driver genes)")
        lines.append("")
        lines.append("| gene | in candidates | rank | top-10 | consensus | ClinGen | pLI | Expression status |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for c in cc:
            if not c.get("in_candidates"):
                lines.append(f"| {c['control']} | Not in candidates | — | — | — | — | — | — |")
            else:
                lines.append(
                    f"| {c['control']} | ✓ | {c['rank']} | {'✓' if c['top10'] else '✗'} | "
                    f"{c['consensus_score']:.3f} | {show(c.get('clinGen_hi_score'))} | "
                    f"{show(c.get('gnomad_pLI'))} | {show(c.get('compensation_class'))} |")
        lines.append("")
        lines.append(f"Positive-control top-10 hit rate: **{meta['controls_top10_n']}/{meta['controls_total']}**")
        lines.append("")
    elif meta.get("controls_list") == []:
        lines.append("No positive controls configured; this check was not evaluated.")
        lines.append("")

    # ---- v2.2: validation panel (Mode A) ----
    vp = meta.get("validation_panel")
    if vp:
        lines.append("## Validation report — evidence panels for known driver genes (Mode A)")
        lines.append("")
        lines.append("| gene | rank | ClinGen HI | gnomAD pLI | consensus | Preliminary genetic support | Expression label |")
        lines.append("|---|---|---|---|---|---|---|")
        for p in vp:
            lines.append(
                f"| {p['gene']} | {p['rank']} | {show(p.get('clinGen_hi_score'))} | "
                f"{show(p.get('gnomad_pLI'))} | {p['consensus_score']:.3f} | "
                f"{'Supported (HI>=2 or pLI>=0.9, without an HI=40 conflict)' if p['target_ok'] else 'Unsupported or conflicting evidence'} | {show(p.get('compensation_class'))} |")
        lines.append("")
        lines.append(f"Check result: **{'Preliminary genetic support found' if meta.get('validation_pass') else 'No preliminary genetic support found'}**; this does not validate SINEUP targeting conditions.")
        lines.append("")

    # ---- v2.2: exploratory disclaimer (Mode D) ----
    if mode == "exploratory":
        lines.append("## ⚠️ Exploratory-mode statement (Mode D)")
        lines.append("")
        lines.append("This interval lacks a trusted curated benchmark (low ClinGen coverage) or known driver genes. This ranking is "
                     "**exploratory prioritization** based on population constraint and expression context; it "
                     "**does not establish definitive targets**. Add literature or experimental positive controls before interpreting the ranking.")
        lines.append("")

    lines.append(f"## Top {min(top_n, len(out))}")
    lines.append("")
    hdr = "| rank | hgnc_id | symbol | score |"
    sep = "|---|---|---|---|"
    if "borda_rank" in out.columns:
        hdr += " borda_rank |"
        sep += "---|"
    lines += [hdr, sep]
    for _, r in out.head(top_n).iterrows():
        row = (f"| {int(r['rank'])} | {r['hgnc_id']} | {r.get('input_symbol', '')} "
               f"| {r['consensus_score']:.3f} |")
        if "borda_rank" in out.columns:
            row += f" {int(r['borda_rank'])} |"
        lines.append(row)
    lines.append("")

    if meta.get("spearman") is not None:
        lines.append(f"Additive-rank vs Borda-rank Spearman ρ = **{meta['spearman']:.3f}** "
                     f"(>0.8: the two aggregation schemes largely agree)")
        lines.append("(Borda aggregates evidence columns only; HPA penalty is additive-only.)")
        lines.append("")

    # ---- v2.1: warnings ----
    for w in meta.get("warnings", []):
        lines.append(w)
        lines.append("")

    t = meta.get("tuning")
    if t:
        lines.append("## Leave-one-out weight tuning (positive controls)")
        lines.append("")
        if t.get("note"):
            lines.append(f"⚠️ {t['note']}")
        else:
            lines.append(f"Controls present: {', '.join(t['controls_present'])} "
                         f"(missing from candidates: {', '.join(t['controls_missing']) or 'none'})")
            lines.append("")
            lines.append("| held-out | rank | in top-N | train recall |")
            lines.append("|---|---|---|---|")
            for f in t["folds"]:
                lines.append(f"| {f['held_out']} | {f['held_out_rank']} | "
                             f"{'✓' if f['held_out_in_topN'] else '✗'} | {f['train_recall']:.2f} |")
            lines.append("")
            lines.append(f"LOO hit rate: **{t['loo_hit_rate']:.0%}** · "
                         f"final recall@N on all controls: {t['final_recall']:.2f}")
            lines.append("")
            lines.append("Final weights: " + ", ".join(
                f"{c}={w:g}" for c, w in zip(meta["cols"], t["final_weights"])))
        lines.append("")

    lines.append("## Data-source verification")
    lines.append("```")
    lines.append("Merge stage: run.json / data_checksums.txt in the specified audit directory. Scoring stage: <ranked CSV>.audit.json (or the --audit path).")
    lines.append("```")

    # ---- v2.1: weight sensitivity ----
    sens = meta.get("sensitivity")
    if sens:
        lines.append("")
        lines.append("## ⚖️ Weight sensitivity analysis")
        lines.append("")
        lines.append(f"- {sens['n_perturbations_tested']} weight perturbation tests (+/-1 per evidence column)")
        lines.append(f"- Maximum rank shift: {sens['max_rank_shift']} positions")
        lines.append(f"- Perturbations changing the top 3: {sens['n_causing_top3_change']}/{sens['n_perturbations_tested']}")
        lines.append(f"- Ranking robustness: **{sens['robustness']}**")
        lines.append("")
        if sens["robustness"] == "low":
            lines.append("> ⚠️ Ranking is sensitive to weight choices. Consider orthogonal validation of the top N candidates "
                         "instead of selecting rank #1 as the sole target.")
    
    # ---- v2.1: experiment recommendations ----
    if "experiments" in meta:
        lines.append("")
        lines.append("## 🧪 Experimental validation suggestions")
        lines.append("")
        for exp in meta["experiments"]:
            lines.append(f"### {exp.get('title', '')}")
            lines.append(exp.get("body", ""))
            lines.append("")
    lines.extend(render_phenotype_panels(out, meta.get("phenotype_panels", [])))
    return "\n".join(lines)


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--ai-scores", default=None)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--controls", default=None,
                    help="txt with one positive-control symbol per line (overrides config)")
    ap.add_argument("--tune", action="store_true",
                    help="force-enable leave-one-out weight tuning")
    ap.add_argument("--sensitivity", action="store_true",
                    help="run weight sensitivity analysis (±1 per evidence weight)")
    ap.add_argument("--compensation", default=None,
                    help="CSV from src/compensation.py: mRNA compensation per gene")
    ap.add_argument("--mode", default=None, choices=["auto", "validate", "rank", "full", "exploratory"],
                    help="pipeline mode: auto|validate|rank|full|exploratory (default: auto)")
    ap.add_argument("--warnings", action="store_true",
                    help="emit sparse-evidence warnings for Mode D")
    ap.add_argument("--audit", default=None, help="run JSON (default: <out>.audit.json)")
    args = ap.parse_args()

    inputs = [args.evidence, args.config, args.ai_scores, args.controls, args.compensation]
    audit_path = Path(args.audit) if args.audit else Path(str(args.out) + ".audit.json")
    try:
        validate_output_paths(inputs, [args.out, args.report, audit_path])
    except (ValueError, OSError, RuntimeError) as exc:
        ap.error(str(exc))

    def merge_optional(extra, key, source):
        overlap = sorted((set(ev.columns) & set(extra.columns)) - {key})
        if overlap:
            ap.error(f"{source}: columns overlap existing evidence: {', '.join(overlap)}; "
                     "choose one source for each column before rerunning (no automatic overwrite)")
        return ev.merge(extra, on=key, how="left", validate="many_to_one")

    cfg = yaml.safe_load(open(args.config))
    try:
        cfg2 = load_v2_config(cfg)
    except ValueError as exc:
        ap.error(str(exc))

    ev = pd.read_parquet(args.evidence)
    if "hgnc_id" not in ev.columns:
        raise ValueError("evidence must have hgnc_id")
    ev["hgnc_id"] = ev["hgnc_id"].astype("string").str.strip().replace("", pd.NA)

    if args.ai_scores:
        ai = pd.read_csv(args.ai_scores)
        ai["hgnc_id"] = ai["hgnc_id"].astype("string").str.strip().replace("", pd.NA)
        if ai["hgnc_id"].isna().any():
            raise ValueError("AI scores must have non-null HGNC IDs")
        ev = merge_optional(ai, "hgnc_id", args.ai_scores)

    # v2.1: merge compensation data if provided
    if args.compensation:
        comp = pd.read_csv(args.compensation)
        ev = merge_optional(comp.rename(columns={"gene": "input_symbol"}),
                            "input_symbol", args.compensation)

    try:
        out, meta = run_consensus(ev, cfg2, controls_path=args.controls,
                                  tune_override=args.tune, sensitivity=args.sensitivity or cfg2.get("sensitivity", {}).get("enabled", False),
                                  mode=args.mode or cfg2["pipeline_mode"])
    except ValueError as exc:
        ap.error(str(exc))

    # ---- v2.1: mode-dependent metadata ----
    mode = args.mode or cfg2["pipeline_mode"]
    if mode not in ("auto", "validate", "rank", "full", "exploratory"):
        raise ValueError(f"unknown pipeline_mode: {mode}")
    meta["mode"] = mode

    clinGen_cov = meta.get("coverage", {}).get("clinGen_hi_score", 0)
    n_genes = len(out)
    top3_scores = out["consensus_score"].head(3).values if len(out) >= 3 else None

    # Warnings for sparse-evidence scenarios (Mode D)
    if args.warnings or mode in ("exploratory", "auto"):
        warnings = []
        if clinGen_cov < cfg2.get("warnings", {}).get("min_clingen_coverage", 0.2):
            warnings.append(
                f"⚠️ ClinGen covers only {clinGen_cov:.0%} of genes in this interval. "
                "Ranking mainly relies on gnomAD constraint, potentially favoring population-intolerant genes and "
                "missing important tissue-specific genes that have not been curated."
            )
        if n_genes > cfg2.get("warnings", {}).get("max_genes_for_single_driver", 20) and top3_scores is not None and len(top3_scores) >= 3:
            score_gap = float(top3_scores[0] - top3_scores[2])
            score_mean = float(out["consensus_score"].std())
            if score_gap < score_mean:
                warnings.append(
                    f"⚠️ This interval contains many genes ({n_genes}) with similar top-3 evidence scores"
                    f" (gap={score_gap:.2f}, σ={score_mean:.2f})."
                    "Multiple genes may contribute to the phenotype. A rank does not establish a unique causal gene. "
                    "Consider orthogonal validation of the top N candidates."
                )
        if clinGen_cov == 0:
            warnings.append(
                "⚠️ No genes in this interval have ClinGen dosage-sensitivity ratings. "
                "Ranking relies entirely on gnomAD constraint and HPA expression data, with low confidence. "
                "Review molecular biology literature before experimental validation."
            )
        meta["warnings"] = warnings

    # ---- v2.2: mode-specific analysis ----
    # Positive-control check (used by rank/full/validate; informative for exploratory)
    controls_check = []
    symbol_col = meta.get("symbol_col", "input_symbol")
    symbols = out[symbol_col].astype(str).str.strip().str.upper()
    rank_map = dict(zip(symbols, out["rank"]))
    for ctrl in meta.get("controls_list", []):
        if ctrl in rank_map:
            row = out.loc[symbols == ctrl].iloc[0]
            controls_check.append({
                "control": ctrl,
                "in_candidates": True,
                "rank": int(row["rank"]),
                "top10": bool(row["rank"] <= 10),
                "consensus_score": float(row["consensus_score"]),
                "clinGen_hi_score": row.get("clinGen_hi_score"),
                "gnomad_pLI": row.get("gnomad_pLI"),
                "compensation_class": row.get("compensation_class", ""),
            })
        else:
            controls_check.append({"control": ctrl, "in_candidates": False})
    meta["controls_check"] = controls_check
    meta["controls_top10_n"] = sum(1 for c in controls_check if c.get("top10"))
    meta["controls_total"] = len(controls_check)
    meta["controls_present_n"] = sum(1 for c in controls_check if c.get("in_candidates"))

    # Mode validate (A): known-driver evidence panel for the validation report
    if mode == "validate":
        panel = []
        for c in meta.get("controls_list", []):
            if symbol_col in out.columns:
                m = out.loc[symbols == c]
                if len(m):
                    r = m.iloc[0]
                    clingen = r.get("clinGen_hi_score")
                    pli = r.get("gnomad_pLI")
                    try:
                        clingen_ok = bool(clingen is not None and not (isinstance(clingen, float) and np.isnan(clingen)) and float(clingen) >= 2)
                    except (TypeError, ValueError):
                        clingen_ok = False
                    try:
                        pli_ok = bool(pli is not None and not (isinstance(pli, float) and np.isnan(pli)) and float(pli) >= 0.9)
                    except (TypeError, ValueError):
                        pli_ok = False
                    panel.append({
                        "gene": c, "rank": int(r["rank"]),
                        "clinGen_hi_score": clingen, "gnomad_pLI": pli,
                        "consensus_score": float(r["consensus_score"]),
                        "target_ok": (clingen_ok or pli_ok) and not bool(r["dosage_conflict"]),
                        "compensation_class": r.get("compensation_class", ""),
                    })
        meta["validation_panel"] = panel
        meta["validation_pass"] = any(p["target_ok"] for p in panel) if controls_check else None

    # v2.1: experiment recommendations
    experiments = []
    if mode == "exploratory" or not cfg2.get("experiments", {}).get("enabled", True):
        # Mode D does not assert a single top candidate; no experiment recommendation.
        meta["experiments"] = experiments
    else:
        top_gene = out.iloc[0] if len(out) > 0 else None
        if top_gene is not None and not top_gene["dosage_conflict"] and top_gene["evidence_status"] == "available":
            sym = top_gene.get("input_symbol", top_gene.get("gene_symbol", ""))
            evidence = []
            for c in meta.get("cols", []):
                if c in top_gene.index:
                    val = top_gene[c]
                    if not (isinstance(val, float) and np.isnan(val)) and val != 0:
                        evidence.append(f"{c}={val:.2f}" if isinstance(val, (int, float)) else f"{c}={val}")
            comp = top_gene.get("compensation_class", "")
            experiments.append({
                "title": f"Validation path for the leading candidate: {sym}",
                "body": (
                    f"**Evidence**: {', '.join(evidence) if evidence else 'See evidence panels'}\n\n"
                    f"**Expression status**: {comp if comp else 'Not evaluated (no scRNA-seq data)'}\n\n"
                    "Suggested validation steps:\n"
                    f"1. Use independent DNA evidence to check whether the deletion includes {sym}\n"
                    "2. Use Western blot to assess protein deficiency (without sufficient restoration at the translation level)\n"
                    f"3. If protein is deficient: design SINEUP-{sym} and test protein restoration in a cellular model\n"
                    f"4. If protein levels are normal: consider alternative genes below"
                )
            })
            # backup candidates
            top5_symbols = out["input_symbol" if "input_symbol" in out.columns else "hgnc_id"].head(5).tolist()
            backup = [s for s in top5_symbols[1:] if s != sym]
            experiments.append({
                "title": "Alternative-target strategy",
                "body": (
                    f"If {sym} protein levels are normal, consider these alternatives:\n"
                    + "\n".join(f"- {b}" for b in backup[:3])
                    + f"\n\nSwitching criterion: at least one orthogonal evidence source (pLI/ClinGen/expression) independent of {sym}"
                )
            })
            if len(backup) >= 3:
                experiments.append({
                    "title": "Suggestions when ranking is uncertain",
                    "body": (
                        "The leading candidates have similar evidence levels. Compare candidates "
                        "using appropriate controls and measure phenotypes after functional restoration:\n"
                        "- Restore expression of each candidate gene\n"
                        "- Compare phenotypic restoration\n"
                        "- Keep independent validation separate from weight-tuning data to avoid circular validation"
                    )
                })
        meta["experiments"] = experiments
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    with open(args.report, "w") as f:
        f.write(render_report(out, meta))
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    # pandas serializes numpy values and non-finite floats as standard JSON/null.
    record = {"stage": "consensus_v2", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "arguments": vars(args), "resolved_config": cfg2, "results": meta,
              "runtime": {"python": sys.version, "pandas": pd.__version__, "numpy": np.__version__},
              "code_sha256": sha256_file(__file__),
              "input_checksums": {str(p): sha256_file(p) for p in inputs if p},
              "output_sha256": sha256_file(args.out), "report_sha256": sha256_file(args.report)}
    audit_path.write_text(json.dumps(json.loads(pd.Series(record).to_json()), ensure_ascii=False, indent=2))

    print(f"wrote {args.out} with {len(out)} rows; report at {args.report}")
    if meta.get("tuning") and meta["tuning"].get("loo_hit_rate") is not None:
        print(f"LOO hit rate on positive controls: {meta['tuning']['loo_hit_rate']:.0%}")


if __name__ == "__main__":
    main()
