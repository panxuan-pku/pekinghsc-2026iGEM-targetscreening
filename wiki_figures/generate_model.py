"""Export three Wiki Model figures; captions and source records stay outside images."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import tempfile

# Keep font caches outside the user's home and keep optional imports headless.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "igem-wiki-matplotlib"))
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, FancyArrowPatch, Patch
import seaborn as sns

# Team wiki-plot-spec.md: shared palette and plotting defaults.
WIKI = {
    "blue": "#169bb7", "green": "#059669", "green_deep": "#047857",
    "ink": "#1a3348", "muted": "#7a8896", "line": "#dfe7ed",
    "paper": "#f6f9fa", "white": "#ffffff", "note": "#0e7490", "warn": "#92400e",
}
WIKI_CAT = [WIKI[k] for k in ("blue", "green", "ink", "note", "muted")]
WIKI_SEQ = LinearSegmentedColormap.from_list("wiki_seq", ["#f6f9fa", "#c5e8ef", "#169bb7", "#0e7490"])
WIKI_DIV = LinearSegmentedColormap.from_list("wiki_div", ["#047857", "#059669", "#f6f9fa", "#169bb7", "#0e7490"])


def apply_wiki_plot_style():
    sns.set_theme(style="ticks", palette=WIKI_CAT, color_codes=False)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Inter", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 13, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.labelsize": 11, "axes.labelcolor": WIKI["ink"],
        "axes.edgecolor": WIKI["line"], "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.facecolor": WIKI["white"], "figure.facecolor": WIKI["white"],
        "xtick.color": WIKI["ink"], "ytick.color": WIKI["ink"],
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "text.color": WIKI["ink"],
        "grid.color": WIKI["line"], "grid.linewidth": 0.7,
        "legend.frameon": False, "legend.fontsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": WIKI["white"],
    })


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
MODEL = OUT / "model"


def canvas(title):
    fig = plt.figure(figsize=(7.2, 4.2))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 7.2), ylim=(0, 4.2))
    ax.axis("off")
    ax.text(.22, 3.94, title, fontsize=13, weight="semibold", va="center")
    return fig, ax


def box(ax, x, y, w, h, title, detail="", color="blue"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=WIKI["paper"], edgecolor=WIKI["line"], lw=.8))
    ax.plot([x, x + w], [y + h, y + h], color=WIKI[color], lw=1.4)
    ax.text(x + w / 2, y + h * (.68 if detail else .5), title,
            ha="center", va="center", fontsize=11, weight="semibold")
    if detail:
        ax.text(x + w / 2, y + h * .27, detail, ha="center", va="center", fontsize=9,
                color=WIKI["ink"], linespacing=1.3)


def arrow(ax, start, end, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
                                lw=.8, linestyle="--" if dashed else "-", color=WIKI["ink"]))


def save(fig, name):
    # Preserve the team's exact full-width dimensions rather than auto-cropping.
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(MODEL / f"{name}.svg", metadata={"Date": None})
        fig.savefig(MODEL / f"{name}.png", dpi=300)
    plt.close(fig)


def workflow():
    fig, ax = canvas("From a deletion to a reviewable shortlist")
    box(ax, .22, 2.96, 1.80, .60, "Deletion interval", "GRCh38 overlap")
    box(ax, .22, 2.10, 1.80, .60, "Candidate gene list", "IDs or symbols")
    ax.plot([2.02, 2.23, 2.23, 2.02], [3.26, 3.26, 2.40, 2.40], color=WIKI["ink"], lw=.8)
    arrow(ax, (2.23, 2.78), (2.48, 2.78))
    box(ax, 2.50, 2.36, 1.97, .84, "Resolve identity", "HGNC + GENCODE\nUnique coding genes")
    arrow(ax, (4.47, 2.78), (4.90, 2.78))
    box(ax, 4.92, 2.36, 2.06, .84, "Evidence ranking", "ClinGen + gnomAD\nScore = 3H + 2L + 2P")
    arrow(ax, (5.95, 2.36), (5.95, 1.58))
    box(ax, 4.92, .68, 2.06, .88, "Design checks", "Transcript · 3′UTR\nModule budget", "green")
    arrow(ax, (4.92, 1.12), (4.49, 1.12))
    box(ax, 2.50, .68, 1.97, .88, "Reviewable outputs", "All ranks retained\nSeparate selection flags", "green")
    box(ax, .22, .68, 1.80, .88, "Context only", "Open Targets · HPA\nOptional IMPC", "muted")
    arrow(ax, (2.02, 1.12), (2.48, 1.12), dashed=True)
    ax.text(.22, .30, "Solid arrows: decision workflow     Dashed arrow: annotations without rescoring",
            fontsize=9, color=WIKI["muted"])
    save(fig, "model-screening-workflow")


def contributions(rows):
    top = rows[:6]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.subplots_adjust(left=.13, right=.79, bottom=.17, top=.76)
    fig.text(.04, .94, "WHS ranking: where each score comes from", fontsize=13, weight="semibold")
    labels = ("ClinGen (3H)", "LOEUF (2L)", "pLI (2P)")
    terms = ("contrib_clinGen_hi_score", "contrib_gnomad_LOEUF", "contrib_gnomad_pLI")
    patterns = ("", "///", "...")
    left = [0.] * len(top)
    for term, label, color, pattern in zip(terms, labels, WIKI_CAT, patterns):
        vals = [float(r[term]) for r in top]
        ax.barh(range(len(top)), vals, left=left, height=.56, color=color, hatch=pattern,
                edgecolor=WIKI["white"], linewidth=.4, label=label, zorder=3)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(range(len(top)), [r["gene_name"] for r in top])
    ax.set_ylim(len(top) - .45, -.55)
    ax.set_xlim(0, 7.7)
    ax.set_xticks(range(8))
    ax.set_xlabel("Weighted score contribution (points)", labelpad=10)
    ax.grid(axis="y", which="major")
    ax.grid(axis="x", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for i, row in enumerate(top):
        ax.text(left[i] + .10, i, f"{left[i]:.2f}", va="center", fontsize=9)
        value = row["clinGen_hi_score"]
        status = "Unavailable" if not value else ("HI = 0" if float(value) == 0 else f"HI = {float(value):g}")
        ax.text(1.06, i, status, transform=ax.get_yaxis_transform(), va="center", fontsize=9,
                color=WIKI["muted"] if not value else WIKI["ink"])
    fig.text(.83, .79, "ClinGen record", fontsize=9, weight="semibold")
    fig.legend(handles=[Patch(facecolor=c, edgecolor="white", hatch=p, label=l)
                        for c, p, l in zip(WIKI_CAT, patterns, labels)],
               loc="upper left", bbox_to_anchor=(.12, .89), ncol=3, handlelength=1.5, columnspacing=1.5)
    save(fig, "model-whs-score-contributions")


def design():
    fig, ax = canvas("From rank to design eligibility")
    ax.text(.22, 3.51, "Apply the checks to each candidate, in ranking order", fontsize=11)
    positions = (.22, 2.57, 4.92)
    boxes = (
        ("1  Resolve transcript", "MANE Select → canonical\n→ sole coding transcript"),
        ("2  Check 3′UTR", "Spliced length\nAt least 30 bp"),
        ("3  Apply budget", "4,400 bp available\n300 bp per module"),
    )
    failures = ("Unresolved transcript\nor incomplete UTR", "3′UTR below\n30 bp", "Module budget\nexhausted")
    for x, (title, detail), failure in zip(positions, boxes, failures):
        box(ax, x, 2.26, 2.06, .88, title, detail, "green")
        arrow(ax, (x + 1.03, 2.24), (x + 1.03, 1.78), dashed=True)
        ax.text(x + 1.03, 1.50, failure, ha="center", va="center", fontsize=9.5,
                color=WIKI["warn"], linespacing=1.3)
    for x in positions[:2]:
        arrow(ax, (x + 2.06, 2.70), (x + 2.33, 2.70))
    ax.plot([.22, 6.98], [.99, .99], color=WIKI["line"], lw=.8)
    ax.text(.22, .68, "Pass all checks", fontsize=11, weight="semibold", color=WIKI["green"])
    ax.text(2.18, .68, "Selected for design; up to 14 modules at these settings", fontsize=9.5)
    ax.text(.22, .30, "Fail a check", fontsize=11, weight="semibold", color=WIKI["warn"])
    ax.text(2.18, .30, "Retained in the ranked table with a reason", fontsize=9.5)
    save(fig, "model-screening-design-checks")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed interval output directory")
    args = parser.parse_args()
    run = args.run.resolve()
    manifest_path = run / "interval_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] not in {"complete", "completed_with_annotation_gaps"}:
        raise ValueError("The source run is not complete")
    for name, expected in manifest["output_sha256"].items():
        if digest(run / name) != expected:
            raise ValueError(f"Source output checksum mismatch: {name}")
    for name, expected in manifest["code"]["file_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Pipeline changed since source run: {name}; rerun screening first")
    if manifest["input"]["interval"] != "chr4:337779-2009235" or manifest["input"]["genome_build"] != "GRCh38":
        raise ValueError("This WHS figure requires the documented GRCh38 interval")
    if (manifest["module_budget_bp"], manifest["assumed_module_bp"]) != (4400, 300):
        raise ValueError("Design figure assumes a 4,400 bp budget and 300 bp modules")
    with (run / "interval_ranked.csv").open(encoding="utf-8-sig") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda r: int(r["rank"]))
    if [int(r["rank"]) for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Expected unique, sequential ranks")
    if set(r["gene_id"] for r in rows) != set(manifest["candidate_gene_ids"]):
        raise ValueError("Candidate set differs from the manifest")
    for row in rows:
        total = sum(float(row[k]) for k in ("contrib_clinGen_hi_score", "contrib_gnomad_LOEUF", "contrib_gnomad_pLI"))
        if not math.isclose(total, float(row["consensus_score"]), abs_tol=1e-9):
            raise ValueError("Contributions do not sum to the recorded score")
        if int(row["project_min_utr3_bp"]) != 30:
            raise ValueError("Design figure assumes a 30 bp UTR threshold")
    MODEL.mkdir(exist_ok=True)
    apply_wiki_plot_style()
    workflow()
    contributions(rows)
    design()
    figures = [
        {"file": "model-screening-workflow", "title": "01 · Screening workflow", "placement": "Model: below Two input routes, one ranking",
         "alt": "Interval and gene-list inputs converge on identity checks, evidence ranking and design selection, with context annotations kept separate from scoring.",
         "caption": "Conceptual workflow of the current screening entry point. ClinGen and gnomAD determine priority; Open Targets, HPA and optional IMPC provide annotations without changing rank. All candidates remain in the output, with separate design-selection flags."},
        {"file": "model-whs-score-contributions", "title": "02 · WHS score contributions", "placement": "Model: below Evidence sets priority; context aids review",
         "alt": "Stacked score contributions for the six highest-ranked WHS candidates, with separate ClinGen, LOEUF and pLI segments and a ClinGen availability column.",
         "caption": f"Top six of {len(rows)} protein-coding candidates from GRCh38 chr4:337779–2009235 (ClinGen ISCA-37429), using the completed run at {manifest['completed_utc']}. Bars show recorded weighted contributions (3H + 2L + 2P): H is HI/3, L is reversed min–max LOEUF, and P is min–max pLI. Scaling uses all {len(rows)} candidates, not only the six displayed. ClinGen ‘Unavailable’ differs from an observed HI score of 0; both contribute no points here. The two gnomAD terms overlap in information. These are computational priorities, not probabilities or experimental rescue results."},
        {"file": "model-screening-design-checks", "title": "03 · Design checks", "placement": "Model: inside Design assumptions to review",
         "alt": "Three sequential design checks: resolve a transcript, require a spliced 3-prime UTR of at least 30 bp, and apply the module budget in rank order. Excluded candidates remain in the full table.",
         "caption": "Illustration of the current project's design rules. Transcript selection must be unambiguous and UTR annotation complete. The 4,400 bp budget assumes 300 bp per module, permitting at most 14 eligible candidates. Passing does not establish binding specificity, an optimal gene combination, complete-vector feasibility or functional rescue; missing core evidence does not itself exclude a gene from this design shortlist."},
    ]
    record = {"source_run": str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run),
              "source_manifest_sha256": digest(manifest_path), "source_run_manifest": manifest,
              "top_six": [{k: r[k] for k in ("rank", "gene_name", "gene_id", "consensus_score", "clinGen_hi_score", "contrib_clinGen_hi_score", "contrib_gnomad_LOEUF", "contrib_gnomad_pLI")} for r in rows[:6]],
              "generator_sha256": digest(Path(__file__)), "plot_libraries": {"matplotlib": matplotlib.__version__, "seaborn": sns.__version__},
              "figures": figures}
    (MODEL / "provenance.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    cards = []
    for figure in figures:
        name = figure["file"]
        cards.append(f'<section><h2>{html.escape(figure["title"])}</h2><p>{html.escape(figure["placement"])}</p>'
                     f'<img src="model/{name}.svg" alt="{html.escape(figure["alt"])}">'
                     f'<p class="caption">{html.escape(figure["caption"])}</p>'
                     f'<a href="model/{name}.svg">SVG master</a> · <a href="model/{name}.png">300 dpi PNG</a></section>')
    (OUT / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Model figure review</title><style>body{max-width:1000px;margin:40px auto;padding:0 24px;font:16px/1.6 Arial,sans-serif;color:#1a3348;background:white}'
        'h1,h2{line-height:1.25}section{padding:28px 0;border-top:1px solid #dfe7ed}img{display:block;width:100%;max-width:900px;margin:18px auto}'
        'a{color:#0e7490}.caption{background:#f6f9fa;padding:14px}header p{color:#7a8896}</style>'
        '<header><h1>Model · Figure review</h1><p>Local staging only. Review captions, then upload the chosen SVGs to iGEM static hosting. No Wiki page has been changed.</p></header>'
        + ''.join(cards) + '</html>')
    print(f"[OK] Verified {len(rows)} candidates and all recorded output/source-code checksums")
    print(f"[OK] Exported 3 SVGs and 3 PNGs to {MODEL}")
    print(f"[OK] Captions and provenance: {MODEL / 'provenance.json'}")
    print(f"[OK] Preview: {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
