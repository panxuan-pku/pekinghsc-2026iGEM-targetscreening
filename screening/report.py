"""Human-facing views of an existing ranking; never changes scores or candidates."""
from html import escape
import math


SUMMARY_FIELDS = {
    "rank": "Rank within this candidate set",
    "symbol": "Approved gene symbol",
    "gene_id": "Unique Ensembl gene ID",
    "consensus_score": "Core evidence score, not a probability",
    "clinGen_hi_score": "ClinGen haploinsufficiency evidence used for scoring",
    "gnomad_LOEUF": "Quality-filtered LOEUF; lower values generally indicate stronger constraint",
    "gnomad_pLI": "Quality-filtered pLI",
    "evidence_status": "Availability of core evidence for ranking",
    "gnomad_status": "gnomAD matching and quality status",
    "interval_overlap": "Full or partial interval overlap; gene-list input does not confirm a deletion",
    "utr3_bp": "Selected transcript 3′UTR length (bp)",
    "selected": "Meets project design conditions and module budget; not experimental validation",
    "selection_reason": "Reason for design selection or exclusion",
    "ot_status": "OT query status; no record differs from query failure",
    "ot_disease_score": "OT disease-association annotation; excluded from the core score",
    "evidence_notes": "Review notes for partial overlap, missing evidence or conflicts",
}
INPUT_TYPES = {"interval": "Deletion interval", "gene_list": "Gene list"}


def output_names(input_type):
    if input_type not in INPUT_TYPES:
        raise ValueError(f"unsupported input type: {input_type}")
    return {name: f"{input_type}_{name}" for name in (
        "report.html", "summary.csv", "ranked.csv", "candidates.csv",
        "manifest.json", "config.yaml", "ot_snapshot.json",
    )}


REASONS = {
    "selected_within_module_budget": "Selected for design",
    "module_budget_limit": "Over module budget",
    "below_project_utr3_minimum": "3′UTR below minimum",
    "unresolved_transcript_or_utr": "Unresolved transcript / UTR",
}
OT = {"available": "Association available", "no_record": "No direct record", "query_failed": "Query failed"}


def display(value, digits=None):
    if value is None or str(value) in ("nan", "<NA>", ""):
        return "—"
    if digits is not None:
        try:
            number = float(value)
            return f"{number:.{digits}f}" if math.isfinite(number) else "—"
        except (TypeError, ValueError):
            pass
    return escape(str(value))


def write_summary(output, ranked, input_type):
    ranked[list(SUMMARY_FIELDS)].to_csv(output / output_names(input_type)["summary.csv"], index=False, encoding="utf-8-sig")


def report(output, ranked, manifest):
    input_type = manifest["input"]["input_type"]
    names = manifest["outputs"]
    rows = []
    for item in ranked.to_dict("records"):
        selected = bool(item["selected"])
        notes = display(item.get("evidence_notes"))
        notes = notes if notes != "—" else "No additional flags; interpret alongside the original evidence."
        gene = display(item["symbol"])
        reason = display(REASONS.get(item["selection_reason"], item["selection_reason"]))
        usable = item.get("evidence_status") != "no_usable_evidence"
        core = (f"HI <b>{display(item.get('clinGen_hi_score'), 0)}</b> · "
                f"LOEUF <b>{display(item.get('gnomad_LOEUF'), 3)}</b><br>"
                f"pLI <b>{display(item.get('gnomad_pLI'), 3)}</b>")
        if not usable:
            core += '<br><span class="caution">No usable core evidence</span>'
        details = f"""<details><summary>Details</summary><div class="gene-details">
<p><b>Position</b> {display(item.get('chrom'))}:{display(item.get('start'))}–{display(item.get('end'))}<br>
<b>Interval overlap</b> {display(item.get('interval_overlap'))} · <b>HGNC</b> {display(item.get('hgnc_id'))}</p>
<p><b>Score contributions</b> ClinGen {display(item.get('contrib_clinGen_hi_score'), 3)} / LOEUF {display(item.get('contrib_gnomad_LOEUF'), 3)} / pLI {display(item.get('contrib_gnomad_pLI'), 3)}<br>
<b>Evidence status</b> {display(item.get('evidence_status'))} · gnomAD {display(item.get('gnomad_status'))}<br>
<b>ClinGen status</b> {display(item.get('clinGen_haploinsufficiency_status'))}</p>
<p><b>Transcript</b> {display(item.get('transcript_id'))} ({display(item.get('transcript_choice'))})<br>
<b>3′UTR</b> {display(item.get('utr3_bp'), 0)} bp · {display(item.get('utr3_status'))}<br>
<b>OT annotation score</b> {display(item.get('ot_disease_score'), 3)} (excluded from ranking)</p>
<p class="note">{notes}</p></div></details>"""
        rows.append(f"""<tr><td class="rank">{display(item['rank'])}</td>
<td><strong class="gene">{gene}</strong><small>{display(item['gene_id'])}</small></td>
<td class="score">{display(item['consensus_score'], 3)}</td><td class="core">{core}</td>
<td>{display(OT.get(item.get('ot_status'), item.get('ot_status')))}</td>
<td><span class="badge {'positive' if selected else 'neutral'}">{reason}</span></td><td>{details}</td></tr>""")
    inp = manifest["input"]
    interval = inp.get("interval")
    scope = display(interval) if interval else f"Submitted gene list ({len(inp.get('genes') or [])} identifiers)"
    source = display(inp.get("source"))
    source_html = (f'<a href="{source}" target="_blank" rel="noopener noreferrer">{source}</a>'
                   if str(inp.get("source", "")).startswith(("https://", "http://")) else source)
    gaps = int(ranked.ot_status.eq("query_failed").sum())
    no_evidence = int(ranked.evidence_status.eq("no_usable_evidence").sum())
    status = "Complete" if manifest["status"] == "complete" else "Annotation gaps"
    alerts = []
    if gaps:
        alerts.append(f"OT queries failed for {gaps} candidates; missing annotations do not mean no association.")
    if no_evidence:
        alerts.append(f"{no_evidence} candidates lack usable core evidence; low scores are not negative evidence.")
    if "interval_overlap" in ranked and ranked.interval_overlap.eq("partial").any():
        alerts.append("Some genes only partially overlap the interval. Review their functional impact in the gene details.")
    alert_html = '<aside class="notice">' + '<br>'.join(alerts) + '</aside>' if alerts else ''
    fields = ''.join(f'<tr><td><code>{key}</code></td><td>{escape(value)}</td></tr>' for key, value in SUMMARY_FIELDS.items())
    references = ''.join(f'<li><b>{escape(name)}</b> · {display(record.get("version"))}</li>'
                         for name, record in manifest.get("references", {}).get("files", {}).items())
    content = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{INPUT_TYPES[input_type]} screening report · {display(manifest['disease_id'])}</title>
<style>
:root{{--ink:#193a3b;--muted:#5d7273;--green:#17695e;--line:#dbe5df;--paper:#fffefa}}
*{{box-sizing:border-box}}body{{margin:0;background:#f3f5ef;color:var(--ink);font:15px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif}}
a{{color:var(--green);text-underline-offset:3px}}main{{max-width:1340px;margin:auto;padding:42px 32px 60px}}
header{{border-top:5px solid var(--green);padding:25px 0}}.eyebrow{{font-size:12px;font-weight:700;letter-spacing:2px;color:var(--green)}}
h1{{font-size:36px;letter-spacing:-1px;margin:8px 0}}h2{{font-size:22px;margin:0 0 8px}}p{{margin:8px 0}}.subtitle,small,.muted{{color:var(--muted)}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:8px 0 26px}}.stat{{padding:19px 22px;background:var(--paper);border:1px solid var(--line);border-radius:12px}}
.stat strong{{display:block;font-size:28px;line-height:1.4}}.stat span{{font-size:13px;color:var(--muted)}}
section{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:24px;margin:18px 0}}
.scope{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.scope small{{display:block}}.source{{overflow-wrap:anywhere;font-size:13px}}
.notice{{border-left:4px solid #b48328;background:#fff4d9;padding:13px 18px;margin:18px 0;color:#72531c}}
.toolbar{{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;margin:16px 0}}
input{{font:inherit;border:1px solid #b7cbc2;background:white;padding:10px 13px;border-radius:8px;width:280px;max-width:100%}}
.button{{display:inline-block;border:1px solid #b6cfc5;background:#eaf4ee;padding:9px 16px;border-radius:8px;text-decoration:none;font-weight:600}}
.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}th{{text-align:left;background:#eef3ec;color:#4b6561;font-size:12px;white-space:nowrap;padding:13px 10px}}
td{{padding:15px 10px;border-bottom:1px solid #e6ece5;vertical-align:top}}tbody tr:hover{{background:#f7faf4}}.rank{{color:#74877f;font-variant-numeric:tabular-nums}}
.gene{{font-size:16px}}td small{{display:block;font-size:11px;white-space:nowrap}}.score{{font-size:18px;font-weight:700;font-variant-numeric:tabular-nums}}.core{{min-width:160px;font-size:12px}}
.badge{{display:inline-block;padding:4px 9px;border-radius:6px;white-space:nowrap;font-size:12px}}.positive{{background:#e2f1e6;color:#216448}}.neutral{{background:#f0eee5;color:#6d6651}}
summary{{cursor:pointer;color:var(--green);font-weight:600;white-space:nowrap}}details[open]>summary{{margin-bottom:12px}}.gene-details{{width:290px;font-size:12px;overflow-wrap:anywhere}}.note{{border-left:2px solid #b6cfc5;padding-left:10px}}
.caution{{color:#916b20}}.downloads{{display:flex;gap:10px;flex-wrap:wrap}}.technical{{margin-top:20px}}.technical p,.technical li{{font-size:13px}}code{{overflow-wrap:anywhere}}footer{{font-size:12px;color:var(--muted);padding:10px 0}}
@media(max-width:700px){{main{{padding:20px 14px}}h1{{font-size:28px}}.stats{{grid-template-columns:repeat(2,1fr);gap:8px}}.stat{{padding:14px}}section{{padding:18px}}.scope{{grid-template-columns:1fr}}}}
@media print{{input,.toolbar{{display:none}}main{{padding:0}}section{{break-inside:avoid}}.table-wrap{{overflow:visible}}}}
</style></head><body><main>
<header><div class="eyebrow">EVIDENCE-GUIDED PRIORITIZATION</div><h1>{INPUT_TYPES[input_type]} screening report</h1>
<p class="subtitle">Prioritize research candidates using genetic evidence to inform design and experiments.</p></header>
<div class="stats"><div class="stat"><strong>{len(ranked)}</strong><span>Candidate genes · all retained</span></div>
<div class="stat"><strong>{int(ranked.selected.sum())}</strong><span>Within design conditions and budget</span></div>
<div class="stat"><strong>{no_evidence}</strong><span>Without usable core evidence</span></div>
<div class="stat"><strong style="font-size:22px">{status}</strong><span>OT query failures: {gaps}</span></div></div>
<section><h2>01 / Analysis scope</h2><div class="scope"><div><small>Input type: {INPUT_TYPES[input_type]} ({input_type}) · {display(inp.get('genome_build'))} · 1-based inclusive</small><b>{scope}</b></div>
<div><small>Disease ID · completion time (UTC)</small><b>{display(manifest['disease_id'])}</b> · {display(manifest.get('completed_utc'))}</div></div>
<p class="source">Source: {source_html}</p><p class="muted">Interval input retrieves overlapping protein-coding genes; list input covers only the submitted genes. This scope does not automatically represent a patient.</p></section>
<section><h2>02 / Candidate ranking</h2><p class="muted">Read ranks and core evidence first, then design status. Scores compare this candidate set; they are not probabilities of disease or treatment success.</p>
{alert_html}<div class="toolbar"><label>Find a gene <input id="search" type="search" placeholder="Gene name or Ensembl ID" aria-label="Search gene names or IDs"></label>
<a class="button" href="{names['summary.csv']}" download>Download summary · {len(SUMMARY_FIELDS)} columns</a></div>
<div class="table-wrap"><table id="ranking"><thead><tr><th>Rank</th><th>Gene / unique ID</th><th>Evidence score</th><th>Core evidence</th><th>OT disease annotation</th><th>Design status</th><th>Evidence details</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p id="empty" hidden>No matching genes. Check the name or ID.</p>
<p class="muted">“—” means no usable value, not zero. ClinGen HI, LOEUF and pLI contribute to the core score; OT, HPA and IMPC are annotations only.</p>
<details class="technical"><summary>What does design status mean?</summary><p>Selected genes meet the project 3′UTR minimum of {display(ranked.project_min_utr3_bp.iloc[0], 0)} bp and fit the fixed module budget of {display(manifest.get('module_budget_bp'), 0)} bp, assuming {display(manifest.get('assumed_module_bp'), 0)} bp per module. Candidates outside the budget remain ranked.</p>
<p>These checks do not validate full constructs, sequence specificity, protein restoration or functional rescue. ClinGen recessive associations and IMPC mouse viability do not automatically exclude candidates.</p></details></section>
<section><h2>03 / Results and provenance</h2><p class="muted">Use the summary for routine review and the full table to audit evidence and calculations. Both retain the same candidates, ranks and raw scores.</p>
<div class="downloads"><a class="button" href="{names['summary.csv']}" download>Summary</a><a class="button" href="{names['ranked.csv']}" download>Full evidence · {len(ranked.columns)} columns</a><a class="button" href="{names['candidates.csv']}" download>Candidates</a></div>
<details class="technical"><summary>Summary field guide ({len(SUMMARY_FIELDS)} columns)</summary><div class="table-wrap"><table>{fields}</table></div></details>
<details class="technical"><summary>Why retain the full evidence table?</summary><p>The full table retains source names, standard IDs, raw and quality-filtered values, transcript choices, references, tissue expression and detailed OT responses. Repeated metadata supports gene-level traceability.</p><p>Flagged gnomAD values are retained but excluded from scoring. No direct record, a failed query and data not supplied are distinct states; an empty column in one run is not necessarily redundant.</p></details>
<details class="technical"><summary>Versions, sources and reproducibility</summary><ul>{references}</ul><p><a href="{names['manifest.json']}">Run manifest and checksums</a> · <a href="{names['config.yaml']}">Run configuration</a> · <a href="{names['ot_snapshot.json']}">OT response snapshot</a></p><p>Reproduction requires the same input, reference versions, configuration and OT snapshot. Displayed numbers are rounded; CSV files retain the original precision.</p></details></section>
<footer>For research prioritization; candidates require experimental and source-evidence review. Works offline without external fonts or scripts.</footer>
</main><script>
const search=document.getElementById('search');
search.addEventListener('input',()=>{{let visible=0;const term=search.value.trim().toLowerCase();
for(const row of document.querySelectorAll('#ranking tbody tr')){{row.hidden=!row.cells[1].textContent.toLowerCase().includes(term);if(!row.hidden)visible++;}}
document.getElementById('empty').hidden=visible>0;}});
</script></body></html>"""
    (output / names["report.html"]).write_text(content, encoding="utf-8")
