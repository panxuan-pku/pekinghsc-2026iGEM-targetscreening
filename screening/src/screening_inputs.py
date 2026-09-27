"""One GRCh38 candidate contract for intervals and explicit gene lists."""
import gzip
import re

import pandas as pd

from screening.src.cnv.gene_order import genes_in_interval, parse_gtf_genes
from screening.src.normalize import load_alias_map


def build_candidates(gtf, hgnc_path, *, interval=None, genes=None):
    if (interval is None) == (genes is None):
        raise ValueError("provide exactly one interval or gene list")
    annotation = parse_gtf_genes(gtf)
    if annotation.gene_id.duplicated().any():
        raise ValueError("annotation has ambiguous gene positions")
    hgnc = pd.read_csv(hgnc_path, sep="\t", dtype=str, keep_default_na=False)
    required = {"hgnc_id", "symbol", "status", "ensembl_gene_id"}
    if not required <= set(hgnc.columns):
        raise ValueError("HGNC requires: " + ", ".join(sorted(required)))
    hgnc = hgnc[hgnc.status.eq("Approved")].copy()
    if hgnc.hgnc_id.duplicated().any() or hgnc.symbol.duplicated().any():
        raise ValueError("ambiguous approved HGNC identities")
    if interval is not None:
        match = re.fullmatch(r"(?:chr)?([1-9][0-9]?|X|Y):(\d+)-(\d+)", interval)
        if not match or not 1 <= int(match[2]) <= int(match[3]):
            raise ValueError("interval must be GRCh38 chrN:start-end, 1-based inclusive")
        chrom, start, end = "chr" + match[1], int(match[2]), int(match[3])
        selected = genes_in_interval(annotation, chrom, start, end)
        selected["input_symbol"] = selected.gene_name
        selected["interval_overlap"] = ["full" if start <= s and e <= end else "partial"
                                         for s, e in zip(selected.start, selected.end)]
        selected["input_interval"] = interval
    else:
        aliases = load_alias_map(hgnc_path)
        rows = []
        for value in genes:
            label = str(value).strip()
            if label.startswith("ENSG"):
                hits = hgnc[hgnc.ensembl_gene_id.eq(label.split(".")[0])]
            else:
                hid = label if label.startswith("HGNC:") else aliases.get(label.lower())
                hits = hgnc[hgnc.hgnc_id.eq(hid)]
            if len(hits) != 1 or not hits.iloc[0].ensembl_gene_id:
                raise ValueError(f"unresolved or ambiguous gene identity: {label}")
            gid = hits.iloc[0].ensembl_gene_id
            positions = annotation[annotation.gene_id.eq(gid)]
            if len(positions) != 1:
                raise ValueError(f"no unique protein-coding position in pinned annotation: {label} ({gid})")
            row = positions.iloc[0].to_dict()
            row.update(input_symbol=label, interval_overlap="not_provided", input_interval="")
            rows.append(row)
        selected = pd.DataFrame(rows)
    if selected.empty:
        raise ValueError("empty candidate set")
    mapping = hgnc[hgnc.ensembl_gene_id.isin(selected.gene_id)]
    if mapping.ensembl_gene_id.duplicated().any():
        raise ValueError("ambiguous Ensembl to HGNC identity")
    selected = selected.merge(mapping[["ensembl_gene_id", "hgnc_id", "symbol"]],
                              left_on="gene_id", right_on="ensembl_gene_id", how="left", validate="many_to_one")
    if selected.hgnc_id.isna().any():
        raise ValueError("unresolved HGNC IDs: " + ", ".join(selected.loc[selected.hgnc_id.isna(), "gene_id"]))
    if selected.hgnc_id.duplicated().any():
        raise ValueError("duplicate candidate identity; provide each gene once (original labels are not rewritten)")
    selected["genome_build"] = "GRCh38"
    selected["input_type"] = "interval" if interval is not None else "gene_list"
    return selected.sort_values("gene_id").reset_index(drop=True)


def union_blocks(blocks):
    merged = []
    for start, end in sorted(blocks):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def transcript_design(gtf, candidates):
    """Count spliced bases beyond CDS + stop codon; never count introns as UTR."""
    wanted = set(candidates.gene_id)
    transcripts = {}
    opener = gzip.open if str(gtf).endswith(".gz") else open
    with opener(gtf, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) != 9 or parts[2] not in {"transcript", "exon", "CDS", "stop_codon"}:
                continue
            attributes = re.findall(r'(\w+) "([^"]*)"', parts[8])
            attrs = dict(attributes)
            gid, tid = attrs.get("gene_id", "").split(".")[0], attrs.get("transcript_id", "")
            if gid not in wanted or not tid:
                continue
            tx = transcripts.setdefault((gid, tid), {"tags": set(), "exon": [], "CDS": [], "stop_codon": [], "strand": parts[6]})
            tx["tags"].update(v for k, v in attributes if k == "tag")
            if parts[2] != "transcript":
                tx[parts[2]].append((int(parts[3]), int(parts[4])))
    rows = []
    for gid in candidates.gene_id:
        options = [(tid, tx) for (g, tid), tx in transcripts.items() if g == gid and tx["CDS"]]
        chosen, choice = [], "unresolved"
        for tag in ("MANE_Select", "Ensembl_canonical"):
            tagged = [(tid, tx) for tid, tx in options if tag in tx["tags"]]
            if tagged:
                chosen, choice = tagged, tag
                break
        if not chosen and len(options) == 1:
            chosen, choice = options, "only_coding_transcript"
        row = {"gene_id": gid, "transcript_id": "", "transcript_choice": choice,
               "utr3_bp": None, "utr3_status": "ambiguous_or_missing_transcript"}
        if len(chosen) == 1:
            tid, tx = chosen[0]
            row["transcript_id"] = tid
            if tx["exon"] and tx["stop_codon"] and "cds_end_NF" not in tx["tags"]:
                coding = tx["CDS"] + tx["stop_codon"]
                exons = union_blocks(tx["exon"])
                if tx["strand"] == "+":
                    end = max(e for _, e in coding)
                    length = sum(max(0, e - max(s, end + 1) + 1) for s, e in exons)
                else:
                    start = min(s for s, _ in coding)
                    length = sum(max(0, min(e, start - 1) - s + 1) for s, e in exons)
                row.update(utr3_bp=length, utr3_status="ok")
            else:
                row["utr3_status"] = "incomplete_coding_annotation"
        rows.append(row)
    return candidates.merge(pd.DataFrame(rows), on="gene_id", validate="one_to_one")
