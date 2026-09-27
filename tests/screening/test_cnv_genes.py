"""Tests for GTF parsing and interval→gene mapping."""
import gzip

from screening.src.cnv.gene_order import genes_in_interval, parse_gtf_genes

GTF = """##description: test fixture
chr1\ttest\tgene\t100\t200\t.\t+\t.\tgene_id "ENSG1.1"; gene_type "protein_coding"; gene_name "AAA";
chr1\ttest\ttranscript\t100\t200\t.\t+\t.\tgene_id "ENSG1.1"; transcript_id "T1";
chr1\ttest\tgene\t300\t400\t.\t-\t.\tgene_id "ENSG2.1"; gene_type "lncRNA"; gene_name "LNC1";
chr7\ttest\tgene\t1000\t2000\t.\t+\t.\tgene_id "ENSG3.1"; gene_type "protein_coding"; gene_name "ELN";
chr7\ttest\tgene\t5000\t6000\t.\t+\t.\tgene_id "ENSG4.1"; gene_type "protein_coding"; gene_name "GTF2I";
chrM\ttest\tgene\t1\t100\t.\t+\t.\tgene_id "ENSG5.1"; gene_type "protein_coding"; gene_name "MTX";
"""


def _write_gtf(tmp_path):
    p = tmp_path / "test.gtf.gz"
    with gzip.open(p, "wt") as f:
        f.write(GTF)
    return p


def test_parse_filters_and_sorts(tmp_path):
    genes = parse_gtf_genes(_write_gtf(tmp_path))
    assert list(genes["gene_name"]) == ["AAA", "ELN", "GTF2I"]  # lncRNA + chrM excluded
    assert list(genes["chrom"]) == ["chr1", "chr7", "chr7"]
    assert genes.iloc[0]["start"] == 100


def test_genes_in_interval_overlap():
    import pandas as pd
    g = pd.DataFrame({"gene_name": ["A", "B", "C"], "gene_id": ["1", "2", "3"],
                      "chrom": ["chr7", "chr7", "chr7"],
                      "start": [100, 500, 900], "end": [200, 600, 1000],
                      "strand": ["+", "+", "+"]})
    hit = genes_in_interval(g, "chr7", 150, 550)
    assert list(hit["gene_name"]) == ["A", "B"]       # overlap, not containment
    assert genes_in_interval(g, "7", 150, 550).shape[0] == 2  # chr prefix tolerated
    assert genes_in_interval(g, "chr7", 700, 800).empty
