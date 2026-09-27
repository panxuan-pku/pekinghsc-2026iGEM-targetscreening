"""Input conversion uses synthetic files, never downloaded scientific results."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "screening/prepare_input.py"


def convert(tmp_path, kind, file_format, text, *options):
    raw = tmp_path / "download.txt"
    raw.write_text(text, encoding="utf-8")
    output = tmp_path / "prepared.json"
    result = subprocess.run([sys.executable, str(SCRIPT), kind, "--input", str(raw),
                             "--format", file_format, "--source", "synthetic fixture",
                             "--output", str(output), *options], capture_output=True, text=True)
    assert raw.read_text(encoding="utf-8") == text
    return result, output


@pytest.mark.parametrize("file_format,text,options", [
    ("txt", "\ufeffOLD_A\nHGNC:2\n\n", []),
    ("csv", 'Gene stable ID,note\nOLD_A,"a,b"\nHGNC:2,\n', ["--gene-column", "Gene stable ID"]),
    ("tsv", "Gene stable ID\tGene name\nOLD_A\tA\nHGNC:2\tB\n", ["--gene-column", "Gene stable ID"]),
])
def test_gene_formats_preserve_identifiers_and_provenance(tmp_path, file_format, text, options):
    result, output = convert(tmp_path, "genes", file_format, text, *options)
    assert result.returncode == 0, result.stderr
    assert "[OK] Input preparation complete" in result.stdout
    assert "2 gene identifiers" in result.stdout
    data = json.loads(output.read_text())
    assert data["genes"] == ["OLD_A", "HGNC:2"]
    assert data["source"] == "synthetic fixture"
    assert data["original_file"]["sha256"] == hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize("file_format,text,options,error", [
    ("txt", "", [], "empty"),
    ("txt", "A\nA\n", [], "duplicate"),
    ("txt", "A;B\n", [], "one identifier"),
    ("csv", "symbol\nA\n", ["--gene-column", "gene"], "missing columns"),
    ("csv", "gene,gene\nA,B\n", ["--gene-column", "gene"], "duplicate table headers"),
    ("tsv", "gene\tother\n\tx\n", ["--gene-column", "gene"], "invalid gene"),
    ("csv", "gene,other\nA\n", ["--gene-column", "gene"], "number of columns"),
])
def test_invalid_genes_leave_no_output(tmp_path, file_format, text, options, error):
    result, output = convert(tmp_path, "genes", file_format, text, *options)
    assert result.returncode != 0 and error in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("file_format,text,options", [
    ("bed", "track name=example\n#comment\nchr4\t99\t699\tdeletion\n", []),
    ("tsv", "chrom\tstart\tend\n4\t100\t699\n",
     ["--coordinates", "1-based-inclusive", "--chrom-column", "chrom", "--start-column", "start", "--end-column", "end"]),
    ("csv", "chrom,start,end\n4,99,699\n",
     ["--coordinates", "0-based-half-open", "--chrom-column", "chrom", "--start-column", "start", "--end-column", "end"]),
])
def test_interval_formats_have_same_boundaries(tmp_path, file_format, text, options):
    result, output = convert(tmp_path, "interval", file_format, text, "--genome-build", "GRCh38", *options)
    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text())
    assert data["interval"] == "chr4:100-699"
    assert data["coordinate_system"] == "1-based-inclusive"


@pytest.mark.parametrize("text,options,error", [
    ("chr4\t100\t100\n", [], "nonempty"),
    ("chr4\t200\t100\n", [], "nonempty"),
    ("chr4\t-1\t100\n", [], "integers"),
    ("chr4\t1.5\t100\n", [], "integers"),
    ("chr99\t0\t100\n", [], "chromosome"),
    ("chr4\t0\t100\nchr4\t200\t300\n", [], "found 2 intervals"),
    ("chr4\t0\t100\n", ["--coordinates", "1-based-inclusive"], "BED requires"),
    ("chr4\t0\t100\n", ["--row", "0"], "--row must"),
])
def test_invalid_bed_is_not_silently_fixed(tmp_path, text, options, error):
    result, output = convert(tmp_path, "interval", "bed", text, "--genome-build", "GRCh38", *options)
    assert result.returncode != 0 and error in result.stderr
    assert not output.exists()


def test_select_one_interval_and_keep_selection(tmp_path):
    result, output = convert(tmp_path, "interval", "bed", "chr4\t0\t100\nchr4\t200\t300\n",
                             "--genome-build", "GRCh38", "--row", "2")
    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text())
    assert data["interval"] == "chr4:201-300"
    assert data["original_file"]["selected_row"] == 2
    assert data["original_file"]["record_count"] == 2


@pytest.mark.parametrize("options,error", [
    (["--genome-build", "GRCh37"], "invalid choice"),
    (["--genome-build", "GRCh38"], "require --coordinates"),
])
def test_unknown_assembly_and_coordinate_convention_fail(tmp_path, options, error):
    result, output = convert(tmp_path, "interval", "tsv", "chrom\tstart\tend\n4\t1\t100\n", *options)
    assert result.returncode != 0 and error in result.stderr
    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path):
    output = tmp_path / "prepared.json"
    output.write_text("keep me")
    result, _ = convert(tmp_path, "genes", "txt", "A\n")
    assert result.returncode != 0
    assert output.read_text() == "keep me"


CLINGEN = ("#ClinGen Region Curation Results\n#date\n"
           "#Genomic Locations are reported on GRCh38 (hg38)\n"
           "#ISCA ID\tISCA Region Name\tGenomic Location\n"
           "ISCA-TEST1\tfirst synthetic region\tchr4:100-699\n"
           "ISCA-TEST2\tsecond synthetic region\tchr5:1-100\n")


def test_clingen_comments_header_and_stable_region_id(tmp_path):
    result, output = convert(tmp_path, "interval", "clingen-tsv", CLINGEN,
                             "--genome-build", "GRCh38", "--region-id", "ISCA-TEST2")
    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text())
    assert data["interval"] == "chr5:1-100"
    assert data["original_file"]["region_name"] == "second synthetic region"
    assert data["original_file"]["selected_row"] == 2
    assert data["original_file"]["record_count"] == 2


@pytest.mark.parametrize("text,region,error", [
    (CLINGEN.replace("GRCh38", "GRCh37"), "ISCA-TEST1", "explicit GRCh38"),
    (CLINGEN, "ISCA-NOTFOUND", "found 0"),
    (CLINGEN.replace("ISCA-TEST2", "ISCA-TEST1"), "ISCA-TEST1", "found 2"),
    (CLINGEN.replace("Genomic Location\n", "location\n"), "ISCA-TEST1", "missing required columns"),
])
def test_clingen_rejects_wrong_build_and_ambiguous_selection(tmp_path, text, region, error):
    result, output = convert(tmp_path, "interval", "clingen-tsv", text,
                             "--genome-build", "GRCh38", "--region-id", region)
    assert result.returncode != 0 and error in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("payload", [
    {"schema_version": 1, "source": "test", "input_type": "gene_list", "genes": []},
    {"schema_version": 1, "source": "test", "input_type": "gene_list", "genes": ["A"], "interval": "chr4:1-10"},
    {"schema_version": 1, "source": "test", "input_type": "interval", "interval": "chr4:1-10",
     "genome_build": "GRCh37", "coordinate_system": "1-based-inclusive"},
])
def test_runner_rejects_bad_input_before_references_or_output(tmp_path, payload):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    result = subprocess.run([sys.executable, str(ROOT / "screening/run.py"), "run",
                             "--input-file", str(path), "--references", str(tmp_path / "absent-refs"),
                             "--output", str(tmp_path / "result")], capture_output=True, text=True)
    assert result.returncode != 0 and "Screening stopped" in result.stderr
    assert "No such file" not in result.stderr
    assert not (tmp_path / "result").exists()
