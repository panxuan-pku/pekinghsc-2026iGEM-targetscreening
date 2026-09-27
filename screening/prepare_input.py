#!/usr/bin/env python3
"""Convert downloaded gene tables or deletion intervals into a screening input (no network)."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def check_genes(genes):
    if not isinstance(genes, list) or not genes:
        raise ValueError("gene list is empty or invalid")
    seen = set()
    for row, gene in enumerate(genes, 1):
        if not isinstance(gene, str) or not re.fullmatch(r"(?:HGNC:[0-9]+|[A-Za-z0-9][A-Za-z0-9_.-]*)", gene):
            raise ValueError(f"invalid gene value at record {row}: {gene!r}; use one identifier per cell")
        if gene in seen:
            raise ValueError(f"duplicate gene at record {row}: {gene}; review the source, no rows were merged")
        seen.add(gene)


def normalize_interval(chrom, start, end, coordinates):
    chrom = chrom.strip()
    if chrom.startswith("chr"):
        chrom = chrom[3:]
    if not re.fullmatch(r"(?:[1-9]|1[0-9]|2[0-2]|X|Y)", chrom):
        raise ValueError(f"unsupported human chromosome: {chrom}")
    if not re.fullmatch(r"[0-9]+", start) or not re.fullmatch(r"[0-9]+", end):
        raise ValueError("coordinates must be integers without commas or decimal points")
    start, end = int(start), int(end)
    if coordinates == "0-based-half-open":
        start += 1
    elif coordinates != "1-based-inclusive":
        raise ValueError("specify 0-based-half-open or 1-based-inclusive coordinates")
    if not 1 <= start <= end:
        raise ValueError("interval must be nonempty, with start <= end and valid coordinates")
    return f"chr{chrom}:{start}-{end}"


def load_prepared_input(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("unsupported prepared input schema")
    if not isinstance(data.get("source"), str) or not data["source"].strip():
        raise ValueError("prepared input requires a source")
    if data.get("input_type") == "gene_list":
        if "interval" in data:
            raise ValueError("prepared input cannot contain both genes and interval")
        check_genes(data.get("genes"))
    elif data.get("input_type") == "interval":
        if "genes" in data:
            raise ValueError("prepared input cannot contain both genes and interval")
        if data.get("genome_build") != "GRCh38" or data.get("coordinate_system") != "1-based-inclusive":
            raise ValueError("prepared interval must be GRCh38, 1-based-inclusive")
        match = re.fullmatch(r"chr([^:]+):([0-9]+)-([0-9]+)", str(data.get("interval", "")))
        if not match or normalize_interval(*match.groups(), "1-based-inclusive") != data["interval"]:
            raise ValueError("invalid prepared interval")
    else:
        raise ValueError("prepared input must contain a gene_list or interval")
    return data


def read_table(path, file_format, columns):
    if any(not column for column in columns):
        raise ValueError("specify the input column names explicitly")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="," if file_format == "csv" else "\t")
        headers = reader.fieldnames or []
        if len(set(headers)) != len(headers):
            raise ValueError("duplicate table headers")
        if not set(columns) <= set(headers):
            raise ValueError(f"missing columns: {sorted(set(columns) - set(headers))}; available: {headers}")
        rows = []
        for number, row in enumerate(reader, 1):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"inconsistent number of columns at record {number}")
            rows.append([row[column].strip() for column in columns])
    return rows


def prepare(args):
    if not args.source.strip():
        raise ValueError("provide a nonempty source URL, accession or citation")
    data = {"schema_version": 1, "source": args.source.strip(),
            "prepared_utc": datetime.now(timezone.utc).isoformat(),
            "original_file": {"path": str(args.input.resolve()), "format": args.format,
                              "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest()}}
    if args.kind == "genes":
        if args.format == "txt":
            if args.gene_column:
                raise ValueError("TXT has no header; do not specify --gene-column")
            genes = [line.strip() for line in args.input.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        else:
            genes = [row[0] for row in read_table(args.input, args.format, [args.gene_column])]
            data["original_file"]["gene_column"] = args.gene_column
        check_genes(genes)
        data.update(input_type="gene_list", genes=genes)
        message = f"{len(genes)} gene identifiers; identity checks run during screening"
    else:
        if args.format == "clingen-tsv":
            if not args.region_id:
                raise ValueError("ClinGen region TSV requires --region-id, e.g. ISCA-37429")
            if args.row is not None or any((args.chrom_column, args.start_column, args.end_column)):
                raise ValueError("ClinGen region TSV selects by --region-id, not row or column arguments")
            if args.coordinates not in (None, "1-based-inclusive"):
                raise ValueError("ClinGen Genomic Location uses 1-based-inclusive coordinates")
            lines = args.input.read_text(encoding="utf-8-sig").splitlines()
            header = next((i for i, line in enumerate(lines) if line.startswith("#ISCA ID\t")), None)
            if header is None or not any("Genomic Locations are reported on GRCh38" in line for line in lines[:header]):
                raise ValueError("expected ClinGen region TSV with explicit GRCh38 header")
            reader = csv.DictReader(lines[header:], delimiter="\t")
            if not {"#ISCA ID", "ISCA Region Name", "Genomic Location"} <= set(reader.fieldnames):
                raise ValueError("ClinGen region TSV is missing required columns")
            records = list(reader)
            matches = [(i, record) for i, record in enumerate(records, 1) if record["#ISCA ID"] == args.region_id]
            if len(matches) != 1:
                raise ValueError(f"expected one region for {args.region_id}; found {len(matches)}")
            number, record = matches[0]
            match = re.fullmatch(r"([^:]+):([0-9]+)-([0-9]+)", record.get("Genomic Location") or "")
            if not match:
                raise ValueError(f"no unambiguous Genomic Location for {args.region_id}")
            rows = [list(match.groups())]
            coordinates = "1-based-inclusive"
            data["original_file"].update(region_id=args.region_id, region_name=record["ISCA Region Name"],
                                         selected_row=number, record_count=len(records))
        elif args.format == "bed":
            if args.coordinates not in (None, "0-based-half-open"):
                raise ValueError("BED requires 0-based-half-open coordinates")
            if any((args.chrom_column, args.start_column, args.end_column)):
                raise ValueError("BED uses its first three columns; do not specify column names")
            coordinates = "0-based-half-open"
            rows = []
            for line in args.input.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line or line.startswith(("#", "track ", "browser ")):
                    continue
                fields = line.split()
                if len(fields) < 3:
                    raise ValueError("BED requires at least three columns")
                rows.append(fields[:3])
        else:
            if not args.coordinates:
                raise ValueError("table intervals require --coordinates; coordinates are never guessed")
            coordinates = args.coordinates
            columns = [args.chrom_column, args.start_column, args.end_column]
            if len(set(columns)) != 3:
                raise ValueError("chromosome, start and end columns must be distinct")
            rows = read_table(args.input, args.format, columns)
            data["original_file"]["interval_columns"] = columns
        if not rows:
            raise ValueError("interval input is empty")
        if args.row is None and len(rows) != 1:
            raise ValueError(f"found {len(rows)} intervals; use --row to select one record after reviewing its identity")
        row = args.row if args.row is not None else 1
        if not 1 <= row <= len(rows):
            raise ValueError(f"--row must be between 1 and {len(rows)} (data records, excluding headers)")
        interval = normalize_interval(*rows[row - 1], coordinates)
        if args.format != "clingen-tsv":
            data["original_file"].update(selected_row=row, record_count=len(rows))
        data["original_file"].update(genome_build=args.genome_build, coordinate_system=coordinates)
        data.update(input_type="interval", genome_build="GRCh38",
                    coordinate_system="1-based-inclusive", interval=interval)
        selected = data["original_file"]["selected_row"]
        count = data["original_file"]["record_count"]
        message = f"{interval} (GRCh38, 1-based-inclusive); source record {selected}/{count}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print("\n[OK] Input preparation complete")
    print(f"  Input type: {data['input_type']}")
    print(f"  Input: {message}")
    print(f"  Source: {data['source']}")
    print(f"  Input file: {args.output.resolve()}")
    print("  Format conversion only; screening has not run.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="kind", required=True)
    genes = sub.add_parser("genes", help="extract one identifier per row from TXT, CSV or TSV")
    interval = sub.add_parser("interval", help="select one known deletion interval from BED, CSV/TSV or ClinGen TSV")
    for command, formats in ((genes, ["txt", "csv", "tsv"]), (interval, ["bed", "csv", "tsv", "clingen-tsv"])):
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--format", choices=formats, required=True)
        command.add_argument("--source", required=True, help="source URL, accession or citation")
        command.add_argument("--output", type=Path, required=True, help="new JSON file; never overwrite")
    genes.add_argument("--gene-column", help="exact CSV/TSV header; required for tables")
    interval.add_argument("--genome-build", choices=["GRCh38"], required=True, help="source assembly; no liftover")
    interval.add_argument("--coordinates", choices=["0-based-half-open", "1-based-inclusive"])
    interval.add_argument("--chrom-column")
    interval.add_argument("--start-column")
    interval.add_argument("--end-column")
    interval.add_argument("--row", type=int, help="1-based data record to select; required for multiple intervals")
    interval.add_argument("--region-id", help="ClinGen region accession, only for --format clingen-tsv")
    args = parser.parse_args()
    try:
        if args.kind == "interval" and args.region_id and args.format != "clingen-tsv":
            raise ValueError("--region-id is only supported for --format clingen-tsv")
        prepare(args)
    except (ValueError, OSError, csv.Error) as exc:
        parser.exit(1, f"[FAIL] Input preparation stopped: {exc}\n")


if __name__ == "__main__":
    main()
