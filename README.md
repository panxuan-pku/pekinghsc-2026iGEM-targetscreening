# iGEM 2026: deletion-gene prioritization

**English** | [简体中文](README.zh-CN.md)

**Start with a deletion interval or a candidate gene list; obtain an evidence ranking and a design-check report.** The pipeline integrates ClinGen and gnomAD evidence, adds Open Targets (OT) disease annotations, and checks the project's **3′UTR ≥30 bp** and module-budget requirements.

| Your starting point | Workflow |
|---|---|
| A deletion interval, or your first WHS example | [2. Deletion interval](#2-deletion-interval-whs-example) |
| A candidate list from a sample, publication or analysis | [3. Gene list](#3-gene-list) |
| Results and custom output locations | [4. Outputs](#4-outputs) |
| Understanding the repository | [7. Repository structure](#7-repository-structure) |

The separate **VirtualCellTool (VCT)** explores expression changes using single-cell data. It does not contribute to the genetic-evidence ranking or demonstrate SINEUP protein restoration. [Start the VCT demo](docs/vct.md).

## 1. Installation

Use Terminal on macOS/Linux or a Conda-enabled terminal on Windows (for example, Miniforge Prompt). **Git and Conda must be available.** These commands are the same on all three systems. Lines starting with `#` explain each command; in Windows Miniforge Prompt (cmd.exe), copy only the command lines:

```text
# Download the project from GitHub
git clone https://github.com/panxuan-pku/pekinghsc-2026iGEM-targetscreening.git

# Enter the repository root
cd pekinghsc-2026iGEM-targetscreening

# Create an isolated Conda environment with the basic screening dependencies
conda env create --prefix ./workspace/envs/virtual-screening --file screening/environment.yml

# Activate the screening environment for this terminal
conda activate ./workspace/envs/virtual-screening

# Check dependency declarations, installed versions and imports
python -m screening.check --suite environment
```

Continue after **`[OK] Checks passed (environment)`**. Run all commands below from the repository root. In a new terminal, return to this directory and run `conda activate ./workspace/envs/virtual-screening` again. If `conda` is not found, open Miniforge Prompt (Windows) or initialize your installed Conda shell and reopen Terminal (macOS/Linux). Environment prefixes keep this clone separate from existing environments.

## 2. Deletion interval: WHS example

This example uses the **ClinGen WHS reference interval ISCA-37429, GRCh38**. The pipeline extracts overlapping protein-coding genes from reference annotations; no BioMart export is needed. This public reference interval is not a patient-specific deletion or the historical 19-gene study set.

### 2.1 Prepare the input

Create `workspace/screening/input/raw/`. Download the [ClinGen GRCh38 region table](https://ftp.clinicalgenome.org/ClinGen_region_curation_list_GRCh38.tsv) into it, keeping the original filename. Check the interval against the [WHS record](https://search.clinicalgenome.org/kb/gene-dosage/region/ISCA-37429).

```text
# Convert the ClinGen WHS region record to the interval input JSON
python -m screening.prepare_input interval --input workspace/screening/input/raw/ClinGen_region_curation_list_GRCh38.tsv --format clingen-tsv --region-id ISCA-37429 --genome-build GRCh38 --source "https://search.clinicalgenome.org/kb/gene-dosage/region/ISCA-37429" --output workspace/screening/input/whs_interval_input.json
```

**Expected:** `[OK] Input preparation complete`, the interval coordinates and the path to `whs_interval_input.json`. For this record: **chr4:337779–2009235**, 1-based inclusive. Check the coordinates; no manual JSON editing is needed.

### 2.2 Prepare references

```text
# Download the five reference files, or verify and reuse existing files
python -m screening.run prepare
```

**Expected:** `[OK] Reference preparation complete (5/5)`. The first download is approximately **617 MiB**, saved in `workspace/screening/references/`. Subsequent runs verify and reuse completed files; rerun the same command after an interruption.

| Reference | Purpose |
|---|---|
| GENCODE and HGNC | Gene identity, position and transcript annotation |
| ClinGen and gnomAD | Core ranking evidence |
| HPA | Tissue-expression annotations |

The ClinGen **region table** in step 2.1 defines the input interval; the **gene evidence table** downloaded here contributes to scoring.

### 2.3 Run screening

```text
# Rank genes in the WHS interval and save reports in whs_interval_01
python -m screening.run run --input-file workspace/screening/input/whs_interval_input.json --output workspace/screening/results/whs_interval_01
```

**Expected:** `[OK] Screening complete`, candidate counts and the path to **`interval_report.html`**. Open that file in a browser. With the references checked on 28 September 2026: **30 candidates, 14 selected, NSD2 ranked first**. Live evidence updates can change results. See [section 4](#4-outputs) for the other outputs.

<details>
<summary>Use your own deletion interval: BED input</summary>

Save a GRCh38 BED file as `workspace/screening/input/raw/deletion.bed` and replace step 2.1 with:

```text
# Convert your GRCh38 BED interval to input JSON; replace the source placeholder
python -m screening.prepare_input interval --input workspace/screening/input/raw/deletion.bed --format bed --genome-build GRCh38 --source "actual interval source" --output workspace/screening/input/deletion_interval_input.json
```

Keep step 2.2. In step 2.3, use `--input-file workspace/screening/input/deletion_interval_input.json` and a new output directory. The converter changes BED's 0-based start to 1-based inclusive coordinates; it does **not** convert genome builds. Select one row from a multi-row BED with `--row`. For CSV/TSV interval tables, run `python -m screening.prepare_input interval --help`.

</details>

## 3. Gene list

Use this route when you **already have a defined candidate set**. No deletion coordinates are required. Ensembl/HGNC IDs and uniquely resolvable gene symbols are accepted.

### 3.1 Prepare the input

Save your table as `workspace/screening/input/raw/genes.csv`, with one candidate per row. Keep its header and other columns. This example assumes a gene column named **`gene_id`**; replace `--gene-column` and `--source` with the actual column name and source URL or publication ID.

```text
# Read candidate IDs from the gene_id column; replace the column and source as needed
python -m screening.prepare_input genes --input workspace/screening/input/raw/genes.csv --format csv --gene-column gene_id --source "actual source URL or publication ID" --output workspace/screening/input/gene_list_input.json
```

**Expected:** `[OK] Input preparation complete`, the number of identifiers and the path to `gene_list_input.json`. Identity checks occur during screening; duplicates or ambiguous identities produce an explicit error.

<details>
<summary>Other formats: TXT, TSV and Excel</summary>

TXT files contain one gene per line with no header:

```text
# Convert a one-gene-per-line TXT file to input JSON; replace the source placeholder
python -m screening.prepare_input genes --input workspace/screening/input/raw/genes.txt --format txt --source "actual list source" --output workspace/screening/input/gene_list_input.json
```

For TSV, use the CSV command with your `.tsv` path, `--format tsv` and the actual column name. Export Excel sheets as UTF-8 CSV first. Choose one format to create the input JSON.

</details>

<details>
<summary>No list yet? Practice with the complete WHS candidates from section 2</summary>

After completing section 2, use this command instead of the CSV conversion above. It reads **all candidates**, not only the highest-ranked genes:

```text
# Use all WHS candidates from the interval run to prepare a gene-list input
python -m screening.prepare_input genes --input workspace/screening/results/whs_interval_01/interval_candidates.csv --format csv --gene-column gene_id --source "WHS ISCA-37429; whs_interval_01/interval_candidates.csv; see whs_interval_01/interval_manifest.json" --output workspace/screening/input/gene_list_input.json
```

Continue with steps 3.2 and 3.3. This demonstrates another input route for the same candidate set, not an independent source of candidates.

</details>

### 3.2 Prepare references

```text
# Verify and reuse the shared references, downloading any missing files
python -m screening.run prepare
```

**Expected:** `[OK] Reference preparation complete (5/5)`. Both input routes share references; files verified in step 2.2 are reused.

### 3.3 Run screening

```text
# Rank the submitted gene list and save reports in gene_list_01
python -m screening.run run --input-file workspace/screening/input/gene_list_input.json --output workspace/screening/results/gene_list_01
```

**Expected:** `[OK] Screening complete`, candidate counts and the path to **`gene_list_report.html`**. This report covers the submitted list.

Both routes default to WHS disease ID `MONDO_0008684` for OT annotations. For another disease, add `--disease-id YOUR_DISEASE_ID` to `run`. OT annotations do not change the core ranking.

## 4. Outputs

### 4.1 Which files to read

Filenames follow **`input_type_purpose.extension`**: `interval` for deletion intervals, `gene_list` for gene lists.

| Purpose | Interval input | Gene-list input |
|---|---|---|
| Browser report | `interval_report.html` | `gene_list_report.html` |
| Summary, 16 columns | `interval_summary.csv` | `gene_list_summary.csv` |
| Full evidence | `interval_ranked.csv` | `gene_list_ranked.csv` |
| Standardized candidates | `interval_candidates.csv` | `gene_list_candidates.csv` |
| Run manifest | `interval_manifest.json` | `gene_list_manifest.json` |
| Run configuration | `interval_config.yaml` | `gene_list_config.yaml` |
| OT response snapshot | `interval_ot_snapshot.json` | `gene_list_ot_snapshot.json` |

CSV files use UTF-8 with BOM for Excel compatibility; JSON, HTML and YAML use UTF-8. The manifest records the input type, output filenames and checksums. Shared references remain in `workspace/screening/references/`.

Both result tables preserve the same candidates, ranks and scores. **The score indicates research priority; `selected` indicates compliance with design-length and budget conditions.** Neither establishes experimental validation.

**`[WARN] Ranking generated…`** means the ranking is available with OT annotation gaps; review the report. **`[FAIL]`** means the step stopped; resolve the reported error before retrying.

### 4.2 Choose your output directory

Set **`--output "directory"`** on the `run` command:

```text
# Save this run in a custom output directory that does not yet exist
python -m screening.run run --input-file workspace/screening/input/gene_list_input.json --output "workspace/screening/results/gene_list_02"
```

| System | Absolute-path example |
|---|---|
| macOS | `--output "/Users/yourname/Documents/screening/gene_list_02"` |
| Linux | `--output "/home/yourname/screening/gene_list_02"` |
| Windows | `--output "D:\screening\gene_list_02"` |

Relative paths resolve from your terminal's current directory. **The output directory must not already exist**; the program creates it. For another run, change `_01` to `_02`.

Without `--output`, the program creates `workspace/screening/results/interval_YYYYMMDD_HHMMSS_microseconds/` or `gene_list_YYYYMMDD_HHMMSS_microseconds/` inside this repository. Custom directory names are used unchanged; files inside still carry the input-type prefix. Name prepared inputs `project_interval_input.json` or `project_gene_list_input.json`. Existing results retain their original names.

To relocate references, supply the same `--references "directory"` to both `prepare` and `run`.

## 5. Methods and reproduction

The unified configuration is **`screening/config/screening.yaml`**. ClinGen and gnomAD drive the ranking; OT, HPA and optional IMPC provide annotations. The 3′UTR check and module budget determine design selection without deleting ranking rows. DeepLOF is not enabled in this profile.

`references.json` records reference versions, sources and checksums. To replay an analysis, retain its input, references and configuration, then add `--ot-snapshot previous_run/interval_ot_snapshot.json` to `run` (use `gene_list_ot_snapshot.json` for a list run).

See [methods and provenance](docs/methods.md) for evidence roles and [VCT](docs/vct.md) for downstream exploration. Neither tool establishes therapeutic benefit.

## 6. Optional features and development

Basic screening only needs section 1. VCT has a **separate environment**, including its own single-cell dependencies; installing CNV extras first is unnecessary.

| Purpose | Instructions |
|---|---|
| VCT: install, model, data and interactive demo | [VCT guide](docs/vct.md) |
| Optional single-cell/CNV processing | `python -m pip install -r screening/requirements-cnv.txt` in the screening environment; excluded from the core workflow |
| Development | Commands below; not required for users |

```text
# Install optional developer test dependencies
python -m pip install -r screening/requirements-dev.txt

# Run the core screening tests without downloading references
python -m screening.check --suite core

# Check release files for bundled data, models and broken local documentation links
python tests/check_release.py
```

Expected: **`[OK] Checks passed (core)`** and **`Release check passed`**. Run VCT tests inside the VCT environment as explained in its guide. Optional CNV tests require the CNV dependencies; run `python -m screening.check --suite full` after installing them. CNV exploration is retained but is not a validated way to infer this WHS candidate set from scRNA-seq.

## 7. Repository structure

```text
pekinghsc-2026iGEM-targetscreening/  # Repository root
├── README.md                 # Start here: two screening input routes
├── README.zh-CN.md           # Simplified Chinese guide
├── screening/                # Gene-prioritization pipeline
│   ├── prepare_input.py      # Downloaded table → typed input JSON
│   ├── run.py                # Reference preparation and screening
│   ├── report.py             # English HTML reports
│   ├── check.py              # Environment and developer checks
│   ├── config/               # screening.yaml is the current profile
│   ├── src/                  # Evidence, ranking and optional analyses
│   └── environment.yml       # Basic Conda environment
├── vct/                      # VirtualCellTool
│   ├── src/                  # Data preparation and perturbation engines
│   ├── web/                  # Interactive local application
│   ├── analysis/             # Optional research analyses
│   ├── prepare_model.py      # Verified official model download
│   └── environment.yml       # Independent VCT Conda environment
├── tests/                    # screening/, vct/ and release checks
├── docs/                     # Methods, VCT guide and third-party attribution
├── examples/                 # Example sources and reproduction notes
└── workspace/                # Local only: environments, data, models, results
```

There are no historical disease directories or bundled results. `workspace/` is created locally and ignored by Git. The older `screening/config/pipeline.yaml` and CNV configuration support optional analysis and regression tests; **use `screening/config/screening.yaml` through `python -m screening.run` for the workflow above**.

## License and sources

Team code is distributed under [MIT](LICENSE). See [attributions](docs/ATTRIBUTIONS.md) for third-party software, data and model sources; their own terms continue to apply.
