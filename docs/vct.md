# VirtualCellTool: local interactive demo

VCT is separate from the screening score. **Start with public PBMC3k data to check the tool**, then prepare an appropriate disease dataset for research. The PBMC demonstration is not a WHS experiment.

## 1. Install a separate environment

From the repository root, in Terminal (macOS/Linux) or Miniforge Prompt (Windows):

```text
conda env create --prefix ./workspace/envs/virtual-cell --file vct/environment.yml
conda activate ./workspace/envs/virtual-cell
python -m pip check
```

Git and Conda are required. This environment includes Scanpy, PyTorch, SIGnature and the VCT web dependencies; the screening environment does not need them. SIGnature is installed from a pinned upstream commit. A separate third-party checkout is unnecessary.

## 2. Download the model

```text
python -m vct.prepare_model
```

Expected: **`[OK] Model verified and extracted`**. The script downloads the approximately 118 MB [official SIGnature model archive](https://zenodo.org/records/17903196), verifies its size and published MD5, and extracts the SCimilarity files under `workspace/vct/models/`. Sources and SHA-256 are saved beside them. It refuses to overwrite an existing model directory.

## 3. Prepare the public example

```text
python vct/src/prepare_data.py --computed-groups 6 --output workspace/vct/data
```

The first run downloads the approximately 6 MB public PBMC3k matrix through Scanpy. It aligns genes to SCimilarity, computes 128-dimensional embeddings and UMAP, and saves a complete seven-file bundle. Expected: **2,700 cells**, a `(2700, 128)` embedding and `(2700, 2)` UMAP, followed by the output path.

**Computed group 1–6 are algorithmic groups, not confirmed cell types.** Their labels come from seeded KMeans on the same embeddings. `pbmc_info.json` records the input checksum, model location, grouping policy and UMAP seed. No historical annotations or precomputed results are bundled.

To use an independently downloaded copy, add `--input path/to/pbmc3k_raw.h5ad`. If you have verified annotations, replace `--computed-groups 6` with `--metadata labels.csv`; the CSV must contain unique `cell` IDs and nonempty `cell_type` labels covering exactly the input cells. Choose a new output directory for each preparation run.

## 4. Start and explore

```text
python -m uvicorn web.app:app --app-dir vct --host 127.0.0.1 --port 8377
```

Wait for **`Application startup complete`**, then open **http://127.0.0.1:8377/**. Stop with **Ctrl+C** in the same terminal. This is a local application: another user clones the repository and runs their own instance. It is not a hosted multi-user service.

Try the **PBMC** dataset, select a cell on UMAP, inspect gene attribution, and search for **MS4A1**. Compare expression, encoder-based perturbation, CIPHER response and the baseline. Unavailable datasets and unsupported genes are reported explicitly. The existing full interface retains its Chinese labels; this guide gives the English workflow.

| View | What it means |
|---|---|
| Expression and UMAP | Observed expression and an embedding visualization |
| Attribution | Which inputs affect the encoder's representation under the chosen comparison |
| Encoder perturbation | Re-encoded expression after the selected change; model-space displacement |
| CIPHER and baseline | Dataset-wide linear response and comparison predictions |
| Group response | Relationship between a shared prediction and group expression patterns |

The current web CIPHER model fits the selected dataset as a whole. Group views **are not independently fitted cell-type causal predictions**. None of these outputs demonstrates SINEUP-mediated protein restoration or functional rescue.

## 5. Other datasets and optional research functions

Runtime assets belong in `workspace/vct/`. Set `VCT_WORKSPACE`, `VCT_DATA_DIR` or `VCT_MODEL_DIR` before starting the application to use another location. `SIGNATURE_DIR` optionally selects a source checkout; normally the installed package is used.

The MS and Williams preparation scripts now require explicit inputs:

```text
python vct/src/prepare_ws_web.py --input path/to/annotated_williams.h5ad
```

Inspect each script's `--help` and its required metadata fields before using a new dataset. Williams input needs an `obs["condition"]` annotation. The MS web preparer additionally needs `var["feature_name"]`, `obs["disease"]` and an existing `ms_embeddings.npy` generated from the same matrix and cell order by `compute_ms_attribution.py` (which also requires donor and lesion metadata). Place that run’s embeddings in `VCT_DATA_DIR` before calling `prepare_ms_web.py --input ...`; this is not an automatic download workflow. These research-specific workflows assume suitable annotations and gene coverage; an arbitrary h5ad is not automatically compatible. Download the original matrices from the study providers and retain their identifiers, versions and annotation provenance. See [sources](ATTRIBUTIONS.md). These larger studies were not rerun during this migration.

The optional **GEARS** branch needs Norman Perturb-seq data and a compatible checkpoint, independently of PBMC. Follow [GEARS data instructions](https://github.com/snap-stanford/GEARS) and the [checkpoint source](https://huggingface.co/matthewshu/gears-norman). Its default locations are `workspace/vct/gears_data/` and `workspace/vct/gears_ckpt/`; `python vct/src/load_gears.py` checks a prepared installation. Predictable targets depend on the actual training graph. GEARS training and pretrained predictions were not rerun here. Their absence does not prevent PBMC attribution or CIPHER use.

`vct/analysis/` retains research analyses. `vct/src/history_panel.py` serves only explicitly listed local historical files; a fresh clone shows these as missing. Earlier experiments remain in the [original project](https://github.com/panxuan-pku/igem-backup). Missing archives do not prevent core operation.

## 6. Developer checks

In the VCT environment:

```text
python -m pip install -r screening/requirements-dev.txt
python -m pytest tests/vct -q
```

With the PBMC service running on port 8377, use another activated terminal:

```text
python tests/vct/tests_api.py --datasets pbmc --skip-gears
```

This exercises the live core routes with the prepared PBMC dataset. It intentionally excludes GEARS and unavailable MS/Williams datasets; it is not evidence of those workflows passing. Offline tests include synthetic safety checks. A historical-archive test skips if its optional material is absent.

The installation, real PBMC preparation and core service were tested on macOS Apple Silicon. Windows and Linux command forms are provided, but were not executed on those systems during this migration.
