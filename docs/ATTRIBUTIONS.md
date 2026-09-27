# Attribution and external resources

## Team contribution

The team maintains the screening integration, input conversion, evidence provenance, reports, VCT data preparation, interactive interface and local perturbation/baseline implementation. These integrate external evidence and methods; the team did not train the foundation model or generate the public reference datasets.

Code was migrated from [igem-backup](https://github.com/panxuan-pku/igem-backup), including its current development changes. The destination retains its existing MIT license for team code. Earlier records identify AI assistance from Hermes Agent; Codex assisted with migration, dependency/path updates, documentation and testing. Scientific interpretation remains the team's responsibility.

## Screening evidence

| Resource | Source and role |
|---|---|
| GENCODE v44 | [Human annotation](https://www.gencodegenes.org/human/release_44.html): genomic overlap, transcripts and UTRs |
| HGNC | [Complete-set download](https://www.genenames.org/download/statistics-and-files/): standardized identities |
| ClinGen | [Dosage sensitivity](https://search.clinicalgenome.org/kb/gene-dosage): gene evidence and the WHS reference region |
| gnomAD v4.1.1 | [Downloads](https://gnomad.broadinstitute.org/downloads): loss-of-function constraint |
| HPA v24 | [Archived downloads](https://v24.proteinatlas.org/about/download): expression context |
| Open Targets | [Platform](https://platform.opentargets.org/): disease annotations, separate from the score |

Exact download URLs and versions are in `screening/config/screening.yaml`; local manifests record the bytes actually used. Provider terms apply to downloaded data.

## VCT software and models

| Resource | Use and attribution |
|---|---|
| [SIGnature](https://github.com/Genentech/SIGnature) | Genentech model wrappers and alignment; installed from commit `c1546cb9857acb3d91e50d19233a7b2a28bf54d6`. Its [Genentech Non-Commercial Software License](https://github.com/Genentech/SIGnature/blob/c1546cb9857acb3d91e50d19233a7b2a28bf54d6/LICENSE) applies separately. |
| [Official model helper archive](https://zenodo.org/records/17903196) | SCimilarity weights and gene order, downloaded at runtime. SIGnature's model support does not imply every upstream model is implemented in VCT. |
| [Captum](https://github.com/pytorch/captum) | Integrated Gradients attribution |
| [GEARS](https://github.com/snap-stanford/GEARS) | Optional perturbation model; [Norman checkpoint source](https://huggingface.co/matthewshu/gears-norman). No weights or training data are redistributed here. |
| [CIPHER](https://github.com/GoyalLab/CIPHER)-inspired covariance response | Team implementation in `vct/src/cipher_engine.py`; the repository documents the implemented approximation and its limits, not equivalence to an externally validated therapeutic model. |
| [Linear baseline study](https://doi.org/10.1038/s41592-025-02772-6) | Motivation for explicit baseline comparisons; local baseline implementation is team code. |
| [Plotly.js 2.32.0](https://github.com/plotly/plotly.js/tree/v2.32.0) | Browser visualization bundle in `vct/web/plotly.min.js`; original copyright/license header is retained. Plotly's [MIT license](https://github.com/plotly/plotly.js/blob/v2.32.0/LICENSE) applies to that bundle. |

## Single-cell data

- **PBMC3k:** 10x Genomics, distributed through [Scanpy's PBMC3k loader](https://scanpy.readthedocs.io/en/stable/generated/scanpy.datasets.pbmc3k.html). The download URL and checksum are recorded during preparation. Group labels are computed for this demonstration, not imported from a cell-type atlas.
- **MS oligodendrocytes:** earlier work used a CELLxGENE study (recorded collection prefix `16c1e722`). A complete, verified download identifier is not yet provided in this release; obtain and verify the original study matrix and annotations before using the retained research scripts. The PBMC demonstration does not depend on it.
- **Williams syndrome:** original study-specific input is not bundled. Retained scripts accept an explicit AnnData path; the old project preserves the study context. This migration does not claim independent reproduction of that dataset.
- **Norman Perturb-seq:** optional GEARS exploration uses the data sources documented by GEARS. Earlier experiments recorded Zenodo `17252307` as a mirror; the core PBMC demonstration does not download or rely on it.

CIPHER reference: Kuznets-Speck and colleagues, *Fluctuation structure predicts genome-wide perturbation outcomes*, [bioRxiv preprint (2025)](https://www.biorxiv.org/content/10.1101/2025.06.27.661814v1). This cites the theoretical source; migration checks establish software behavior, not equivalence to the paper’s benchmarks.
