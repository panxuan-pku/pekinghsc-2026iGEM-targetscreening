# Methods and provenance

## Screening

The two input routes converge on a single evidence-ranking workflow:

```text
Deletion interval ──► GENCODE overlap ─┐
                                      ├─► gene identity ─► evidence ranking ─► design checks ─► report
Candidate gene list ──────────────────┘
```

The default profile is `screening/config/screening.yaml`. It uses GRCh38 and 1-based inclusive coordinates. BED conversion adjusts the coordinate convention, not the assembly. Gene identities must map unambiguously; duplicates and ambiguous symbols stop the formal ranking.

| Component | Role in the current profile |
|---|---|
| ClinGen haploinsufficiency | Core score, weight 3 |
| gnomAD LOEUF and pLI | Core score, weight 2 each |
| GENCODE v44 and HGNC | Identity, position and transcript/3′UTR annotation |
| HPA v24 | Tissue-expression context; zero ranking penalty in this profile |
| Open Targets | Disease annotation only, expected release 26.09 |
| Optional IMPC | Phenotype annotation, not core scoring |
| DeepLOF | Optional analysis support remains; not enabled in this profile |

The profile's transformation and missingness rules are implemented in the existing consensus code. Evidence availability and conflicts remain visible. Missing evidence is not experimental evidence of absence. OT request failures are explicitly distinguished from a successful query with no record, and leave the genetic-evidence score unchanged.

**Design selection is separate from rank.** A qualifying annotated 3′UTR must be at least **30 bp**. The default module budget is 4,400 bp with an assumed 300 bp per module. This caps the number selected; it does not remove other candidates from the table. Transcript annotation and this length check do not establish SINEUP binding efficacy.

## Reproduction

The WHS example uses ClinGen region **ISCA-37429**, **chr4:337779–2009235** (GRCh38). The September 2026 run yields 30 protein-coding candidates, 14 selected and NSD2 first. This is distinct from the historical 19-gene study set; do not mix their denominators or results.

Keep the typed input, reference directory (including `references.json`), configuration, output manifest and OT snapshot. The manifest records source checksums and code provenance. Reuse `--ot-snapshot` when comparing the two input routes or code versions so annotation service availability does not confound comparisons. Live HGNC and ClinGen downloads can change; their downloaded bytes and checksums define the snapshot used.

The report and 16-column summary are the first reading layer. The full evidence table keeps technical fields needed to audit identity, scoring, transcript selection and missing evidence. All data and results are local, excluded from Git.

## VCT boundary

VCT explores expression-based hypotheses downstream of target selection. It does not feed into the ranking. Its current CIPHER implementation fits a shared dataset-level linear response; group summaries compare that response with group expression patterns. Encoder attribution and model-space displacement do not validate a causal therapeutic effect. See [VCT](vct.md) for the practical workflow and optional GEARS branch.
