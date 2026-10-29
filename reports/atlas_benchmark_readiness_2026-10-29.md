# ATLAS benchmark readiness — 2026-10-29

The portable benchmark code and CPU preparation are complete. Actual ESM-2 GPU inference and pretrained/fine-tuned score comparison have **not** been run locally. No affinity performance result is claimed in this readiness report.

User instructions: [Persian step-by-step guide](../DSMBind/ATLAS_BENCHMARK_FA.md).

## Real input preparation

Source: `atlas dataset/outputs/atlas_peptide_ranking/atlas_peptide_ranking_index.csv` and its corresponding `atlas_model_test/model_inputs/manifest.csv`, with PDBs and optional assay metadata from that package. The original workbook/index/PDBs were not modified.

| Quantity | Value |
|---|---:|
| Original observations preserved | 236 |
| Distinct structural scoring inputs | 206 |
| Biological sequence units | 206 |
| Retained TCR–pMHC chain pairs | 871 |
| Required unique ESM sequences | 511 |
| Required residues | 46,719 |
| Raw float16 ESM storage estimate | 239,201,280 bytes (228.1 MiB) |
| Peptide groups | 129 |
| Peptide groups with >=3 units, before checking label/prediction ties | 19 |
| Units in those peptide groups | 75 |
| Assay-context groups with >=3 units, before ties | 7 |
| Missing N/C/O chain-atom report records | 2 |

Sequence-based chain mapping was verified while packing. Nonstandard ATLAS whitespace ATOM records are parsed explicitly, and missing atoms remain zero-masked. The default strict behavior of the original PDB loader is preserved for existing callers; the benchmark explicitly opts into the inference parser settings.

## Files ready to transfer (not tracked by Git)

- `data/atlas_benchmark/bundle.zip`: 14,035,638 bytes; includes all 206 original PDB contents plus label-separated portable manifests.
- `artifacts/checkpoints/finetuned_best.pt`: 19,707,515 bytes; copied from the user's full fine-tune `Downloads/best.pt` without changing the weights.

Bundle ZIP SHA-256: `8a98b4c34846f6e25d75bc7552a51809e08f6713e8349e06b002fa3c6f6f32e3`.

Fine-tuned checkpoint SHA-256: `305002a101fe2465324edb695034239ee9a6aa81e52f807c12a5a39fb2af3017`.

The original checkpoint is already in `DSMBind/ckpts/model.skempi.allatom`.

## Validation performed

- Full CPU unittest suite: 17 tests passed, including 10 benchmark tests covering portable inputs, explicit sign conventions, ranking ties, repeated measurements, strict score-ID coverage, checksums/cache invalidation, whitespace coordinates, missing-backbone masking, and single-checkpoint inference CLI.
- An end-to-end evaluator test uses deliberately synthetic scores with known rank order; these are software tests only and are never presented as real Atlas predictions.
- Actual original and fine-tuned checkpoint architectures loaded on CPU and checked for patch size 50, ESM dimension 2560 and atom cutoff 14 Angstrom.
- Real data preparation completed for all 206 structures; no structures were silently excluded.
- Compileall, shell syntax checks and git whitespace checks passed.
- The old model-device smoke tests use FakeSRUpp; passing them does not certify real SRU CUDA behavior or higher-order training gradients.

## Evaluation contract

Primary score is predeclared as symmetric chain-pair `score_sum`; larger scores predict stronger affinity. Experimental strength is `-delta_g_kcal_mol`. Secondary scores include peptide-only symmetric sum, mean and per-residue-contact normalization. Neither direction nor aggregation is selected by maximizing Atlas performance.

Metrics include within-peptide Spearman, Kendall tau-b, tie-aware pairwise accuracy and paired differences between checkpoints on common eligible groups. A separate assay-context analysis controls peptide/HLA/PMID/temperature/method when available. Constant or undersized groups remain visible with explicit undefined correlations. A geometry counting baseline tests whether learned scores add information beyond N14.

Same-input repeated measurements are retained at observation level and aggregated using median experimental delta G for unit-level metrics. Distinct structures for the same biological unit are averaged once each, not weighted by repeat count.

## Remaining server work

1. Commit/push code using the explicit file list in the guide.
2. Clone/pull on a CUDA devel Vast instance (Python 3.10–3.12, preferably 24GB VRAM).
3. Upload the bundle ZIP and fine-tuned checkpoint above.
4. Run `bash scripts/setup_inference_vast.sh`, followed by the documented preparation-only preflight.
5. Run `bash scripts/run_atlas_benchmark.sh`.
6. Download `evaluation/report.md`, all evaluation CSV/JSON files, scoring outputs and `run_manifest.json`; preserve embeddings if desired, then stop the billed GPU instance.

The runner computes only missing embeddings, scores both checkpoints on the same patches/cache, verifies completed stage hashes before reuse, and records data/model/code/required-embedding fingerprints and runtime versions. Single-model `infer` supports future PDB bundles without labels.

## Limits on interpretation

This is an external affinity-ranking diagnostic, not the official IMMREP26 reactivity metric. Original pretraining and fine-tuning data overlap with Atlas is unknown. The previous checkpoint audit findings are not repaired by this benchmark. A fair outcome may favor either checkpoint, neither, or the geometry baseline; the report does not assume a fine-tuning improvement. Absolute predicted kcal/mol and RMSE between raw score and delta G are not reported.

ATLAS reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC5860664/
