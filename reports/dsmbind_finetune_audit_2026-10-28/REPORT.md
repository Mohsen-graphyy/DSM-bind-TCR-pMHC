# DSMBind fine-tuning technical audit

Date: 2026-10-28 (Asia/Tehran).

## Verdict

The positive-only DSM fine-tuning formulation and most pipeline mechanics are appropriate, and encoder parameters did change. The present checkpoints cannot be accepted as a validated successful adaptation. The observed scoring function is almost reducible to atom-pair counting, and the intended all-parameter training did not update the energy heads. These findings are stronger than the previous log-only assessment.

## Scope and limitations

Reviewed local training/data/model/embedding/testing code, the original released checkpoint, the supplied four fine-tuning/pilot checkpoints, config/history/final metrics, and the original DSMBind paper and SRU implementation. Recomputed geometric masks for all 13,229 scored test pairs using the prepared local test SQLite database and aligned by structure/chain identifiers with the supplied baseline and fine-tuned CSVs.

This is not a fresh GPU training run. There is no Python executable on the current command path. Checkpoint tensors and optimizer metadata were inspected with a read-only Node ZIP/pickle data decoder which treats Python globals as inert metadata. Tensor storage byte comparisons and SHA-256 hashes independently verified head equality. Parameter/optimizer mappings were checked against tensor shapes; SRU's `scale_x` is a buffer and was excluded from parameter indexing. All tensor comparisons concern float32, contiguous model tensors in these specific archives.

## Inputs and provenance

- Local code HEAD/run commit previously recorded: `aedf7312793b89945c9fa85e24ba0613a642809b`.
- Original checkpoint: `DSMBind/ckpts/model.skempi.allatom`.
- Fine-tuned checkpoints: `C:/Users/hajmo/Downloads/best.pt`, `last.pt`.
- Pilot checkpoints: `C:/Users/hajmo/Downloads/best (1).pt`, `last (1).pt`.
- Original checkpoint SHA-256: `002fd5bde1d56c06b44e37a647574ef919d7be416fd8743647f897909dab67b2`.
- Fine-tuned best SHA-256: `305002a101fe2465324edb695034239ee9a6aa81e52f807c12a5a39fb2af3017`.
- Fine-tuned last SHA-256: `38e0326641ae58e9f7f41bc64f835d81325d5938f728b7529c07294b7452d027`.
- Baseline pairs: `C:/Users/hajmo/Downloads/pair_scores.csv`.
- Fine-tuned test pairs: `C:/Users/hajmo/Downloads/pair_scores (1).csv`.
- Prepared geometry: `data/dsmbind_tcr_pmhc/test.sqlite`.

## Finding 1: Output scores nearly reduce to counting masked atom pairs

For every saved test pair, decode the exact stored patch indices and float32 coordinates. Retain populated atom14 coordinate slots with norm > 1e-4, then count cross-partner atom pairs at Euclidean distance < 14 Angstrom, matching the checkpoint's internal scoring cutoff. Call this count N14.

This count is different from the CSV `contact_count`, which counts residue pairs at 5 Angstrom on full chains.

Fit one coefficient through the origin using the supplied symmetric scores: `S = c * N14`. This is a descriptive diagnostic fitted on test scores, not an independently evaluated binding model.

| Quantity | Pretrained | Fine-tuned best |
|---|---:|---:|
| Test pairs | 13,229 | 13,229 |
| Pearson(S, N14) | 0.9672889622 | 0.9999612708 |
| Fitted coefficient c | 0.2140292381 | 0.6144646047 |
| R-squared of through-origin approximation | 0.8881691985 | 0.9997962985 |
| Mean S/N14 | 0.1765520233 | 0.6187217888 |
| SD S/N14 | 0.0644042683 | 0.0057431908 |

All pairs had N14 > 0. The fine-tuned relation is not merely caused by mixing large MHC interfaces and small peptide interfaces:

| Chain pair | n | Pearson(fine-tuned S, N14) |
|---|---:|---:|
| A-M | 2,749 | 0.9999568790 |
| A-N | 1,048 | 0.9999926627 |
| A-P | 2,787 | 0.9999308661 |
| B-M | 2,826 | 0.9999891585 |
| B-N | 1,042 | 0.9999759433 |
| B-P | 2,777 | 0.9999649241 |

Example 2520/A-M: N14=10,204, baseline symmetric score=2,344.855, fine-tuned score=6,260.3643. Example 2520/A-P: N14=10,958, baseline=1,225.366, fine-tuned=6,857.1309.

Interpretation: very strong evidence of effective degeneration of the raw scoring function toward a nearly constant contribution per masked atom pair. This does not by itself prove that every hidden feature or residual score is useless; matched-mutant, fixed-geometry sequence perturbation, and native/decoy tests are still needed. Increased symmetry and peptide score share are not independent evidence of improved recognition: atom-pair count itself is symmetric and changes across interfaces.

## Finding 2: Energy heads did not train

All six tensors in `W_o.0`, `W_o.2`, and `U_o.0` (weights and biases; 197,376 scalar values) are byte-identical between the original checkpoint, full best, full last, and both pilot checkpoints. None has an entry in the saved Adam state.

The code constructs `Adam(model.parameters())`, has no intentional head freeze, and uses `zero_grad(set_to_none=True)`. With a newly constructed optimizer and no state removal, absent Adam state indicates that these parameters were never processed with a non-None gradient. Their exact equality is not just small or rounded learning.

The original released checkpoint ALSO lacks Adam state for these heads and the SRU recurrence parameters. Thus this behavior cannot be blamed exclusively on the user's PyTorch 2.11 environment or on the new data adapter. It may be inherited from the original implementation/runtime path; this audit does not assert a defect in every published DSMBind configuration.

There are 20 parameter tensors; 11 have Adam state and changed during adaptation. Best records 34,466 optimizer steps, last records 103,398. This agrees with one and three complete epochs over 137,863 pairs at batch size 4. Fine-tuning did occur in part of the encoder.

The force-based objective needs mixed second derivatives d/dtheta[dE/dX]. The model explicitly requests `create_graph=True`. SRU's custom CUDA backward is therefore a high-priority suspect: successful ordinary backward or CUDA compilation does not certify double backward. The existing CPU smoke test substitutes FakeSRUpp and only checks that ANY parameter has a gradient; it cannot detect this problem.

Confirmed: unchanged heads and absent optimizer states. Not confirmed in this CPU-only audit: the exact runtime operation that cuts off the graph, the full correct gradient, or whether a different SRU/PyTorch backend fixes it. The next diagnostic must compare per-group gradients with finite differences and a differentiable reference recurrence, rather than merely downgrading PyTorch or setting requires_grad=True.

## Finding 3: Existing loss reporting cannot establish useful adaptation

Full training history:

| Epoch (human numbering) | Train DSM loss | Validation DSM loss |
|---|---:|---:|
| 1 | 4.855198174971 | 4.133659135699 |
| 2 | 4.145319251510 | 4.133662451506 |
| 3 | 3.949517543173 | 4.133663021326 |

Validation covers a fixed seeded subset of 200 batches (800/12,532 pairs, about 6.38%) and a fixed seeded noise realization. Best is epoch 1; early stopping after two non-improving epochs matches the code. Test DSM loss=3.954972559985 uses all test batches and the selected best checkpoint. Validation is not detached by `no_grad`, and test does not drive checkpoint selection.

However, there is no pretrained DSM-loss reference on the same evaluation draws, no zero-force predictor reference, no translation/backbone/side-chain component reporting, no gradient diagnostics, and no independent binding or pose-ranking evaluation. A tiny change in validation loss or a falling training average cannot distinguish useful force learning from saturation/collapse, incomplete higher-order differentiation, or noise effects.

## What is technically appropriate

- Positive-only training is compatible with unsupervised DSMBind. Experimental non-binder labels are not required for the DSM objective.
- AllAtomEnergyModel and original checkpoint architecture are loaded strictly.
- ESM-2 t36 3B, layer 36, 2,560 dimensions, special-token removal, full-chain embedding followed by patch indexing match the upstream design. Fixed ESM features may be shared across splits without training on held-out labels.
- Atom14 layouts and centroid patching follow the upstream protein adapter. TCR A/B versus pMHC M/N/P pair decomposition is a defensible pairwise approximation, not a full five-chain model.
- Random partner swapping, Adam, clipping, and real parameter updates are present.
- Checkpoint selection uses validation, then reloads best for final test evaluation.

## Additional methodological limitations

- Each chain pair is treated as an independent complex; cooperative multi-chain context is omitted. An intact positive complex does not imply that each isolated weak-contact subpair is a stable autonomous binding complex.
- A structure with six retained pairs is sampled 1.5 times as often as one with four pairs. The effective training objective is pair-weighted rather than complex-weighted.
- At least one 5-Angstrom residue contact is a permissive interface criterion; patch retention and weak-interface behavior need targeted checks before further training.
- A sum or mean of separately contextualized pair scores is a chosen aggregation, not a proven whole-complex affinity estimator. Dividing by a 5-Angstrom residue count does not remove the observed 14-Angstrom atom-pair dependence.
- Provided split labels are respected. Template/sequence-homology leakage and overlap with the original pretrained corpus were not independently audited here; 'unseen' should not be expanded into a stronger claim without that evidence.
- No native/decoy success metric, affinity correlation, or experimental negative discrimination is provided. AUROC for native versus decoy would measure pose discrimination, not binder versus non-binder discrimination.

## Required next work, in order

1. Preserve existing checkpoints and results as the first run; do not resume long training to fix this by more epochs.
2. On the actual CUDA environment, log parameter-wise gradient presence/norms, verify mixed derivatives against fixed-noise finite differences, and compare with a differentiable SRU recurrence retaining the same parameterization. Include energy-head weights and selected biases rather than assuming every bias must have a nonzero derivative.
3. Evaluate pretrained, best, last, and a zero-force reference on the same validation structures and identical noise draws; report separate loss components, force/torque norms, both partner orientations, and several evaluation seeds.
4. Add matched controls: identical geometry with altered peptide sequence; controlled translations/rotations; native versus docking decoys. Compare to the N14 counting baseline and stratify or match interface size. Score variation alone does not establish biological specificity.
5. Only after these checks pass, rerun a short fine-tuning pilot from the original checkpoint. Verify that the intended parameter groups update and that scores do not reduce to contact count. Select hyperparameters on validation; reserve untouched test data for final reporting.
6. For binding-strength or specificity claims, obtain suitable held-out experimental affinity/mutant data or verified negatives. Structure-positive training remains valid, but experimental claims require appropriate evaluation targets.

## Reproduce the new offline findings

Run from the workspace root using Node 22+ (the audit was run with Node 22.13):

```powershell
node reports/dsmbind_finetune_audit_2026-10-28/checkpoint_audit.cjs --verify
node reports/dsmbind_finetune_audit_2026-10-28/geometry_audit.cjs
```

Scripts use local Downloads paths listed above and read-only SQLite access. They do not modify input data, model weights, training code, or model dependencies. The checkpoint decoder is intentionally limited to these PyTorch archive formats, not a general-purpose pickle library.

## Primary references

- DSMBind original paper, method and protein-protein appendix: https://www.mlsb.io/papers_2023/SE3_denoising_score_matching_for_unsupervised_binding_energy_prediction_and_nanobody_design.pdf
- Original code: https://github.com/wengong-jin/DSMBind
- SRU custom CUDA backward: https://raw.githubusercontent.com/asappresearch/sru/3.0.0-dev/sru/cuda_functional.py
- Pinned SRU recurrence backend selection: https://raw.githubusercontent.com/asappresearch/sru/c2d44e62b90115db59ab92ccea4b2e5b77077cf9/sru/modules.py
- PyTorch double-backward requirements and gradgradcheck: https://docs.pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html
