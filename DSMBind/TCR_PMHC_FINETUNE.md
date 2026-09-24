# TCR-pMHC fine-tuning

This repository represents each five-chain complex as interacting cross-partner
chain pairs. TCR chains are `A/B`; pMHC chains are `M/N/P`. Only `A/B` versus
`M/N/P` pairs with at least one heavy-atom residue contact are retained. Internal
TCR (`A-B`) and internal pMHC pairs are never training examples.

The pipeline initializes `AllAtomEnergyModel` from
`ckpts/model.skempi.allatom`. The checkpoint configuration is preserved exactly:
hidden size 256, ESM dimension 2560, depth 1, patch size 50, and interaction
threshold 14 Angstrom. Fine-tuning uses the original unsupervised rotation,
translation, and side-chain DSM losses; `split_label` is not a target label.

## 1. Vast.ai environment

Use a persistent disk with enough free space for the PDBs, prepared SQLite
databases, ESM-2 cache, and checkpoints. ESM-2 3B embedding generation and model
training should be separate phases so their GPU memory is not held together.

Recommended baseline: Ubuntu, Python 3.10, CUDA PyTorch, at least 24 GB GPU RAM,
and at least 150 GB free persistent disk. Disk demand depends mainly on the
number and length of unique chain sequences.

Either build `Dockerfile.tcr-pmhc`, or start from a PyTorch CUDA image and run:

```bash
cd DSMBind
bash scripts/setup_vast.sh
```

The upstream model was tested with PyTorch 1.13 and SRU++ `3.0.0-dev`; the
provided Dockerfile pins that environment and the current SRU branch commit
(`c2d44e62b90115db59ab92ccea4b2e5b77077cf9`). Do not install CPU-only PyTorch
on the server after starting from a CUDA image.

## 2. Data layout

The default runner expects this layout next to the DSMBind repository:

```text
data/
  linked_data/final_dataset_datasail_like_C1f_4labels_fixed.csv
  new_structure/<str_id>_Complex.pdb
DSMBind/
```

Data and generated databases are ignored by Git. Keep them on Vast persistent
storage or an object-store volume, not in the source repository.

## 3. CPU preprocessing

Preprocessing does not need CUDA and can be run before renting a GPU:

```bash
python -m bindenergy.apps.tcr_pmhc.prepare \
  --csv ../data/linked_data/final_dataset_datasail_like_C1f_4labels_fixed.csv \
  --pdb-dir ../data/new_structure \
  --output-dir ../data/dsmbind_tcr_pmhc \
  --patch-size 50 \
  --min-residue-contacts 1 \
  --resume
```

The command is resumable and creates one database per original split. It fails
immediately on malformed input instead of silently discarding data. Use
`--skip-errors` only if intentional omissions are acceptable.

For a quick end-to-end preparation check, use `--limit 20` and a separate output
directory. Do not train on that limited database.

## 4. ESM-2 cache

```bash
python -m bindenergy.apps.tcr_pmhc.embeddings \
  --datasets \
    ../data/dsmbind_tcr_pmhc/train.sqlite \
    ../data/dsmbind_tcr_pmhc/validation.sqlite \
    ../data/dsmbind_tcr_pmhc/test.sqlite \
    ../data/dsmbind_tcr_pmhc/final_unseen_data.sqlite \
  --output ../data/dsmbind_tcr_pmhc/esm2_t36_3B.sqlite \
  --device cuda \
  --token-budget 1024
```

The cache uses layer 36 of `esm2_t36_3B_UR50D`, stores float16 on disk, returns
float32 to DSMBind, and resumes by sequence hash. Lower `--token-budget` if ESM
runs out of GPU memory.

## 5. Pretrained baseline

Always score the held-out test set before fine-tuning:

```bash
python -m bindenergy.apps.tcr_pmhc.score \
  --dataset ../data/dsmbind_tcr_pmhc/test.sqlite \
  --embeddings ../data/dsmbind_tcr_pmhc/esm2_t36_3B.sqlite \
  --checkpoint ckpts/model.skempi.allatom \
  --output-dir outputs/tcr_pmhc_allatom/pretrained_test \
  --device cuda
```

Pair scores are the sum of both DSMBind directions. Complex outputs include sum,
mean, and contact-normalized scores. These aggregation choices are explicit
project conventions; the TCR-pMHC benchmark papers did not publish their exact
multi-chain aggregation code.

## 6. Pilot then full fine-tuning

Run a short pilot first:

```bash
python -m bindenergy.apps.tcr_pmhc.train \
  --train-db ../data/dsmbind_tcr_pmhc/train.sqlite \
  --validation-db ../data/dsmbind_tcr_pmhc/validation.sqlite \
  --embeddings ../data/dsmbind_tcr_pmhc/esm2_t36_3B.sqlite \
  --init-checkpoint ckpts/model.skempi.allatom \
  --output-dir outputs/pilot \
  --device cuda \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --epochs 2 \
  --max-train-pairs 1000 \
  --validation-batches 20
```

After the pilot completes with finite train/validation losses, run the full job:

```bash
python -m bindenergy.apps.tcr_pmhc.train \
  --train-db ../data/dsmbind_tcr_pmhc/train.sqlite \
  --validation-db ../data/dsmbind_tcr_pmhc/validation.sqlite \
  --test-db ../data/dsmbind_tcr_pmhc/test.sqlite \
  --embeddings ../data/dsmbind_tcr_pmhc/esm2_t36_3B.sqlite \
  --init-checkpoint ckpts/model.skempi.allatom \
  --output-dir outputs/tcr_pmhc_allatom/finetune \
  --device cuda \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --epochs 5 \
  --patience 2 \
  --checkpoint-every-batches 1000
```

The source fine-tuning study used 100 epochs on a much smaller structural set.
Here, chain-pair expansion yields roughly two orders of magnitude more examples,
so five epochs already represent a large number of optimizer updates. Extend the
run only when validation DSM loss is still improving; resume rather than starting
again.

`best.pt` is selected by minimum validation DSM loss. `last.pt` is written every
1,000 training batches and after every epoch. It includes the shuffled pair
order, next position, running loss, optimizer/scheduler state, and all random
number generator states, so a Vast interruption resumes at the next unprocessed
batch. Keep the batch size and training-data options unchanged when resuming.
Repeat the command with:

```text
--resume outputs/tcr_pmhc_allatom/finetune/last.pt
```

The test DSM loss is computed only after training and best-checkpoint selection.

## 7. Final scoring

```bash
python -m bindenergy.apps.tcr_pmhc.score \
  --dataset ../data/dsmbind_tcr_pmhc/final_unseen_data.sqlite \
  --embeddings ../data/dsmbind_tcr_pmhc/esm2_t36_3B.sqlite \
  --checkpoint outputs/tcr_pmhc_allatom/finetune/best.pt \
  --output-dir outputs/tcr_pmhc_allatom/finetuned_final_unseen \
  --device cuda
```

The CSV contains no affinity, binder/non-binder, or ddG target. Therefore this
pipeline performs valid unsupervised domain adaptation and reports DSM loss and
energy scores; it cannot claim supervised binding accuracy without an external
experimental or carefully defined proxy target.

## One-command run

After setup, the full sequence is available as:

```bash
bash scripts/run_tcr_pmhc_pipeline.sh
```

Override paths and GPU-sensitive parameters through environment variables, for
example:

```bash
DATA_ROOT=/workspace/data BATCH_SIZE=2 ESM_TOKEN_BUDGET=512 \
  bash scripts/run_tcr_pmhc_pipeline.sh
```
