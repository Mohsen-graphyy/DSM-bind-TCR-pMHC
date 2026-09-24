"""Fine-tune DSMBind's all-atom PPI model on TCR-pMHC chain pairs."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from bindenergy.data.tcr_pmhc import EmbeddingStore, TCRpMHCPairDataset, make_model_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-db", type=Path, required=True)
    parser.add_argument("--validation-db", type=Path, required=True)
    parser.add_argument("--test-db", type=Path)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, default=Path("ckpts/model.skempi.allatom"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--anneal-rate", type=float, default=1.0)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument(
        "--checkpoint-every-batches",
        type=int,
        default=1000,
        help="Write last.pt during an epoch; 0 disables mid-epoch checkpoints",
    )
    parser.add_argument("--validation-batches", type=int, default=200)
    parser.add_argument("--test-batches", type=int, default=0, help="0 means all test batches")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--embedding-cache-size", type=int, default=128)
    parser.add_argument("--chain-cache-size", type=int, default=128)
    parser.add_argument("--no-sidechain", action="store_true")
    parser.add_argument("--max-train-pairs", type=int)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch 1.13, used by the upstream DSMBind release
        return torch.load(path, map_location=device)


def architecture_from_checkpoint(checkpoint: Any) -> Namespace:
    if isinstance(checkpoint, tuple) and len(checkpoint) == 3:
        architecture = checkpoint[2]
    elif isinstance(checkpoint, dict) and "architecture" in checkpoint:
        architecture = checkpoint["architecture"]
    else:
        raise ValueError("Unsupported checkpoint format")
    if isinstance(architecture, Namespace):
        return architecture
    if isinstance(architecture, dict):
        return Namespace(**architecture)
    return Namespace(**vars(architecture))


def model_state_from_checkpoint(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, tuple) and len(checkpoint) == 3:
        return checkpoint[0]
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"]
    raise ValueError("Unsupported checkpoint format")


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def make_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    architecture: Namespace,
    args: argparse.Namespace,
    best_validation: float,
    stale_epochs: int,
    history: list[dict[str, float]],
    training_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "architecture": vars(architecture),
        "training_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        # Kept for readable inspection and compatibility with format version 1.
        "epoch": int(training_state["epoch"]) - (training_state["next_position"] == 0),
        "best_validation_loss": best_validation,
        "stale_epochs": stale_epochs,
        "history": history,
        "training_state": training_state,
        "rng_state": capture_rng_state(),
    }


def evaluate_loss(
    model: AllAtomEnergyModel,
    dataset: TCRpMHCPairDataset,
    embeddings: EmbeddingStore,
    device: torch.device,
    batch_size: int,
    max_batches: int,
    use_sidechain: bool,
    seed: int,
) -> float:
    # DSM forward computes forces with autograd, so validation cannot use no_grad.
    model.eval()
    state_python = random.getstate()
    state_numpy = np.random.get_state()
    state_torch = torch.random.get_rng_state()
    state_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    set_seed(seed)
    losses: list[float] = []
    try:
        evaluation_indices = list(range(len(dataset)))
        random.shuffle(evaluation_indices)
        if max_batches:
            evaluation_indices = evaluation_indices[: max_batches * batch_size]
        with torch.enable_grad():
            for batch in dataset.iter_batches(batch_size, evaluation_indices):
                binder, target = make_model_batch(batch, embeddings, device)
                loss = model(binder, target, use_sidechain=use_sidechain)
                value = float(loss.detach().cpu())
                if not math.isfinite(value):
                    raise FloatingPointError("Non-finite validation loss")
                losses.append(value)
                del loss, binder, target
    finally:
        random.setstate(state_python)
        np.random.set_state(state_numpy)
        torch.random.set_rng_state(state_torch)
        if state_cuda is not None:
            torch.cuda.set_rng_state_all(state_cuda)
    if not losses:
        raise RuntimeError("Validation dataset produced no batches")
    return float(np.mean(losses))


def main() -> int:
    args = parse_args()
    from bindenergy.models.energy import AllAtomEnergyModel

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.batch_size <= 0 or args.epochs <= 0:
        raise ValueError("batch-size and epochs must be positive")
    if args.checkpoint_every_batches < 0:
        raise ValueError("checkpoint-every-batches must be non-negative")
    device = torch.device(args.device)
    torch.set_num_threads(args.num_threads)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_config.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2),
        encoding="utf-8",
    )

    source_path = args.resume or args.init_checkpoint
    source_checkpoint = torch_load(source_path, device)
    architecture = architecture_from_checkpoint(source_checkpoint)
    model = AllAtomEnergyModel(architecture).to(device)
    model.load_state_dict(model_state_from_checkpoint(source_checkpoint), strict=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.anneal_rate)
    start_epoch = 0
    best_validation = float("inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []
    resume_indices: list[int] | None = None
    resume_position = 0
    resume_running_loss = 0.0
    resume_seen_batches = 0
    saved_rng_state: dict[str, Any] | None = None
    if args.resume:
        if not isinstance(source_checkpoint, dict):
            raise ValueError("--resume requires a checkpoint created by this training script")
        saved_config = source_checkpoint.get("training_config", {})
        for key in ("batch_size", "max_train_pairs", "no_sidechain"):
            if key in saved_config and saved_config[key] != getattr(args, key):
                raise ValueError(
                    f"Cannot change {key} when resuming: checkpoint has "
                    f"{saved_config[key]!r}, command requested {getattr(args, key)!r}"
                )
        optimizer.load_state_dict(source_checkpoint["optimizer_state"])
        scheduler.load_state_dict(source_checkpoint["scheduler_state"])
        best_validation = float(source_checkpoint.get("best_validation_loss", float("inf")))
        stale_epochs = int(source_checkpoint.get("stale_epochs", 0))
        history = list(source_checkpoint.get("history", []))
        if "training_state" in source_checkpoint:
            training_state = source_checkpoint["training_state"]
            start_epoch = int(training_state["epoch"])
            if training_state.get("indices") is not None:
                resume_indices = [int(index) for index in training_state["indices"]]
                resume_position = int(training_state["next_position"])
                resume_running_loss = float(training_state["running_loss"])
                resume_seen_batches = int(training_state["seen_batches"])
            saved_rng_state = source_checkpoint.get("rng_state")
        else:
            start_epoch = int(source_checkpoint["epoch"]) + 1

    train_data = TCRpMHCPairDataset(args.train_db, args.chain_cache_size)
    validation_data = TCRpMHCPairDataset(args.validation_db, args.chain_cache_size)
    test_data = TCRpMHCPairDataset(args.test_db, args.chain_cache_size) if args.test_db else None
    embeddings = EmbeddingStore(args.embeddings, args.embedding_cache_size)
    required_databases = [args.train_db, args.validation_db]
    if args.test_db:
        required_databases.append(args.test_db)
    embeddings.require_sequences(required_databases)
    if int(embeddings.metadata.get("dimension", architecture.esm_size)) != architecture.esm_size:
        raise RuntimeError(
            f"Embedding dimension {embeddings.metadata.get('dimension')} does not match "
            f"checkpoint dimension {architecture.esm_size}"
        )
    if len(train_data) == 0 or len(validation_data) == 0:
        raise RuntimeError("Train and validation databases must contain at least one pair")
    if resume_indices is not None:
        if resume_position < 0 or resume_position > len(resume_indices):
            raise RuntimeError("Resume checkpoint has an invalid training position")
        if any(index < 0 or index >= len(train_data) for index in resume_indices):
            raise RuntimeError("Resume checkpoint does not match the training database")
    if saved_rng_state is not None:
        restore_rng_state(saved_rng_state)

    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            if epoch == start_epoch and resume_indices is not None:
                indices = resume_indices
                next_position = resume_position
                running_loss = resume_running_loss
                seen_batches = resume_seen_batches
            else:
                indices = list(range(len(train_data)))
                random.shuffle(indices)
                if args.max_train_pairs:
                    indices = indices[: args.max_train_pairs]
                next_position = 0
                running_loss = 0.0
                seen_batches = 0
            progress = tqdm(
                train_data.iter_batches(args.batch_size, indices[next_position:]),
                total=math.ceil(len(indices) / args.batch_size),
                initial=seen_batches,
                desc=f"epoch {epoch + 1}/{args.epochs}",
                unit="batch",
            )
            for batch in progress:
                optimizer.zero_grad(set_to_none=True)
                binder, target = make_model_batch(
                    batch,
                    embeddings,
                    device,
                    swap_partners=(random.random() < 0.5),
                )
                loss = model(binder, target, use_sidechain=not args.no_sidechain)
                value = float(loss.detach().cpu())
                if not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite training loss at epoch {epoch + 1}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                optimizer.step()
                running_loss += value
                seen_batches += 1
                next_position += len(batch)
                progress.set_postfix(loss=f"{running_loss / seen_batches:.4f}")
                if (
                    args.checkpoint_every_batches
                    and seen_batches % args.checkpoint_every_batches == 0
                    and next_position < len(indices)
                ):
                    training_state = {
                        "epoch": epoch,
                        "indices": indices,
                        "next_position": next_position,
                        "running_loss": running_loss,
                        "seen_batches": seen_batches,
                    }
                    atomic_torch_save(
                        make_checkpoint(
                            model, optimizer, scheduler, architecture, args,
                            best_validation, stale_epochs, history, training_state,
                        ),
                        args.output_dir / "last.pt",
                    )

            validation_loss = evaluate_loss(
                model=model,
                dataset=validation_data,
                embeddings=embeddings,
                device=device,
                batch_size=args.batch_size,
                max_batches=args.validation_batches,
                use_sidechain=not args.no_sidechain,
                seed=args.seed + 100_000,
            )
            scheduler.step()
            train_loss = running_loss / max(seen_batches, 1)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            improved = validation_loss < best_validation
            if improved:
                best_validation = validation_loss
                stale_epochs = 0
            else:
                stale_epochs += 1
            training_state = {
                "epoch": epoch + 1,
                "indices": None,
                "next_position": 0,
                "running_loss": 0.0,
                "seen_batches": 0,
            }
            payload = make_checkpoint(
                model, optimizer, scheduler, architecture, args,
                best_validation, stale_epochs, history, training_state,
            )
            atomic_torch_save(payload, args.output_dir / "last.pt")
            if improved:
                atomic_torch_save(payload, args.output_dir / "best.pt")
            (args.output_dir / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            print(
                f"epoch={epoch + 1} train_loss={train_loss:.6f} "
                f"validation_loss={validation_loss:.6f} best={best_validation:.6f}"
            )
            if args.patience and stale_epochs >= args.patience:
                print(f"Early stopping after {stale_epochs} epochs without improvement")
                break
            resume_indices = None

        report = {"best_validation_loss": best_validation, "epochs_completed": len(history)}
        if test_data is not None:
            best_checkpoint = torch_load(args.output_dir / "best.pt", device)
            model.load_state_dict(best_checkpoint["model_state"], strict=True)
            report["test_dsm_loss"] = evaluate_loss(
                model=model,
                dataset=test_data,
                embeddings=embeddings,
                device=device,
                batch_size=args.batch_size,
                max_batches=args.test_batches,
                use_sidechain=not args.no_sidechain,
                seed=args.seed + 200_000,
            )
        (args.output_dir / "final_metrics.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
    finally:
        train_data.close()
        validation_data.close()
        if test_data is not None:
            test_data.close()
        embeddings.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
