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
    parser.add_argument("--validate-every-batches", type=int, default=0,
                        help="Also validate/select best within each epoch; 0 means epoch boundaries only")
    parser.add_argument("--test-batches", type=int, default=0, help="0 means all test batches")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--embedding-cache-size", type=int, default=128)
    parser.add_argument("--chain-cache-size", type=int, default=128)
    parser.add_argument("--no-sidechain", action="store_true")
    parser.add_argument("--max-train-pairs", type=int)
    parser.add_argument("--recurrence-backend", choices=("reference", "native"), default="reference",
                        help="Reference fixes the observed mixed-gradient issue; native must pass the same head-gradient guard")
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
    details: dict | None = None,
) -> float:
    # DSM forward computes forces with autograd, so validation cannot use no_grad.
    model.eval()
    state_python = random.getstate()
    state_numpy = np.random.get_state()
    state_torch = torch.random.get_rng_state()
    state_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    set_seed(seed)
    losses: list[float] = []
    components: dict[str, list[float]] = {}
    try:
        evaluation_indices = list(range(len(dataset)))
        random.shuffle(evaluation_indices)
        if max_batches:
            evaluation_indices = evaluation_indices[: max_batches * batch_size]
        with torch.enable_grad():
            for batch in dataset.iter_batches(batch_size, evaluation_indices):
                binder, target = make_model_batch(batch, embeddings, device)
                if details is not None:
                    diagnostic = model(binder, target, use_sidechain=use_sidechain, return_diagnostics=True)
                    loss = diagnostic["loss"]
                    for name, tensor in diagnostic.items():
                        if tensor.numel() == 1:
                            component = float(tensor.detach().cpu())
                            if not math.isfinite(component):
                                raise FloatingPointError(f"Non-finite validation diagnostic: {name}")
                            components.setdefault(name, []).append(component)
                    del diagnostic
                    del tensor
                else:
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
    if details is not None:
        details.update({name: float(np.mean(values)) for name, values in components.items()})
        details.update(batches=len(losses), seed=seed, batch_size=batch_size,
                       note="Mean batch diagnostics on a fixed validation subset/noise; no Atlas labels used")
    return float(np.mean(losses))


def check_head_gradients(model: torch.nn.Module) -> dict:
    """Fail before optimizer.step if any energy-head gradient is absent/non-finite."""
    report = {}
    for name, parameter in model.named_parameters():
        if name.startswith(("W_o.", "U_o.")) and parameter.requires_grad:
            gradient = parameter.grad
            if gradient is None:
                raise RuntimeError(f"Missing mixed gradient for {name}; refusing optimizer.step. Use --recurrence-backend reference")
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError(f"Non-finite energy-head gradient: {name}")
            report[name] = float(gradient.detach().norm().cpu())
    if not report or not any(value > 0 for value in report.values()):
        raise RuntimeError("No nonzero energy-head gradient; pilot is not certified")
    return report


def main() -> int:
    args = parse_args()
    from bindenergy.models.energy import AllAtomEnergyModel

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.batch_size <= 0 or args.epochs <= 0:
        raise ValueError("batch-size and epochs must be positive")
    if args.checkpoint_every_batches < 0 or args.validate_every_batches < 0:
        raise ValueError("Checkpoint and validation intervals must be non-negative")
    device = torch.device(args.device)
    torch.set_num_threads(args.num_threads)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new output-dir for a new run; --resume is required to continue an existing run")
    source_path = args.resume or args.init_checkpoint
    source_checkpoint = torch_load(source_path, device)
    architecture = architecture_from_checkpoint(source_checkpoint)
    model = AllAtomEnergyModel(architecture).to(device)
    model.load_state_dict(model_state_from_checkpoint(source_checkpoint), strict=True)

    from .gradient_check import recurrence_backend
    if args.recurrence_backend == "reference":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    recurrence_audit = []
    with recurrence_backend(model, args.recurrence_backend, required_device=device, device_audit=recurrence_audit):
        return train_model(args, model, architecture, source_checkpoint, device, recurrence_audit=recurrence_audit)


def train_model(args, model, architecture, source_checkpoint, device, recurrence_audit=None) -> int:
    from .benchmark_io import write_json, digest_file

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
    validation_history = []
    best_selection = None
    if args.resume:
        if not isinstance(source_checkpoint, dict):
            raise ValueError("--resume requires a checkpoint created by this training script")
        saved_config = source_checkpoint.get("training_config", {})
        saved_backend = saved_config.get("recurrence_backend", "native")
        if saved_backend != args.recurrence_backend:
            raise ValueError("Cannot resume optimizer across recurrence backends; start afresh from the original pretrained checkpoint")
        for key in ("batch_size", "max_train_pairs", "no_sidechain", "seed", "validation_batches", "validate_every_batches"):
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
        validation_history = list(source_checkpoint.get("validation_history", []))
        best_selection = source_checkpoint.get("best_selection")
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
        if args.resume:
            baseline = source_checkpoint.get("baseline_validation")
            if not isinstance(baseline, dict):
                raise ValueError("Resume checkpoint lacks the original fixed-noise validation baseline")
            if digest_file(args.init_checkpoint) != baseline.get("init_checkpoint_sha256"):
                raise ValueError("Resume requires the same original init-checkpoint as the fixed validation baseline")
        else:
            baseline = {}
            print("Evaluating original checkpoint on fixed validation/noise before any update", flush=True)
            best_validation = evaluate_loss(model, validation_data, embeddings, device, args.batch_size,
                args.validation_batches, not args.no_sidechain, args.seed + 100_000, details=baseline)
            baseline["init_checkpoint_sha256"] = digest_file(args.init_checkpoint)
        dataset_hashes = {key: digest_file(path) for key, path in
                          (("train", args.train_db), ("validation", args.validation_db))}
        if args.resume and baseline.get("dataset_sha256", dataset_hashes) != dataset_hashes:
            raise ValueError("Resume requires unchanged train and validation databases")
        baseline["dataset_sha256"] = dataset_hashes
        write_json(args.output_dir / "sru_device_audit.json", {
            "requested_device": str(device), "backend": args.recurrence_backend,
            "model_parameter_devices": sorted({str(p.device) for p in model.parameters()}),
            "cells": recurrence_audit or [],
            "note": "Actual first-call tensor devices; checked on every reference call. No CPU conversion or detach is introduced."})
        write_json(args.output_dir / "training_config.json", {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
        write_json(args.output_dir / "baseline_validation.json", baseline)
        initial_state = model_state_from_checkpoint(torch_load(args.init_checkpoint, torch.device("cpu")))
        initial_heads = {name: value.detach().cpu().clone() for name, value in initial_state.items()
                         if name.startswith(("W_o.", "U_o."))}
        del initial_state
        if not args.resume:
            initial_payload = make_checkpoint(model, optimizer, scheduler, architecture, args, best_validation,
                0, [], {"epoch": 0, "indices": None, "next_position": 0, "running_loss": 0., "seen_batches": 0})
            initial_payload["baseline_validation"] = baseline
            initial_payload["best_selection"] = None
            initial_payload["validation_history"] = []
            atomic_torch_save(initial_payload, args.output_dir / "best.pt")
            del initial_payload
        gradient_checks = []
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
                head_gradients = check_head_gradients(model)
                if len(gradient_checks) < 3:
                    gradient_checks.append({"epoch": epoch, "batch": seen_batches, "norms": head_gradients})
                    write_json(args.output_dir / "head_gradient_checks.json", gradient_checks)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm, error_if_nonfinite=True)
                optimizer.step()
                del loss, binder, target
                running_loss += value
                seen_batches += 1
                next_position += len(batch)
                progress.set_postfix(loss=f"{running_loss / seen_batches:.4f}")
                validation_interval = getattr(args, "validate_every_batches", 0)
                intermediate_validation = bool(validation_interval and
                    seen_batches % validation_interval == 0 and next_position < len(indices))
                intermediate_improved = False
                if intermediate_validation:
                    diagnostics = {}
                    current_loss = evaluate_loss(model, validation_data, embeddings, device,
                        args.batch_size, args.validation_batches, not args.no_sidechain,
                        args.seed + 100_000, details=diagnostics)
                    record = {"epoch": epoch, "batch": seen_batches, "validation_loss": current_loss,
                              "kind": "within_epoch", "diagnostics": diagnostics}
                    validation_history.append(record)
                    intermediate_improved = current_loss < best_validation
                    if intermediate_improved:
                        best_validation = current_loss
                        best_selection = {k: record[k] for k in ("epoch", "batch", "kind", "validation_loss")}
                    write_json(args.output_dir / "validation_history.json", validation_history)
                    print(f"validation epoch={epoch + 1} batch={seen_batches} loss={current_loss:.8f} best={best_validation:.8f}", flush=True)
                    model.train()
                if (
                    intermediate_validation or (args.checkpoint_every_batches
                    and seen_batches % args.checkpoint_every_batches == 0
                    and next_position < len(indices))
                ):
                    training_state = {
                        "epoch": epoch,
                        "indices": indices,
                        "next_position": next_position,
                        "running_loss": running_loss,
                        "seen_batches": seen_batches,
                    }
                    mid_payload = make_checkpoint(
                            model, optimizer, scheduler, architecture, args,
                            best_validation, stale_epochs, history, training_state,
                        )
                    mid_payload["baseline_validation"] = baseline
                    mid_payload["best_selection"] = best_selection
                    mid_payload["validation_history"] = validation_history
                    atomic_torch_save(mid_payload, args.output_dir / "last.pt")
                    if intermediate_improved:
                        atomic_torch_save(mid_payload, args.output_dir / "best.pt")
                    del mid_payload

            validation_details = {}
            validation_loss = evaluate_loss(
                model=model,
                dataset=validation_data,
                embeddings=embeddings,
                device=device,
                batch_size=args.batch_size,
                max_batches=args.validation_batches,
                use_sidechain=not args.no_sidechain,
                seed=args.seed + 100_000,
                details=validation_details,
            )
            write_json(args.output_dir / f"validation_epoch_{epoch + 1}.json", validation_details)
            validation_record = {"epoch": epoch, "batch": seen_batches, "validation_loss": validation_loss,
                                 "kind": "epoch_end", "diagnostics": validation_details}
            validation_history.append(validation_record)
            write_json(args.output_dir / "validation_history.json", validation_history)
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
                best_selection = {k: validation_record[k] for k in ("epoch", "batch", "kind", "validation_loss")}
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
            payload["baseline_validation"] = baseline
            payload["best_selection"] = best_selection
            payload["validation_history"] = validation_history
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

        best_checkpoint = torch_load(args.output_dir / "best.pt", torch.device("cpu"))
        head_updates = {name: float((best_checkpoint["model_state"][name].cpu() - value).abs().max())
                        for name, value in initial_heads.items()}
        last_updates = {name: float((model.state_dict()[name].detach().cpu() - value).abs().max())
                        for name, value in initial_heads.items()}
        report = {"best_validation_loss": best_validation, "epochs_completed": len(history),
                  "baseline_validation_loss": baseline["loss"], "baseline_zero_force_loss": baseline["zero_force_loss"],
                  "best_checkpoint_is_initial": not bool(best_checkpoint.get("best_selection") or best_checkpoint["history"]),
                  "best_selection": best_checkpoint.get("best_selection"),
                  "best_head_max_abs_updates": head_updates, "last_head_max_abs_updates": last_updates,
                  "recurrence_backend": args.recurrence_backend}
        write_json(args.output_dir / "head_updates.json", {"best": head_updates, "last": last_updates})
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
