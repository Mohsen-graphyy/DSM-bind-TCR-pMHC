"""Score TCR-pMHC chain pairs and aggregate them to complex-level scores."""

from __future__ import annotations

import argparse
import csv
import json
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from bindenergy.data.tcr_pmhc import EmbeddingStore, TCRpMHCPairDataset, make_model_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--embedding-cache-size", type=int, default=128)
    parser.add_argument("--chain-cache-size", type=int, default=128)
    return parser.parse_args()


def load_parts(path: Path, device: torch.device) -> tuple[dict, Namespace]:
    try:
        checkpoint: Any = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch 1.13 compatibility
        checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, tuple) and len(checkpoint) == 3:
        state, architecture = checkpoint[0], checkpoint[2]
    elif isinstance(checkpoint, dict):
        state, architecture = checkpoint["model_state"], checkpoint["architecture"]
    else:
        raise ValueError("Unsupported checkpoint format")
    if not isinstance(architecture, Namespace):
        architecture = Namespace(**(architecture if isinstance(architecture, dict) else vars(architecture)))
    return state, architecture


def main() -> int:
    args = parse_args()
    from bindenergy.models.energy import AllAtomEnergyModel

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    device = torch.device(args.device)
    state, architecture = load_parts(args.checkpoint, device)
    model = AllAtomEnergyModel(architecture).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    dataset = TCRpMHCPairDataset(args.dataset, args.chain_cache_size)
    embeddings = EmbeddingStore(args.embeddings, args.embedding_cache_size)
    embeddings.require_sequences([args.dataset])
    if int(embeddings.metadata.get("dimension", architecture.esm_size)) != architecture.esm_size:
        raise RuntimeError(
            f"Embedding dimension {embeddings.metadata.get('dimension')} does not match "
            f"checkpoint dimension {architecture.esm_size}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = args.output_dir / "pair_scores.csv"
    aggregate: dict[str, dict[str, list[float] | float | str]] = defaultdict(
        lambda: {"scores": [], "contacts": [], "split": ""}
    )

    try:
        with pair_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "structure_id", "split", "binder_chain", "target_chain",
                    "contact_count", "min_distance", "forward_score",
                    "reverse_score", "symmetric_score",
                ),
            )
            writer.writeheader()
            with torch.inference_mode():
                for batch in tqdm(
                    dataset.iter_batches(args.batch_size),
                    total=(len(dataset) + args.batch_size - 1) // args.batch_size,
                    desc="DSMBind scoring",
                    unit="batch",
                ):
                    binder, target = make_model_batch(batch, embeddings, device)
                    forward = model.predict(binder, target).detach().cpu().numpy()
                    reverse = model.predict(target, binder).detach().cpu().numpy()
                    symmetric = forward + reverse
                    for index, entry in enumerate(batch):
                        writer.writerow(
                            {
                                "structure_id": entry["structure_id"],
                                "split": entry["split"],
                                "binder_chain": entry["binder_chain"],
                                "target_chain": entry["target_chain"],
                                "contact_count": entry["contact_count"],
                                "min_distance": f"{entry['min_distance']:.4f}",
                                "forward_score": f"{forward[index]:.8g}",
                                "reverse_score": f"{reverse[index]:.8g}",
                                "symmetric_score": f"{symmetric[index]:.8g}",
                            }
                        )
                        group = aggregate[entry["structure_id"]]
                        group["split"] = entry["split"]
                        group["scores"].append(float(symmetric[index]))
                        group["contacts"].append(float(entry["contact_count"]))
    finally:
        dataset.close()
        embeddings.close()

    complex_path = args.output_dir / "complex_scores.csv"
    with complex_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "structure_id", "split", "pair_count", "score_sum", "score_mean",
                "total_contacts", "score_per_contact",
            ),
        )
        writer.writeheader()
        for structure_id in sorted(aggregate):
            group = aggregate[structure_id]
            scores = np.asarray(group["scores"], dtype=np.float64)
            contacts = np.asarray(group["contacts"], dtype=np.float64)
            total_contacts = float(contacts.sum())
            writer.writerow(
                {
                    "structure_id": structure_id,
                    "split": group["split"],
                    "pair_count": len(scores),
                    "score_sum": f"{scores.sum():.10g}",
                    "score_mean": f"{scores.mean():.10g}",
                    "total_contacts": f"{total_contacts:.10g}",
                    "score_per_contact": f"{scores.sum() / max(total_contacts, 1.0):.10g}",
                }
            )
    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "pair_count": len(dataset),
        "complex_count": len(aggregate),
        "pair_scores": str(pair_path),
        "complex_scores": str(complex_path),
    }
    (args.output_dir / "scoring_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
