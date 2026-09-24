"""Build a resumable, disk-backed ESM-2 embedding cache."""

from __future__ import annotations

import argparse
import json
import sqlite3
import zlib
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from tqdm import tqdm

from bindenergy.data.tcr_pmhc import EMBEDDING_SCHEMA, distinct_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="esm2_t36_3B_UR50D")
    parser.add_argument("--layer", type=int, default=36)
    parser.add_argument("--token-budget", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--compression-level", type=int, default=1)
    return parser.parse_args()


def token_batches(
    sequences: list[tuple[str, str]], token_budget: int
) -> Iterator[list[tuple[str, str]]]:
    batch: list[tuple[str, str]] = []
    max_tokens = 0
    for item in sequences:
        length = len(item[1]) + 2
        candidate_max = max(max_tokens, length)
        if batch and candidate_max * (len(batch) + 1) > token_budget:
            yield batch
            batch = []
            max_tokens = 0
        batch.append(item)
        max_tokens = max(max_tokens, length)
    if batch:
        yield batch


def load_esm(name: str):
    import esm

    try:
        loader = getattr(esm.pretrained, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown fair-esm model: {name}") from exc
    return loader()


def main() -> int:
    args = parse_args()
    if args.token_budget <= 0:
        raise ValueError("--token-budget must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    sequences = distinct_sequences(args.datasets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(args.output)
    connection.executescript(EMBEDDING_SCHEMA)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    existing_metadata = {
        key: json.loads(value)
        for key, value in connection.execute("SELECT key, value FROM metadata")
    }
    completed = {
        row[0] for row in connection.execute("SELECT sequence_key FROM embeddings")
    }
    requested_metadata = {
        "model": args.model,
        "layer": args.layer,
        "storage_dtype": args.storage_dtype,
    }
    if existing_metadata:
        for key, value in requested_metadata.items():
            if existing_metadata.get(key) != value:
                raise ValueError(
                    f"Embedding cache uses {key}={existing_metadata.get(key)!r}, "
                    f"requested {value!r}; use a new output path"
                )
    elif completed:
        raise ValueError("Embedding cache contains rows but has no model metadata")
    pending = [item for item in sequences if item[0] not in completed]
    print(f"distinct sequences={len(sequences):,}; cached={len(completed):,}; pending={len(pending):,}")
    if not pending:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.close()
        return 0

    model, alphabet = load_esm(args.model)
    model = model.eval().to(args.device)
    batch_converter = alphabet.get_batch_converter()
    storage_dtype = np.dtype(args.storage_dtype)

    metadata = {
        **requested_metadata,
        "dimension": int(model.embed_dim),
    }
    for key, value in metadata.items():
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
    connection.commit()

    batches = list(token_batches(pending, args.token_budget))
    with torch.inference_mode():
        for batch in tqdm(batches, desc="ESM-2 batches", unit="batch"):
            labels_and_sequences = [(key, sequence) for key, sequence in batch]
            _, _, tokens = batch_converter(labels_and_sequences)
            tokens = tokens.to(args.device)
            result = model(tokens, repr_layers=[args.layer], return_contacts=False)
            representations = result["representations"][args.layer]
            rows = []
            for batch_index, (key, sequence) in enumerate(batch):
                embedding = (
                    representations[batch_index, 1 : len(sequence) + 1]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(storage_dtype, copy=False)
                )
                rows.append(
                    (
                        key,
                        sequence,
                        len(sequence),
                        embedding.shape[1],
                        storage_dtype.name,
                        zlib.compress(embedding.tobytes(), args.compression_level),
                    )
                )
            with connection:
                connection.executemany(
                    """
                    INSERT OR REPLACE INTO embeddings(
                        sequence_key, sequence, length, dimension, dtype, data
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            del tokens, result, representations

    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
