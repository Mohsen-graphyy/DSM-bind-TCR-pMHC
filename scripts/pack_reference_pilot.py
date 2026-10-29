"""Export a small, fixed train/validation pair subset. Standard library, no CUDA/ESM."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sqlite3
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export_subset(source: Path, target: Path, count: int, seed: int, split: str) -> dict:
    if count < 1:
        raise ValueError("Pair count must be positive")
    if target.exists():
        raise FileExistsError(target)
    connection = sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    output = None
    try:
        connection.execute("BEGIN")
        available = [r[0] for r in connection.execute("SELECT id FROM pairs ORDER BY id")]
        if count > len(available):
            raise ValueError(f"Requested {count} pairs, available {len(available)}")
        selected = sorted(random.Random(seed).sample(available, count))
        pairs = [dict(connection.execute("SELECT * FROM pairs WHERE id=?", (i,)).fetchone()) for i in selected]
        if {r["split"] for r in pairs} != {split}:
            raise ValueError(f"Expected only {split} source pairs")
        structure_ids = sorted({r["structure_id"] for r in pairs})
        chain_ids = sorted({(r["structure_id"], r[c]) for r in pairs for c in ("binder_chain", "target_chain")})
        structures = []
        for sid in structure_ids:
            row = dict(connection.execute("SELECT * FROM structures WHERE structure_id=?", (sid,)).fetchone())
            row["pair_count"] = sum(p["structure_id"] == sid for p in pairs)
            structures.append(row)
        chains = [dict(connection.execute("SELECT * FROM chains WHERE structure_id=? AND chain_id=?", key).fetchone())
                  for key in chain_ids]
        metadata = [dict(r) for r in connection.execute("SELECT * FROM metadata")]
        target.parent.mkdir(parents=True, exist_ok=True)
        output = sqlite3.connect(target)
        for table in ("metadata", "chains", "structures", "pairs"):
            schema = connection.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()[0]
            output.execute(schema)
        for table, rows in (("metadata", metadata), ("chains", chains), ("structures", structures), ("pairs", pairs)):
            fields = list(rows[0])
            output.executemany(f"INSERT INTO {table} ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
                               [tuple(r[k] for k in fields) for r in rows])
        output.execute("CREATE INDEX idx_chains_sequence_key ON chains(sequence_key)")
        output.execute("CREATE INDEX idx_pairs_structure ON pairs(structure_id)")
        output.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", ("reference_pilot", json.dumps({"seed": seed, "pairs": count})))
        output.commit()
        if output.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Subset SQLite integrity check failed")
        output.close()
        output = None
        return {"source_name": source.name, "pairs": count, "structures": len(structure_ids),
                "structure_ids": structure_ids, "seed": seed, "selected_source_pair_ids": selected,
                "database_sha256": sha256(target), "database_bytes": target.stat().st_size}
    finally:
        if output is not None:
            output.close()
        connection.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-db", type=Path, default=Path("data/dsmbind_tcr_pmhc/train.sqlite"))
    parser.add_argument("--validation-db", type=Path, default=Path("data/dsmbind_tcr_pmhc/validation.sqlite"))
    parser.add_argument("--train-pairs", type=int, default=128)
    parser.add_argument("--validation-pairs", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("data/reference_pilot/bundle"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if args.output_dir.exists() or args.output_dir.with_suffix(".zip").exists():
        parser.error("Use a new output directory/archive; existing pilot data will not be overwritten")
    train = export_subset(args.train_db, args.output_dir / "train.sqlite", args.train_pairs, args.seed, "train")
    validation = export_subset(args.validation_db, args.output_dir / "validation.sqlite", args.validation_pairs, args.seed + 1, "validation")
    if set(train["structure_ids"]) & set(validation["structure_ids"]):
        raise ValueError("Train/validation selected structure IDs overlap")
    manifest = {"format_version": 1, "train": train, "validation": validation,
                "purpose": "Short diagnostic pilot from the user's original splits; no Atlas data/labels.",
                "sampling": "Uniform seeded source-pair subset; no changes to coordinates, patches, interface filter or loss."}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.make_archive(str(args.output_dir), "zip", args.output_dir.parent, args.output_dir.name)
    print(json.dumps({"archive": str(args.output_dir.with_suffix('.zip')), "train_pairs": args.train_pairs,
                      "validation_pairs": args.validation_pairs,
                      "archive_bytes": args.output_dir.with_suffix('.zip').stat().st_size}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
