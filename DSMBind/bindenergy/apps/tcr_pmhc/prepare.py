"""Convert five-chain TCR-pMHC PDBs into pair-level DSMBind databases."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from bindenergy.data.tcr_pmhc import (
    PMHC_CHAINS,
    TCR_CHAINS,
    TCRpMHCDataError,
    initialize_dataset_database,
    insert_structure,
    parse_pdb_chains,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CSV = WORKSPACE_ROOT / "data" / "linked_data" / "final_dataset_datasail_like_C1f_4labels_fixed.csv"
DEFAULT_PDB_DIR = WORKSPACE_ROOT / "data" / "new_structure"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "data" / "dsmbind_tcr_pmhc"
SPLITS = ("train", "validation", "test", "final_unseen_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--pdb-dir", type=Path, default=DEFAULT_PDB_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--patch-size", type=int, default=50)
    parser.add_argument("--min-residue-contacts", type=int, default=1)
    parser.add_argument("--heavy-atom-cutoff", type=float, default=5.0)
    parser.add_argument("--ca-prefilter-cutoff", type=float, default=12.0)
    parser.add_argument("--pdb-template", default="{str_id}_Complex.pdb")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true", help="Skip structures already committed")
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Record malformed/missing structures instead of failing immediately",
    )
    return parser.parse_args()


def read_rows(path: Path, limit: int | None) -> list[dict[str, str]]:
    if not path.is_file():
        raise TCRpMHCDataError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"str_id", "split_label"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise TCRpMHCDataError(f"CSV is missing columns: {sorted(missing)}")
        rows = []
        for row in reader:
            split = row["split_label"].strip()
            structure_id = row["str_id"].strip()
            if split not in SPLITS:
                raise TCRpMHCDataError(f"Unknown split {split!r} for {structure_id}")
            if not structure_id or not structure_id.isalnum():
                raise TCRpMHCDataError(f"Unsafe/blank str_id: {structure_id!r}")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def main() -> int:
    args = parse_args()
    if args.patch_size <= 0 or args.min_residue_contacts < 0:
        raise TCRpMHCDataError("patch/contact counts must be non-negative, patch-size must be positive")
    rows = read_rows(args.csv, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "format_version": 1,
        "source_csv": str(args.csv.resolve()),
        "source_pdb_dir": str(args.pdb_dir.resolve()),
        "patch_size": args.patch_size,
        "min_residue_contacts": args.min_residue_contacts,
        "heavy_atom_cutoff": args.heavy_atom_cutoff,
        "ca_prefilter_cutoff": args.ca_prefilter_cutoff,
        # Lists round-trip through JSON without changing type, which keeps
        # resume metadata validation stable.
        "tcr_chains": list(TCR_CHAINS),
        "pmhc_chains": list(PMHC_CHAINS),
    }
    connections = {
        split: initialize_dataset_database(args.output_dir / f"{split}.sqlite", metadata)
        for split in SPLITS
    }
    processed = Counter()
    pairs = Counter()
    failures: list[dict[str, str]] = []

    try:
        for row in tqdm(rows, desc="TCR-pMHC structures", unit="structure"):
            structure_id = row["str_id"].strip()
            split = row["split_label"].strip()
            connection = connections[split]
            if args.resume:
                exists = connection.execute(
                    "SELECT 1 FROM structures WHERE structure_id = ?", (structure_id,)
                ).fetchone()
                if exists:
                    processed["resumed"] += 1
                    continue
            pdb_name = args.pdb_template.format(str_id=structure_id)
            pdb_path = args.pdb_dir / pdb_name
            try:
                if not pdb_path.is_file():
                    raise TCRpMHCDataError(f"PDB not found: {pdb_path}")
                chains = parse_pdb_chains(pdb_path)
                missing_tcr = [chain for chain in TCR_CHAINS if chain not in chains]
                if missing_tcr:
                    raise TCRpMHCDataError(
                        f"{pdb_name}: missing required TCR chains {','.join(missing_tcr)}"
                    )
                if "M" not in chains or "P" not in chains:
                    raise TCRpMHCDataError(f"{pdb_name}: MHC chain M and peptide P are required")
                pair_count = insert_structure(
                    connection=connection,
                    structure_id=structure_id,
                    split=split,
                    source_pdb=pdb_path,
                    chains=chains,
                    patch_size=args.patch_size,
                    min_residue_contacts=args.min_residue_contacts,
                    heavy_atom_cutoff=args.heavy_atom_cutoff,
                    ca_prefilter_cutoff=args.ca_prefilter_cutoff,
                )
                processed[split] += 1
                pairs[split] += pair_count
                if pair_count == 0:
                    processed["zero_pair_structures"] += 1
            except (OSError, ValueError, TCRpMHCDataError) as exc:
                failure = {"str_id": structure_id, "split": split, "error": str(exc)}
                failures.append(failure)
                if not args.skip_errors:
                    raise
    finally:
        for connection in connections.values():
            connection.close()

    database_totals = {}
    for split in SPLITS:
        connection = sqlite3.connect(args.output_dir / f"{split}.sqlite")
        try:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            database_totals[split] = {
                "structures": connection.execute("SELECT COUNT(*) FROM structures").fetchone()[0],
                "pairs": connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0],
                "chains": connection.execute("SELECT COUNT(*) FROM chains").fetchone()[0],
            }
        finally:
            connection.close()

    report = {
        "rows_requested": len(rows),
        "current_invocation": {
            "processed_structures": dict(processed),
            "retained_pairs": dict(pairs),
        },
        "database_totals": database_totals,
        "failures": failures,
        "configuration": metadata,
    }
    report_path = args.output_dir / "preprocessing_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        print(f"Completed with {len(failures)} failures; see {report_path}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
