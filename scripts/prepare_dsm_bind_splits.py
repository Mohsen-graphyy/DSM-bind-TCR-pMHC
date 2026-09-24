#!/usr/bin/env python3
"""Validate and organize TCR-pMHC PDB files using the dataset split CSV.

The source directory used by this project contains one five-chain complex per
row: TCR chains A/B and pMHC chains M/N/P.  This script can:

1. verify the CSV-to-PDB mapping and DSMBind-relevant PDB properties;
2. organize the combined PDB files into train/validation/test/final_unseen_data;
3. optionally extract the two partners named by ``tcr_pdb_member`` and
   ``pmhc_pdb_member`` in the CSV.

Only Python's standard library is required.  The resulting PDB files are
structurally suitable for a DSMBind preprocessor, but DSMBind's protein and
antibody dataset classes do not consume raw PDB files directly.  They require
preprocessed 14-heavy-atom coordinate tensors in pickle/JSONL form.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "data" / "linked_data" / "final_dataset_datasail_like_C1f_4labels_fixed.csv"
DEFAULT_PDB_DIR = REPO_ROOT / "data" / "new_structure"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "pdb_splits"

REQUIRED_COLUMNS = {
    "str_id",
    "split_label",
    "tcr_pdb_member",
    "pmhc_pdb_member",
}
ALLOWED_SPLITS = ("train", "validation", "test", "final_unseen_data")
TCR_CHAINS = frozenset({"A", "B"})
PMHC_CHAINS = frozenset({"M", "N", "P"})
SUPPORTED_CHAINS = TCR_CHAINS | PMHC_CHAINS
BASE_EXPECTED_CHAINS = frozenset({"A", "B", "M", "P"})

# DSMBind's bindenergy/data/constants.py uses these 20 residue and heavy-atom
# names.  Hydrogens are ignored by the checker because the model does not use
# them.
AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
DSMBIND_ATOMS = {
    "N", "CA", "C", "O", "CB", "CG", "CG1", "CG2", "OG", "OG1",
    "SG", "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD",
    "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2",
    "CH2", "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
}
DSMBIND_RESIDUE_ATOMS = {
    "ALA": frozenset({"N", "CA", "C", "O", "CB"}),
    "ARG": frozenset({"N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"}),
    "ASN": frozenset({"N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"}),
    "ASP": frozenset({"N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"}),
    "CYS": frozenset({"N", "CA", "C", "O", "CB", "SG"}),
    "GLN": frozenset({"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"}),
    "GLU": frozenset({"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"}),
    "GLY": frozenset({"N", "CA", "C", "O"}),
    "HIS": frozenset({"N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"}),
    "ILE": frozenset({"N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"}),
    "LEU": frozenset({"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"}),
    "LYS": frozenset({"N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"}),
    "MET": frozenset({"N", "CA", "C", "O", "CB", "CG", "SD", "CE"}),
    "PHE": frozenset({"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"}),
    "PRO": frozenset({"N", "CA", "C", "O", "CB", "CG", "CD"}),
    "SER": frozenset({"N", "CA", "C", "O", "CB", "OG"}),
    "THR": frozenset({"N", "CA", "C", "O", "CB", "OG1", "CG2"}),
    "TRP": frozenset({"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"}),
    "TYR": frozenset({"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"}),
    "VAL": frozenset({"N", "CA", "C", "O", "CB", "CG1", "CG2"}),
}
SEQUENCE_COLUMNS = {
    # The generated structures use B for TCR alpha/delta and A for beta/gamma.
    # This is verified against the sequences in the supplied CSV; it is not
    # inferred from alphabetical chain order.
    "A": "FV_beta/gamma",
    "B": "FV_alpha/delta",
    "M": "mhc.aseq",
    "N": "mhc.bseq",
    "P": "antigen.epitope",
}


class DatasetError(RuntimeError):
    """Raised for an invalid CSV, unsafe output path, or inconsistent mapping."""


@dataclass
class PDBCheck:
    chains: set[str] = field(default_factory=set)
    sequences: dict[str, str] = field(default_factory=dict)
    atom_count: int = 0
    residue_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def compatible_for_preprocessing(self) -> bool:
        return not self.errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and organize combined TCR-pMHC PDB files by CSV split."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Split CSV path")
    parser.add_argument("--pdb-dir", type=Path, default=DEFAULT_PDB_DIR, help="Combined PDB directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output root")
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="How combined PDBs are materialized (default: hardlink, no extra file data)",
    )
    parser.add_argument(
        "--extract-members",
        action="store_true",
        help="Also write chain-filtered TCR (A/B) and pMHC (M/N/P) PDB files",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Perform all checks but do not create directories or files",
    )
    parser.add_argument(
        "--validate",
        choices=("mapping", "structure", "sequence"),
        default="sequence",
        help="Validation depth; sequence is the strictest (default: sequence)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many CSV rows (useful for a smoke test)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing destination files instead of failing",
    )
    parser.add_argument(
        "--allow-extra-chains",
        action="store_true",
        help="Warn instead of fail if chains other than A/B/M/N/P are present",
    )
    return parser.parse_args(argv)


def clean_cell(row: dict[str, str], name: str) -> str:
    return (row.get(name) or "").strip()


def read_dataset(csv_path: Path, limit: int | None) -> tuple[list[dict[str, str]], list[str]]:
    if not csv_path.is_file():
        raise DatasetError(f"CSV file not found: {csv_path}")
    if limit is not None and limit <= 0:
        raise DatasetError("--limit must be a positive integer")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing_columns = REQUIRED_COLUMNS - set(fieldnames)
        if missing_columns:
            raise DatasetError(f"CSV is missing columns: {sorted(missing_columns)}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            row["__line__"] = str(line_number)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break

    seen_ids: dict[str, int] = {}
    seen_destinations: dict[str, int] = {}
    for row in rows:
        line_number = int(row["__line__"])
        structure_id = clean_cell(row, "str_id")
        split = clean_cell(row, "split_label")
        if not structure_id:
            raise DatasetError(f"CSV line {line_number}: blank str_id")
        if any(char in structure_id for char in ("/", "\\")) or structure_id in {".", ".."}:
            raise DatasetError(f"CSV line {line_number}: unsafe str_id {structure_id!r}")
        if structure_id in seen_ids:
            raise DatasetError(
                f"Duplicate str_id {structure_id!r} on lines {seen_ids[structure_id]} and {line_number}"
            )
        seen_ids[structure_id] = line_number
        if split not in ALLOWED_SPLITS:
            raise DatasetError(
                f"CSV line {line_number}: unsupported split_label {split!r}; "
                f"expected one of {ALLOWED_SPLITS}"
            )
        for column in ("tcr_pdb_member", "pmhc_pdb_member"):
            member = clean_cell(row, column)
            if not member or Path(member).name != member or not member.lower().endswith(".pdb"):
                raise DatasetError(f"CSV line {line_number}: unsafe or blank {column}={member!r}")
            destination_key = f"{split}/{column}/{member.casefold()}"
            if destination_key in seen_destinations:
                raise DatasetError(
                    f"Duplicate destination filename {member!r} in split {split!r}"
                )
            seen_destinations[destination_key] = line_number

    return rows, fieldnames


def _parse_atom_line(line: str, line_number: int, pdb_path: Path) -> tuple[str, str, str, str, str, str]:
    if len(line) < 54:
        raise ValueError(f"line {line_number}: ATOM record is shorter than 54 columns")
    atom = line[12:16].strip()
    altloc = line[16:17]
    residue = line[17:20].strip().upper()
    chain = line[21:22]
    residue_id = line[22:26].strip()
    insertion_code = line[26:27]
    try:
        float(line[30:38])
        float(line[38:46])
        float(line[46:54])
    except ValueError as exc:
        raise ValueError(f"line {line_number}: invalid XYZ coordinate") from exc
    if not chain.strip():
        raise ValueError(f"line {line_number}: blank chain identifier")
    if not residue_id:
        raise ValueError(f"line {line_number}: blank residue number")
    return atom, altloc, residue, chain, residue_id, insertion_code


def check_pdb(
    pdb_path: Path,
    row: dict[str, str],
    check_sequences: bool,
    allow_extra_chains: bool,
) -> PDBCheck:
    result = PDBCheck()
    residues: dict[str, dict[tuple[str, str], dict[str, object]]] = defaultdict(dict)
    duplicate_atoms = 0
    ignored_altlocs = 0
    unsupported_atoms: Counter[str] = Counter()

    try:
        with pdb_path.open("r", encoding="ascii", errors="strict") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.startswith("ATOM  "):
                    continue
                result.atom_count += 1
                try:
                    atom, altloc, residue, chain, residue_id, insertion_code = _parse_atom_line(
                        line, line_number, pdb_path
                    )
                except ValueError as exc:
                    result.errors.append(str(exc))
                    continue

                result.chains.add(chain)
                if altloc not in {" ", "A"}:
                    ignored_altlocs += 1
                    continue
                if residue not in AA3_TO_AA1:
                    result.errors.append(
                        f"line {line_number}: non-canonical residue {residue!r} is not in DSMBind's alphabet"
                    )
                    continue
                if atom not in DSMBIND_ATOMS and not atom.startswith("H"):
                    unsupported_atoms[atom] += 1

                key = (residue_id, insertion_code)
                chain_residues = residues[chain]
                if key not in chain_residues:
                    chain_residues[key] = {"name": residue, "atoms": set()}
                elif chain_residues[key]["name"] != residue:
                    result.errors.append(
                        f"line {line_number}: residue identity changes at chain {chain} {residue_id}{insertion_code}"
                    )
                atoms = chain_residues[key]["atoms"]
                assert isinstance(atoms, set)
                if atom in atoms and altloc in {" ", "A"}:
                    duplicate_atoms += 1
                atoms.add(atom)
    except (OSError, UnicodeError) as exc:
        result.errors.append(f"cannot read PDB: {exc}")
        return result

    if result.atom_count == 0:
        result.errors.append("no ATOM records")
        return result

    # Chain N is the second MHC chain and is absent when mhc.bseq is blank.
    expected_chains = set(BASE_EXPECTED_CHAINS)
    if clean_cell(row, "mhc.bseq"):
        expected_chains.add("N")
    missing_chains = expected_chains - result.chains
    extra_chains = result.chains - SUPPORTED_CHAINS
    if missing_chains:
        result.errors.append(f"missing expected chains: {','.join(sorted(missing_chains))}")
    if extra_chains:
        message = f"unexpected chains: {','.join(sorted(extra_chains))}"
        (result.warnings if allow_extra_chains else result.errors).append(message)

    missing_atoms: list[str] = []
    for chain, chain_residues in residues.items():
        sequence: list[str] = []
        for (residue_id, insertion_code), residue_data in chain_residues.items():
            name = str(residue_data["name"])
            atoms = residue_data["atoms"]
            assert isinstance(atoms, set)
            sequence.append(AA3_TO_AA1[name])
            missing = DSMBIND_RESIDUE_ATOMS[name] - atoms
            if missing and len(missing_atoms) < 10:
                missing_atoms.append(
                    f"{chain}:{residue_id}{insertion_code.strip()} missing {','.join(sorted(missing))}"
                )
        result.sequences[chain] = "".join(sequence)
        result.residue_count += len(chain_residues)

    if missing_atoms:
        result.errors.append(
            "residues with incomplete DSMBind heavy atoms (first 10): " + "; ".join(missing_atoms)
        )
    if duplicate_atoms:
        result.warnings.append(f"{duplicate_atoms} duplicate primary-conformer atom records")
    if ignored_altlocs:
        result.warnings.append(f"ignored {ignored_altlocs} alternate-location atom records")
    if unsupported_atoms:
        names = ",".join(sorted(unsupported_atoms))
        result.warnings.append(f"heavy atom names not used by DSMBind: {names}")

    if check_sequences:
        for chain, column in SEQUENCE_COLUMNS.items():
            expected = clean_cell(row, column).replace(" ", "").upper()
            observed = result.sequences.get(chain, "")
            # A blank mhc.bseq is valid in some CSV rows; structure checks still
            # ensure that chain N exists.
            if expected and observed != expected:
                result.errors.append(
                    f"chain {chain} sequence differs from CSV column {column}: "
                    f"PDB length={len(observed)}, CSV length={len(expected)}"
                )

    return result


def materialize_file(source: Path, destination: Path, mode: str, overwrite: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            # Idempotent reruns are safe when both paths identify the same file.
            try:
                if source.samefile(destination):
                    return "existing"
            except OSError:
                pass
            raise DatasetError(f"Destination already exists (use --overwrite): {destination}")
        if destination.is_dir():
            raise DatasetError(f"Destination is a directory, refusing to replace it: {destination}")
        destination.unlink()

    if mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        shutil.copy2(source, destination)
    return "created"


def extract_chains(source: Path, destination: Path, chains: frozenset[str], overwrite: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            raise DatasetError(f"Destination already exists (use --overwrite): {destination}")
        if destination.is_dir():
            raise DatasetError(f"Destination is a directory, refusing to replace it: {destination}")
        destination.unlink()

    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    selected_atoms = 0
    last_chain: str | None = None
    try:
        with source.open("r", encoding="ascii", errors="strict") as input_handle, temporary.open(
            "w", encoding="ascii", newline="\n"
        ) as output_handle:
            for line in input_handle:
                record = line[:6].strip()
                if record in {"HEADER", "TITLE", "COMPND", "SOURCE", "REMARK", "CRYST1"}:
                    output_handle.write(line.rstrip("\r\n") + "\n")
                elif record == "ATOM" and len(line) >= 22 and line[21] in chains:
                    chain = line[21]
                    if last_chain is not None and chain != last_chain:
                        output_handle.write("TER\n")
                    output_handle.write(line.rstrip("\r\n") + "\n")
                    selected_atoms += 1
                    last_chain = chain
            if last_chain is not None:
                output_handle.write("TER\n")
            output_handle.write("END\n")
        if selected_atoms == 0:
            raise DatasetError(f"No atoms for chains {sorted(chains)} in {source}")
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return "created"


def write_split_manifest(
    path: Path,
    rows: Iterable[dict[str, str]],
    original_fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(original_fieldnames) + ["combined_pdb_path", "tcr_pdb_path", "pmhc_pdb_path"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for input_row in rows:
            row = dict(input_row)
            split = clean_cell(row, "split_label")
            structure_id = clean_cell(row, "str_id")
            row["combined_pdb_path"] = str(Path(split) / "complex" / f"{structure_id}_Complex.pdb")
            row["tcr_pdb_path"] = str(Path(split) / "tcr" / clean_cell(row, "tcr_pdb_member"))
            row["pmhc_pdb_path"] = str(Path(split) / "pmhc" / clean_cell(row, "pmhc_pdb_member"))
            writer.writerow(row)


def run(args: argparse.Namespace) -> int:
    csv_path = args.csv.resolve()
    pdb_dir = args.pdb_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not pdb_dir.is_dir():
        raise DatasetError(f"PDB directory not found: {pdb_dir}")
    if output_dir == pdb_dir or pdb_dir in output_dir.parents:
        raise DatasetError("--output-dir must not be the source PDB directory or one of its children")

    rows, fieldnames = read_dataset(csv_path, args.limit)
    split_counts = Counter(clean_cell(row, "split_label") for row in rows)
    expected_source_names = {f"{clean_cell(row, 'str_id')}_Complex.pdb" for row in rows}
    extra_source_pdbs: list[str] = []
    if args.limit is None:
        extra_source_pdbs = sorted(
            path.name
            for path in pdb_dir.glob("*_Complex.pdb")
            if path.name not in expected_source_names
        )
    failures: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    checked = 0
    total_atoms = 0
    total_residues = 0
    operation_counts: Counter[str] = Counter()

    print(f"CSV: {csv_path}")
    print(f"PDB directory: {pdb_dir}")
    print(f"Rows: {len(rows):,}")
    print("Splits: " + ", ".join(f"{name}={split_counts[name]:,}" for name in ALLOWED_SPLITS))
    if extra_source_pdbs:
        print(
            f"Extra source PDBs not referenced by the CSV: {len(extra_source_pdbs):,} "
            f"({', '.join(extra_source_pdbs[:10])})"
        )

    for index, row in enumerate(rows, start=1):
        structure_id = clean_cell(row, "str_id")
        split = clean_cell(row, "split_label")
        source = pdb_dir / f"{structure_id}_Complex.pdb"
        if not source.is_file():
            failures.append({"str_id": structure_id, "file": str(source), "errors": ["source PDB is missing"]})
            continue

        if args.validate != "mapping":
            check = check_pdb(
                source,
                row,
                check_sequences=args.validate == "sequence",
                allow_extra_chains=args.allow_extra_chains,
            )
            checked += 1
            total_atoms += check.atom_count
            total_residues += check.residue_count
            if check.errors:
                failures.append({"str_id": structure_id, "file": str(source), "errors": check.errors})
            if check.warnings:
                warnings.append({"str_id": structure_id, "file": str(source), "warnings": check.warnings})

        if index % 1000 == 0 or index == len(rows):
            print(
                f"Processed {index:,}/{len(rows):,}; "
                f"validation failures={len(failures):,}, warnings={len(warnings):,}",
                flush=True,
            )

    # Validate the complete selection before writing anything.  This avoids a
    # half-built split tree when a missing or malformed structure is found.
    if not args.check_only and not failures:
        print("Validation passed; materializing split directories ...")
        for index, row in enumerate(rows, start=1):
            structure_id = clean_cell(row, "str_id")
            split = clean_cell(row, "split_label")
            source = pdb_dir / f"{structure_id}_Complex.pdb"
            combined_destination = output_dir / split / "complex" / source.name
            result = materialize_file(source, combined_destination, args.mode, args.overwrite)
            operation_counts[f"combined_{result}"] += 1
            if args.extract_members:
                tcr_destination = output_dir / split / "tcr" / clean_cell(row, "tcr_pdb_member")
                pmhc_destination = output_dir / split / "pmhc" / clean_cell(row, "pmhc_pdb_member")
                extract_chains(source, tcr_destination, TCR_CHAINS, args.overwrite)
                extract_chains(source, pmhc_destination, PMHC_CHAINS, args.overwrite)
                operation_counts["tcr_created"] += 1
                operation_counts["pmhc_created"] += 1
            if index % 1000 == 0 or index == len(rows):
                print(f"Materialized {index:,}/{len(rows):,}", flush=True)

    report = {
        "csv": str(csv_path),
        "pdb_directory": str(pdb_dir),
        "output_directory": None if args.check_only else str(output_dir),
        "check_only": args.check_only,
        "validation_level": args.validate,
        "rows": len(rows),
        "split_counts": {name: split_counts[name] for name in ALLOWED_SPLITS},
        "pdb_files_checked": checked,
        "atom_records_checked": total_atoms,
        "residues_checked": total_residues,
        "failed_files": len(failures),
        "warning_files": len(warnings),
        "extra_source_pdbs": extra_source_pdbs,
        "failure_examples": failures[:100],
        "warning_examples": warnings[:100],
        "operations": dict(operation_counts),
        "dsm_bind_assessment": {
            "pdb_compatible_for_preprocessing": (
                None if args.validate == "mapping" else len(failures) == 0
            ),
            "direct_raw_pdb_input_supported_by_protein_or_antibody_dataset": False,
            "required_next_format": "preprocessed sequence + [residue, 14, 3] coordinate arrays",
        },
    }

    if not args.check_only and not failures:
        for split in ALLOWED_SPLITS:
            split_rows = [row for row in rows if clean_cell(row, "split_label") == split]
            write_split_manifest(output_dir / split / "manifest.csv", split_rows, fieldnames)
        report_path = output_dir / "validation_report.json"
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    print("\nValidation summary")
    print(f"  checked PDBs: {checked:,}")
    print(f"  failed PDBs:  {len(failures):,}")
    print(f"  warning PDBs: {len(warnings):,}")
    if failures:
        print("  first failures:")
        for failure in failures[:10]:
            print(f"    {failure['str_id']}: {'; '.join(failure['errors'])}")
    if warnings:
        print("  first warnings:")
        for warning in warnings[:10]:
            print(f"    {warning['str_id']}: {'; '.join(warning['warnings'])}")
    if args.check_only:
        print("  no files were written (--check-only)")
    elif failures:
        print("  no files were written because validation failed")
    else:
        print(f"  output: {output_dir}")
        print("  operations: " + ", ".join(f"{key}={value:,}" for key, value in sorted(operation_counts.items())))

    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (DatasetError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
