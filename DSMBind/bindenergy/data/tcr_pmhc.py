"""Scalable TCR-pMHC input pipeline for the DSMBind protein-protein model.

The published DSMBind protein pipeline consumes pairs of protein chains.  A
TCR-pMHC complex is therefore represented by the cross-partner chain pairs
between TCR chains A/B and pMHC chains M/N/P.  Coordinates are stored once per
chain in SQLite; pair rows hold the two interface patches used by DSMBind.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch

from bindenergy.data.constants import ALPHABET, ATOM_TYPES, RES_ATOM14


TCR_CHAINS = ("A", "B")
PMHC_CHAINS = ("M", "N", "P")
STANDARD_CHAINS = frozenset(TCR_CHAINS + PMHC_CHAINS)

AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


class TCRpMHCDataError(RuntimeError):
    """Raised when a structure cannot be converted without ambiguity."""


@dataclass(frozen=True)
class ChainRecord:
    sequence: str
    coords: np.ndarray  # [L, 14, 3], float32


def sequence_key(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _encode_array(array: np.ndarray, compression_level: int = 1) -> bytes:
    return zlib.compress(np.ascontiguousarray(array).tobytes(), compression_level)


def _decode_array(blob: bytes, dtype: np.dtype, shape: Sequence[int]) -> np.ndarray:
    raw = zlib.decompress(blob)
    array = np.frombuffer(raw, dtype=dtype)
    expected = int(np.prod(shape))
    if array.size != expected:
        raise TCRpMHCDataError(
            f"Corrupt array: expected {expected} values, found {array.size}"
        )
    return array.reshape(tuple(shape)).copy()


def parse_pdb_chains(
    pdb_path: str | Path,
    wanted_chains: Iterable[str] = STANDARD_CHAINS,
) -> dict[str, ChainRecord]:
    """Parse canonical ATOM records into DSMBind's residue atom14 layout.

    Alternative locations are resolved by highest occupancy, with blank/A
    locations preferred when occupancies tie.  A residue must contain N, CA,
    and C; incomplete side chains are represented by zero coordinates.
    """

    pdb_path = Path(pdb_path)
    wanted = frozenset(wanted_chains)
    residues: dict[str, OrderedDict[tuple[int, str], dict]] = {
        chain: OrderedDict() for chain in wanted
    }

    with pdb_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.startswith("ATOM  ") or len(line) < 54:
                continue
            chain = line[21:22]
            if chain not in wanted:
                continue
            atom_name = line[12:16].strip()
            altloc = line[16:17]
            residue_name = line[17:20].strip().upper()
            if residue_name not in AA3_TO_AA1:
                raise TCRpMHCDataError(
                    f"{pdb_path.name}:{line_number}: unsupported residue {residue_name!r}"
                )
            try:
                residue_number = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                occupancy = float(line[54:60].strip() or "0")
            except ValueError as exc:
                raise TCRpMHCDataError(
                    f"{pdb_path.name}:{line_number}: malformed coordinate record"
                ) from exc
            insertion_code = line[26:27]
            key = (residue_number, insertion_code)
            residue = residues[chain].setdefault(
                key, {"name": residue_name, "atoms": {}}
            )
            if residue["name"] != residue_name:
                raise TCRpMHCDataError(
                    f"{pdb_path.name}:{line_number}: residue identity changes at "
                    f"{chain}:{residue_number}{insertion_code.strip()}"
                )

            # Higher occupancy wins.  On a tie prefer blank, then A.
            altloc_rank = 2 if altloc == " " else 1 if altloc == "A" else 0
            previous = residue["atoms"].get(atom_name)
            candidate = (occupancy, altloc_rank, np.asarray((x, y, z), dtype=np.float32))
            if previous is None or candidate[:2] > previous[:2]:
                residue["atoms"][atom_name] = candidate

    result: dict[str, ChainRecord] = {}
    for chain in sorted(wanted):
        chain_residues = residues[chain]
        if not chain_residues:
            continue
        sequence_chars: list[str] = []
        coordinate_rows: list[np.ndarray] = []
        for (residue_number, insertion_code), residue in chain_residues.items():
            atoms = residue["atoms"]
            missing_backbone = [name for name in ("N", "CA", "C") if name not in atoms]
            if missing_backbone:
                raise TCRpMHCDataError(
                    f"{pdb_path.name}: {chain}:{residue_number}{insertion_code.strip()} "
                    f"missing backbone atoms {','.join(missing_backbone)}"
                )
            aa = AA3_TO_AA1[residue["name"]]
            atom_layout = RES_ATOM14[ALPHABET.index(aa)]
            row = np.zeros((14, 3), dtype=np.float32)
            for atom_index, atom_name in enumerate(atom_layout):
                if atom_name and atom_name in atoms:
                    row[atom_index] = atoms[atom_name][2]
            sequence_chars.append(aa)
            coordinate_rows.append(row)
        result[chain] = ChainRecord(
            sequence="".join(sequence_chars),
            coords=np.stack(coordinate_rows).astype(np.float32, copy=False),
        )
    return result


def interface_statistics(
    first: ChainRecord,
    second: ChainRecord,
    heavy_atom_cutoff: float = 5.0,
    ca_prefilter_cutoff: float = 12.0,
) -> tuple[int, float]:
    """Return residue-contact count and minimum heavy-atom distance."""

    first_ca = first.coords[:, 1]
    second_ca = second.coords[:, 1]
    ca_distances = np.linalg.norm(first_ca[:, None, :] - second_ca[None, :, :], axis=-1)
    candidate_pairs = np.argwhere(ca_distances <= ca_prefilter_cutoff)
    contact_count = 0
    minimum = float("inf")
    cutoff_sq = heavy_atom_cutoff * heavy_atom_cutoff

    for first_index, second_index in candidate_pairs:
        first_atoms = first.coords[first_index]
        second_atoms = second.coords[second_index]
        first_atoms = first_atoms[np.any(first_atoms != 0, axis=1)]
        second_atoms = second_atoms[np.any(second_atoms != 0, axis=1)]
        squared = np.sum(
            (first_atoms[:, None, :] - second_atoms[None, :, :]) ** 2,
            axis=-1,
        )
        pair_min_sq = float(squared.min())
        minimum = min(minimum, pair_min_sq ** 0.5)
        if pair_min_sq <= cutoff_sq:
            contact_count += 1

    if not np.isfinite(minimum):
        # The CA prefilter found no candidates.  This value is exact enough for
        # rejection/reporting and avoids a full all-atom Cartesian product.
        minimum = float(ca_distances.min())
    return contact_count, minimum


def select_centroid_patch(
    chain: ChainRecord,
    partner: ChainRecord,
    patch_size: int,
) -> np.ndarray:
    """Reproduce the centroid-based patch selection in ProteinDataset."""

    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    partner_center = partner.coords[:, 1].mean(axis=0, keepdims=True)
    distances = np.linalg.norm(chain.coords[:, 1] - partner_center, axis=-1)
    count = min(patch_size, len(chain.sequence))
    return np.sort(np.argsort(distances)[:count]).astype(np.int32)


DATASET_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chains (
    structure_id TEXT NOT NULL,
    chain_id TEXT NOT NULL,
    sequence TEXT NOT NULL,
    sequence_key TEXT NOT NULL,
    length INTEGER NOT NULL,
    coords BLOB NOT NULL,
    PRIMARY KEY (structure_id, chain_id)
);
CREATE TABLE IF NOT EXISTS structures (
    structure_id TEXT PRIMARY KEY,
    split TEXT NOT NULL,
    source_pdb TEXT NOT NULL,
    pair_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    structure_id TEXT NOT NULL,
    split TEXT NOT NULL,
    binder_chain TEXT NOT NULL,
    target_chain TEXT NOT NULL,
    binder_indices BLOB NOT NULL,
    binder_count INTEGER NOT NULL,
    target_indices BLOB NOT NULL,
    target_count INTEGER NOT NULL,
    contact_count INTEGER NOT NULL,
    min_distance REAL NOT NULL,
    source_pdb TEXT NOT NULL,
    UNIQUE (structure_id, binder_chain, target_chain)
);
CREATE INDEX IF NOT EXISTS idx_pairs_structure ON pairs(structure_id);
CREATE INDEX IF NOT EXISTS idx_chains_sequence_key ON chains(sequence_key);
"""


EMBEDDING_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    sequence_key TEXT PRIMARY KEY,
    sequence TEXT NOT NULL,
    length INTEGER NOT NULL,
    dimension INTEGER NOT NULL,
    dtype TEXT NOT NULL,
    data BLOB NOT NULL
);
"""


def initialize_dataset_database(path: str | Path, metadata: Mapping[str, object]) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(DATASET_SCHEMA)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    immutable_keys = {
        "format_version",
        "patch_size",
        "min_residue_contacts",
        "heavy_atom_cutoff",
        "ca_prefilter_cutoff",
        "tcr_chains",
        "pmhc_chains",
    }
    existing = {
        key: json.loads(value)
        for key, value in connection.execute("SELECT key, value FROM metadata")
    }
    for key in immutable_keys:
        if key in existing and key in metadata and existing[key] != metadata[key]:
            connection.close()
            raise TCRpMHCDataError(
                f"Database {path} was created with {key}={existing[key]!r}; "
                f"requested {metadata[key]!r}. Use a new output directory."
            )
    for key, value in metadata.items():
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (key, json.dumps(value, sort_keys=True)),
        )
    connection.commit()
    return connection


def insert_structure(
    connection: sqlite3.Connection,
    structure_id: str,
    split: str,
    source_pdb: str | Path,
    chains: Mapping[str, ChainRecord],
    patch_size: int,
    min_residue_contacts: int,
    heavy_atom_cutoff: float,
    ca_prefilter_cutoff: float,
) -> int:
    """Atomically insert one structure and return its retained pair count."""

    pair_rows: list[tuple] = []
    used_chains: set[str] = set()
    for binder_chain in TCR_CHAINS:
        if binder_chain not in chains:
            continue
        for target_chain in PMHC_CHAINS:
            if target_chain not in chains:
                continue
            binder = chains[binder_chain]
            target = chains[target_chain]
            contact_count, minimum = interface_statistics(
                binder,
                target,
                heavy_atom_cutoff=heavy_atom_cutoff,
                ca_prefilter_cutoff=ca_prefilter_cutoff,
            )
            if contact_count < min_residue_contacts:
                continue
            binder_indices = select_centroid_patch(binder, target, patch_size)
            target_indices = select_centroid_patch(target, binder, patch_size)
            pair_rows.append(
                (
                    structure_id,
                    split,
                    binder_chain,
                    target_chain,
                    _encode_array(binder_indices),
                    len(binder_indices),
                    _encode_array(target_indices),
                    len(target_indices),
                    contact_count,
                    minimum,
                    str(source_pdb),
                )
            )
            used_chains.update((binder_chain, target_chain))

    with connection:
        connection.execute(
            "DELETE FROM chains WHERE structure_id = ?",
            (structure_id,),
        )
        for chain_id in sorted(used_chains):
            chain = chains[chain_id]
            connection.execute(
                """
                INSERT OR REPLACE INTO chains(
                    structure_id, chain_id, sequence, sequence_key, length, coords
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    structure_id,
                    chain_id,
                    chain.sequence,
                    sequence_key(chain.sequence),
                    len(chain.sequence),
                    _encode_array(chain.coords),
                ),
            )
        connection.execute("DELETE FROM pairs WHERE structure_id = ?", (structure_id,))
        connection.executemany(
            """
            INSERT INTO pairs(
                structure_id, split, binder_chain, target_chain,
                binder_indices, binder_count, target_indices, target_count,
                contact_count, min_distance, source_pdb
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            pair_rows,
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO structures(structure_id, split, source_pdb, pair_count)
            VALUES (?, ?, ?, ?)
            """,
            (structure_id, split, str(source_pdb), len(pair_rows)),
        )
    return len(pair_rows)


class EmbeddingStore:
    """Read-only, bounded-memory access to a SQLite ESM embedding cache."""

    def __init__(self, path: str | Path, cache_size: int = 128):
        self.path = str(Path(path))
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA query_only=ON")
        self.metadata = {
            key: json.loads(value)
            for key, value in self.connection.execute("SELECT key, value FROM metadata")
        }
        self._get_cached = lru_cache(maxsize=cache_size)(self._get_uncached)

    def _get_uncached(self, key: str) -> torch.Tensor:
        row = self.connection.execute(
            "SELECT length, dimension, dtype, data FROM embeddings WHERE sequence_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Embedding not found for sequence key {key}")
        length, dimension, dtype_name, blob = row
        dtype = np.dtype(dtype_name)
        array = _decode_array(blob, dtype, (length, dimension)).astype(np.float32)
        return torch.from_numpy(array)

    def __getitem__(self, sequence_or_key: str) -> torch.Tensor:
        key = sequence_or_key if len(sequence_or_key) == 64 else sequence_key(sequence_or_key)
        return self._get_cached(key)

    def close(self) -> None:
        self._get_cached.cache_clear()
        self.connection.close()

    def require_sequences(self, dataset_paths: Sequence[str | Path]) -> None:
        required = {key for key, _ in distinct_sequences(dataset_paths)}
        available = {
            row[0] for row in self.connection.execute("SELECT sequence_key FROM embeddings")
        }
        missing = required - available
        if missing:
            examples = ", ".join(sorted(missing)[:5])
            raise TCRpMHCDataError(
                f"Embedding cache is missing {len(missing):,} required sequences "
                f"(examples: {examples})"
            )


class TCRpMHCPairDataset:
    """Lazy pair dataset backed by a preprocessing SQLite database."""

    def __init__(self, path: str | Path, chain_cache_size: int = 128):
        self.path = str(Path(path))
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA query_only=ON")
        self.pair_ids = [row[0] for row in self.connection.execute("SELECT id FROM pairs ORDER BY id")]
        self._load_chain_cached = lru_cache(maxsize=chain_cache_size)(self._load_chain_uncached)

    def __len__(self) -> int:
        return len(self.pair_ids)

    def _load_chain_uncached(self, structure_id: str, chain_id: str) -> ChainRecord:
        row = self.connection.execute(
            """
            SELECT sequence, length, coords FROM chains
            WHERE structure_id = ? AND chain_id = ?
            """,
            (structure_id, chain_id),
        ).fetchone()
        if row is None:
            raise TCRpMHCDataError(f"Missing chain {structure_id}:{chain_id}")
        sequence, length, blob = row
        coords = _decode_array(blob, np.dtype("float32"), (length, 14, 3))
        return ChainRecord(sequence=sequence, coords=coords)

    def __getitem__(self, index: int) -> dict:
        pair_id = self.pair_ids[index]
        row = self.connection.execute(
            """
            SELECT structure_id, split, binder_chain, target_chain,
                   binder_indices, binder_count, target_indices, target_count,
                   contact_count, min_distance, source_pdb
            FROM pairs WHERE id = ?
            """,
            (pair_id,),
        ).fetchone()
        if row is None:
            raise IndexError(index)
        (
            structure_id, split, binder_chain, target_chain,
            binder_blob, binder_count, target_blob, target_count,
            contact_count, min_distance, source_pdb,
        ) = row
        binder = self._load_chain_cached(structure_id, binder_chain)
        target = self._load_chain_cached(structure_id, target_chain)
        binder_indices = _decode_array(binder_blob, np.dtype("int32"), (binder_count,))
        target_indices = _decode_array(target_blob, np.dtype("int32"), (target_count,))
        return {
            "pair_id": pair_id,
            "structure_id": structure_id,
            "split": split,
            "source_pdb": source_pdb,
            "binder_chain": binder_chain,
            "target_chain": target_chain,
            "contact_count": contact_count,
            "min_distance": min_distance,
            "binder_full": binder.sequence,
            "binder_seq": "".join(binder.sequence[i] for i in binder_indices),
            "binder_coords": torch.from_numpy(binder.coords[binder_indices]),
            "binder_idx": torch.from_numpy(binder_indices.astype(np.int64)),
            "target_full": target.sequence,
            "target_seq": "".join(target.sequence[i] for i in target_indices),
            "target_coords": torch.from_numpy(target.coords[target_indices]),
            "target_idx": torch.from_numpy(target_indices.astype(np.int64)),
        }

    def iter_batches(
        self,
        batch_size: int,
        indices: Sequence[int] | None = None,
    ) -> Iterator[list[dict]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        selected = range(len(self)) if indices is None else indices
        for start in range(0, len(selected), batch_size):
            positions = selected[start : start + batch_size]
            yield [self[position] for position in positions]

    def close(self) -> None:
        self._load_chain_cached.cache_clear()
        self.connection.close()


def _atom_types(sequence: str) -> torch.Tensor:
    return torch.tensor(
        [[ATOM_TYPES.index(atom) for atom in RES_ATOM14[ALPHABET.index(aa)]] for aa in sequence],
        dtype=torch.long,
    )


def make_model_batch(
    batch: Sequence[dict],
    embeddings: EmbeddingStore,
    device: torch.device,
    swap_partners: bool = False,
) -> tuple[tuple, tuple]:
    """Create the two padded AllAtomEnergyModel inputs for a pair batch."""

    binder_name, target_name = ("target", "binder") if swap_partners else ("binder", "target")

    def featurize_partner(name: str) -> tuple:
        max_length = max(len(entry[f"{name}_seq"]) for entry in batch)
        embedding_dim = embeddings[batch[0][f"{name}_full"]].shape[1]
        coords = torch.zeros((len(batch), max_length, 14, 3), dtype=torch.float32, device=device)
        sequence_features = torch.zeros(
            (len(batch), max_length, embedding_dim), dtype=torch.float32, device=device
        )
        atom_types = torch.zeros((len(batch), max_length, 14), dtype=torch.long, device=device)
        dihedrals = torch.zeros((len(batch), max_length, 6), dtype=torch.float32, device=device)
        for batch_index, entry in enumerate(batch):
            sequence = entry[f"{name}_seq"]
            length = len(sequence)
            indices = entry[f"{name}_idx"]
            coords[batch_index, :length] = entry[f"{name}_coords"].to(device)
            atom_types[batch_index, :length] = _atom_types(sequence).to(device)
            full_embedding = embeddings[entry[f"{name}_full"]]
            sequence_features[batch_index, :length] = full_embedding[indices].to(device)
        return coords, sequence_features, atom_types, dihedrals

    return featurize_partner(binder_name), featurize_partner(target_name)


def distinct_sequences(dataset_paths: Sequence[str | Path]) -> list[tuple[str, str]]:
    """Collect distinct (hash, sequence) pairs from one or more dataset DBs."""

    sequences: dict[str, str] = {}
    for path in dataset_paths:
        connection = sqlite3.connect(str(path))
        try:
            for key, sequence in connection.execute(
                "SELECT DISTINCT sequence_key, sequence FROM chains"
            ):
                previous = sequences.setdefault(key, sequence)
                if previous != sequence:
                    raise TCRpMHCDataError(f"SHA-256 collision for {key}")
        finally:
            connection.close()
    return sorted(sequences.items(), key=lambda item: (len(item[1]), item[0]))
