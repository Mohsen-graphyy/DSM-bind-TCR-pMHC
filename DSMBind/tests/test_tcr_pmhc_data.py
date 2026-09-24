from __future__ import annotations

import sqlite3
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np
import torch

from bindenergy.data.constants import ALPHABET, RES_ATOM14
from bindenergy.data.tcr_pmhc import (
    EMBEDDING_SCHEMA,
    ChainRecord,
    EmbeddingStore,
    TCRpMHCPairDataset,
    initialize_dataset_database,
    insert_structure,
    make_model_batch,
    parse_pdb_chains,
    sequence_key,
)


def chain(sequence: str, offset: tuple[float, float, float]) -> ChainRecord:
    rows = np.zeros((len(sequence), 14, 3), dtype=np.float32)
    offset_array = np.asarray(offset, dtype=np.float32)
    for residue_index, aa in enumerate(sequence):
        atom_names = RES_ATOM14[ALPHABET.index(aa)]
        for atom_index, atom_name in enumerate(atom_names):
            if atom_name:
                rows[residue_index, atom_index] = (
                    offset_array
                    + np.asarray((residue_index * 3.8, atom_index * 0.1, 0.0), dtype=np.float32)
                )
    return ChainRecord(sequence=sequence, coords=rows)


def atom_line(serial: int, atom: str, residue: str, chain_id: str, residue_id: int, xyz) -> str:
    return (
        f"ATOM  {serial:5d} {atom:>4s} {residue:>3s} {chain_id}{residue_id:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{1.0:6.2f}{20.0:6.2f}          {atom[0]:>2s}\n"
    )


class TCRpMHCDataTests(unittest.TestCase):
    def test_database_metadata_can_be_reopened_for_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.sqlite"
            metadata = {
                "format_version": 1,
                "tcr_chains": ["A", "B"],
                "pmhc_chains": ["M", "N", "P"],
            }
            initialize_dataset_database(path, metadata).close()
            initialize_dataset_database(path, metadata).close()

    def test_pdb_parser_produces_atom14(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.pdb"
            lines = []
            serial = 1
            for chain_id, shift in (("A", 0.0), ("M", 4.0)):
                for atom, xyz in (
                    ("N", (shift, 0.0, 0.0)),
                    ("CA", (shift + 1.0, 0.0, 0.0)),
                    ("C", (shift + 2.0, 0.0, 0.0)),
                    ("O", (shift + 3.0, 0.0, 0.0)),
                    ("CB", (shift + 1.0, 1.0, 0.0)),
                ):
                    lines.append(atom_line(serial, atom, "ALA", chain_id, 1, xyz))
                    serial += 1
            path.write_text("".join(lines) + "END\n", encoding="ascii")
            parsed = parse_pdb_chains(path, ("A", "M"))
            self.assertEqual(parsed["A"].sequence, "A")
            self.assertEqual(parsed["A"].coords.shape, (1, 14, 3))
            np.testing.assert_allclose(parsed["A"].coords[0, 1], (1.0, 0.0, 0.0))

    def test_database_retains_only_cross_partner_pairs_and_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "train.sqlite"
            connection = initialize_dataset_database(dataset_path, {"format_version": 1})
            chains = {
                "A": chain("AAAA", (1.0, 0.0, 0.0)),
                "B": chain("RRRR", (1.0, 1.0, 0.0)),
                "M": chain("GGGG", (1.0, 0.0, 3.0)),
                "P": chain("SS", (1.0, 0.0, 2.0)),
            }
            pair_count = insert_structure(
                connection,
                structure_id="sample",
                split="train",
                source_pdb="sample.pdb",
                chains=chains,
                patch_size=3,
                min_residue_contacts=1,
                heavy_atom_cutoff=5.0,
                ca_prefilter_cutoff=12.0,
            )
            connection.close()
            self.assertEqual(pair_count, 4)

            embedding_path = root / "embeddings.sqlite"
            embedding_connection = sqlite3.connect(embedding_path)
            embedding_connection.executescript(EMBEDDING_SCHEMA)
            for record in chains.values():
                array = np.arange(len(record.sequence) * 8, dtype=np.float16).reshape(
                    len(record.sequence), 8
                )
                embedding_connection.execute(
                    """
                    INSERT OR IGNORE INTO embeddings(
                        sequence_key, sequence, length, dimension, dtype, data
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sequence_key(record.sequence),
                        record.sequence,
                        len(record.sequence),
                        8,
                        "float16",
                        zlib.compress(array.tobytes(), 1),
                    ),
                )
            embedding_connection.commit()
            embedding_connection.close()

            dataset = TCRpMHCPairDataset(dataset_path)
            store = EmbeddingStore(embedding_path)
            try:
                self.assertEqual(len(dataset), 4)
                chain_pairs = {
                    (dataset[index]["binder_chain"], dataset[index]["target_chain"])
                    for index in range(len(dataset))
                }
                self.assertEqual(chain_pairs, {("A", "M"), ("A", "P"), ("B", "M"), ("B", "P")})
                binder, target = make_model_batch(
                    [dataset[0], dataset[1]], store, torch.device("cpu")
                )
                self.assertEqual(binder[0].shape, (2, 3, 14, 3))
                self.assertEqual(binder[1].shape, (2, 3, 8))
                self.assertEqual(target[0].shape[0], 2)
                self.assertTrue(torch.isfinite(binder[1]).all())
            finally:
                dataset.close()
                store.close()


if __name__ == "__main__":
    unittest.main()
