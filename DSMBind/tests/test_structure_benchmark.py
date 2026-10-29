from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bindenergy.apps.tcr_pmhc.benchmark_io import (
    pack_bundle, prepare_bundle, read_csv, verify_bundle, write_csv, write_json,
    digest_file, atlas_rows,
)
from bindenergy.apps.tcr_pmhc.benchmark import completed_score
from bindenergy.apps.tcr_pmhc.ranking import rank_metrics, aggregate_units, evaluate, SCORE_FIELDS
from bindenergy.data.tcr_pmhc import parse_pdb_chains, TCRpMHCDataError
from bindenergy.data.constants import ALPHABET, RES_ATOM14, RESTYPE_1to3


def pdb(path, variant="AA"):
    lines, serial = [], 1
    for chain, sequence, offset in (("D", "ST", 0), ("E", variant, 1), ("H", "AG", 3), ("L", "GS", 2)):
        for number, residue in enumerate(sequence, 1):
            for atom_index, atom in enumerate(RES_ATOM14[ALPHABET.index(residue)]):
                if not atom:
                    continue
                xyz = (number * 3.8 + 1, offset + (atom_index % 3) * .3, 2 + (atom_index // 3) * .2)
                lines.append(f"ATOM  {serial:5d} {atom:>4s} {RESTYPE_1to3[residue]:>3s} {chain}{number:4d}    "
                             f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{1.0:6.2f}{20.0:6.2f}          {atom[0]:>2s}\n")
                serial += 1
    path.write_text("".join(lines) + "END\n", encoding="ascii")


def manifest_row(oid, path, energy=""):
    return {"observation_id": oid, "pdb_path": str(path), "chain_map": json.dumps({"D": "B", "E": "A", "H": "M", "L": "P"}),
            "peptide": "GS", "delta_g_kcal_mol": energy, "strict_group_id": "GS|HLA|study|298|SPR"}


class StructureBenchmarkTests(unittest.TestCase):
    def test_64_residue_sequence_is_not_mistaken_for_embedding_hash(self):
        from bindenergy.data.tcr_pmhc import EmbeddingStore, sequence_key
        store = object.__new__(EmbeddingStore)
        store._get_cached = lambda key: key
        self.assertEqual(store["A" * 64], sequence_key("A" * 64))
        key = sequence_key("AA")
        self.assertEqual(store[key], key)

    def test_single_checkpoint_inference_cli_needs_no_second_model(self):
        from bindenergy.apps.tcr_pmhc.benchmark import parser
        args = parser().parse_args(["infer", "--bundle", "bundle", "--work-dir", "out",
                                    "--checkpoint", "best.pt", "--embeddings", "cache.sqlite"])
        self.assertEqual(args.command, "infer")
        self.assertEqual(args.checkpoint, Path("best.pt"))

    def test_raw_score_direction_and_average_ties(self):
        self.assertAlmostEqual(rank_metrics([8, 7, 6], [3, 2, 1])["spearman"], 1)
        self.assertAlmostEqual(rank_metrics([8, 7, 6], [1, 2, 3])["spearman"], -1)
        self.assertEqual(rank_metrics([8, 7], [1, 1])["pairwise_accuracy"], .5)
        constant = rank_metrics([8, 7, 6], [1, 1, 1])
        self.assertIsNone(constant["spearman"])
        self.assertEqual(constant["ineligible_reason"], "constant_prediction")

    def test_repeat_measurements_not_counted_as_independent_structures(self):
        observations = [{"observation_id": str(i), "peptide": "P", "unit_id": "U", "structure_id": sid,
                         "delta_g_kcal_mol": energy} for i, (sid, energy) in enumerate([("S1", -6), ("S1", -8), ("S2", -7)])]
        scores = {name: {sid: {field: value for field in SCORE_FIELDS} for sid, value in (("S1", 5), ("S2", 9))}
                  for name in ("pretrained", "finetuned")}
        unit = aggregate_units(observations, scores, "peptide")[0]
        self.assertEqual(unit["n_observations"], 3)
        self.assertEqual(unit["n_structures"], 2)
        self.assertEqual(unit["delta_g_median"], -7)
        self.assertEqual(unit["pretrained_score_sum"], 7)

    def test_bundle_portability_sequence_identity_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "source.pdb"
            pdb(path)
            source = root / "source.csv"
            source.write_text("input", encoding="utf-8")
            bundle = root / "portable"
            result = pack_bundle([manifest_row("o1", path, -7), manifest_row("o2", path, -8)], bundle, source)
            self.assertEqual(result["structures"], 1)
            self.assertEqual(result["observations"], 2)
            self.assertEqual(result["biological_units"], 1)
            target = root / "moved"
            bundle.rename(target)
            structures, obs = verify_bundle(target)
            self.assertEqual(len(obs), 2)
            self.assertTrue((target / structures[0]["pdb_path"]).is_file())
            (target / structures[0]["pdb_path"]).write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_bundle(target)

    def test_invalid_peptide_and_duplicate_ids_fail_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "source.pdb"
            pdb(path)
            source = root / "source.csv"
            source.write_text("input", encoding="utf-8")
            row = manifest_row("o1", path, -7)
            row["peptide"] = "WRONG"
            with self.assertRaisesRegex(ValueError, "peptide"):
                pack_bundle([row], root / "out", source)
            self.assertFalse((root / "out").exists())
            row = manifest_row("o1", path, -7)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                pack_bundle([row, row], root / "out", source)

    def test_prepare_reuses_exact_database_and_masks_missing_backbone(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "source.pdb"
            pdb(path)
            path.write_text("\n".join(l for l in path.read_text().splitlines() if not (l.startswith("ATOM") and l[12:16].strip() == "N")))
            source = root / "source.csv"
            source.write_text("input", encoding="utf-8")
            pack_bundle([manifest_row("o1", path, -7)], root / "bundle", source)
            report = prepare_bundle(root / "bundle", root / "work")
            self.assertEqual(report["structures"], 1)
            self.assertGreater(report["missing_backbone_records"], 0)
            self.assertEqual(prepare_bundle(root / "bundle", root / "work"), report)
            with self.assertRaisesRegex(ValueError, "config changed"):
                prepare_bundle(root / "bundle", root / "work", patch_size=2)
            with self.assertRaises(TCRpMHCDataError):
                parse_pdb_chains(path, ("D", "E", "H", "L"))

    def test_nonstandard_atlas_spacing_uses_correct_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "offset.pdb"
            path.write_text("ATOM   2954  CA  PRO C 1      63.681  -2.946  -0.029  1.00 48.02           C  \n")
            parsed = parse_pdb_chains(path, ("C",), required_backbone_atoms=("CA",), allow_whitespace_records=True)
            np.testing.assert_allclose(parsed["C"].coords[0, 1], [63.681, -2.946, -.029], atol=1e-5)
            with self.assertRaises(TCRpMHCDataError):
                parse_pdb_chains(path, ("C",), required_backbone_atoms=("CA",))

    def test_score_cache_checks_input_and_output_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            score = root / "pair_scores.csv"
            score.write_text("example", encoding="utf-8")
            self.assertFalse(completed_score(root, "f"))
            write_json(root / "completion.json", {"fingerprint": "f", "output_sha256": {"pair_scores.csv": digest_file(score)}})
            self.assertTrue(completed_score(root, "f"))
            with self.assertRaisesRegex(ValueError, "changed"):
                completed_score(root, "different")
            score.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                completed_score(root, "f")

    def test_end_to_end_evaluation_joins_by_id_and_detects_worse_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.csv"
            source.write_text("source", encoding="utf-8")
            rows = []
            for i, variant in enumerate(("AA", "AV", "AG")):
                path = root / f"{i}.pdb"
                pdb(path, variant)
                rows.append(manifest_row(f"o{i}", path, -8 + i))
            pack_bundle(rows, root / "bundle", source)
            structures, obs = verify_bundle(root / "bundle")
            sid_by_id = {r["observation_id"]: r["structure_id"] for r in obs}
            work = root / "work"
            work.mkdir()
            write_csv(work / "geometry_pairs.csv", [{"structure_id": sid_by_id[f"o{i}"], "atom_pairs_14A": 10 + i} for i in range(3)])
            for name in ("pretrained", "finetuned"):
                complexes, pairs = [], []
                for i in range(3):
                    sid = sid_by_id[f"o{i}"]
                    score = 3 - i if name == "pretrained" else i + 1
                    complexes.append({"structure_id": sid, "pair_count": 2, "score_sum": score,
                                      "score_mean": score / 2, "total_contacts": 20, "score_per_contact": score / 20})
                    for target in ("M", "P"):
                        pairs.append({"structure_id": sid, "binder_chain": "A", "target_chain": target,
                                      "contact_count": 10, "min_distance": 3, "symmetric_score": score / 2,
                                      "forward_score": score / 4, "reverse_score": score / 4})
                write_csv(work / name / "complex_scores.csv", list(reversed(complexes)))
                write_csv(work / name / "pair_scores.csv", pairs)
            result = evaluate(root / "bundle", work, root / "report", bootstrap=10)
            change = next(r for r in result["paired_comparisons"] if r["grouping"] == "peptide" and r["score_field"] == "score_sum")
            self.assertAlmostEqual(change["mean_delta_spearman_finetuned_minus_pretrained"], -2)
            self.assertIsNone(change["paired_group_bootstrap_95ci"])
            self.assertTrue((root / "report" / "report.md").is_file())
            self.assertEqual(len(read_csv(root / "report" / "observation_scores.csv")), 3)
            self.assertNotIn("NaN", (root / "report" / "metrics.json").read_text())
            # Reject incomplete inference instead of evaluating a favorable subset.
            write_csv(work / "finetuned" / "complex_scores.csv", complexes[:-1])
            with self.assertRaisesRegex(ValueError, "Missing/extra"):
                evaluate(root / "bundle", work, root / "report")


if __name__ == "__main__":
    unittest.main()
