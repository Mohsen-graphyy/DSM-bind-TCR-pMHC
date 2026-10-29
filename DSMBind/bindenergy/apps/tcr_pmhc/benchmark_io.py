"""Portable, label-separated PDB bundles for repeatable inference (no GPU needed)."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signature(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def ca_sequences(path: Path) -> dict[str, str]:
    """Read observed CA sequences; refuse multistate PDBs rather than mix models."""
    aa = dict(zip(
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
        "ARNDCQEGHILKMFPSTWYV",
    ))
    chains: dict[str, dict] = {}
    models = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MODEL "):
            models += 1
            if models > 1:
                raise ValueError(f"Multiple MODELs in {path}; supply one structural model")
        if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
            continue
        residue = line[17:20].strip()
        if residue not in aa:
            raise ValueError(f"Unsupported residue {residue} in {path}")
        key = line[22:27]
        chain = chains.setdefault(line[21], {})
        if key in chain and chain[key] != aa[residue]:
            raise ValueError(f"Ambiguous residue {path}:{line[21]}:{key}")
        chain[key] = aa[residue]
    return {chain: "".join(residues.values()) for chain, residues in chains.items()}


def atlas_rows(index: Path, manifest: Path | None, atlas_root: Path | None) -> list[dict]:
    source = read_csv(index)
    if not source:
        raise ValueError("Empty ATLAS index")
    required = {"Peptide", "TCR ID", "ATLAS observation ID", "delta G", "PDB path",
                "observed_tcr_alpha_aa", "observed_tcr_beta_aa"}
    if not required.issubset(source[0]):
        raise ValueError(f"ATLAS index missing columns: {required - set(source[0])}")
    metadata = {}
    labels = {}
    if manifest:
        metadata = {r["tcr_id"]: r for r in read_csv(manifest)}
        label_path = manifest.parent.parent / "evaluation_private" / "labels.csv"
        if label_path.is_file():
            labels = {r["tcr_id"]: r for r in read_csv(label_path)}
    result = []
    for row in source:
        oid = row["ATLAS observation ID"]
        meta = metadata.get(oid, {})
        if manifest and not meta:
            raise ValueError(f"Observation absent from ATLAS manifest: {oid}")
        pdb = (index.parent / row["PDB path"]).resolve()
        if atlas_root and meta:
            pdb = (atlas_root / meta["structure_relpath"]).resolve()
        if not pdb.is_file():
            raise FileNotFoundError(f"PDB for {oid}: {pdb}; supply --atlas-root and --atlas-manifest")
        expected_hash = meta.get("structure_sha256")
        if expected_hash and digest_file(pdb) != expected_hash:
            raise ValueError(f"PDB checksum changed for {oid}: {pdb}")
        sequences = ca_sequences(pdb)
        # Match exact observed sequences, never infer biological role from chain letters.
        roles = {
            "B": row["observed_tcr_alpha_aa"],  # same alpha/beta convention as training
            "A": row["observed_tcr_beta_aa"],
            "P": row["Peptide"],
        }
        if meta.get("observed_mhc_alpha_aa"):
            roles["M"] = meta["observed_mhc_alpha_aa"]
        if meta.get("observed_mhc_beta_aa"):
            roles["N"] = meta["observed_mhc_beta_aa"]
        mapping = {}
        for canonical, expected in roles.items():
            matches = [chain for chain, sequence in sequences.items() if sequence == expected]
            if len(matches) != 1:
                raise ValueError(f"{oid}: {canonical} sequence has {len(matches)} chain matches")
            mapping[matches[0]] = canonical
        if "M" not in mapping.values():
            raise ValueError("ATLAS chain roles need --atlas-manifest with observed_mhc_alpha_aa")
        if len(mapping) != len(roles):
            raise ValueError(f"Overlapping chain roles for {oid}")
        context = labels.get(oid, {})
        if context and float(context["DeltaG_kcal_per_mol"]) != float(row["delta G"]):
            raise ValueError(f"Index/label delta G mismatch for {oid}")
        strict = [row["Peptide"], meta.get("hla_long", ""), context.get("PMID", ""),
                  context.get("Temperature_K", ""), context.get("Exp_Method", "")]
        result.append({
            "observation_id": oid, "pdb_path": str(pdb), "chain_map": json.dumps(mapping, sort_keys=True),
            "peptide": row["Peptide"], "tcr_id": row["TCR ID"],
            "delta_g_kcal_mol": row["delta G"], "array_index": row["array_index"],
            "hla": meta.get("hla_long", ""), "pmid": context.get("PMID", ""),
            "temperature_k": context.get("Temperature_K", ""), "assay": context.get("Exp_Method", ""),
            "strict_group_id": "|".join(strict) if all(strict) else "",
        })
    return result


def pack_bundle(rows: list[dict], output: Path, source: Path, make_zip: bool = False) -> dict:
    """Input manifest: observation_id,pdb_path,chain_map,peptide[,delta_g_kcal_mol]."""
    if not rows:
        raise ValueError("No observations")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Bundle directory must be new/empty: {output}")
    observations, structures = [], {}
    ids = set()
    # Validate all input rows before materializing the bundle.
    for row in rows:
        oid = row.get("observation_id", "").strip()
        if not oid or oid in ids:
            raise ValueError(f"Duplicate/empty observation_id: {oid!r}")
        ids.add(oid)
        pdb = Path(row["pdb_path"])
        if not pdb.is_absolute():
            pdb = source.parent / pdb
        pdb = pdb.resolve()
        mapping = json.loads(row["chain_map"])
        if (not isinstance(mapping, dict) or any(len(k) != 1 for k in mapping)
                or any(not isinstance(v, str) or len(v) != 1 or v not in "ABMNP" for v in mapping.values())
                or len(set(mapping.values())) != len(mapping)
                or not {"A", "B", "M", "P"}.issubset(mapping.values())):
            raise ValueError(f"{oid}: chain_map must map unique PDB letters to A,B,M,P and optional N")
        sequences = ca_sequences(pdb)
        if not set(mapping).issubset(sequences):
            raise ValueError(f"Missing mapped chain in {pdb}")
        canonical = {role: sequences[chain] for chain, role in mapping.items()}
        peptide = row["peptide"].strip().upper()
        if peptide != canonical["P"]:
            raise ValueError(f"{oid}: experimental peptide does not match PDB peptide")
        energy = row.get("delta_g_kcal_mol", "")
        if energy != "" and (not math.isfinite(float(energy))):
            raise ValueError(f"Non-finite experimental delta G: {oid}")
        sha = digest_file(pdb)
        sid = "S_" + signature([sha, mapping])[:24]
        unit = "U_" + signature(canonical)[:24]
        relative = f"pdbs/{sha}.pdb"
        structures.setdefault(sid, {"structure_id": sid, "pdb_path": relative,
                                   "pdb_sha256": sha, "chain_map": json.dumps(mapping, sort_keys=True),
                                   "peptide": peptide, "_source": pdb})
        observations.append({**{k: v for k, v in row.items() if k not in {"pdb_path", "chain_map"}},
                             "observation_id": oid, "structure_id": sid, "unit_id": unit,
                             "peptide": peptide, "delta_g_kcal_mol": energy})
    output.mkdir(parents=True, exist_ok=True)
    (output / "pdbs").mkdir()
    for entry in structures.values():
        destination = output / entry["pdb_path"]
        if not destination.exists():
            shutil.copyfile(entry["_source"], destination)
    public_structures = [{k: v for k, v in entry.items() if k != "_source"} for entry in structures.values()]
    write_csv(output / "structures.csv", public_structures)
    write_csv(output / "observations.csv", observations)
    result = {"format_version": 1, "observations": len(observations), "structures": len(structures),
              "biological_units": len({r["unit_id"] for r in observations}),
              "peptide_counts": dict(Counter(r["peptide"] for r in observations)),
              "source_name": source.name, "source_sha256": digest_file(source),
              "structures_csv_sha256": digest_file(output / "structures.csv"),
              "observations_csv_sha256": digest_file(output / "observations.csv"),
              "score_direction": "higher DSMBind score = stronger; lower experimental delta G = stronger",
              "unit_identity": "all mapped observed TCR/pMHC chain sequences; repeats aggregated only in evaluation"}
    write_json(output / "bundle.json", result)
    if make_zip:
        archive = output.with_suffix(".zip")
        if archive.exists():
            raise FileExistsError(f"Refusing to overwrite archive {archive}")
        shutil.make_archive(str(output), "zip", root_dir=output.parent, base_dir=output.name)
        result["archive"] = str(archive)
    return result


def verify_bundle(bundle: Path) -> tuple[list[dict], list[dict]]:
    meta = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    for name in ("structures", "observations"):
        if digest_file(bundle / f"{name}.csv") != meta[f"{name}_csv_sha256"]:
            raise ValueError(f"Bundle {name}.csv changed; create a new bundle")
    structures, observations = read_csv(bundle / "structures.csv"), read_csv(bundle / "observations.csv")
    for row in structures:
        path = (bundle / row["pdb_path"]).resolve()
        if bundle.resolve() not in path.parents or digest_file(path) != row["pdb_sha256"]:
            raise ValueError(f"Bundle PDB path/checksum mismatch: {row['structure_id']}")
    if len({r["observation_id"] for r in observations}) != len(observations):
        raise ValueError("Nonunique observation IDs")
    return structures, observations


def prepare_bundle(bundle: Path, work: Path, patch_size: int = 50, backbone_policy: str = "ca") -> dict:
    """Build the existing DSMBind pair DB, keeping score inputs separate from labels."""
    import sqlite3
    import numpy as np
    from bindenergy.data.tcr_pmhc import (
        initialize_dataset_database, parse_pdb_chains, insert_structure,
        distinct_sequences, select_centroid_patch,
    )
    structures, _ = verify_bundle(bundle)
    if patch_size <= 0:
        raise ValueError("Patch size must be positive")
    fingerprint = signature({"structures": structures, "patch_size": patch_size,
                             "backbone_policy": backbone_policy, "implementation": signature(Path(__file__).read_text(encoding="utf-8")),
                             "data_implementation": signature((Path(__file__).parents[2] / "data" / "tcr_pmhc.py").read_text(encoding="utf-8"))})
    report_path = work / "preparation_report.json"
    db_path = work / "dataset.sqlite"
    if report_path.is_file() and db_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["fingerprint"] != fingerprint:
            raise ValueError("Prepared inputs/config changed; use a new --work-dir")
        if digest_file(db_path) != report["dataset_sha256"]:
            raise ValueError("Prepared database changed; use a new --work-dir")
        if digest_file(work / "geometry_pairs.csv") != report["geometry_sha256"]:
            raise ValueError("Prepared geometry changed; use a new --work-dir")
        return report
    work.mkdir(parents=True, exist_ok=True)
    if db_path.is_file():
        previous = sqlite3.connect(db_path)
        try:
            existing = previous.execute("SELECT value FROM metadata WHERE key='inference_fingerprint'").fetchone()
        finally:
            previous.close()
        if not existing or json.loads(existing[0]) != fingerprint:
            raise ValueError("Partial preprocessing inputs changed; use a new --work-dir")
    connection = initialize_dataset_database(db_path, {"format_version": 1, "patch_size": patch_size,
                                                       "inference_fingerprint": fingerprint})
    geometry, missing = [], []
    try:
        for entry in structures:
            mapping = json.loads(entry["chain_map"])
            original = parse_pdb_chains(bundle / entry["pdb_path"], mapping,
                                       required_backbone_atoms=("CA",) if backbone_policy == "ca" else ("N", "CA", "C"),
                                       allow_whitespace_records=True)
            chains = {canonical: original[source] for source, canonical in mapping.items()}
            for chain, record in chains.items():
                for atom, index in (("N", 0), ("C", 2), ("O", 3)):
                    count = int((~np.any(record.coords[:, index] != 0, axis=-1)).sum())
                    if count:
                        missing.append({"structure_id": entry["structure_id"], "chain": chain,
                                        "atom": atom, "missing_residues": count})
            pair_count = insert_structure(connection, entry["structure_id"], "inference", entry["pdb_path"],
                                          chains, patch_size, 1, 5.0, 12.0)
            if not pair_count:
                raise ValueError(f"No cross-partner contacts for {entry['structure_id']}")
            for binder_chain, target_chain in connection.execute(
                "SELECT binder_chain,target_chain FROM pairs WHERE structure_id=?", (entry["structure_id"],)
            ):
                first, second = chains[binder_chain], chains[target_chain]
                a = first.coords[select_centroid_patch(first, second, patch_size)].reshape(-1, 3)
                b = second.coords[select_centroid_patch(second, first, patch_size)].reshape(-1, 3)
                a, b = a[np.linalg.norm(a, axis=1) > 1e-4], b[np.linalg.norm(b, axis=1) > 1e-4]
                distances = np.linalg.norm(a[:, None] - b[None, :], axis=-1)
                geometry.append({"structure_id": entry["structure_id"], "binder_chain": binder_chain,
                                 "target_chain": target_chain, "atom_pairs_14A": int((distances < 14.0).sum())})
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    sequences = distinct_sequences([db_path])
    write_csv(work / "geometry_pairs.csv", geometry)
    write_csv(work / "missing_backbone_atoms.csv", missing,
              ["structure_id", "chain", "atom", "missing_residues"])
    report = {"fingerprint": fingerprint, "structures": len(structures), "pairs": len(geometry),
              "unique_sequences": len(sequences), "residues": sum(len(s) for _, s in sequences),
              "esm_float16_raw_bytes": sum(len(s) for _, s in sequences) * 2560 * 2,
              "patch_size": patch_size, "backbone_policy": backbone_policy,
              "missing_backbone_records": len(missing), "dataset_sha256": digest_file(db_path),
              "geometry_sha256": digest_file(work / "geometry_pairs.csv"),
              "geometry_cutoff_angstrom": 14.0}
    write_json(report_path, report)
    return report
