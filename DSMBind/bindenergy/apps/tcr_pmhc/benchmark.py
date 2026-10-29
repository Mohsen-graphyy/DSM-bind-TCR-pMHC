"""Portable PDB inference: pack -> prepare -> cached ESM -> two checkpoints -> ranking."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from .benchmark_io import (
    atlas_rows, pack_bundle, prepare_bundle, read_csv, verify_bundle,
    digest_file, signature, write_json,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    atlas = commands.add_parser("pack-atlas", help="Create a portable ATLAS bundle from the ranking index")
    atlas.add_argument("--index", type=Path, required=True)
    atlas.add_argument("--atlas-manifest", type=Path, required=True)
    atlas.add_argument("--atlas-root", type=Path)
    atlas.add_argument("--output-dir", type=Path, required=True)
    atlas.add_argument("--zip", action="store_true")
    generic = commands.add_parser("pack", help="Bundle new structures using an explicit CSV chain map")
    generic.add_argument("--manifest", type=Path, required=True)
    generic.add_argument("--output-dir", type=Path, required=True)
    generic.add_argument("--zip", action="store_true")
    for name in ("prepare", "run", "infer"):
        command = commands.add_parser(name)
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--work-dir", type=Path, required=True)
        command.add_argument("--patch-size", type=int, default=50)
        command.add_argument("--backbone-policy", choices=("strict", "ca"), default="ca")
        if name in ("run", "infer"):
            if name == "run":
                command.add_argument("--pretrained", type=Path, required=True)
                command.add_argument("--finetuned", type=Path, required=True)
            else:
                command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--embeddings", type=Path, required=True,
                                 help="New or existing compatible ESM SQLite cache; only missing sequences are computed")
            command.add_argument("--device", default="cuda")
            command.add_argument("--batch-size", type=int, default=4)
            command.add_argument("--esm-token-budget", type=int, default=512)
            command.add_argument("--bootstrap", type=int, default=2000)
            command.add_argument("--seed", type=int, default=7)
            command.add_argument("--preparation-only", action="store_true",
                                 help="Check bundle/checkpoints and print workload, without GPU/ESM loading")
    evaluation = commands.add_parser("evaluate", help="Recompute ranking reports from existing scores on CPU")
    evaluation.add_argument("--bundle", type=Path, required=True)
    evaluation.add_argument("--work-dir", type=Path, required=True)
    evaluation.add_argument("--output-dir", type=Path)
    evaluation.add_argument("--bootstrap", type=int, default=2000)
    evaluation.add_argument("--seed", type=int, default=7)
    return root


def run_process(arguments: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    package_root = str(Path(__file__).resolve().parents[3])
    environment["PYTHONPATH"] = package_root + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("OMP_NUM_THREADS", "8")
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\nCOMMAND: " + json.dumps(arguments) + "\n")
        stream.flush()
        process = subprocess.Popen([sys.executable, "-u", *arguments], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace", env=environment)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
        result = process.wait()
        if result:
            raise RuntimeError(f"Stage failed with exit code {result}; see {log}")


def embedding_fingerprint(cache: Path, dataset: Path) -> str:
    """Hash only the embeddings actually used, not a possibly 33-GB shared cache."""
    import hashlib
    from bindenergy.data.tcr_pmhc import distinct_sequences
    required = distinct_sequences([dataset])
    connection = sqlite3.connect(f"{cache.resolve().as_uri()}?mode=ro", uri=True)
    h = hashlib.sha256()
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if (json.loads(metadata.get("model", '""')) != "esm2_t36_3B_UR50D"
                or json.loads(metadata.get("layer", "0")) != 36
                or json.loads(metadata.get("dimension", "0")) != 2560):
            raise ValueError("Expected ESM-2 t36 3B layer 36, dimension 2560")
        h.update(json.dumps(metadata, sort_keys=True).encode())
        for key, sequence in required:
            row = connection.execute("SELECT sequence,length,dimension,dtype,data FROM embeddings WHERE sequence_key=?", (key,)).fetchone()
            if row is None or row[0] != sequence or row[1] != len(sequence) or row[2] != 2560:
                raise ValueError(f"Missing/incompatible embedding {key}")
            h.update(key.encode())
            h.update(row[3].encode())
            h.update(row[4])
    finally:
        connection.close()
    return h.hexdigest()


def completed_score(directory: Path, fingerprint: str) -> bool:
    marker = directory / "completion.json"
    if not marker.is_file():
        return False
    record = json.loads(marker.read_text(encoding="utf-8"))
    if record["fingerprint"] != fingerprint:
        raise ValueError(f"Score inputs/checkpoint/runtime changed; use a new work directory: {directory}")
    for name, checksum in record["output_sha256"].items():
        if not (directory / name).is_file() or digest_file(directory / name) != checksum:
            raise ValueError(f"Completed score output was changed: {directory / name}")
    return True


def run(args) -> dict:
    import torch
    from .score import load_parts
    if args.batch_size <= 0 or args.esm_token_budget <= 0 or args.bootstrap < 0:
        raise ValueError("Invalid batch, token budget or bootstrap count")
    args.bundle = args.bundle.resolve()
    args.work_dir = args.work_dir.resolve()
    args.embeddings = args.embeddings.resolve()
    checkpoints = ({"model": args.checkpoint.resolve()} if args.command == "infer" else
                   {"pretrained": args.pretrained.resolve(), "finetuned": args.finetuned.resolve()})
    if len(checkpoints) == 2 and digest_file(checkpoints["pretrained"]) == digest_file(checkpoints["finetuned"]):
        raise ValueError("Pretrained and fine-tuned checkpoint files are identical")
    architectures = {}
    for name, checkpoint_path in checkpoints.items():
        state, architecture = load_parts(checkpoint_path, torch.device("cpu"))
        if (architecture.esm_size != 2560 or architecture.bert_size != 2560
                or architecture.patch_size != args.patch_size or architecture.threshold != 14):
            raise ValueError("This benchmark expects 2560-D embeddings, 14-Angstrom cutoff and checkpoint-matching patch size")
        architectures[name] = vars(architecture)
        del state
    report = prepare_bundle(args.bundle, args.work_dir, args.patch_size, args.backbone_policy)
    print(json.dumps({"workload": report, "architectures": architectures}, indent=2), flush=True)
    if args.preparation_only:
        return {"status": "prepared_only", **report}
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use preparation-only locally, then run on Vast")
    # Fail before paying for ESM if real SRU is not importable/compatible on this machine.
    from bindenergy.models.energy import AllAtomEnergyModel
    del AllAtomEnergyModel
    run_process(["-m", "bindenergy.apps.tcr_pmhc.embeddings", "--datasets", str(args.work_dir / "dataset.sqlite"),
                 "--output", str(args.embeddings), "--device", args.device,
                 "--token-budget", str(args.esm_token_budget)], args.work_dir / "logs" / "embeddings.log")
    emb_fingerprint = embedding_fingerprint(args.embeddings, args.work_dir / "dataset.sqlite")
    package_root = Path(__file__).parents[2]
    implementation = {str(p.relative_to(package_root)): signature(p.read_text(encoding="utf-8")) for p in (
        package_root / "data" / "tcr_pmhc.py", package_root / "models" / "energy.py",
        package_root / "models" / "frame.py", Path(__file__).with_name("score.py"))}
    runtime = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
               "platform": platform.platform()}
    for package in ("sru", "fair-esm", "numpy", "scipy"):
        runtime[package] = importlib.metadata.version(package)
    import sru
    sru_root = Path(sru.__file__).parent
    runtime["sru_source_sha256"] = {
        name: digest_file(sru_root / name) for name in
        ("modules.py", "ops.py", "cuda_functional.py", "csrc/sru_cuda_kernel.cu")
        if (sru_root / name).is_file()
    }
    if args.device.startswith("cuda"):
        runtime["gpu"] = torch.cuda.get_device_name(torch.device(args.device))
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
    for name, checkpoint_path in checkpoints.items():
        directory = args.work_dir / name
        identity = {"checkpoint_sha256": digest_file(checkpoint_path), "prepared": report["fingerprint"],
                    "dataset_sha256": report["dataset_sha256"], "embedding_sha256_required_subset": emb_fingerprint,
                    "batch_size": args.batch_size, "device": args.device, "implementation": implementation,
                    "runtime": runtime, "score_direction": "higher"}
        fingerprint = signature(identity)
        if completed_score(directory, fingerprint):
            print(f"Reusing verified complete {name} scores", flush=True)
            continue
        start = time.monotonic()
        run_process(["-m", "bindenergy.apps.tcr_pmhc.score", "--dataset", str(args.work_dir / "dataset.sqlite"),
                     "--embeddings", str(args.embeddings), "--checkpoint", str(checkpoint_path),
                     "--output-dir", str(directory), "--device", args.device,
                     "--batch-size", str(args.batch_size)], args.work_dir / "logs" / f"{name}.log")
        write_json(directory / "completion.json", {"fingerprint": fingerprint, "inputs": identity,
                   "elapsed_seconds": time.monotonic() - start, "git_commit": git_commit,
                   "output_sha256": {file: digest_file(directory / file) for file in
                                     ("pair_scores.csv", "complex_scores.csv", "scoring_summary.json")}})
    from .ranking import evaluate, load_scores
    if args.command == "run":
        result = evaluate(args.bundle, args.work_dir, args.work_dir / "evaluation", args.bootstrap, args.seed)
        outcome = {"status": "complete", "report": str(args.work_dir / "evaluation" / "report.md"),
                   "observations": result["observations"], "structures": result["unique_structures"]}
    else:
        from .benchmark_io import write_csv
        structures, observations = verify_bundle(args.bundle)
        predictions = load_scores(args.work_dir / "model", {r["structure_id"] for r in structures})
        write_csv(args.work_dir / "predictions.csv", [
            {"observation_id": r["observation_id"], "structure_id": r["structure_id"],
             "peptide": r["peptide"], **predictions[r["structure_id"]]} for r in observations])
        outcome = {"status": "complete", "predictions": str(args.work_dir / "predictions.csv")}
    write_json(args.work_dir / "run_manifest.json", {"git_commit": git_commit, "runtime": runtime,
               "checkpoints_sha256": {name: digest_file(path) for name, path in checkpoints.items()},
               "bundle": json.loads((args.bundle / "bundle.json").read_text(encoding="utf-8")),
               "preparation": report, "required_embedding_sha256": emb_fingerprint,
               "evaluation": {"seed": args.seed, "bootstrap": args.bootstrap, "primary_score": "score_sum"}})
    return outcome


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.command == "pack-atlas":
        rows = atlas_rows(args.index.resolve(), args.atlas_manifest.resolve(), args.atlas_root)
        result = pack_bundle(rows, args.output_dir.resolve(), args.index, args.zip)
    elif args.command == "pack":
        result = pack_bundle(read_csv(args.manifest), args.output_dir.resolve(), args.manifest.resolve(), args.zip)
    elif args.command == "prepare":
        result = prepare_bundle(args.bundle.resolve(), args.work_dir.resolve(), args.patch_size, args.backbone_policy)
    elif args.command in ("run", "infer"):
        result = run(args)
    else:
        from .ranking import evaluate
        result = evaluate(args.bundle, args.work_dir, args.output_dir or args.work_dir / "evaluation", args.bootstrap, args.seed)
        result = {"report": str(args.output_dir or args.work_dir / "evaluation"), "observations": result["observations"]}
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
