"""Read-only mixed-derivative diagnostic on real pair inputs; no optimizer steps."""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import math
import random
import time
import types
import warnings
from pathlib import Path

import numpy as np
import torch

from .benchmark_io import digest_file, write_csv, write_json


@contextlib.contextmanager
def fixed_rng(seed):
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda is not None:
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


@contextlib.contextmanager
def recurrence_backend(model, backend, required_device=None, device_audit=None):
    """Temporarily select the installed SRU pure-PyTorch reference recurrence."""
    if backend == "native":
        yield 0
        return
    if backend != "reference":
        raise ValueError(backend)
    from sru.ops import elementwise_recurrence_naive
    expected_device = torch.device(required_device) if required_device is not None else None
    verified_cells = set()

    def apply_reference(cell, U, V, residual, c0, scale_val, mask_c, mask_pad):
        # grad must remain enabled: upstream naive otherwise dispatches to
        # its inference kernel, which is unsuitable for mixed derivatives.
        if not torch.is_grad_enabled():
            raise RuntimeError("Reference diagnostic requires grad_enabled=True")
        actual = U.device
        if expected_device is not None and (actual.type != expected_device.type or
                (expected_device.index is not None and actual.index != expected_device.index)):
            raise RuntimeError(f"SRU reference input is on {actual}; required {expected_device}. Refusing device fallback.")
        inputs = dict(U=U, V=V, residual=residual, c0=c0, scale_val=scale_val,
                      mask_c=mask_c, mask_pad=mask_pad)
        for name, value in list(inputs.items()) + list(cell.named_parameters(recurse=False)):
            if isinstance(value, torch.Tensor) and value.device != actual:
                raise RuntimeError(f"SRU reference {name} is on {value.device}, input is on {actual}")
        # Upstream's naive implementation warns about CPU unconditionally when
        # grad is enabled, including with CUDA tensors. Suppress only that exact
        # misleading warning, only after checking the CUDA tensors/parameters.
        with warnings.catch_warnings():
            if actual.type == "cuda":
                warnings.filterwarnings("ignore",
                    message=r"^Running SRU on CPU with grad_enabled=True\. Are you sure\?$",
                    category=UserWarning, module=r"^sru\.ops$")
            result = elementwise_recurrence_naive(
                U, residual, V, cell.bias, c0, cell.activation_type,
                cell.hidden_size, cell.bidirectional, cell.has_skip_term,
                scale_val, mask_c, mask_pad,
            )
        if not all(isinstance(value, torch.Tensor) and value.device == actual for value in result):
            raise RuntimeError("SRU reference output changed device; refusing device fallback")
        if id(cell) not in verified_cells:
            verified_cells.add(id(cell))
            record = {"cell": cell_names[id(cell)], "input_device": str(actual),
                      "input_devices": {name: str(value.device) for name, value in inputs.items()
                                        if isinstance(value, torch.Tensor)},
                      "parameter_devices": {name: str(value.device) for name, value in cell.named_parameters(recurse=False)},
                      "output_devices": [str(value.device) for value in result],
                      "grad_enabled": torch.is_grad_enabled(), "native_custom_backward": False}
            if device_audit is not None:
                device_audit.append(record)
            print(f"[SRU reference verified] cell={record['cell']} input={actual} outputs={record['output_devices']} grad_enabled=True", flush=True)
        return result

    saved = []
    cell_names = {}
    try:
        for name, module in model.named_modules():
            if hasattr(module, "apply_recurrence") and hasattr(module, "weight_c"):
                saved.append((module, "apply_recurrence" in module.__dict__, module.apply_recurrence))
                cell_names[id(module)] = name
                module.apply_recurrence = types.MethodType(apply_reference, module)
        if not saved:
            raise RuntimeError("No real SRU cells found; reference backend not installed (FakeSRU is unsupported)")
        yield len(saved)
    finally:
        for module, had_instance_method, original in saved:
            if had_instance_method:
                module.apply_recurrence = original
            else:
                del module.apply_recurrence


def value_and_gradients(model, closure):
    """Return every parameter's gradient, keeping absent gradients distinct from zero."""
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    diagnostics = closure()
    objective = diagnostics["signal_loss"]
    if objective.requires_grad:
        gradients = torch.autograd.grad(objective, [p for _, p in parameters], allow_unused=True)
    else:
        gradients = [None] * len(parameters)
    values = {key: (float(value.detach()) if value.numel() == 1 else value.detach().cpu().reshape(-1).tolist())
              for key, value in diagnostics.items()}
    if not all(bool(np.isfinite(value).all()) for value in values.values()):
        raise FloatingPointError("Non-finite diagnostic: singular inertia or unstable example; try another --pair-index")
    copied = {name: None if grad is None else grad.detach().cpu().clone()
              for (name, _), grad in zip(parameters, gradients)}
    del diagnostics, objective, gradients
    return values, copied


def finite_difference(parameter, index, closure, epsilon):
    """Perturb only an in-memory scalar and always restore its exact original value."""
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Finite-difference epsilon must be finite and positive")
    flat = parameter.view(-1)
    original = flat[index].detach().clone()
    try:
        with torch.no_grad():
            flat[index].copy_(original + epsilon)
        plus = closure()
        high = float(plus["signal_loss"].detach())
        del plus
        with torch.no_grad():
            flat[index].copy_(original - epsilon)
        minus = closure()
        low = float(minus["signal_loss"].detach())
        del minus
    finally:
        with torch.no_grad():
            flat[index].copy_(original)
    return (high - low) / (2 * epsilon)


def assess_probe(automatic, differences, atol=1e-9, rtol=.2):
    """Do not pass numerically flat or epsilon-unstable finite differences."""
    if len(differences) < 2 or not all(math.isfinite(v) for v in differences):
        return "inconclusive"
    centre = float(np.median(differences))
    if max(abs(v - centre) for v in differences) > atol + rtol * abs(centre):
        return "inconclusive_fd_unstable"
    if abs(centre) <= 10 * atol:
        return "inconclusive_low_signal"
    if automatic is None:
        return "fail_missing_gradient"
    return "pass" if all(abs(automatic - v) <= atol + rtol * max(abs(automatic), abs(v)) for v in differences) else "fail_gradient_mismatch"


def crop_pair(entry, maximum):
    """Diagnostic-only small real interface; never overwrite benchmark/training DB."""
    if maximum < 3:
        raise ValueError("Use at least 3 residues per partner for backbone inertia")
    result = dict(entry)
    for name, partner in (("binder", "target"), ("target", "binder")):
        xyz = entry[f"{name}_coords"][:, 1]
        centre = entry[f"{partner}_coords"][:, 1].mean(0)
        indices = (xyz - centre).norm(dim=-1).argsort()[:maximum].sort().values
        result[f"{name}_coords"] = entry[f"{name}_coords"][indices].clone()
        result[f"{name}_idx"] = entry[f"{name}_idx"][indices].clone()
        result[f"{name}_seq"] = "".join(entry[f"{name}_seq"][i] for i in indices.tolist())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--finetuned", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--max-residues", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--component", choices=("translation", "full"), default="translation")
    parser.add_argument("--fd-atol", type=float, default=1e-9)
    parser.add_argument("--fd-rtol", type=float, default=.2)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[.01, .003])
    parser.add_argument("--probe-parameters", nargs="+", default=["W_o.2.weight", "U_o.0.weight"])
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args(argv)
    if len(args.epsilons) < 2 or any(not math.isfinite(x) or x <= 0 for x in args.epsilons):
        parser.error("Supply at least two positive finite epsilon values")
    if not all(math.isfinite(x) and x >= 0 for x in (args.fd_atol, args.fd_rtol)):
        parser.error("FD tolerances must be finite and nonnegative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use a new/empty output directory to preserve earlier diagnostics")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable: run on the existing Vast instance")
    # Import genuine SRU, never the CPU smoke-test replacement.
    import sru
    if not getattr(sru, "__file__", None):
        raise RuntimeError("A real installed SRU is required")
    from bindenergy.data.tcr_pmhc import TCRpMHCPairDataset, EmbeddingStore, make_model_batch
    from bindenergy.models.energy import AllAtomEnergyModel
    from .score import load_parts
    torch.set_num_threads(args.num_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    dataset, embeddings = TCRpMHCPairDataset(args.dataset), EmbeddingStore(args.embeddings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gradient_rows, probe_rows, model_reports = [], [], {}
    try:
        entry = crop_pair(dataset[args.pair_index], args.max_residues)
        binder, target = make_model_batch([entry], embeddings, device)
        for name, path in (("pretrained", args.pretrained), ("finetuned", args.finetuned)):
            print(f"Loading {name}; diagnostic crop={args.max_residues} residues per partner", flush=True)
            state, architecture = load_parts(path, device)
            model = AllAtomEnergyModel(architecture).to(device)
            model.load_state_dict(state, strict=True)
            del state
            model.eval()  # Same differentiable path, dropout disabled for comparison.
            named = dict(model.named_parameters())
            snapshots = {}
            probe_indices = {parameter_name: 0 for parameter_name in args.probe_parameters}
            for backend in ("reference", "native"):
                print(f"{name}: {backend} mixed derivatives", flush=True)
                start = time.monotonic()
                try:
                    with recurrence_backend(model, backend):
                        def closure():
                            with fixed_rng(args.seed), torch.enable_grad():
                                return model(binder, target, return_diagnostics=True,
                                             diagnostic_translation_only=args.component == "translation")
                        values, gradients = value_and_gradients(model, closure)
                        snapshots[backend] = {"values": values}
                        for parameter_name, gradient in gradients.items():
                            gradient_rows.append({"checkpoint": name, "backend": backend,
                                "parameter": parameter_name, "gradient_present": gradient is not None,
                                "gradient_norm": float(gradient.norm()) if gradient is not None else None,
                                "gradient_finite": bool(torch.isfinite(gradient).all()) if gradient is not None else None})
                        if backend == "reference":
                            probe_indices = {}
                            for parameter_name in args.probe_parameters:
                                if parameter_name not in gradients:
                                    raise ValueError(f"Unknown/non-trainable parameter: {parameter_name}")
                                g = gradients[parameter_name]
                                probe_indices[parameter_name] = int(g.abs().argmax()) if g is not None else 0
                        for parameter_name in args.probe_parameters:
                            index = probe_indices[parameter_name]
                            gradient = gradients[parameter_name]
                            if gradient is not None and not bool(torch.isfinite(gradient).all()):
                                raise FloatingPointError(f"Non-finite gradient: {parameter_name}")
                            automatic = float(gradient.reshape(-1)[index]) if gradient is not None else None
                            differences = []
                            for epsilon in args.epsilons:
                                differences.append(finite_difference(named[parameter_name], index, closure, epsilon))
                            status = assess_probe(automatic, differences, args.fd_atol, args.fd_rtol)
                            row = {"checkpoint": name, "backend": backend, "parameter": parameter_name,
                                   "flat_index": index, "automatic_derivative": automatic,
                                   "finite_differences": differences, "epsilons": args.epsilons, "status": status}
                            probe_rows.append(row)
                            print(f"  {parameter_name}: {status}; AD={automatic}; FD={differences}", flush=True)
                        del gradients
                except Exception as exc:
                    snapshots[backend] = {"error": f"{type(exc).__name__}: {exc}"}
                    print(f"  {backend} failed: {snapshots[backend]['error']}", flush=True)
                snapshots[backend]["seconds"] = time.monotonic() - start
                write_csv(args.output_dir / "parameter_gradients.csv", gradient_rows)
                write_json(args.output_dir / "finite_differences.json", probe_rows)
            if all("values" in snapshots.get(k, {}) for k in ("native", "reference")):
                a, b = snapshots["native"]["values"], snapshots["reference"]["values"]
                snapshots["backend_value_agreement"] = {
                    key: {"native": a[key], "reference": b[key],
                          "close": math.isclose(a[key], b[key], rel_tol=.01, abs_tol=1e-5)}
                    for key in ("loss", "force_rms", "perturbed_energy")}
                if "force_vector" in a and "force_vector" in b:
                    av, bv = np.asarray(a["force_vector"]), np.asarray(b["force_vector"])
                    snapshots["backend_value_agreement"]["force_vector"] = {
                        "relative_l2_difference": float(np.linalg.norm(av - bv) / max(np.linalg.norm(bv), 1e-12)),
                        "close": bool(np.allclose(av, bv, rtol=.02, atol=1e-7))}
            snapshots["checkpoint_sha256"] = digest_file(path)
            model_reports[name] = snapshots
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        dataset.close()
        embeddings.close()
    results = {"models": model_reports, "pair": {k: entry[k] for k in
               ("structure_id", "pair_id", "binder_chain", "target_chain", "binder_seq", "target_seq")},
               "max_residues": args.max_residues, "seed": args.seed, "component": args.component,
               "fd_tolerances": {"atol": args.fd_atol, "rtol": args.fd_rtol},
               "source_sha256": {"diagnostic": digest_file(Path(__file__)),
                                 "energy": digest_file(Path(__file__).parents[2] / "models" / "energy.py")},
               "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "sru": importlib.metadata.version("sru")},
               "probes": probe_rows,
               "note": "Diagnostic crop and fixed noise; not a ranking result or a retraining. No checkpoint files changed."}
    write_json(args.output_dir / "diagnostic.json", results)
    lines = ["# DSMBind mixed-derivative diagnostic", "", results["note"], "",
             f"DSM component: {args.component}; residues per partner <= {args.max_residues}; fixed seed={args.seed}.", "",
             "| Checkpoint | Backend | Parameter | Status |", "|---|---|---|---|"]
    lines += [f"| {r['checkpoint']} | {r['backend']} | {r['parameter']} | {r['status']} |" for r in probe_rows]
    lines += ["", "## Interpretation", "",
              "- Native failure + sensitive, passing reference + matching first-order values supports a CUDA mixed-gradient problem.",
              "- Native/reference energy and force disagreement prevents attributing a difference solely to double backward.",
              "- Inconclusive low signal/unstable FD is not a pass; inspect pretrained and another real pair.",
              "- Reference errors or failed probes mean no training backend has been certified.",
              "- Even a pass on this small crop is a necessary diagnostic, not proof of full training or binding accuracy.",
              "- Inspect diagnostic.json for backend errors and value comparisons, parameter_gradients.csv for all parameter groups."]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Diagnostic saved to {args.output_dir / 'report.md'}", flush=True)
    if any("error" in backend for m in model_reports.values() for backend in (m.get("native", {}), m.get("reference", {}))):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
