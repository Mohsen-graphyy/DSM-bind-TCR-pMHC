"""Profile a few real DSMBind updates in a separate process; never overwrite training checkpoints."""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "DSMBind"))

import torch

from bindenergy.apps.tcr_pmhc.train import (
    architecture_from_checkpoint, check_head_gradients, model_state_from_checkpoint,
    set_seed, torch_load,
)
from bindenergy.apps.tcr_pmhc.gradient_check import recurrence_backend
from bindenergy.apps.tcr_pmhc.benchmark_io import digest_file, write_json
from bindenergy.data.tcr_pmhc import EmbeddingStore, TCRpMHCPairDataset, make_model_batch


@contextlib.contextmanager
def instrument_recurrences(model, shapes):
    """Add profiler scopes without replacing any recurrence math or detaching its graph."""
    saved = []
    def wrap(original, name, cell):
        def call(*args, **kwargs):
            U = args[0]
            shapes.setdefault(name, {"U_shape": list(U.shape), "device": str(U.device),
                "bidirectional": bool(cell.bidirectional), "hidden_size": cell.hidden_size})
            with torch.profiler.record_function("SRU_reference/" + name):
                return original(*args, **kwargs)
        return call
    try:
        for name, cell in model.named_modules():
            if hasattr(cell, 'apply_recurrence') and hasattr(cell, 'weight_c'):
                saved.append((cell, cell.apply_recurrence))
                cell.apply_recurrence = wrap(cell.apply_recurrence, name, cell)
        yield
    finally:
        for cell, original in saved:
            cell.apply_recurrence = original


def choose_indices(checkpoint, count, dataset_size, seed):
    """Use the next saved pairs when available; otherwise use a seeded permutation."""
    state = checkpoint.get('training_state', {}) if isinstance(checkpoint, dict) else {}
    indices = state.get('indices')
    if indices is not None:
        remaining = list(indices[int(state.get('next_position', 0)):])
    else:
        remaining = []
    if len(remaining) < count:
        additional = list(range(dataset_size))
        random.Random(seed).shuffle(additional)
        remaining.extend(additional)
    selected = remaining[:count]
    if len(selected) < count or any(i < 0 or i >= dataset_size for i in selected):
        raise ValueError('Not enough valid pair indices for this profile')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/profile_reference_b16_01'))
    parser.add_argument('--warmup-batches', type=int, default=1)
    parser.add_argument('--timing-batches', type=int, default=2)
    parser.add_argument('--trace', action='store_true', help='Optional large Chrome trace; operator tables are always saved')
    args = parser.parse_args()
    if args.warmup_batches < 0 or args.timing_batches < 1:
        parser.error('Need non-negative warmup and at least one timing batch')
    if not torch.cuda.is_available():
        raise RuntimeError('This diagnostic needs the real CUDA training environment')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Use a new output-dir; previous profiles are preserved')
    config = json.loads((args.run_dir / 'training_config.json').read_text())
    if config.get('recurrence_backend') != 'reference':
        raise ValueError('Expected a reference training run')
    device = torch.device(config.get('device', 'cuda'))
    if device.type != 'cuda':
        raise ValueError('Expected a CUDA training run')
    torch.set_num_threads(int(config.get('num_threads', 8)))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(int(config['seed']))
    source = args.run_dir / 'last.pt'
    if not source.exists():
        source = Path(config['init_checkpoint'])
    checkpoint = torch_load(source, torch.device('cpu'))
    architecture = architecture_from_checkpoint(checkpoint)
    from bindenergy.models.energy import AllAtomEnergyModel
    model = AllAtomEnergyModel(architecture).to(device)
    model.load_state_dict(model_state_from_checkpoint(checkpoint), strict=True)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['learning_rate']))
    if isinstance(checkpoint, dict) and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    batch_size = int(config['batch_size'])
    shapes, audit = {}, []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        dataset = TCRpMHCPairDataset(Path(config['train_db']), int(config.get('chain_cache_size', 128)))
        stack.callback(dataset.close)
        embeddings = EmbeddingStore(Path(config['embeddings']), int(config.get('embedding_cache_size', 128)))
        stack.callback(embeddings.close)
        if int(embeddings.metadata.get('dimension', architecture.esm_size)) != architecture.esm_size:
            raise ValueError('Embedding/checkpoint dimension mismatch')
        count = (args.warmup_batches + args.timing_batches + 1) * batch_size
        indices = choose_indices(checkpoint, count, len(dataset), int(config['seed']))
        iterator = iter(dataset.iter_batches(batch_size, indices))
        stack.enter_context(recurrence_backend(model, 'reference', required_device=device, device_audit=audit))
        stack.enter_context(instrument_recurrences(model, shapes))

        def batch_update():
            times = {}
            def phase(name, function):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                with torch.profiler.record_function(name):
                    value = function()
                torch.cuda.synchronize(device)
                times[name] = time.perf_counter() - start
                return value
            phase('zero_grad', lambda: optimizer.zero_grad(set_to_none=True))
            batch = phase('sqlite_and_pair_loading', lambda: next(iterator))
            binder, target = phase('esm_lookup_featurization_H2D', lambda: make_model_batch(
                batch, embeddings, device, swap_partners=(random.random() < .5)))
            loss = phase('model_forward_including_coordinate_gradient', lambda: model(
                binder, target, use_sidechain=not config['no_sidechain']))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Non-finite diagnostic loss')
            phase('loss_backward_mixed_derivatives', loss.backward)
            def guard_and_clip():
                check_head_gradients(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config['clip_norm']), error_if_nonfinite=True)
            phase('gradient_checks_and_clipping', guard_and_clip)
            phase('Adam_step_in_memory_only', optimizer.step)
            return {'pairs': len(batch), 'loss': float(loss.detach()), 'seconds': times,
                    'total_seconds': sum(times.values())}

        for i in range(args.warmup_batches):
            print('warmup', i + 1, batch_update()['total_seconds'], flush=True)
        torch.cuda.reset_peak_memory_stats(device)
        timings = [batch_update() for _ in range(args.timing_batches)]
        # Persist unprofiled timings first: tracing can be expensive for this recurrence.
        report = {'checkpoint': str(source), 'checkpoint_sha256': digest_file(source),
                  'gpu': torch.cuda.get_device_name(device), 'torch': torch.__version__,
                  'cuda': torch.version.cuda, 'cpu_threads': torch.get_num_threads(),
                  'batch_size': batch_size, 'recurrence_shapes': shapes, 'device_audit': audit,
                  'timing_batches': timings, 'peak_allocated_MiB': torch.cuda.max_memory_allocated(device) / 2**20,
                  'note': 'Separate-process diagnostic. Updates only its in-memory model. No training checkpoint is written. Phase boundaries synchronize CUDA. Does not time validation/checkpoint I/O.'}
        phase_names = list(timings[0]['seconds'])
        report['mean_seconds_by_phase'] = {name: statistics.mean(t['seconds'][name] for t in timings) for name in phase_names}
        elapsed = sum(t['total_seconds'] for t in timings)
        report['pairs_per_second'] = sum(t['pairs'] for t in timings) / elapsed
        report['estimated_training_only_epoch_hours'] = len(dataset) / report['pairs_per_second'] / 3600
        write_json(args.output_dir / 'timings.json', report)
        print(json.dumps(report['mean_seconds_by_phase'], indent=2), flush=True)
        print('Profiling one extra batch; tracing adds overhead and is excluded from timing estimates.', flush=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=False, profile_memory=False, with_stack=False) as prof:
            profiled = batch_update()
        averages = prof.key_averages()
        (args.output_dir / 'operators_cpu.txt').write_text(averages.table(sort_by='self_cpu_time_total', row_limit=40), encoding='utf-8')
        (args.output_dir / 'operators_cuda.txt').write_text(averages.table(sort_by='self_device_time_total', row_limit=40), encoding='utf-8')
        if args.trace:
            prof.export_chrome_trace(str(args.output_dir / 'trace.json'))
        report['profiled_batch_seconds_with_tracing_overhead'] = profiled['total_seconds']
        write_json(args.output_dir / 'timings.json', report)
    print('Saved profile to', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
