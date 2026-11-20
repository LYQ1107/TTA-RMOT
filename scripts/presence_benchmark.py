"""GT-free FP32 batch-size engineering benchmark on the three fixed real frames.

This does not tune a method parameter. Every execution shape must reproduce the
sealed native margins, including sign decisions, before it can be recommended.
It never changes or restarts a running export.
"""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--target', choices=['KITTI-V2', 'Dance', 'SportsMOT'])
    p.add_argument('--summarize', action='store_true')
    a = p.parse_args()
    if a.summarize:
        from flex_uda.presence.runtime import RUN, PUBLIC, IN_DOMAIN
        from flex_tta.runtime.pf_receipts import atomic_json, file_sha
        summaries = []
        for target in IN_DOMAIN:
            path = sorted((RUN / 'batch_benchmark' / target).glob('*/benchmark.json'))[-1]
            record = json.loads(path.read_text())
            if record['GT_open_attempts'] or not all(r['passed'] for r in record['results']):
                raise ValueError('Batch benchmark did not pass all real-frame checks')
            baseline = sum(r['seconds'] for r in record['results'] if r['candidate_batch'] == 8 and r['query_batch'] == 8)
            chosen = sum(r['seconds'] for r in record['results'] if r['candidate_batch'] == 8 and r['query_batch'] == 64)
            summaries.append(dict(target=target, source=str(path), sha256=file_sha(path),
                checked_shapes=len(record['results']), fixed_frames=record['frames'],
                maximum_margin_error=max(r['maximum_margin_error'] for r in record['results']), sign_flips=0,
                decode_seconds_batch8=baseline, decode_seconds_batch64=chosen, decode_ratio=baseline/chosen,
                peak_allocated_bytes=record['peak_allocated_bytes']))
        atomic_json(PUBLIC / 'batch_benchmark.json', dict(status='THREE_REAL_TARGETS_ALL_SHAPES_PASS', records=summaries,
            selected_candidate_batch=8, selected_query_batch=64,
            caveat='one pass on shared GPU, includes shape/kernel warmup; decode ratio is not end-to-end speedup',
            existing_healthy_exports_restarted=False, method_hyperparameters_changed=False))
        print(json.dumps(summaries, indent=2), flush=True)
        return
    if not a.target: p.error('--target is required for a real GPU benchmark')
    import compat.sitecustomize
    import torch
    from flex_uda.presence.runtime import RUN, IN_DOMAIN, target_metadata, sealed_margins, runtime_guard, load_model, episode_from_bank, cache_for
    from flex_uda.presence.features import margins_and_features
    from flex_uda.model.flexhook_uda import prepared_inputs
    from flex_tta.runtime.pf_receipts import atomic_json
    access = runtime_guard(); name = IN_DOMAIN[a.target]; meta = target_metadata(a.target)
    torch.cuda.set_per_process_memory_fraction(min(.9, (16 << 30)/torch.cuda.get_device_properties(0).total_memory))
    torch.manual_seed(0)
    folder = RUN / 'batch_benchmark' / a.target / str(time.time_ns()); folder.mkdir(parents=True)
    device = torch.device('cuda:0')
    model, tokenizer, transform, loaded = load_model(a.target, name, folder / 'model_log', device)
    reference, reference_record = sealed_margins(a.target, name)
    video = sorted(meta['videos'])[0]; bank = meta['videos'][video]
    queries = [q for q in meta['episodes'] if q['video_id'] == video]
    episode = episode_from_bank(video, bank, a.target, transform)
    cache = cache_for(model, tokenizer, episode, device)
    texts = [cache._text(q['query_text']) for q in queries]
    frames = sorted({episode.frames[0], episode.frames[len(episode.frames)//2], episode.frames[-1]})
    positions = {tuple(o): j for j, o in enumerate(bank['observations'])}
    rows = []
    with torch.no_grad():
        for frame in frames:
            candidates = list(episode.current_candidates(frame))
            # Warm the encoder separately, so batch comparisons measure decode
            # work, not cache misses due to which shape happened to run first.
            prepared_inputs(cache, candidates[:1], queries[0]['query_text'], frame)
            for cb in (8, 24):
                for qb in (8, 16, 32, 64):
                    torch.cuda.synchronize(); start = time.perf_counter()
                    maximum_error = 0.; flips = fallback = count = 0
                    for ca in range(0, len(candidates), cb):
                        prepared = prepared_inputs(cache, candidates[ca:ca+cb], queries[0]['query_text'], frame)
                        old = np.asarray([positions[frame, i] for i in prepared['track_ids']])
                        for qa in range(0, len(queries), qb):
                            qt = texts[qa:qa+qb]
                            value, _ = margins_and_features(model, prepared, qt)
                            for j in (value.abs() <= 1e-3).any(dim=0).nonzero().flatten().tolist():
                                isolated, _ = margins_and_features(model, prepared, [qt[j]])
                                value[:, j] = isolated[:, 0]; fallback += 1
                            cpu = value.cpu().numpy()
                            expected = np.stack([reference[q['start']+old] for q in queries[qa:qa+qb]], 1)
                            maximum_error = max(maximum_error, float(np.abs(cpu-expected).max()))
                            flips += int(((cpu > 0) != (expected > 0)).sum()); count += cpu.size
                    torch.cuda.synchronize(); seconds = time.perf_counter()-start
                    record = dict(frame=frame, candidate_batch=cb, query_batch=qb, rows=count,
                        seconds=seconds, rows_per_second=count/seconds,
                        maximum_margin_error=maximum_error, sign_flips=flips, isolated_fallbacks=fallback,
                        passed=maximum_error <= 1e-3 and flips == 0)
                    rows.append(record); print(json.dumps(record), flush=True)
    result = dict(target=a.target, model=name, video=video, frames=frames, results=rows,
        checkpoint=loaded, margin_reference=reference_record, GT_open_attempts=len(access),
        physical_gpu=os.environ.get('PRESENCE_PHYSICAL_GPU'), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        existing_exports_restarted=False, algorithm_hyperparameters_changed=False)
    atomic_json(folder / 'benchmark.json', result)
    if access: raise RuntimeError('GT access during engineering benchmark')


if __name__ == '__main__': main()
