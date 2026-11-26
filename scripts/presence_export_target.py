"""Frozen complete-candidate PCD export. Independent of all evaluation GT."""
from argparse import ArgumentParser
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha


def export_video(model, tokenizer, transform, device, meta, video, target, name, output, reference, smoke, *, source=False):
    import torch
    from flex_uda.presence.features import margins_and_features, sentence_vector
    from flex_uda.presence.runtime import episode_from_bank, cache_for
    from flex_uda.model.flexhook_uda import prepared_inputs
    from flex_uda.evaluation import query_margins
    folder = output / ('smoke' if smoke else 'videos') / video
    seal_path = folder / 'seal.json'
    if seal_path.exists():
        old = json.loads(seal_path.read_text())
        if any(file_sha(folder / p) != sha for p, sha in old['files_sha256'].items()):
            raise ValueError('Changed completed feature shard; preserve and investigate')
        return old
    folder.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    bank = meta['videos'][video]
    episode = episode_from_bank(video, bank, target, transform)
    cache = cache_for(model, tokenizer, episode, device)
    queries = [q for q in meta['episodes'] if q['video_id'] == video]
    texts = [cache._text(q['query_text']) for q in queries]
    np.save(folder / 'sentence.npy', torch.stack([sentence_vector(t) for t in texts]).cpu().numpy())
    frames = (bank['image_frames'][::(3 if target == 'KITTI-V2' else 10)] if source else
              sorted(int(p.stem) for p in episode.image_root.glob('*' + episode.extension) if p.stem.isdigit()))
    check_frames = frames if source else episode.frames
    checks = sorted({check_frames[0], check_frames[len(check_frames)//2], check_frames[-1]})
    reference_checks = set(checks) if video == sorted(meta['videos'])[0] else set()
    selected_frames = checks if smoke else frames
    selected_set = set(selected_frames)
    observations = [(t, i) for t, i in bank['observations'] if t in selected_set]
    positions = {tuple(obs): j for j, obs in enumerate(observations)}
    original = {tuple(obs): j for j, obs in enumerate(bank['observations'])}
    n, nq = len(observations), len(queries)
    pcd = np.lib.format.open_memmap(folder / 'pcd.npy', mode='w+', dtype='float16', shape=(nq*n, 768))
    margin = np.lib.format.open_memmap(folder / 'margin.npy', mode='w+', dtype='float32', shape=(nq*n,))
    box = np.zeros((n, 4), dtype='float32')
    np.save(folder / 'observations.npy', np.asarray(observations, dtype='int32'))
    index = np.lib.format.open_memmap(folder / 'index.npy', mode='w+', dtype='int32', shape=(nq*n, 3))
    for j in range(nq):
        index[j*n:(j+1)*n, 0] = j
        index[j*n:(j+1)*n, 1:] = observations
    index.flush()
    parity, maximum_error, sign_flips, extra_tolerance_rows = [], 0., 0, 0
    fixed_maximum_error = fixed_sign_flips = 0
    with torch.no_grad():
        for t in selected_frames:
            candidates = list(episode.current_candidates(t)) if t in episode.observations_by_frame else []
            for ca in range(0, len(candidates), 8):
                chosen = candidates[ca:ca+8]
                prepared = prepared_inputs(cache, chosen, queries[0]['query_text'], t)
                local = np.asarray([positions[t, i] for i in prepared['track_ids']])
                old = np.asarray([original[t, i] for i in prepared['track_ids']])
                box[local] = prepared['boxes'][:, -1].cpu().numpy()
                for qa in range(0, nq, 64):
                    qt = texts[qa:qa+64]
                    value, feature = margins_and_features(model, prepared, qt)
                    # Same pre-existing near-zero query-isolation policy as sealed banks.
                    for j in (value.abs() <= 1e-3).any(dim=0).nonzero().flatten().tolist():
                        isolated, fi = margins_and_features(model, prepared, [qt[j]])
                        value[:, j] = isolated[:, 0]; feature[:, j] = fi[:, 0]
                    cpu = value.cpu().numpy()
                    expected = (np.stack([reference[q['start'] + old] for q in queries[qa:qa+64]], 1)
                                if reference is not None else cpu)
                    error = float(np.abs(cpu - expected).max())
                    flips = int(((cpu > 0) != (expected > 0)).sum())
                    maximum_error = max(maximum_error, error); sign_flips += flips
                    if t in reference_checks:
                        fixed_maximum_error = max(fixed_maximum_error, error)
                        fixed_sign_flips += flips
                    extra_tolerance_rows += int((np.abs(cpu-expected) > 1e-3).sum())
                    # The user gate is the pre-fixed THREE real frames, not a
                    # new per-row acceptance criterion. Historical V2/KITTI
                    # scores were already documented to differ by0.0010823
                    # under FP32 frozen-encoder replay (old smoke_V2-best.json).
                    # Keep all extra errors visible; never rewrite old scores.
                    # Native / head identity inputs use the original sealed
                    # bank (FeatureUnits), including its near-zero decisions.
                    # Extra replay differences remain audit, not extra gates.
                    if t in reference_checks and (error > 1e-3 or flips):
                        raise RuntimeError(f'Sealed-margin parity failed: {target}/{name}/{video}/{t}: {error}, {flips}')
                    if t in checks:
                        base = query_margins(model, prepared, qt, batched=True)
                        err = float((base - value).abs().max())
                        parity.append(dict(frame=t, candidate_start=ca, query_start=qa,
                            sealed_max_error=error, sealed_sign_flips=flips, query_margins_error=err))
                        if err > 1e-3:
                            raise RuntimeError('Final-PCD hook changed query_margins')
                    if not torch.isfinite(feature).all():
                        raise FloatingPointError('Nonfinite frozen PCD feature')
                    for j in range(len(qt)):
                        dest = (qa+j)*n + local
                        pcd[dest] = feature[:, j].cpu().numpy().astype('float16')
                        margin[dest] = cpu[:, j]
    pcd.flush(); margin.flush()
    np.save(folder / 'boxes.npy', box)
    if source:
        from flex_uda.presence.source import candidate_labels
        labels = candidate_labels(bank, queries, observations)
        np.save(folder / 'labels.npy', labels)
    atomic_json(folder / 'metadata.json', dict(video_id=video, target=target, model=name,
        queries=queries, observations=n, frames=selected_frames, width=episode.width, height=episode.height,
        row_order='query-major; index.npy=(local_query_index, frame, original_track_id)',
        frame_rule='KITTI image-index 0 based; raw prediction MOT remains 1 based' if target == 'KITTI-V2' else 'original 1-based MOT/image frame'))
    files = ['pcd.npy', 'margin.npy', 'sentence.npy', 'boxes.npy', 'index.npy', 'observations.npy', 'metadata.json']
    if source: files.append('labels.npy')
    receipt = dict(status='SMOKE_PASS' if smoke else 'FEATURES_SEALED', video_id=video, target=target, model=name,
        path=str(folder), candidate_query_rows=n*nq, units=len(selected_frames)*nq,
        candidate_batch=8, query_batch=64,
        maximum_margin_error=maximum_error, sign_flips=sign_flips, parity=parity,
        fixed_reference_tolerance_frames=sorted(reference_checks), extra_audit_rows_over_1e_3=extra_tolerance_rows,
        fixed_reference_max_error=fixed_maximum_error, fixed_reference_sign_flips=fixed_sign_flips,
        identity_input_policy='sealed complete official margins; replay margin.npy retained as audit',
        old_KITTI_replay_discrepancy_reference='results/multidomain_generality/smoke_V2-best.json',
        files_sha256={p: file_sha(folder / p) for p in files},
        wall_seconds=time.perf_counter()-begin, cache=cache.stats)
    atomic_json(seal_path, receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('files_sha256', 'parity', 'cache')}), flush=True)
    return receipt


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--target', required=True, choices=['KITTI-V2', 'Dance', 'SportsMOT'])
    p.add_argument('--model', required=True, choices=['V2-best', 'Dance-best', 'LaMOT-best'])
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    import compat.sitecustomize
    import torch
    from flex_uda.presence.runtime import RUN, OLD, TARGET_KEY, load_model, target_metadata, sealed_margins, runtime_guard
    from flex_uda.presence.resources import admitted_device
    admitted_device()
    access = runtime_guard([OLD / 'runtime' / TARGET_KEY[a.target]])
    folder = RUN / 'features' / a.target / a.model
    folder.mkdir(parents=True, exist_ok=True)
    # A source-head validation bank may be prioritized on one of the SAME
    # registered physical GPUs while the original queue is still exporting.
    # The queued copy waits here and reuses the completed bank, never writes it
    # concurrently or restarts the healthy priority worker.
    inherited = os.environ.get('PRESENCE_EXPORT_LOCK_FD')
    lock_handle = os.fdopen(int(inherited), 'a') if inherited else (folder / '.export.lock').open('a')
    fcntl.flock(lock_handle, fcntl.LOCK_EX)
    completion = folder / ('smoke_receipt.json' if a.smoke else 'export_receipt.json')
    if completion.exists():
        receipt = json.loads(completion.read_text())
        for video in receipt['videos']:
            for name, sha in video['files_sha256'].items():
                if file_sha(Path(video['path']) / name) != sha: raise ValueError('Changed completed feature bank')
        print(json.dumps(dict(status='ALREADY_SEALED_NO_GPU_RELOAD', target=a.target, model=a.model)), flush=True)
        return
    # Serialize new frozen-export contexts on one physical card. Existing
    # healthy pre-lease workers remain untouched; memory admission still waits
    # for enough real space rather than assuming a lock reserves VRAM.
    resource_root = RUN / 'resources'; resource_root.mkdir(exist_ok=True)
    gpu_lock = (resource_root / (os.environ['CUDA_VISIBLE_DEVICES']+'.export.lock')).open('a')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX)
    admitted_device(20480)
    started = time.perf_counter()
    device = torch.device('cuda:0')
    torch.manual_seed(0)
    torch.cuda.set_per_process_memory_fraction(min(.9, (16384 << 20) / torch.cuda.get_device_properties(0).total_memory))
    model, tokenizer, transform, loaded = load_model(a.target, a.model, folder / 'model_log', device)
    versions = {n: p._version for n, p in model.named_parameters()}
    reference, reference_record = sealed_margins(a.target, a.model)
    meta = target_metadata(a.target)
    videos = sorted(meta['videos'])[:1] if a.smoke else sorted(meta['videos'])
    receipts = [export_video(model, tokenizer, transform, device, meta, v, a.target, a.model, folder, reference, a.smoke) for v in videos]
    torch.cuda.synchronize()
    if access or any(p._version != versions[n] for n, p in model.named_parameters()):
        raise RuntimeError('GT access or frozen parameter mutation')
    result = dict(status='SMOKE_PASS' if a.smoke else 'TARGET_FEATURES_COMPLETE', target=a.target, model=a.model,
        load=loaded, margin_reference=reference_record, GT_open_attempts=len(access), videos=receipts,
        rows=sum(r['candidate_query_rows'] for r in receipts), wall_seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), physical_gpu=os.environ.get('PRESENCE_PHYSICAL_GPU'))
    atomic_json(folder / ('smoke_receipt.json' if a.smoke else 'export_receipt.json'), result)
    print(json.dumps(dict(status=result['status'], target=a.target, model=a.model, rows=result['rows'],
                          seconds=result['wall_seconds'], peak=result['peak_allocated_bytes'])), flush=True)


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        print(json.dumps(dict(status='TARGET_EXPORT_FAILED', error=repr(exc))), flush=True)
        raise
