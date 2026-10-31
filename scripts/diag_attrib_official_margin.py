#!/usr/bin/env python3
"""Recover genuinely missing official Dance logits without changing old files.

One GPU, no training/GT. Preserve known logits exactly. If full-bank native
selection differs from the historical official result, stop attribution.
"""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diag_attrib_audit import (RUN, PUBLIC, OFFICIAL, TRACKS, DATA,
    deny_gt, require_prereg, runtime_rows, prediction, expected_rows)
from flex_tta.runtime.pf_receipts import atomic_json, file_sha

CHECKPOINT = Path('/data3/user/TTA-RMOT-assets/flexhook/SOTA_ckpts/refer-dance-best.pth')
CHECKPOINT_SHA = 'fbb5df7339a1201f34300c105df81e893a5edbb8ef939b7f6eaf8780348eac6c'


def admit_gpu():
    gpu_text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,memory.total',
        '--format=csv,noheader,nounits'], text=True)
    process_text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True)
    counts = Counter(line.split(',')[0].strip() for line in process_text.splitlines() if line.strip())
    devices = []
    for line in gpu_text.splitlines():
        index, uuid, free, total = [x.strip() for x in line.split(',')]
        devices.append(dict(index=int(index), uuid=uuid, free_MiB=int(free), total_MiB=int(total),
                            process_count=counts[uuid]))
    eligible = [d for d in devices if d['free_MiB'] >= 12288]
    if not eligible:
        raise RuntimeError('no single GPU meets fixed 12GiB admission budget')
    chosen = min(eligible, key=lambda d: (d['process_count'], -d['free_MiB'], d['index']))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(chosen['index'])
    atomic_json(RUN / 'official_gpu_admission.json', dict(devices=devices, chosen=chosen,
        maximum_physical_GPUs=1, admission_MiB=12288, allocator_limit_MiB=8192,
        selection='fewest processes, then most free memory, then numeric index',
        raw_GPU_query=gpu_text, raw_process_query=process_text, other_processes_stopped=0))
    return chosen


def run():
    gate = require_prereg()
    attempts = deny_gt()
    started = time.perf_counter()
    chosen = admit_gpu()
    import compat.sitecustomize  # noqa: F401; canonical local BPE
    import torch
    from transformers import RobertaTokenizerFast
    from flex_tta.runtime.dance_episode import DanceEpisode
    from flex_tta.runtime.frozen_features import FrozenFeatureCache
    from flex_tta.dance_io import load_tracker_lines
    from scripts.run_flexhook_dance_pcd_tta import _load_model, _model_margin
    if file_sha(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('wrong official Dance checkpoint')
    source_path = OFFICIAL / 'result.json'
    source_sha = file_sha(source_path)
    if source_sha != 'de8ff46428818cbd1ee0dd2b6ea1a7692eae2b7e0133c521c1470b67b58da110':
        raise ValueError('historical official logits changed')
    source = json.loads(source_path.read_text())
    missing = json.loads((RUN / 'official_missing_observations.json').read_text())
    by_video, queries = defaultdict(list), defaultdict(list)
    for row in missing:
        by_video[row['video']].append(row)
    for row in runtime_rows():
        queries[row['video_id']].append(row)
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction((8192 << 20) / torch.cuda.get_device_properties(device).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gpu_started = time.perf_counter()
    _, model, _ = _load_model(ROOT / 'vendor/FlexHook', CHECKPOINT, DATA, TRACKS,
                             RUN / 'official_model_log', device, 8, strict_checkpoint=True)
    tokenizer = RobertaTokenizerFast.from_pretrained(str(ROOT / 'vendor/FlexHook/pretrained/roberta-base'))
    if tokenizer.encode('person', add_special_tokens=False) != [5970]:
        raise ValueError('canonical tokenizer failed')
    parity = []
    new_positives = []
    all_supplements = []
    original_states = {n: p._version for n, p in model.named_parameters()}
    for video in sorted(by_video):
        episode = DanceEpisode(DATA, TRACKS, video)
        cache = FrozenFeatureCache(model, episode, tokenizer, device, max_bytes=256 << 20)
        # Original released forward and cached decode are compared to each
        # other AND actual historical logits on known rows before recovery.
        if len(parity) < 8:
            for t in episode.frames:
                candidates = [c for c in episode.current_candidates(t)
                              if str(t) in source[video].get(str(c[0]), {})][:4]
                if not candidates:
                    continue
                row = queries[video][0]
                with torch.no_grad():
                    logits, ids = cache.logits(candidates, row['query_text'], t)
                    direct, direct_ids = _model_margin(model, episode, candidates, row['query_text'], tokenizer, device)
                margins = logits[:, 1] - logits[:, 0]
                old = torch.tensor([source[video][str(i)][str(t)][row['query_text']][1] -
                    source[video][str(i)][str(t)][row['query_text']][0] for i in ids], device=device)
                err = float((margins-direct).abs().max())
                historical_err = float((margins-old).abs().max())
                flips = int(((margins > 0) != (old > 0)).sum())
                parity.append(dict(video=video, frame=t, candidates=len(ids),
                    cache_vs_original_forward_max_error=err, cache_vs_historical_max_error=historical_err,
                    historical_selection_mismatches=flips))
                if ids != direct_ids or err > 1e-3 or historical_err > 1e-3 or flips:
                    atomic_json(RUN / 'official_parity_failure.json', parity)
                    raise RuntimeError('official model/old-logit parity failed; no attribution allowed')
                if len(parity) >= 8:
                    break
        output = RUN / 'official_supplement' / (video + '.json')
        wanted = {(r['frame'], r['track_id']) for r in by_video[video]}
        if output.exists():
            record = json.loads(output.read_text())
            if record['checkpoint_sha256'] != CHECKPOINT_SHA or record['original_score_sha256'] != source_sha:
                raise ValueError('recovery input changed')
            entries = record['entries']
        else:
            entries = []
            for t in sorted({r['frame'] for r in by_video[video]}):
                candidates = list(episode.current_candidates(t, {i for f, i in wanted if f == t}))
                for row in queries[video]:
                    for a in range(0, len(candidates), 8):
                        with torch.no_grad():
                            logits, ids = cache.logits(candidates[a:a+8], row['query_text'], t)
                        for i, pair in zip(ids, logits.cpu().tolist()):
                            entries.append(dict(video=video, expression=row['expression_id'],
                                query_text=row['query_text'], frame=t, track_id=i,
                                logits=pair, margin=pair[1]-pair[0]))
            atomic_json(output, dict(checkpoint_sha256=CHECKPOINT_SHA, original_score_sha256=source_sha,
                entries=entries, optimizer_steps=0, target_GT_open_attempts=attempts, cache_stats=cache.stats))
        if len(entries) != 17*len(wanted) or len({(e['expression'], e['frame'], e['track_id']) for e in entries}) != len(entries):
            raise ValueError('incomplete/duplicated missing-score recovery')
        positive = [e for e in entries if e['margin'] > 0]
        new_positives.extend(positive)
        all_supplements.append(dict(path=str(output), sha256=file_sha(output), entries=len(entries),
                                    positive_entries=len(positive)))
        print(json.dumps(dict(event='OFFICIAL_MISSING_MARGIN_RECOVERED', video=video,
                              entries=len(entries), positive_entries=len(positive))), flush=True)
        del cache
    if any(p._version != original_states[n] for n, p in model.named_parameters()):
        raise ValueError('frozen official model parameters changed')
    torch.cuda.synchronize()
    gpu_seconds = time.perf_counter()-gpu_started
    # This is a no-GT reproduction self-check, not a new official evaluation.
    added_by_episode = Counter((e['video'], e['expression']) for e in new_positives)
    status = 'STOP_OFFICIAL_COMPLETEBANK_CANNOT_REPRODUCE_HISTORICAL_NATIVE' if new_positives else 'OFFICIAL_COMPLETEBANK_NATIVE_REPRODUCTION_PASS'
    report = dict(status=status, prereg=gate, original_score_sha256=source_sha,
        checkpoint=str(CHECKPOINT), checkpoint_sha256=CHECKPOINT_SHA,
        missing_MOT_observations=len(missing), recovered_candidate_query_scores=sum(r['entries'] for r in all_supplements),
        added_positive_rows=len(new_positives), changed_episodes=len(added_by_episode),
        historical_prediction_rows=254118, completebank_native_prediction_rows=254118+len(new_positives),
        official_historical_HOTA=32.134, completebank_HOTA=None,
        completebank_HOTA_not_evaluated='stop immediately on row reproduction failure; no GT opened',
        new_positive_examples=new_positives[:20],
        added_rows_per_episode=[dict(video=v, expression=q, added_rows=n) for (v,q),n in sorted(added_by_episode.items())],
        supplements=all_supplements, parity=parity, physical_GPU=chosen['index'],
        precision='FP32', tokenizer_person=[5970], optimizer_steps=0, trainable_parameters=0,
        target_GT_open_attempts=attempts, original_files_written=0,
        GPU_worker_seconds=gpu_seconds, GPU_worker_hours=gpu_seconds/3600,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), total_wall_seconds=time.perf_counter()-started)
    atomic_json(RUN / 'official_reproduction.json', report)
    atomic_json(PUBLIC / 'official_reproduction.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('supplements','new_positive_examples','added_rows_per_episode')}, indent=2), flush=True)
    if new_positives:
        raise SystemExit(3)


if __name__ == '__main__':
    run()
