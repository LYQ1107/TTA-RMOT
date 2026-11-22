"""Fresh A100 three-anchor PCD, retaining the observations before averaging.

All ordinary and proposed learners in a later study receive the same states.
This is shared representation preparation, not an innovation-quality result
or a single-method deployment benchmark. No teacher QA or target GT is read.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, generation_guard, metadata
from scripts.a100_six_direction import OUT as SIX, PUBLIC, source
from scripts.a100_anchor_observability import samples_for

OUT = SIX/'anchor_PCD_A100_20261123'
STATE = OUT/'supervisor/state.json'
REGISTRATION = PUBLIC/'anchor_PCD_collection_registration_v1.json'
SUCCESS = 'PASS_FRESH_A100_THREE_ANCHOR_PCD_SIX_SETTINGS_NO_GT'


def prepare():
    guard = generation_guard(); cases = []
    for pair, (model, target) in PAIRS.items():
        counts = dict(videos=0, queries=0, provided_tracks=0, anchor_observations=0,
            distinct_frames_within_videos=0, stored_feature_elements=0)
        for video, bank in sorted(metadata(target)['videos'].items()):
            obs = np.asarray(bank['observations'], dtype=int); tracks, rows = samples_for(obs)
            nq = sum(q['video_id'] == video for q in metadata(target)['episodes'])
            counts['videos'] += 1; counts['queries'] += nq; counts['provided_tracks'] += len(tracks)
            counts['anchor_observations'] += sum(map(len, rows))
            counts['distinct_frames_within_videos'] += len(np.unique(obs[np.concatenate(rows), 0]))
            counts['stored_feature_elements'] += nq*len(tracks)*3*768
        cases.append(dict(pair=pair, source_model=model, target=target, **counts))
    assert not guard and sum(r['videos'] for r in cases) == 148
    result = dict(status='SEALED_FRESH_ANCHOR_PCD_INPUT_PLAN', settings=cases,
        identical_three_source_observation_indices=True, official_candidates_and_TrackIDs_unchanged=True,
        query_batch=8, candidate_batch=8, original_temporal_context_preserved=True,
        no_teacher_answers_or_predictions_or_target_GT=True, manual_GT_open_attempts=0)
    atomic_json(OUT/'preparation.json', result)
    (PUBLIC/'anchor_PCD_collection_input_audit.json').write_text(json.dumps(result, separators=(',', ':'))+'\n')
    return result


def run(pair):
    import torch
    from flex_uda.a100.runtime import load_student, source_head
    from flex_uda.presence.runtime import episode_from_bank
    from flex_tta.runtime.frozen_features import FrozenFeatureCache
    from flex_uda.active.v6_frames import FrameCacheV6
    from scripts.a100_visual_classical_cold_slice import SparsePCD
    guard = generation_guard(); torch.set_num_threads(4); torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    folder = OUT/pair; folder.mkdir(parents=True, exist_ok=True)
    if (folder/'completion.json').is_file():
        return
    start = time.perf_counter(); device = torch.device('cuda:0'); torch.cuda.set_device(device)
    model_name, target = PAIRS[pair]; ledger_path = folder/'phase_ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.is_file() else dict(attempts=[])
    attempt = dict(started_unix=time.time(), physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'), phases=[])
    ledger['attempts'].append(attempt); atomic_json(ledger_path, ledger)
    def phase(name, stamp, video=None):
        attempt['phases'].append(dict(phase=name, seconds=time.perf_counter()-stamp, video=video))
        attempt['worker_wall_seconds_to_checkpoint'] = time.perf_counter()-start
        atomic_json(ledger_path, ledger)
    stamp = time.perf_counter()
    model, tokenizer, transform, loaded = load_student(folder/'student_load', device,
        model_name=model_name, target=target)
    phase('source_student_load_and_integrity', stamp)
    assert not any(p.requires_grad for p in model.parameters())
    versions = tuple(p._version for p in model.parameters()); receipts = []
    for video, bank in sorted(metadata(target)['videos'].items()):
        local = folder/'videos'/video; receipt_path = local/'receipt.json'
        if receipt_path.is_file():
            receipts.append(json.loads(receipt_path.read_text())); continue
        stamp = time.perf_counter(); local.mkdir(parents=True, exist_ok=True)
        queries = [q for q in metadata(target)['episodes'] if q['video_id'] == video]
        obs = np.asarray(bank['observations'], dtype=int); sink = SparsePCD(obs, len(queries))
        frames = np.unique(obs[np.concatenate(sink.samples), 0]); ids = np.flatnonzero(np.isin(obs[:, 0], frames))
        scoring_bank = dict(bank, observations=obs[ids].tolist(), lines=[bank['lines'][i] for i in ids])
        def receive(qa, local_ids, features):
            sink(qa, ids[local_ids], features)
        episode = episode_from_bank(video, bank, target, transform)
        cache = FrozenFeatureCache(model, episode, tokenizer, device, max_bytes=512 << 20)
        fc = FrameCacheV6(model, cache, scoring_bank, queries, frames, source_head(model_name),
            local/'frame_cache', store_pcd=False, pcd_sink=receive)
        fc.score(model_step=0); mean = sink.mean(); fc.close(); torch.cuda.synchronize()
        assert np.isfinite(sink.values).all() and tuple(p._version for p in model.parameters()) == versions
        assert list(map(int, sink.tracks)) == fc.tracks
        phase('fresh_student_anchor_forward_and_cache_close', stamp, video)
        stamp = time.perf_counter()
        sample_rows = np.full((len(sink.tracks), 3), -1, dtype=int)
        sample_frames = sample_rows.copy(); present = np.zeros(sample_rows.shape, dtype=bool)
        for ti, rows in enumerate(sink.samples):
            sample_rows[ti, :len(rows)] = rows; sample_frames[ti, :len(rows)] = obs[rows, 0]
            present[ti, :len(rows)] = True
        np.savez_compressed(local/'anchor_pcd.npz', features=sink.values.astype('float16'),
            seen=sink.seen, present=present, tracks=sink.tracks, sample_rows=sample_rows,
            sample_frames=sample_frames, mean=mean, query_ids=[q['expression_id'] for q in queries])
        with np.load(source(pair)/'source'/video/'sparse_inputs.npz') as previous:
            assert np.array_equal(previous['tracks'], sink.tracks) and previous['pcd'].shape == mean.shape
            error = np.abs(previous['pcd'].astype(float)-mean)
        receipt = dict(pair=pair, video=video, queries=len(queries), tracks=len(sink.tracks),
            sampled_observations=int(present.sum()), selected_frames=len(frames),
            new_source_candidate_query_rows=sum(r['candidate_query_rows'] for r in fc.score_records),
            PCD_shape=list(sink.values.shape), stored_dtype='float16; identical to sink FP16-roundtrip values',
            previous_A100_mean_max_error=float(error.max()), previous_A100_mean_absolute_error_sum=float(error.sum()),
            previous_A100_mean_elements=int(error.size), previous_A100_exact_mean_elements=int((error == 0).sum()),
            previous_A100_mean_exact=bool((error == 0).all()),
            later_comparison_rule='Both ordinary/proposed heads retrain on these same fresh states/means; never pair old heads with new features.',
            fixed32_metadata_created_by_shared_cache_but_never_used=True,
            slow_refresh_or_LoRA_updates=0, source_parameters_frozen=True, manual_GT_open_attempts=len(guard))
        phase('save_anchor_states_and_compare_old_mean', stamp, video)
        atomic_json(receipt_path, receipt); receipts.append(receipt)
        atomic_json(folder/'progress.json', dict(completed_videos=len(receipts), total_videos=len(metadata(target)['videos']), video=video))
        del fc, cache, episode, sink; torch.cuda.empty_cache()
    assert not guard
    attempt['worker_wall_seconds'] = time.perf_counter()-start; atomic_json(ledger_path, ledger)
    atomic_json(folder/'completion.json', dict(status='SEALED_FRESH_ANCHOR_PCD_NO_GT', pair=pair,
        source=loaded, videos=receipts, attempts=ledger['attempts'], frozen_source=True,
        teacher_answers_or_predictions_or_GT_read=0, raw_states_not_to_Git=True,
        quality_prep_not_single_method_deployment=True, goal_status='ACTIVE'))


def main(prepare_only=False):
    plan = json.loads(REGISTRATION.read_text())
    for name, sha in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == sha, name
    STATE.parent.mkdir(parents=True, exist_ok=True); prepare()
    if prepare_only:
        return
    pending = [p for p in PAIRS if not (OUT/p/'completion.json').is_file()]; active = {}
    def state(status, failure=None):
        arms = {}
        for pid, row in active.items():
            p = OUT/row['pair']/'progress.json'; progress = json.loads(p.read_text()) if p.is_file() else {}
            arms[row['pair']] = dict(physical_GPU=row['GPU'], **progress)
        atomic_json(STATE, dict(status=status, pid=os.getpid(), completed=6-len(pending)-len(active),
            pending=pending, active=active, arms=arms, failure=failure, goal_status='ACTIVE'))
    state('RUNNING_FRESH_SOURCE_ANCHOR_PCD_NO_GT'); processes = {}
    while pending or active:
        for pid, proc in list(processes.items()):
            if proc.poll() is None:
                continue
            row = active.pop(pid); del processes[pid]
            if proc.returncode or not (OUT/row['pair']/'completion.json').is_file():
                state('FAILED_FRESH_ANCHOR_PCD', dict(pair=row['pair'], exit_code=proc.returncode, log=row['log']))
                raise RuntimeError(Path(row['log']).read_text()[-7000:])
        if pending and len(active) < 3:
            raw = subprocess.check_output(['nvidia-smi','-i','0,1,2,3','--query-gpu=index,memory.used','--format=csv,noheader,nounits'], text=True)
            memory = {int(r.split(',')[0]):int(r.split(',')[1]) for r in raw.strip().splitlines()}
            used = {r['GPU'] for r in active.values()}
            free = [g for g in (0,1,2,3) if g not in used and memory[g] < 512]
            for gpu in free[:min(3-len(active),len(pending))]:
                pair = pending.pop(0); log = STATE.parent/(pair+'_worker.log')
                with log.open('a', buffering=1) as stream:
                    proc = subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--pair',pair], cwd=ROOT,
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
                active[proc.pid] = dict(pair=pair, GPU=gpu, log=str(log)); processes[proc.pid] = proc
        state('RUNNING_FRESH_SOURCE_ANCHOR_PCD_NO_GT')
        if pending or active:
            time.sleep(15)
    completed = {p:json.loads((OUT/p/'completion.json').read_text()) for p in PAIRS}
    rows = [v for d in completed.values() for v in d['videos']]; assert len(rows) == 148
    report = dict(status=SUCCESS, settings={p:dict(source=d['source'], videos=len(d['videos']), attempts=d['attempts']) for p,d in completed.items()},
        setting_videos=len(rows), expected_model_features='Same three source anchors, query/candidate batch8, original temporal context; retain states and mean.',
        previous_mean_max_error=max(r['previous_A100_mean_max_error'] for r in rows),
        previous_mean_fraction_exact=sum(r['previous_A100_exact_mean_elements'] for r in rows)/sum(r['previous_A100_mean_elements'] for r in rows),
        previous_mean_exact_videos=sum(r['previous_A100_mean_exact'] for r in rows),
        ordinary_and_proposed_inputs_identical_in_future_study=True,
        teacher_answers_or_predictions_or_GT_read=0, new_teacher_calls_or_adaptation_fits=0,
        cost_scope='Shared neural input preparation with real phase costs, not a single-method cold deployment.',
        no_claim_of_RMOT_gain_or_new_innovation=True, raw_feature_weights_not_to_Git=True, goal_status='ACTIVE')
    atomic_json(OUT/'completion.json', report)
    (PUBLIC/'anchor_PCD_collection_receipt.json').write_text(json.dumps(report, separators=(',', ':'))+'\n')
    state(SUCCESS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--pair', choices=tuple(PAIRS)); parser.add_argument('--prepare-only', action='store_true'); args = parser.parse_args()
    if args.pair:
        folder = OUT/args.pair; folder.mkdir(parents=True, exist_ok=True)
        with (folder/'condition.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            run(args.pair)
    else:
        try:
            main(args.prepare_only)
        except Exception as error:
            previous = json.loads(STATE.read_text()) if STATE.is_file() else {}
            previous.update(status='FAILED_FRESH_ANCHOR_PCD', error=repr(error), pid=os.getpid(), goal_status='ACTIVE')
            atomic_json(STATE, previous); raise
