"""Collect the frozen four-policy observation experiment on real GPUs.

The union broker avoids paying several times for an identical frozen input.
Each policy receives only its own requested answers, with its own logical
budget. This is shared evidence preparation, never single-policy deployment
cost. It consumes no old response probabilities, target GT or predictions.
"""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, metadata, generation_guard
from scripts.a100_six_direction import PUBLIC
from scripts.a100_observation_design_prepare import OUT as PREP, POLICIES
from scripts.a100_observation_teacher_pilot import FIELDS, key

OUT = PREP.parent/'full_collection'
STATE = OUT/'supervisor/state.json'
REGISTRATION = PUBLIC/'observation_design_collection_registration.json'
SUCCESS = 'PASS_FOUR_POLICY_OBSERVATION_REAL_TEACHER_COLLECTION_NO_GT'
TEACHER_SHA = '9a13380b4279bc98f4680bd083ff4995fdcfec120cfded459513be90d041a956'


def requests_for(target, video):
    union = {}; own = {}
    for pair, (_, domain) in PAIRS.items():
        if domain != target:
            continue
        for policy in POLICIES:
            source = PREP/pair/video/(policy+'_requests.json')
            rows = json.loads(source.read_text())['requests']
            assert all(r['target'] == target and r['video_id'] == video for r in rows)
            own[pair, policy] = rows
            for row in rows:
                union[key(row)] = {field:row[field] for field in FIELDS}
    return union, own


def prepare():
    begin = time.perf_counter(); guard = generation_guard(); plan = json.loads(REGISTRATION.read_text())
    groups = []; totals = Counter(); policy_counts = Counter()
    # Targets are shared by two source settings; response identity is defined
    # by actual teacher input, not the source model or owning strategy.
    for target in sorted({v[1] for v in PAIRS.values()}):
        for video in sorted(metadata(target)['videos']):
            union, own = requests_for(target, video)
            groups.append(dict(target=target, video=video, unique_requests=len(union)))
            totals['unique_physical_requests'] += len(union)
            for (pair, policy), rows in own.items():
                policy_counts[pair+'/'+policy] += len(rows)
                totals['logical_requests_over_all_settings_and_policies'] += len(rows)
    assert len(groups) == 74
    assert {sum(v for k,v in policy_counts.items() if k.endswith('/'+p)) for p in POLICIES} == {55019}
    assignments = [[] for _ in range(plan['worker_shards'])]; loads = [0]*len(assignments)
    for group in sorted(groups, key=lambda r:(-r['unique_requests'], r['target'], r['video'])):
        selected = min(range(len(loads)), key=lambda i:(loads[i], i))
        assignments[selected].append(group); loads[selected] += group['unique_requests']
    for i, rows in enumerate(assignments):
        rows.sort(key=lambda r:(r['target'], r['video']))
        atomic_json(OUT/f'shard{i}/manifest.json', dict(groups=rows,
            actual_unique_requests=loads[i], no_teacher_answers_used=True))
    assert not guard
    receipt = dict(status='SEALED_FULL_OBSERVATION_COLLECTION_UNION_INPUTS',
        groups=len(groups), **dict(totals), logical_requests_per_policy=55019,
        per_setting_and_policy=dict(policy_counts), worker_shard_unique_loads=loads,
        manual_GT_open_attempts=0, old_teacher_answers_or_predictions_read=0,
        all_new_policy_manifests_unchanged=True, shared_union_is_not_policy_budget_increase=True,
        model_or_method_quality_claim=False, wall_seconds=time.perf_counter()-begin, goal_status='ACTIVE')
    atomic_json(OUT/'preparation.json', receipt)
    (PUBLIC/'observation_design_collection_input_audit.json').write_text(
        json.dumps(receipt, separators=(',', ':'))+'\n')
    return assignments


def worker(shard):
    import torch
    from scripts.active_run import video_spec
    from flex_uda.a100.runtime import teacher
    guard = generation_guard(); start = time.perf_counter(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    root = OUT/f'shard{shard}'; progress_path = root/'progress.json'
    manifest = json.loads((root/'manifest.json').read_text())
    groups = manifest['groups']; expected = manifest['actual_unique_requests']
    if (root/'completion.json').is_file():
        return
    torch.cuda.set_device(0)
    reservation = torch.empty(256 << 20, dtype=torch.float32, device='cuda:0')
    specs = []; metadata_cache = {}; phases = []; completed = 0; logical = 0
    def progress(phase):
        atomic_json(progress_path, dict(phase=phase, completed_videos=completed,
            total_videos=len(groups), completed_unique_answers=logical,
            total_unique_requests=expected, physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES')))
    progress('metadata_and_frozen_teacher_load')
    for target in sorted({g['target'] for g in groups}):
        metadata_cache[target] = metadata(target)
    for group in groups:
        specs.append(video_spec(group['target'], group['video'],
            metadata_cache[group['target']]['videos'][group['video']]))
    fresh_cache = not (root/'teacher.sqlite').exists()
    t = time.perf_counter(); client = teacher(specs, root/'teacher.sqlite')
    torch.cuda.synchronize(); phases.append(dict(phase='teacher_load_and_integrity', seconds=time.perf_counter()-t))
    del reservation; torch.cuda.empty_cache(); resumed = 0; policies = Counter()
    t = time.perf_counter()
    for group in groups:
        target, video = group['target'], group['video']; union, own = requests_for(target, video)
        assert len(union) == group['unique_requests']
        folder = root/'videos'/target/video; checkpoint = folder/'answers_checkpoint.json'
        answers = json.loads(checkpoint.read_text())['answers'] if checkpoint.is_file() else {}
        assert set(answers) <= set(union); resumed += len(answers)
        pending = [(k,r) for k,r in union.items() if k not in answers]
        logical += len(answers); progress('real_frozen_teacher_service')
        for offset in range(0, len(pending), 8):
            batch = pending[offset:offset+8]
            results = client.ask_many([r for _, r in batch])
            for (request_key, request), answer in zip(batch, results):
                assert key(answer) == request_key and 0 <= answer['yes_probability'] <= 1
                assert answer['frozen_weight_sha256'] == TEACHER_SHA
                answers[request_key] = answer
            logical += len(batch)
            # This durable checkpoint bounds lost work without rewriting a
            # large JSON after every tiny teacher microbatch.
            if (offset//8) % 8 == 7 or offset+8 >= len(pending):
                atomic_json(checkpoint, dict(answers=answers, not_to_Git=True,
                    current_session_stats=client.stats, GT_open_attempts=len(guard)))
                progress('real_frozen_teacher_service')
        assert set(answers) == set(union)
        assert all(a['frozen_weight_sha256'] == TEACHER_SHA for a in answers.values())
        for (pair, policy), requests in own.items():
            selected = [dict(answers[key(row)], local_query=row['local_query'],
                local_track=row['local_track'], allocation_kind=row['allocation_kind'],
                order_within_query=row['order_within_query']) for row in requests]
            assert len(selected) == len(requests)
            assert all(key(a) == key(r) for a,r in zip(selected, requests))
            atomic_json(OUT/'policy_answers'/pair/video/(policy+'_answers.json'), dict(
                answers=selected, logical_requested=len(requests),
                only_this_policy_requested_responses=True, GT_open_attempts=len(guard), not_to_Git=True))
            policies[pair+'/'+policy] += len(selected)
        atomic_json(folder/'completion.json', dict(status='SEALED_POLICY_SPECIFIC_OBSERVATION_ANSWERS',
            unique_physical_answers=len(answers), all_owned_answers_exact_requested_subsets=True,
            all_teacher_weights_frozen_SHA=TEACHER_SHA, GT_open_attempts=len(guard)))
        completed += 1; progress('real_frozen_teacher_service')
    torch.cuda.synchronize(); phases.append(dict(phase='teacher_service_including_image_IO', seconds=time.perf_counter()-t))
    t = time.perf_counter(); teacher_receipt = client.close()
    phases.append(dict(phase='teacher_final_weight_integrity', seconds=time.perf_counter()-t))
    assert logical == expected and completed == len(groups) and not guard
    assert teacher_receipt['teacher_weight_sha256'] == TEACHER_SHA
    if fresh_cache and resumed == 0:
        assert teacher_receipt['new_calls'] == expected and teacher_receipt['cache_hits'] == 0
    receipt = dict(status='SEALED_FULL_OBSERVATION_TEACHER_SHARD', shard=shard,
        completed_videos=completed, unique_answers=logical, resumed_answers=resumed,
        fresh_teacher_cache_at_session_start=fresh_cache,
        policies=dict(policies), teacher=teacher_receipt, phases=phases,
        physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'), wall_seconds=time.perf_counter()-start,
        source_features_prepaid=True, shared_union_preparation_not_single_policy_deployment=True,
        manual_GT_open_attempts=0, new_student_encoder_or_head_fits=0, new_quality_predictions=0,
        raw_answers_not_to_Git=True, goal_status='ACTIVE')
    atomic_json(root/'completion.json', receipt); progress('sealed')


def main(prepare_only=False):
    start = time.perf_counter(); plan = json.loads(REGISTRATION.read_text())
    for name, sha in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == sha, name
    STATE.parent.mkdir(parents=True, exist_ok=True)
    assignments = prepare()
    if prepare_only:
        return
    pending = list(range(len(assignments))); active = {}; finished = []
    failure = None; last = 0.
    def state(status):
        arms = {}
        for gpu, (proc, _, shard) in active.items():
            p = OUT/f'shard{shard}/progress.json'
            arms[str(gpu)] = json.loads(p.read_text()) if p.is_file() else dict(phase='initializing', shard=shard)
        atomic_json(STATE, dict(status=status, pid=os.getpid(), completed=len(finished),
            finished_shards=finished, active={str(g):dict(pid=p.pid, shard=s) for g,(p,_,s) in active.items()},
            pending=pending, arms=arms, failure=failure, wall_seconds=time.perf_counter()-start, goal_status='ACTIVE'))
    state('AWAITING_FREE_GPU_0_TO_3_FULL_OBSERVATION_COLLECTION')
    while pending or active:
        changed = False
        for gpu, (proc, handle, shard) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            handle.close(); del active[gpu]
            if code or not (OUT/f'shard{shard}/completion.json').is_file():
                log = OUT/f'shard{shard}/worker.log'
                failure = dict(shard=shard, physical_GPU=gpu, exit_code=code, last_log=log.read_text()[-7000:])
            else:
                finished.append(shard)
            changed = True
        candidates = [g for g in (0,1,2,3) if g not in active]
        if pending and candidates and not failure:
            text = subprocess.check_output(['nvidia-smi', '-i', ','.join(map(str, candidates)),
                '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
            memory = {int(r.split(',')[0]):int(r.split(',')[1]) for r in text.strip().splitlines()}
            for gpu in candidates:
                if not pending or memory[gpu] >= 512:
                    continue
                shard = pending.pop(0); log = OUT/f'shard{shard}/worker.log'
                if (OUT/f'shard{shard}/completion.json').is_file():
                    finished.append(shard); changed = True; continue
                handle = log.open('a', buffering=1)
                proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--shard', str(shard)],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
                active[gpu] = proc, handle, shard; changed = True
        if changed or time.perf_counter()-last > 45:
            state('RUNNING_FOUR_POLICY_REAL_TEACHER_COLLECTION_NO_GT' if not failure else 'PRESERVING_HEALTHY_TEACHER_SHARDS_AFTER_FAILURE')
            last = time.perf_counter()
        if failure and not active:
            raise RuntimeError(json.dumps(failure))
        if pending or active:
            time.sleep(5)
    rows = [json.loads((OUT/f'shard{i}/completion.json').read_text()) for i in sorted(finished)]
    inputs = json.loads((OUT/'preparation.json').read_text()); counts = Counter()
    for row in rows:
        counts.update(row['policies'])
    assert dict(counts) == inputs['per_setting_and_policy']
    assert sum(r['unique_answers'] for r in rows) == inputs['unique_physical_requests']
    report = dict(status=SUCCESS, input_audit=inputs, shards=rows,
        teacher_new_calls=sum(r['teacher']['new_calls'] for r in rows),
        teacher_cache_hits=sum(r['teacher']['cache_hits'] for r in rows),
        source_features_prepaid=True, shared_preparation_not_policy_deployment_cost=True,
        controller_parallel_makespan_seconds=time.perf_counter()-start,
        manual_target_GT_open_attempts=0, new_quality_predictions=0,
        policy_quality_not_yet_evaluated=True, goal_status='ACTIVE', research_goal_complete=False)
    atomic_json(OUT/'completion.json', report)
    (PUBLIC/'observation_design_collection_receipt.json').write_text(json.dumps(report, separators=(',', ':'))+'\n')
    state(SUCCESS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--shard', type=int)
    parser.add_argument('--prepare-only', action='store_true'); args = parser.parse_args()
    if args.shard is not None:
        worker(args.shard)
    else:
        try:
            main(args.prepare_only)
        except Exception as error:
            previous = json.loads(STATE.read_text()) if STATE.is_file() else {}
            previous.update(status='FAILED_FULL_OBSERVATION_TEACHER_COLLECTION', error=repr(error), goal_status='ACTIVE')
            atomic_json(STATE, previous)
            raise
