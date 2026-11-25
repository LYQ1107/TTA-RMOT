"""Stronger ordinary frame KD with causal teacher-only capacity continuation.

Keep every passed prefix byte-equivalent and repair only the five remaining
failed prefixes. Selection uses deployed float32 teacher KL, never RMOT GT.
All 162 own raw/count outputs precede this supplement's isolated evaluation.
"""
import argparse
import concurrent.futures
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
from flex_uda.a100.mechanism import PAIRS, metadata, history, generation_guard
from flex_uda.active.relational_features import geometry, extra_features, moments, frame_supervision
from flex_uda.active.relational_transport import horizontal_direction
from scripts.a100_six_direction import OUT as SIX, PUBLIC, SEEDS, source
from scripts.a100_relation_transport import coarse_features
from scripts.a100_object_reference_fit_complete import OUT as PREDECESSOR
from scripts.a100_relation_transport_prepare import OUT as RELATION_PREPARED

OUT = SIX/'object_reference/capacity_supplement'
STATE = OUT/'supervisor/state.json'
SUCCESS = 'PASS_ORDINARY_CAPACITY_SIX_SETTINGS_RESEARCH_GOAL_ACTIVE'
PRIMARY = 'REFERENCE_FULL'
BASELINES = ('CAPACITY_FRAME', 'CAPACITY_HYBRID')
CONTRASTS = {}
for _b in BASELINES:
    CONTRASTS['FULL_vs_'+_b] = (PRIMARY, _b)
    CONTRASTS['RANK_AT_FULL_COUNT_vs_'+_b] = (PRIMARY+'__AT_'+PRIMARY+'_COUNT', _b+'__AT_'+PRIMARY+'_COUNT')
    CONTRASTS['RANK_AT_BASE_COUNT_vs_'+_b] = (PRIMARY+'__AT_'+_b+'_COUNT', _b+'__AT_'+_b+'_COUNT')
REGISTRATION = PUBLIC/'object_reference_capacity_registration_v1.json'


def conditions():
    return [dict(pair=p, seed=s, folder=str(OUT/p/f'seed{s}')) for s in SEEDS for p in PAIRS]


def predict(checkpoint, raw, coords, columns, directions):
    import torch
    from torch import nn
    width = checkpoint['model']['0.weight'].shape[0]
    model = nn.Sequential(nn.Linear(1087, width), nn.GELU(), nn.Linear(width, 1)).cuda().eval()
    model.load_state_dict(checkpoint['model']); score = np.empty((len(raw), len(coords)), float)
    mean, std = checkpoint['mean'], checkpoint['std']
    with torch.no_grad():
        for qi, direction in enumerate(directions):
            for start in range(0, len(coords), 4096):
                stop = min(start+4096, len(coords)); ids = columns[start:stop]
                x = np.concatenate((raw[qi, ids], extra_features(coords[start:stop], direction)), axis=1)
                z = model(torch.as_tensor((x-mean)/std, dtype=torch.float32, device='cuda:0')).squeeze(-1)
                score[qi, start:stop] = torch.sigmoid(z).cpu().numpy()
    return score


def run(pair, seed):
    import torch
    from flex_uda.active.fit_capacity import continue_capacity
    from flex_uda.presence.predictions import seal_mask
    from scripts.a100_visual_classical_strict_prefix import plain, immediate_predictions
    guard = generation_guard(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    root = OUT/pair/f'seed{seed}'; original = PREDECESSOR/pair/f'seed{seed}'
    if (root/'completion.json').is_file():
        return
    previous = json.loads((original/'completion.json').read_text())
    needs_repair = any(not r['fits']['FRAME_COMPLETE']['teacher_fit_within_0_005'] for r in previous['receipts'])
    plan = json.loads(REGISTRATION.read_text()); source_model, target_domain = PAIRS[pair]
    m = metadata(target_domain); root.mkdir(parents=True, exist_ok=True)
    if not (root/PRIMARY).exists():
        (root/PRIMARY).symlink_to(original/PRIMARY, target_is_directory=True)
    begin = time.perf_counter(); total = np.zeros(1087); square = np.zeros(1087); row_count = 0
    bx, by, receipts = [], [], []; next_video = 0; state_path = root/'stream.pt'
    if state_path.is_file():
        saved = torch.load(state_path, map_location='cpu', weights_only=False)
        total, square, row_count, bx, by, receipts, next_video = [saved[k] for k in
            ('total', 'square', 'row_count', 'bx', 'by', 'receipts', 'next_video')]
    videos = sorted(m['videos'].items())
    for vi, (video, bank) in enumerate(videos):
        if vi < next_video:
            continue
        started = time.perf_counter(); queries = [q for q in m['episodes'] if q['video_id'] == video]
        old_receipt = previous['receipts'][vi]; assert old_receipt['video'] == video
        old_fit = old_receipt['fits']['FRAME_COMPLETE']; fit = dict(old_fit)
        with np.load(original/'videos'/video/'frame_scores.npz') as a:
            tracks, direct, hybrid, primary = [a[k] for k in
                ('tracks', 'FIT_COMPLETE_FRAME', 'FIT_COMPLETE_HYBRID', PRIMARY)]
        calls = reused_probe = 0
        if needs_repair:
            obs = np.asarray(bank['observations']); coords, width, height = geometry(bank, target_domain)
            directions = np.asarray([horizontal_direction(q['query_text']) for q in queries], int)
            with np.load(source(pair)/'source'/video/'sparse_inputs.npz') as a:
                pcd, colours, prior, original_tracks = [a[k] for k in ('pcd', 'colours', 'prior', 'tracks')]
            assert np.array_equal(tracks, original_tracks)
            lookup = {int(t):i for i,t in enumerate(tracks)}
            columns = np.asarray([lookup[int(t)] for _, t in obs])
            raw = coarse_features(pcd, colours, prior, tracks, obs, [q['query_text'] for q in queries])
            records = json.loads((history(pair)/'videos'/video/'questions.json').read_text())
            cells, teacher, teacher_coords, evidence = frame_supervision(records, queries, tracks, width, height)
            qidx, tidx, _ = cells.T
            bx.append(np.concatenate((raw[qidx, tidx], extra_features(teacher_coords, directions[qidx])), axis=1).astype('float32'))
            by.append(teacher); t, s, n = moments(raw, coords, columns, directions)
            total += t; square += s; row_count += n
            mean = total/row_count; std = np.maximum(np.sqrt(np.maximum(square/row_count-mean**2, 0)), 1e-3)
            if not old_fit['teacher_fit_within_0_005']:
                x, y = np.concatenate(bx), np.concatenate(by)
                checkpoint = None
                for cache in plan['causal_probe_caches']:
                    if (pair, seed, video) == (cache['pair'], cache['seed'], cache['video']):
                        assert vi+1 == cache['prefix_videos'] and len(x) == cache['teacher_examples']
                        checkpoint = torch.load(cache['head_path'], map_location='cpu', weights_only=False)
                        reused_probe = 1
                if checkpoint is None:
                    checkpoint = torch.load(original/'videos'/video/'fit_complete_head.pt', map_location='cpu', weights_only=False)
                    assert np.array_equal(checkpoint['mean'], mean) and np.array_equal(checkpoint['std'], std)
                    checkpoint, optimized = continue_capacity(x, y, checkpoint, seed=seed,
                        widths=tuple(plan['widths']), blocks=plan['blocks_per_width'])
                    calls = 1
                else:
                    optimized = checkpoint['receipt']
                assert np.array_equal(checkpoint['mean'], mean) and np.array_equal(checkpoint['std'], std)
                assert optimized['teacher_examples'] == len(x)
                fit = dict(old_fit, **optimized, reused_exact_passed_prefix_output=False,
                    original_64_unit_teacher_KL=old_fit['teacher_KL'])
                local = root/'videos'/video; local.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, local/'capacity_head.pt')
                direct = predict(checkpoint, raw, coords, columns, directions)
                # The same previously registered eligibility and DS fallback.
                # No new query-dependent or GT-dependent deployment choice.
                beta = json.loads((RELATION_PREPARED/pair/video/'prefix_fit.json').read_text())['beta']
                active = (directions != 0) & (beta > 0)
                hybrid[active] = direct[active]
        if old_fit['teacher_fit_within_0_005']:
            fit.update(reused_exact_passed_prefix_output=True, hidden_width=64)
        scores = {PRIMARY:primary, 'CAPACITY_FRAME':direct, 'CAPACITY_HYBRID':hybrid}
        local = root/'videos'/video; local.mkdir(parents=True, exist_ok=True)
        for name in BASELINES:
            bits = plain(scores[name], False); destination = root/name/'videos'/video
            destination.mkdir(parents=True, exist_ok=True); np.save(destination/'mask.npy', bits)
            immediate_predictions(root/name, type('Context', (), dict(bank=bank, queries=queries))(), bits)
        np.savez_compressed(local/'frame_scores.npz', tracks=tracks, **scores)
        receipt = dict(video=video, index=vi, fits={'FRAME_CAPACITY':fit},
            teacher_evidence=old_receipt['teacher_evidence'], capacity_calls=calls,
            reused_same_prefix_probe=reused_probe, passed_original_reused=bool(old_fit['teacher_fit_within_0_005']),
            reference_head_gate_failures=old_receipt['reference_head_gate_failures'],
            GT_open_attempts=len(guard), no_future_teacher_or_head=True,
            wall_seconds=time.perf_counter()-started, quality_replay_not_deployment_cost=True)
        atomic_json(local/'receipt.json', receipt); receipts.append(receipt); next_video = vi+1
        saved = dict(total=total, square=square, row_count=row_count, bx=bx, by=by,
            receipts=receipts, next_video=next_video)
        temporary = state_path.with_suffix('.tmp'); torch.save(saved, temporary); temporary.replace(state_path)
        print(json.dumps(dict(event='CAPACITY_ORDINARY_VIDEO', pair=pair, seed=seed,
            completed=next_video, total=len(videos))), flush=True)
    methods = []
    for name in BASELINES:
        mask = np.zeros(m['candidate_query_rows'], bool)
        for video, _ in videos:
            bits = np.load(root/name/'videos'/video/'mask.npy')
            for qi, q in enumerate(q for q in m['episodes'] if q['video_id'] == video):
                mask[q['start']:q['stop']] = bits[qi]
        seal = seal_mask(root/name, m, mask, target=target_domain, model=source_model,
            method=name, GT_open_attempts=len(guard), extra=dict(seed=seed,
                teacher_only_capacity_continuation=True, no_GT_or_future_head_selection=True))
        methods.append(dict(method=name, rows=seal['prediction_rows']))
    assert not guard
    atomic_json(root/'completion.json', dict(status='SEALED_CAPACITY_ORDINARY_RAW_NO_GT',
        pair=pair, seed=seed, methods=methods, receipts=receipts, GT_open_attempts=0,
        actual_capacity_calls=sum(r['capacity_calls'] for r in receipts),
        actual_reused_prefix_probes=sum(r['reused_same_prefix_probe'] for r in receipts),
        new_teacher_encoder_calls=0, wall_seconds=time.perf_counter()-begin, goal_status='ACTIVE'))


def counts(record):
    import scripts.a100_relation_transport as common
    common.PRIMARY = PRIMARY; common.BASELINES = BASELINES
    return common.counts(record)


def analyze():
    from scripts.a100_visual_classical_analyze import read_stats
    from scripts.a100_six_direction_analyze import comparison
    from scripts.a100_visual_classical_sparse_observation_analyze import fit_audit
    from scripts.analyze_pf_video_uncertainty import pooled_hota
    assert json.loads((OUT/'all_predictions_sealed_before_GT.json').read_text())['own_methods']==162
    data={};statistics={};summary={};error=0.
    for record in conditions():
        root=Path(record['folder']);slug=record['pair']+f"_seed{record['seed']}"
        raw=json.loads((root/'completion.json').read_text());count=json.loads((root/'countmatched/completion.json').read_text())
        paths={r['method']:root/r['method'] for r in raw['methods']}
        paths.update({r['method']:root/'countmatched'/r['method'] for r in count['methods']})
        # CountSelf has byte-identical predictions; this reads only our own
        # completed GT result, never writes an active predecessor result path.
        paths[PRIMARY]=root/'countmatched'/(PRIMARY+'__AT_'+PRIMARY+'_COUNT')
        data[slug]={name:read_stats(path) for name,path in paths.items()}
        for name,row in data[slug].items():
            num,den=row['parent_stats'];error=max(error,abs(float(pooled_hota((num,den),np.ones(len(num))))-row['HOTA']))
            statistics[slug+'/'+name]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],numerator=num.tolist(),denominator=den.tolist())
        summary[slug]=dict(raw_HOTA={n:data[slug][n]['HOTA'] for n in (PRIMARY,)+BASELINES},
            teacher_fit_audit=fit_audit(raw['receipts']),CountSelf_changed_query_files=count['CountSelf_changed_query_files'],
            actual_capacity_calls=raw['actual_capacity_calls'],actual_reused_prefix_probes=raw['actual_reused_prefix_probes'],
            source_reference_gate_failures=sum(r['reference_head_gate_failures'].get('REFERENCE',False) for r in raw['receipts']))
    assert error<1e-8,error
    repeated={p:dict(mean_HOTA={n:float(np.mean([data[p+f'_seed{s}'][n]['HOTA'] for s in SEEDS])) for n in (PRIMARY,)+BASELINES},
        comparisons={name:comparison([data[p+f'_seed{s}'][a] for s in SEEDS],[data[p+f'_seed{s}'][b] for s in SEEDS])
            for name,(a,b) in CONTRASTS.items()}) for p in PAIRS}
    result=dict(status='OBJECT_REFERENCE_CAPACITY_SIX_SETTINGS_ANALYZED',primary=PRIMARY,conditions=summary,repeated=repeated,
        HOTA_parent_sum_max_error=error,own162_outputs_sealed_before_GT=True,predecessor162_also_sealed_before_our_GT=True,
        primary_stats_from_byte_identical_CountSelf=True,all_data_development=True,new_teacher_encoder_calls=0,
        fitting_gate_not_relaxed=True,quality_replay_not_deployment_cost=True,goal_status='ACTIVE',research_goal_complete=False)
    atomic_json(OUT/'analysis.json',result)
    (PUBLIC/'object_reference_capacity_analysis.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    (PUBLIC/'object_reference_capacity_parent_statistics.json').write_text(json.dumps(statistics,separators=(',',':'))+'\n')
    print(json.dumps(dict(event='CAPACITY_ORDINARY_ANALYSIS',repeated=repeated,goal_status='ACTIVE')),flush=True)


def main():
    plan=json.loads(REGISTRATION.read_text())
    for name,sha in plan['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    pending=conditions();done=[];active={};failure=None;started=time.perf_counter();last=0.
    def state(status):atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=len(done),finished=done,
        active={str(g):job for g,(_,_,job) in active.items()},pending=pending,failure=failure,
        wall_seconds=time.perf_counter()-started,goal_status='ACTIVE'))
    state('AWAITING_PREDECESSOR_GPU_STAGE_RELEASE_NO_GT')
    while True:
        previous=json.loads((PREDECESSOR/'supervisor/state.json').read_text())
        if previous.get('failure') or previous['status'].startswith('FAILED_'):raise RuntimeError('Resolve predecessor failure first')
        if previous.get('completed')==18 and not previous.get('active') and not previous.get('pending'):break
        time.sleep(5)
    while pending or active:
        changed=False
        for gpu,(proc,handle,job) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            handle.close();del active[gpu]
            if code or not (Path(job['folder'])/'completion.json').is_file():failure=dict(job=job,exit_code=code,last_log=Path(job['log']).read_text()[-6000:])
            else:done.append(job)
            changed=True
        candidates=[g for g in (0,1,2,3) if g not in active]
        if pending and candidates and not failure:
            output=subprocess.check_output(['nvidia-smi','-i',','.join(map(str,candidates)),'--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            memory={int(r.split(',')[0]):int(r.split(',')[1]) for r in output.strip().splitlines()}
            for gpu in candidates:
                if not pending or memory[gpu]>=512:continue
                job=pending.pop(0)
                if (Path(job['folder'])/'completion.json').is_file():done.append(dict(job,reused_sealed=True));changed=True;continue
                log=OUT/'supervisor'/(job['pair']+'_seed'+str(job['seed'])+'.log');handle=log.open('a',buffering=1)
                proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--pair',job['pair'],'--seed',str(job['seed'])],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
                active[gpu]=proc,handle,dict(job,pid=proc.pid,log=str(log),physical_GPU=gpu);changed=True
        if changed or time.perf_counter()-last>45:
            state('RUNNING_ORDINARY_FIT_COMPLETION_NO_GT' if not failure else 'PRESERVING_ACTIVE_FIT_COMPLETION_AFTER_FAILURE');last=time.perf_counter()
        if failure and not active:raise RuntimeError(json.dumps(failure))
        if pending or active:time.sleep(3)
    state('ALL18_CAPACITY_RAW_SEALED_GENERATING_COUNTS_NO_GT')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:roots=list(pool.map(counts,conditions()))
    tasks=[]
    for record,count_root in zip(conditions(),roots):
        for folder in (Path(record['folder']),Path(count_root)):
            row=json.loads((folder/'completion.json').read_text());assert row['GT_open_attempts']==0
            assert not any(row.get('CountSelf_changed_query_files',{}).values())
            tasks.extend((str(folder/r['method']),record['pair']) for r in row['methods'])
    assert len(tasks)==162
    state('ALL162_OWN_OUTPUTS_SEALED_WAITING_PREDECESSOR162_SEAL_NO_GT')
    barrier=PREDECESSOR/'all_predictions_sealed_before_GT.json'
    while not barrier.is_file():time.sleep(5)
    assert json.loads(barrier.read_text())['own_methods']==162
    atomic_json(OUT/'all_predictions_sealed_before_GT.json',dict(own_methods=162,borrowed_primary_raw_methods=18,
        predecessor_all162_sealed=True,GT_open_attempts=0))
    state('ALL162_CAPACITY_OUTPUTS_SEALED_OFFICIAL_EVALUATING')
    from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:results=dict(pool.map(evaluate_one,tasks))
    atomic_json(OUT/'official.json',dict(methods=162,results=results,goal_status='ACTIVE'))
    analyze();state(SUCCESS);print(SUCCESS,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pair',choices=tuple(PAIRS));parser.add_argument('--seed',type=int);args=parser.parse_args()
    if args.pair:
        import fcntl
        root=OUT/args.pair/f'seed{args.seed}';root.mkdir(parents=True,exist_ok=True)
        with (root/'condition.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX);run(args.pair,args.seed)
    else:
        try:main()
        except Exception as error:
            old=json.loads(STATE.read_text()) if STATE.is_file() else {}
            old.update(status='FAILED_CAPACITY_ORDINARY_REQUIRES_ROOT_CAUSE',error=repr(error),goal_status='ACTIVE');atomic_json(STATE,old);raise
