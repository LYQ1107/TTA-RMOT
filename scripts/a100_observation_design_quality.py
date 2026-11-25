"""Isolate observation-policy effects using the same ordinary KD learner.

This is an I1 component benchmark, not the completed three-component method.
The coarse learner does not establish a state-identification interaction.
All policy answers are real, frozen, budget-matched and broker-isolated.
"""
import argparse
from collections import Counter, defaultdict
import concurrent.futures
import hashlib
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
from flex_uda.a100.mechanism import PAIRS, metadata, generation_guard
from scripts.a100_six_direction import OUT as SIX, PUBLIC, SEEDS, source
from scripts.a100_observation_design_prepare import POLICIES
from scripts.a100_observation_design_collect import OUT as COLLECTED, SUCCESS as COLLECTION_SUCCESS, TEACHER_SHA

OUT = SIX/'active_observation_design/quality'
STATE = OUT/'supervisor/state.json'
REGISTRATION = PUBLIC/'observation_policy_registration_v1.json'
SUCCESS = 'PASS_OBSERVATION_POLICY_QUALITY_SIX_SETTINGS_GOAL_ACTIVE'
METHODS = tuple(p+'_'+d for p in POLICIES for d in ('FINAL', 'MEAN', 'TEACHER_FILL'))
PRIMARY = 'DESIGN_MEAN'
BASELINES = tuple(m for m in METHODS if m != PRIMARY)
CONTRASTS = {'PRIMARY_vs_'+b:(PRIMARY, b) for b in BASELINES}
for _decoder in ('FINAL', 'MEAN', 'TEACHER_FILL'):
    for _other in ('PRIOR', 'UNIFORM'):
        CONTRASTS['COVERAGE_vs_'+_other+'_'+_decoder] = ('COVERAGE_ONLY_'+_decoder, _other+'_'+_decoder)
    for _other in ('COVERAGE_ONLY', 'PRIOR', 'UNIFORM'):
        CONTRASTS['DESIGN_vs_'+_other+'_'+_decoder] = ('DESIGN_'+_decoder, _other+'_'+_decoder)
for _b in BASELINES:
    CONTRASTS['RANK_AT_PRIMARY_COUNT_vs_'+_b] = (PRIMARY+'__AT_'+PRIMARY+'_COUNT', _b+'__AT_'+PRIMARY+'_COUNT')
    CONTRASTS['RANK_AT_BASE_COUNT_vs_'+_b] = (PRIMARY+'__AT_'+_b+'_COUNT', _b+'__AT_'+_b+'_COUNT')


def conditions():
    return [dict(pair=p, seed=s, folder=str(OUT/p/f'seed{s}')) for s in SEEDS for p in PAIRS]


def labels(pair, video, policy, queries, tracks):
    document = json.loads((COLLECTED/'policy_answers'/pair/video/(policy+'_answers.json')).read_text())
    assert document['GT_open_attempts'] == 0 and document['only_this_policy_requested_responses']
    unique = {}; grouped = defaultdict(list)
    for row in document['answers']:
        qi, ti = row['local_query'], row['local_track']
        assert queries[qi]['expression_id'] == row['query_id'] and int(tracks[ti]) == row['track_id']
        assert row['frozen_weight_sha256'] == TEACHER_SHA
        # Exact request repetition does not become another pseudo-example.
        if row['request_id'] in unique:
            assert unique[row['request_id']]['yes_probability'] == row['yes_probability']
            continue
        unique[row['request_id']] = row
        grouped[qi, ti].append(float(row['yes_probability']))
    cells = np.asarray(sorted(grouped), int)
    target = np.asarray([np.mean(grouped[tuple(cell)]) for cell in cells], float)
    assert len(document['answers']) == document['logical_requested']
    assert np.isfinite(target).all() and ((target >= 0) & (target <= 1)).all()
    # This ordinary track-level target intentionally projects paid frame
    # answers to a coarse query/track mean. State-aware learners remain a
    # separate experiment; this benchmark must not claim to validate them.
    return cells, target, dict(logical_requests=document['logical_requested'],
        unique_request_inputs=len(unique), query_track_cells=len(cells),
        target='Raw mean of own actual teacher frames; one query-track training cell',
        confidence='1+3*abs(2*unclipped_track_mean-1); not multiplied by repeated frame count',
        effective_views=dict(Counter(r['effective_view'] for r in document['answers'])),
        CLIP_fallbacks=sum(bool(r['clip_fallback']) for r in document['answers']),
        teacher_weight_SHA=TEACHER_SHA, borrowed_policy_answers=0)


def predict(checkpoint, current):
    import torch
    from torch import nn
    width = checkpoint['model']['0.weight'].shape[0]
    model = nn.Sequential(nn.Linear(768, width), nn.GELU(), nn.Linear(width, 1)).cuda().eval()
    model.load_state_dict(checkpoint['model'])
    result = []
    with torch.no_grad():
        for start in range(0, len(current), 4096):
            x = (current[start:start+4096]-checkpoint['mean'])/checkpoint['std']
            result.append(torch.sigmoid(model(torch.as_tensor(x, dtype=torch.float32,
                device='cuda:0')).squeeze(-1)).cpu().numpy())
    return np.concatenate(result)


def run(pair, seed):
    import torch
    from flex_uda.active.confidence_teacher import fit_head
    from flex_uda.active.fit_capacity import continue_capacity
    from flex_uda.presence.predictions import seal_mask
    from scripts.a100_visual_classical_strict_prefix import plain, immediate_predictions
    guard = generation_guard(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    root = OUT/pair/f'seed{seed}'; root.mkdir(parents=True, exist_ok=True)
    if (root/'completion.json').is_file():
        return
    plan = json.loads(REGISTRATION.read_text()); source_model, target_domain = PAIRS[pair]
    m = metadata(target_domain); begin = time.perf_counter(); next_video = 0
    total = np.zeros(768); square = np.zeros(768); rows = 0
    bx = {p:[] for p in POLICIES}; by = {p:[] for p in POLICIES}; receipts = []
    state_path = root/'stream.pt'
    if state_path.is_file():
        saved = torch.load(state_path, map_location='cpu', weights_only=False)
        total, square, rows, bx, by, receipts, next_video = [saved[k] for k in
            ('total', 'square', 'rows', 'bx', 'by', 'receipts', 'next_video')]
    for vi, (video, bank) in enumerate(sorted(m['videos'].items())):
        if vi < next_video:
            continue
        started = time.perf_counter(); queries = [q for q in m['episodes'] if q['video_id'] == video]
        with np.load(source(pair)/'source'/video/'sparse_inputs.npz') as a:
            pcd, prior, tracks = [a[k] for k in ('pcd', 'prior', 'tracks')]
        current = pcd.reshape(-1, 768).astype('float32')
        total += current.sum(0, dtype=float); square += np.square(current.astype(float)).sum(0)
        rows += len(current); mean = total/rows
        std = np.maximum(np.sqrt(np.maximum(square/rows-mean**2, 0)), 1e-3)
        scores, fits, evidence, cache = {}, {}, {}, {}
        actual_fits = capacity_calls = 0
        for policy in POLICIES:
            cells, raw_target, ev = labels(pair, video, policy, queries, tracks)
            bx[policy].append(pcd[cells[:, 0], cells[:, 1]]); by[policy].append(raw_target)
            x = np.concatenate(bx[policy]); y = np.concatenate(by[policy])
            weight = 1.+3.*np.abs(2.*y-1.); digest = hashlib.sha256()
            for value in (x, y, weight):
                digest.update(np.ascontiguousarray(value).tobytes())
            fingerprint = digest.hexdigest()
            if fingerprint in cache:
                owner, early, final, fit = cache[fingerprint]
                fit = dict(fit, identical_prefix_learner_reused_from_policy=owner)
            else:
                output, fit, checkpoint = fit_head(x, np.zeros(len(x)), np.clip(y, .02, .98),
                    current, np.zeros(len(current)), mean, std, sample_weight=weight, soft=True, seed=seed)
                actual_fits += 1; early = output['400']; final = output['CONVERGED']
                if not fit['teacher_fit_within_0_005']:
                    checkpoint, optimized = continue_capacity(x, y, checkpoint, seed=seed,
                        widths=tuple(plan['capacity_widths']), blocks=plan['capacity_blocks_per_width'])
                    fit = dict(fit, **optimized, original_64_unit_teacher_KL=fit['teacher_KL'])
                    capacity_calls += 1; final = predict(checkpoint, current)
                torch.save(checkpoint, root/(policy+'_latest_head.pt'))
                cache[fingerprint] = policy, early, final, fit
            early = early.reshape(prior.shape); final = final.reshape(prior.shape)
            ensemble = .5*(early+final)
            teacher_fill = final.copy(); teacher_fill[cells[:, 0], cells[:, 1]] = raw_target
            scores.update({policy+'_FINAL':final, policy+'_MEAN':ensemble,
                policy+'_TEACHER_FILL':teacher_fill})
            fits[policy] = dict(fit, mean_decoder='Fixed0.5*(step400+adequate_final); gate applies to final component',
                no_GT_checkpoint_selection=True, source_residual=False)
            evidence[policy] = ev
        obs = np.asarray(bank['observations']); lookup = {int(t):i for i,t in enumerate(tracks)}
        columns = np.asarray([lookup[int(t)] for _,t in obs])
        for name in METHODS:
            bits = plain(scores[name][:, columns], False); destination = root/name/'videos'/video
            destination.mkdir(parents=True, exist_ok=True); np.save(destination/'mask.npy', bits)
            immediate_predictions(root/name, type('Context', (), dict(bank=bank, queries=queries))(), bits)
        local = root/'videos'/video; local.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(local/'coarse_scores.npz', tracks=tracks, **scores)
        receipt = dict(video=video, index=vi, fits=fits, teacher_evidence=evidence,
            actual_head_fits=actual_fits, conditional_capacity_calls=capacity_calls,
            exact_identical_prefix_fit_reuse=4-actual_fits, GT_open_attempts=len(guard),
            wall_seconds=time.perf_counter()-started, quality_replay_not_deployment_cost=True)
        atomic_json(local/'receipt.json', receipt); receipts.append(receipt); next_video = vi+1
        saved = dict(total=total, square=square, rows=rows, bx=bx, by=by,
            receipts=receipts, next_video=next_video)
        temporary = state_path.with_suffix('.tmp'); torch.save(saved, temporary); temporary.replace(state_path)
        print(json.dumps(dict(event='OBSERVATION_POLICY_VIDEO', pair=pair, seed=seed,
            completed=next_video, total=len(m['videos']))), flush=True)
    methods = []
    for name in METHODS:
        mask = np.zeros(m['candidate_query_rows'], bool)
        for video in sorted(m['videos']):
            bits = np.load(root/name/'videos'/video/'mask.npy')
            for qi, q in enumerate(q for q in m['episodes'] if q['video_id'] == video):
                mask[q['start']:q['stop']] = bits[qi]
        seal = seal_mask(root/name, m, mask, target=target_domain, model=source_model,
            method=name, GT_open_attempts=len(guard), extra=dict(seed=seed,
                policy_specific_teacher_evidence=True, no_GT_or_future_data_selection=True))
        methods.append(dict(method=name, rows=seal['prediction_rows']))
    assert not guard
    atomic_json(root/'completion.json', dict(status='SEALED_OBSERVATION_POLICY_RAW_NO_GT',
        pair=pair, seed=seed, methods=methods, receipts=receipts, GT_open_attempts=0,
        actual_head_fits=sum(r['actual_head_fits'] for r in receipts),
        actual_capacity_calls=sum(r['conditional_capacity_calls'] for r in receipts),
        new_teacher_encoder_calls=0, wall_seconds=time.perf_counter()-begin, goal_status='ACTIVE'))


def counts(record):
    import scripts.a100_view_fusion_launch as common
    common.PRIMARY = PRIMARY; common.BASELINES = BASELINES
    return common.counts(record)


def analyze():
    from scripts.a100_visual_classical_analyze import read_stats
    from scripts.a100_six_direction_analyze import comparison
    from scripts.a100_visual_classical_sparse_observation_analyze import fit_audit
    from scripts.analyze_pf_video_uncertainty import pooled_hota
    assert json.loads((OUT/'all_predictions_sealed_before_GT.json').read_text())['methods'] == 828
    data, statistics, summary = {}, {}, {}; error = 0.; head_fits = capacity_calls = 0
    for record in conditions():
        root = Path(record['folder']); slug = record['pair']+f"_seed{record['seed']}"
        raw = json.loads((root/'completion.json').read_text())
        count = json.loads((root/'countmatched/completion.json').read_text())
        paths = {r['method']:root/r['method'] for r in raw['methods']}
        paths.update({r['method']:root/'countmatched'/r['method'] for r in count['methods']})
        data[slug] = {name:read_stats(path) for name,path in paths.items()}
        for name, row in data[slug].items():
            num, den = row['parent_stats']
            error = max(error, abs(float(pooled_hota((num,den), np.ones(len(num))))-row['HOTA']))
            statistics[slug+'/'+name] = dict(parents=row['parent_videos'], HOTA=row['HOTA'],
                numerator=num.tolist(), denominator=den.tolist())
        head_fits += raw['actual_head_fits']; capacity_calls += raw['actual_capacity_calls']
        summary[slug] = dict(raw_HOTA={n:data[slug][n]['HOTA'] for n in METHODS},
            teacher_fit_audit=fit_audit(raw['receipts']),
            CountSelf_changed_query_files=count['CountSelf_changed_query_files'],
            actual_head_fits=raw['actual_head_fits'], actual_capacity_calls=raw['actual_capacity_calls'])
    assert error < 1e-8 and len(statistics) == 828, (error, len(statistics))
    repeated = {p:dict(mean_HOTA={n:float(np.mean([data[p+f'_seed{s}'][n]['HOTA'] for s in SEEDS])) for n in METHODS},
        comparisons={name:comparison([data[p+f'_seed{s}'][a] for s in SEEDS],
            [data[p+f'_seed{s}'][b] for s in SEEDS]) for name,(a,b) in CONTRASTS.items()}) for p in PAIRS}
    result = dict(status='OBSERVATION_POLICY_QUALITY_SIX_SETTINGS_ANALYZED', primary=PRIMARY,
        conditions=summary, repeated=repeated, HOTA_parent_sum_max_error=error,
        actual_ordinary_head_fits=head_fits, conditional_capacity_calls=capacity_calls,
        all828_outputs_sealed_before_this_GT=True, manual_target_GT_in_adaptation=False,
        all_data_development=True, independent_blind_validation=False,
        evidence='Each policy owns55019logicalrequests acrosssixsettings;sharedfrozenresponseunion71153;three neural seeds do not tripleteacherQA.',
        fixed_ordinary_learner='Natural768D directhead, rawowntrackmeanpseudo-target, confidenceweight, adequatefinal optimization; fixedstep400/finalmean anddirectteacherfill controls.',
        scope='I1policycomponenttest. Coarse learner doesnotdemonstratestateexcitationorobject-relationdecoderinteractions,andnot thefinalthreeinnovationmethod.',
        fit_gate='TeacherKL0.005appliestofinalhead,notnecessarilytheearly/finalprobabilityensemble;allremainingfailuresmuststayexplicit.',
        quality_replay_is_not_single_policy_deployment_cost=True,
        no_posthoc_decoder_or_policy_promoted=True, goal_status='ACTIVE', research_goal_complete=False)
    atomic_json(OUT/'analysis.json', result)
    (PUBLIC/'observation_policy_analysis.json').write_text(json.dumps(result, separators=(',', ':'))+'\n')
    (PUBLIC/'observation_policy_parent_statistics.json').write_text(json.dumps(statistics, separators=(',', ':'))+'\n')


def main():
    plan=json.loads(REGISTRATION.read_text())
    for name,sha in plan['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    pending=conditions();done=[];active={};failure=None;started=time.perf_counter();last=0.
    def state(status):atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=len(done),finished=done,
        active={str(g):job for g,(_,_,job) in active.items()},pending=pending,failure=failure,
        wall_seconds=time.perf_counter()-started,goal_status='ACTIVE'))
    state('AWAITING_ALL_REAL_OBSERVATION_ANSWERS_NO_GT')
    while True:
        completion=COLLECTED/'completion.json'
        status_path=COLLECTED/'supervisor/state.json'
        previous=json.loads(status_path.read_text()) if status_path.is_file() else {}
        if previous.get('failure') or previous.get('status','').startswith('FAILED_'):
            raise RuntimeError('Resolve real-teacher collection failure first')
        if completion.is_file():
            collected=json.loads(completion.read_text())
            assert collected['status']==COLLECTION_SUCCESS
            assert collected['input_audit']['logical_requests_per_policy']==55019
            break
        time.sleep(10)
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
    state('ALL18_OBSERVATION_POLICY_RAW_SEALED_GENERATING_COUNTS_NO_GT')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:roots=list(pool.map(counts,conditions()))
    tasks=[]
    for record,count_root in zip(conditions(),roots):
        for folder in (Path(record['folder']),Path(count_root)):
            row=json.loads((folder/'completion.json').read_text());assert row['GT_open_attempts']==0
            assert not any(row.get('CountSelf_changed_query_files',{}).values())
            tasks.extend((str(folder/r['method']),record['pair']) for r in row['methods'])
    assert len(tasks)==828
    atomic_json(OUT/'all_predictions_sealed_before_GT.json',dict(methods=828,
        frozen_real_teacher_collection_complete=True,GT_open_attempts=0))
    state('ALL828_OBSERVATION_POLICY_OUTPUTS_SEALED_OFFICIAL_EVALUATING')
    from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:results=dict(pool.map(evaluate_one,tasks))
    atomic_json(OUT/'official.json',dict(methods=828,results=results,goal_status='ACTIVE'))
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
            old.update(status='FAILED_OBSERVATION_POLICY_ORDINARY_REQUIRES_ROOT_CAUSE',error=repr(error),goal_status='ACTIVE');atomic_json(STATE,old);raise
