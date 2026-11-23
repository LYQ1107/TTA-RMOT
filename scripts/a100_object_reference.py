"""Learn a reference representation before reconstructing referring states.

This independent fixed-evidence study leaves the running bounded-transport
study intact. Adequate direct and residual ordinary frame heads, a same-fallback
hybrid, target-decomposition controls and per-frame quantity controls remain.
Every raw/count output seals before this study's isolated official evaluation.
"""
import argparse
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
from flex_uda.a100.mechanism import PAIRS, metadata, history, generation_guard
from flex_uda.active.object_reference import reconstruct, oriented_position
from flex_uda.active.relational_transport import horizontal_direction
from flex_uda.active.relational_features import geometry, extra_features, moments, frame_supervision
from scripts.a100_six_direction import OUT as SIX, PUBLIC, SEEDS, source
from scripts.a100_view_fusion import OUT as CORE
from scripts.a100_relation_transport import OUT as BOUNDED, coarse_features, predict_frame_head
from scripts.a100_object_reference_prepare import OUT as PREPARED

OUT = SIX/'object_reference/quality_study'
STATE = OUT/'supervisor/state.json'
SUCCESS = 'PASS_OBJECT_REFERENCE_SIX_SETTINGS_RESEARCH_GOAL_ACTIVE'
PRIMARY = 'REFERENCE_FULL'
METHODS = ('COARSE_DS','FRAME_DIRECT','FRAME_RESIDUAL','FRAME_DIRECT_HYBRID',PRIMARY,
           'REFERENCE_UNSHIFTED','REFERENCE_TIME_AVERAGE','REFERENCE_SHUFFLED','REFERENCE_AT_CENTER')
BASELINES = ('COARSE_DS','FRAME_DIRECT','FRAME_RESIDUAL','FRAME_DIRECT_HYBRID','REFERENCE_TIME_AVERAGE')
CONTRASTS = {'FULL_vs_'+b:(PRIMARY,b) for b in BASELINES}
CONTRASTS.update(TARGET_DECOMPOSITION=(PRIMARY,'REFERENCE_UNSHIFTED'),
    DYNAMIC_vs_AT_CENTER=(PRIMARY,'REFERENCE_AT_CENTER'),
    CURRENT_STATE_vs_SHUFFLE=(PRIMARY,'REFERENCE_SHUFFLED'))
for _baseline in BASELINES:
    CONTRASTS['RANK_AT_FULL_COUNT_vs_'+_baseline] = (
        PRIMARY+'__AT_'+PRIMARY+'_COUNT',_baseline+'__AT_'+PRIMARY+'_COUNT')
    CONTRASTS['RANK_AT_BASE_COUNT_vs_'+_baseline] = (
        PRIMARY+'__AT_'+_baseline+'_COUNT',_baseline+'__AT_'+_baseline+'_COUNT')


def conditions():
    return [dict(pair=p,seed=s,folder=str(OUT/p/f'seed{s}')) for s in SEEDS for p in PAIRS]


def reference_fields(ds, direct, residual, learned, unshifted, coords, columns,
                     directions, beta, obs, tracks, video):
    coarse = ds[:,columns]
    active = (directions != 0) & (beta > 0)
    full = coarse.copy()
    control = coarse.copy()
    at_center = coarse.copy()
    hybrid = coarse.copy()
    if active.any():
        position = np.stack([oriented_position(coords,d) for d in directions])
        full[active] = reconstruct(learned[:,columns],position,beta)[active]
        control[active] = reconstruct(unshifted[:,columns],position,beta)[active]
        at_center[active] = learned[active][:,columns]
        hybrid[active] = direct[active]
    average = full.copy()
    shuffled = full.copy()
    for track in tracks:
        ids = np.flatnonzero(obs[:,1] == track)
        for qi in np.flatnonzero(active):
            average[qi,ids] = full[qi,ids].mean()
        if len(ids) > 1:
            key = int(hashlib.sha256((video+':'+str(int(track))+':object-reference-state-permutation').encode()).hexdigest()[:8],16)
            source_ids = np.roll(ids,1+key%(len(ids)-1))
            shuffled[:,ids] = full[:,source_ids]
    values = dict(COARSE_DS=coarse,FRAME_DIRECT=direct,FRAME_RESIDUAL=residual,
        FRAME_DIRECT_HYBRID=hybrid,REFERENCE_FULL=full,REFERENCE_UNSHIFTED=control,
        REFERENCE_TIME_AVERAGE=average,REFERENCE_SHUFFLED=shuffled,REFERENCE_AT_CENTER=at_center)
    for name in METHODS[3:]:
        assert np.array_equal(values[name][~active],coarse[~active]),name
    return values


def run(pair, seed):
    import torch
    from flex_uda.active.confidence_teacher import fit_head
    from flex_uda.presence.predictions import seal_mask
    from scripts.a100_visual_classical_strict_prefix import plain, immediate_predictions
    guard = generation_guard()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = OUT/pair/f'seed{seed}'
    if (root/'completion.json').is_file():
        return
    assert (BOUNDED/pair/f'seed{seed}'/'completion.json').is_file()
    assert json.loads((PREPARED/'completion.json').read_text())['target_manual_GT_open_attempts'] == 0
    root.mkdir(parents=True,exist_ok=True)
    model,target = PAIRS[pair]
    m = metadata(target)
    begin = time.perf_counter()
    totals = np.zeros(1087)
    squares = np.zeros(1087)
    frame_rows = 0
    reference_total = np.zeros(1081)
    reference_square = np.zeros(1081)
    reference_rows = 0
    frame_x,frame_y,reference_x,receipts = [],[],[],[]
    next_video = 0
    state_path = root/'stream.pt'
    keys = ('totals','squares','frame_rows','reference_total','reference_square','reference_rows',
            'frame_x','frame_y','reference_x','receipts','next_video')
    if state_path.is_file():
        saved = torch.load(state_path,map_location='cpu',weights_only=False)
        (totals,squares,frame_rows,reference_total,reference_square,reference_rows,
         frame_x,frame_y,reference_x,receipts,next_video) = [saved[k] for k in keys]
    for vi,(video,bank) in enumerate(sorted(m['videos'].items())):
        if vi < next_video:
            continue
        started = time.perf_counter()
        queries = [q for q in m['episodes'] if q['video_id'] == video]
        texts = [q['query_text'] for q in queries]
        obs = np.asarray(bank['observations'])
        coords,width,height = geometry(bank,target)
        directions = np.asarray([horizontal_direction(text) for text in texts],int)
        with np.load(source(pair)/'source'/video/'sparse_inputs.npz') as a:
            pcd,colours,prior,tracks = [a[k] for k in ('pcd','colours','prior','tracks')]
        lookup = {int(t):i for i,t in enumerate(tracks)}
        columns = np.asarray([lookup[int(t)] for _,t in obs])
        raw = coarse_features(pcd,colours,prior,tracks,obs,texts)
        with np.load(CORE/'quality'/pair/f'seed{seed}'/'videos'/video/'coarse_scores.npz') as a:
            assert np.array_equal(a['tracks'],tracks)
            ds = a['DS_FULL']
        with np.load(BOUNDED/pair/f'seed{seed}'/'videos'/video/'frame_scores.npz') as a:
            assert np.array_equal(a['tracks'],tracks)
            residual = a['FRAME_KD_FINAL']
        records = json.loads((history(pair)/'videos'/video/'questions.json').read_text())
        cells,teacher,teacher_coords,evidence = frame_supervision(records,queries,tracks,width,height)
        qidx,tidx,_ = cells.T
        frame_x.append(np.concatenate((raw[qidx,tidx],extra_features(teacher_coords,directions[qidx])),axis=1).astype('float32'))
        frame_y.append(teacher)
        total,square,count = moments(raw,coords,columns,directions)
        totals += total
        squares += square
        frame_rows += count
        mean = totals/frame_rows
        std = np.maximum(np.sqrt(np.maximum(squares/frame_rows-mean**2,0)),1e-3)
        x,y = np.concatenate(frame_x),np.concatenate(frame_y)
        _,fit,checkpoint = fit_head(x,np.zeros(len(x)),np.clip(y,.02,.98),x[:1],np.zeros(1),
            mean,std,sample_weight=1.+3.*np.abs(2.*y-1.),soft=True,seed=seed)
        fit.update(architecture='1087->64 GELU->1 direct ordinary frame head; no source or coarse residual',
            all_queries_all_paid_answers_same_geometry=True)
        fits = {'FRAME_DIRECT':fit}
        torch.save(checkpoint,root/'FRAME_DIRECT_latest_head.pt')
        direct = predict_frame_head(checkpoint,raw,coords,columns,directions,np.full(prior.shape,.5))
        beta = 0.
        learned = unshifted = ds
        if directions.any():
            prepared = PREPARED/pair/video
            preparation = json.loads((prepared/'completion.json').read_text())
            beta = preparation['beta']
            with np.load(prepared/'prefix_targets.npz') as a:
                reference_target,control,weight,cell_q,cell_t,cell_v = [a[k] for k in
                    ('reference_target','unshifted_target','sample_weight','cell_query','cell_track','cell_video')]
            current_cells = cell_v == len(preparation['ordered_videos'])-1
            reference_x.append(raw[cell_q[current_cells],cell_t[current_cells]])
            current = raw.reshape(-1,1081)
            reference_total += current.sum(0,dtype=float)
            reference_square += np.square(current.astype(float)).sum(0)
            reference_rows += len(current)
            ref_mean = reference_total/reference_rows
            ref_std = np.maximum(np.sqrt(np.maximum(reference_square/reference_rows-ref_mean**2,0)),1e-3)
            ref_x = np.concatenate(reference_x)
            assert len(ref_x) == len(reference_target) == preparation['prefix_reference_groups']
            fields = {}
            for name,label in (('REFERENCE',reference_target),('UNSHIFTED',control)):
                predictions,receipt,head = fit_head(ref_x,np.zeros(len(ref_x)),label,current,np.zeros(prior.size),
                    ref_mean,ref_std,sample_weight=weight,soft=True,seed=seed)
                receipt.update(architecture='1081->64 GELU->1 direct reference head; all prefix queries',
                    no_reclipping_after_reference_transport=True,raw_frame_confidence_group_sums=True)
                fits[name] = receipt
                fields[name] = predictions['CONVERGED'].reshape(prior.shape)
                torch.save(head,root/(name+'_latest_head.pt'))
            learned,unshifted = fields['REFERENCE'],fields['UNSHIFTED']
        scores = reference_fields(ds,direct,residual,learned,unshifted,coords,columns,
            directions,beta,obs,tracks,video)
        local = root/'videos'/video
        local.mkdir(parents=True,exist_ok=True)
        for name in METHODS:
            assert scores[name].shape == (len(queries),len(obs)) and np.isfinite(scores[name]).all()
            bits = plain(scores[name],False)
            destination = root/name/'videos'/video
            destination.mkdir(parents=True,exist_ok=True)
            np.save(destination/'mask.npy',bits)
            immediate_predictions(root/name,type('Context',(),dict(bank=bank,queries=queries))(),bits)
        np.savez_compressed(local/'frame_scores.npz',tracks=tracks,**scores)
        receipt = dict(video=video,index=vi,fits=fits,teacher_evidence=evidence,beta=beta,
            reference_head_fits=2 if directions.any() else 0,GT_open_attempts=len(guard),
            wall_seconds=time.perf_counter()-started,current_video_transductive=True,
            no_future_teacher_heads_or_normalization=True,new_teacher_encoder_calls=0,
            reused_original_residual_frame_comparator=True,quality_replay_not_deployment_cost=True)
        atomic_json(local/'receipt.json',receipt)
        receipts.append(receipt)
        next_video = vi+1
        saved = dict(zip(keys,(totals,squares,frame_rows,reference_total,reference_square,reference_rows,
            frame_x,frame_y,reference_x,receipts,next_video)))
        temporary = state_path.with_suffix('.tmp')
        torch.save(saved,temporary)
        temporary.replace(state_path)
        print(json.dumps(dict(event='OBJECT_REFERENCE_VIDEO',pair=pair,seed=seed,completed=next_video,total=len(m['videos']))),flush=True)
    methods = []
    for name in METHODS:
        mask = np.zeros(m['candidate_query_rows'],bool)
        for video in sorted(m['videos']):
            bits = np.load(root/name/'videos'/video/'mask.npy')
            for qi,q in enumerate(q for q in m['episodes'] if q['video_id'] == video):
                mask[q['start']:q['stop']] = bits[qi]
        seal = seal_mask(root/name,m,mask,target=target,model=model,method=name,GT_open_attempts=len(guard),
            extra=dict(seed=seed,same_fixed_paid_QA=True,independent_per_query_decoder=True,
                learned_reference_before_state_reconstruction=True,quality_replay_not_deployment_cost=True))
        methods.append(dict(method=name,rows=seal['prediction_rows']))
    assert not guard
    atomic_json(root/'completion.json',dict(status='SEALED_OBJECT_REFERENCE_RAW_NO_GT',pair=pair,seed=seed,
        methods=methods,receipts=receipts,GT_open_attempts=0,new_teacher_encoder_calls=0,
        actual_head_fits=sum(len(r['fits']) for r in receipts),wall_seconds=time.perf_counter()-begin,goal_status='ACTIVE'))


def counts(record):
    # Process-local names for the already fixed per-frame count algorithm;
    # no running module/file or other process's state is changed.
    import scripts.a100_relation_transport as common
    common.PRIMARY = PRIMARY
    common.BASELINES = BASELINES
    return common.counts(record)


def analyze():
    from scripts.a100_visual_classical_analyze import read_stats
    from scripts.a100_six_direction_analyze import comparison
    from scripts.a100_visual_classical_sparse_observation_analyze import fit_audit
    from scripts.analyze_pf_video_uncertainty import pooled_hota
    assert json.loads((OUT/'all_predictions_sealed_before_GT.json').read_text())['methods'] == 450
    data,statistics,summary = {},{},{}
    error = 0.
    head_fits = 0
    for record in conditions():
        root = Path(record['folder'])
        slug = record['pair']+f"_seed{record['seed']}"
        raw = json.loads((root/'completion.json').read_text())
        count = json.loads((root/'countmatched/completion.json').read_text())
        paths = {r['method']:root/r['method'] for r in raw['methods']}
        paths.update({r['method']:root/'countmatched'/r['method'] for r in count['methods']})
        data[slug] = {name:read_stats(path) for name,path in paths.items()}
        for name,row in data[slug].items():
            num,den = row['parent_stats']
            error = max(error,abs(float(pooled_hota((num,den),np.ones(len(num))))-row['HOTA']))
            statistics[slug+'/'+name] = dict(parents=row['parent_videos'],HOTA=row['HOTA'],numerator=num.tolist(),denominator=den.tolist())
        head_fits += raw['actual_head_fits']
        summary[slug] = dict(raw_HOTA={n:data[slug][n]['HOTA'] for n in METHODS},
            teacher_fit_audit=fit_audit(raw['receipts']),CountSelf_changed_query_files=count['CountSelf_changed_query_files'])
    assert error < 1e-8 and head_fits == 492,(error,head_fits)
    repeated = {p:dict(mean_HOTA={n:float(np.mean([data[p+f'_seed{s}'][n]['HOTA'] for s in SEEDS])) for n in METHODS},
        comparisons={n:comparison([data[p+f'_seed{s}'][a] for s in SEEDS],
                                 [data[p+f'_seed{s}'][b] for s in SEEDS]) for n,(a,b) in CONTRASTS.items()}) for p in PAIRS}
    result = dict(status='OBJECT_REFERENCE_SIX_SETTINGS_ANALYZED',primary=PRIMARY,conditions=summary,repeated=repeated,
        HOTA_parent_sum_max_error=error,actual_head_fits_total=head_fits,new_teacher_encoder_calls=0,
        all450_outputs_sealed_before_this_GT=True,manual_target_labels_in_adaptation=False,
        all_data_development=True,independent_blind_validation=False,goal_status='ACTIVE',research_goal_complete=False,
        scope='Current narrow reference learner changes only eligible descriptions; the other four settings exactly retain DS_FULL. Ordinary direct frame KD is newly trained in all six settings.',
        theory_scope='Reference target decomposition and conditional state reconstruction under an additive logit model, not a semantic or HOTA theorem.',
        quality_replay_is_not_real_deployment_cost=True,no_posthoc_control_promoted=True)
    atomic_json(OUT/'analysis.json',result)
    (PUBLIC/'object_reference_analysis.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    (PUBLIC/'object_reference_parent_statistics.json').write_text(json.dumps(statistics,separators=(',',':'))+'\n')
    print(json.dumps(dict(event='OBJECT_REFERENCE_ANALYSIS',repeated=repeated,goal_status='ACTIVE')),flush=True)


def main():
    plan = json.loads((PUBLIC/'object_reference_quality_registration_v1.json').read_text())
    for name,sha in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == sha,name
    assert plan['total_outputs'] == 450 and plan['actual_neural_fits'] == 492
    pending,done,active = conditions(),[],{}
    failure = None
    start = time.perf_counter()
    last = 0.
    def state(status):
        atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=len(done),finished=done,
            active={str(g):job for g,(_,_,job) in active.items()},pending=pending,failure=failure,
            wall_seconds=time.perf_counter()-start,goal_status='ACTIVE'))
    state('AWAITING_BOUNDED_RELATION_GPU_RELEASE_NO_GT')
    while True:
        old = json.loads((BOUNDED/'supervisor/state.json').read_text())
        if old.get('failure') or old['status'].startswith('FAILED_'):
            raise RuntimeError('Predecessor failed; preserve its workers and resolve its root cause')
        if old.get('completed') == 18 and not old.get('active') and not old.get('pending'):
            break
        time.sleep(5)
    while pending or active:
        changed = False
        for gpu,(proc,handle,job) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            handle.close()
            del active[gpu]
            if code or not (Path(job['folder'])/'completion.json').is_file():
                failure = dict(job=job,exit_code=code,last_log=Path(job['log']).read_text()[-6000:])
            else:
                done.append(job)
            changed = True
        candidates = [g for g in (0,1,2,3) if g not in active]
        if pending and candidates and not failure:
            output = subprocess.check_output(['nvidia-smi','-i',','.join(map(str,candidates)),
                '--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            memory = {int(r.split(',')[0]):int(r.split(',')[1]) for r in output.strip().splitlines()}
            for gpu in candidates:
                if not pending or memory[gpu] >= 512:
                    continue
                job = pending.pop(0)
                if (Path(job['folder'])/'completion.json').is_file():
                    done.append(dict(job,reused_sealed=True))
                    changed = True
                    continue
                log = OUT/'supervisor'/(job['pair']+'_seed'+str(job['seed'])+'.log')
                handle = log.open('a',buffering=1)
                proc = subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--pair',job['pair'],'--seed',str(job['seed'])],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
                active[gpu] = proc,handle,dict(job,pid=proc.pid,log=str(log),physical_GPU=gpu)
                changed = True
        if changed or time.perf_counter()-last > 45:
            state('RUNNING_OBJECT_REFERENCE_SIX_SETTINGS_NO_GT' if not failure else 'PRESERVING_ACTIVE_OBJECT_REFERENCE_AFTER_FAILURE')
            last = time.perf_counter()
        if failure and not active:
            raise RuntimeError(json.dumps(failure))
        if pending or active:
            time.sleep(3)
    assert len(done) == 18
    state('ALL18_REFERENCE_RAW_SEALED_GENERATING_COUNTS_NO_GT')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        roots = list(pool.map(counts,conditions()))
    tasks = []
    for record,count_root in zip(conditions(),roots):
        for folder in (Path(record['folder']),Path(count_root)):
            row = json.loads((folder/'completion.json').read_text())
            assert row['GT_open_attempts'] == 0
            assert not any(row.get('CountSelf_changed_query_files',{}).values())
            tasks.extend((str(folder/r['method']),record['pair']) for r in row['methods'])
    assert len(tasks) == 450
    atomic_json(OUT/'all_predictions_sealed_before_GT.json',dict(methods=450,conditions=18,GT_open_attempts=0))
    state('ALL450_REFERENCE_OUTPUTS_SEALED_OFFICIAL_EVALUATING')
    from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        results = dict(pool.map(evaluate_one,tasks))
    atomic_json(OUT/'official.json',dict(methods=450,results=results,goal_status='ACTIVE'))
    analyze()
    state(SUCCESS)
    print(SUCCESS,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair',choices=tuple(PAIRS))
    parser.add_argument('--seed',type=int)
    args = parser.parse_args()
    if args.pair:
        import fcntl
        condition = OUT/args.pair/f'seed{args.seed}'
        condition.mkdir(parents=True,exist_ok=True)
        with (condition/'condition.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            run(args.pair,args.seed)
    else:
        try:
            main()
        except Exception as error:
            previous = json.loads(STATE.read_text()) if STATE.is_file() else {}
            previous.update(status='FAILED_OBJECT_REFERENCE_REQUIRES_ROOT_CAUSE',error=repr(error),goal_status='ACTIVE')
            atomic_json(STATE,previous)
            raise
