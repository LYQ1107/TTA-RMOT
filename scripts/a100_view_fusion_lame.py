"""Fixed published LAME comparators for RMOT, not new claimed innovations.

CPU-only reuse of immutable multiview heads. Two feature/head scopes are
reported separately; their difference is not an attribution to reliability.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import PAIRS,metadata,generation_guard
from flex_uda.presence.predictions import seal_mask
from flex_uda.active.lame_reference import binary
from scripts.a100_six_direction import SEEDS,PUBLIC,source
from scripts.a100_view_fusion import OUT as CORE

OUT=CORE/'lame_reference';STATE=OUT/'supervisor/state.json'
SUCCESS='PASS_SIX_SETTINGS_LAME_REFERENCE_RESEARCH_GOAL_ACTIVE'
FULL='DS_FULL';BASES=('RAW_LAME_KNN5','DS_RGB_LAME_KNN5')
CONTRASTS={
    'FULL_vs_ORDINARY_KD_LAME':(FULL,BASES[0]),
    'FULL_vs_SAME_DS_HEAD_RGB_LAME':(FULL,BASES[1]),
    'LAME_ON_ORDINARY_HEAD':(BASES[0],'RAW_FINAL'),
    'RGB_LAME_ON_DS_HEAD':(BASES[1],'DS_FINAL')}
for name in BASES:
    CONTRASTS['RANK_AT_FULL_COUNT_vs_'+name]=(FULL+'__AT_'+FULL+'_COUNT',name+'__AT_'+FULL+'_COUNT')
    CONTRASTS['RANK_AT_BASE_COUNT_vs_'+name]=(FULL+'__AT_'+name+'_COUNT',name+'__AT_'+name+'_COUNT')


def conditions():return [dict(pair=p,seed=s,folder=str(OUT/'quality'/p/f'seed{s}')) for s in SEEDS for p in PAIRS]


def generate(pair,seed):
    from flex_uda.active.context_cohort import distillation_features
    from flex_uda.active.semantic_colour import appearance_cues
    from scripts.a100_visual_classical_strict_prefix import plain,immediate_predictions
    guard=generation_guard();root=OUT/'quality'/pair/f'seed{seed}';core=CORE/'quality'/pair/f'seed{seed}'
    assert (core/'completion.json').is_file()
    if (root/'completion.json').is_file():return
    root.mkdir(parents=True,exist_ok=True);start=time.perf_counter();model,target=PAIRS[pair];m=metadata(target)
    for name in (FULL,'RAW_FINAL','DS_FINAL'):
        if not (root/name).exists():(root/name).symlink_to(core/name,target_is_directory=True)
    masks={name:np.zeros(m['candidate_query_rows'],bool) for name in BASES};receipts=[]
    for video,bank in sorted(m['videos'].items()):
        begin=time.perf_counter();queries=[q for q in m['episodes'] if q['video_id']==video];texts=[q['query_text'] for q in queries]
        with np.load(source(pair)/'source'/video/'sparse_inputs.npz') as a:pcd,colours,tracks=[a[k] for k in ('pcd','colours','tracks')]
        assert np.all(np.diff(tracks)>0),'Stable tie rule requires ascending Track IDs'
        obs=np.asarray(bank['observations']);lengths=np.asarray([(obs[:,1]==t).sum() for t in tracks])
        cue,available,_,_=appearance_cues(colours,texts,lengths)
        rgb=np.concatenate([distillation_features(pcd,colours,texts,'extended'),
            np.where(available,2*cue-1,0).astype('float32')[:,:,None]],axis=2)
        with np.load(core/'videos'/video/'coarse_scores.npz') as a:
            assert np.array_equal(a['tracks'],tracks)
            full=a[FULL];heads={'RAW_FINAL':a['RAW_FINAL'],'DS_FINAL':a['DS_FINAL']}
        scores={FULL:full};details={}
        for name,head,features in ((BASES[0],'RAW_FINAL',pcd),(BASES[1],'DS_FINAL',rgb)):
            score=np.empty_like(heads[head],dtype=float);records=[]
            for qi in range(len(queries)):score[qi],receipt=binary(heads[head][qi],features[qi]);records.append(receipt)
            scores[name]=score;details[name]=dict(head=head,feature_dimension=features.shape[-1],
                max_iterations=max(r['iterations'] for r in records),energy_stopping_met_queries=sum(r['energy_stopping_met'] for r in records),queries=len(queries))
        lookup={int(t):i for i,t in enumerate(tracks)};columns=np.asarray([lookup[int(t)] for _,t in obs])
        for name in BASES:
            bits=plain(scores[name][:,columns],False);dest=root/name/'videos'/video;dest.mkdir(parents=True,exist_ok=True)
            np.save(dest/'mask.npy',bits);immediate_predictions(root/name,type('Context',(),dict(bank=bank,queries=queries))(),bits)
            for qi,q in enumerate(queries):masks[name][q['start']:q['stop']]=bits[qi]
        local=root/'videos'/video;local.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(local/'coarse_scores.npz',tracks=tracks,**scores)
        receipts.append(dict(video=video,details=details,wall_seconds=time.perf_counter()-begin,GT_open_attempts=len(guard)))
    records=[]
    for name in BASES:
        seal=seal_mask(root/name,m,masks[name],target=target,model=model,method=name,GT_open_attempts=len(guard),
            extra=dict(published_LAME_adapter_not_our_innovation=True,neighbors=5,lambda_value=1,max_steps=100,
                no_new_teacher_encoder_head_calls=True,independent_decoder=True,current_video_transductive=True))
        records.append(dict(method=name,rows=seal['prediction_rows']))
    assert not guard
    atomic_json(root/'completion.json',dict(status='SEALED_LAME_REFERENCE_OUTPUTS',pair=pair,seed=seed,methods=records,
        receipts=receipts,GT_open_attempts=0,new_teacher_encoder_head_calls=0,wall_seconds=time.perf_counter()-start))


def counts(record):
    # Isolated count process; local method-name constants only, no source edit.
    import scripts.a100_view_fusion_launch as common
    common.PRIMARY=FULL;common.BASELINES=BASES
    return common.counts(record)


def analyze():
    from scripts.a100_visual_classical_analyze import read_stats
    from scripts.a100_six_direction_analyze import comparison
    from scripts.analyze_pf_video_uncertainty import pooled_hota
    assert json.loads((OUT/'all_predictions_sealed_before_GT.json').read_text())['methods']==162
    names=sorted({n for pair in CONTRASTS.values() for n in pair});data={};sufficient={};error=0.
    for r in conditions():
        root=Path(r['folder']);values={n:read_stats(root/'countmatched'/n if '__AT_' in n else root/n) for n in names}
        data[r['pair'],r['seed']]=values
        for name,row in values.items():
            num,den=row['parent_stats'];error=max(error,abs(float(pooled_hota((num,den),np.ones(len(num))))-row['HOTA']))
            sufficient[r['pair']+f"_seed{r['seed']}/"+name]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],numerator=num.tolist(),denominator=den.tolist())
    assert error<1e-8,error
    repeated={p:dict(mean_HOTA={n:float(np.mean([data[p,s][n]['HOTA'] for s in SEEDS])) for n in (FULL,*BASES,'RAW_FINAL','DS_FINAL')},
        comparisons={n:comparison([data[p,s][a] for s in SEEDS],[data[p,s][b] for s in SEEDS]) for n,(a,b) in CONTRASTS.items()}) for p in PAIRS}
    result=dict(status='PUBLISHED_LAME_RMOT_REFERENCE_ANALYZED',repeated=repeated,
        all162_supplement_outputs_sealed_before_supplement_GT=True,core468_separate_GT_group=True,
        new_teacher_encoder_head_calls=0,HOTA_parent_sum_max_error=error,primary_unchanged=True,all_data_development=True,
        comparator_scope='RAW_LAME uses RAW_FINAL/768PCD; DS_RGB_LAME uses DS_FINAL/1081PCD+RGB+cue. Do not attribute their difference to one module. Both are query-track binary adaptations of published LAME, not original classification benchmark reproduction.',
        no_new_innovation_claim_for_LAME_or_graph_output_correction=True,goal_status='ACTIVE')
    atomic_json(OUT/'analysis.json',result)
    (PUBLIC/'lame_reference_analysis.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    (PUBLIC/'lame_reference_parent_statistics.json').write_text(json.dumps(sufficient,separators=(',',':'))+'\n')
    print(json.dumps(dict(event='LAME_RMOT_REFERENCE_ANALYSIS',repeated=repeated,goal_status='ACTIVE')),flush=True)


def main():
    plan=json.loads((PUBLIC/'lame_reference_registration_v1.json').read_text())
    assert plan['source_sha256']==file_sha(Path(__file__)) and plan['adapter_sha256']==file_sha(ROOT/'flex_uda/active/lame_reference.py')
    pending=conditions();done=[];active={};failure=None;start=time.perf_counter();last=0.
    def state(status):atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=len(done),finished=done,
        active={str(s):j for s,(_,_,j) in active.items()},pending=pending,failure=failure,wall_seconds=time.perf_counter()-start,goal_status='ACTIVE'))
    while pending or active:
        changed=False
        for slot,(proc,handle,job) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            handle.close();del active[slot]
            if code or not (Path(job['folder'])/'completion.json').is_file():failure=dict(job=job,exit_code=code,last_log=Path(job['log']).read_text()[-6000:])
            else:done.append(job)
            changed=True
        runnable=[r for r in pending if (CORE/'quality'/r['pair']/f"seed{r['seed']}"/'completion.json').is_file()]
        for slot in range(2):
            if failure or slot in active or not runnable:continue
            job=runnable.pop(0);pending.remove(job)
            if (Path(job['folder'])/'completion.json').is_file():done.append(dict(job,reused_sealed=True));changed=True;continue
            log=OUT/'supervisor'/(job['pair']+'_seed'+str(job['seed'])+'.log');handle=log.open('a',buffering=1)
            proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--pair',job['pair'],'--seed',str(job['seed'])],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
            active[slot]=proc,handle,dict(job,pid=proc.pid,log=str(log));changed=True
        if changed or time.perf_counter()-last>45:
            state('AWAITING_OR_GENERATING_FIXED_LAME_COMPARATORS_NO_GT' if not failure else 'PRESERVING_ACTIVE_LAME_AFTER_FAILURE');last=time.perf_counter()
        if failure and not active:raise RuntimeError(json.dumps(failure))
        if pending or active:time.sleep(3)
    assert len(done)==18
    state('ALL18_LAME_RAW_CONDITIONS_SEALED_GENERATING_COUNTS_NO_GT')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:roots=list(pool.map(counts,conditions()))
    tasks=[]
    for r,count_root in zip(conditions(),roots):
        for folder in (Path(r['folder']),Path(count_root)):
            row=json.loads((folder/'completion.json').read_text());assert row['GT_open_attempts']==0
            assert not any(row.get('CountSelf_changed_query_files',{}).values())
            tasks.extend((str(folder/e['method']),r['pair']) for e in row['methods'])
    assert len(tasks)==162
    atomic_json(OUT/'all_predictions_sealed_before_GT.json',dict(methods=162,conditions=18,GT_open_attempts=0))
    state('ALL162_LAME_OUTPUTS_SEALED_AWAITING_CORE_ANALYSIS')
    while not (CORE/'analysis.json').is_file():time.sleep(5)
    from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one
    state('ALL162_LAME_OUTPUTS_SEALED_OFFICIAL_EVALUATING')
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:results=dict(pool.map(evaluate_one,tasks))
    atomic_json(OUT/'official.json',dict(methods=162,results=results,goal_status='ACTIVE'))
    analyze();state(SUCCESS);print(SUCCESS,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pair',choices=tuple(PAIRS));parser.add_argument('--seed',type=int);args=parser.parse_args()
    if args.pair:generate(args.pair,args.seed)
    else:
        try:main()
        except Exception as error:
            previous=json.loads(STATE.read_text()) if STATE.is_file() else {}
            previous.update(status='FAILED_LAME_REFERENCE_REQUIRES_ROOT_CAUSE',error=repr(error),goal_status='ACTIVE')
            atomic_json(STATE,previous);raise
