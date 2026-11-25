"""Four fixed scope diagnostics, then sealed-only cache-free metric verification."""
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
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_benchmark_scope import AUDIT,FRESH
from scripts.a100_visual_classical_prepare import OUT


def clone_for_uncached():
    from flex_uda.a100.mechanism import metadata,generation_guard
    from flex_uda.presence.predictions import seal_mask
    guard=generation_guard();m=metadata('SportsMOT');methods=[]
    sources={'COLD_KD':FRESH/'methods/CLIP_KD',
             'COLD_GRAPH':OUT/'cold_full_c4/C4_key_single_CLIP_COHORT_seed0/methods/CLIP_COHORT'}
    for name,source in sources.items():
        mask=np.zeros(m['candidate_query_rows'],dtype=bool)
        for video in sorted(m['videos']):
            local=np.load(source/'videos'/video/'mask.npy')
            for qi,q in enumerate(q for q in m['episodes'] if q['video_id']==video):mask[q['start']:q['stop']]=local[qi]
        folder=AUDIT/'uncached'/name
        seal=seal_mask(folder,m,mask,target='SportsMOT',model='Dance-best',method='UNCACHED_VERIFY_'+name,
            GT_open_attempts=len(guard),extra=dict(immutable_prediction_reference=str(source),no_algorithm_change=True,
                no_new_teacher_calls=True,cache_bypass_re_evaluation_only=True))
        original=json.loads((source/'predictions_seal.json').read_text())
        a={(r['video'],r['expression']):r['prediction_sha256'] for r in original['episodes']}
        b={(r['video'],r['expression']):r['prediction_sha256'] for r in seal['episodes']}
        assert a==b
        methods.append(dict(method=name,source=str(source),identical_query_predictions=len(a)))
    assert not guard
    atomic_json(AUDIT/'uncached/completion.json',dict(methods=methods,GT_open_attempts=0))


def evaluate_job(job):
    folder,pair,uncached=job
    if uncached:
        from scripts.a100_mechanism_offline import evaluate
    else:
        from flex_uda.a100.evaluation_reuse import evaluate
    return str(folder),evaluate(folder,pair)


def membership_diagnostic():
    # Post-seal evaluation only. These labels are never an adaptation input.
    from flex_uda.a100.mechanism import metadata,PAIRS,OUT as DIAG
    result={}
    for pair in ('C4','C1'):
        m=metadata(PAIRS[pair][1]);totals={};volume={'all_pairs':0,'asked_pairs':0,'eligible_pairs':0,'queries':0,'asked_observation_rows':0}
        for video,bank in sorted(m['videos'].items()):
            truth=np.load(DIAG/'offline_GT'/pair/video/'labels.npy')>=0
            obs=np.asarray(bank['observations']);tracks=sorted(set(obs[:,1].tolist()))
            qn=len([q for q in m['episodes'] if q['video_id']==video]);length=np.asarray([(obs[:,1]==t).sum() for t in tracks])
            path=FRESH/'source'/video/'answers.json' if pair=='C4' else OUT/'causal_teacher'/pair/'COVER'/(video+'.json')
            answers=json.loads(path.read_text())['answers'];known=np.zeros((qn,len(tracks)),bool)
            for a in answers:known[int(a['local_query']),int(a['local_track'])]=True
            columns=np.asarray([tracks.index(int(t)) for _,t in obs])
            observed=known[:,columns]
            volume['all_pairs']+=known.size;volume['asked_pairs']+=int(known.sum());volume['eligible_pairs']+=qn*int((length>=3).sum())
            volume['queries']+=qn;volume['asked_observation_rows']+=int(observed.sum())
            folders={name:AUDIT/pair/'CURRENT_ONLY'/name for name in ['SOURCE_ONLY','DIRECT_TEACHER','CURRENT_ONLY','PREFIX_REPRO']}
            folders['PAST_ONLY']=AUDIT/pair/'PAST_ONLY'/'PAST_ONLY'
            for name,folder in folders.items():
                mask=np.load(folder/'videos'/video/'mask.npy')
                for scope,choose in [('asked',observed),('unasked',~observed)]:
                    counter=totals.setdefault(name,{}).setdefault(scope,dict(TP=0,FP=0,FN=0,rows=0))
                    counter['TP']+=int((mask&truth&choose).sum());counter['FP']+=int((mask&~truth&choose).sum())
                    counter['FN']+=int((~mask&truth&choose).sum());counter['rows']+=int(choose.sum())
        result[pair]=dict(volume=volume,candidate_IoU_membership_counts=totals,
            scope='Post-seal diagnostic only; candidate-IoU counts are not official HOTA or an additive attribution')
    return result


def main():
    AUDIT.mkdir(parents=True,exist_ok=True);supervisor=OUT/'supervisor';state=supervisor/'benchmark_scope_state.json'
    begin=time.perf_counter();jobs=[]
    for gpu,(pair,scope) in enumerate([('C4','CURRENT_ONLY'),('C4','PAST_ONLY'),('C1','CURRENT_ONLY'),('C1','PAST_ONLY')]):
        log_path=supervisor/f'benchmark_scope_{pair}_{scope}.log';log=log_path.open('a',buffering=1)
        process=subprocess.Popen([sys.executable,'scripts/a100_visual_classical_benchmark_scope.py','--pair',pair,'--scope',scope],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
        jobs.append((process,log,dict(pair=pair,scope=scope,pid=process.pid,GPU=gpu,log=str(log_path))))
    atomic_json(state,dict(status='RUNNING_FIXED_BASELINE_SCOPE_AUDIT',pid=os.getpid(),jobs=[r for _,_,r in jobs]))
    for process,log,record in jobs:
        record['exit_code']=process.wait();log.close()
        atomic_json(state,dict(status='RUNNING_FIXED_BASELINE_SCOPE_AUDIT',pid=os.getpid(),jobs=[r for _,_,r in jobs]))
        if record['exit_code']:raise RuntimeError('Scope worker failed; preserve and inspect '+record['log'])
        sealed=json.loads((AUDIT/record['pair']/record['scope']/'completion.json').read_text());assert sealed['GT_open_attempts']==0
    subprocess.run([sys.executable,__file__,'--clone-for-uncached'],cwd=ROOT,check=True)
    atomic_json(state,dict(status='ALL_SCOPE_PREDICTIONS_SEALED_EVALUATING',pid=os.getpid(),jobs=[r for _,_,r in jobs]))
    from flex_uda.a100.evaluation_reuse import bootstrap
    bootstrap();tasks=[]
    for _,_,record in jobs:
        root=AUDIT/record['pair']/record['scope'];sealed=json.loads((root/'completion.json').read_text())
        tasks.extend((root/r['method'],record['pair'],False) for r in sealed['methods'])
    for name in ('COLD_KD','COLD_GRAPH'):tasks.append((AUDIT/'uncached'/name,'C4',True))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:evaluated=dict(pool.map(evaluate_job,tasks))
    values={pair:{} for pair in ('C4','C1')}
    for folder,pair,uncached in tasks:
        if not uncached:values[pair][folder.name]=evaluated[str(folder)]['HOTA_unrounded']
    uncached={name:evaluated[str(AUDIT/'uncached'/name)] for name in ('COLD_KD','COLD_GRAPH')}
    refs={'COLD_KD':FRESH/'methods/CLIP_KD','COLD_GRAPH':OUT/'cold_full_c4/C4_key_single_CLIP_COHORT_seed0/methods/CLIP_COHORT'}
    differences={name:r['HOTA_unrounded']-json.loads((refs[name]/'official_completion.json').read_text())['HOTA_unrounded'] for name,r in uncached.items()}
    assert max(map(abs,differences.values()))<1e-8
    # Exact re-fit diagnostic: record any threshold-sensitive differences,
    # rather than replacing a newly produced prediction by the reference.
    reproduction={}
    for pair,reference in [('C4',refs['COLD_KD']),('C1',OUT/'strong_cover_seed0_clipped/C1/SOFT_KD')]:
        a=json.loads((reference/'predictions_seal.json').read_text())
        b=json.loads((AUDIT/pair/'CURRENT_ONLY/PREFIX_REPRO/predictions_seal.json').read_text())
        lookup={(r['video'],r['expression']):r['prediction_sha256'] for r in a['episodes']}
        changed=sum(lookup[r['video'],r['expression']]!=r['prediction_sha256'] for r in b['episodes'])
        reproduction[pair]=dict(changed_query_prediction_files=changed,
            reference_HOTA=json.loads((reference/'official_completion.json').read_text())['HOTA_unrounded'],refit_HOTA=values[pair]['PREFIX_REPRO'])
    result=dict(status='PASS_BASELINE_SCOPE_AND_UNCACHED_EVALUATION_AUDIT',HOTA=values,
        scope_is_diagnostic_not_matching_different_label_budgets=True,uncached_HOTA_differences=differences,
        uncached_evaluated_queries={k:v['evaluated_episodes'] for k,v in uncached.items()},
        prefix_reproduction=reproduction,membership=membership_diagnostic(),
        new_teacher_calls=0,all_development_data=True,prepaid_features_and_feedback=True,
        not_a_cold_deployment_benchmark=True,total_study_wall_seconds=time.perf_counter()-begin,
        user_research_goal_still_active=True)
    atomic_json(OUT/'analysis/benchmark_scope_audit.json',result)
    atomic_json(state,dict(status='PASS_BASELINE_SCOPE_AUDIT_GOAL_STILL_ACTIVE',pid=os.getpid(),jobs=[r for _,_,r in jobs]))
    print(json.dumps(dict(event='BENCHMARK_SCOPE_AUDIT_COMPLETE',HOTA=values,uncached_HOTA_differences=differences)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--clone-for-uncached',action='store_true');a=p.parse_args()
    if a.clone_for_uncached:clone_for_uncached()
    else:
        try:main()
        except Exception:
            atomic_json(OUT/'supervisor/benchmark_scope_state.json',dict(status='FAILED_SCOPE_AUDIT_REQUIRES_ROOT_CAUSE',pid=os.getpid()))
            raise
