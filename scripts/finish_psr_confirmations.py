#!/usr/bin/env python3
"""One blocking CPU closure of registered seed pairs and parent-weight control."""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.pf_diag_common import official_metrics,official_video_values,FROZEN,DATA,TRACKS
from scripts.analyze_pf_video_uncertainty import paired_clusters
from scripts.analyze_pf_full_offline import analyze as attribution
from scripts.analyze_psr_matrix import official_query_groups
from scripts.analyze_psr_training_dynamics import summarize

PRIMARY=ROOT/'results/runs/psr_val_20261029_v1'


def closure(confirm,parent,public):
    confirm,parent,public=map(lambda p:Path(p).resolve(),(confirm,parent,public))
    public.mkdir(parents=True,exist_ok=True);records=[];last=None
    while True:
        pairs=[(s,m) for s in (0,42) for m in ('NAIVE_PAIRED','PSR')
               if (confirm/f'seed{s}'/'evaluation/merged'/m/'full_audit.json').exists()]
        parent_done=(parent/'completion.json').exists()
        launchers=list((confirm/'launchers').glob('*/gpu_admission.json'))
        parents_done=bool(launchers) and all((p.parent/'launcher_status.json').exists() for p in launchers)
        primary_done=(PRIMARY/'cpu_pipeline/completion.json').exists()
        current=(tuple(pairs),parent_done,parents_done,primary_done)
        if current!=last:
            print(json.dumps(dict(event='CONFIRMATION_CPU_WAIT',sealed_seed_pairs=pairs,parent_control_complete=parent_done,
                confirmation_scheduler_exited=parents_done,primary_CPU_complete=primary_done,GPU_calls=0)),flush=True);last=current
        if len(pairs)==4 and parent_done and parents_done and primary_done:break
        time.sleep(20)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    def phase(name,args):
        folder=confirm/'cpu_pipeline'/name;folder.mkdir(parents=True,exist_ok=True);receipt=folder/'completion.json'
        if receipt.exists() and json.loads(receipt.read_text())['status']=='COMPLETE':return
        start=time.perf_counter();print(json.dumps(dict(event='CONFIRMATION_CPU_PHASE_STARTED',phase=name)),flush=True)
        with (folder/'stdout.txt').open('a') as log:
            child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/args[0]),*map(str,args[1:])],cwd=ROOT,
                env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in child.stdout:
                log.write(line);log.flush()
                if line.startswith('{"event":'):print(line.rstrip(),flush=True)
            code=child.wait()
        item=dict(status='COMPLETE' if code==0 else 'FAILED_RETAINED',exit_code=code,seconds=time.perf_counter()-start,GPU_calls=0)
        atomic_json(receipt,item);records.append(dict(phase=name,**item))
        if code:raise RuntimeError('resume only failed confirmation CPU stage: '+name)
    for seed in (0,42):
        run=confirm/f'seed{seed}';evaluation=run/'evaluation/merged';dest=public/f'seed{seed}'
        phase(f'seed{seed}_controls',['prepare_psr_controls.py','--evaluation-root',evaluation,'--training-root',run/'students','--methods','NAIVE_PAIRED','PSR'])
        phase(f'seed{seed}_official',['evaluate_mllm_trackeval.py','--output-root',evaluation,'--max-jobs','4'])
        phase(f'seed{seed}_analysis',['analyze_psr_matrix.py','--evaluation-root',evaluation,'--training-root',run/'students','--output',dest])
        phase(f'seed{seed}_tables',['render_psr_tables.py','--public-root',dest])
        phase(f'seed{seed}_export',['export_psr_results.py','--run-root',run,'--output',dest])
    evaluation=parent/'evaluation/merged';method='PSR_PARENT_WEIGHT_CONTROL';dest=public/'parent_weight';dest.mkdir(exist_ok=True)
    phase('parent_controls',['prepare_psr_controls.py','--evaluation-root',evaluation,'--training-root',parent/'students','--methods',method])
    phase('parent_official',['evaluate_mllm_trackeval.py','--output-root',evaluation,'--max-jobs','4'])
    with (dest/'offline_analysis_stdout.txt').open('w') as log,redirect_stdout(log):
        report=attribution(evaluation,method,FROZEN,DATA,TRACKS,dest/'offline_attribution.json')
    metrics=official_metrics(evaluation/method/'results');videos,values=official_video_values(evaluation/method/'results/pedestrian_detailed.csv')
    draws=np.random.default_rng(0).multinomial(25,np.ones(25)/25,size=2000);comparisons={}
    for name,folder in [(m,PRIMARY/'evaluation/merged'/m) for m in ('FROZEN','PSR_DISCOVERY','PSR_GEOMETRY','PSR')]+[
            (m,evaluation/m) for m in ('FROZEN_AT_'+method+'_COUNT',method+'_AT_FROZEN_COUNT')]:
        other_videos,other=official_video_values(folder/'results/pedestrian_detailed.csv')
        if other_videos!=videos:raise ValueError('video cluster misalignment')
        boot=paired_clusters(values,other,draws);per=paired_clusters(values,other,np.eye(25))
        comparisons[name]=dict(delta_HOTA=float(paired_clusters(values,other,np.ones(25))),
            CI95=np.percentile(boot,[2.5,97.5]).tolist(),improved_videos=int((per>1e-9).sum()),declined_videos=int((per< -1e-9).sum()),
            per_video=[dict(video=v,delta_HOTA=float(x)) for v,x in zip(videos,per)])
    training=parent/'students'/method;receipt=json.loads((training/'training_receipt.json').read_text())
    run_metadata=json.loads((evaluation/method/'run_metadata.json').read_text())
    parent_result=dict(status='COMPLETE_POST_HOC_MECHANISM_CONTROL',protocol='25VAL_SAME_SET_NOT_BLIND_TEST',
        official_metrics=metrics,query_groups=official_query_groups(evaluation/method,report),comparisons=comparisons,
        control=receipt['parent_weight_control'],training=receipt,training_dynamics=summarize(training),
        inference=run_metadata,count_control_metrics={m:official_metrics(evaluation/m/'results') for m in (
            'FROZEN_AT_'+method+'_COUNT',method+'_AT_FROZEN_COUNT')},
        interpretation_limit='opportunity/positive-weight control, not an exact gradient intervention or independent replication',
        checkpoint_selection='source-val CE only',target_GT_training_reads=0)
    atomic_json(dest/'summary.json',parent_result)
    phase('parent_export',['export_psr_results.py','--run-root',parent,'--output',dest])
    seeds={17:json.loads((ROOT/'results/psr/full/summary.json').read_text()),
           **{s:json.loads((public/f'seed{s}'/'summary.json').read_text()) for s in (0,42)}}
    metrics_by_seed={str(s):{m:r['official_metrics'][m] for m in ('NAIVE_PAIRED','PSR')} for s,r in seeds.items()}
    aggregate={}
    for m in ('NAIVE_PAIRED','PSR'):
        aggregate[m]={k:dict(mean=float(np.mean([r['official_metrics'][m][k] for r in seeds.values()])),
            sample_std=float(np.std([r['official_metrics'][m][k] for r in seeds.values()],ddof=1)))
            for k in ('HOTA','DetA','AssA','IDF1','DetRe','IDSW')}
    deltas=[r['official_metrics']['PSR']['HOTA']-r['official_metrics']['NAIVE_PAIRED']['HOTA'] for r in seeds.values()]
    result=dict(status='COMPLETE_REGISTERED_3SEED_PAIRS_AND_PARENT_CONTROL',seeds=[17,0,42],
        protocol='USER_AUTHORIZED25VAL_SAME_SET_NOT_TRAIN_TO_VAL',metrics_by_seed=metrics_by_seed,aggregate=aggregate,
        PSR_minus_Naive_HOTA=dict(mean=float(np.mean(deltas)),sample_std=float(np.std(deltas,ddof=1)),values=dict(zip(map(str,seeds),deltas))),
        deterministic_Frozen_HOTA=8.2180,Frozen_is_one_reused_reference_not_three_independent_runs=True,
        seed_uncertainty_distinct_from_video_cluster_bootstrap=True,teacher_sampling_seed=17,teacher_caches_unchanged=True,
        original_ablations_seed17_only=True,target_GT_checkpoint_threshold_selection=False,
        parent_weight_summary='parent_weight/summary.json',private_roots=dict(confirm=str(confirm),parent_control=str(parent)))
    atomic_json(public/'summary.json',result)
    for run,name in ((confirm,'confirmations'),(parent,'parent_control')):
        for source in run.glob('**/*completion.json'):
            if 'evaluation' in source.relative_to(run).parts:continue
            target=public/'receipts'/name/source.relative_to(run);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        for source in run.glob('launchers/*/*'):
            if source.is_file():
                target=public/'receipts'/name/source.relative_to(run);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    lines=['# Registered three-seed confirmation','',
        '25Val same-set study; not Train→Val or blind test. Fixed caches, budgets and source-val-only checkpoint selection.','',
        '| Seed | Naive HOTA | PSR HOTA | PSR−Naive | PSR−Frozen |','|---|---:|---:|---:|---:|']
    for s,r in seeds.items():
        n,p=(r['official_metrics'][m]['HOTA'] for m in ('NAIVE_PAIRED','PSR'))
        lines.append(f'| {s} | {n:.4f} | {p:.4f} | {p-n:+.4f} | {p-8.218:+.4f} |')
    lines+=['',f"PSR mean±sample std: {aggregate['PSR']['HOTA']['mean']:.4f} ± {aggregate['PSR']['HOTA']['sample_std']:.4f}.",
        f"Naive mean±sample std: {aggregate['NAIVE_PAIRED']['HOTA']['mean']:.4f} ± {aggregate['NAIVE_PAIRED']['HOTA']['sample_std']:.4f}.",
        '',f"Post-hoc parent-weight control HOTA: {metrics['HOTA']:.4f}; no independent new weak labels.",
        'Uniform/Confidence/Discovery/Geometry/Temporal/MeanTeacher remain seed17-only; do not claim three-seed superiority over them.','']
    (public/'TABLES.md').write_text('\n'.join(lines))
    atomic_json(confirm/'cpu_pipeline/completion.json',result)
    print(json.dumps(dict(event='PSR_CONFIRMATIONS_AND_WEIGHT_CONTROL_COMPLETE',aggregate=aggregate)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--confirmation-root',type=Path,default=ROOT/'results/runs/psr_confirm_20261030_v1')
    p.add_argument('--parent-root',type=Path,default=ROOT/'results/runs/psr_parent_weight_20261030_v1')
    p.add_argument('--output',type=Path,default=ROOT/'results/psr/confirmations')
    a=p.parse_args();closure(a.confirmation_root,a.parent_root,a.output)
