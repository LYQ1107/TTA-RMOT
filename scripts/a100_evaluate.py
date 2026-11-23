"""Independent post-seal official HOTA evaluation and causal/cost analysis."""
import argparse
from collections import defaultdict
import contextlib
import csv
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'vendor/FlexHook/TrackEval'))
import compat.sitecustomize
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.runtime import OUTPUT,HISTORY,metadata,relocate
from flex_uda.presence.predictions import seal_mask,verify_predictions


def evaluate(folder,meta,scope='full'):
    folder=Path(folder);seal=verify_predictions(folder)
    if (folder/'official_completion.json').exists():return json.loads((folder/'official_completion.json').read_text())
    # This is the first GT access, exclusively in a separate offline process.
    gt_manifest=json.loads((ROOT/'results/multidomain_generality/sports_GT_reconstruction_v4.json').read_text())
    gt_root=relocate(gt_manifest['evaluation_root'])/'gt_template'
    seq_info={}
    for q in meta['episodes']:
        v,e=q['video_id'],q['expression_id'];n=int(meta['videos'][v]['frame_count'])
        seq_info[v+'+'+e]=n
        p=folder/'results'/v/e/'gt.txt';p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():continue
        original=gt_root/v/e/'gt.txt'
        if scope=='full':p.symlink_to(original)
        else:
            stop=meta['videos'][v]['evaluation_last_frame']
            text=''.join(line+'\n' for line in original.read_text().splitlines() if line.strip() and 1<=int(float(line.split(',')[0]))<=stop)
            p.write_text(text)
    import trackeval
    cfg=dict(GT_FOLDER=str(gt_root),TRACKERS_FOLDER=str(folder/'results'),TRACKERS_TO_EVAL=[str(folder/'results')],
        SKIP_SPLIT_FOL=True,GT_LOC_FORMAT='{gt_folder}/{video_id}/{expression_id}/gt.txt',SEQ_INFO=seq_info,PRINT_CONFIG=False)
    evcfg=dict(USE_PARALLEL=len(seq_info)>20,NUM_PARALLEL_CORES=2,PRINT_RESULTS=False,PRINT_ONLY_COMBINED=True,
        PRINT_CONFIG=False,TIME_PROGRESS=False,DISPLAY_LESS_PROGRESS=True,OUTPUT_SUMMARY=True,OUTPUT_DETAILED=True,PLOT_CURVES=False,
        BREAK_ON_ERROR=True,RETURN_ON_ERROR=False,LOG_ON_ERROR=str(folder/'trackeval_error.log'))
    started=time.perf_counter()
    with (folder/'trackeval.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        evaluator=trackeval.Evaluator(evcfg)
        evaluator.evaluate([trackeval.datasets.MotChallenge2DBox(cfg)],
            [m(dict(PRINT_CONFIG=False,THRESHOLD=.5)) for m in (trackeval.metrics.HOTA,trackeval.metrics.CLEAR,trackeval.metrics.Identity)])
    from scripts.pf_diag_common import official_metrics
    from scripts.active_eval import official_stats
    values=official_metrics(folder/'results');stats=official_stats(folder)
    assert int(values['Dets'])==seal['prediction_rows']
    result=dict(status='PASS',metrics=values,HOTA_unrounded=stats['HOTA_unrounded'],scope=scope,episodes=len(seq_info),
        wall_seconds=time.perf_counter()-started,detailed_sha256=file_sha(folder/'results/pedestrian_detailed.csv'),
        prediction_seal_sha256=file_sha(folder/'predictions_seal.json'),GT_protocol=gt_manifest['protocol'])
    atomic_json(folder/'official_completion.json',result)
    print(json.dumps(dict(event='OFFICIAL_EVAL_DONE',folder=str(folder),HOTA=stats['HOTA_unrounded'])),flush=True)
    return result


def bootstrap(a,b):
    from scripts.active_eval import official_stats
    from scripts.analyze_pf_video_uncertainty import pooled_hota,paired_clusters
    aa,bb=official_stats(a),official_stats(b);assert aa['videos']==bb['videos']
    n=len(aa['videos']);rng=np.random.default_rng(0);counts=rng.multinomial(n,np.ones(n)/n,size=10000)
    delta=paired_clusters(aa['stats'],bb['stats'],np.ones(n));ci=np.percentile(paired_clusters(aa['stats'],bb['stats'],counts),[2.5,97.5])
    individual=paired_clusters(aa['stats'],bb['stats'],np.eye(n))
    return dict(delta_HOTA=float(delta),paired_video_CI95=ci.tolist(),CI_excludes_zero=bool(ci[0]>0 or ci[1]<0),
        samples=10000,seed=0,videos=n,improved_videos=int((individual>1e-9).sum()),declined_videos=int((individual< -1e-9).sum()),
        per_video=[dict(video=v,delta_HOTA=float(d)) for v,d in zip(aa['videos'],individual)])


def cache_runtime_profile(method):
    groups={}
    for p in (OUTPUT/'stage2'/method/'videos').glob('*/receipt.json'):
        r=json.loads(p.read_text());capacity=r.get('feature_cache_mib') or 512
        g=groups.setdefault(str(capacity),dict(videos=0,optimizer_steps=0,slow_update_phase_wall_seconds=0.,
            video_start_score_phase_wall_seconds=0.,fixed32_score_phase_wall_seconds=0.,recalc_events=0,
            recorded_backbone_calls=0,recorded_visual_hits=0))
        g['videos']+=1;g['optimizer_steps']+=r['slow_updates']
        g['slow_update_phase_wall_seconds']+=sum(x['slow']['GPU_seconds'] for x in r['rounds'])
        for score in r['initial_and_recalc_score_records']:
            key='video_start_score_phase_wall_seconds' if score['kind']=='VIDEO_START_ALL' else 'fixed32_score_phase_wall_seconds'
            g[key]+=score['GPU_seconds'];g['recalc_events']+=score['kind']=='FIXED32'
        cache=r.get('frozen_encoder_cache_stats') or {}
        g['recorded_backbone_calls']+=cache.get('backbone_calls',0);g['recorded_visual_hits']+=cache.get('visual_hits',0)
    for g in groups.values():
        g['slow_phase_wall_seconds_per_optimizer_step']=(g['slow_update_phase_wall_seconds']/g['optimizer_steps'] if g['optimizer_steps'] else None)
    return groups


def countmatched():
    meta=metadata();root=OUTPUT/'stage2';methods=('SLOW_REPLAY_MATCHED','MINUS_RECALC_ONLY')
    for method in methods:verify_predictions(root/method)
    pairs=[(r,c) for r in methods for c in methods]
    masks={(r,c):np.zeros(meta['candidate_query_rows'],dtype=bool) for r,c in pairs}
    for video,bank in sorted(meta['videos'].items()):
        queries=[q for q in meta['episodes'] if q['video_id']==video]
        obs=np.asarray(bank['observations']);by_frame={f:np.flatnonzero(obs[:,0]==f) for f in np.unique(obs[:,0])}
        scores={r:np.load(root/r/'videos'/video/'frame_cache/margin.npy',mmap_mode='r') for r in methods}
        original={c:np.load(root/c/'videos'/video/'mask.npy') for c in methods}
        for r,c in pairs:
            local=np.zeros_like(original[c])
            for qi,q in enumerate(queries):
                for f,ids in by_frame.items():
                    k=int(original[c][qi,ids].sum())
                    # Stable physical ID tie, identical to the original top-k rule.
                    ranked=sorted(ids.tolist(),key=lambda i:(-float(scores[r][qi,i]),int(obs[i,1])))
                    local[qi,ranked[:k]]=True
                    assert int(local[qi,ids].sum())==k
                masks[r,c][q['start']:q['stop']]=local[qi]
    for (r,c),mask in masks.items():
        name=('LORA' if r==methods[0] else 'FROZEN')+'_RANK__'+('LORA' if c==methods[0] else 'RECALC')+'_COUNT'
        seal_mask(root/'countmatched'/name,meta,mask,target='SportsMOT',model='Dance-best',method=name,GT_open_attempts=0,
            extra=dict(role='fixed predicted-count diagnostic, no GT counts',ranking_source=r,count_source=c,
                rank='actual cached candidate margin; no teacher posterior or fast intercept used',count_exact_per_query_frame=True))
    print(json.dumps(dict(event='COUNTMATCHED_SEALED',comparisons=4,GT_used=False)),flush=True)


def cost_ledger(folder,supervisor):
    p=folder/'phase_events.jsonl';events=[json.loads(line) for line in p.read_text().splitlines()]
    active={};durations=defaultdict(float);attempted=returned=new=hits=0;forward=0.
    end=min(supervisor['wall_seconds'],supervisor['budget_seconds']) if supervisor['budget_seconds'] else supervisor['wall_seconds']
    for event in events:
        name=event['phase']
        if event['event']=='BEGIN':
            active[name]=event
            attempted+=event.get('attempted_requests',0)
        else:
            begin=active.pop(name)
            durations[name]+=max(0.,min(end,event['elapsed_seconds'])-begin['elapsed_seconds'])
            if name=='teacher_request':
                returned+=event['returned_requests'];cost=event['costs']
                new+=cost['new_calls'];hits+=cost['cache_hits'];forward+=cost['forward_seconds']
    unfinished=[]
    for name,event in active.items():
        duration=max(0.,end-event['elapsed_seconds']);durations[name]+=duration
        unfinished.append(dict(phase=name,billed_seconds=duration,attempted_requests=event.get('attempted_requests')))
    result=dict(phase_wall_seconds=dict(durations),accounted_wall_seconds=end,
        import_setup_snapshot_and_other_seconds=max(0.,end-sum(durations.values())),
        actual_total_process_wall_seconds=supervisor['wall_seconds'],
        deadline_stop_overhead_seconds=max(0.,supervisor['wall_seconds']-end),
        attempted_logical_requests=attempted,returned_requests=returned,completed_new_teacher_calls=new,completed_cache_hits=hits,
        synchronized_backend_forward_wall_seconds=forward,unfinished_phases=unfinished,
        in_flight_teacher_completion_unknown=any(r['phase']=='teacher_request' for r in unfinished),
        note='phase wall includes CPU/IO and GPU work, not GPU utilization integral; unfinished work billed without prediction credit')
    atomic_json(folder/'cost_ledger.json',result);return result


def stage2():
    root=OUTPUT/'stage2';meta=metadata();results={}
    for method in ('MINUS','MINUS_RECALC_ONLY','SLOW_REPLAY_MATCHED'):
        results[method]=evaluate(root/method,meta)
    for folder in sorted((root/'countmatched').iterdir()):results[folder.name]=evaluate(folder,meta)
    comparisons={}
    for a,b in [('MINUS_RECALC_ONLY','MINUS'),('SLOW_REPLAY_MATCHED','MINUS_RECALC_ONLY'),('SLOW_REPLAY_MATCHED','MINUS'),
        ('LORA_RANK__RECALC_COUNT','FROZEN_RANK__RECALC_COUNT'),('LORA_RANK__LORA_COUNT','FROZEN_RANK__LORA_COUNT')]:
        comparisons[a+'_vs_'+b]=bootstrap(root/a if (root/a).exists() else root/'countmatched'/a,root/b if (root/b).exists() else root/'countmatched'/b)
    source=HISTORY/'redesign_v8p/active/C4/MINUS_SLOW/rho_0.3'
    original=json.loads((source/'predictions_seal.json').read_text());ours=verify_predictions(root/'MINUS')
    index={(e['video'],e['expression']):e for e in original['episodes']}
    differences=[(e['video'],e['expression']) for e in ours['episodes'] if e['prediction_sha256']!=index[e['video'],e['expression']]['prediction_sha256']]
    updates={};events={};costs={};mechanism={}
    for method in ('MINUS','MINUS_RECALC_ONLY','SLOW_REPLAY_MATCHED'):
        receipts=[json.loads(p.read_text()) for p in (root/method/'videos').glob('*/receipt.json')]
        updates[method]=sum(r['slow_updates'] for r in receipts)
        events[method]=sum(len([x for x in r['initial_and_recalc_score_records'] if x['kind']=='FIXED32']) for r in receipts)
        costs[method]=dict(process_wall_seconds=ours.get('wall_seconds') if method=='MINUS' else verify_predictions(root/method)['wall_seconds'],
            LoRA_wall_seconds=sum(rr['slow']['GPU_seconds'] for v in receipts for rr in v['rounds']),
            initial_and_recalc_wall_seconds=sum(x['GPU_seconds'] for v in receipts for x in v['initial_and_recalc_score_records']),
            matched_schedule_wait_seconds=sum(rr.get('matched_schedule_wait_seconds',0.) for v in receipts for rr in v['rounds']),
            new_teacher_calls=0,replayed_answers=sum(v['questions'] for v in receipts))
        slopes=[]
        for p in sorted((root/method/'videos').glob('*/belief.npz')):
            with np.load(p) as b:slopes.append(dict(video=p.parent.name,slope=float(b['slope']),mean_fraction=float(b['fractions'].mean())))
        mechanism[method]=dict(per_video_fast_state=slopes,zero_slope_videos=sum(abs(r['slope'])<1e-12 for r in slopes),
            fast_identity_MAP_rounds=sum(r['fast']['identity']['status']=='MAP_UPDATED' for v in receipts for r in v['rounds']),
            optimized_positive_samples=sum(r['slow'].get('actual_optimized_positive_samples',0) for v in receipts for r in v['rounds']),
            optimized_negative_samples=sum(r['slow'].get('actual_optimized_negative_samples',0) for v in receipts for r in v['rounds']),
            AMP_retry_records=sum(sum(not step.get('optimizer_updated',True) for step in r['slow'].get('records',[])) for v in receipts for r in v['rounds']))
    assert events['MINUS_RECALC_ONLY']==events['SLOW_REPLAY_MATCHED'] and updates['MINUS_RECALC_ONLY']==0
    from scripts.active_eval import official_stats
    historical={}
    for method in ('FULL','SLOW_REPLAY'):
        p=HISTORY/'redesign_v8p/active/C4'/method/'rho_0.3'
        r=json.loads((p/'official_completion.json').read_text())
        historical[method]=dict(HOTA_unrounded=official_stats(p)['HOTA_unrounded'],metrics=r['metrics'],
            role='A800 historical descriptive reference, not a matched A100 causal control',path=str(p))
    result=dict(metrics=results,comparisons=comparisons,actual_optimizer_updates=updates,recalc_events=events,costs=costs,
        frozen_cache_runtime={method:cache_runtime_profile(method) for method in ('SLOW_REPLAY_MATCHED','MINUS_RECALC_ONLY')},
        mechanism_diagnostics=mechanism,
        historical_references=historical,
        MINUS_original_prediction_bytes_equal=not differences,MINUS_A100_vs_A800_different_sequences=differences,
        preregistration_sha256=file_sha(root/'preregistration.json'),
        interpretation='Fixed Q&A and matched recalc schedule; LoRA total effect includes its downstream fast fit. CountMatched tests raw score ranking at fixed predicted counts.',
        cost_note='Causal arms wait for canonical input/schedule producers; their process wall is not a self-contained deployment benchmark. Use stage3 for actual cold end-to-end cost.')
    atomic_json(root/'analysis.json',result)
    return result


def stage3():
    from scripts.a100_wallclock import slice_metadata,GRID,METHODS
    root=OUTPUT/'stage3';rows=[]
    for scope in ('short','full'):
        meta=slice_metadata(scope)
        for repeat in (range(3) if scope=='short' else range(1)):
            for method in METHODS:
                folder=root/scope/f'repeat_{repeat}'/method
                s=json.loads((folder/'supervisor_receipt.json').read_text())
                worker=json.loads((folder/'worker_receipt.json').read_text()) if (folder/'worker_receipt.json').exists() else None
                ledger=cost_ledger(folder,s)
                snaps=[json.loads(line) for line in (folder/'snapshots.jsonl').read_text().splitlines()]
                visibility={r['index']:r['parent_observed_seconds'] for r in (json.loads(line) for line in (folder/'availability.jsonl').read_text().splitlines())}
                for snap in snaps:
                    snap['credit_seconds']=max(snap['elapsed_seconds'],visibility.get(snap['index'],float('inf')))
                grid=GRID if scope=='short' else [s['wall_seconds']]
                for budget in grid:
                    available=[r for r in snaps if r['credit_seconds']<=budget]
                    latest=available[-1] if available else None
                    mask=np.load(latest['path'])['mask'] if latest else np.zeros(meta['candidate_query_rows'],dtype=bool)
                    target=folder/'at_budget'/str(int(budget))
                    seal_mask(target,meta,mask,target='SportsMOT',model='Dance-best',method=method,GT_open_attempts=0,
                        extra=dict(budget_seconds=budget,scope=scope,repeat=repeat,snapshot=latest,
                            all_unprocessed_samples_retained=True,cold_start_included=True,worker_receipt=worker,
                            cost_ledger=ledger,
                            real_total_process_wall_seconds=s['wall_seconds'],physical_GPU=s['physical_GPU']))
                    ev=evaluate(target,meta,scope)
                    rows.append(dict(scope=scope,method=method,repeat=repeat,budget_seconds=budget,HOTA=ev['HOTA_unrounded'],
                        DetA=ev['metrics']['DetA'],AssA=ev['metrics']['AssA'],IDF1=ev['metrics']['IDF1'],
                        prediction_rows=int(mask.sum()),teacher_calls=latest['actual_calls'] if latest else 0,
                        latest_commit_seconds=latest['elapsed_seconds'] if latest else None,
                        total_process_wall_seconds=s['wall_seconds'],physical_GPU=s['physical_GPU'],folder=str(target)))
    with (root/'cost_utility.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    atomic_json(root/'analysis.json',dict(rows=rows,scope_note='short two-video slice, three cold-process repeats; full45-video actual cost once per method',
        budget_accounting='model/process cold, OS cache shared; hard cutoff; committed output only',
        no_unprocessed_samples_removed=True,preregistration_sha256=file_sha(root/'preregistration.json')))
    plot_short(rows)
    full_curves()
    return rows


def plot_short(rows):
    from scripts.a100_wallclock import METHODS
    root=OUTPUT/'stage3'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.2,4.5));colors=dict(MINUS_v8P='#2369bd',YT_PRIOR='#e48a22',UNIFORM_K='#2c9a67')
    for method in METHODS:
        data=[r for r in rows if r['scope']=='short' and r['method']==method]
        x=sorted({r['budget_seconds'] for r in data});means=[np.mean([r['HOTA'] for r in data if r['budget_seconds']==t]) for t in x]
        low=[min(r['HOTA'] for r in data if r['budget_seconds']==t) for t in x];high=[max(r['HOTA'] for r in data if r['budget_seconds']==t) for t in x]
        ax.step(x,means,where='post',label=method,color=colors[method],linewidth=2,marker='o',markersize=3)
        ax.fill_between(x,low,high,step='post',color=colors[method],alpha=.13)
    title='C4 fixed short slice: complete cold startup included'
    if not any(r['teacher_calls'] for r in rows if r['scope']=='short'):
        title+='\nNo teacher feedback credited within 600 seconds'
    ax.set(xlabel='Total process / GPU wall-clock budget (seconds)',ylabel='Official HOTA (%)',xlim=(0,600),title=title)
    ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.text(.5,.015,'Measured budget grid: mean and min-max over three fresh-process repeats.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(root/'cost_utility.png',dpi=220);fig.savefig(root/'cost_utility.pdf');plt.close(fig)


def full_curves():
    from scripts.a100_wallclock import slice_metadata,METHODS
    root=OUTPUT/'stage3';meta=slice_metadata('full');runs={}
    for method in METHODS:
        folder=root/'full/repeat_0'/method;s=load_json(folder/'supervisor_receipt.json')
        snaps=[json.loads(line) for line in (folder/'snapshots.jsonl').read_text().splitlines()]
        visible={r['index']:r['parent_observed_seconds'] for r in (json.loads(line) for line in (folder/'availability.jsonl').read_text().splitlines())}
        for snap in snaps:snap['credit_seconds']=max(snap['elapsed_seconds'],visible[snap['index']])
        runs[method]=dict(folder=folder,supervisor=s,snaps=snaps)
    grid=sorted(set([0.,600.,1200.,2400.,4800.,9600.,19200.]+[r['supervisor']['wall_seconds'] for r in runs.values()]))
    rows=[];evaluated={}
    for method,run in runs.items():
        folder=run['folder'];end=run['supervisor']['wall_seconds']
        for budget in grid:
            available=[s for s in run['snaps'] if s['credit_seconds']<=budget];last=available[-1] if available else None
            if budget>=end:
                target=folder/'at_budget'/str(int(end));ev=load_json(target/'official_completion.json')
            else:
                key=(method,last['index']) if last else ('EMPTY',None)
                if key in evaluated:ev,target=evaluated[key]
                else:
                    mask=np.load(last['path'])['mask'] if last else np.zeros(meta['candidate_query_rows'],dtype=bool)
                    target=folder/'full_curve'/f'{budget:.6f}'
                    seal_mask(target,meta,mask,target='SportsMOT',model='Dance-best',method=method,GT_open_attempts=0,
                        extra=dict(role='supplementary full C4 anytime curve',budget_seconds=budget,snapshot=last,
                            all_unprocessed_samples_retained=True,cold_start_included=True))
                    ev=evaluate(target,meta,'full');evaluated[key]=(ev,target)
            rows.append(dict(method=method,budget_seconds=budget,HOTA=ev['HOTA_unrounded'],teacher_calls=last['actual_calls'] if last else 0,
                task_finished_by_budget=budget>=end,actual_total_wall_seconds=end,folder=str(target)))
    with (root/'full_cost_utility.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    atomic_json(root/'full_curve_analysis.json',dict(rows=rows,time_grid_rule='predeclared powers of2 from600 plus all actual completion times',
        GPU_experiments_added=0,ground_truth_used_for_time_grid=False,all_unprocessed_samples_retained=True))
    plot_full(rows)
    return rows


def plot_full(rows):
    from scripts.a100_wallclock import METHODS
    root=OUTPUT/'stage3'
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.2,4.5))
    for method in METHODS:
        part=[r for r in rows if r['method']==method]
        line,=ax.step([r['budget_seconds'] for r in part],[r['HOTA'] for r in part],where='post',label=method,linewidth=2,marker='o',markersize=2.5)
        end=next(r for r in part if r['task_finished_by_budget'])
        ax.plot(end['actual_total_wall_seconds'],end['HOTA'],'o',color=line.get_color(),markersize=5)
    ax.set(xlabel='Total process / GPU wall-clock budget (seconds)',ylabel='Official full C4 HOTA (%)',
        title='Full C4: cold startup and unprocessed samples included')
    ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.text(.5,.015,'Measured budgets; single runs, shared NAS and OS cache. Large dots = completion.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(root/'full_cost_utility.png',dpi=220);fig.savefig(root/'full_cost_utility.pdf');plt.close(fig)


def load_json(path):return json.loads(Path(path).read_text())


def preflight():
    amendment=OUTPUT/'stage2/runtime_cache_amendment.json'
    if amendment.exists():
        proofs=[json.loads(p.read_text()) for p in (OUTPUT/'stage2/canonical').glob('*/resume_numerical_equivalence.json')]
        assert proofs and all(p['all_arrays_bitwise_equal'] for p in proofs), 'Validate512MiB-to8192MiB frozen-cache equivalence before any GT evaluation'
    # The very first worker was launched before the within-round pool was
    # moved outside the chunk loop. Retain it only if that code was never
    # exercised twice within a round; otherwise require a clean rerun BEFORE GT.
    folder=OUTPUT/'stage3/short/repeat_0/MINUS_v8P'
    runtime=json.loads((folder/'runtime_metadata.json').read_text())
    if runtime.get('acquisition_version')=='fixed_pool_per_registered_round':return
    counts=defaultdict(int);p=folder/'answer_batches.jsonl'
    if p.exists():
        for line in p.read_text().splitlines():
            r=json.loads(line);counts[r['video'],r['round']]+=1
    if any(n>1 for n in counts.values()):
        atomic_json(OUTPUT/'stage3/NEEDS_FIRST_TRIAL_RERUN.json',dict(reason='Initial worker refreshed the question pool within a round; registration requires one fixed pool per round.',
            affected_folder=str(folder),observed_batch_counts={str(k):v for k,v in counts.items()},GT_not_read=True))
        raise RuntimeError('Rerun initial MINUS short trial with corrected fixed per-round pool before GT evaluation')
    atomic_json(folder/'implementation_equivalence_audit.json',dict(status='PASS_UNEXERCISED_LATENT_POOL_DIFFERENCE',
        batch_counts={str(k):v for k,v in counts.items()},reason='No second completed batch within any round; credited predictions use the same registered pool.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['countmatched','stage2','stage3','all'],default='all');a=p.parse_args()
    if a.phase in ('stage2','stage3','all'):preflight()
    if a.phase in ('countmatched','all'):countmatched()
    if a.phase in ('stage2','all'):stage2()
    if a.phase in ('stage3','all'):stage3()
