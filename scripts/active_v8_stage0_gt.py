"""Independent CPU-only GT diagnostics after the stage-0 GT-free seal."""
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import expit,logit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.presence.predictions import verify_predictions
from scripts.active_run import PAIRS

PUBLIC=ROOT/'results/presence_active/redesign_v8/amendments_A2_A3'


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    names=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=names);writer.writeheader();writer.writerows(rows)


def prior_pi_counterfactual(query_index,questions,original_presence,rounds,initial_rel,pi_by_stratum,weak):
    """Replace only pi in old v7's per-answer presence rule, no new answers."""
    rho=original_presence['before']['rho_hat'];total=logit(rho)+weak
    probability=float(expit(total));stopped=probability<.1
    rel=initial_rel
    for r in range(1,5):
        for a in (x for x in questions if x['round']==r and x['_qi']==query_index and x['kind']=='PRESENCE'):
            s,c=rel[a['effective_view']];pi=pi_by_stratum[a['prior_stratum']]
            yes=a['yes_probability']>=.5
            under=pi*s+(1-pi)*(1-c) if yes else pi*(1-s)+(1-pi)*c
            total+=float(np.clip(np.log(under)-np.log(1-c if yes else c),-2,2))
            if not stopped and expit(total)<.1:stopped=True;probability=float(expit(total))
        rel={v:(x['sensitivity'],x['specificity']) for v,x in rounds[r-1]['ds']['reliability'].items()}
        # DS refits reevaluate all observed presence answers with current rel.
        total=logit(rho)+weak
        for a in (x for x in questions if x['round']<=r and x['_qi']==query_index and x['kind']=='PRESENCE'):
            s,c=rel[a['effective_view']];pi=pi_by_stratum[a['prior_stratum']];yes=a['yes_probability']>=.5
            under=pi*s+(1-pi)*(1-c) if yes else pi*(1-s)+(1-pi)*c
            total+=float(np.clip(np.log(under)-np.log(1-c if yes else c),-2,2))
        if not stopped and expit(total)<.1:stopped=True;probability=float(expit(total))
    return stopped,probability if stopped else float(expit(total))


def main():
    seal=json.loads((PUBLIC/'stage0_gt_free_seal.json').read_text())
    assert seal['status']=='SEALED_BEFORE_GT' and file_sha(seal['path'])==seal['sha256']
    replay=json.loads(Path(seal['path']).read_text());assert replay['status']=='PASS'
    for item in seal['inputs']:
        assert file_sha(item['predictions_seal'])==item['sha256']
        verify_predictions(Path(item['predictions_seal']).parent)
    # First numerical GT access is below, in this separate process only.
    from scripts.active_eval import offline_labels
    truth_cache={};estimator=[];decisions=[];details=[];kit=[];reliability=[];streams=[]
    oldroot=ROOT/'results/presence_active/redesign_v7'
    initial={v:(.8,.8) for v in ('CROP','CLIP','FRAME_BOX')}
    for pair,(model,target) in PAIRS.items():
        if target not in truth_cache:truth_cache[target]=offline_labels(target)
        meta,truth,by_video=truth_cache[target]
        entries=[x for x in replay['records'] if x['pair']==pair]
        totals=dict(present=0,empty=0,old_false_absent=0,old_true_absent=0,
            v8_false_absent=0,v8_true_absent=0,v8_false_absent_output=0,v8_true_absent_output=0,
            low_tentative=0,low_tentative_present=0)
        numerator=np.zeros(3,int);denominator=np.zeros(3,int)
        asked_num=np.zeros(3,int);asked_den=np.zeros(3,int)
        prepared=[];view_counts={v:Counter() for v in initial};last_ds=None
        for entry in entries:
            video=entry['video'];directory=Path(entry['original_directory'])
            assert file_sha(entry['counterfactual_path'])==entry['counterfactual_sha256']
            assert file_sha(directory/'questions.json')==entry['questions_sha256']
            assert file_sha(directory/'belief.npz')==entry['belief_sha256']
            assert file_sha(directory/'receipt.json')==entry['receipt_sha256']
            original=json.loads((directory/'presence.json').read_text())
            receipt=json.loads((directory/'receipt.json').read_text())
            questions=json.loads((directory/'questions.json').read_text())
            with np.load(entry['counterfactual_path']) as cf:
                queries=cf['queries'].tolist();tracks=cf['tracks'].tolist();strata=cf['prior_strata'].copy()
                qid={q:i for i,q in enumerate(queries)};tid={t:i for i,t in enumerate(tracks)}
                for a in questions:a['_qi']=qid[a['query_id']];a['_ti']=tid[a['track_id']]
                observations=np.asarray(meta['videos'][video]['observations'])
                assert np.array_equal(observations,np.asarray(by_video[video]['observations']))
                track_positions={ti:np.flatnonzero(observations[:,1]==track) for ti,track in enumerate(tracks)}
                Y=np.zeros_like(strata,dtype=bool);is_present=np.zeros(len(queries),bool)
                asked={(a['_qi'],a['_ti']) for a in questions}
                for q,query in enumerate(queries):
                    t=truth[video,query];is_present[q]=bool(t['counts'].sum())
                    for ti,ids in track_positions.items():
                        Y[q,ti]=bool(np.mean(t['labels'][ids])>=.5)
                        k=int(strata[q,ti])
                        if k<0 or not is_present[q]:continue
                        denominator[k]+=1;numerator[k]+=Y[q,ti]
                        if (q,ti) in asked:asked_den[k]+=1;asked_num[k]+=Y[q,ti]
                    empty=not is_present[q];old=bool(cf['old_absent'][q]);new=bool(cf['absent'][q]);out=bool(cf['output_absent'][q])
                    totals['present' if is_present[q] else 'empty']+=1
                    totals['old_false_absent' if is_present[q] else 'old_true_absent']+=int(old)
                    totals['v8_false_absent' if is_present[q] else 'v8_true_absent']+=int(new)
                    totals['v8_false_absent_output' if is_present[q] else 'v8_true_absent_output']+=int(out)
                    totals['low_tentative']+=int(cf['low_tentative'][q])
                    totals['low_tentative_present']+=int(cf['low_tentative'][q] and is_present[q])
                    decisions.append(dict(pair=pair,video=video,query=query,GT_present=bool(is_present[q]),
                        old_absent=old,v8_absent=new,v8_output_absent=out,
                        low_tentative=bool(cf['low_tentative'][q]),probability=float(cf['presence_probability'][q]),
                        decision_source='EVIDENCE' if new else 'PRIOR_BUDGET_EXHAUSTED' if out else 'NOT_REJECTED'))
                for a in questions:
                    target_positive=bool(Y[a['_qi'],a['_ti']]);yes=a['yes_probability']>=.5
                    view_counts[a['effective_view']][('TP' if yes else 'FN') if target_positive else ('FP' if yes else 'TN')]+=1
                prepared.append(dict(entry=entry,queries=queries,tracks=tracks,strata=strata,Y=Y,
                    is_present=is_present,questions=questions,original=original,receipt=receipt,
                    initial_rel=initial,weak=cf['weak_presence_logits'].copy(),old_absent=cf['old_absent'].copy()))
            initial={v:(x['sensitivity'],x['specificity']) for v,x in receipt['rounds'][-1]['ds']['reliability'].items()}
            last_ds=initial
            streams.append(dict(pair=pair,video=video,video_index=entry['video_index'],
                rho_hat=entry['after']['rho_hat'],pi_0=entry['after']['pi_strata'][0],
                pi_1=entry['after']['pi_strata'][1],pi_2=entry['after']['pi_strata'][2],
                iterations=entry['after']['iterations'],converged=entry['after']['converged'],
                objective_trace_monotone=entry['after']['objective_trace_monotone'],
                contraction_rate=entry['after']['contraction_rate']))
        gt_pi=np.divide(numerator,denominator,out=np.full(3,np.nan),where=denominator>0)
        gt_asked=np.divide(asked_num,asked_den,out=np.full(3,np.nan),where=asked_den>0)
        final=entries[-1]['after']
        oldfinal=json.loads((Path(entries[-1]['original_directory'])/'presence.json').read_text())['after']
        for k in range(3):
            estimator.append(dict(pair=pair,target=target,stratum=k,v7_rho=oldfinal['rho_hat'],
                v7_global_pi=oldfinal['pi_hat'],v8_rho=final['rho_hat'],v8_pi=final['pi_strata'][k],
                GT_rho=totals['present']/(totals['present']+totals['empty']),
                GT_pi_answered=float(gt_asked[k]) if asked_den[k] else None,
                GT_pi_all_eligible=float(gt_pi[k]) if denominator[k] else None,
                GT_present_answered_tracks=int(asked_den[k]),GT_present_all_tracks=int(denominator[k]),
                answered_track_pairs=final['tracks_per_stratum'][k],
                identification_status=final['identification_status'][k],
                presence_likelihood_evaluations=final['presence_likelihood_evaluations_per_stratum'][k]))
        flipped={name:0 for name in ('GT_PI_ANSWERED','GT_PI_ALL_ELIGIBLE')}
        if pair in ('C2','C5'):
            for item in prepared:
                for q in np.flatnonzero(item['is_present'] & item['old_absent']):
                    outcome={}
                    for name,values in [('GT_PI_ANSWERED',gt_asked),('GT_PI_ALL_ELIGIBLE',gt_pi)]:
                        if np.isnan(values).any():raise RuntimeError('Registered KITTI pi counterfactual has undefined GT stratum')
                        stopped,prob=prior_pi_counterfactual(q,item['questions'],item['original'],
                            item['receipt']['rounds'],item['initial_rel'],values,item['weak'][q])
                        flipped[name]+=int(not stopped);outcome[name]=prob
                    for a in (x for x in item['questions'] if x['_qi']==q):
                        k=a['prior_stratum'];r=a['round'];view=a['effective_view'];yes=a['yes_probability']>=.5
                        rel=item['initial_rel'] if r==1 else {v:(x['sensitivity'],x['specificity']) for v,x in item['receipt']['rounds'][r-2]['ds']['reliability'].items()}
                        s,c=rel[view];pi=item['original']['before']['pi_hat']
                        raw=np.log(pi*s+(1-pi)*(1-c) if yes else pi*(1-s)+(1-pi)*c)-np.log(1-c if yes else c)
                        details.append(dict(pair=pair,video=item['entry']['video'],query=item['queries'][q],
                            track_id=a['track_id'],frame_id=a['frame_id'],round=r,kind=a['kind'],stratum=k,
                            view=view,yes=yes,GT_track_positive=bool(item['Y'][q,a['_ti']]),
                            original_pi=pi,EM_pi_video_start=item['entry']['before']['pi_strata'][k],
                            GT_pi_answered=float(gt_asked[k]),GT_pi_all_eligible=float(gt_pi[k]),
                            sensitivity=s,specificity=c,old_presence_contribution_used=a['kind']=='PRESENCE',
                            old_answer_LLR_unclipped=float(raw) if a['kind']=='PRESENCE' else None,
                            old_answer_LLR_clipped=float(np.clip(raw,-2,2)) if a['kind']=='PRESENCE' else None,
                            **{name+'_probability':p for name,p in outcome.items()}))
            kit.append(dict(pair=pair,original_false_absent=totals['old_false_absent'],
                            flipped_with_GT_pi_answered=flipped['GT_PI_ANSWERED'],
                            flipped_with_GT_pi_all_eligible=flipped['GT_PI_ALL_ELIGIBLE'],
                            approximate_fixed_rho_DS_weak_replay=True,not_deployable=True))
        for view,count in view_counts.items():
            specificity=count['TN']/(count['TN']+count['FP']) if count['TN']+count['FP'] else None
            reliability.append(dict(pair=pair,view=view,DS_specificity=last_ds[view][1],
                GT_specificity=specificity,DS_minus_GT=None if specificity is None else last_ds[view][1]-specificity,
                TP=count['TP'],FN=count['FN'],FP=count['FP'],TN=count['TN'],
                label='GT track majority IoU>=0.5; asked answer rows, not unbiased population'))
        atomic_json(PUBLIC/f'stage0_presence_{pair}.json',dict(pair=pair,**totals,not_RMOT_evaluation=True))
        initial={v:(.8,.8) for v in initial}
    write_csv(PUBLIC/'stage0_estimators.csv',estimator)
    write_csv(PUBLIC/'stage0_counterfactual.csv',decisions)
    write_csv(PUBLIC/'stage0_kitti_answer_decomposition.csv',details)
    write_csv(PUBLIC/'stage0_kitti_pi_counterfactual.csv',kit)
    write_csv(PUBLIC/'stage0_reliability_vs_GT.csv',reliability)
    write_csv(PUBLIC/'stage0_em_stream.csv',streams)
    plot(estimator,streams)
    atomic_json(PUBLIC/'stage0_GT_diagnostics.json',dict(status='COMPLETE',
        estimator_rows=len(estimator),query_rows=len(decisions),KITTI_answer_rows=len(details),
        source_gt_free_seal_sha256=file_sha(PUBLIC/'stage0_gt_free_seal.json'),
        source_gt_free_payload_sha256=seal['sha256'],GT_feedback_to_runtime=False,
        all_v7_predictions_verified_before_GT=True,GPU_calls=0,teacher_calls=0,not_new_RMOT_evaluation=True))
    print(json.dumps(dict(event='V8_STAGE0_GT_COMPLETE',estimator_rows=len(estimator),queries=len(decisions),KITTI_answers=len(details))),flush=True)


def plot(estimator,streams):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key in ('rho_hat','pi_0','pi_1','pi_2'):
        fig,axes=plt.subplots(2,3,figsize=(15,7))
        for ax,pair in zip(axes.flat,PAIRS):
            rows=[x for x in streams if x['pair']==pair];truth=[x for x in estimator if x['pair']==pair]
            gt=truth[0]['GT_rho'] if key=='rho_hat' else truth[int(key[-1])]['GT_pi_all_eligible']
            ax.plot([x['video_index']+1 for x in rows],[x[key] for x in rows],label='v8 EM')
            if gt is not None:ax.axhline(gt,ls='--',label='GT diagnostic only')
            old=truth[0]['v7_rho'] if key=='rho_hat' else truth[0]['v7_global_pi']
            ax.axhline(old,ls=':',label='v7 final (global pi)')
            ax.set(title=pair,xlabel='Video ID order',ylabel=key,ylim=(0,1));ax.legend(fontsize=7);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(PUBLIC/(key+'_diagnostic.png'),dpi=150);plt.close(fig)


if __name__=='__main__':main()
