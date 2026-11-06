"""Independent post-seal CPU GT diagnostics; never imported by estimators."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.active_v9_inputs import PUBLIC,SOURCE,DATA,PAIRS,write_csv
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def read_sealed(name,pair):
    seal=json.loads((DATA/name/pair/'estimates_seal.json').read_text())
    assert file_sha(seal['path'])==seal['sha256'] and seal['GT_open_attempts']==0
    return json.loads(Path(seal['path']).read_text())


def diagnostics():
    # Fail before opening any GT unless every E0 direction is completely sealed.
    e0={p:read_sealed('E0',p) for p in PAIRS}
    cached=PUBLIC/'presence_truth.csv'
    if (PUBLIC/'D1_D2_diagnostics.json').exists():
        with cached.open() as handle:
            return {(r['pair'],r['video'],r['query']):r['GT_present']=='True' for r in csv.DictReader(handle)}
    from scripts.active_eval import offline_labels
    cache={};d1=[];d2=[];d1_stats=[];d2_stats=[];truth_by_query={}
    for pair,(_,target) in PAIRS.items():
        if target not in cache:cache[target]=offline_labels(target)
        meta,truth,_=cache[target]
        original={r['video']:r for r in e0[pair]['records']}
        pair_rows=[]
        for video,bank in sorted(meta['videos'].items()):
            directory=SOURCE/pair/'MINUS_SLOW/rho_0.3/videos'/video
            answers=json.loads((directory/'questions.json').read_text())
            with np.load(directory/'belief.npz') as b:
                tracks=b['tracks'];names=b['queries'].tolist();strata=b['prior_strata']
            obs=np.asarray(bank['observations']);ti=np.searchsorted(tracks,obs[:,1]);counts=np.bincount(ti,minlength=len(tracks))
            eligible=counts>=3;track_lookup={int(t):i for i,t in enumerate(tracks)}
            old_queries={q['query']:q for q in original[video]['queries']}
            for qi,name in enumerate(names):
                label=truth[video,name];present=bool(label['counts'].sum())
                assert len(label['labels'])==len(obs)
                positive=np.bincount(ti,weights=label['labels'],minlength=len(tracks))>=.5*counts
                truth_by_query[pair,video,name]=present
                if not present:continue
                for group in ('ALL',0,1,2):
                    mask=eligible if group=='ALL' else eligible & (strata[qi]==group)
                    n=int(mask.sum());pos=int((positive&mask).sum())
                    if n:d1.append(dict(pair=pair,target=target,video=video,query=name,stratum=group,
                        tracks=n,GT_positive_tracks=pos,pi_q=pos/n))
                aa=[a for a in answers if a['query_id']==name]
                asked={track_lookup[a['track_id']] for a in aa}
                pos_answers=[a for a in aa if positive[track_lookup[a['track_id']]]]
                neg_answers=[a for a in aa if not positive[track_lookup[a['track_id']]]]
                row=dict(pair=pair,target=target,video=video,query=name,
                    GT_positive_tracks=int((positive&eligible).sum()),eligible_tracks=int(eligible.sum()),
                    unique_asked_tracks=len(asked),unique_asked_GT_positive=sum(bool(positive[t]) for t in asked),
                    unique_asked_GT_negative=sum(not positive[t] for t in asked),
                    positive_track_answers=len(pos_answers),negative_track_answers=len(neg_answers),
                    positive_track_Yes=sum(a['yes_probability']>=.5 for a in pos_answers),
                    negative_track_Yes=sum(a['yes_probability']>=.5 for a in neg_answers),
                    positive_track_Yes_rate=sum(a['yes_probability']>=.5 for a in pos_answers)/len(pos_answers) if pos_answers else None,
                    negative_track_Yes_rate=sum(a['yes_probability']>=.5 for a in neg_answers)/len(neg_answers) if neg_answers else None,
                    all_answers_No=bool(aa) and not any(a['yes_probability']>=.5 for a in aa),
                    no_answers=not bool(aa),v8_output_false_absent=old_queries[name]['output_absent'],
                    v8_evidence_false_absent=old_queries[name]['evidence_absent'])
                d2.append(row);pair_rows.append(row)
        for group in ('ALL',0,1,2):
            rows=[r for r in d1 if r['pair']==pair and r['stratum']==group]
            rates=np.array([r['pi_q'] for r in rows]);ns=np.array([r['tracks'] for r in rows])
            null_pi=sum(r['GT_positive_tracks'] for r in rows)/ns.sum()
            variance=float(np.var(rates,ddof=1));binomial=float(np.mean(null_pi*(1-null_pi)/ns))
            d1_stats.append(dict(pair=pair,target=target,stratum=group,queries=len(rows),
                pi_mean=float(rates.mean()),pooled_pi=null_pi,variance=variance,
                quantiles=np.quantile(rates,[0,.1,.25,.5,.75,.9,1]).tolist(),
                homogeneous_binomial_variance=binomial,overdispersion=variance/binomial if binomial else None,
                null='shared pi, per-query actual track counts; descriptive, not independent-track significance'))
        rejected=[r for r in pair_rows if r['v8_output_false_absent']]
        def rate(rows,n,d):
            denominator=sum(r[d] for r in rows);return sum(r[n] for r in rows)/denominator if denominator else None
        d2_stats.append(dict(pair=pair,target=target,GT_present_queries=len(pair_rows),
            positive_track_Yes_rate=rate(pair_rows,'positive_track_Yes','positive_track_answers'),
            negative_track_Yes_rate=rate(pair_rows,'negative_track_Yes','negative_track_answers'),
            all_No_queries=sum(r['all_answers_No'] for r in pair_rows),no_answer_queries=sum(r['no_answers'] for r in pair_rows),
            v8_output_false_absent=len(rejected),rejected_with_zero_GT_positive_asked=sum(r['unique_asked_GT_positive']==0 for r in rejected),
            rejected_with_GT_positive_asked=sum(r['unique_asked_GT_positive']>0 for r in rejected),
            rejected_unique_GT_positive_tracks_asked=sum(r['unique_asked_GT_positive'] for r in rejected),
            rejected_unique_GT_negative_tracks_asked=sum(r['unique_asked_GT_negative'] for r in rejected)))
    write_csv(PUBLIC/'D1_query_matching_rates.csv',d1);write_csv(PUBLIC/'D1_overdispersion.csv',d1_stats)
    write_csv(PUBLIC/'D2_query_teacher_answers.csv',d2);write_csv(PUBLIC/'D2_teacher_summary.csv',d2_stats)
    write_csv(cached,[dict(pair=p,video=v,query=q,GT_present=y) for (p,v,q),y in truth_by_query.items()])
    atomic_json(PUBLIC/'D1_D2_diagnostics.json',dict(status='POST_SEAL_GT_DIAGNOSTICS',
        D1=d1_stats,D2=d2_stats,E0_seals=[str(DATA/'E0'/p/'estimates_seal.json') for p in PAIRS],
        GT_used_only_in_this_process=True,teacher_or_GPU_calls=0))
    return truth_by_query


def evaluate_estimators(truth):
    manifest=json.loads((PUBLIC/'manifest.json').read_text());rows=[]
    for name in ('E0','E1','E3'):
        for pair in PAIRS:
            if not (DATA/name/pair/'estimates_seal.json').exists():continue
            data=read_sealed(name,pair);qq=[(r['video'],q) for r in data['records'] for q in r['queries']]
            present=sum(truth[pair,v,q['query']] for v,q in qq);empty=len(qq)-present
            count=lambda condition:sum(condition(q,truth[pair,v,q['query']]) for v,q in qq)
            false=count(lambda q,y:y and q['output_absent']);correct=count(lambda q,y:not y and q['output_absent'])
            final=data['final'];error=abs(final['rho_hat']-present/len(qq))
            false_pass=false<=manifest['false_absent_max'][pair]
            empty_pass=correct>=manifest['correct_empty_min'][pair] if pair in manifest['correct_empty_min'] else True
            prior_correct=count(lambda q,y:not y and q['decision_source']=='PRIOR_BUDGET_EXHAUSTED')
            rows.append(dict(estimator=name,pair=pair,present=present,empty=empty,
                output_false_absent=false,output_correct_empty=correct,
                evidence_false_absent=count(lambda q,y:y and q['evidence_absent']),
                evidence_correct_empty=count(lambda q,y:not y and q['evidence_absent']),
                low_tentative=count(lambda q,y:q['low_tentative']),
                PRIOR_BUDGET_EXHAUSTED=count(lambda q,y:q['decision_source']=='PRIOR_BUDGET_EXHAUSTED'),
                prior_budget_GT_present=count(lambda q,y:y and q['decision_source']=='PRIOR_BUDGET_EXHAUSTED'),
                prior_budget_correct_empty=prior_correct,
                correct_empty_majority_from_prior_budget=bool(correct and prior_correct>correct/2),
                rho_hat=final['rho_hat'],rho_GT=present/len(qq),rho_error=error,
                pi_or_mu=json.dumps(final['pi_strata']),tau=final.get('tau'),rho_laplace_std=final.get('rho_laplace_std'),
                false_absent_gate=false_pass,correct_empty_gate=empty_pass,rho_gate=error<=.15,
                ALL_GATES_PASS=false_pass and empty_pass and error<=.15,
                uncapped_output_disagreements=count(lambda q,y:q['uncapped_output_absent'] is not None and q['output_absent']!=q['uncapped_output_absent']),
                uncapped_evidence_disagreements=count(lambda q,y:q['uncapped_evidence_absent'] is not None and q['evidence_absent']!=q['uncapped_evidence_absent'])))
    write_csv(PUBLIC/'estimator_gates.csv',rows)
    e1=[r for r in rows if r['estimator']=='E1']
    if len(e1)<6:selection=dict(status='PENDING',selected=None)
    elif all(r['ALL_GATES_PASS'] for r in e1):selection=dict(status='PASS',selected='E1')
    else:selection=dict(status='STOP_NO_ESTIMATOR_PASSED',selected=None,E2='NOT_APPLICABLE')
    atomic_json(PUBLIC/'selection.json',dict(selection,rows=rows,official_val_is_development_not_blind=True))
    return selection,rows


if __name__=='__main__':
    truth=diagnostics();selection,rows=evaluate_estimators(truth)
    print(json.dumps(dict(selection,completed_direction_estimates=len(rows),GPU_calls=0,teacher_calls=0)),flush=True)
