"""Post-queue protocol audit from sealed runtime records; no GT/model calls."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import expit,logit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.active.execution import authorize_active
from flex_uda.presence.runtime import target_metadata,runtime_guard
from flex_uda.presence.predictions import verify_predictions
from scripts.active_run import PAIRS
from scripts.active_v6_prepare import sample32
from flex_uda.active.v7 import balanced_weights,weak_rank_logits


def validate_recheck(question, history, original_question=None):
    """A fixed replay's selection eligibility belongs to its original MINUS state."""
    reference=original_question if original_question is not None else question
    assert reference['kind']=='POSITIVE_RECHECK'
    assert len(history)==1 and history[0]['yes_probability']>=.5
    assert .5<=reference['posterior_probability_before']<.85
    assert question['requested_view']==history[0]['effective_view']
    assert question['frame_id']!=history[0]['frame_id']


def validate(execution):
    attempts=runtime_guard();public=execution.public
    plan=json.loads((public/'registered_jobs.json').read_text());jobs=list(plan['jobs'])
    decisions=json.loads((public/'baseline_budget_decisions.json').read_text());jobs+=decisions['jobs']
    assert file_sha(plan['prereg_path'])==plan['prereg_sha256']
    old6=json.loads((Path(execution.manifest['prior_v6_public_root'])/'artifacts.json').read_text())
    for row in old6['new']:
        assert file_sha(Path(row['path'])/'predictions_seal.json')==row['seal_sha256']
    records=[];meta_cache={};sum_episodes=sum_rows=0
    for job in jobs:
        pair,method=job['pair'],job['method'];target=PAIRS[pair][1]
        folder=execution.active/pair/method/'rho_0.3';seal=verify_predictions(folder)
        if target not in meta_cache:meta_cache[target]=target_metadata(target)
        meta=meta_cache[target];qmap={(q['video_id'],q['expression_id']):q for q in meta['episodes']}
        sealed={(e['video'],e['expression']):e for e in seal['episodes']};assert set(qmap)==set(sealed)
        assert seal['teacher']['GT_open_attempts']==0
        rho_alpha=rho_beta=pi_alpha=pi_beta=1.
        updates=positive=negative=zero_grad=logical=0
        for video,bank in sorted(meta['videos'].items()):
            directory=folder/'videos'/video;receipt=json.loads((directory/'receipt.json').read_text())
            mask=np.load(directory/'mask.npy');assert file_sha(directory/'mask.npy')==receipt['mask_sha256']
            queries=[q for q in meta['episodes'] if q['video_id']==video]
            assert mask.shape==(len(queries),len(bank['observations']))
            for qi,q in enumerate(queries):
                assert Path(sealed[video,q['expression_id']]['prediction_path']).read_text()==''.join(
                    line+'\n' for line,keep in zip(bank['lines'],mask[qi]) if keep)
            if method.startswith('YESTRACK'):
                full=json.loads((execution.active/pair/'FULL/rho_0.3/videos'/video/'receipt.json').read_text())
                assert receipt['actual_calls']==full['questions'];continue
            answers=json.loads((directory/'questions.json').read_text());logical+=len(answers)
            assert len(answers)==receipt['questions']<=receipt['budget']
            frozen=None
            if method=='SLOW_REPLAY':
                source=execution.active/pair/'MINUS_SLOW/rho_0.3/videos'/video/'questions.json'
                frozen=json.loads(source.read_text());assert len(frozen)==len(answers)
                keys=('request_id','frame_id','track_id','query_id','effective_view','yes_probability','yes_logit','no_logit','round','kind')
                assert all(tuple(a[k] for k in keys)==tuple(b[k] for k in keys) for a,b in zip(answers,frozen))
            by_track=defaultdict(list)
            for j,(_,t) in enumerate(bank['observations']):by_track[t].append(j)
            observation_keys=set(map(tuple,bank['observations']))
            with np.load(directory/'belief.npz') as b:
                tracks=b['tracks'].tolist();qid={q:i for i,q in enumerate(b['queries'].tolist())};tid={t:i for i,t in enumerate(tracks)}
                eligible=np.asarray([len(by_track[t])>=3 for t in tracks]);evidence=np.zeros_like(b['evidence']);asked=np.zeros_like(b['asked_mask'])
                history=defaultdict(list);presence_groups=defaultdict(list);per_round=set()
                for ai,a in enumerate(answers):
                    q,t=qid[a['query_id']],tid[a['track_id']];key=(q,t)
                    assert eligible[t] and (a['frame_id'],a['track_id']) in observation_keys
                    assert (q,t,a['round']) not in per_round;per_round.add((q,t,a['round']))
                    assert abs(a['evidence_delta'])<=2 and np.isclose(a['evidence_delta'],np.clip(a['unclipped_evidence_delta'],-2,2))
                    evidence[q,t]+=a['evidence_delta'];asked[q,t]=True
                    if method!='SLOW_REPLAY':assert a['presence_probability_before']>=.1
                    if a['kind']=='POSITIVE_RECHECK':
                        validate_recheck(a,history[key],None if frozen is None else frozen[ai])
                    if a['kind']=='PRESENCE':presence_groups[key].append(a)
                    assert a['prior_stratum']==int(b['prior_strata'][q,t])
                    assert a['identity_used_as_DS_annotator'] is False
                    history[key].append(a)
                assert np.array_equal(asked,b['asked_mask']) and np.allclose(evidence,b['evidence'],atol=1e-12)
                f=b['fractions'];p=np.clip(f,.05,.95);ell=np.clip(np.log(p/(1-p)),-2,2)
                assert ((0<=f)&(f<=1)).all() and np.array_equal(ell,b['base_logits'])
                assert 0<=float(b['slope'])<=1
                expected=np.clip(b['intercept'][:,None]+float(b['slope'])*ell,-2,2)
                assert np.allclose(expected[:,eligible],b['prior'][:,eligible],rtol=0,atol=1e-12)
                posterior=expit(b['prior']+evidence);posterior[b['absent']]=0
                assert np.allclose(posterior,b['posterior'],rtol=0,atol=1e-12)
                pseudo=np.where(posterior>.85,1,np.where(posterior<.15,0,-1)).astype('int8')
                pseudo[~asked]=-1;pseudo[:,~eligible]=-1
                assert np.array_equal(pseudo,b['pseudo_labels'])
                presence=json.loads((directory/'presence.json').read_text());before=presence['before'];after=presence['after']
                assert before['rho_alpha']==rho_alpha and before['rho_beta']==rho_beta
                assert np.isclose(before['pi_alpha'],pi_alpha) and np.isclose(before['pi_beta'],pi_beta)
                rho=rho_alpha/(rho_alpha+rho_beta);pi=pi_alpha/(pi_alpha+pi_beta)
                reliability=receipt['rounds'][-1]['ds']['reliability']
                weak=weak_rank_logits(b['presence_max']);assert np.array_equal(weak,b['weak_presence_logits'])
                for q,row in enumerate(presence['queries']):
                    raw=clipped=0.;count=0;n=0
                    for (qi,_),aa in presence_groups.items():
                        if qi!=q:continue
                        for a in aa:
                            s=reliability[a['effective_view']]['sensitivity'];c=reliability[a['effective_view']]['specificity']
                            yes=a['yes_probability']>=.5
                            under=pi*s+(1-pi)*(1-c) if yes else pi*(1-s)+(1-pi)*c
                            value=np.log(under)-np.log(1-c if yes else c)
                            raw+=value;clipped+=np.clip(value,-2,2);count+=abs(value)>2;n+=1
                    assert np.isclose(raw,row['logLR_unclipped_sum']) and np.isclose(clipped,row['logLR_clipped_sum'])
                    assert count==row['clipped_answers'] and n==row['presence_answers']
                    assert np.isclose(expit(logit(rho)+weak[q]+clipped),row['recomputed_probability'])
                    if row['absent']:assert row['decision_probability']<.1 and not mask[q].any()
                    else:assert np.isclose(row['decision_probability'],row['recomputed_probability'])
                confirmed=int(((b['presence_probability']>.9)&~b['absent']).sum());absent=int(b['absent'].sum())
                assert presence['confirmed_present']==confirmed and presence['confirmed_absent']==absent
                soft={(x['query'],x['track']):x for x in presence['pi_pair_soft_counts']}
                expected_soft={}
                for key,aa in presence_groups.items():
                    q,t=key
                    if b['absent'][q] or b['presence_probability'][q]<=.9:continue
                    log1,log0=np.log(pi),np.log1p(-pi)
                    for a in aa:
                        s=reliability[a['effective_view']]['sensitivity'];c=reliability[a['effective_view']]['specificity'];yes=a['yes_probability']>=.5
                        log1+=np.log(s if yes else 1-s);log0+=np.log(1-c if yes else c)
                    expected_soft[key]=float(np.exp(log1-np.logaddexp(log1,log0)))
                assert set(soft)==set(expected_soft)
                for key,value in expected_soft.items():assert np.isclose(soft[key]['probability'],value,atol=1e-12)
                rho_alpha+=confirmed;rho_beta+=absent
                pi_alpha+=sum(expected_soft.values());pi_beta+=sum(1-v for v in expected_soft.values())
                assert np.isclose(after['pi_alpha'],pi_alpha) and np.isclose(after['pi_beta'],pi_beta)
                assert after['rho_alpha']==rho_alpha and after['rho_beta']==rho_beta
            for rr in receipt['rounds']:
                ds=rr['ds'];assert ds['Beta_prior']==[2.,2.] and not ds['identity_annotator']
                slow=rr['slow'];n=slow['optimizer_updates'];updates+=n
                positive+=slow['actual_optimized_positive_samples'];negative+=slow['actual_optimized_negative_samples']
                if slow['status']!='DISABLED':
                    weights=balanced_weights(slow['training_pool_positive'],slow['training_pool_negative'],exclusive=receipt['exclusive_group'])
                    assert np.isclose(weights['binary_positive_weight'],slow['binary_positive_weight'])
                    assert np.isclose(weights['none_class_weight'],slow['none_class_weight'])
                success=[x for x in slow.get('records',[]) if x.get('optimizer_updated')]
                assert len(success)==n
                assert sum(x['actual_positive_samples'] for x in success)==slow['actual_optimized_positive_samples']
                assert sum(x['actual_negative_samples'] for x in success)==slow['actual_optimized_negative_samples']
                assert slow['actual_optimized_positive_samples']+slow['actual_optimized_negative_samples']==32*n
                for x in success:
                    assert np.isfinite([x['loss'],x['gradient_norm']]).all();zero_grad+=x['gradient_norm']==0
            if method=='MINUS_SLOW':
                assert not (directory/'frame_cache').exists() and receipt['rescore_GPU_seconds']==receipt['slow_GPU_seconds']==0
            else:
                local=json.loads((ROOT/'results/runs/presence_active_20261102/features'/target/PAIRS[pair][0]/'videos'/video/'metadata.json').read_text())
                fixed=sample32(local['frames']);assert receipt['fixed32_frames']==fixed
                cache=json.loads((directory/'frame_cache/cache_receipt.json').read_text())
                assert cache['records'][0]['kind']=='VIDEO_START_ALL' and cache['records'][0]['frames']==local['frames']
                assert len([x for x in cache['records'] if x['kind']=='FIXED32'])==sum(r['slow']['optimizer_updates']>0 for r in receipt['rounds'])
                for x in cache['records']:
                    if x['kind']=='FIXED32':assert x['frames']==fixed
                for name,shape in [('margin.npy',mask.shape),('pcd.npy',mask.shape+(768,))]:
                    value=np.load(directory/'frame_cache'/name,mmap_mode='r');assert value.dtype==np.float32 and value.shape==shape
            if method=='SLOW_REPLAY':
                assert not (folder/'teacher_sessions').exists()
        if not method.startswith('YESTRACK'):assert logical==seal['teacher']['logical_calls']
        records.append(dict(pair=pair,method=method,episodes=len(sealed),prediction_rows=seal['prediction_rows'],
            optimizer_updates=updates,actual_optimized_positive_samples=positive,actual_optimized_negative_samples=negative,
            zero_gradient_updates=int(zero_grad),GT_open_attempts=0,prediction_seal_sha256=file_sha(folder/'predictions_seal.json')))
        sum_episodes+=len(sealed);sum_rows+=seal['prediction_rows']
    assert not attempts
    result=dict(status='V7_AUTHORIZED_TIER_VALIDATED',primary_jobs=len(plan['jobs']),conditional_jobs=len(decisions['jobs']),
        total_episodes=sum_episodes,prediction_rows=sum_rows,methods=records,old_v6_seals_unchanged=True,
        fixed_MOT_rows_exact=True,EB_likelihood_and_clipping_verified=True,EB_online_Beta_and_unclipped_pi_updates_verified=True,
        asked_only_pseudo_labels_and_balanced_loss_weights_verified=True,actual_optimizer_input_accounting_verified=True,
        fixed_replay_questions_and_original_answers_verified=True,MINUS_no_frame_rescore_verified=True,
        full_frame_and_fixed32_coverage_verified=True,GT_open_attempts=0,GPU_or_teacher_called=False,
        v6_C4_C6_and_old80_not_started=True)
    atomic_json(public/'validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='methods'},indent=2))


if __name__=='__main__':
    execution,_=authorize_active();validate(execution)
