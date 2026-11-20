"""GT-free paired-view decisions; independent offline bank analysis is separate."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import expit,logit
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import OUT,PAIRS,FrozenInputs,generation_guard
from flex_uda.active.v6 import ReliabilityV6
from flex_uda.active.belief import evidence_increment
from flex_uda.active.acquisition import route,information_gain,entropy

VIEWS=('CROP','CLIP','FRAME_BOX')


def generate():
    attempts=generation_guard();begin=time.perf_counter()
    samples=json.loads((OUT/'evidence_manifest.json').read_text())['samples']
    completion=json.loads((OUT/'evidence/completion.json').read_text())
    assert completion['manifest_sha256']==file_sha(OUT/'evidence_manifest.json')
    responses={(r['sample_id'],r['requested_view']):r for r in
               map(json.loads,(OUT/'evidence/responses.jsonl').read_text().splitlines())}
    assert len(responses)==3*len(samples)
    ds={pair:ReliabilityV6() for pair in PAIRS}
    for s in samples:
        if s['split']!='calibration':continue
        for v in VIEWS:
            a=responses[s['sample_id'],v]
            ds[s['pair']].add(s['sample_id'],a['effective_view'],a['yes_probability']>=.5)
    for d in ds.values():d.fit()
    priors={}
    for pair,video in sorted({(s['pair'],s['video_id']) for s in samples}):
        fc=FrozenInputs(pair,video);fraction,_,_=fc.decode()
        tid={t:i for i,t in enumerate(fc.tracks)};qid={q['expression_id']:i for i,q in enumerate(fc.queries)}
        for s in (s for s in samples if s['pair']==pair and s['video_id']==video):
            priors[s['sample_id']]=float(np.clip(logit(np.clip(fraction[qid[s['query_id']],tid[s['track_id']]],.05,.95)),-2,2))
    rows=[]
    for s in samples:
        answer={v:responses[s['sample_id'],v] for v in VIEWS}
        p={v:float(a['yes_probability']) for v,a in answer.items()}
        prior=priors[s['sample_id']];reliability=ds[s['pair']].values
        def increment(v,learned):
            effective=answer[v]['effective_view'];sc=reliability[effective] if learned else (.8,.8)
            return float(np.clip(evidence_increment(p[v]>=.5,*sc),-2,2))
        # Replay the original registered routing choice, including its later
        # information-gain view selection. TEXT_ROUTE is a separate diagnostic.
        routed=s['original_view'];text_route=route(s['query_text'])
        posterior={
            'CROP_FIXED':float(expit(prior+increment('CROP',False))),
            'ROUTED_FIXED':float(expit(prior+increment(routed,False))),
            'CROP_DS':float(expit(prior+increment('CROP',True))),
            'ROUTED_DS':float(expit(prior+increment(routed,True))),
            'CLIP_FIXED':float(expit(prior+increment('CLIP',False))),
            'FRAME_BOX_FIXED':float(expit(prior+increment('FRAME_BOX',False))),
            'TEXT_ROUTE_FIXED':float(expit(prior+increment(text_route,False))),
            'ALL_MEAN':float(expit(prior+np.clip(logit(np.clip(np.mean(list(p.values())),1e-6,1-1e-6)),-6,6))),
            'ALL_VOTE_FIXED':float(expit(prior+sum(increment(v,False) for v in VIEWS))),
            'ALL_DS':float(expit(prior+sum(increment(v,True) for v in VIEWS))),
            'SOURCE_PRIOR':float(expit(prior))}
        rows.append(dict(sample_id=s['sample_id'],pair=s['pair'],video=s['video_id'],parent_video=s['parent_video'],
            query=s['query_id'],track=s['track_id'],frame=s['frame_id'],query_type=s['query_type'],split=s['split'],
            original_routed_view=routed,text_routed_view=text_route,prior_logit=prior,
            teacher_probability=p,posterior=posterior,choices={k:v>.5 for k,v in posterior.items()},
            teacher_request_ids={v:answer[v]['request_id'] for v in VIEWS},
            actual_effective_views={v:answer[v]['effective_view'] for v in VIEWS}))
    # Conditional acquisition diagnostic on the SAME prepaid, historically
    # sampled pool. No claims about live cost, unseen-pool recall or optimality.
    acquisition=[]
    for pair in PAIRS:
        pool=[r for r in rows if r['pair']==pair and r['split']=='evaluation'];n=len(pool)
        for seed in (0,17,42):
            rng=np.random.default_rng(seed);tie=rng.random(n)
            scores={
                'UNIFORM':tie,
                'UNCERTAINTY':np.asarray([entropy(expit(r['prior_logit'])) for r in pool]),
                'INFORMATION_GAIN':np.asarray([information_gain(expit(r['prior_logit']),*ds[pair].values[r['original_routed_view']]) for r in pool])}
            for frac in (.1,.3,1.):
                k=max(1,int(np.floor(n*frac)))
                for method in ('UNIFORM','INFORMATION_GAIN','UNCERTAINTY','UNCERTAINTY_EXPLORE'):
                    if method=='UNCERTAINTY_EXPLORE':
                        explore=max(1,int(np.ceil(.2*k)));rank=np.lexsort((tie,-scores['UNCERTAINTY']))
                        chosen=list(rank[:k-explore]);remaining=[i for i in range(n) if i not in chosen]
                        chosen+=rng.choice(remaining,explore,replace=False).tolist()
                    else:chosen=np.lexsort((tie,-scores[method]))[:k].tolist()
                    chosen=set(chosen)
                    acquisition.append(dict(pair=pair,seed=seed,method=method,budget_fraction=frac,teacher_calls=k,
                        predictions=[dict(sample_id=r['sample_id'],posterior=r['posterior']['ROUTED_DS'] if i in chosen else r['posterior']['SOURCE_PRIOR'],queried=i in chosen)
                                     for i,r in enumerate(pool)]))
    result=dict(status='SEALED_BEFORE_BANK_GT',rows=rows,reliability={p:d.receipt() for p,d in ds.items()},
        acquisition=acquisition,GT_open_attempts=len(attempts),neural_optimizer_updates=0,
        manifest_sha256=file_sha(OUT/'evidence_manifest.json'),responses_sha256=file_sha(OUT/'evidence/responses.jsonl'),
        routing='original predeclared v8P requested view; includes text-routing then IG; text-only route supplementary',
        scope='conditional mechanism bank, all methods share source prior and queried physical positions',
        bank_selection='historical active requested positions; not unbiased coverage of all candidates',
        preparation_teacher_receipt=completion,process_wall_seconds=time.perf_counter()-begin)
    assert not attempts
    atomic_json(OUT/'mechanism/bank_decisions.json',result)
    print(json.dumps(dict(event='MECHANISM_BANK_SEALED',rows=len(rows),methods=list(rows[0]['posterior']),GT_open_attempts=0)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['generate'],default='generate');p.parse_args();generate()
