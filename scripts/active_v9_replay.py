"""Fixed-v8-answer CPU counterfactual stream, sealed before offline GT."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import json
import pickle
import time
from pathlib import Path
import sys
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.active_v9_inputs import PUBLIC,V8,SOURCE,DATA,PAIRS,rel_from,estimation_guard
from scripts.active_v6_preflight import stage_arrays
from flex_uda.active.v8 import PresenceEM,BeliefStateV8
from flex_uda.active.v7 import OnlinePresenceEB
from flex_uda.active.v9_offline import RandomEffectsPresence,RandomEffectsBelief
from flex_uda.presence.runtime import runtime_guard,target_metadata
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def parameters(em,name):
    r=em.receipt()
    if name=='E3':r=dict(r,pi_strata=[em.pi]*3,estimator='ORIGINAL_V7_NUMBERS_ON_FIXED_V8_STREAM')
    return r


def replay(name,pair):
    opened=estimation_guard();begin=time.perf_counter()
    inventory=json.loads((PUBLIC/'inventory_D3.json').read_text())
    assert file_sha(PUBLIC/'inventory_D3.json')==json.loads((PUBLIC/'inputs_seal.json').read_text())['sha256']
    if name=='E2' and inventory['E2_status']=='NOT_APPLICABLE':
        result=dict(status='NOT_APPLICABLE',estimator=name,pair=pair,reason=inventory['E2_reason'],GT_open_attempts=0)
        atomic_json(PUBLIC/f'{name}_{pair}.json',result);return result
    model,target=PAIRS[pair];source=SOURCE/pair/'MINUS_SLOW/rho_0.3'
    out=DATA/name/pair;out.mkdir(parents=True,exist_ok=True)
    seal=out/'estimates_seal.json'
    if seal.exists():return json.loads(seal.read_text())
    meta=target_metadata(target);ordered=sorted(meta['videos'])
    em=OnlinePresenceEB() if name=='E3' else PresenceEM() if name=='E0' else RandomEffectsPresence()
    if name=='E2':raise RuntimeError('E2 not dispatched when protocol applicability is not established')
    rel={v:(.8,.8) for v in ('CROP','CLIP','FRAME_BOX')};temperature=1.;bias=0.;records=[];start=0
    checkpoints=sorted(out.glob('stream_*.pkl'))
    if checkpoints:
        with checkpoints[-1].open('rb') as handle:saved=pickle.load(handle)
        em=saved['em'];rel=saved['rel'];temperature=saved['temperature'];bias=saved['bias']
        records=saved['records'];start=saved['next_video']
    for vi,video in enumerate(ordered):
        if vi<start:continue
        directory=source/'videos'/video;bank=meta['videos'][video]
        original=json.loads((directory/'receipt.json').read_text())
        answers=json.loads((directory/'questions.json').read_text())
        with np.load(directory/'belief.npz') as b:belief={k:b[k].copy() for k in b.files}
        tracks=belief['tracks'].tolist();names=belief['queries'].tolist()
        qid={q:i for i,q in enumerate(names)};tid={t:i for i,t in enumerate(tracks)}
        obs=np.asarray(bank['observations']);lengths=np.bincount(np.searchsorted(tracks,obs[:,1]),minlength=len(tracks)).tolist()
        raw=np.asarray(stage_arrays(Path(V8['stage_A_root']),target,model,video,len(names))[0]['presence_logits'],float)
        before=parameters(em,name)
        def make(uncapped=False):
            extra=dict(tau=em.tau,uncapped_decision=uncapped) if name in ('E1','E2') else {}
            klass=RandomEffectsBelief if name in ('E1','E2') else BeliefStateV8
            state=klass(np.zeros_like(belief['fractions']),belief['fractions'],expit(raw/temperature+bias).max(1),
                track_lengths=lengths,rho_hat=before['rho_hat'],pi_strata=before['pi_strata'],**extra)
            state.strata=belief['prior_strata'].copy();state.recompute_presence(rel)
            return state
        state=make();uncapped=make(True) if name in ('E1','E2') else None
        for r in range(4):
            for s in (state,uncapped):
                if s is not None:s.recompute_presence(rel)
            for a in (a for a in answers if a['round']==r+1):
                q,t=qid[a['query_id']],tid[a['track_id']];view=a['effective_view']
                for s in (state,uncapped):
                    if s is not None:s.observe(q,t,a['yes_probability']>=.5,view,rel[view],
                        presence_question=a['kind'] in ('GLANCE','PRESENCE','PRESENCE_COVER'))
            rr=original['rounds'][r];rel=rel_from(rr['ds'])
            for s in (state,uncapped):
                if s is not None:s.recompute_presence(rel)
            temperature=rr['fast']['temperature'];bias=rr['fast']['bias']
            maximum=expit(raw/temperature+bias).max(1)
            for s in (state,uncapped):
                if s is not None:s.refresh_presence(maximum)
        if name=='E3':
            state.pi_start=em.pi;fit=em.finish_video(state,rel);after=parameters(em,name)
        else:
            em.add_video(video,state);after=em.fit(rel)
            if not after['converged']:
                atomic_json(out/f'{video}_fit_failure.json',dict(before=before,after=after,GT_open_attempts=0))
                raise RuntimeError(f'{name}/{pair}/{video} estimator nonconvergence; preserve completed units')
        reproduction={}
        if name=='E0':
            expected=original['EB']['after']
            error=float(np.max(np.abs(np.r_[em.rho,em.pi]-np.r_[expected['rho_hat'],expected['pi_strata']])))
            mask_equal=np.array_equal(state.output_absence(),belief['output_absent'])
            probability_error=float(np.max(np.abs(state.presence_probability-belief['presence_probability'])))
            reproduction=dict(parameter_max_error=error,output_absence_exact=mask_equal,probability_max_error=probability_error)
            if error>=1e-9 or probability_error>=1e-9 or not mask_equal:
                atomic_json(out/'reproduction_failure.json',dict(video=video,**reproduction))
                raise RuntimeError('E0 must reproduce v8; inspect '+str(out/'reproduction_failure.json'))
        queries=[]
        for q,text in enumerate(names):
            queries.append(dict(query=text,probability=float(state.presence_probability[q]),
                recomputed_probability=float(state.presence_probability_recomputed[q]),
                evidence_absent=bool(state.absent[q]),low_tentative=bool(state.low_tentative[q]),
                output_absent=bool(state.output_absence()[q]),
                decision_source='EVIDENCE' if state.absent[q] else 'PRIOR_BUDGET_EXHAUSTED' if state.output_absence()[q] else 'NOT_REJECTED',
                uncapped_evidence_absent=bool(uncapped.absent[q]) if uncapped else None,
                uncapped_output_absent=bool(uncapped.output_absence()[q]) if uncapped else None))
        row=dict(pair=pair,estimator=name,video=video,video_index=vi,before=before,after=after,queries=queries,
            original_budget=original['budget'],fixed_answer_count=len(answers),
            reproduction=reproduction,GT_open_attempts=0,
            weak_rank_protocol='Original v8 video-start and per-round T/beta applied to frozen StageA logits; no final-rank lookahead; no counterfactual fast re-fit')
        atomic_json(out/f'{video}.json',row);records.append(row)
        checkpoint=out/f'stream_{vi:03d}.pkl'
        with checkpoint.open('wb') as handle:pickle.dump(dict(em=em,rel=rel,temperature=temperature,bias=bias,
            records=records,next_video=vi+1),handle,protocol=pickle.HIGHEST_PROTOCOL)
        print(json.dumps(dict(event='V9_OFFLINE_VIDEO',estimator=name,pair=pair,video=video,
            rho=after['rho_hat'],pi=after['pi_strata'],tau=after.get('tau'),iterations=after.get('iterations'))),flush=True)
    assert not opened
    result=dict(status='SEALED_BEFORE_GT',estimator=name,pair=pair,target=target,model=model,
        records=records,final=parameters(em,name),GT_open_attempts=0,GPU_calls=0,teacher_calls=0,
        wall_seconds=time.perf_counter()-begin,source_seal_sha256=file_sha(source/'predictions_seal.json'),
        input_inventory_sha256=file_sha(PUBLIC/'inventory_D3.json'),manifest_sha256=file_sha(PUBLIC/'manifest.json'))
    result_path=PUBLIC/f'{name}_{pair}.json';atomic_json(result_path,result)
    sealed=dict(status='SEALED_BEFORE_GT',estimator=name,pair=pair,path=str(result_path),
        sha256=file_sha(result_path),GT_open_attempts=0,GPU_calls=0,teacher_calls=0)
    atomic_json(seal,sealed);return sealed


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--estimator',choices=['E0','E1','E2','E3'],required=True)
    p.add_argument('--pair',choices=list(PAIRS));a=p.parse_args()
    for pair in ([a.pair] if a.pair else PAIRS):print(json.dumps(replay(a.estimator,pair)),flush=True)
