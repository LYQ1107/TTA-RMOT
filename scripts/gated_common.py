"""Fixed gated-decode definitions and GT-free numerical operations."""
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import xquery_common as xq
from scripts import count_structure_common as cs
from scripts import diag_attrib_v2_common as v2
from flex_tta.runtime.pf_receipts import atomic_json,file_sha

OLD=xq.RUN;RUN=ROOT/'results/runs/gated_decode_20261101';PUBLIC=ROOT/'results/gated_decode'
MARGINS=xq.MARGINS
GENERAL=('G0_GZ_X0','G0_GZ_X1','G1_GMM_REJECT_X0','TG_X0','TG_X0_REASSIGN','TG_GZ_X0','TG_EARLY_X1')
SOURCE=xq.SOURCE
NORMAL=(.05,.17,.29,.41,.53,.65,.77,.89)
EARLY=(.025,.05,.075,.10,.125,.15,.175,.20)
TEACHER_PY='/home/szgx/anaconda3/envs/vllmedit/bin/python'
if (RUN/'deps').is_dir():sys.path.insert(0,str(RUN/'deps'))

def setup():
    RUN.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True);v2.EVAL=RUN/'evaluation'

def require_plan():
    r=json.loads((RUN/'prereg_publication.json').read_text())
    if not r['remote_verified'] or file_sha(PUBLIC/'PREREG.md')!=r['prereg_sha256']:raise ValueError('publish unchanged plan before execution')
    return r

def deny_annotations():
    attempts=[]
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute();r=p.resolve()
        if any(a.name in ('gt.txt','labels.json') or any(k in a.parts for k in ('gt_template','gt_templates','labels_with_ids','expression')) for a in (p,r)):
            attempts.append(str(p));raise PermissionError('GT-free gated decoder may not read annotations: '+str(p))
    sys.addaudithook(hook);return attempts

def inputs(verify=False):
    # Redirect only this helper's NEW audit destination, never old artifacts.
    previous=xq.RUN;xq.RUN=RUN
    try:return xq.inputs(verify=verify)
    finally:xq.RUN=previous

def source_inputs(verify=False):
    meta=json.loads((OLD/'source/metadata.json').read_text())
    seal=json.loads((OLD/'source/scores_seal.json').read_text())
    expected={'FROZEN':meta['frozen_array_sha256'],'T2V_D1_s17':seal['models']['T2V_D1_s17']['array_sha256']}
    if verify:
        original=json.loads((xq.PUBLIC/'source_inputs.json').read_text())
        if file_sha(OLD/'source/metadata.json')!=original['metadata_sha256']:raise ValueError('source metadata changed')
        for m,h in expected.items():
            if file_sha(OLD/'source'/(m+'.npy'))!=h:raise ValueError('source array changed '+m)
        for b in meta['videos'].values():
            for p in b['tracker_files']:
                if file_sha(p['path'])!=p['sha256']:raise ValueError('source MOT changed')
        atomic_json(RUN/'source_input_audit.json',dict(status='PASS',metadata_sha256=file_sha(OLD/'source/metadata.json'),
            arrays={m:dict(path=str(OLD/'source'/(m+'.npy')),sha256=h) for m,h in expected.items()},rows=2355528,episodes=862))
    return meta,{m:np.load(OLD/'source'/(m+'.npy'),mmap_mode='r') for m in expected}

def means(matrix,inverse):
    n=int(inverse.max())+1
    return np.stack([cs.track_mean(x,inverse,n) for x in matrix])

def raw_decodes(matrix,inverse):
    n=int(inverse.max())+1;mu=means(matrix,inverse)
    causal=np.stack([cs.causal_track_mean(x,inverse,n) for x in matrix])
    q=np.arange(len(matrix))[:,None]
    return q==mu.argmax(0)[inverse][None,:],q==causal.argmax(0)[None,:],mu

def global_moments(meta,array):
    # Exact sentence text is the key (same 17 sentences across target videos).
    chunks=defaultdict(list)
    for q in meta['episodes']:chunks[q['query_text']].append(array[q['start']:q['stop']])
    return {text:dict(mean=float((x:=np.concatenate(xs)).mean()),std=float(x.std(ddof=0)),observations=len(x)) for text,xs in sorted(chunks.items())}

def global_z(matrix,queries,stats):
    out=np.zeros_like(matrix,dtype=np.float64)
    for j,q in enumerate(queries):
        r=stats[q['query_text']]
        if r['std']!=0:out[j]=(matrix[j]-r['mean'])/r['std']
    return out

def gmm_reject(mu,inverse):
    from sklearn.mixture import GaussianMixture
    winners=mu.argmax(0);s=mu[winners,np.arange(mu.shape[1])];keep=np.ones(len(s),bool)
    report=dict(tracks=len(s),fit=False,BIC1=None,BIC2=None,means2=None,converged1=None,converged2=None,removed=0)
    if len(s)>=4:
        x=s[:,None];a=GaussianMixture(1,random_state=0,n_init=5,covariance_type='full').fit(x)
        b=GaussianMixture(2,random_state=0,n_init=5,covariance_type='full').fit(x)
        bic1,bic2=a.bic(x),b.bic(x)
        if bic2<bic1:keep=b.predict(x)!=int(b.means_[:,0].argmin())
        report.update(fit=True,BIC1=float(bic1),BIC2=float(bic2),means2=b.means_[:,0].tolist(),converged1=bool(a.converged_),converged2=bool(b.converged_),removed=int((~keep).sum()))
    return (np.arange(len(mu))[:,None]==winners[inverse][None,:])&keep[inverse],report

def sampled_frames(first,last,available,fractions):
    available=np.asarray(sorted(set(map(int,available))),dtype=np.int64)
    wanted=[int(np.floor(first+p*(last-first)+.5)) for p in fractions]
    return [int(available[np.argmin(np.abs(available-t))]) for t in wanted]

def gate_and_exclusivity(frame_probabilities,nq):
    votes=np.zeros(nq,dtype=int);frames=[set() for _ in range(nq)];positive=multiple=0
    for t,p in sorted(frame_probabilities.items()):
        p=np.asarray(p,dtype=np.float64)
        if p.size==0:continue
        above=(p>=.5).sum(1);positive+=int((above>=1).sum());multiple+=int((above>=2).sum())
        for q in p.argmax(1)[p.max(1)>=.5]:votes[q]+=1;frames[q].add(int(t))
    gate=np.asarray([n>=2 and len(f)>=2 for n,f in zip(votes,frames)],dtype=bool)
    return gate,dict(votes=votes.tolist(),distinct_frames=[len(f) for f in frames],positive_candidates=positive,multiple_candidates=multiple,
        E=multiple/positive if positive else None,zero_denominator=positive==0)

def gated_masks(matrix,inverse,present,early,stats,queries):
    x0,x1,mu=raw_decodes(matrix,inverse);z=global_z(matrix,queries,stats);z0,z1,_=raw_decodes(z,inverse)
    gmm,detail=gmm_reject(mu,inverse);reassign=np.zeros_like(x0)
    if np.any(present):
        winners=mu.argmax(0);absent=~present[winners]
        winners[absent]=np.where(present[:,None],mu,-np.inf).argmax(0)[absent]
        reassign=np.arange(len(matrix))[:,None]==winners[inverse][None,:]
    return dict(G0_GZ_X0=z0,G0_GZ_X1=z1,G1_GMM_REJECT_X0=gmm,TG_X0=x0&present[:,None],
        TG_X0_REASSIGN=reassign,TG_GZ_X0=z0&present[:,None],TG_EARLY_X1=x1&early[:,None]),detail

def sa_mask(matrix,inverse,present,E):
    if E is None:return np.zeros_like(matrix,dtype=bool),'EMPTY_NO_TEACHER_EVIDENCE'
    if E<=.5:return raw_decodes(matrix,inverse)[0]&present[:,None],'TG_X0'
    return matrix>0,'NATIVE'
