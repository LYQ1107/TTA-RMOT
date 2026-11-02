"""Fixed source-only two-slope fit and separate GT-free cardinality decode."""
from argparse import ArgumentParser
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_uda.presence.runtime import RUN, PUBLIC, IN_DOMAIN, sealed_margins, target_metadata
from flex_uda.presence.cardinality import ARCHITECTURE, BernoulliCardinalityHead
from flex_tta.runtime.pf_receipts import atomic_json, file_sha


def fit(target, out, label_root=None):
    from scripts.multidomain_runtime import guard
    attempts = guard(); started = time.perf_counter()
    name = IN_DOMAIN[target]
    out = Path(out).resolve(); out.mkdir(parents=True,exist_ok=True)
    lock = (out / '.fit.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX)
    if (out / 'training_receipt.json').exists():
        print(json.dumps(dict(status='ALREADY_COMPLETE_NO_REFIT',model=name)),flush=True);return
    train_root = RUN / 'features/source_train' / target / name
    val_root = RUN / 'features' / target / name
    label_root = Path(label_root) if label_root else None
    allowed = [train_root.resolve(),val_root.resolve()] + ([label_root.resolve()] if label_root else [])
    def numerical_guard(event,args):
        if event == 'open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.name in ('labels.npy','GT_counts.npy') and not any(path.is_relative_to(p) for p in allowed):
                attempts.append(str(path));raise PermissionError('Other-domain numerical GT is not source calibration input')
    sys.addaudithook(numerical_guard)
    if label_root:
        labels_manifest = json.loads((label_root / 'labels_manifest.json').read_text())
        if labels_manifest['source_domain'] != target or labels_manifest['model'] != name:
            raise ValueError('Wrong corrected source labels')
        for row in labels_manifest['files']:
            if file_sha(row['path']) != row['sha256']:raise ValueError('Changed corrected source labels')
    x, y, source_rows = [], [], []
    for directory in sorted((train_root / 'videos').iterdir()):
        md = json.loads((directory / 'metadata.json').read_text())
        lp = label_root / 'videos' / directory.name / 'labels.npy' if label_root else directory / 'labels.npy'
        m = np.load(directory / 'margin.npy').astype('float64')
        labels = np.load(lp)
        if m.shape != labels.shape: raise ValueError('Source candidate-label alignment differs')
        x.append(m);y.append(labels)
        source_rows.append(dict(video=directory.name,rows=len(m),queries=len(md['queries']),
            score_path=str(directory / 'margin.npy'),score_sha256=file_sha(directory / 'margin.npy'),
            labels_path=str(lp),labels_sha256=file_sha(lp),feature_seal_sha256=file_sha(directory / 'seal.json')))
    x=np.concatenate(x);y=np.concatenate(y)
    fits=[];scales=[]
    for sign,mask in [('negative',x<0),('positive',x>=0)]:
        signed = (1-2*y[mask].astype('float64'))*x[mask]
        if not len(signed):raise ValueError('Fixed source fit requires both margin sign groups')
        def objective(alpha):return float(np.logaddexp(0,alpha*signed).mean())
        solved = minimize_scalar(objective,bounds=(.05,20.),method='bounded',
            options=dict(xatol=1e-6,maxiter=128))
        if not solved.success:raise RuntimeError('Source convex scalar calibration did not converge')
        scales.append(float(solved.x))
        fits.append(dict(sign=sign,rows=len(signed),alpha=float(solved.x),temperature=1/float(solved.x),
            initial_NLL=objective(1.),fitted_NLL=float(solved.fun),iterations=int(solved.nit),evaluations=int(solved.nfev)))
    model=BernoulliCardinalityHead(*scales).eval()
    def measures(m,labels):
        z=m*np.where(m>=0,scales[1],scales[0]);p=expit(z);prior=expit(m)
        return dict(rows=len(m),positive_rows=int(labels.sum()),
            native_NLL=float(np.logaddexp(0,(1-2*labels.astype(float))*m).mean()),
            calibrated_NLL=float(np.logaddexp(0,(1-2*labels.astype(float))*z).mean()),
            native_Brier=float(np.square(prior-labels).mean()),calibrated_Brier=float(np.square(p-labels).mean()),
            candidate_sign_changes=int(((m>0)!=(z>0)).sum()))
    training=measures(x,y)
    margin,reference=sealed_margins(target,name)
    validation_labels=[];cursor=0
    for directory in sorted((val_root / 'videos').iterdir()):
        md=json.loads((directory / 'metadata.json').read_text())
        a,b=md['queries'][0]['start'],md['queries'][-1]['stop']
        if a!=cursor:raise ValueError('Validation shards must match canonical sealed score order')
        cursor=b
        labels=np.load(directory / 'labels.npy')
        if len(labels)!=b-a:raise ValueError('Own-source validation query order differs')
        validation_labels.append(labels)
    vy=np.concatenate(validation_labels)
    if len(vy)!=len(margin):raise ValueError('Incomplete own-source validation labels')
    validation=measures(np.asarray(margin,dtype='float64'),vy)
    state=dict(architecture=ARCHITECTURE,head=model.state_dict(),target=target,model=name,seed=0,
        selection='source TRAIN candidate NLL convex minimizer; validation diagnostic only',
        fits=fits,learning_rate=None,train_metrics=training,validation_metrics=validation)
    checkpoint=out / 'checkpoint_best_source.pt'
    temporary=checkpoint.with_suffix('.partial');torch.save(state,temporary);temporary.replace(checkpoint)
    initial=dict(metrics=dict(loss=validation['native_NLL'],candidate_NLL=validation['native_NLL']),
        selected=False,reason='uncalibrated probability diagnostic, NOT native count-equivalence control')
    atomic_json(out / 'initial_calibration.json',initial)
    atomic_json(out / 'source_inputs.json',dict(source_domain=target,model=name,files=source_rows,
        source_validation_reference=reference,other_domain_GT_open_attempts=len(attempts)))
    result=dict(status='SOURCE_CALIBRATION_COMPLETE',architecture=ARCHITECTURE,target=target,model=name,
        best_source_loss=validation['calibrated_NLL'],selection=state['selection'],
        train_units=len(x),validation_units=len(vy),parameters=2,epochs=0,optimizer_updates=0,
        scalar_solver_evaluations=sum(r['evaluations'] for r in fits),fits=fits,train_metrics=training,
        validation_metrics=validation,learning_rate=None,physical_gpu=None,peak_allocated_bytes=0,
        GPU_seconds=0.,wall_seconds=time.perf_counter()-started,target_GT_open_attempts=len(attempts),
        head_only=True,frozen_FlexHook_loaded=False,PCD_or_sentence_features_used=False,
        checkpoint=str(checkpoint),checkpoint_sha256=file_sha(checkpoint))
    atomic_json(out / 'training_receipt.json',result)
    print(json.dumps(result),flush=True)


def decode(target,name,head_path,out,batch_size=4096):
    from flex_uda.presence.runtime import runtime_guard
    from flex_uda.presence.residual import head_from_checkpoint
    from flex_uda.presence.predictions import seal_mask,verify_predictions
    from scripts.diag_pf_countmatched import topk_ids
    attempts=runtime_guard();started=time.perf_counter()
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out / '.decode.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    if (out / 'decode_receipt.json').exists():
        r=json.loads((out / 'decode_receipt.json').read_text())
        for p,sha in r['files_sha256'].items():
            if file_sha(out / p)!=sha:raise ValueError('Changed completed count arrays')
        for method in ('PRES_TOPK','PRES_TOPK_SMOOTH'):verify_predictions(out / method)
        print(json.dumps(dict(status='ALREADY_SEALED_NO_RERUN',target=target,model=name)),flush=True);return
    state=torch.load(head_path,map_location='cpu',weights_only=False)
    if state['architecture']!=ARCHITECTURE or state['model']!=name:raise ValueError('Wrong v4 head')
    head=head_from_checkpoint(state).eval()
    meta=target_metadata(target);margins,reference=sealed_margins(target,name)
    queries={(q['video_id'],q['expression_id']):q for q in meta['episodes']}
    units=[];by_query=defaultdict(list);cap=0
    for directory in sorted((RUN / 'features' / target / name / 'videos').iterdir()):
        local=json.loads((directory / 'metadata.json').read_text());video=local['video_id']
        obs=np.asarray(meta['videos'][video]['observations'])
        fs,starts,sizes=np.unique(obs[:,0],return_index=True,return_counts=True)
        ranges={int(t):(int(a),int(a+n)) for t,a,n in zip(fs,starts,sizes)}
        for item in local['queries']:
            q=queries[video,item['expression_id']]
            for t in local['frames']:
                a,b=ranges.get(t,(0,0));cap=max(cap,b-a)
                by_query[video,item['expression_id']].append(len(units))
                units.append((video,item['expression_id'],t,q['start']+a,q['start']+b,a,b))
    count=len(units)
    presence=np.empty(count,'float32');joint=np.zeros((count,cap+1),'float32')
    conditional=np.zeros_like(joint);raw_k=np.empty(count,'int64')
    with torch.no_grad():
        for start in range(0,count,batch_size):
            part=units[start:start+batch_size];n=max(u[4]-u[3] for u in part)
            matrix=np.zeros((len(part),n),'float64');padding=np.ones_like(matrix,dtype=bool)
            for j,u in enumerate(part):
                size=u[4]-u[3];matrix[j,:size]=margins[u[3]:u[4]];padding[j,:size]=False
            result=head(margin=torch.from_numpy(matrix),padding_mask=torch.from_numpy(padding))
            width=n+1;stop=start+len(part)
            joint[start:stop,:width]=result['joint'].numpy()
            presence[start:stop]=result['presence'].numpy();raw_k[start:stop]=result['logits'].argmax(-1).numpy()
            if n:
                positive=result['logits'][:,1:].softmax(-1).numpy()
                positive[padding.all(1)]=0
                conditional[start:stop,1:width]=positive
    smoothed=np.empty_like(raw_k);radius=round({'KITTI-V2':10,'Dance':20,'SportsMOT':25}[target]/2)
    for ids in by_query.values():
        frames=np.asarray([units[j][2] for j in ids]);values=raw_k[ids]
        for j,index in enumerate(ids):
            a=np.searchsorted(frames,frames[j]-radius,side='left');b=np.searchsorted(frames,frames[j]+radius,side='right')
            smoothed[index]=int(np.median(values[a:b]))
    masks={method:np.zeros(len(margins),dtype=bool) for method in ('PRES_TOPK','PRES_TOPK_SMOOTH')}
    for j,(video,query,t,start,stop,a,b) in enumerate(units):
        ids=[i for _,i in meta['videos'][video]['observations'][a:b]];scores=dict(zip(ids,margins[start:stop]))
        for method,counts in [('PRES_TOPK',raw_k),('PRES_TOPK_SMOOTH',smoothed)]:
            chosen=topk_ids(ids,scores,int(counts[j]));masks[method][start:stop]=[i in chosen for i in ids]
    for key,value in dict(presence_logits=presence,count_probs=conditional,joint_count_probs=joint,raw_k=raw_k,smooth_k=smoothed).items():
        np.save(out / (key+'.npy'),value)
    extra=dict(head_checkpoint=str(head_path),head_checkpoint_sha256=file_sha(head_path),head_architecture=ARCHITECTURE,
        count_decoder='Poisson_binomial_joint_MAP_0_to_n',
        presence_definition='1-product(1-p_i); MAP count may disagree with presence>=0.5',
        conditional_count_probabilities=True,count_cap=None,source_learning_rate=None,
        margin_reference=reference,teacher_calls=0,test_time_optimizer_updates=0,native_count_floor=False)
    for method,mask in masks.items():
        seal_mask(out / method,meta,mask,target=target,model=name,method=method,GT_open_attempts=len(attempts),extra=extra)
    receipt=dict(status='SEALED_BEFORE_GT',target=target,model=name,units=count,GT_open_attempts=len(attempts),
        future_frames_in_smoothing=True,source_validation_early_stopping=False,CPU_only=True,
        wall_seconds=time.perf_counter()-started,files_sha256={p.name:file_sha(p) for p in out.glob('*.npy')})
    atomic_json(out / 'decode_receipt.json',receipt);print(json.dumps(receipt),flush=True)


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['fit','decode'],required=True);p.add_argument('--target',choices=list(IN_DOMAIN),required=True)
    p.add_argument('--model',choices=list(IN_DOMAIN.values()));p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--head-checkpoint',type=Path);p.add_argument('--source-label-root',type=Path)
    a=p.parse_args()
    if a.phase=='fit':fit(a.target,a.output_dir,a.source_label_root)
    else:decode(a.target,a.model,a.head_checkpoint,a.output_dir)


if __name__=='__main__':main()
