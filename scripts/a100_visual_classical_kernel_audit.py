"""Teacher-only, held-video diagnostic for nonparametric exemplar memory.

This diagnostic does not evaluate RMOT or read target ground truth. It is a
batch teacher generalization audit, not a video-stream experiment.
"""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import logit
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import generation_guard
from flex_uda.active.exemplar_memory import ExemplarMemory
from scripts.a100_visual_classical_prepare import OUT


def loss(probability,teacher):
    p=np.clip(probability,1e-8,1-1e-8);y=np.asarray(teacher)
    return float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))


def run(pair,group_unit='original_parent',compare_mlp=False):
    guard=generation_guard();begin=time.perf_counter();x=[];z=[];y=[];fold=[]
    paths=sorted((OUT/'causal_teacher'/pair/'COVER').glob('*.json'))
    group=lambda v:v.rsplit('_c',1)[0] if group_unit=='original_parent' else v
    groups=sorted({group(p.stem) for p in paths});group_index={g:i for i,g in enumerate(groups)}
    for vi,path in enumerate(paths):
        rows=json.loads(path.read_text())['answers']
        with np.load(OUT/'inputs'/pair/path.stem/'inputs.npz') as a:
            features=a['features'];source=np.clip(logit(np.clip(a['prior'],.05,.95)),-2,2)
        for row in rows:
            q,t=int(row['local_query']),int(row['local_track'])
            x.append(features[q,t]);z.append(source[q,t]);y.append(row['yes_probability']);fold.append(group_index[group(path.stem)]%3)
    x=np.asarray(x);z=np.asarray(z);y=np.asarray(y);fold=np.asarray(fold)
    kernel_prediction=np.empty_like(y);prior_prediction=1/(1+np.exp(-z));receipts=[];mlp_prediction=np.empty_like(y);variance=np.empty_like(y)
    for held in range(3):
        training=fold!=held;testing=~training;memory=ExemplarMemory()
        fit=memory.fit(x[training],z[training],y[training]);p,v=memory.predict(x[testing],z[testing]);kernel_prediction[testing]=p;variance[testing]=v
        receipts.append(dict(fold=held,training_examples=int(training.sum()),held_examples=int(testing.sum()),
            fit=fit,teacher_CE=loss(p,y[testing]),mean_variance=float(v.mean())))
        if compare_mlp:
            from flex_uda.active.prefix_teacher import PrefixTeacherHead
            head=PrefixTeacherHead('cuda:0',0);head.add(x[training],z[training],y[training]);receipt=head.fit()
            mlp_prediction[testing]=head.predict(x[testing],z[testing])
            receipts[-1]['MLP_teacher_CE']=loss(mlp_prediction[testing],y[testing]);receipts[-1]['MLP_fit']=receipt
        print(json.dumps(dict(event='EXEMPLAR_TEACHER_AUDIT',pair=pair,**receipts[-1])),flush=True)
    entropy=loss(y,y)
    result=dict(pair=pair,teacher_only=True,target_GT_used=False,GT_open_attempts=len(guard),
        kernel_teacher_CE=loss(kernel_prediction,y),source_teacher_CE=loss(prior_prediction,y),teacher_entropy=entropy,
        kernel_excess_KL=loss(kernel_prediction,y)-entropy,folds=receipts,
        held_out_unit=group_unit,groups=groups,fold_rule='lexicographic groups modulo3',wall_seconds=time.perf_counter()-begin,
        not_a_causal_stream_experiment=True,not_RMOT_accuracy=True)
    if compare_mlp:
        from scipy.special import expit
        kz=logit(np.clip(kernel_prediction,1e-6,1-1e-6));mz=logit(np.clip(mlp_prediction,1e-6,1-1e-6))
        variants=dict(KD=mlp_prediction,KERNEL=kernel_prediction,MEAN=expit(.5*(kz+mz)),
            SUPPORT_MIX=expit((1-variance)*mz+variance*kz),SUPPORT_GATED=expit(z+(1-variance)*(mz-z)))
        result.update(MLP_teacher_CE=loss(mlp_prediction,y),MLP_excess_KL=loss(mlp_prediction,y)-entropy,
            variants_teacher_CE={k:loss(p,y) for k,p in variants.items()})
        destination=OUT/'teacher_only_audit'/pair;destination.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(destination/('held_'+group_unit+'_predictions.npz'),teacher=y,source_logits=z,
            kernel=kernel_prediction,MLP=mlp_prediction,variance=variance,fold=fold)
    assert not guard;atomic_json(OUT/'teacher_only_audit'/pair/('exemplar_memory_'+group_unit+('_vs_MLP' if compare_mlp else '')+'.json'),result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',required=True,choices=['C4','C1'])
    p.add_argument('--group-unit',choices=['original_parent','clip'],default='original_parent');p.add_argument('--compare-mlp',action='store_true')
    a=p.parse_args();run(a.pair,a.group_unit,a.compare_mlp)
