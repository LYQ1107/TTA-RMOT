"""Same fixed heads, only the applicability of classical evidence inference."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import metadata,PAIRS,generation_guard
from flex_uda.active.acquisition import exclusive_group
from flex_uda.active.color_cohort import augmented_features,part_mask
from flex_uda.active.context_cohort import context_part_mask
from flex_uda.active.visual_classical import refine_distilled_prior
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_benchmark_scope import inputs,FRESH
from scripts.a100_visual_classical_budget_prefix import BUDGET
from scripts.a100_visual_classical_context_prefix import CONTEXT
from scripts.a100_visual_classical_strict_prefix import plain,immediate_predictions

SCOPE=OUT/'query_scope'
POLICIES=('CLOTHING','EXTENDED','ANCHORS')


def head_sources(pair,seed):
    original=BUDGET/f'rho0.1_seed{seed}'/pair
    result=dict(ORIGINAL_MEAN=original/'SOFT_MEAN_400_FINAL',ORIGINAL_400=original/'SOFT_400')
    if pair=='C4':
        source=CONTEXT/f'seed{seed}'/pair
        result.update(MASKED_MEAN=source/'MASKED_MEAN',CONTEXT_MEAN=source/'CONTEXT_MEAN',CONTEXT_400=source/'CONTEXT_SOFT400')
    return result


def run(pair,seed):
    guard=generation_guard();started=time.perf_counter()
    model,target=PAIRS[pair];m=metadata(target)
    root=SCOPE/f'seed{seed}'/pair
    if (root/'completion.json').is_file(): return
    root.mkdir(parents=True,exist_ok=True)
    sources=head_sources(pair,seed)
    method_names=[head+'_'+policy for head in sources for policy in POLICIES]
    receipts=[]
    for video,bank in sorted(m['videos'].items()):
        completed=root/'videos'/video/'receipt.json'
        if completed.is_file():
            row=json.loads(completed.read_text())
            assert row['GT_open_attempts']==row['original_graph_changed_rows']==0
            receipts.append(row)
            continue
        stamp=time.perf_counter()
        raw,_,_,context,columns,gates=inputs(pair,video,m)
        path=FRESH/'source'/video/'sparse_inputs.npz' if pair=='C4' else OUT/'color_features'/pair/video/'features.npz'
        with np.load(path) as a: colours=a['colours']
        texts=[q['query_text'] for q in context.queries]
        geometry,pcd,_=augmented_features(raw[:,:,:768],colours,texts)
        old_parts=np.stack([part_mask(text) for text in texts])
        new_parts=np.stack([context_part_mask(text) for text in texts])
        parsed=np.stack([(colours*mask[None,:,None]).reshape(len(colours),-1) for mask in new_parts])
        parsed/=np.maximum(np.linalg.norm(parsed,axis=2,keepdims=True),1e-8)
        extended_geometry=np.concatenate([pcd,parsed],axis=2).astype('float32')
        teachers=json.loads((BUDGET/'evidence/rho0.1'/pair/video/'answers.json').read_text())['answers']
        teacher=np.full(raw.shape[:2],np.nan)
        for row in teachers:teacher[int(row['local_query']),int(row['local_track'])]=row['yes_probability']
        first=[int(np.flatnonzero(columns==ti)[0]) for ti in range(raw.shape[1])]
        predictions={}
        for head,folder in sources.items():
            unary=np.load(folder/'videos'/video/'scores.npy')[:,first]
            assert unary.dtype==np.float32,(head,unary.dtype)
            for policy in POLICIES:
                score=unary.copy()
                for q in range(len(texts)):
                    parts=old_parts[q] if policy=='CLOTHING' else new_parts[q]
                    if not parts.any():continue
                    visible=np.all(colours[:,parts>0].sum(2)>0,axis=1)
                    feature=geometry[q] if policy=='CLOTHING' else extended_geometry[q]
                    score[q]=refine_distilled_prior(unary[q],feature,teacher[q],
                        graph=policy!='ANCHORS',source_weight=.1,node_mask=visible)
                predictions[head+'_'+policy]=score[:,columns]
        exclusive=exclusive_group(texts,target)
        masks={}
        for name,score in predictions.items():
            mask=plain(score,exclusive)
            for q,gate in gates.items():mask[q]&=gate
            masks[name]=mask
        oldmask=np.load(BUDGET/f'rho0.1_seed{seed}'/pair/'MEAN_GRAPH/videos'/video/'mask.npy')
        regression=int(np.count_nonzero(masks['ORIGINAL_MEAN_CLOTHING']!=oldmask))
        assert regression==0,('Original mean graph changed',pair,seed,video,regression)
        if pair=='C1':
            assert np.array_equal(geometry,extended_geometry)
            for head in sources:
                assert np.array_equal(masks[head+'_CLOTHING'],masks[head+'_EXTENDED'])
        precision_changes=0
        if pair=='C4':
            old=np.load(CONTEXT/f'seed{seed}'/pair/'CONTEXT_VISUAL/videos'/video/'mask.npy')
            precision_changes=int(np.count_nonzero(old!=masks['CONTEXT_MEAN_EXTENDED']))
        for name,mask in masks.items():
            folder=root/name;local=folder/'videos'/video
            local.mkdir(parents=True,exist_ok=True)
            np.save(local/'mask.npy',mask);np.save(local/'scores.npy',predictions[name])
            immediate_predictions(folder,context,mask)
        row=dict(video=video,methods=len(masks),original_graph_changed_rows=regression,
            contextual_common_precision_changed_rows=precision_changes,
            teacher_examples=len(teachers),head_updates=0,GT_open_attempts=len(guard),
            wall_seconds=time.perf_counter()-stamp)
        receipts.append(row);atomic_json(root/'videos'/video/'receipt.json',row)
    methods=[]
    for name in method_names:
        folder=root/name;mask=np.zeros(m['candidate_query_rows'],bool)
        for video in m['videos']:
            local=np.load(folder/'videos'/video/'mask.npy')
            for q,episode in enumerate(q for q in m['episodes'] if q['video_id']==video):mask[episode['start']:episode['stop']]=local[q]
        seal=seal_mask(folder,m,mask,target=target,model=model,method='QUERY_SCOPE_'+name,
            GT_open_attempts=len(guard),extra=dict(same_actual_head_weights=True,head_updates=0,
                same_frozen10percent_QA=True,new_teacher_calls=0,no_target_GT_in_inference=True,
                boxes_and_TrackIDs_fixed=True,not_deployment_timing=True))
        methods.append(dict(method=name,rows=seal['prediction_rows']))
    assert not guard
    atomic_json(root/'completion.json',dict(status='SEALED_SAME_HEAD_QUERY_SCOPE',pair=pair,seed=seed,
        methods=methods,GT_open_attempts=0,new_teacher_calls=0,head_updates=0,receipts=receipts,
        original_graph_changed_rows=sum(r['original_graph_changed_rows'] for r in receipts),
        contextual_common_precision_changed_rows=sum(r['contextual_common_precision_changed_rows'] for r in receipts),
        logical_teacher_calls=sum(r['teacher_examples'] for r in receipts),
        wall_seconds=time.perf_counter()-started,implementation_sha256=file_sha(Path(__file__)),goal_status='ACTIVE'))
    print(json.dumps(dict(event='QUERY_SCOPE_SEALED',pair=pair,seed=seed,methods=len(methods),goal_status='ACTIVE')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=['C4','C1'],required=True);p.add_argument('--seed',type=int,default=0)
    a=p.parse_args();run(a.pair,a.seed)
