"""Teacher-only task-contract safeguard on completed strict RGB predictions."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import metadata,PAIRS,generation_guard
from flex_uda.active.task_decode import nonempty_from_reference,retain_nonempty_distillation
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_rgb_parity import PARITY,source_spec
from scripts.a100_visual_classical_strict_prefix import immediate_predictions

NULL=OUT/'null_set_safeguard'
HEADS={'MEAN':('RGB_KD_MEAN','RGB_GRAPH_MEAN'),'400':('RGB_KD400','RGB_GRAPH400'),
    'FINAL':('RGB_KD_FINAL','RGB_GRAPH_FINAL'),'HARD400':('RGB_HARD400','RGB_GRAPH_HARD400'),
    'HARD_FINAL':('RGB_HARD_FINAL','RGB_GRAPH_HARD_FINAL')}


def run(seed):
    guard=generation_guard();begin=time.perf_counter();pair='C4';root=NULL/f'seed{seed}'/pair
    if (root/'completion.json').is_file():return
    source=PARITY/pair/f'seed{seed}';complete=json.loads((source/'completion.json').read_text());assert complete['GT_open_attempts']==0
    protocol=json.loads((OUT/'preregistration_v1.json').read_text())['protocol'][pair];assert protocol=='PRESUPPOSED'
    model,target=PAIRS[pair];m=metadata(target);receipts=[];method_names=[]
    for video,bank in sorted(m['videos'].items()):
        stamp=time.perf_counter();queries=[q for q in m['episodes'] if q['video_id']==video];obs=np.asarray(bank['observations'])
        with np.load(source_spec(pair)[0]/'source'/video/'sparse_inputs.npz') as values:prior,tracks=values['prior'],values['tracks']
        first=np.asarray([np.flatnonzero(obs[:,1]==track)[0] for track in tracks])
        ref=np.load(source/'RGB_KD_MEAN/videos'/video/'scores.npy');graph_ref=np.load(source/'RGB_GRAPH_MEAN/videos'/video/'scores.npy')
        masks={};details={};mean_scores={}
        for head,(kd,graph) in HEADS.items():
            before_kd=np.load(source/kd/'videos'/video/'mask.npy');before_graph=np.load(source/graph/'videos'/video/'mask.npy')
            protected_kd,kd_changes=nonempty_from_reference(before_kd,ref,obs,tracks,prior,protocol)
            protected_graph,graph_changes=nonempty_from_reference(before_graph,ref,obs,tracks,prior,protocol)
            safe,rejected=retain_nonempty_distillation(before_graph,protected_kd,protocol)
            assert protected_kd.any(1).all() and protected_graph.any(1).all() and safe.any(1).all()
            assert np.array_equal(safe[before_graph.any(1)],before_graph[before_graph.any(1)])
            assert np.array_equal(retain_nonempty_distillation(before_graph,protected_kd,'CHECKLIST')[0],before_graph)
            masks.update({f'KD_{head}_PRESENCE':protected_kd,f'GRAPH_{head}_PRESENCE':protected_graph,f'SAFE_{head}':safe})
            details[head]=dict(KD_singleton_fallback=kd_changes,GRAPH_singleton_fallback=graph_changes,
                graph_empty_set_rejections=int(rejected.sum()),retained_multiple_KD_tracks_queries=int((rejected&(protected_kd[:,first].sum(1)>1)).sum()))
            if head=='MEAN':
                # CountSelf uses the actual fallback reference on every null
                # query, with the exact same tie order as the common decoder.
                mixed=graph_ref.copy();mixed[~before_graph.any(1)]=ref[~before_graph.any(1)]
                mean_scores=dict(KD_MEAN_PRESENCE=ref,GRAPH_MEAN_PRESENCE=mixed,SAFE_MEAN=mixed)
        for name,mask in masks.items():
            folder=root/name;local=folder/'videos'/video;local.mkdir(parents=True,exist_ok=True)
            np.save(local/'mask.npy',mask)
            if name in mean_scores:np.save(local/'scores.npy',mean_scores[name])
            immediate_predictions(folder,SimpleNamespace(bank=bank,queries=queries),mask)
        method_names=list(masks);receipt=dict(video=video,changes=details,CPU_seconds=time.perf_counter()-stamp,GT_open_attempts=len(guard))
        receipts.append(receipt);atomic_json(root/'videos'/video/'receipt.json',receipt)
        print(json.dumps(dict(event='NULL_SET_VIDEO',seed=seed,video=video,completed=len(receipts),total=45)),flush=True)
    methods=[]
    for name in method_names:
        mask=np.zeros(m['candidate_query_rows'],bool);folder=root/name
        for video in sorted(m['videos']):
            local=np.load(folder/'videos'/video/'mask.npy')
            for q,episode in enumerate(q for q in m['episodes'] if q['video_id']==video):mask[episode['start']:episode['stop']]=local[q]
        seal=seal_mask(folder,m,mask,target=target,model=model,method=name,GT_open_attempts=len(guard),extra=dict(
            explicit_task_contract=protocol,same_common_reference_for_every_arm=True,no_new_teacher_calls=True,no_new_head_fit=True,
            official_boxes_and_Track_IDs_unchanged=True,current_video_transductive=True,not_cold_timing=True))
        methods.append(dict(method=name,rows=seal['prediction_rows']))
    assert not guard
    atomic_json(root/'completion.json',dict(status='SEALED_NULL_SET_SAFEGUARD',pair=pair,seed=seed,methods=methods,
        receipts=receipts,source=str(source),GT_open_attempts=0,new_teacher_calls=0,new_head_fits=0,protocol=protocol,
        wall_seconds=time.perf_counter()-begin,implementation_sha256=file_sha(Path(__file__)),goal_status='ACTIVE'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);a=p.parse_args();run(a.seed)
