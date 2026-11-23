"""Fixed-candidate attribution: raw anchors, graph edges, visual correspondence."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import PAIRS,metadata,generation_guard
from flex_uda.active.acquisition import exclusive_group
from flex_uda.active.grounded_graph import refine_queries
from flex_uda.active.semantic_colour import appearance_cues,shuffled_cues,crossfit_predictions,correct_belief,append_history
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_rgb_parity import source_spec
from scripts.a100_visual_classical_semantic_colour import empty_history
from scripts.a100_visual_classical_protocol_calibration import STUDY as REFERENCE
from scripts.a100_visual_classical_strict_prefix import plain,immediate_predictions

STUDY=OUT/'pure_components'
METHODS=('ANCHOR_ONLY','ANCHOR_CAL','SHUFFLED_CAL','SHUFFLED_JOINT')


def run(pair,seed):
    guard=generation_guard();begin=time.perf_counter();root=STUDY/pair/f'seed{seed}'
    if (root/'completion.json').is_file():return
    root.mkdir(parents=True,exist_ok=True);model,target=PAIRS[pair];m=metadata(target);source=source_spec(pair)[0]
    implementation={path:file_sha(ROOT/path) for path in ('scripts/a100_visual_classical_pure_components.py','flex_uda/active/semantic_colour.py','flex_uda/active/grounded_graph.py','flex_uda/active/visual_classical.py')}
    history=empty_history();shuffled_history=empty_history();receipts=[];calls=0
    for vi,(video,bank) in enumerate(sorted(m['videos'].items())):
        stamp=time.perf_counter();local=root/'videos'/video;local.mkdir(parents=True,exist_ok=True)
        queries=[e for e in m['episodes'] if e['video_id']==video];texts=[q['query_text'] for q in queries]
        with np.load(source/'source'/video/'sparse_inputs.npz') as a:pcd,colours,prior,tracks=a['pcd'],a['colours'],a['prior'],a['tracks']
        with np.load(REFERENCE/pair/f'seed{seed}'/'videos'/video/'coarse_scores.npz') as a:
            assert np.array_equal(tracks,a['tracks'])
            values={name:a[name] for name in ('KD','PROTOCOL_GRAPH','PROTOCOL_JOINT','FINAL_CAL')}
        obs=np.asarray(bank['observations']);index={int(t):i for i,t in enumerate(tracks)}
        columns=np.asarray([index[int(t)] for _,t in obs]);lengths=np.asarray([(obs[:,1]==t).sum() for t in tracks])
        cue,available,families,_=appearance_cues(colours,texts,lengths);teacher=np.full(cue.shape,np.nan)
        for answer in json.loads((source/'source'/video/'answers.json').read_text())['answers']:
            teacher[int(answer['local_query']),int(answer['local_track'])]=answer['yes_probability']
        calls+=int(np.isfinite(teacher).sum())
        probability,base,active,_=crossfit_predictions(cue,available,families,teacher,tracks,video,history)
        shuffled=shuffled_cues(cue,available,video)
        placebo,pbase,pactive,_=crossfit_predictions(shuffled,available,families,teacher,tracks,video,shuffled_history)
        errors=dict(FINAL_CAL=float(np.max(np.abs(correct_belief(values['KD'],probability,base,active)-values['FINAL_CAL']))),
            PRIMARY=float(np.max(np.abs(correct_belief(values['PROTOCOL_GRAPH'],probability,base,active)-values['PROTOCOL_JOINT']))))
        assert not any(errors.values()),(pair,seed,video,errors)
        values['ANCHOR_ONLY']=refine_queries(values['KD'],pcd,colours,texts,teacher,graph=False)
        values['ANCHOR_CAL']=correct_belief(values['ANCHOR_ONLY'],probability,base,active)
        values['SHUFFLED_CAL']=correct_belief(values['KD'],placebo,pbase,pactive)
        values['SHUFFLED_JOINT']=correct_belief(values['PROTOCOL_GRAPH'],placebo,pbase,pactive)
        exclusive=exclusive_group(texts,target)
        assert np.array_equal(plain(values['PROTOCOL_JOINT'][:,columns],exclusive),np.load(REFERENCE/pair/f'seed{seed}'/'PROTOCOL_JOINT/videos'/video/'mask.npy'))
        for name in METHODS:
            mask=plain(values[name][:,columns],exclusive);folder=root/name/'videos'/video;folder.mkdir(parents=True,exist_ok=True)
            np.save(folder/'mask.npy',mask);immediate_predictions(root/name,type('Context',(),dict(bank=bank,queries=queries))(),mask)
        np.savez_compressed(local/'coarse_scores.npz',tracks=tracks,**values)
        receipt=dict(video=video,index=vi,implementation=implementation,primary_and_CAL_replay_max_error=errors,
            teacher_examples=int(np.isfinite(teacher).sum()),GT_open_attempts=len(guard),wall_seconds=time.perf_counter()-stamp)
        atomic_json(local/'receipt.json',receipt);receipts.append(receipt)
        append_history(history,cue,available,families,teacher);append_history(shuffled_history,shuffled,available,families,teacher)
        print(json.dumps(dict(event='PURE_COMPONENTS_VIDEO',pair=pair,seed=seed,completed=vi+1,total=len(m['videos']))),flush=True)
    methods=[]
    for name in METHODS:
        mask=np.zeros(m['candidate_query_rows'],bool)
        for video in sorted(m['videos']):
            saved=np.load(root/name/'videos'/video/'mask.npy')
            for q,e in enumerate(e for e in m['episodes'] if e['video_id']==video):mask[e['start']:e['stop']]=saved[q]
        seal=seal_mask(root/name,m,mask,target=target,model=model,method='PURE_COMPONENTS_'+name,GT_open_attempts=len(guard),extra=dict(
            seed=seed,implementation=implementation,same_exact_converged_FINAL=True,new_teacher_calls=0,new_encoder_calls=0,new_head_fits=0,
            current_video_transductive=True,no_future_teacher=True,quality_replay_not_cold_time=True))
        methods.append(dict(method=name,rows=seal['prediction_rows']))
    assert not guard and calls==(1160 if pair=='C4' else 1103)
    atomic_json(root/'completion.json',dict(status='SEALED_PURE_COMPONENTS',pair=pair,seed=seed,methods=methods,receipts=receipts,
        logical_teacher_calls=calls,GT_open_attempts=0,new_teacher_calls=0,new_encoder_calls=0,new_head_fits=0,
        implementation=implementation,wall_seconds=time.perf_counter()-begin,goal_status='ACTIVE'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=['C4','C1'],required=True);p.add_argument('--seed',type=int,required=True)
    args=p.parse_args();run(args.pair,args.seed)
