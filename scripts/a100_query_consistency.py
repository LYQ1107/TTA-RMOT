"""Frozen six-direction query-logic regularisation, identical decoder controls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import PAIRS,metadata,generation_guard
from flex_uda.active.query_consistency import relations,project
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_strict_prefix import plain,immediate_predictions
from scripts.a100_six_direction import OUT as TRANSFER

OUT=TRANSFER/'query_consistency'
METHODS=('KD_FINAL','KD_MEAN','HARD_FINAL','BASE_PROTOCOL','RAW_TEACHER_FILL','UNPROJECTED_JOINT',
         'LATTICE_KD','EQUIV_JOINT','LATTICE_JOINT','SHUFFLED_JOINT')


def run(pair,seed):
    guard=generation_guard();root=OUT/'quality'/pair/f'seed{seed}'
    if (root/'completion.json').is_file():return
    root.mkdir(parents=True,exist_ok=True);model,target=PAIRS[pair];m=metadata(target);start=time.perf_counter();receipts=[]
    previous=TRANSFER/'quality'/pair/f'seed{seed}'
    assert (previous/'completion.json').is_file()
    for vi,(video,bank) in enumerate(sorted(m['videos'].items())):
        stamp=time.perf_counter();queries=[q for q in m['episodes'] if q['video_id']==video];texts=[q['query_text'] for q in queries]
        with np.load(previous/'videos'/video/'coarse_scores.npz') as a:
            tracks=a['tracks'];values={k:a[k].copy() for k in ('KD_FINAL','KD_MEAN','HARD_FINAL','BASE_PROTOCOL','RAW_TEACHER_FILL')}
            joint=a['JOINT'].copy()
        parsed,edges,equivalent=relations(texts)
        equiv=np.asarray([edge for a,b in equivalent for edge in ((a,b),(b,a))],int).reshape(-1,2)
        seed_hash=int(hashlib.sha256((video+':query-implication-placebo').encode()).hexdigest()[:8],16)
        permutation=np.random.default_rng(seed_hash).permutation(len(queries));shuffled=permutation[edges]
        values['UNPROJECTED_JOINT']=joint
        values['LATTICE_KD'],kd_receipt=project(values['KD_FINAL'],edges)
        values['EQUIV_JOINT'],eq_receipt=project(joint,equiv)
        values['LATTICE_JOINT'],full_receipt=project(joint,edges)
        values['SHUFFLED_JOINT'],shuffle_receipt=project(joint,shuffled)
        obs=np.asarray(bank['observations']);lookup={int(t):i for i,t in enumerate(tracks)}
        columns=np.asarray([lookup[int(t)] for _,t in obs])
        for name in METHODS:
            mask=plain(values[name][:,columns],False);destination=root/name/'videos'/video
            destination.mkdir(parents=True,exist_ok=True);np.save(destination/'mask.npy',mask)
            immediate_predictions(root/name,type('Context',(),dict(bank=bank,queries=queries))(),mask)
        local=root/'videos'/video;local.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(local/'coarse_scores.npz',tracks=tracks,**values)
        receipt=dict(video=video,index=vi,projection=dict(KD=kd_receipt,EQUIVALENT=eq_receipt,FULL=full_receipt,SHUFFLED=shuffle_receipt),
            same_independent_decoder_all_methods=True,old_17query_mutual_exclusivity_removed_for_every_arm=True,
            parsed_queries=sum(p['supported'] for p in parsed),equivalent_pairs=len(equivalent),edges=edges.tolist(),
            new_teacher_encoder_head_calls=0,GT_open_attempts=len(guard),wall_seconds=time.perf_counter()-stamp)
        atomic_json(local/'receipt.json',receipt);receipts.append(receipt)
        print(json.dumps(dict(event='QUERY_CONSISTENCY_VIDEO',pair=pair,seed=seed,completed=vi+1,total=len(m['videos']),edges=len(edges))),flush=True)
    methods=[]
    for name in METHODS:
        mask=np.zeros(m['candidate_query_rows'],bool)
        for video in sorted(m['videos']):
            bits=np.load(root/name/'videos'/video/'mask.npy')
            for qi,q in enumerate(q for q in m['episodes'] if q['video_id']==video):mask[q['start']:q['stop']]=bits[qi]
        seal=seal_mask(root/name,m,mask,target=target,model=model,method='QUERY_CONSISTENCY_'+name,GT_open_attempts=len(guard),
            extra=dict(seed=seed,independent_per_query_decoder_every_arm=True,no_future_feedback=True,
                current_video_transductive=True,no_new_teacher_encoder_head_calls=True,text_parser_not_GT=True))
        methods.append(dict(method=name,rows=seal['prediction_rows']))
    assert not guard
    atomic_json(root/'completion.json',dict(status='SEALED_QUERY_CONSISTENCY',pair=pair,seed=seed,methods=methods,receipts=receipts,
        GT_open_attempts=0,new_teacher_encoder_head_calls=0,same_independent_decoder_every_arm=True,
        wall_seconds=time.perf_counter()-start,goal_status='ACTIVE'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=tuple(PAIRS),required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();run(a.pair,a.seed)
