#!/usr/bin/env python3
"""GT-free, post-hoc count×ranking controls using the unchanged v2 operators."""
import gzip
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts import diag_attrib_v2_common as common
from scripts.run_train2val_training import RUN,PUBLIC,ARMS

OLD=ROOT/'results/runs/diag_attrib_20261031_v2/evaluation'
REFERENCES={'FROZEN':OLD/'M_FROZEN__FROZEN','SAME_SET_D1':OLD/'M_D1__D1',
    'SAME_SET_D3':OLD/'M_D3__D3','B1_SRC_RATE_GLOBAL':OLD/'B1_SRC_RATE_GLOBAL',
    'OFFICIAL_DANCE_COMPLETE':OLD/'OFFICIAL_DANCE_COMPLETE'}


def verify_native_predictions():
    sealed=json.loads((RUN/'predictions_seal.json').read_text())
    if sealed['status']!='ALL_TRAINED_ARMS_COMPLETE425_SEALED_BEFORE_GT':raise ValueError('seal every trained arm first')
    for name,r in sealed['methods'].items():
        if file_sha(r['run_metadata_path'])!=r['run_metadata_sha256']:raise ValueError('changed sealed native metadata')
        for e in r['episodes']:
            for kind in ('prediction','margins'):
                if file_sha(e[kind+'_path'])!=e[kind+'_sha256']:raise ValueError('changed native prediction: '+name)
    return sealed


def prepare():
    started=time.perf_counter();sealed=verify_native_predictions();attempts=common.deny_gt()
    meta,old_scores,groups,_=common.load_bank();lines=common.candidate_lines(meta)
    common.EVAL=RUN/'evaluation'
    sizes=meta['candidate_query_rows'];g=meta['query_frame_groups']
    if sizes!=3588445 or len(meta['episodes'])!=425:raise ValueError('full candidate bank changed')
    scores={'FROZEN':old_scores['FROZEN'],'OFFICIAL_DANCE_COMPLETE':old_scores['OFFICIAL_DANCE_COMPLETE']}
    methods=[];reused=[]
    for name,original in REFERENCES.items():
        selected={}
        for q in meta['episodes']:
            p=original/'results'/q['video_id']/q['expression_id']/'predict.txt'
            selected[q['video_id'],q['expression_id']]=p.read_text().splitlines()
        common.seal_plain(name,meta['episodes'],selected,protocol='EXACT_REUSED_SEALED_V2_REFERENCE')
        common.assert_equivalent(name,original);methods.append(name);reused.append(name)
    cache=RUN/'score_bank';cache.mkdir(parents=True,exist_ok=True)
    # The bank is query-major; original v2 group ranks retain its deterministic
    # same-frame/ID tie rule. No GT counts or target-selected threshold enters.
    tracks={v:common.load_tracker_lines(common.TRACKS,v) for v in {q['video_id'] for q in meta['episodes']}}
    for name,r in sealed['methods'].items():
        dest=cache/(name+'.npy');array=np.lib.format.open_memmap(dest,mode='w+',dtype='float64',shape=(sizes,))
        episodes=[]
        for q in meta['episodes']:
            folder=RUN/'evaluation'/name/'results'/q['video_id']/q['expression_id']
            p=folder/'candidate_margins.jsonl.gz'
            with gzip.open(p,'rt') as stream:
                for expected,line in zip(q['groups'],stream,strict=True):
                    t,a,b,group=expected;row=json.loads(line)
                    if row['frame']!=t or row['ids']!=sorted(tracks[q['video_id']].get(t,{})) or len(row['adapted'])!=b-a:
                        raise ValueError('full candidate/score alignment differs')
                    array[a:b]=row['adapted']
            path=folder/'predict.txt';count=sum(bool(x.strip()) for x in path.read_text().splitlines())
            if count!=int((array[q['start']:q['stop']]>0).sum()):raise ValueError('native output not margin>0')
            episodes.append(dict(video=q['video_id'],expression=q['expression_id'],prediction_path=str(path),
                prediction_sha256=file_sha(path),selected_rows=count,frames=q['frame_count'],margins_path=str(p),margins_sha256=file_sha(p)))
        array.flush();scores[name]=array
        # Add a dedicated seal without rewriting native run_metadata/full_audit.
        seal=dict(status='SEALED',method=name,episode_count=425,episodes=episodes,
            prediction_rows=sum(e['selected_rows'] for e in episodes),protocol='TRAIN_TO_VAL_FINAL_STEP',
            source_native_run_metadata_sha256=r['run_metadata_sha256'],immutable_MOT_rows=True)
        path=RUN/'evaluation'/name/'predictions_seal.json'
        if path.exists() and json.loads(path.read_text())!=seal:raise ValueError('different native seal')
        atomic_json(path,seal);methods.append(name)
    counts={name:np.bincount(groups,weights=(values>0),minlength=g).astype(np.int32) for name,values in scores.items()}
    frozen_rank=common.rank_within_groups(scores['FROZEN'],groups,g)
    control='FROZEN_AT_OFFICIAL_COUNT'
    common.write_selection(control,frozen_rank<counts['OFFICIAL_DANCE_COMPLETE'][groups],meta,lines,
        protocol='POST_HOC_COUNT_RANK_DIAGNOSTIC_NOT_INDEPENDENT_DEPLOYABLE_TTA')
    common.assert_equivalent(control,OLD/'M_FROZEN__OFFICIAL_DANCE_COMPLETE')
    methods.append(control);reused.append(control)
    for name in sealed['methods']:
        rank=common.rank_within_groups(scores[name],groups,g)
        for label,order,budget in [('FROZEN_AT_'+name+'_COUNT',frozen_rank,counts[name]),
            (name+'_AT_FROZEN_COUNT',rank,counts['FROZEN']),
            (name+'_AT_OFFICIAL_COUNT',rank,counts['OFFICIAL_DANCE_COMPLETE'])]:
            mask=order<budget[groups]
            if not np.array_equal(np.bincount(groups,weights=mask,minlength=g).astype(np.int32),budget):
                raise ValueError('per-frame count control budget mismatch')
            common.write_selection(label,mask,meta,lines,protocol='POST_HOC_COUNT_RANK_DIAGNOSTIC_NOT_INDEPENDENT_DEPLOYABLE_TTA')
            methods.append(label)
    if attempts:raise RuntimeError('GT-free control generator touched target GT')
    result=dict(status='ALL_COUNT_RANK_CONTROLS_SEALED_BEFORE_GT',methods=methods,reused_reference_methods=reused,
        new_native_methods=list(sealed['methods']),GT_free_target_open_attempts=0,
        CPU_wall_seconds=time.perf_counter()-started,score_cache_paths={n:str(cache/(n+'.npy')) for n in sealed['methods']},
        score_cache_sha256={n:file_sha(cache/(n+'.npy')) for n in sealed['methods']},
        controls_inherit_parent_inference_and_training_cost=True)
    atomic_json(RUN/'controls_seal.json',result);atomic_json(PUBLIC/'controls_seal.json',result)
    print(json.dumps(dict(event='T2V_COUNT_RANK_SEALED',methods=len(methods))),flush=True)


if __name__=='__main__':prepare()
