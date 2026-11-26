"""Prepare chronological full-six paid inputs with the shared9352D bank.

This does not train an old rejected method. Main-video evidence is stored
separately by video; the existing collection-transductive100 auxiliary
events are a distinct block. Future learners may only join past/current
video blocks, with exactly the same evidence supplied to ordinary KD.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.a100_six_direction import OUT as SIX

OUT=SIX/'fullsix_learning_common_20261125'
PLAN=OUT/'registration.json'
PUBLIC=ROOT/'docs/a100_six_direction_20261122'
SOURCES=('scripts/a100_fullsix_learning_inputs.py','flex_uda/active/full_appearance_learning_inputs.py',
 'flex_uda/active/appearance_learning_inputs.py','scripts/a100_dino_fullsix_collect.py',
 'scripts/a100_dino_appearance_collect.py','flex_uda/active/joint_observation_inputs.py',
 'flex_uda/active/observation_learning_inputs.py','flex_uda/active/factor_rsi_learning.py',
 'flex_uda/active/native_binary_inputs.py','scripts/a100_observation_local_cache.py',
 'flex_uda/a100/mechanism.py','flex_uda/a100/runtime.py','configs/teacher/prompts_v1.yaml')


def prepare():
    from flex_uda.a100.mechanism import PAIRS,metadata,generation_guard
    from scripts.a100_dino_fullsix_collect import OUT as FULL
    guard=generation_guard();assert not PLAN.exists()
    complete=json.loads((FULL/'completion.json').read_text())
    assert complete['rows']==complete['valid_rows']==517686
    alignment=SIX/'full_appearance_input_alignment_20261125/completion.json'
    assert json.loads(alignment.read_text())['exact_old_3844D_rows']==17399
    cases=[dict(pair=pair,model=model,target=target,videos=sorted(metadata(target)['videos']))
           for pair,(model,target) in PAIRS.items()]
    report=dict(status='REGISTERED_FULL_SIX_CHRONOLOGICAL_COMMON_INPUTS',cases=cases,
        source_sha256={n:file_sha(ROOT/n) for n in SOURCES},
        full_appearance_completion_sha256=file_sha(FULL/'completion.json'),
        full_appearance_alignment_sha256=file_sha(alignment),
        main_video_units=sum(len(c['videos']) for c in cases),auxiliary_blocks=6,
        features='Historical5508D original query/anchor/current/coarse and observable state, plus3844D shared frozen image appearance. No query-expanded dense tensor.',
        evidence='Each setting own sealed MINUS_SLOW history questions; original literal query/prompts/actual effective views preserved. Repeats within an event/effective view averaged with explicit counts.',
        auxiliary='Same100three-view target-collection events used in the prior pilot, separately stored per source setting; explicitly transductive auxiliary information, never an independent validation split.',
        chronological_contract='Never feed future-main-video answers to an earlier video head. Main blocks and auxiliary block separate; repeated events can be joined by per-view answer counts. No model implementation or quality acceptance is defined here.',
        same_inputs_required_for_all_future_ordinary=True,
        no_new_teacher_backbone_training_predictions_or_GT=True,
        CPU_workers=3,all_data_DEV=True,goal_status='ACTIVE')
    assert report['main_video_units']==148 and not guard
    atomic_json(PLAN,report);atomic_json(PUBLIC/'fullsix_learning_common_registration_20261125.json',report)
    print(json.dumps(dict(status=report['status'],main_video_units=148,auxiliary_blocks=6)),flush=True)


def registration():
    plan=json.loads(PLAN.read_text())
    for n,sha in plan['source_sha256'].items():assert file_sha(ROOT/n)==sha,n
    return plan


def worker(case):
    import numpy as np
    import yaml
    from flex_uda.a100.mechanism import generation_guard,history,OUT as BANK
    from flex_uda.active.joint_observation_inputs import JointObservationVideo,use_verified_local_features
    from flex_uda.active.factor_rsi_learning import Inputs
    from flex_uda.active.full_appearance_learning_inputs import augment_paid
    from flex_uda.teacher.decompose_inputs import CLIP_PROMPT,FRAME_PROMPT
    plan=registration();guard=generation_guard();use_verified_local_features()
    started=time.perf_counter();pair,target=case['pair'],case['target']
    templates=yaml.safe_load((ROOT/'configs/teacher/prompts_v1.yaml').read_text())
    pattern=re.escape(templates['official_single']).replace(re.escape('{coord}'),r'-?\d+(?:\.\d+)? -?\d+(?:\.\d+)?')
    pattern=pattern.replace(re.escape('{sentence}'),'__LITERAL_QUERY__');before,after=pattern.split('__LITERAL_QUERY__')
    aux_path=BANK/'evidence/responses.jsonl';aux_receipt=json.loads((BANK/'evidence/completion.json').read_text())
    assert file_sha(aux_path)==aux_receipt['responses_sha256']
    original_sha=aux_receipt['teacher']['teacher_weight_sha256']
    auxiliary=[json.loads(s) for s in aux_path.read_text().splitlines()]
    auxiliary=[r for r in auxiliary if r['target']==target];assert len(auxiliary)==300
    reports=[]
    for unit in ('AUX',*case['videos']):
        folder=OUT/pair/unit;folder.mkdir(parents=True,exist_ok=True);final=folder/'completion.json'
        if final.exists():
            receipt=json.loads(final.read_text());assert receipt['registration_sha256']==file_sha(PLAN)
            for leaf,sha in receipt['output_sha256'].items():assert file_sha(folder/leaf)==sha
            reports.append(receipt);continue
        path=aux_path if unit=='AUX' else history(pair)/'videos'/unit/'questions.json'
        rows=auxiliary if unit=='AUX' else json.loads(path.read_text())
        groups=defaultdict(list)
        for row in rows:
            assert row['frozen_weight_sha256']==original_sha
            view=row['effective_view'];text=row['query_text'];prompt=row['input']['prompt']
            assert row['input']['method']==view
            if view=='CROP':assert re.fullmatch(before+re.escape(text)+after,prompt)
            else:assert prompt==(CLIP_PROMPT if view=='CLIP' else FRAME_PROMPT).format(expression=text)
            key=row['video_id'],row['query_id'],int(row['track_id']),int(row['frame_id'])
            if unit!='AUX':assert key[0]==unit
            groups[key].append(row)
        keys=sorted(groups);probability=np.full((len(keys),3),np.nan,dtype='float64')
        counts=np.zeros((len(keys),3),dtype='int32')
        for i,key in enumerate(keys):
            for j,view in enumerate(('CROP','CLIP','FRAME_BOX')):
                values=[r['yes_probability'] for r in groups[key] if r['effective_view']==view]
                if values:probability[i,j]=np.mean(values);counts[i,j]=len(values)
        base=np.empty((len(keys),5508),dtype='float32')
        for video in sorted({k[0] for k in keys}):
            observation=JointObservationVideo(pair,video);inputs=Inputs(observation)
            qi={q['expression_id']:i for i,q in enumerate(observation.queries)}
            ti={int(t):i for i,t in enumerate(observation.tracks)}
            selected=[i for i,k in enumerate(keys) if k[0]==video];records=[]
            for i in selected:
                _,query,track,frame=keys[i]
                assert all(r['query_text']==observation.queries[qi[query]]['query_text'] for r in groups[keys[i]])
                records.append(dict(local_query=qi[query],local_track=ti[track],track_id=track,frame_id=frame))
            a=inputs.arrays(records)
            base[selected]=np.column_stack((a['bags'].reshape(len(selected),-1),a['present'],a['current'],a['coarse']))
        features=augment_paid(base,keys,target)
        assert np.array_equal(features[:,:5508],base) and np.isfinite(features).all()
        assert np.isfinite(probability).any(1).all()
        np.savez(folder/'teacher_inputs.npz',features=features,probability=probability,response_counts=counts)
        atomic_json(folder/'row_keys.json',dict(keys=keys,view_order=['CROP','CLIP','FRAME_BOX']))
        receipt=dict(pair=pair,target=target,unit=unit,events=len(keys),raw_responses=len(rows),
            input_dimension=9352,teacher_weight_sha256=original_sha,source_answers_sha256=file_sha(path),
            original5508_coordinates_byte_equal=True,GT_reads=0,new_teacher_or_neural_updates=0,
            registration_sha256=file_sha(PLAN),output_sha256={leaf:file_sha(folder/leaf) for leaf in ('teacher_inputs.npz','row_keys.json')})
        atomic_json(final,receipt);reports.append(receipt)
        print(json.dumps(dict(status='FULL_SIX_LEARNING_INPUT_UNIT_COMPLETE',**receipt)),flush=True)
    assert len(reports)==len(case['videos'])+1 and not guard
    result=dict(pair=pair,target=target,units=reports,wall_seconds=time.perf_counter()-started)
    atomic_json(OUT/pair/'completion.json',result);return result


def main():
    plan=registration();started=time.perf_counter()
    print(json.dumps(dict(status='STARTED_FULL_SIX_COMMON_PAID_INPUT_PREPARATION',CPU_workers=3)),flush=True)
    with ProcessPoolExecutor(max_workers=3) as pool:reports=list(pool.map(worker,plan['cases']))
    summary=[dict(pair=r['pair'],target=r['target'],main_video_units=len(r['units'])-1,
        main_events=sum(u['events'] for u in r['units'] if u['unit']!='AUX'),
        auxiliary_events=next(u['events'] for u in r['units'] if u['unit']=='AUX'),wall_seconds=r['wall_seconds']) for r in reports]
    result=dict(status='COMPLETE_FULL_SIX_CHRONOLOGICAL_COMMON_LEARNING_INPUTS',cases=summary,
        main_video_units=148,auxiliary_blocks=6,dimension=9352,wall_seconds=time.perf_counter()-started,
        no_new_teacher_backbone_training_RMOT_predictions_or_GT=True,
        not_a_full_six_quality_or_cold_cost_result=True,goal_status='ACTIVE')
    atomic_json(OUT/'completion.json',result);atomic_json(PUBLIC/'fullsix_learning_common_completion_20261125.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');args=p.parse_args()
    if args.prepare:prepare()
    else:main()
