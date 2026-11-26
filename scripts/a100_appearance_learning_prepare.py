"""Shared9352D learning inputs; no adaptation fitting or task evaluation."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import generation_guard
from flex_uda.active.appearance_learning_inputs import augment_paid,DIMENSION,BASE_DIMENSION,OUT as DINO_OUT,PLAN as DINO_PLAN
from scripts.a100_six_direction import OUT as SIX

OUT=SIX/'appearance_learning_common_20261125'
PUBLIC=ROOT/'docs/a100_six_direction_20261122'


def main():
    guard=generation_guard()
    completed=json.loads((DINO_OUT/'completion.json').read_text())
    assert completed['status']=='COMPLETE_SHARED_FROZEN_QUERY_INDEPENDENT_APPEARANCE_INPUTS'
    old=SIX/'joint_teacher_noise_primary_20261125'
    previous=json.loads((old/'registration.json').read_text())
    cases=[]
    for case in previous['cases']:
        pair=case['pair'];folder=OUT/pair;folder.mkdir(parents=True,exist_ok=True)
        source=old/pair/'teacher_inputs.npz'
        assert file_sha(source)==case['data_sha256']
        keys_path=old/pair/'row_keys.json';assert file_sha(keys_path)==case['row_keys_sha256']
        keys=json.loads(keys_path.read_text())['keys']
        with np.load(source) as saved:arrays={n:saved[n] for n in saved.files}
        original=arrays['features'];arrays['features']=augment_paid(original,keys,case['target'])
        assert arrays['features'].shape==(len(keys),DIMENSION)
        assert np.array_equal(arrays['features'][:,:BASE_DIMENSION],original)
        physical={};duplicate_query_rows=0
        for key,row in zip(keys,arrays['features'][:,BASE_DIMENSION:]):
            identifier=key[0],int(key[2]),int(key[3])
            if identifier in physical:
                assert np.array_equal(physical[identifier],row)
                duplicate_query_rows+=1
            else:physical[identifier]=row.copy()
        destination=folder/'teacher_inputs.npz';np.savez(destination,**arrays)
        (folder/'row_keys.json').write_bytes(keys_path.read_bytes())
        report=dict(pair=pair,target=case['target'],evaluated_video=case['evaluated_video'],
            events=len(keys),input_dimension=DIMENSION,base_dimension_preserved=BASE_DIMENSION,
            base_coordinates_byte_equal=True,appearance_coordinates=3844,
            physical_ROIs=len(physical),query_independence_duplicate_rows_checked=duplicate_query_rows,
            source_data_sha256=case['data_sha256'],data_sha256=file_sha(destination),
            row_keys_sha256=file_sha(folder/'row_keys.json'),GT_open_attempts=0,
            same_visual_teacher_information_for_all_future_models=True)
        atomic_json(folder/'completion.json',report);cases.append(report)
        print(json.dumps(dict(status='SHARED_APPEARANCE_LEARNING_CASE_PREPARED',**report)),flush=True)
    assert not guard
    report=dict(status='COMPLETE_SHARED9352D_APPEARANCE_LEARNING_INPUTS_NO_GT',cases=cases,
        descriptor_input_plan_sha256=file_sha(DINO_PLAN),descriptor_completion_sha256=file_sha(DINO_OUT/'completion.json'),
        source_sha256={name:file_sha(ROOT/name) for name in ('flex_uda/active/appearance_learning_inputs.py','scripts/a100_appearance_learning_prepare.py')},
        new_teacher_answers_neural_training_or_RMOT_quality=0,
        no_final_learning_method_or_three_contribution_acceptance_claim=True,goal_status='ACTIVE')
    atomic_json(OUT/'data_preparation.json',report)
    atomic_json(PUBLIC/'appearance_learning_shared_inputs_20261125.json',report)
    print(json.dumps(dict(status=report['status'],cases=len(cases),input_dimension=DIMENSION)),flush=True)


if __name__=='__main__':main()
