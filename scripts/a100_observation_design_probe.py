"""Unlabelled real-input probe of the proposed coverage/excitation planner.

This is not teacher collection or a quality experiment. Historical request
query/track IDs define only a sample quota/coverage reference; answers ignored.
"""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_uda.a100.mechanism import PAIRS,metadata,history,generation_guard
from flex_uda.active.observation_design import plan
from flex_uda.active.relational_transport import horizontal_direction
from flex_uda.active.relational_features import geometry
from scripts.a100_six_direction import source,PUBLIC
from scripts.a100_relation_transport import coarse_features
from flex_tta.runtime.pf_receipts import atomic_json


def main():
    guard=generation_guard();result=[]
    for pair in ('C1','C2','C4'):
        m=metadata(PAIRS[pair][1]);video,bank=next(iter(sorted(m['videos'].items())))
        queries=[q for q in m['episodes'] if q['video_id']==video];obs=np.asarray(bank['observations'])
        coords,width,height=geometry(bank,PAIRS[pair][1])
        with np.load(source(pair)/'source'/video/'sparse_inputs.npz') as a:pcd,colours,prior,tracks,anchors=[a[k] for k in ('pcd','colours','prior','tracks','anchors')]
        raw=coarse_features(pcd,colours,prior,tracks,obs,[q['query_text'] for q in queries])
        qi=next((i for i,q in enumerate(queries) if horizontal_direction(q['query_text'])),0)
        samples=[];positions=[];lengths=[]
        for track in tracks:
            ids=np.flatnonzero(obs[:,1]==track);chosen=ids[np.unique(np.linspace(0,len(ids)-1,min(3,len(ids))).round().astype(int))]
            samples.append([int(obs[i,0]) for i in chosen]);positions.append({int(obs[i,0]):float(coords[i,0]) for i in chosen});lengths.append(len(ids))
        records=json.loads((history(pair)/'videos'/video/'questions.json').read_text())
        used=[r for r in records if r['query_id']==queries[qi]['expression_id']];budget=len(used)
        actions,receipt=plan(raw[qi],np.asarray(lengths,float),prior[qi],samples,anchors,positions,budget,horizontal_direction(queries[qi]['query_text']))
        assert len(actions)==budget and not guard
        result.append(dict(pair=pair,video=video,query_id=queries[qi]['expression_id'],tracks=len(tracks),
            historical_request_count_as_budget=budget,historical_distinct_requested_tracks=len({r['track_id'] for r in used}),
            direction=horizontal_direction(queries[qi]['query_text']),paid_anchors=receipt['paid_anchors'],state_contrasts=receipt['state_contrasts'],
            distinct_objects=receipt['distinct_objects'],new_teacher_GPU_or_GT_calls=0,all_planned_frames_are_actual_source_samples=True))
    assert not guard
    report=dict(status='OBSERVATION_DESIGN_PROTOTYPE_MATH_AND_REAL_INPUT_CHECKS_PASS',
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('flex_uda/active/observation_design.py','tests/test_observation_design.py','scripts/a100_observation_design_probe.py')},
        tests_passed=3,real_probes=result,
        interpretation='Geometry-only prototype planning checks, not a collected active-policy experiment. Historical raw QA values are ignored; query/track IDs used only for quota/coverage reference.',
        no_new_teacher_or_GPU_calls=True,manual_GT_open_attempts=0,new_candidate_predictions=0,
        theory='Coverage plus log of summed per-object maximum contrast excitation is conditionally monotone submodular for fixed paid anchors/actions/nonnegative weights and equal one-question costs. Greedy1-1/e concerns conditional proxy gain only.',
        not_true_Fisher_information=True,not_global_two_stage_or_HOTA_or_wall_time_guarantee=True,goal_status='ACTIVE')
    atomic_json(PUBLIC/'observation_design_prototype.json',report);print(json.dumps(report))


if __name__=='__main__':main()
