"""Independent metadata-only reconstruction of every held-out teacher scope.

Does not import the forecast implementation, read RMOT model outputs, or use
teacher answer values as a score. Uses Python bisect and raw AUX frame traces.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import file_sha,atomic_json
from flex_uda.a100.mechanism import generation_guard

DEST=ROOT/'a100_runs/visual_classical_20261121/supervisor/conditional_state_teacher_forecast_20261127'


def main():
    guard=generation_guard();plan_path=DEST/'registration.json';inventory_path=DEST/'inventory.json'
    output=DEST/'independent_inventory_verification.json'
    if output.exists():raise FileExistsError(output)
    read=lambda p:json.loads(Path(p).read_text())
    plan=read(plan_path);inventory=read(inventory_path)
    assert file_sha(plan_path)==inventory['registration_sha256']
    source_sha=file_sha(__file__);inventory_sha=file_sha(inventory_path)
    for path,expected in plan['inputs'].items():assert file_sha(path)==expected,path
    for path,expected in plan['original48_source_sha256'].items():assert file_sha(ROOT/path)==expected,path
    aux=[json.loads(s) for s in (ROOT/'a100_runs/mechanism_20261121/evidence/responses.jsonl').read_text().splitlines() if s]
    units={(r['pair'],r['video']):r for r in inventory['rows']};totals=Counter();per=defaultdict(Counter)
    assert len(units)==148==len(plan['cases'])
    for case in plan['cases']:
        row=units[case['pair'],case['evaluated_video']]
        obs=np.load(case['observations'],allow_pickle=False);queries=read(case['temporal_queries'])['queries']
        assert len({tuple(o) for o in obs})==len(obs)
        by_track=defaultdict(list)
        for i,(frame,track) in enumerate(obs):by_track[int(track)].append((int(frame),i))
        for track in by_track:by_track[track].sort()
        delta={'Dance':2,'KITTI-V2':1,'SportsMOT':3}[case['target']]
        def event(reply):
            q,node=divmod(int(reply['uid']),len(obs));frame,track=map(int,obs[node])
            actual=[node]
            if reply['effective_view']=='CLIP':
                options=by_track[track];times=[p[0] for p in options];actual=[]
                for wanted in [frame-3*delta,frame-2*delta,frame-delta,frame]:
                    k=bisect_right(times,wanted)-1
                    if k>=0:actual.append(options[k][1])
                assert len(actual)>=2 and actual[-1]==node
            return dict(uid=reply['uid'],q=q,node=node,text=queries[q]['query_text'],view=reply['effective_view'],
                scope=sorted(set(actual),key=lambda i:int(obs[i,0])),actual=actual,request_id=reply['teacher_request_id'])
        uniform=[event(json.loads(s)) for s in Path(case['uniform_answers']).read_text().splitlines() if s]
        active=[event(json.loads(s)) for s in Path(case['active_answers']).read_text().splitlines() if s]
        normalize=lambda t:' '.join(t.split()).casefold()
        prior=set();requests=set();full_frames=set();physical=set()
        for e in uniform:
            requests.add(e['request_id'])
            prior.update((normalize(e['text']),int(obs[i,0]),int(obs[i,1])) for i in e['scope'])
            if e['view']=='FRAME_BOX':full_frames.add(int(obs[e['node'],0]))
        aux_keys={tuple(k) for k in read(case['auxiliary_keys'])['keys'] if k[0]==case['evaluated_video']}
        for a in aux:
            key=(a['video_id'],a['query_id'],int(a['track_id']),int(a['frame_id']))
            if a['target']!=case['target'] or key not in aux_keys:continue
            frames=a['clip']['frames'] if a['effective_view']=='CLIP' else [int(a['frame_id'])]
            physical.update((int(t),int(a['track_id'])) for t in frames)
            if a['effective_view']=='FRAME_BOX':full_frames.update(map(int,frames))
        kept={};rejected=Counter()
        for e in active:
            points=[tuple(map(int,obs[i])) for i in e['scope']]
            if e['request_id'] in requests:reason='same_teacher_request'
            elif any((normalize(e['text']),f,t) in prior for f,t in points):reason='same_literal_query_track_any_scope_overlap'
            elif any(p in physical for p in points):reason='AUX_actual_physical_scope_any_query'
            elif any(f in full_frames for f,_ in points):reason='training_or_AUX_full_frame_image_overlap'
            else:reason=None
            if reason:rejected[reason]+=1
            else:kept[e['uid']]=e
        actual={e['uid']:e for e in row['retained']};assert len(actual)==len(row['retained'])
        assert actual==kept,(case['pair'],case['evaluated_video'])
        assert dict(rejected)==row['rejections'] and len(active)==row['active_events'] and len(uniform)==row['uniform_training_events']
        supported=len({e['q'] for e in kept.values()})
        assert row['queries']==len(queries) and row['supported_queries']==supported
        assert row['retained_events']==len(kept) and row['empty_queries']==len(queries)-supported
        counts=dict(videos=1,queries=len(queries),active_events=len(active),uniform_training_events=len(uniform),
            retained_events=len(kept),supported_queries=supported,empty_queries=len(queries)-supported)
        totals.update(counts);per[case['pair']].update(counts)
    for pair,values in per.items():
        for key,value in values.items():assert inventory['settings'][pair][key]==value,(pair,key)
    assert not guard and file_sha(__file__)==source_sha and file_sha(inventory_path)==inventory_sha
    for path,expected in plan['inputs'].items():assert file_sha(path)==expected,path
    result=dict(status='PASS_INDEPENDENT_ALL148_SCOPE_EXCLUSIONS_AND_SUPPORT_COUNTS',totals=totals,
        source_sha256=source_sha,inventory_sha256=inventory_sha,source_forecaster_not_imported=True,
        raw_AUX_frame_traces_and_Python_bisect=True,no_RMOT_scores_quality_GT_new_neural_teacher_or_GPU=True)
    atomic_json(output,result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
