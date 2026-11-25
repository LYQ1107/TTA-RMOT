"""Evidence preparation for protocol-aware search, with a frozen QA broker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,append_json,file_sha
from flex_uda.a100.mechanism import PAIRS,metadata,generation_guard
from flex_uda.active.coverage_search import CoverageSearch
from flex_uda.active.visual_classical import visual_graph,acquisition_order
from scripts.a100_visual_classical_rgb_parity import source_spec
from scripts.a100_visual_classical_prepare import OUT

SEARCH=OUT/'coverage_search'
EVIDENCE=SEARCH/'evidence'


def problem(video,bank,queries,source):
    with np.load(source/'source'/video/'sparse_inputs.npz') as a:
        pcd,prior,tracks,anchors=a['pcd'],a['prior'],a['tracks'],a['anchors']
    observations=np.asarray(bank['observations']);lengths=np.asarray([(observations[:,1]==t).sum() for t in tracks])
    eligible=np.flatnonzero(lengths>=3);budget=int(np.floor(.1*len(queries)*len(eligible)));orders=[]
    for qi in range(len(queries)):
        _,distance=visual_graph(pcd[qi]);seed=int(hashlib.sha256((video+str(qi)).encode()).hexdigest()[:8],16)
        orders.append(acquisition_order(prior[qi],distance,eligible,'COVER',np.random.default_rng(seed)))
    return orders,budget,tracks,anchors


def collect():
    guard=generation_guard();begin=time.perf_counter();pair='C4';_,target=PAIRS[pair];m=metadata(target);source=source_spec(pair)[0]
    root=EVIDENCE/pair;root.mkdir(parents=True,exist_ok=True)
    if (root/'completion.json').is_file():return
    cache=root/'raw.sqlite'
    if not cache.is_file():
        with sqlite3.connect('file:'+str(source/'raw.sqlite')+'?mode=ro',uri=True) as original,sqlite3.connect(cache) as copied:original.backup(copied)
    from scripts.active_run import video_spec
    from flex_uda.a100.runtime import teacher
    stamp=time.perf_counter();client=teacher([video_spec(target,v,b) for v,b in m['videos'].items()],cache);load=time.perf_counter()-stamp
    protocol=json.loads((OUT/'preregistration_v1.json').read_text())['protocol'][pair];receipts=[]
    for vi,(video,bank) in enumerate(sorted(m['videos'].items())):
        local=root/'source'/video;local.mkdir(parents=True,exist_ok=True)
        if (local/'answers.json').is_file():
            receipts.append(json.loads((local/'receipt.json').read_text()));continue
        stamp=time.perf_counter();queries=[q for q in m['episodes'] if q['video_id']==video]
        orders,budget,tracks,anchors=problem(video,bank,queries,source);selector=CoverageSearch(orders,protocol);answers=[];planning=0.;service=0.
        for offset in range(0,budget,8):
            t=time.perf_counter();chosen=selector.select_batch(min(8,budget-offset));planning+=time.perf_counter()-t;requests=[]
            for item in chosen:
                qi,ti=item['local_query'],item['local_track'];q=queries[qi]
                requests.append(dict(target=target,video_id=video,query_id=q['expression_id'],query_text=q['query_text'],track_id=int(tracks[ti]),frame_id=int(anchors[ti]),view='CROP'))
            append_json(root/'request_journal.jsonl',dict(stage='SELECTED_BEFORE_ANSWERS',video=video,offset=offset,positions=chosen,requests=requests))
            t=time.perf_counter();returned=client.ask_many(requests);service+=time.perf_counter()-t
            for item,value in zip(chosen,returned):
                qi,ti=item['local_query'],item['local_track'];selector.consume(qi,ti,value['yes_probability'])
                row=dict(value,local_query=qi,local_track=ti);answers.append(row);append_json(root/'responses.jsonl',dict(row,video=video))
        assert len(answers)==budget and len({(a['local_query'],a['local_track']) for a in answers})==budget and not selector.pending
        if budget>=len(queries):assert np.all(selector.count>=1)
        old=json.loads((source/'source'/video/'answers.json').read_text())['answers'];old_keys={(a['local_query'],a['local_track']) for a in old}
        receipt=dict(video=video,index=vi,logical_calls=budget,cache_hits=sum(a['cache_hit'] for a in answers),
            changed_question_positions=len({(a['local_query'],a['local_track']) for a in answers}-old_keys),
            per_query_calls=selector.count.tolist(),acquisition_CPU_seconds=planning,teacher_service_wall_seconds=service,GT_open_attempts=len(guard),wall_seconds=time.perf_counter()-stamp)
        atomic_json(local/'receipt.json',receipt);atomic_json(local/'answers.json',dict(answers=answers,logical_calls=budget,GT_open_attempts=len(guard)))
        receipts.append(receipt);print(json.dumps(dict(receipt,event='COVERAGE_SEARCH_EVIDENCE_VIDEO',completed=vi+1,total=45)),flush=True)
    t=time.perf_counter();cost=client.close();close=time.perf_counter()-t
    assert not guard and sum(r['logical_calls'] for r in receipts)==1160
    atomic_json(root/'completion.json',dict(status='SEALED_COVERAGE_SEARCH_TEACHER_EVIDENCE',videos=receipts,teacher=cost,teacher_load_wall_seconds=load,teacher_close_seconds=close,
        logical_calls=1160,GT_open_attempts=0,selection_reads_only_requested_answers=True,source_inputs=str(source),private_cache_seeded_from_frozen_COVER=True,
        copied_cache_quality_preparation_not_cold_deployment=True,implementation_sha256=file_sha(ROOT/'flex_uda/active/coverage_search.py'),wall_seconds=time.perf_counter()-begin,physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES')))


if __name__=='__main__':collect()
