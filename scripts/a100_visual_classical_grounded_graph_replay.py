"""GT-free equivalence of shared graph code to the156-study's strict inputs."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import metadata,PAIRS,generation_guard
from flex_uda.active.grounded_graph import refine_queries
from flex_uda.active.acquisition import exclusive_group
from scripts.a100_visual_classical_rgb_parity import PARITY,source_spec
from scripts.a100_visual_classical_strict_prefix import plain


def main():
    guard=generation_guard();changed=0;checked=0;checklist_guard=0
    for pair in ('C4','C1'):
        target=PAIRS[pair][1];m=metadata(target);base=PARITY/pair/'seed0'
        for video,bank in sorted(m['videos'].items()):
            queries=[q for q in m['episodes'] if q['video_id']==video];texts=[q['query_text'] for q in queries];obs=np.asarray(bank['observations'])
            source=source_spec(pair)[0]/'source'/video
            with np.load(source/'sparse_inputs.npz') as a:pcd,colours,tracks=a['pcd'],a['colours'],a['tracks']
            lookup={int(t):i for i,t in enumerate(tracks)};cols=np.asarray([lookup[int(t)] for _,t in obs]);first=np.asarray([np.flatnonzero(cols==i)[0] for i in range(len(tracks))])
            unary=np.load(base/'RGB_KD_MEAN/videos'/video/'scores.npy')[:,first];teacher=np.full(unary.shape,np.nan)
            for r in json.loads((source/'answers.json').read_text())['answers']:teacher[int(r['local_query']),int(r['local_track'])]=r['yes_probability']
            score=refine_queries(unary,pcd,colours,texts,teacher,kind='combined')
            mask=plain(score[:,cols],exclusive_group(texts,target));expected=np.load(base/'RGB_GRAPH_MEAN/videos'/video/'mask.npy')
            changed+=int(np.count_nonzero(mask!=expected));checked+=1
            if pair=='C1':
                a=refine_queries(unary,pcd,colours,texts,teacher,kind='pcd')
                b=refine_queries(unary,pcd,colours,texts,teacher,kind='pcd',positive_guard=True,early=unary,final=unary,protocol='CHECKLIST')
                checklist_guard+=int(np.count_nonzero(a!=b))
    assert changed==checklist_guard==len(guard)==0,(changed,checklist_guard,guard)
    result=dict(status='PASS_SHARED_GROUNDED_GRAPH_REPLAY',videos=checked,changed_candidate_rows=changed,
        CHECKLIST_guard_changed_probabilities=checklist_guard,GT_open_attempts=0,new_teacher_or_source_calls=0,new_head_updates=0,
        existing_combined_graph_preserved=True,not_new_quality_or_cost_measurement=True)
    atomic_json(ROOT/'docs/a100_visual_classical_20261121/grounded_graph_replay.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
