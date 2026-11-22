"""No-GT integration replay; no new model calls, quality or cold-time claim."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import metadata,PAIRS,generation_guard
from flex_uda.active.acquisition import exclusive_group,expression_types
from flex_uda.active.appearance_runtime import empty_appearance_history,regularize_appearance,commit_appearance_history
from scripts.a100_visual_classical_joint_visual_launch import conditions
from scripts.a100_visual_classical_rgb_parity import source_spec,PARITY
from scripts.a100_visual_classical_strict_prefix import plain
from scripts.a100_visual_classical_canonical_deployment_replay import deployment_graph


def main():
    records=conditions();assert all((Path(r['folder'])/'completion.json').is_file() for r in records)
    guard=generation_guard();checks=0;error=0.;changed=0;maps=0;per_condition=[]
    for record in records:
        pair,seed=record['pair'],record['seed'];root=Path(record['folder']);source=source_spec(pair)[0]
        m=metadata(PAIRS[pair][1]);history=empty_appearance_history();condition_checks=0
        for video,bank in sorted(m['videos'].items()):
            queries=[e for e in m['episodes'] if e['video_id']==video];texts=[q['query_text'] for q in queries]
            # The actual cold worker also has a source motion gate. These two
            # registered directions have no such query, checked explicitly.
            assert not any('MOTION' in expression_types(text) for text in texts)
            with np.load(source/'source'/video/'sparse_inputs.npz') as a:
                pcd,colours,prior,tracks=a['pcd'],a['colours'],a['prior'],a['tracks']
            obs=np.asarray(bank['observations']);lookup={int(t):i for i,t in enumerate(tracks)}
            columns=np.asarray([lookup[int(t)] for _,t in obs]);lengths=np.asarray([(obs[:,1]==t).sum() for t in tracks])
            teacher=np.full(prior.shape,np.nan)
            for answer in json.loads((source/'source'/video/'answers.json').read_text())['answers']:
                teacher[int(answer['local_query']),int(answer['local_track'])]=answer['yes_probability']
            with np.load(root/'videos'/video/'coarse_scores.npz') as a:
                unary=a['KD'];expected={name:a[name] for name in ('CAL','JOINT')}
            graph=deployment_graph(unary,pcd,colours,texts,teacher,'combined',np.full(unary.shape,.1))
            for name,base in [('CAL',unary),('JOINT',graph)]:
                actual,inputs,receipt=regularize_appearance(base,colours,texts,lengths,teacher,tracks,video,history)
                maximum=float(np.max(np.abs(actual-expected[name])));error=max(error,maximum)
                mask=plain(actual[:,columns],exclusive_group(texts,PAIRS[pair][1]))
                difference=int(np.count_nonzero(mask!=np.load(root/name/'videos'/video/'mask.npy')))
                assert maximum==difference==0,(pair,seed,video,name,maximum,difference)
                checks+=1;condition_checks+=1;changed+=difference;maps+=receipt['crossfit_models']
            commit_appearance_history(history,inputs,teacher)
        per_condition.append(dict(pair=pair,seed=seed,video_method_checks=condition_checks))
    assert not guard
    result=dict(status='PASS_APPEARANCE_DEPLOYMENT_NUMERICAL_REPLAY',conditions=per_condition,video_method_checks=checks,
        maximum_probability_error=error,changed_candidate_rows=changed,crossfit_map_models_checked=maps,
        GT_open_attempts=0,new_teacher_calls=0,new_encoder_calls=0,new_head_fits=0,
        actual_cold_worker_executed=False,no_new_quality_or_cold_timing_claim=True,
        implementation_sha256={name:file_sha(ROOT/path) for name,path in dict(glue='flex_uda/active/appearance_runtime.py',
            cold_worker='scripts/a100_visual_classical_cold_full_c4.py',calibration='flex_uda/active/semantic_colour.py').items()})
    atomic_json(ROOT/'docs/a100_visual_classical_20261121/appearance_deployment_replay.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
