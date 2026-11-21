"""Post-seal error accounting: track coverage is not actual anchor coverage."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import metadata,PAIRS,FrozenInputs,OUT as DIAG,baseline_mask
from flex_uda.active.color_cohort import part_mask
from scripts.a100_visual_classical_prepare import OUT


def run(seed=0):
    results={};group='strong_cover_seed'+str(seed);policy='COVER' if seed==0 else 'COVER_S'+str(seed)
    for pair in ('C4','C1'):
        completion=json.loads((OUT/group/pair/'completion.json').read_text());assert completion['GT_open_attempts']==completion['forbidden_read_attempts']==0
        _,target=PAIRS[pair];m=metadata(target)
        rawaudit=json.loads((DIAG/'audit'/pair/'completion.json').read_text())
        truth={(r['video'],r['query']):bool(r['query_present']) for r in rawaudit['episodes']}
        totals=Counter();methods={k:Counter() for k in ('COHORT','SOFT_KD','HARD_KD','SOFT_KD_400','HARD_KD_400','NO_GRAPH','STRONG_PRIOR')};rows=[]
        for video in sorted(m['videos']):
            fc=FrozenInputs(pair,video);labels=np.load(DIAG/'offline_GT'/pair/video/'labels.npy')>=0
            responses=json.loads((OUT/'causal_teacher'/pair/policy/(video+'.json')).read_text())['answers']
            lookup={tuple(map(int,o)):i for i,o in enumerate(fc.observations)}
            old=baseline_mask(pair,video)
            masks={k:np.load(OUT/group/pair/k/'videos'/video/'mask.npy') for k in methods}
            for qi,q in enumerate(fc.queries):
                present=truth[video,q['expression_id']];selected=[r for r in responses if r['local_query']==qi]
                asked_tracks={int(r['track_id']) for r in selected};track_hit=any(labels[qi,fc.by_track[t]].any() for t in asked_tracks)
                anchors=[lookup[int(r['frame_id']),int(r['track_id'])] for r in selected]
                anchor_hit=bool(labels[qi,anchors].any());static=bool(part_mask(q['query_text']).any())
                count=dict(queries=1,query_present=int(present),positive_candidate_query=int(labels[qi].any()),
                    no_positive_track_asked=int(present and not track_hit),no_positive_anchor_asked=int(present and not anchor_hit),
                    positive_track_seen_but_not_at_queried_anchor=int(present and track_hit and not anchor_hit),
                    source_zero_present_query=int(present and not old[qi].any()))
                if static:
                    count['static_queries']=1
                    for r,i in zip(selected,anchors):
                        fraction=float(labels[qi,fc.by_track[int(r['track_id'])]].mean())
                        count['static_queried_anchors']=count.get('static_queried_anchors',0)+1
                        count['static_anchor_sign_differs_from_track_majority']=count.get('static_anchor_sign_differs_from_track_majority',0)+int(bool(labels[qi,i])!=(fraction>.5))
                totals.update(count);detail=dict(video=video,query=q['expression_id'],query_text=q['query_text'],static_appearance_query=static,**count,outputs={})
                for name,total in methods.items():
                    bits=masks[name][qi];positive=bits&labels[qi]
                    c=dict(positive_selected_rows=int(positive.sum()),negative_selected_rows=int((bits&~labels[qi]).sum()),
                        missed_positive_rows=int((~bits&labels[qi]).sum()),selected_rows=int(bits.sum()),
                        present_zero_output=int(present and not bits.any()),empty_query_output=int(not present and bits.any()),
                        old_zero_now_has_matched_positive=int(present and not old[qi].any() and positive.any()),
                        no_positive_anchor_but_now_has_matched_positive=int(present and not anchor_hit and positive.any()))
                    total.update(c);detail['outputs'][name]=c
                rows.append(detail)
        results[pair]=dict(seed=seed,acquisition=dict(totals),outputs={k:dict(v) for k,v in methods.items()},queries=rows,
            GT_only_after_this_seed_predictions_sealed=True,
            labels='IoU>=0.5 matched candidate observations, not an oracle of pure semantic correctness',
            anchor='actual frame shown to Qwen; a correct track at some other frame is not counted as a correct anchor',
            error_counts_not_additive_HOTA_losses=True,all_development_data=True)
        print(json.dumps(dict(event='STRONG_METHOD_OFFLINE_DIAGNOSIS',pair=pair,seed=seed,acquisition=dict(totals),
            output_query_errors={k:{f:v[f] for f in ('present_zero_output','empty_query_output','old_zero_now_has_matched_positive')} for k,v in methods.items()})),flush=True)
    atomic_json(OUT/'analysis'/('strong_diagnosis_seed'+str(seed)+'.json'),results)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);a=p.parse_args();run(a.seed)
