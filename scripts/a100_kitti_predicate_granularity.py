"""Isolated post-seal KITTI predicate/physical-identity granularity diagnosis.

Uses byte-identical restored GT only here, never in adaptation. Candidate
matching and annotation membership are kept distinct. No new model or HOTA
experiment is launched; row-error bounds are not HOTA upper bounds.
"""
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import metadata,OUT as DIAG
from scripts.a100_six_direction import OUT as SIX,PUBLIC
from scripts.analyze_pf_full_offline import load_eval_gt,overlaps,matched_pairs
from scripts.a100_language_scope_audit import classify

OUT=SIX/'offline_GT/kitti_predicate_granularity'


def main():
    barrier=json.loads((SIX/'all_predictions_sealed_before_GT.json').read_text())
    assert barrier['methods']==234 and barrier['adaptation_GT_open_attempts']==0
    restored=SIX/'offline_GT/kitti_original_hash_restore'
    receipt=json.loads((restored/'completion.json').read_text())
    assert receipt['status']=='PASS_ALL862_ORIGINAL_GT_BYTE_HASHES' and receipt['matching_original_files']==862
    start=time.perf_counter();m=metadata('KITTI-V2');all_queries=[];videos=[];physical=Counter()
    for video,bank in sorted(m['videos'].items()):
        obs=np.asarray(bank['observations']);queries=[q for q in m['episodes'] if q['video_id']==video]
        frames=np.asarray([int(float(line.split(',')[0])) for line in bank['lines']])
        offsets=np.unique(frames-obs[:,0]);assert len(offsets)==1,offsets
        boxes=np.asarray([[float(v) for v in line.split(',')[2:6]] for line in bank['lines']]);boxes[:,2:]+=boxes[:,:2]
        by_frame={int(f):np.flatnonzero(frames==f) for f in np.unique(frames)}
        truths=[load_eval_gt(restored,video,q['expression_id']) for q in queries]
        union=defaultdict(dict)
        for truth in truths:
            for frame,rows in truth.items():
                for gid,box in rows:
                    if gid in union[frame]:assert np.array_equal(union[frame][gid],box)
                    union[frame][gid]=box
        ids_union=np.full(len(obs),-1,dtype=np.int64)
        for frame,indices in by_frame.items():
            entries=list(union.get(frame,{}).items());pairs=matched_pairs(overlaps(boxes[indices],[b for _,b in entries]),list(range(len(indices))))
            for i,j in pairs.items():ids_union[indices[i]]=entries[j][0]
        visible=ids_union>=0;semantic=np.zeros((len(queries),len(obs)),bool)
        for qi,truth in enumerate(truths):
            for frame,rows in truth.items():
                indices=by_frame.get(frame)
                if indices is not None:semantic[qi,indices]=np.isin(ids_union[indices],[gid for gid,_ in rows])
        per_query=[Counter(query_track_pairs=0,matched_union_rows=int(visible.sum()),
            annotation_positive_rows_on_matched_union=int((semantic[qi]&visible).sum())) for qi in range(len(queries))]
        vphysical=Counter();changes={}
        masks={p:{name:np.load(SIX/'quality'/p/'seed0'/name/'videos'/video/'mask.npy') for name in ('KD_FINAL','JOINT')} for p in ('C2','C5')}
        for p,methods in masks.items():
            for name,mask in methods.items():assert mask.shape==semantic.shape;changes[p+'/'+name]=0
        for track in sorted(set(obs[:,1])):
            indices=np.flatnonzero(obs[:,1]==track);valid=indices[visible[indices]]
            for p,methods in masks.items():
                for name,mask in methods.items():changes[p+'/'+name]+=int(np.count_nonzero(np.any(mask[:,indices]!=mask[:,indices[:1]],axis=1)))
            if not len(valid):continue
            identities,counts=np.unique(ids_union[valid],return_counts=True)
            vphysical.update(tracks_with_visible_GT=1,tracks_with_multiple_GTIDs=int(len(identities)>1),matched_rows=len(valid),majority_GTID_rows=int(counts.max()))
            positive=semantic[:,valid].sum(1);negative=len(valid)-positive;mixed=(positive>0)&(negative>0)
            nodes=indices[np.unique(np.linspace(0,len(indices)-1,min(3,len(indices))).round().astype(int))]
            nodes=nodes[visible[nodes]];seen=semantic[:,nodes]
            visible_mixed=(seen.any(1)&~seen.all(1)) if len(nodes)>1 else np.zeros(len(queries),bool)
            within=np.zeros(len(queries),np.int64);within_groups=np.zeros(len(queries),np.int64)
            for gid in identities:
                group=valid[ids_union[valid]==gid];n=semantic[:,group].sum(1)
                within+=np.minimum(n,len(group)-n);within_groups+=(n>0)&(n<len(group))
            for qi,row in enumerate(per_query):row.update(query_track_pairs=1,mixed_annotation_membership_tracks=int(mixed[qi]),
                mixed_tracks_visible_in_three_nodes=int(mixed[qi] and visible_mixed[qi]),
                constant_track_minimum_binary_row_errors=int(min(positive[qi],negative[qi])),
                same_GTID_minimum_binary_row_errors=int(within[qi]),same_GTID_mixed_groups=int(within_groups[qi]))
        physical.update(vphysical)
        legacy=np.load(DIAG/'offline_GT/C5'/video/'labels.npy')>=0;assert legacy.shape==semantic.shape
        videos.append(dict(video=video,query_count=len(queries),candidate_rows=len(obs),evaluation_frame_offset_from_observation=int(offsets[0]),
            physical=dict(vphysical),seed0_raw_masks_nonconstant_query_tracks=changes,
            legacy_positive_membership_rows=int(legacy.sum()),union_then_annotation_positive_rows=int(semantic.sum()),
            legacy_vs_union_annotation_disagreement_rows=int((legacy!=semantic).sum())))
        for qi,q in enumerate(queries):
            flags=classify(q['query_text']);all_queries.append(dict(video=video,query=q['expression_id'],text=q['query_text'],
                types=flags['types'],appearance_cue_supported=flags['appearance_cue_available'],
                compositional_colour_without_current_cue=flags['compositional_colour_not_in_current_cue'],counts=dict(per_query[qi])))
    groups={}
    for name,predicate in [('ALL',lambda r:True)]+[(t,lambda r,t=t:t in r['types']) for t in ('APPEARANCE','SPATIAL','MOTION','ROLE','OTHER')]+[
            ('CUE_UNSUPPORTED',lambda r:not r['appearance_cue_supported']),('COMPOSITIONAL_COLOUR_WITHOUT_CUE',lambda r:r['compositional_colour_without_current_cue'])]:
        chosen=[r for r in all_queries if predicate(r)];c=Counter()
        for row in chosen:c.update(row['counts'])
        groups[name]=dict(queries=len(chosen),counts=dict(c),
            constant_track_min_error_fraction_of_matched_rows=c['constant_track_minimum_binary_row_errors']/max(c['matched_union_rows'],1),
            same_GTID_min_error_fraction_of_matched_rows=c['same_GTID_minimum_binary_row_errors']/max(c['matched_union_rows'],1),
            same_GTID_min_error_relative_to_positive_rows=c['same_GTID_minimum_binary_row_errors']/max(c['annotation_positive_rows_on_matched_union'],1))
    result=dict(status='POST_SEAL_KITTI_PREDICATE_GRANULARITY_DIAGNOSIS',groups=groups,videos=videos,physical_identity=dict(physical),
        physical_majority_ID_row_purity=physical['majority_GTID_rows']/max(physical['matched_rows'],1),
        GT_used_only_in_this_isolated_diagnosis=True,manual_target_labels_in_adaptation=False,new_teacher_encoder_or_GPU_calls=0,
        matching='Unmodified IoU>=.5 maximum-cardinality/IoU matcher over union of query GT; then use that matched physical GTID membership in each query annotation at each official evaluation frame.',
        bound_scope='On visible matched-union rows, any per-query constant Track-ID binary decision incurs at least the computed row errors. Same-GTID subgroup changes separate annotation dynamics from ID switches; not a HOTA bound or full semantic accuracy.',
        legacy_scope='Existing cached KITTI membership is binary only and uses its historical diagnostic matching; disagreements are reported, never used to replace the official GT or metric.',
        all234_predictions_preceded_this_diagnosis=True,groups_overlap_not_additive=True,all_data_development=True,
        parameters_or_predictions_selected_by_this_GT=False,wall_seconds=time.perf_counter()-start,goal_status='ACTIVE')
    atomic_json(OUT/'full_query_diagnosis.json',dict(summary=result,queries=all_queries))
    atomic_json(OUT/'completion.json',result)
    (PUBLIC/'kitti_predicate_granularity.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
