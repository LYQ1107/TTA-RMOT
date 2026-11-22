"""Post-I1-seal audit of linguistic event assumptions against raw annotations.

No candidate matching: compare official (frame, original ID) sets, conditional
on shared ID namespaces, same video/horizon/parsed subject. Interpretation
errors and annotation conventions cannot be distinguished by this audit.
No current unified-model prediction, parameter or threshold is consulted.
"""
from collections import Counter,defaultdict
import itertools
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import metadata
from flex_uda.a100.runtime import relocate
from scripts.a100_scoped_supervision_audit import graph
from scripts.a100_query_scope_teacher import OUT as SCOPES,text_key
from scripts.a100_observation_design_quality import OUT as I1
from scripts.a100_six_direction import OUT as SIX,PUBLIC
from flex_uda.a100.kitti_gt_restore import OUT as RESTORED_KITTI

OUT=SIX/'offline_GT/query_scope_assumptions_20261123'
PLAN=PUBLIC/'query_scope_postseal_assumptions_plan.json'


def roots():
    restore=json.loads((RESTORED_KITTI/'completion.json').read_text())
    assert restore['status']=='PASS_ALL862_ORIGINAL_GT_BYTE_HASHES'
    sports=json.loads((ROOT/'results/multidomain_generality/sports_GT_reconstruction_v4.json').read_text())
    return {'Dance':relocate('/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance')/'gt_template',
        'SportsMOT':relocate(sports['evaluation_root'])/'gt_template',
        'KITTI-V2':RESTORED_KITTI/'gt_template'}


def read_rows(path,horizon,namespace):
    result=set();counts=Counter();maximum=0.
    for line in path.read_text().splitlines():
        fields=line.strip().split(',');assert len(fields)>=8,path
        counts['raw_annotation_rows']+=1
        if int(float(fields[6]))==0 or int(float(fields[7]))!=1:
            counts['official_mark_or_class_filtered_rows']+=1;continue
        frame,identity=int(float(fields[0])),int(float(fields[1]))
        if not 1<=frame<=horizon:
            counts['outside_official_horizon_rows']+=1;continue
        box=tuple(float(x) for x in fields[2:6]);key=frame,identity
        if key in namespace:
            error=max(abs(a-b) for a,b in zip(box,namespace[key]))
            maximum=max(maximum,error)
            counts['cross_query_same_ID_geometry_nonidentical']+=int(error>0)
        else:namespace[key]=box
        result.add(key)
    return result,counts,maximum


def pair_counts(queries,annotations,structures):
    result={name:Counter() for name in ('all_accepted','no_ambiguity_notes')}
    witnesses=[]
    for left,right in itertools.combinations(queries,2):
        a,b=left['expression_id'],right['expression_id'];ta,tb=left['query_text'],right['query_text']
        sa,sb=structures[text_key(ta)],structures[text_key(tb)]
        for group in result.values():group['all_query_pairs']+=1
        if int(left['frame_count'])!=int(right['frame_count']):
            for group in result.values():group['different_horizons_not_compared']+=1
            continue
        ga,gb=graph(sa),graph(sb)
        if ga is None or gb is None:continue
        pa,pb=sa['accepted_structure'],sb['accepted_structure']
        if pa['subject']!=pb['subject']:
            for group in result.values():group['different_parsed_subjects_not_compared']+=1
            continue
        eligible=['all_accepted']
        if not sa['all_ambiguity_notes'] and not sb['all_ambiguity_notes']:eligible.append('no_ambiguity_notes')
        if ga==gb:
            difference=annotations[a]^annotations[b]
            for name in eligible:
                c=result[name];c['same_graph_pairs']+=1;c['same_graph_pairs_with_nonempty_GT']+=bool(annotations[a] or annotations[b])
                c['same_graph_pairs_with_different_GT_sets']+=bool(difference)
                c['same_graph_symmetric_difference_GT_rows_pair_repeated']+=len(difference)
                if ta==tb:
                    c['identical_text_pairs']+=1;c['identical_text_pairs_with_different_GT_sets']+=bool(difference)
            if difference:witnesses.append(dict(kind='same_graph_different_GT_sets',left=a,right=b,rows=len(difference)))
        elif ga<gb or gb<ga:
            weak,strong=(a,b) if ga<gb else (b,a)
            violation=annotations[strong]-annotations[weak]
            for name in eligible:
                c=result[name];c['strict_subset_parse_pairs']+=1
                c['nonempty_stronger_GT_pairs']+=bool(annotations[strong])
                c['pairs_with_stronger_GT_outside_weaker_GT']+=bool(violation)
                c['stronger_GT_rows_pair_repeated']+=len(annotations[strong])
                c['stronger_GT_outside_weaker_GT_rows_pair_repeated']+=len(violation)
            if violation:witnesses.append(dict(kind='stronger_GT_not_subset',weaker=weak,stronger=strong,rows=len(violation)))
    return {name:dict(c) for name,c in result.items()},witnesses


def main():
    start=time.perf_counter();plan=json.loads(PLAN.read_text());assert file_sha(Path(__file__))==plan['source_sha256']
    assert json.loads((I1/'all_predictions_sealed_before_GT.json').read_text())['methods']==828
    assert json.loads((I1/'official.json').read_text())['methods']==828
    assert json.loads((SCOPES/'completion.json').read_text())['status']=='PASS_QUERY_TEXT_SCOPE_TEACHER_PREPARATION_NO_GT'
    gt_roots=roots();structures=json.loads((SCOPES/'structures.json').read_text())['structures'];summaries={}
    for target in ('Dance','KITTI-V2','SportsMOT'):
        md=metadata(target);totals={name:Counter() for name in ('all_accepted','no_ambiguity_notes')};videos=[]
        for video in sorted(md['videos']):
            queries=[q for q in md['episodes'] if q['video_id']==video]
            namespace={};annotations={};io_counts=Counter();max_geometry_error=0.
            for q in queries:
                value,counts,error=read_rows(gt_roots[target]/video/q['expression_id']/'gt.txt',int(q['frame_count']),namespace)
                annotations[q['expression_id']]=value;io_counts.update(counts);max_geometry_error=max(max_geometry_error,error)
            # Exact equality is a conservative annotation-namespace check,
            # not a learned threshold or a candidate localization metric.
            comparable=max_geometry_error==0
            result,witnesses=pair_counts(queries,annotations,structures) if comparable else ({name:{} for name in totals},[])
            for name,c in result.items():totals[name].update(c)
            row=dict(video=video,queries=len(queries),GT_rows=sum(map(len,annotations.values())),
                annotation_IO=dict(io_counts),cross_query_ID_geometry_max_absolute_error=max_geometry_error,
                namespace_exactly_coherent=comparable,logical_counts=result)
            videos.append(row)
            atomic_json(OUT/target/(video+'_private_witnesses.json'),dict(relationships=witnesses,
                no_actual_GT_rows_exported=True,not_to_Git=True))
        summaries[target]=dict(videos=len(videos),queries=len(md['episodes']),
            videos_with_exact_shared_ID_namespace=sum(r['namespace_exactly_coherent'] for r in videos),
            logical_counts={name:dict(c) for name,c in totals.items()},per_video=videos)
    report=dict(status='PASS_POST_I1_SEAL_QUERY_EVENT_ASSUMPTION_AUDIT',targets=summaries,
        scope='Raw official positive-mark/class1 (frame, original ID) annotation sets; no frozen-candidate matching.',
        comparison='Same video/evaluation horizon/accepted parsed subject; full AND/nonopaque fixed teacher interpretations. Report with and without ambiguity notes.',
        namespace_rule='Only exact cross-query ID-to-box consistency permits logical set comparison; failures remain explicit.',
        interpretation='A violation can reflect parser semantics, pragmatic reference, annotation conventions or actual label inconsistency. This audit does not attribute its cause or certify any atom label.',
        I1_all828_sealed_and_official_complete_before_this_GT=True,explicit_posthoc_DEV_diagnostic=True,
        current_unified_study_predictions_heads_and_parameters_not_read_or_changed=True,
        no_annotation_or_per_query_GT_supplied_to_adaptation=True,no_new_prediction_training_or_teacher_calls=True,
        pairs_and_rows_correlated_no_population_CI_or_HOTA_error_decomposition=True,
        worker_wall_seconds=time.perf_counter()-start,goal_status='ACTIVE')
    atomic_json(OUT/'completion.json',report)
    atomic_json(PUBLIC/'query_scope_postseal_assumptions.json',report)
    print(json.dumps(dict(status=report['status'],targets={t:{k:v for k,v in r.items() if k!='per_video'} for t,r in summaries.items()})))


if __name__=='__main__':main()
