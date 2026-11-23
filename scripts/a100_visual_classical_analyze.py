"""Post-seal statistics and error coverage for the registered first round."""
import json
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import metadata,PAIRS,OUT as DIAG,baseline_mask
from scripts.a100_visual_classical_prepare import OUT


def read_stats(folder):
    from scripts.active_eval import official_stats
    from scripts.a100_mechanism_analyze import grouped_official
    path=Path(folder)
    if (path/'official_equivalent.json').is_file():
        path=Path(json.loads((path/'official_equivalent.json').read_text())['identical_prediction_reference'])
    return grouped_official(path)


def analyze():
    from scripts.a100_mechanism_analyze import compare_official
    analysis=OUT/'analysis';analysis.mkdir(parents=True,exist_ok=True)
    results={};sufficient={};coverage={};cost={}
    for pair,(_,target) in [(p,PAIRS[p]) for p in ('C4','C1')]:
        complete=json.loads((OUT/'evaluation'/pair/'completion.json').read_text())
        longer=json.loads((OUT/'convergence'/pair/'official_evaluation.json').read_text())
        complete['results'].extend([dict(row,group='convergence') for row in longer['results']])
        groups={};allstats={}
        for row in complete['results']:
            key=row['group']+'/'+row['method'];folder=Path(row['folder'])
            groups[key]=dict(HOTA=row['evaluation']['HOTA_unrounded'],metrics=row['evaluation']['metrics'],
                prediction_rows=json.loads((folder/'predictions_seal.json').read_text())['prediction_rows'])
            allstats[key]=read_stats(folder)
        comparisons={}
        primary='predictions/COVER__FULL'
        references=['COVER__DIRECT_HARD','COVER__DIRECT_SOFT','COVER__DIRECT_NATIVE','COVER__KD_HARD','COVER__KD_SOFT',
            'COVER__FRAME_KD_HARD','COVER__FRAME_KD_SOFT','COVER__FRAME_KD_HARD_STRUCTURED','COVER__FRAME_KD_SOFT_STRUCTURED',
            'COVER__NO_CORRECTION','COVER__NO_GRAPH','PRIOR__FULL','UNIFORM__FULL','UNCERTAINTY__FULL']
        references += [a+'__'+m for a in ('PRIOR','UNIFORM','UNCERTAINTY') for m in ('DIRECT_NATIVE','KD_HARD','KD_SOFT')]
        references += [a+'__'+m for a in ('PRIOR','UNIFORM','UNCERTAINTY') for m in ('FRAME_KD_HARD','FRAME_KD_SOFT')]
        for reference in references:
            second='predictions/'+reference
            comparisons['COVER_FULL_vs_'+reference]=compare_official(allstats[primary],allstats[second])
        for policy in ('PRIOR','UNIFORM','UNCERTAINTY','COVER'):
            for label in ('HARD','SOFT'):
                reference=policy+'__KD_'+label+'_4000'
                comparisons['COVER_FULL_vs_'+reference]=compare_official(allstats[primary],allstats['convergence/'+reference])
        for rank in ('KD_HARD','KD_SOFT','FRAME_KD_HARD','FRAME_KD_SOFT','DIRECT_SOFT','NO_CORRECTION','NO_GRAPH'):
            a='countmatched/FULL__COUNT_FULL';b='countmatched/'+rank+'__COUNT_FULL'
            comparisons['COUNT_FULL__FULL_vs_'+rank]=compare_official(allstats[a],allstats[b])
        comparisons['COUNT_KD_SOFT__FULL_vs_KD_SOFT']=compare_official(
            allstats['countmatched/FULL__COUNT_KD_SOFT'],allstats['countmatched/KD_SOFT__COUNT_KD_SOFT'])
        comparisons['COUNT_FRAME_KD_SOFT__FULL_vs_FRAME_KD_SOFT']=compare_official(
            allstats['countmatched/FULL__COUNT_FRAME_KD_SOFT'],allstats['countmatched/FRAME_KD_SOFT__COUNT_FRAME_KD_SOFT'])
        results[pair]=dict(methods=groups,comparisons=comparisons,seed=0,
            all_development_data=True,no_target_GT_in_generation=True,
            confidence_scope='descriptive paired original-parent bootstrap of fixed seed0 outputs; not independent domain confirmation')
        sufficient[pair]={key:dict(videos=data['videos'],parents=data['parent_videos'],
            numerator=data['parent_stats'][0].tolist(),denominator=data['parent_stats'][1].tolist(),HOTA=data['HOTA']) for key,data in allstats.items()}
        plan=json.loads((OUT/'evidence'/pair/'acquisition_plan.json').read_text())['positions']
        meta=metadata(target);totals={a:Counter() for a in ('PRIOR','UNIFORM','UNCERTAINTY','COVER')};details=[]
        method_totals={method:Counter() for method in ('DIRECT_NATIVE','KD_SOFT','KD_HARD_4000','KD_SOFT_4000','FRAME_KD_HARD','FRAME_KD_SOFT','FRAME_KD_SOFT_STRUCTURED','FULL','NO_CORRECTION','NO_GRAPH')}
        for video,bank in sorted(meta['videos'].items()):
            labels=np.load(DIAG/'offline_GT'/pair/video/'labels.npy')>=0
            obs=np.asarray(bank['observations']);tracks=sorted(set(obs[:,1].tolist()))
            queries=[q for q in meta['episodes'] if q['video_id']==video]
            old=baseline_mask(pair,video)
            with np.load(OUT/'inputs'/pair/video/'inputs.npz') as a:anchors=a['anchors']
            for policy,total in totals.items():
                asked={(int(q),tracks[int(t)]) for q,t in plan[video][policy]}
                for q,query in enumerate(queries):
                    positive=labels[q];asked_rows=np.asarray([(q,int(t)) in asked for _,t in obs])
                    correct=bool((positive&asked_rows).any())
                    counts=dict(queries=1,present=int(positive.any()),positive_candidate_rows=int(positive.sum()),
                        positive_rows_on_asked_tracks=int((positive&asked_rows).sum()),
                        no_correct_track_asked=int(positive.any() and not correct),
                        selected_distinct_tracks=sum(qi==q for qi,_ in asked))
                    total.update(counts)
                    details.append(dict(pair=pair,video=video,query=query['expression_id'],policy=policy,**counts))
            for method,total in method_totals.items():
                group='convergence' if method.endswith('_4000') else 'predictions'
                mask=np.load(OUT/group/pair/('COVER__'+method)/'videos'/video/'mask.npy')
                total.update(dict(selected_rows=int(mask.sum()),positive_selected_rows=int((mask&labels).sum()),
                    negative_selected_rows=int((mask&~labels).sum()),missed_positive_rows=int((~mask&labels).sum())))
                for q,query in enumerate(queries):
                    present=bool(labels[q].any())
                    total['present_zero_queries']+=present and not bool(mask[q].any())
                    total['old_zero_query_now_positive_match']+=present and not bool(old[q].any()) and bool((mask[q]&labels[q]).any())
                    total['empty_query_false_output']+=(not present) and bool(mask[q].any())
        coverage[pair]=dict(acquisition={k:dict(v) for k,v in totals.items()},
            outputs={k:dict(v) for k,v in method_totals.items()},queries=details,
            definition='fixed IoU-matched candidate observation labels for diagnosis; not an additive HOTA attribution')
        teacher=json.loads((OUT/'evidence'/pair/'completion.json').read_text())
        preparation=json.loads((OUT/'inputs'/pair/'completion.json').read_text())
        cost[pair]=dict(evidence_preparation=teacher,input_preparation=preparation,
            caveat='shared prepaid teacher pool and historical frozen-feature reuse are not a measured cold end-to-end deployment',
            evidence_process_wall_includes_input_preparation_due_to_collection_module_import_scope=True,
            teacher_service_and_load_phase_wall_sum_seconds=sum(teacher[k] for k in ('teacher_load_wall_seconds','teacher_service_wall_seconds','final_weight_seal_wall_seconds')),
            generation=json.loads((OUT/'predictions'/pair/'completion.json').read_text()),
            converged_distillation={a+'__KD_'+label+'_4000':json.loads((OUT/'convergence'/pair/(a+'__KD_'+label+'_4000')/'training_receipt.json').read_text())
                for a in ('PRIOR','UNIFORM','UNCERTAINTY','COVER') for label in ('HARD','SOFT')},
            distillation={a+'__'+m:json.loads((OUT/'predictions'/pair/(a+'__'+m)/'training_receipt.json').read_text())
                for a in ('PRIOR','UNIFORM','UNCERTAINTY','COVER') for m in ('KD_HARD','KD_SOFT','FRAME_KD_HARD','FRAME_KD_SOFT')})
    for name,value in [('official_results',results),('official_sufficient_statistics',sufficient),('coverage',coverage),('cost_ledger',cost)]:
        atomic_json(analysis/(name+'.json'),value)
    print(json.dumps({pair:{k:v['HOTA'] for k,v in r['methods'].items() if k.startswith('predictions/COVER')} for pair,r in results.items()}),flush=True)


if __name__=='__main__':analyze()
