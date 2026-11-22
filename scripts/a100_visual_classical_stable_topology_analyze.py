"""Fixed-plan teacher-only topology selection, all outputs sealed before GT."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_stable_topology_launch import conditions
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_sparse_observation_analyze import effect
from scripts.a100_visual_classical_sparse_observation_report import describe
from scripts.a100_visual_classical_sparse_statistics_reproduce import resolve_record,share_identical_statistics,methods

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'
ORDINARY=('RES_KD','RES_KD400','RES_KD_FINAL','RES_HARD400','RES_HARD_FINAL','RES_PCD_KD_MEAN',
    'RES_ZERO_HSV_MEAN','RES_SEM_MEAN','RES_SEM400','RES_SEM_HARD400','RGB_KD400','RGB_KD_FINAL','RGB_KD_MEAN','RGB_HARD400','RGB_HARD_FINAL')


def contrasts():
    primary='SELECTED_JOINT'
    result={primary+'_vs_'+name:(primary,name) for name in ('KD','CAL','SELECTED_GRAPH','LOCAL_JOINT','GLOBAL_JOINT','MEAN_JOINT','STRICT_JOINT','TWO_SE_JOINT','MEDIAN_JOINT','DELETE_ONE_JOINT')+ORDINARY}
    result.update(SELECTED_GRAPH_vs_KD=('SELECTED_GRAPH','KD'),SELECTED_GRAPH_vs_LOCAL_GRAPH=('SELECTED_GRAPH','LOCAL_GRAPH'),
        SELECTED_GRAPH_vs_GLOBAL_GRAPH=('SELECTED_GRAPH','GLOBAL_GRAPH'),GLOBAL_JOINT_vs_GLOBAL_GRAPH=('GLOBAL_JOINT','GLOBAL_GRAPH'),
        GLOBAL_JOINT_vs_KD=('GLOBAL_JOINT','KD'),LOCAL_JOINT_vs_KD=('LOCAL_JOINT','KD'))
    for name in ('KD','CAL','SELECTED_GRAPH','LOCAL_JOINT','GLOBAL_JOINT','STRICT_JOINT'):
        result['COUNT_OWN_'+primary+'_vs_'+name]=(primary+'__AT_'+primary+'_COUNT',name+'__AT_'+primary+'_COUNT')
        result['COUNT_BASELINE_'+primary+'_vs_'+name]=(primary+'__AT_'+name+'_COUNT',name+'__AT_'+name+'_COUNT')
    assert all(tuple(methods(label))==value for label,value in result.items())
    return result


def diagnosis(records):
    from collections import Counter
    from flex_uda.a100.mechanism import OUT as DIAG,metadata,PAIRS
    from scripts.a100_visual_classical_direct_teacher_pcd import DIRECT
    from scripts.a100_visual_classical_native_factorial import FACTORIAL
    result={}
    for record in records:
        pair,seed=record['pair'],record['seed'];root=Path(record['folder']);m=metadata(PAIRS[pair][1])
        names=('KD','CAL','LOCAL_JOINT','GLOBAL_JOINT','SELECTED_GRAPH','SELECTED_JOINT','MEAN_JOINT')
        refs={name:DIRECT/pair/f'seed{seed}'/original for name,original in dict(KD='KD_MEAN',CAL='CAL',LOCAL_JOINT='JOINT').items()}
        refs['GLOBAL_JOINT']=FACTORIAL/pair/f'seed{seed}'/'COMPLETE'
        refs['MEAN_JOINT']=OUT/'predictive_topology'/pair/f'seed{seed}'/'MEAN_JOINT'
        counts={name:Counter() for name in names};trade={name:Counter() for name in ('KD','LOCAL_JOINT','GLOBAL_JOINT')}
        for video in sorted(m['videos']):
            truth=np.load(DIAG/'offline_GT'/pair/video/'labels.npy')>=0;present=truth.any(1)
            masks={name:np.load((refs[name] if name in refs else root/name)/'videos'/video/'mask.npy') for name in names}
            correct={name:(mask&truth).any(1) for name,mask in masks.items()}
            for name,mask in masks.items():
                counts[name].update(queries=len(mask),recoverable_positive_queries=int(present.sum()),
                    positive_queries_empty=int((present&~mask.any(1)).sum()),positive_queries_without_correct_output=int((present&~correct[name]).sum()),
                    no_positive_candidate_queries_with_output=int((~present&mask.any(1)).sum()),TP=int((mask&truth).sum()),FP=int((mask&~truth).sum()),FN=int((~mask&truth).sum()))
            for name in trade:
                trade[name].update(rescued=int((present&~correct[name]&correct['SELECTED_JOINT']).sum()),
                    regressed=int((present&correct[name]&~correct['SELECTED_JOINT']).sum()))
        result[pair+f'_seed{seed}']=dict(candidate_membership={k:dict(v) for k,v in counts.items()},selected_J_retrieval_versus={k:dict(v) for k,v in trade.items()})
    return dict(conditions=result,postseal_GT_only=True,not_additive_HOTA_accounting=True,
        retrieval_is_any_IoU_positive_candidate_row_not_complete_GT=True,never_passed_to_selector=True)


def main():
    official=json.loads((OUT/'analysis/stable_topology_official.json').read_text());assert official['methods']==144
    data={};summary={};statistics={};cache={};comparisons=contrasts();selector_seed_check={}
    for record in conditions():
        pair,seed=record['pair'],record['seed'];slug=pair+f'_seed{seed}';root=Path(record['folder'])
        raw=json.loads((root/'completion.json').read_text());counts=json.loads((root/'countmatched/completion.json').read_text())
        assert raw['GT_open_attempts']==raw['new_teacher_calls']==raw['new_encoder_calls']==raw['new_head_fits']==0
        assert raw['logical_teacher_calls']==(1160 if pair=='C4' else 1103) and not any(counts['CountSelf_changed_query_files'].values())
        paths={m['method']:root/m['method'] for m in raw['methods']};paths.update({m['method']:root/'countmatched'/m['method'] for m in counts['methods']})
        data[slug]={name:read_stats(folder) for name,folder in paths.items()}
        for name,row in data[slug].items():statistics[slug+'/'+name]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],
            numerator=row['parent_stats'][0].tolist(),denominator=row['parent_stats'][1].tolist())
        refs={name:dict(file='direct_teacher_pcd_parent_statistics.json',record=slug+'/'+original) for name,original in dict(
            KD='KD_MEAN',CAL='CAL',LOCAL_GRAPH='GRAPH',LOCAL_JOINT='JOINT').items()}
        refs['GLOBAL_JOINT']=dict(file='native_factorial_parent_statistics.json',record=slug+'/COMPLETE')
        refs['GLOBAL_GRAPH']=dict(file='predictive_topology_parent_statistics.json',record=slug+'/GLOBAL_GRAPH')
        refs['MEAN_JOINT']=dict(file='predictive_topology_parent_statistics.json',record=slug+'/MEAN_JOINT')
        refs['STRICT_JOINT']=dict(file='predictive_topology_parent_statistics.json',record=slug+'/SELECTED_JOINT')
        refs.update({name:dict(file='direct_teacher_pcd_parent_statistics.json',record=slug+'/'+name) for name in ORDINARY})
        for name,ref in refs.items():
            row=resolve_record(PUBLIC,ref['file'],ref['record'],cache)
            data[slug][name]=dict(parent_videos=row['parents'],HOTA=row['HOTA'],parent_stats=(np.asarray(row['numerator']),np.asarray(row['denominator'])))
            statistics[slug+'/'+name]=dict(reference=ref)
        choices=[(r['video'],r['selector']['selected_model'],r['selector']['BCE_selected_model'],r['selector']['BALANCED_selected_model']) for r in raw['receipts']]
        if pair in selector_seed_check:assert selector_seed_check[pair]==choices,'Selector must not depend on fitted neural seed'
        else:selector_seed_check[pair]=choices
        selections={mode:sum(r['selector'][mode]=='GLOBAL' for r in raw['receipts']) for mode in ('selected_model','BCE_selected_model','BALANCED_selected_model')}
        summary[slug]=dict(pair=pair,seed=seed,raw_HOTA={n:r['HOTA'] for n,r in data[slug].items() if '__AT_' not in n},
            comparisons={n:effect([data[slug][a]],[data[slug][b]]) for n,(a,b) in comparisons.items()},
            stability_global_video_counts={name:sum(r['selector']['control_selected_models'][name]=='GLOBAL' for r in raw['receipts']) for name in ('ONE_SE','TWO_SE','MEDIAN','DELETE_ONE')},
            selector_global_video_counts=selections,total_videos=len(choices),selector_CPU_seconds=sum(r['selector_CPU_seconds'] for r in raw['receipts']),
            frozen_prediction_max_error=max(max(r['frozen_prediction_max_errors'].values()) for r in raw['receipts']),
            CountSelf_changed_query_files=counts['CountSelf_changed_query_files'],choices_use_fitted_NN=False)
    repeated={}
    for pair in ('C4','C1'):
        slugs=[pair+f'_seed{s}' for s in (0,17,42)]
        repeated['STABLE_'+pair]=dict(condition_keys=slugs,neural_seeds=[0,17,42],QA_seed=0,
            mean_HOTA={name:float(np.mean([summary[k]['raw_HOTA'][name] for k in slugs])) for name in summary[slugs[0]]['raw_HOTA']},
            comparisons={n:effect([data[k][a] for k in slugs],[data[k][b] for k in slugs]) for n,(a,b) in comparisons.items()})
    macro={name:float(np.mean([r['mean_HOTA'][name] for r in repeated.values()])) for name in repeated['STABLE_C4']['mean_HOTA']}
    diagnostic=diagnosis(conditions())
    result=dict(status='STABLE_TOPOLOGY_SELECTION_ANALYZED',conditions=summary,repeated=repeated,query_diagnosis=diagnostic,
        equal_domain_macro_points_only=macro,macro_points_are_arithmetic_descriptors_not_new_domain_population_inference=True,
        source='Same sealed native768D direct softmean + fixedPAVA; ONE_SE teacher risk sensitivity; no manualGT or NN unary in validation.',
        old_Gaussian_BMA_negative_preserved=True,strict_minimum_V1_C4_promotion_failure_preserved=True,primary_rule='ONE_SE',SE_is_sensitivity_not_calibrated_CI=True,
        no_fitted_NN_unary_in_validation=True,selector_choices_equal_across_neural_seeds=True,
        new_raw_outputs=30,new_count_outputs=114,official_new_methods=144,new_teacher_calls=0,new_encoder_calls=0,new_head_fits=0,
        all_development_data=True,independent_blind_validation=False,
        uncertainty='Fixed sealed original-parent/shared-neural-seed descriptive bootstrap; no pipeline refit or development/multiplicity correction.',
        goal_status='ACTIVE_REQUIRES_SCIENTIFIC_REVIEW_AND_MATCHING_ACTUAL_COST')
    atomic_json(PUBLIC/'stable_topology_analysis.json',result);atomic_json(OUT/'analysis/stable_topology_analysis.json',result)
    compact,_=share_identical_statistics(statistics,'stable_topology_parent_statistics.json')
    (PUBLIC/'stable_topology_parent_statistics.json').write_text(json.dumps(compact,separators=(',',':'))+'\n')
    lines=['TTA-RMOT：教师留轨迹预测风险驱动的局部/整体参照选择','2026-11-22','',
        '不是按目标域名称选择高分拓扑；只用冻结图片几何、当前教师回答和已声明任务契约。神经头不进入选择器验证。',
        '教师TrackID五折跨查询排除；中性0.5一元项，直接比较真实图算子对未参与拟合教师回答的预测风险。',
        'PRESUPPOSED使用正/负各半softCE，CHECKLIST使用BernoulliCE。主方法ONE_SE仅在均值改进大于配对查询风险SE时转全局；该SE不校正共享轨迹/交叉验证依赖，不是置信区间。',
        '主方法固定ONE_SE；TWO_SE、MEDIAN、DELETE_ONE是预先固定敏感性对照，不按GT择优替换主方法。旧严格最小值、完全图、固定局部与均值控制全部保留。',
        '旧Gaussian证据BMA、三专家及NoYes保护的失败仍保留。本轮是不同的预测选择机制；不把旧负结果改名为成功。',
        '144新输出全部封存后GT；0新增教师/编码器调用、小头拟合。所有数据均为开发数据，非独立盲测。','']
    for pair,row in repeated.items():
        lines += [pair,'三种子平均HOTA：'+json.dumps(row['mean_HOTA'])]
        for name,e in row['comparisons'].items():lines.append(name+'：'+describe(e))
    fields=('raw_HOTA','selector_global_video_counts','stability_global_video_counts','total_videos','selector_CPU_seconds','frozen_prediction_max_error','CountSelf_changed_query_files')
    lines += ['', '逐种子分数、教师选择和数值重放：',json.dumps({k:{n:r[n] for n in fields} for k,r in summary.items()},indent=2),
        '封存后的逐查询检索权衡（只作诊断，不进入选择器）：'+json.dumps(diagnostic,separators=(',',':')),
        '两个固定开发方向等权点均值（无新总体泛化区间）：'+json.dumps(macro),
        '教师预测一致性不等于GT准确率；若此机制没有胜过固定图或强普通KD，必须保留失败，不能按域硬编码覆盖。',
        '这些是CPU复用质量实验，不是新完整冷启动时间。改变最终方法需新的对应物理流与计时，不能套当前固定局部图的时间。']
    (ROOT/'docs/TTA_RMOT_教师预测选择局部整体参照_20261122.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(event='STABLE_TOPOLOGY_ANALYSIS_COMPLETE',mean_HOTA={p:r['mean_HOTA'] for p,r in repeated.items()},goal_status='ACTIVE')),flush=True)


if __name__=='__main__':main()
