"""Post-seal graph/PAVA attribution on the exact shared weighted teacher head."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_weighted_classical_launch import conditions
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_sparse_observation_analyze import effect
from scripts.a100_visual_classical_sparse_observation_report import describe
from scripts.a100_visual_classical_sparse_statistics_reproduce import resolve_record,share_identical_statistics,methods

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'
ORDINARY=('RES_KD','RES_KD400','RES_KD_FINAL','RES_HARD400','RES_HARD_FINAL','RES_PCD_KD_MEAN','RES_ZERO_HSV_MEAN','RES_SEM_MEAN','RES_SEM400','RES_SEM_HARD400','RGB_KD400','RGB_KD_FINAL','RGB_KD_MEAN','RGB_HARD400','RGB_HARD_FINAL')


def references(slug):
    result={name:dict(file='confidence_teacher_parent_statistics.json',record=slug+'/'+source) for name,source in dict(KD='KD_MEAN',KD400='KD400',KD_FINAL='KD_FINAL',HARD400='HARD400',HARD_FINAL='HARD_FINAL').items()}
    result.update({name:dict(file='direct_teacher_pcd_parent_statistics.json',record=slug+'/'+name) for name in ORDINARY})
    for name,source in dict(UNWEIGHTED_KD='KD_MEAN',UNWEIGHTED_JOINT='JOINT').items():result[name]=dict(file='direct_teacher_pcd_parent_statistics.json',record=slug+'/'+source)
    return result


def contrasts():
    names=('KD','KD400','KD_FINAL','HARD400','HARD_FINAL','GRAPH','CAL','UNWEIGHTED_KD','UNWEIGHTED_JOINT')+ORDINARY
    result={'JOINT_vs_'+name:('JOINT',name) for name in names}
    result.update(GRAPH_vs_KD=('GRAPH','KD'),CAL_vs_KD=('CAL','KD'))
    for name in ('KD','GRAPH','CAL'):
        result['COUNT_OWN_JOINT_vs_'+name]=('JOINT__AT_JOINT_COUNT',name+'__AT_JOINT_COUNT')
        result['COUNT_BASELINE_JOINT_vs_'+name]=('JOINT__AT_'+name+'_COUNT',name+'__AT_'+name+'_COUNT')
    assert all(tuple(methods(label))==value for label,value in result.items())
    return result


def diagnosis(records):
    from collections import Counter
    from flex_uda.a100.mechanism import OUT as DIAG,metadata,PAIRS
    result={}
    for record in records:
        pair,seed=record['pair'],record['seed'];root=Path(record['folder']);m=metadata(PAIRS[pair][1])
        names=('KD','GRAPH','CAL','JOINT','UNWEIGHTED_JOINT')
        refs=dict(KD=OUT/'confidence_teacher'/pair/f'seed{seed}'/'KD_MEAN',UNWEIGHTED_JOINT=OUT/'direct_teacher_nores_pcd'/pair/f'seed{seed}'/'JOINT')
        counts={name:Counter() for name in names};trade={name:Counter() for name in ('KD','CAL','UNWEIGHTED_JOINT')}
        for video in sorted(m['videos']):
            truth=np.load(DIAG/'offline_GT'/pair/video/'labels.npy')>=0;present=truth.any(1)
            masks={name:np.load((refs[name] if name in refs else root/name)/'videos'/video/'mask.npy') for name in names}
            correct={name:(mask&truth).any(1) for name,mask in masks.items()}
            for name,mask in masks.items():
                counts[name].update(queries=len(mask),recoverable_positive_queries=int(present.sum()),positive_queries_empty=int((present&~mask.any(1)).sum()),
                    positive_queries_without_correct_output=int((present&~correct[name]).sum()),no_positive_candidate_queries_with_output=int((~present&mask.any(1)).sum()),
                    TP=int((mask&truth).sum()),FP=int((mask&~truth).sum()),FN=int((~mask&truth).sum()))
            for name in trade:trade[name].update(rescued=int((present&~correct[name]&correct['JOINT']).sum()),regressed=int((present&correct[name]&~correct['JOINT']).sum()))
        result[pair+f'_seed{seed}']=dict(candidate_membership={k:dict(v) for k,v in counts.items()},weighted_J_retrieval_versus={k:dict(v) for k,v in trade.items()})
    return dict(conditions=result,postseal_manual_GT_only=True,not_additive_HOTA_attribution=True,query_recall_is_any_IoU_positive_candidate_not_complete_GT=True)


def main():
    official=json.loads((OUT/'analysis/weighted_classical_official.json').read_text());assert official['methods']==78
    while not (PUBLIC/'confidence_teacher_statistical_reproduction.json').is_file():
        state=json.loads((OUT/'supervisor/confidence_teacher_state.json').read_text())
        if state['status'].startswith('FAILED'):raise RuntimeError('Weighted baseline dependency failed')
        time.sleep(5)
    assert json.loads((PUBLIC/'confidence_teacher_statistical_reproduction.json').read_text())['status'].startswith('PASS_')
    data={};summary={};statistics={};cache={};pairs=contrasts()
    for record in conditions():
        pair,seed=record['pair'],record['seed'];slug=pair+f'_seed{seed}';root=Path(record['folder'])
        raw=json.loads((root/'completion.json').read_text());counts=json.loads((root/'countmatched/completion.json').read_text())
        assert raw['GT_open_attempts']==raw['new_teacher_calls']==raw['new_encoder_calls']==raw['new_head_fits']==0
        assert not any(counts['CountSelf_changed_query_files'].values())
        paths={m['method']:root/m['method'] for m in raw['methods']};paths.update({m['method']:root/'countmatched'/m['method'] for m in counts['methods']})
        data[slug]={name:read_stats(path) for name,path in paths.items()}
        for name,row in data[slug].items():statistics[slug+'/'+name]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],numerator=row['parent_stats'][0].tolist(),denominator=row['parent_stats'][1].tolist())
        for name,ref in references(slug).items():
            row=resolve_record(PUBLIC,ref['file'],ref['record'],cache);data[slug][name]=dict(parent_videos=row['parents'],HOTA=row['HOTA'],parent_stats=(np.asarray(row['numerator']),np.asarray(row['denominator'])))
            statistics[slug+'/'+name]=dict(reference=ref)
        summary[slug]=dict(pair=pair,seed=seed,raw_HOTA={n:r['HOTA'] for n,r in data[slug].items() if '__AT_' not in n},
            comparisons={name:effect([data[slug][a]],[data[slug][b]]) for name,(a,b) in pairs.items()},
            full_prediction_replay_max_error=max(max(r['frozen_prediction_max_errors'].values()) for r in raw['receipts']),
            same_weighted_neural_head_as_ordinary=all(r['same_weighted_neural_head_as_ordinary'] for r in raw['receipts']),CountSelf_changed_query_files=counts['CountSelf_changed_query_files'])
    repeated={}
    for pair in ('C4','C1'):
        keys=[pair+f'_seed{s}' for s in (0,17,42)]
        repeated['WEIGHTED_CLASSICAL_'+pair]=dict(condition_keys=keys,neural_seeds=[0,17,42],QA_seed=0,
            mean_HOTA={n:float(np.mean([summary[k]['raw_HOTA'][n] for k in keys])) for n in summary[keys[0]]['raw_HOTA']},
            comparisons={name:effect([data[k][a] for k in keys],[data[k][b] for k in keys]) for name,(a,b) in pairs.items()})
    diagnostic=diagnosis(conditions())
    result=dict(status='SHARED_WEIGHTED_HEAD_CLASSICAL_ANALYZED',conditions=summary,repeated=repeated,query_diagnosis=diagnostic,
        primary_rule='Sharedconfidenceweightedordinary768Ddirectsoftmean;fixedfull3NNgraphthenPAVA',
        same_exact_head_as_competitive_weighted_ordinary=True,all78_new_outputs_sealed_before_manual_GT=True,new_teacher_encoder_head_calls=0,
        all_development_data=True,independent_blind_validation=False,
        uncertainty='Fixedsealedoriginal-parent/shared-seeddescriptivebootstrap,nopipelinerefitordevelopment/multiplicitycorrection.',
        quality_replay_not_actual_cold_time=True,goal_status='ACTIVE_REQUIRES_ROOT_SCIENTIFIC_REVIEW')
    atomic_json(OUT/'analysis/weighted_classical_analysis.json',result)
    (PUBLIC/'weighted_classical_analysis.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    compact,_=share_identical_statistics(statistics,'weighted_classical_parent_statistics.json')
    (PUBLIC/'weighted_classical_parent_statistics.json').write_text(json.dumps(compact,separators=(',',':'))+'\n')
    lines=['TTA-RMOT：在同一个强置信度加权头上的经典模块归因','2026-11-22','',
        '此前旧J对加权普通KD虽然C4平均+.536388，但原视频+种子CI[-.010093,1.127430]跨0。不能据此宣布已稳定胜过更强普通头。',
        '本研究给两边完全相同的加权普通头，再只增减原有图与外观PAVA。图/PAVA/数据/教师/解码均不调参数，比较头本身固定。',
        '6条件18原始+60双向数量=78输出，全部封存后本组GT；0新增教师、主干前向或头拟合，复用已充分拟合的420头。',
        '这是统一替换共享基础，不是按域选择GT最高的头；KD400/最终/均值、硬400/最终及全部旧强头均保留。','']
    for pair,row in repeated.items():
        lines += [pair,'三种子平均HOTA：'+json.dumps(row['mean_HOTA'])]
        for name,e in row['comparisons'].items():lines.append(name+'：'+describe(e))
    lines += ['封存后查询检索/空输出/TP/FP/FN：'+json.dumps(diagnostic,separators=(',',':')),
        '只报告匹配加权头上各模块的条件贡献；不把标准置信度加权本身当我方创新，不挑单一数量投影冒充语义排序。',
        '此前未加权J的真实时间不与本配方精度拼接。如最终保留加权共享头方案，须对应新的真实物理执行。',
        '全部数据为开发数据，区间为固定预测/原视频+共享种子的描述性重采样，不校正多轮开发/多重比较、也不重拟合全流水线。Goal不自动结束。']
    (ROOT/'docs/TTA_RMOT_共享强加权头_经典模块机制实证_20261122.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(event='WEIGHTED_CLASSICAL_ANALYSIS_COMPLETE',repeated=repeated,goal_status='ACTIVE')),flush=True)


if __name__=='__main__':main()
