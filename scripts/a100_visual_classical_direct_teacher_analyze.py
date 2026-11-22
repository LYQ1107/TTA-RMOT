"""Direct-KD fairness: isolate a fixed source residual from structured gains."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_direct_teacher import METHODS
from scripts.a100_visual_classical_direct_teacher_launch import conditions
from scripts.a100_visual_classical_joint_visual_analyze import external as residual_external
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_sparse_observation_analyze import effect,fit_audit
from scripts.a100_visual_classical_sparse_observation_report import describe
from scripts.a100_visual_classical_sparse_statistics_reproduce import resolve_record,share_identical_statistics

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'


def references(record):
    result={'RES_'+name:reference for name,(_,reference) in residual_external(record).items()}
    slug=record['pair']+f"_seed{record['seed']}"
    for name in ('JOINT','CAL'):result['RES_'+name]=dict(file='joint_visual_parent_statistics.json',record=slug+'/'+name)
    return result


def contrasts():
    result={'JOINT_vs_'+n:('JOINT',n) for n in METHODS if n!='JOINT'}
    for n in ('KD400','KD_FINAL','KD_MEAN','HARD400','HARD_FINAL'):result['RES_JOINT_vs_'+n]=('RES_JOINT',n)
    for name in ('RES_JOINT','RES_KD','RES_KD400','RES_KD_FINAL','RES_HARD400','RES_HARD_FINAL',
            'RES_PCD_KD_MEAN','RES_PCD_KD400','RES_ZERO_HSV_MEAN','RES_ZERO_HSV_400','RES_SEM_MEAN','RES_SEM400','RES_SEM_HARD400'):
        result['JOINT_vs_'+name]=('JOINT',name)
    result.update(KD_MEAN_vs_RES_KD=('KD_MEAN','RES_KD'),GRAPH_vs_KD_MEAN=('GRAPH','KD_MEAN'),CAL_vs_KD_MEAN=('CAL','KD_MEAN'))
    result['COUNT_OWN_JOINT_vs_KD_MEAN']=('JOINT__AT_JOINT_COUNT','KD_MEAN__AT_JOINT_COUNT')
    result['COUNT_BASELINE_JOINT_vs_KD_MEAN']=('JOINT__AT_KD_MEAN_COUNT','KD_MEAN__AT_KD_MEAN_COUNT')
    return result


def diagnosis(records):
    from collections import Counter
    from flex_uda.a100.mechanism import OUT as DIAG,metadata,PAIRS
    result={};names=('KD_MEAN','GRAPH','CAL','JOINT')
    for record in records:
        pair=record['pair'];root=Path(record['folder']);m=metadata(PAIRS[pair][1]);counts={name:Counter() for name in names};trade=Counter()
        for video in sorted(m['videos']):
            truth=np.load(DIAG/'offline_GT'/pair/video/'labels.npy')>=0;present=truth.any(1);correct={}
            for name in names:
                mask=np.load(root/name/'videos'/video/'mask.npy');correct[name]=(mask&truth).any(1)
                counts[name].update(queries=len(mask),recoverable_positive_queries=int(present.sum()),
                    positive_queries_empty=int((present&~mask.any(1)).sum()),positive_queries_without_correct_output=int((present&~correct[name]).sum()),
                    no_positive_candidate_queries_with_output=int((~present&mask.any(1)).sum()),TP=int((mask&truth).sum()),FP=int((mask&~truth).sum()),FN=int((~mask&truth).sum()))
            trade.update(rescued=int((present&~correct['KD_MEAN']&correct['JOINT']).sum()),regressed=int((present&correct['KD_MEAN']&~correct['JOINT']).sum()))
        result[pair+f"_seed{record['seed']}"]=dict(candidate_membership={n:dict(c) for n,c in counts.items()},joint_query_retrieval_tradeoff=dict(trade))
    return dict(conditions=result,GT_only_after_all72seals=True,not_additive_HOTA_attribution=True)


def main(stem='direct_teacher',records=None,dimension=1080,extra_references=None,extra_contrasts=None):
    official=json.loads((OUT/'analysis'/(stem+'_official.json')).read_text());assert official['methods']==72
    data={};summary={};statistics={};cache={};records=conditions() if records is None else records
    comparisons=contrasts()
    if extra_contrasts:comparisons.update(extra_contrasts)
    for record in records:
        pair,seed=record['pair'],record['seed'];slug=pair+f'_seed{seed}';root=Path(record['folder'])
        raw=json.loads((root/'completion.json').read_text());counts=json.loads((root/'countmatched/completion.json').read_text())
        assert raw['source_logit_residual_disabled'] and raw['GT_open_attempts']==raw['new_teacher_calls']==raw['new_encoder_calls']==0
        assert raw['logical_teacher_calls']==(1160 if pair=='C4' else 1103) and not any(counts['CountSelf_changed_query_files'].values())
        paths={m['method']:root/m['method'] for m in raw['methods']};paths.update({m['method']:root/'countmatched'/m['method'] for m in counts['methods']})
        data[slug]={name:read_stats(folder) for name,folder in paths.items()}
        for name,row in data[slug].items():statistics[slug+'/'+name]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],
            numerator=row['parent_stats'][0].tolist(),denominator=row['parent_stats'][1].tolist())
        refs=references(record)
        if extra_references:refs.update(extra_references(record))
        for alias,reference in refs.items():
            row=resolve_record(PUBLIC,reference['file'],reference['record'],cache)
            data[slug][alias]=dict(parent_videos=row['parents'],HOTA=row['HOTA'],parent_stats=(np.asarray(row['numerator']),np.asarray(row['denominator'])))
            statistics[slug+'/'+alias]=dict(reference=reference)
        summary[slug]=dict(pair=pair,seed=seed,raw_HOTA={name:row['HOTA'] for name,row in data[slug].items() if '__AT_' not in name},
            comparisons={name:effect([data[slug][a]],[data[slug][b]]) for name,(a,b) in comparisons.items()},
            teacher_fit_audit=fit_audit(raw['receipts']),
            actual_head_fit_seconds={label:sum(r['fits'][label]['wall_seconds'] for r in raw['receipts']) for label in ('SOFT','HARD')},
            CountSelf_changed_query_files=counts['CountSelf_changed_query_files'],source_logit_residual_disabled=True,
            physical_GPU=raw['physical_GPU'],logical_teacher_calls=raw['logical_teacher_calls'],new_teacher_calls=0,new_encoder_calls=0)
    repeated={}
    for pair in ('C4','C1'):
        slugs=[pair+f'_seed{s}' for s in (0,17,42)]
        repeated['DIRECT_'+pair]=dict(condition_keys=slugs,neural_seeds=[0,17,42],QA_seed=0,
            mean_HOTA={name:float(np.mean([summary[k]['raw_HOTA'][name] for k in slugs])) for name in summary[slugs[0]]['raw_HOTA']},
            comparisons={name:effect([data[k][a] for k in slugs],[data[k][b] for k in slugs]) for name,(a,b) in comparisons.items()})
    errors=diagnosis(records)
    result=dict(status='DIRECT_TEACHER_NO_SOURCE_RESIDUAL_ANALYZED',conditions=summary,repeated=repeated,query_diagnosis=errors,
        ordinary_direct_classifier_is_not_point_label_copy=True,
        fixed_source_logit_zero_in_training_and_prediction=True,neural_input_dimension=dimension,
        no_HSV_neural_input=dimension==768,
        prior_still_common_acquisition_and_tie_break_information=True,
        raw_outputs=48,count_outputs=24,official_new_methods=72,new_teacher_calls=0,new_encoder_calls=0,actual_head_fits=420,
        old_RES_aliases_are_previous_source_residual_heads=True,
        no_target_manual_GT_in_adaptation=True,all_development_data=True,independent_blind_validation=False,
        uncertainty='Fixed sealed prediction parent/shared-neural-seed descriptive intervals, no prefix refit or repeated-development correction.',
        cost='Actual head fit quality-replay components only; not new cold wall. Retained recipe/comparator changes require matching physical follow-up.',
        goal_status='ACTIVE_REQUIRES_STRONG_DIRECT_BASELINE_REVIEW')
    atomic_json(PUBLIC/(stem+'_analysis.json'),result);atomic_json(OUT/'analysis'/(stem+'_analysis.json'),result)
    compact,_=share_identical_statistics(statistics,stem+'_parent_statistics.json')
    (PUBLIC/(stem+'_parent_statistics.json')).write_text(json.dumps(compact,separators=(',',':'))+'\n')
    lines=['TTA-RMOT：不叠加固定源域先验的直接教师分类头公平性核验','2026-11-22','',
        '此前充分训练KD是source-logit残差头，本轮补普通直接分类头，排除新模块只是补救固定错误先验。',
        '神经输入维度='+str(dimension)+'；'+('原生PCD，不拼接HSV。' if dimension==768 else 'PCD+查询条件RGB1080。'),
        '教师问答、归一化、优化器、充分训练规则相同；训练与推理的固定源z均为0。',
        '源先验仍参与共同图像选题和数量投影并列时的固定tie-break，但不固定叠加到直接头输出。',
        '保留软400/最终/均值和硬400/最终，另在同一个直接均值头上拆GRAPH/CAL/JOINT；不按方向挑选头。',
        'GPU资源等四份旧物理流全部退出后再分配，不影响正在测量的真实worker墙钟。',
        '6条件420真实小头拟合，48原始+24双向数量共72输出全部封存后本研究GT。新增学生/教师前向0。',
        'RES_前缀表示以前固定源logit残差对照，普通直接分类头不等于只把教师答案原样拷到已问框。','']
    for pair,row in repeated.items():
        lines += [pair,'三种子HOTA：'+json.dumps(row['mean_HOTA'])]
        for name,e in row['comparisons'].items():lines.append(name+'：'+describe(e))
    lines += ['', '封存后检索/FP/FN：',json.dumps(errors,ensure_ascii=False,indent=2),
        '拟合与真实小头时间：',json.dumps({k:{n:v[n] for n in ('teacher_fit_audit','actual_head_fit_seconds','CountSelf_changed_query_files')} for k,v in summary.items()},indent=2),
        '本轮复用已付费特征/问答，仅小头重拟合，不能把新精度贴在旧冷启动时间上。',
        '所有数据均为开发数据，bootstrap不重训全前缀流程。若直接分类头消除了旧优势，必须继续研究并保留失败，不能因此自动关闭Goal。']
    name='TTA_RMOT_原生PCD直接教师_无源残差公平性_20261122.txt' if dimension==768 else 'TTA_RMOT_直接教师分类头_无源先验残差公平性_20261122.txt'
    (ROOT/'docs'/name).write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(event='DIRECT_TEACHER_ANALYSIS_COMPLETE',mean_HOTA={k:r['mean_HOTA'] for k,r in repeated.items()},goal_status='ACTIVE')),flush=True)


if __name__=='__main__':main()
