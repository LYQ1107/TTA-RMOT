"""Post-seal audit of optional identity/local/global teacher-predictive transfer."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_optional_transfer_launch import conditions
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_sparse_observation_analyze import effect
from scripts.a100_visual_classical_sparse_observation_report import describe
from scripts.a100_visual_classical_sparse_statistics_reproduce import resolve_record,share_identical_statistics,methods
from scripts.a100_visual_classical_weighted_classical_analyze import ORDINARY

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'
PRIMARY='OPTIONAL_JOINT'
COUNT_REFERENCES=('KD','CAL','OPTIONAL_GRAPH','FULL_JOINT','FORCED_JOINT','GLOBAL_JOINT')


def references(slug):
    result={name:dict(file='confidence_teacher_parent_statistics.json',record=slug+'/'+source) for name,source in
        dict(KD='KD_MEAN',KD400='KD400',KD_FINAL='KD_FINAL',HARD400='HARD400',HARD_FINAL='HARD_FINAL').items()}
    for name,source in dict(CAL='CAL',FULL_GRAPH='GRAPH',FULL_JOINT='JOINT').items():
        result[name]=dict(file='weighted_classical_parent_statistics.json',record=slug+'/'+source)
    for name,source in dict(UNWEIGHTED_KD='KD_MEAN',UNWEIGHTED_JOINT='JOINT').items():
        result[name]=dict(file='direct_teacher_pcd_parent_statistics.json',record=slug+'/'+source)
    result.update({name:dict(file='direct_teacher_pcd_parent_statistics.json',record=slug+'/'+name) for name in ORDINARY})
    return result


def contrasts():
    names=tuple(references('unused'))+('OPTIONAL_GRAPH','STRICT_JOINT','FORCED_JOINT','GLOBAL_JOINT','ANCHOR_OPTIONAL_JOINT')
    result={PRIMARY+'_vs_'+name:(PRIMARY,name) for name in names}
    result['OPTIONAL_GRAPH_vs_KD']=('OPTIONAL_GRAPH','KD')
    for name in COUNT_REFERENCES:
        result['COUNT_OWN_'+PRIMARY+'_vs_'+name]=(PRIMARY+'__AT_'+PRIMARY+'_COUNT',name+'__AT_'+PRIMARY+'_COUNT')
        result['COUNT_BASELINE_'+PRIMARY+'_vs_'+name]=(PRIMARY+'__AT_'+name+'_COUNT',name+'__AT_'+name+'_COUNT')
    assert all(tuple(methods(label))==value for label,value in result.items())
    return result


def diagnosis(records):
    from collections import Counter
    from flex_uda.a100.mechanism import OUT as DIAG,metadata,PAIRS
    result={}
    for record in records:
        pair,seed=record['pair'],record['seed'];root=Path(record['folder']);m=metadata(PAIRS[pair][1])
        names=('KD','CAL','FULL_JOINT','OPTIONAL_GRAPH',PRIMARY,'STRICT_JOINT','FORCED_JOINT','GLOBAL_JOINT','ANCHOR_OPTIONAL_JOINT')
        weighted=OUT/'weighted_classical'/pair/f'seed{seed}'
        refs=dict(KD=OUT/'confidence_teacher'/pair/f'seed{seed}'/'KD_MEAN',CAL=weighted/'CAL',FULL_JOINT=weighted/'JOINT')
        counts={name:Counter() for name in names};trade={name:Counter() for name in COUNT_REFERENCES}
        for video in sorted(m['videos']):
            truth=np.load(DIAG/'offline_GT'/pair/video/'labels.npy')>=0;present=truth.any(1)
            masks={name:np.load((refs[name] if name in refs else root/name)/'videos'/video/'mask.npy') for name in names}
            correct={name:(mask&truth).any(1) for name,mask in masks.items()}
            for name,mask in masks.items():
                counts[name].update(queries=len(mask),recoverable_positive_queries=int(present.sum()),positive_queries_empty=int((present&~mask.any(1)).sum()),
                    positive_queries_without_correct_output=int((present&~correct[name]).sum()),no_positive_candidate_queries_with_output=int((~present&mask.any(1)).sum()),
                    TP=int((mask&truth).sum()),FP=int((mask&~truth).sum()),FN=int((~mask&truth).sum()))
            for name in trade:
                trade[name].update(rescued=int((present&~correct[name]&correct[PRIMARY]).sum()),regressed=int((present&correct[name]&~correct[PRIMARY]).sum()))
        result[pair+f'_seed{seed}']=dict(candidate_membership={k:dict(v) for k,v in counts.items()},optional_J_retrieval_versus={k:dict(v) for k,v in trade.items()})
    return dict(conditions=result,postseal_manual_GT_only=True,not_additive_HOTA_attribution=True,query_recall_is_any_IoU_positive_candidate_not_complete_GT=True)


def main():
    official=json.loads((OUT/'analysis/optional_transfer_official.json').read_text());assert official['methods']==150
    assert json.loads((PUBLIC/'weighted_classical_statistical_reproduction.json').read_text())['status'].startswith('PASS_')
    data={};summary={};statistics={};cache={};pairs=contrasts()
    for record in conditions():
        pair,seed=record['pair'],record['seed'];slug=pair+f'_seed{seed}';root=Path(record['folder'])
        raw=json.loads((root/'completion.json').read_text());counts=json.loads((root/'countmatched/completion.json').read_text())
        assert raw['GT_open_attempts']==raw['new_teacher_calls']==raw['new_encoder_calls']==raw['new_head_fits']==0
        assert not any(counts['CountSelf_changed_query_files'].values())
        paths={m['method']:root/m['method'] for m in raw['methods']};paths.update({m['method']:root/'countmatched'/m['method'] for m in counts['methods']})
        data[slug]={name:read_stats(path) for name,path in paths.items()}
        for name,row in data[slug].items():
            statistics[slug+'/'+name]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],numerator=row['parent_stats'][0].tolist(),denominator=row['parent_stats'][1].tolist())
        for name,ref in references(slug).items():
            row=resolve_record(PUBLIC,ref['file'],ref['record'],cache)
            data[slug][name]=dict(parent_videos=row['parents'],HOTA=row['HOTA'],parent_stats=(np.asarray(row['numerator']),np.asarray(row['denominator'])))
            statistics[slug+'/'+name]=dict(reference=ref)
        summary[slug]=dict(pair=pair,seed=seed,raw_HOTA={n:r['HOTA'] for n,r in data[slug].items() if '__AT_' not in n},
            comparisons={name:effect([data[slug][a]],[data[slug][b]]) for name,(a,b) in pairs.items()},
            full_prediction_replay_max_error=max(max(r['frozen_prediction_max_errors'].values()) for r in raw['receipts']),
            selected_models={name:sum(r['selector']['selected_model']==name for r in raw['receipts']) for name in ('IDENTITY','LOCAL','GLOBAL')},
            strict_selected_models={name:sum(r['selector']['strict_selected_model']==name for r in raw['receipts']) for name in ('IDENTITY','LOCAL','GLOBAL')},
            forced_selected_models={name:sum(r['selector']['forced_selected_model']==name for r in raw['receipts']) for name in ('LOCAL','GLOBAL')},
            CountSelf_changed_query_files=counts['CountSelf_changed_query_files'])
    repeated={}
    for pair in ('C4','C1'):
        keys=[pair+f'_seed{s}' for s in (0,17,42)]
        repeated['OPTIONAL_TRANSFER_'+pair]=dict(condition_keys=keys,neural_seeds=[0,17,42],QA_seed=0,
            mean_HOTA={n:float(np.mean([summary[k]['raw_HOTA'][n] for k in keys])) for n in summary[keys[0]]['raw_HOTA']},
            comparisons={name:effect([data[k][a] for k in keys],[data[k][b] for k in keys]) for name,(a,b) in pairs.items()})
    diagnostic=diagnosis(conditions())
    result=dict(status='OPTIONAL_IDENTITY_TRANSFER_ANALYZED',conditions=summary,repeated=repeated,query_diagnosis=diagnostic,
        primary_rule='HeldTrackIDneutralteacherCV;bestLOCALorGLOBALmustbeatIDENTITYriskbyonequerySE;elseexactKDidentity;thenexistingPAVA',
        same_exact_head_as_competitive_weighted_ordinary=True,all150_new_outputs_sealed_before_manual_GT=True,new_teacher_encoder_head_calls=0,
        explicit_identity_option_and_forced_graph_control=True,
        all_development_data=True,independent_blind_validation=False,
        uncertainty='Fixedsealedoriginal-parent/shared-seeddescriptivebootstrap,nopipelinerefitordevelopment/multiplicitycorrection.',
        quality_replay_not_actual_cold_time=True,goal_status='ACTIVE_REQUIRES_ROOT_SCIENTIFIC_REVIEW')
    atomic_json(OUT/'analysis/optional_transfer_analysis.json',result)
    (PUBLIC/'optional_transfer_analysis.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    compact,_=share_identical_statistics(statistics,'optional_transfer_parent_statistics.json')
    (PUBLIC/'optional_transfer_parent_statistics.json').write_text(json.dumps(compact,separators=(',',':'))+'\n')
    lines=['TTA-RMOT：允许不传播的教师预测模型选择','2026-11-22','',
        '原先局部/全局选择器被迫使用某种图。教师收据审计显示C4的36/45视频中，两图的留轨迹教师风险均不如中性0.5；C1仅1/25。此诊断不用GT，不是HOTA结论。',
        '新选项IDENTITY在部署中严格保留同一强加权普通KD，不是把已问节点替换为原始教师概率的零边图。验证阶段不使用已拟合KD，排除持出TrackID全部查询的教师答案。',
        '先选教师风险较低的局部/全局图；它相对IDENTITY的查询平均风险下降超过一倍配对查询SE时才传播，否则保留原KD。之后统一执行原PAVA。',
        'PRESUPPOSED验证平衡softCE，CHECKLIST验证BCE。SE是敏感性正则，查询共享轨迹，不称为校准置信度。',
        'STRICT三模型最小风险、被迫局部/全局、固定全局、原完整局部、原CAL均保留作对照；ANCHOR_OPTIONAL在拒绝传播时仍应用局部原始教师锚点，区分跳过锚点与跳过跨人传播。6条件36原始+114数量=150新输出全封存后本组GT，0新增教师/编码器/头拟合。','']
    for pair,row in repeated.items():
        lines += [pair,'三种子平均HOTA：'+json.dumps(row['mean_HOTA'])]
        for name,e in row['comparisons'].items():lines.append(name+'：'+describe(e))
    lines += ['封存后检索/空输出/TPFPFN：'+json.dumps(diagnostic,separators=(',',':')),
        '模型门控是否有效须对比强普通KD、被迫传播和固定图，不能把选择了IDENTITY本身当成功。若与ANCHOR_OPTIONAL对照不同，也不能把全部差值都归成纯跨人图边的作用。',
        '只用教师预测风险不保证GT/HOTA最优；现有数据全为DEV，区间不校正多轮开发/多重比较，也不重拟合整套适配。',
        '如新配方保留，必须对应真实物理成本，不套旧固定图wall；与新SEARCH叠加必须另做同问答组合实测，不能相加两个研究涨幅。Goal不自动完成。']
    (ROOT/'docs/TTA_RMOT_允许不传播的教师预测选择_机制实证_20261122.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(event='OPTIONAL_TRANSFER_ANALYSIS_COMPLETE',repeated=repeated,goal_status='ACTIVE')),flush=True)


if __name__=='__main__':main()
