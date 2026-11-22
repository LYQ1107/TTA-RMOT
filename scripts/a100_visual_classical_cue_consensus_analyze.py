"""Post-seal audit of the registered visual-conflict veto and matched controls."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_cue_consensus_launch import conditions
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_sparse_observation_analyze import effect
from scripts.a100_visual_classical_sparse_observation_report import describe
from scripts.a100_visual_classical_sparse_statistics_reproduce import resolve_record,share_identical_statistics,methods
from scripts.a100_visual_classical_weighted_classical_analyze import ORDINARY

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'
PRIMARY='CONSENSUS_JOINT'
COUNT_REFERENCES=('KD','CAL','CONSENSUS_GRAPH','FULL_JOINT','WEAK_JOINT','RANDOM_JOINT')


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
    names=tuple(references('unused'))+('CONSENSUS_GRAPH','ALL_PROTOCOL_JOINT','WEAK_JOINT','RANDOM_JOINT')
    result={PRIMARY+'_vs_'+name:(PRIMARY,name) for name in names}
    result['CONSENSUS_GRAPH_vs_KD']=('CONSENSUS_GRAPH','KD')
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
        names=('KD','CAL','FULL_JOINT','CONSENSUS_GRAPH',PRIMARY,'ALL_PROTOCOL_JOINT','WEAK_JOINT','RANDOM_JOINT')
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
        result[pair+f'_seed{seed}']=dict(candidate_membership={k:dict(v) for k,v in counts.items()},consensus_J_retrieval_versus={k:dict(v) for k,v in trade.items()})
    return dict(conditions=result,postseal_manual_GT_only=True,not_additive_HOTA_attribution=True,query_recall_is_any_IoU_positive_candidate_not_complete_GT=True)


def main():
    official=json.loads((OUT/'analysis/cue_consensus_official.json').read_text());assert official['methods']==144
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
            matched_edge_total_max_error=max(max(r['contract']['matched_edge_total_errors'].values()) for r in raw['receipts']),
            conflicting_negative_nodes=sum(r['contract']['visual_conflicting_negative_nodes'] for r in raw['receipts']),
            conflicting_queries=sum(r['contract']['affected_queries'] for r in raw['receipts']),
            CountSelf_changed_query_files=counts['CountSelf_changed_query_files'])
    repeated={}
    for pair in ('C4','C1'):
        keys=[pair+f'_seed{s}' for s in (0,17,42)]
        repeated['CUE_CONSENSUS_'+pair]=dict(condition_keys=keys,neural_seeds=[0,17,42],QA_seed=0,
            mean_HOTA={n:float(np.mean([summary[k]['raw_HOTA'][n] for k in keys])) for n in summary[keys[0]]['raw_HOTA']},
            comparisons={name:effect([data[k][a] for k in keys],[data[k][b] for k in keys]) for name,(a,b) in pairs.items()})
    diagnostic=diagnosis(conditions())
    result=dict(status='CUE_CONSENSUS_NEGATIVE_TRANSFER_ANALYZED',conditions=summary,repeated=repeated,query_diagnosis=diagnostic,
        primary_rule='PRESUPPOSED: isolateNoonlyifleaveTrackIDoutvisualevidence>=existingln3cap;CHECKLISTfull',
        same_exact_head_as_competitive_weighted_ordinary=True,all144_new_outputs_sealed_before_manual_GT=True,new_teacher_encoder_head_calls=0,
        random_negative_isolation_and_uniform_weakening_total_edge_weight_matched=True,
        all_development_data=True,independent_blind_validation=False,
        uncertainty='Fixedsealedoriginal-parent/shared-seeddescriptivebootstrap,nopipelinerefitordevelopment/multiplicitycorrection.',
        quality_replay_not_actual_cold_time=True,goal_status='ACTIVE_REQUIRES_ROOT_SCIENTIFIC_REVIEW')
    atomic_json(OUT/'analysis/cue_consensus_analysis.json',result)
    (PUBLIC/'cue_consensus_analysis.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    compact,_=share_identical_statistics(statistics,'cue_consensus_parent_statistics.json')
    (PUBLIC/'cue_consensus_parent_statistics.json').write_text(json.dumps(compact,separators=(',',':'))+'\n')
    lines=['TTA-RMOT：留轨迹视觉冲突约束的负证据传播实验','2026-11-22','',
        '全面阻断No已失败：相对完整图HOTA下降、误报增加。本研究只在排除当前TrackID监督后的视觉信号达到原有ln3正修正上限时，限制No向外传播。',
        '使用同一个置信度加权普通768维头；局部No锚点、解码、教师回答、原有PAVA均不改变。CHECKLIST保持完整传播。',
        '等总边权弱化、等数量随机No隔离、全协议隔离都是预登记对照，不能事后替代主方法。',
        '6条件30原始+114双向数量=144新输出，全部封存后本组GT。0新增教师/编码器/神经头拟合；门槛不是由目标GT搜索。','']
    for pair,row in repeated.items():
        lines += [pair,'三种子平均HOTA：'+json.dumps(row['mean_HOTA'])]
        for name,e in row['comparisons'].items():lines.append(name+'：'+describe(e))
    lines += ['封存后检索、空输出、TP/FP/FN：'+json.dumps(diagnostic,separators=(',',':')),
        '图与视觉校准证据相关，这是一种有界正则化，不把它描述为独立贝叶斯证据。',
        '只减少空输出不等于找对人；相同总边权不等于各节点度相同；全部数据为开发数据，不是独立盲测。',
        '是否通过按预登记标准核验。若新方案失败，保留失败，不挑随机或其他协议控制作为主方法。',
        'CPU复用质量不代表完整端到端真实墙钟成本；若保留新配方，必须有对应真实物理执行。Goal不自动完成。']
    (ROOT/'docs/TTA_RMOT_留轨迹视觉冲突与负证据传播_机制实证_20261122.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(event='CUE_CONSENSUS_ANALYSIS_COMPLETE',repeated=repeated,goal_status='ACTIVE')),flush=True)


if __name__=='__main__':main()
