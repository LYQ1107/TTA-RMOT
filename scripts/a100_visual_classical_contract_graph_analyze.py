"""Post-seal causal audit of negative-evidence transfer, not a new head."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_contract_graph_launch import conditions
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_sparse_observation_analyze import effect
from scripts.a100_visual_classical_sparse_observation_report import describe
from scripts.a100_visual_classical_sparse_statistics_reproduce import resolve_record,share_identical_statistics,methods

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'
ORDINARY=('RES_KD','RES_KD400','RES_KD_FINAL','RES_HARD400','RES_HARD_FINAL','RES_PCD_KD_MEAN','RES_ZERO_HSV_MEAN','RES_SEM_MEAN','RES_SEM400','RES_SEM_HARD400','RGB_KD400','RGB_KD_FINAL','RGB_KD_MEAN','RGB_HARD400','RGB_HARD_FINAL')


def references(slug):
    result={name:dict(file='direct_teacher_pcd_parent_statistics.json',record=slug+'/'+source) for name,source in dict(KD='KD_MEAN',CAL='CAL',FULL_GRAPH='GRAPH',FULL_JOINT='JOINT').items()}
    result.update({name:dict(file='direct_teacher_pcd_parent_statistics.json',record=slug+'/'+name) for name in ORDINARY})
    for name,source in dict(WEIGHTED_KD='KD_MEAN',WEIGHTED_KD400='KD400',WEIGHTED_KD_FINAL='KD_FINAL',WEIGHTED_HARD400='HARD400',WEIGHTED_HARD_FINAL='HARD_FINAL').items():
        result[name]=dict(file='confidence_teacher_parent_statistics.json',record=slug+'/'+source)
    result['GLOBAL_JOINT']=dict(file='native_factorial_parent_statistics.json',record=slug+'/COMPLETE')
    return result


def contrasts():
    primary='CONTRACT_JOINT';names=('KD','CAL','CONTRACT_GRAPH','FULL_JOINT','FULL_GRAPH','GLOBAL_JOINT','POSITIVE_JOINT','WEAK_JOINT','WEIGHTED_KD','WEIGHTED_KD400','WEIGHTED_KD_FINAL','WEIGHTED_HARD400','WEIGHTED_HARD_FINAL')+ORDINARY
    result={primary+'_vs_'+name:(primary,name) for name in names}
    result['CONTRACT_GRAPH_vs_KD']=('CONTRACT_GRAPH','KD')
    for name in ('KD','CAL','CONTRACT_GRAPH','FULL_JOINT','WEAK_JOINT','WEIGHTED_KD'):
        result['COUNT_OWN_'+primary+'_vs_'+name]=(primary+'__AT_'+primary+'_COUNT',name+'__AT_'+primary+'_COUNT')
        result['COUNT_BASELINE_'+primary+'_vs_'+name]=(primary+'__AT_'+name+'_COUNT',name+'__AT_'+name+'_COUNT')
    assert all(tuple(methods(label))==value for label,value in result.items())
    return result


def diagnosis(records):
    from collections import Counter
    from flex_uda.a100.mechanism import OUT as DIAG,metadata,PAIRS
    result={}
    for record in records:
        pair,seed=record['pair'],record['seed'];root=Path(record['folder']);m=metadata(PAIRS[pair][1])
        names=('KD','CAL','FULL_JOINT','WEIGHTED_KD','CONTRACT_GRAPH','CONTRACT_JOINT','POSITIVE_JOINT','WEAK_JOINT')
        native=OUT/'direct_teacher_nores_pcd'/pair/f'seed{seed}'
        refs=dict(KD=native/'KD_MEAN',CAL=native/'CAL',FULL_JOINT=native/'JOINT',WEIGHTED_KD=OUT/'confidence_teacher'/pair/f'seed{seed}'/'KD_MEAN')
        counts={name:Counter() for name in names};trade={name:Counter() for name in ('KD','FULL_JOINT','WEAK_JOINT','WEIGHTED_KD')}
        for video in sorted(m['videos']):
            truth=np.load(DIAG/'offline_GT'/pair/video/'labels.npy')>=0;present=truth.any(1)
            masks={name:np.load((refs[name] if name in refs else root/name)/'videos'/video/'mask.npy') for name in names}
            correct={name:(mask&truth).any(1) for name,mask in masks.items()}
            for name,mask in masks.items():
                counts[name].update(queries=len(mask),recoverable_positive_queries=int(present.sum()),positive_queries_empty=int((present&~mask.any(1)).sum()),
                    positive_queries_without_correct_output=int((present&~correct[name]).sum()),no_positive_candidate_queries_with_output=int((~present&mask.any(1)).sum()),
                    TP=int((mask&truth).sum()),FP=int((mask&~truth).sum()),FN=int((~mask&truth).sum()))
            for name in trade:trade[name].update(rescued=int((present&~correct[name]&correct['CONTRACT_JOINT']).sum()),regressed=int((present&correct[name]&~correct['CONTRACT_JOINT']).sum()))
        result[pair+f'_seed{seed}']=dict(candidate_membership={k:dict(v) for k,v in counts.items()},contract_J_retrieval_versus={k:dict(v) for k,v in trade.items()})
    return dict(conditions=result,postseal_manual_GT_only=True,not_additive_HOTA_attribution=True,query_recall_is_any_IoU_positive_candidate_not_complete_GT=True)


def main():
    official=json.loads((OUT/'analysis/contract_graph_official.json').read_text());assert official['methods']==138
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
            matched_edge_total_max_error=max(r['contract']['matched_edge_total_max_error'] for r in raw['receipts']),CountSelf_changed_query_files=counts['CountSelf_changed_query_files'])
    repeated={}
    for pair in ('C4','C1'):
        keys=[pair+f'_seed{s}' for s in (0,17,42)]
        repeated['CONTRACT_'+pair]=dict(condition_keys=keys,neural_seeds=[0,17,42],QA_seed=0,
            mean_HOTA={n:float(np.mean([summary[k]['raw_HOTA'][n] for k in keys])) for n in summary[keys[0]]['raw_HOTA']},
            comparisons={name:effect([data[k][a] for k in keys],[data[k][b] for k in keys]) for name,(a,b) in pairs.items()})
    diagnostic=diagnosis(conditions())
    result=dict(status='CONTRACT_BOUNDED_NEGATIVE_EVIDENCE_ANALYZED',conditions=summary,repeated=repeated,query_diagnosis=diagnostic,
        primary_rule='PRESUPPOSED isolatesqueriedNoanchors;CHECKLISTunchanged;teacherthreshold.5fixed',
        edge_weight_matched_weakening_control=True,all138_new_outputs_sealed_before_manual_GT=True,new_teacher_encoder_head_calls=0,
        all_development_data=True,independent_blind_validation=False,
        uncertainty='Fixedsealedoriginal-parent/shared-seeddescriptivebootstrap,nopipelinerefitordevelopment/multiplicitycorrection.',
        quality_replay_not_actual_cold_time=True,goal_status='ACTIVE_REQUIRES_ROOT_SCIENTIFIC_REVIEW')
    atomic_json(OUT/'analysis/contract_graph_analysis.json',result)
    (PUBLIC/'contract_graph_analysis.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    compact,_=share_identical_statistics(statistics,'contract_graph_parent_statistics.json')
    (PUBLIC/'contract_graph_parent_statistics.json').write_text(json.dumps(compact,separators=(',',':'))+'\n')
    lines=['TTA-RMOT：任务契约约束的负证据传播审计','2026-11-22','',
        '假设：问到某候选的No在预设存在任务中外溢到未问目标，可能误删短查询。只改变传播边界，不改网络、教师、问题或解码阈值。',
        'PRESUPPOSED断开已问No节点的所有图边，其No锚点仍保留作局部拒绝；CHECKLIST保留原完整传播。单一规则适用于两域，不读取域名/人工GT。',
        '总边权相同的均匀弱化图为强度对照；ALL_POSITIVE在CHECKLIST也隔离No，检验是否需要任务契约。两者不是事后择优主方法。',
        '6条件24原始+114双向数量=138新输出，全部封存后本组GT。0新教师/编码器/神经头拟合。','']
    for pair,row in repeated.items():
        lines += [pair,'三种子平均HOTA：'+json.dumps(row['mean_HOTA'])]
        for name,e in row['comparisons'].items():lines.append(name+'：'+describe(e))
    lines += ['封存后检索、空输出、TP/FP/FN：'+json.dumps(diagnostic,separators=(',',':')),
        '只减少空输出不等于找对人；必须同时检查正确检索、误报、HOTA及等强度对照。相同总边权不意味着每节点度相同。',
        '正增益也只是已用开发数据的描述性实证；如失败必须保留，不能挑弱化图/ALL_POSITIVE改称主方法。',
        'CPU复用质量不等于完整真实墙钟时间；若最终配方改变，需对应真实执行和来源/输出核验。Goal不自动完成。']
    (ROOT/'docs/TTA_RMOT_任务契约与负证据边界_机制实证_20261122.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(event='CONTRACT_GRAPH_ANALYSIS_COMPLETE',repeated=repeated,goal_status='ACTIVE')),flush=True)


if __name__=='__main__':main()
