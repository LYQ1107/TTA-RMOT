"""Post-seal attribution on the shared canonical converged teacher head."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_protocol_calibration_launch import conditions
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_sparse_observation_analyze import effect
from scripts.a100_visual_classical_sparse_observation_report import describe
from scripts.a100_visual_classical_sparse_statistics_reproduce import resolve_record, share_identical_statistics, methods
from scripts.a100_visual_classical_weighted_classical_analyze import ORDINARY

PUBLIC = ROOT/'docs/a100_visual_classical_20261121'
PRIMARY = 'PROTOCOL_JOINT'
COUNT_REFERENCES = ('KD', 'BASE_PROTOCOL', 'FINAL_CAL', 'PROTOCOL_GRAPH')


def references(slug):
    result = {name:dict(file='confidence_teacher_parent_statistics.json', record=slug+'/'+source) for name, source in
        dict(KD='KD_FINAL', KD_MEAN='KD_MEAN', KD400='KD400', HARD400='HARD400', HARD_FINAL='HARD_FINAL').items()}
    for name, source in dict(MEAN_CAL='CAL', MEAN_LOCAL='JOINT').items():
        result[name] = dict(file='weighted_classical_parent_statistics.json', record=slug+'/'+source)
    for name, source in dict(UNWEIGHTED_KD='KD_MEAN', UNWEIGHTED_JOINT='JOINT').items():
        result[name] = dict(file='direct_teacher_pcd_parent_statistics.json', record=slug+'/'+source)
    result.update({name:dict(file='direct_teacher_pcd_parent_statistics.json', record=slug+'/'+name) for name in ORDINARY})
    return result


def contrasts():
    names = tuple(references('unused'))+('BASE_PROTOCOL', 'PROTOCOL_GRAPH', 'FINAL_CAL', 'FINAL_LOCAL', 'FINAL_GLOBAL')
    result = {PRIMARY+'_vs_'+name:(PRIMARY, name) for name in names}
    result.update(PROTOCOL_GRAPH_vs_KD=('PROTOCOL_GRAPH', 'KD'), FINAL_CAL_vs_KD=('FINAL_CAL', 'KD'),
        FINAL_GLOBAL_vs_FINAL_CAL=('FINAL_GLOBAL', 'FINAL_CAL'), FINAL_LOCAL_vs_FINAL_CAL=('FINAL_LOCAL', 'FINAL_CAL'))
    for name in COUNT_REFERENCES:
        result['COUNT_OWN_'+PRIMARY+'_vs_'+name] = (PRIMARY+'__AT_'+PRIMARY+'_COUNT', name+'__AT_'+PRIMARY+'_COUNT')
        result['COUNT_BASELINE_'+PRIMARY+'_vs_'+name] = (PRIMARY+'__AT_'+name+'_COUNT', name+'__AT_'+name+'_COUNT')
    assert all(tuple(methods(label)) == value for label, value in result.items())
    return result


def diagnosis(records):
    from collections import Counter
    from flex_uda.a100.mechanism import OUT as DIAG, metadata, PAIRS
    result = {}
    for record in records:
        pair, seed = record['pair'], record['seed']; root = Path(record['folder']); m = metadata(PAIRS[pair][1])
        names = ('KD', 'BASE_PROTOCOL', 'PROTOCOL_GRAPH', PRIMARY, 'FINAL_CAL', 'FINAL_LOCAL', 'FINAL_GLOBAL')
        kd = OUT/'confidence_teacher'/pair/f'seed{seed}'/'KD_FINAL'
        counts = {name:Counter() for name in names}; trade = {name:Counter() for name in COUNT_REFERENCES}
        for video in sorted(m['videos']):
            truth = np.load(DIAG/'offline_GT'/pair/video/'labels.npy') >= 0; present = truth.any(1)
            masks = {name:np.load((kd if name == 'KD' else root/name)/'videos'/video/'mask.npy') for name in names}
            correct = {name:(mask & truth).any(1) for name, mask in masks.items()}
            for name, mask in masks.items():
                counts[name].update(queries=len(mask), recoverable_positive_queries=int(present.sum()),
                    positive_queries_empty=int((present & ~mask.any(1)).sum()), positive_queries_without_correct_output=int((present & ~correct[name]).sum()),
                    no_positive_candidate_queries_with_output=int((~present & mask.any(1)).sum()),
                    canonical_positive_selected=int((mask & truth).sum()), canonical_negative_selected=int((mask & ~truth).sum()),
                    canonical_positive_missed=int((~mask & truth).sum()))
            for name in trade:
                trade[name].update(rescued=int((present & ~correct[name] & correct[PRIMARY]).sum()), regressed=int((present & correct[name] & ~correct[PRIMARY]).sum()))
        result[pair+f'_seed{seed}'] = dict(candidate_membership={k:dict(v) for k,v in counts.items()}, primary_retrieval_versus={k:dict(v) for k,v in trade.items()})
    return dict(conditions=result, postseal_manual_GT_only=True, not_additive_HOTA_attribution=True,
        canonical_labels='Full provided bank one-to-one per-query frame IoU.5 matching; not teacher semantic truth or official per-method matching',
        query_recall_is_any_IoU_positive_candidate_not_complete_GT=True)


def main():
    official = json.loads((OUT/'analysis/protocol_calibration_official.json').read_text()); assert official['methods'] == 114
    data = {}; summary = {}; statistics = {}; cache = {}; pairs = contrasts()
    for record in conditions():
        pair, seed = record['pair'], record['seed']; slug = pair+f'_seed{seed}'; root = Path(record['folder'])
        raw = json.loads((root/'completion.json').read_text()); counts = json.loads((root/'countmatched/completion.json').read_text())
        assert raw['GT_open_attempts'] == raw['new_teacher_calls'] == raw['new_encoder_calls'] == raw['new_head_fits'] == 0
        assert not any(counts['CountSelf_changed_query_files'].values())
        paths = {m['method']:root/m['method'] for m in raw['methods']}; paths.update({m['method']:root/'countmatched'/m['method'] for m in counts['methods']})
        data[slug] = {name:read_stats(path) for name, path in paths.items()}
        for name, row in data[slug].items():
            statistics[slug+'/'+name] = dict(parents=row['parent_videos'], HOTA=row['HOTA'], numerator=row['parent_stats'][0].tolist(), denominator=row['parent_stats'][1].tolist())
        for name, ref in references(slug).items():
            row = resolve_record(PUBLIC, ref['file'], ref['record'], cache)
            data[slug][name] = dict(parent_videos=row['parents'], HOTA=row['HOTA'], parent_stats=(np.asarray(row['numerator']), np.asarray(row['denominator'])))
            statistics[slug+'/'+name] = dict(reference=ref)
        summary[slug] = dict(pair=pair, seed=seed, raw_HOTA={n:r['HOTA'] for n,r in data[slug].items() if '__AT_' not in n},
            comparisons={name:effect([data[slug][a]], [data[slug][b]]) for name,(a,b) in pairs.items()},
            same_exact_final_head_as_ordinary=all(r['same_exact_final_head_as_ordinary'] for r in raw['receipts']),
            frozen_final_mask_differences=sum(r['frozen_prediction_max_errors']['KD_FINAL_mask_changed'] for r in raw['receipts']),
            CountSelf_changed_query_files=counts['CountSelf_changed_query_files'])
    repeated = {}
    for pair in ('C4', 'C1'):
        keys = [pair+f'_seed{s}' for s in (0,17,42)]
        repeated['PROTOCOL_CALIBRATION_'+pair] = dict(condition_keys=keys, neural_seeds=[0,17,42], QA_seed=0,
            mean_HOTA={n:float(np.mean([summary[k]['raw_HOTA'][n] for k in keys])) for n in summary[keys[0]]['raw_HOTA']},
            comparisons={name:effect([data[k][a] for k in keys], [data[k][b] for k in keys]) for name,(a,b) in pairs.items()})
    result = dict(status='CONVERGED_HEAD_PROTOCOL_CALIBRATION_ANALYZED', conditions=summary, repeated=repeated, query_diagnosis=diagnosis(conditions()),
        primary_rule='Shared confidence-weighted converged FINAL; PRESUPPOSED exact identity, CHECKLIST total-edge-matched global pool; same prefix TrackID-crossfit visual PAVA',
        ordinary_protocol_control='Same FINAL independent Bernoulli marginals conditioned on nonempty candidate set for PRESUPPOSED; CHECKLIST unchanged',
        all114_new_outputs_sealed_before_manual_GT=True, new_teacher_encoder_head_calls=0,
        development_derived_simplification_not_independent_confirmation=True, all_development_data=True, independent_blind_validation=False,
        uncertainty='Fixed sealed original-parent/shared-seed descriptive bootstrap; no pipeline refit or development/multiplicity correction',
        quality_replay_not_actual_cold_time=True, goal_status='ACTIVE_REQUIRES_ROOT_SCIENTIFIC_REVIEW')
    atomic_json(OUT/'analysis/protocol_calibration_analysis.json', result)
    (PUBLIC/'protocol_calibration_analysis.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    compact, _ = share_identical_statistics(statistics, 'protocol_calibration_parent_statistics.json')
    (PUBLIC/'protocol_calibration_parent_statistics.json').write_text(json.dumps(compact,separators=(',',':'))+'\n')
    lines = ['TTA-RMOT：共享收敛蒸馏头上的任务协议与视觉校准', '2026-11-22', '',
        '这是已知开发结果后的简化研究，非独立盲测。旧门控与SEARCH均未通过，不拼接其涨幅。',
        '双方使用完全相同的强置信度加权768D、充分收敛FINAL快照；均值/400步/硬标签/旧RGB及SEM普通基线全部保留。',
        '任务给定PRESUPPOSED时原样保留KD，CHECKLIST时使用原同总边权全局图，然后统一进行原有跨TrackID、过去/当前视频前缀的PAVA视觉证据修正。不按数据集名字或GT切规则。',
        '普通基线也获得同一任务协议：对独立Bernoulli边缘条件化为至少一个候选；清单协议原样。此控制假设固定候选覆盖场景，不提供GT或强制单人。',
        '固定局部/全局以及仅PAVA对照，36原始+78双向数量=114新输出全部封存后本组GT；0新增教师/编码器/拟合。',
        '全局秩一求解与原稠密求解保持同一能量；只求解阶段线性，建图归一化仍可能平方复杂度，不把回放时间当物理冷流。', '']
    for pair, row in repeated.items():
        lines += [pair, '三种子平均HOTA：'+json.dumps(row['mean_HOTA'])]
        lines += [name+'：'+describe(e) for name,e in row['comparisons'].items()]
    lines += ['封存后canonical候选归属与查询诊断：'+json.dumps(result['query_diagnosis'],separators=(',',':')),
        'Canonical标签是全候选IoU.5一对一匹配，不是教师语义准确率，也非各方法官方实际TP/FP；HOTA使用独立官方评测。',
        '数量与排序拆解两种投影均报告；不把压低输出量等同语义排序。所有数据为DEV，多轮开发和多重比较未校正，区间仅描述固定预测的重采样。',
        '仅在对强FINAL/协议基线以及模块条件贡献通过后，启动对应四臂真实成本；不能套旧冷流时间。Goal不随执行PASS自动完成。']
    (ROOT/'docs/TTA_RMOT_共享收敛头的任务协议与视觉校准_机制实证_20261122.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(event='PROTOCOL_CALIBRATION_ANALYSIS_COMPLETE', repeated=repeated, goal_status='ACTIVE')), flush=True)


if __name__ == '__main__':
    main()
