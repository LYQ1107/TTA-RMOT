"""Post-seal logic/decoder attribution; no training or parameter selection."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_six_direction import PAIRS,SEEDS,PUBLIC
from scripts.a100_query_consistency import OUT,METHODS
from scripts.a100_query_consistency_launch import conditions
from scripts.a100_six_direction_analyze import comparison
from scripts.a100_visual_classical_analyze import read_stats
from scripts.analyze_pf_video_uncertainty import pooled_hota

CONTRASTS={
    'FULL_vs_KD_FINAL':('LATTICE_JOINT','KD_FINAL'),
    'FULL_vs_KD_MEAN':('LATTICE_JOINT','KD_MEAN'),
    'FULL_vs_HARD_FINAL':('LATTICE_JOINT','HARD_FINAL'),
    'FULL_vs_BASE_PROTOCOL':('LATTICE_JOINT','BASE_PROTOCOL'),
    'FULL_vs_RAW_TEACHER_FILL':('LATTICE_JOINT','RAW_TEACHER_FILL'),
    'CONSISTENCY_CONDITIONAL':('LATTICE_JOINT','UNPROJECTED_JOINT'),
    'CONSISTENCY_ON_KD':('LATTICE_KD','KD_FINAL'),
    'PROTOCOL_VISUAL_CONDITIONAL':('LATTICE_JOINT','LATTICE_KD'),
    'EQUIVALENCE_CONDITIONAL':('EQUIV_JOINT','UNPROJECTED_JOINT'),
    'IMPLICATION_AFTER_EQUIVALENCE':('LATTICE_JOINT','EQUIV_JOINT'),
    'TRUE_GRAPH_vs_QUERY_SHUFFLE':('LATTICE_JOINT','SHUFFLED_JOINT'),
    'RANK_AT_FULL_COUNT':('LATTICE_JOINT__AT_LATTICE_JOINT_COUNT','BASE_PROTOCOL__AT_LATTICE_JOINT_COUNT'),
    'RANK_AT_BASE_COUNT':('LATTICE_JOINT__AT_BASE_PROTOCOL_COUNT','BASE_PROTOCOL__AT_BASE_PROTOCOL_COUNT')}


def main():
    assert json.loads((OUT/'all_predictions_sealed_before_GT.json').read_text())['methods']==252
    official=json.loads((OUT/'official.json').read_text());assert official['methods']==252
    data={};summary={};sufficient={};max_error=0.
    for record in conditions():
        pair,seed=record['pair'],record['seed'];root=Path(record['folder']);slug=pair+f'_seed{seed}'
        raw=json.loads((root/'completion.json').read_text());count=json.loads((root/'countmatched/completion.json').read_text())
        assert raw['GT_open_attempts']==raw['new_teacher_encoder_head_calls']==count['GT_open_attempts']==0
        assert not any(count['CountSelf_changed_query_files'].values())
        paths={m['method']:root/m['method'] for m in raw['methods']}
        paths.update({m['method']:root/'countmatched'/m['method'] for m in count['methods']})
        data[slug]={n:read_stats(p) for n,p in paths.items()}
        for name,row in data[slug].items():
            num,den=row['parent_stats'];reconstructed=float(pooled_hota((num,den),np.ones(len(num))))
            max_error=max(max_error,abs(reconstructed-row['HOTA']))
            sufficient[slug+'/'+name]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],numerator=num.tolist(),denominator=den.tolist())
        summary[slug]=dict(raw_HOTA={n:data[slug][n]['HOTA'] for n in METHODS},
            same_independent_decoder_every_arm=True,CountSelf_changed_query_files=count['CountSelf_changed_query_files'],
            implication_edges=sum(len(r['edges']) for r in raw['receipts']),
            maximum_projection_iterations=max(r['projection']['FULL']['iterations'] for r in raw['receipts']))
    repeated={}
    for pair in PAIRS:
        slugs=[pair+f'_seed{s}' for s in SEEDS]
        row=dict(source=PAIRS[pair][0],target=PAIRS[pair][1],neural_seeds=list(SEEDS),QA_seed=0,
            mean_HOTA={n:float(np.mean([summary[k]['raw_HOTA'][n] for k in slugs])) for n in METHODS},
            comparisons={name:comparison([data[k][a] for k in slugs],[data[k][b] for k in slugs]) for name,(a,b) in CONTRASTS.items()})
        h=row['mean_HOTA'];row['HOTA_nonadditive_interaction_point']=(h['LATTICE_JOINT']-h['UNPROJECTED_JOINT'])-(h['LATTICE_KD']-h['KD_FINAL'])
        repeated[pair]=row
    assert max_error<1e-8,max_error
    passed={p:all(r['comparisons'][n]['delta_HOTA']>0 for n in ('FULL_vs_KD_FINAL','FULL_vs_KD_MEAN','FULL_vs_HARD_FINAL','FULL_vs_BASE_PROTOCOL')) for p,r in repeated.items()}
    result=dict(status='QUERY_CONSISTENCY_SIX_DIRECTION_ANALYZED',primary='LATTICE_JOINT',conditions=summary,repeated=repeated,
        pointwise_above_registered_ordinary=passed,six_direction_research_goal_complete=False,goal_status='ACTIVE_REQUIRES_SCIENTIFIC_REVIEW',
        all252_predictions_sealed_before_this_GT=True,GT_used_in_adaptation=False,new_teacher_encoder_head_calls=0,
        same_independent_decoder_every_arm=True,decoder_relaxation_is_shared_engineering_change_not_our_module_gain=True,
        parser_assumptions='Conservative linguistic regularizer, not annotation truth; unknown qualifiers opaque; a unique video sport qualifier supplied by query set.',
        all_data_development=True,independent_blind_validation=False,HOTA_parent_sum_max_error=max_error,
        cost='Actual CPU operations/output only on prepaid source/teacher/fitted heads; not independent deployment wall time',
        negative_controls_not_promoted=True)
    atomic_json(OUT/'analysis.json',result)
    (PUBLIC/'query_consistency_analysis.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
    (PUBLIC/'query_consistency_parent_statistics.json').write_text(json.dumps(sufficient,separators=(',',':'))+'\n')
    lines=['TTA-RMOT：六方向查询一致性与解码公平性机制实验','2026-11-22','',
        '人类视觉启发：同义描述保持同一理解，细分描述的正判断不能与其上位描述矛盾。经典算法：无标签文本规则构造偏序，凸概率投影。',
        '这是持续开发，不是新独立盲测。全部十种方法使用相同普通逐查询>.5解码，统一移除旧17句互斥假设；不能把放宽解码的收益只算给我们。',
        '同一真实10%教师路径、同一充分收敛头，无新增教师/主干/拟合。全六方向三种子180raw+72CountMatched=252输出封存后评价。',
        '完整组合对独立解码普通蒸馏、同协议普通、直接老师回填、仅一致性、仅原组合、仅同义和随机置乱图全部对照。',
        'KITTI仅四视频不给总体CI；其它CI是多轮开发后固定预测的描述性原视频/种子敏感性区间。','',
        '方向 | KD Final | KD Mean | KD Hard | 同协议KD | 原组合 | 查询一致性+KD | 完整组合']
    for p,row in repeated.items():
        h=row['mean_HOTA'];lines.append(p+' '+row['source']+'→'+row['target']+' | '+' | '.join(f'{h[n]:.6f}' for n in ('KD_FINAL','KD_MEAN','HARD_FINAL','BASE_PROTOCOL','UNPROJECTED_JOINT','LATTICE_KD','LATTICE_JOINT')))
        lines.append('  模块效应：'+json.dumps(row['comparisons'],ensure_ascii=False,separators=(',',':')))
        lines.append('  HOTA四项交互点值（不把各自涨点相加）：'+str(row['HOTA_nonadditive_interaction_point']))
    lines+=['','六方向均赢普通基线的点值检查：'+json.dumps(passed),
        '所有Goal保持ACTIVE，执行PASS不自动验收。存在未赢、模块无贡献或只靠数量变化时须继续重设计，不把事后更高控制升为主方法。',
        '解析依赖语义假设，尚不能把规则等同GT；必要时通过教师-only自洽性诊断修正，不能用目标标签调阈值。']
    (ROOT/'docs/TTA_RMOT_六方向查询一致性与解码公平性_20261122.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(event='QUERY_CONSISTENCY_SIX_DIRECTION_ANALYSIS',mean_HOTA={p:r['mean_HOTA'] for p,r in repeated.items()},
        pointwise_above_ordinary=passed,goal_status='ACTIVE')),flush=True)


if __name__=='__main__':main()
