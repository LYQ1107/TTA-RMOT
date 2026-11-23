"""Post-seal shared-evidence attribution; never a six-setting success shortcut."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_six_direction import PAIRS,SEEDS,PUBLIC
from scripts.a100_view_fusion import OUT,METHODS
from scripts.a100_view_fusion_launch import conditions,BASELINES,PRIMARY
from scripts.a100_six_direction_analyze import comparison
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_sparse_observation_analyze import fit_audit
from scripts.analyze_pf_video_uncertainty import pooled_hota

CONTRASTS={
    **{'FULL_vs_'+name:(PRIMARY,name) for name in BASELINES},
    'LEARNED_vs_FIXED_RELIABILITY':('DS_FULL','FIXED_FULL'),
    'LEARNED_vs_RAW_PROBABILITY':('DS_FULL','RAW_FULL'),
    'DS_LABELS_ON_SHARED_HEAD':('DS_FINAL','RAW_FINAL'),
    'FIXED_LABELS_ON_SHARED_HEAD':('FIXED_FINAL','RAW_FINAL'),
    'STRUCTURE_ON_DS_HEAD':('DS_FULL','DS_FINAL'),
    'CONTEXT_CONDITIONAL':('DS_JOINT','DS_VISUAL'),
    'VISUAL_CONDITIONAL':('DS_JOINT','DS_CONTEXT'),
    'QUERY_CONSISTENCY_CONDITIONAL':('DS_FULL','DS_JOINT'),
    'TRUE_QUERY_GRAPH_vs_SHUFFLE':('DS_FULL','DS_SHUFFLED')}
for name in BASELINES:
    CONTRASTS['RANK_AT_FULL_COUNT_vs_'+name]=(PRIMARY+'__AT_'+PRIMARY+'_COUNT',name+'__AT_'+PRIMARY+'_COUNT')
    CONTRASTS['RANK_AT_BASE_COUNT_vs_'+name]=(PRIMARY+'__AT_'+name+'_COUNT',name+'__AT_'+name+'_COUNT')


def main():
    assert json.loads((OUT/'all_predictions_sealed_before_GT.json').read_text())['methods']==468
    official=json.loads((OUT/'official.json').read_text());assert official['methods']==468
    data={};summary={};sufficient={};max_error=0.
    for record in conditions():
        pair,seed=record['pair'],record['seed'];root=Path(record['folder']);slug=pair+f'_seed{seed}'
        raw=json.loads((root/'completion.json').read_text());count=json.loads((root/'countmatched/completion.json').read_text())
        assert raw['GT_open_attempts']==count['GT_open_attempts']==raw['new_teacher_encoder_calls']==0
        assert not any(count['CountSelf_changed_query_files'].values())
        paths={r['method']:root/r['method'] for r in raw['methods']}
        paths.update({r['method']:root/'countmatched'/r['method'] for r in count['methods']})
        data[slug]={n:read_stats(p) for n,p in paths.items()}
        for name,row in data[slug].items():
            num,den=row['parent_stats'];reconstructed=float(pooled_hota((num,den),np.ones(len(num))))
            max_error=max(max_error,abs(reconstructed-row['HOTA']))
            sufficient[slug+'/'+name]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],numerator=num.tolist(),denominator=den.tolist())
        summary[slug]=dict(raw_HOTA={n:data[slug][n]['HOTA'] for n in METHODS},teacher_fit_audit=fit_audit(raw['receipts']),
            requested_teacher_questions=raw['logical_teacher_questions'],actual_head_fits=raw['actual_head_fits'],
            CountSelf_changed_query_files=count['CountSelf_changed_query_files'],
            unsupervised_last_prefix_reliability=raw['receipts'][-1]['fusion'])
    repeated={}
    for pair in PAIRS:
        slugs=[pair+f'_seed{s}' for s in SEEDS]
        row=dict(source=PAIRS[pair][0],target=PAIRS[pair][1],neural_seeds=list(SEEDS),
            mean_HOTA={n:float(np.mean([summary[k]['raw_HOTA'][n] for k in slugs])) for n in METHODS},
            comparisons={name:comparison([data[k][a] for k in slugs],[data[k][b] for k in slugs]) for name,(a,b) in CONTRASTS.items()})
        h=row['mean_HOTA'];row['reliability_structure_interaction_point']=(h['DS_FULL']-h['RAW_FULL'])-(h['DS_FINAL']-h['RAW_FINAL'])
        repeated[pair]=row
    assert max_error<1e-8,max_error
    passed={p:all(r['comparisons']['FULL_vs_'+b]['delta_HOTA']>0 for b in BASELINES) for p,r in repeated.items()}
    result=dict(status='FIXED_VIEW_FUSION_SIX_SETTINGS_ANALYZED',primary=PRIMARY,conditions=summary,repeated=repeated,
        pointwise_above_four_registered_ordinary=passed,goal_status='ACTIVE_REQUIRES_SCIENTIFIC_REVIEW',research_goal_complete=False,
        all468_predictions_sealed_before_this_GT=True,manual_target_GT_in_adaptation=False,
        same_requested_teacher_views_candidates_features_training_weights_and_decoder=True,
        new_teacher_encoder_calls=0,new_head_fits_total=sum(r['actual_head_fits'] for r in summary.values()),
        HOTA_parent_sum_max_error=max_error,all_data_development=True,independent_blind_validation=False,
        scope='Fixed actually-requested legacy rho=.3 views, new common three-observation track features. Not a fresh active-policy or end-to-end wall-time comparison.',
        reliability_limits='DS parameters are correlated-view unsupervised estimates, not measured accuracies. Neutral-prior mean-view likelihood ratios are regularized targets, not a raw DS posterior.',
        target_scope='Legacy teacher frame anchors can differ from new PCD observation anchors; every arm shares this track-level evidence/feature mismatch.',
        negative_controls_not_promoted=True)
    atomic_json(OUT/'analysis.json',result)
    (PUBLIC/'view_fusion_analysis.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
    (PUBLIC/'view_fusion_parent_statistics.json').write_text(json.dumps(sufficient,separators=(',',':'))+'\n')
    lines=['TTA-RMOT：三个创新候选在六个跨域实验设置的固定多视角证据机制诊断','2026-11-22','',
        '本研究检验同一批教师证据经过可靠性统筹及结构校准，是否优于直接软/硬教师蒸馏。不是旧v8P66评测重跑。',
        '全体共享旧真实请求rho=.3多视角回答、新三观测PCD、置信度权重、充分训练与独立>.5解码。拒绝复用旧预测、旧belief或identity_vote。',
        '三种子、六个设置，13原始+13数量对照共468输出全部封存后评价。',
        'DS只估计无监督可靠性，不拿拟合参数冒充教师准确率；同视角重复先合并，选择性查询基率不作为目标不存在先验，多视角平均对数似然避免重复累乘。',
        '目标人工标签仅封存后官评使用；全部数据DEV。KITTI四视频无总体置信区间。',
        '固定已有问题路径不验证创新I1的新选题收益，也不能把回放时间当完整部署时间；若组合保留，另做新真实采集/成本。','',
        '设置 | 普通Soft Final | 普通Soft Mean | 普通Hard | 普通含Source残差 | 固定可靠性组合 | 原始概率组合 | DS完整组合']
    for pair,row in repeated.items():
        h=row['mean_HOTA'];lines.append(pair+' '+row['source']+'→'+row['target']+' | '+' | '.join(f'{h[n]:.6f}' for n in (*BASELINES,'FIXED_FULL','RAW_FULL','DS_FULL')))
        lines.append('  因果/数量效应：'+json.dumps(row['comparisons'],ensure_ascii=False,separators=(',',':')))
        lines.append('  可靠性×结构的HOTA交互点值：'+str(row['reliability_structure_interaction_point']))
    lines+=['','完整组合是否逐设置超过四个注册普通对照（仅点值）：'+json.dumps(passed),
        '科研Goal保持ACTIVE，不把队列完成、点值正或固定证据胜出当全部目标已实现。没有用GT挑参数或把事后更高控制升为主方法。']
    (ROOT/'docs/TTA_RMOT_六个跨域设置_多视角证据统筹诊断_20261122.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(event='FIXED_VIEW_FUSION_SIX_SETTINGS_ANALYSIS',mean_HOTA={p:r['mean_HOTA'] for p,r in repeated.items()},
        pointwise_above_ordinary=passed,goal_status='ACTIVE')),flush=True)


if __name__=='__main__':main()
