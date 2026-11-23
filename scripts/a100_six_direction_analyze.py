"""Isolated post-seal six-direction evaluation; no adaptation imports this.

The original v8P table is a historical reference at another budget/decoder,
never an ablation effect. Four-video KITTI is reported without a population CI.
"""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_six_direction import OUT,PUBLIC,PAIRS,SEEDS,METHODS,source
from scripts.a100_six_direction_launch import conditions
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_sparse_observation_analyze import effect,fit_audit
from scripts.analyze_pf_video_uncertainty import pooled_hota

CONTRASTS={
    'JOINT_vs_BASE_PROTOCOL':('JOINT','BASE_PROTOCOL'),
    'JOINT_vs_KD_FINAL':('JOINT','KD_FINAL'),
    'JOINT_vs_KD_MEAN':('JOINT','KD_MEAN'),
    'JOINT_vs_HARD_FINAL':('JOINT','HARD_FINAL'),
    'JOINT_vs_RAW_TEACHER_FILL':('JOINT','RAW_TEACHER_FILL'),
    'CONTEXT_CONDITIONAL':('JOINT','VISUAL_ONLY'),
    'VISUAL_CONDITIONAL':('JOINT','PROTOCOL_GRAPH'),
    'VISUAL_ONLY_vs_KD_FINAL':('VISUAL_ONLY','KD_FINAL'),
    'JOINT_vs_LOCAL_JOINT':('JOINT','LOCAL_JOINT'),
    'RANK_AT_JOINT_COUNT':('JOINT__AT_JOINT_COUNT','BASE_PROTOCOL__AT_JOINT_COUNT'),
    'RANK_AT_BASE_COUNT':('JOINT__AT_BASE_PROTOCOL_COUNT','BASE_PROTOCOL__AT_BASE_PROTOCOL_COUNT')}


def comparison(first,second):
    if len(first[0]['parent_videos'])<5:
        delta=[a['HOTA']-b['HOTA'] for a,b in zip(first,second)]
        return dict(delta_HOTA=float(np.mean(delta)),per_neural_seed_delta=delta,
            original_parent_groups=len(first[0]['parent_videos']),CI95=None,
            limitation='Four original videos: seed sensitivity only, no population confidence interval')
    return effect(first,second)


def diagnosis():
    from flex_uda.a100.mechanism import metadata,OUT as DIAG
    from flex_uda.active.acquisition import expression_types
    result={};canonical={'C1':'C1','C2':'C5','C3':'C4','C4':'C4','C5':'C5','C6':'C1'}
    for record in conditions():
        pair,seed=record['pair'],record['seed'];m=metadata(PAIRS[pair][1]);root=Path(record['folder'])
        totals={name:Counter() for name in ('KD_FINAL','BASE_PROTOCOL','RAW_TEACHER_FILL','JOINT')}
        coverage=Counter();by_type={};available=True
        for video,bank in sorted(m['videos'].items()):
            path=DIAG/'offline_GT'/canonical[pair]/video/'labels.npy'
            if not path.is_file():available=False;break
            truth=np.load(path)>=0;present=truth.any(1);obs=np.asarray(bank['observations'])
            queries=[q for q in m['episodes'] if q['video_id']==video]
            with np.load(source(pair)/'source'/video/'sparse_inputs.npz') as a:tracks=a['tracks']
            asked=np.zeros((len(queries),len(tracks)),bool);positive_yes=asked.copy()
            answers=json.loads((source(pair)/'source'/video/'answers.json').read_text())['answers']
            for row in answers:
                q,t=int(row['local_query']),int(row['local_track']);asked[q,t]=True;positive_yes[q,t]=row['yes_probability']>.5
            lookup={int(t):i for i,t in enumerate(tracks)};columns=np.asarray([lookup[int(t)] for _,t in obs])
            covered=(asked[:,columns]&truth).any(1);yes=(positive_yes[:,columns]&truth).any(1)
            coverage.update(queries=len(queries),recoverable_positive_queries=int(present.sum()),
                no_correct_candidate_asked=int((present&~covered).sum()),no_correct_teacher_yes=int((present&~yes).sum()))
            for name,total in totals.items():
                mask=np.load(root/name/'videos'/video/'mask.npy');assert mask.shape==truth.shape
                good=(mask&truth).any(1)
                total.update(selected_rows=int(mask.sum()),canonical_positive_selected=int((mask&truth).sum()),
                    canonical_negative_selected=int((mask&~truth).sum()),canonical_positive_missed=int((~mask&truth).sum()),
                    recoverable_positive_queries=int(present.sum()),recoverable_positive_without_correct_output=int((present&~good).sum()),
                    recoverable_positive_zero_output=int((present&~mask.any(1)).sum()),
                    no_recoverable_candidate_with_output=int((~present&mask.any(1)).sum()))
                for qi,q in enumerate(queries):
                    for kind in expression_types(q['query_text']):
                        key=kind+'/'+name;by_type.setdefault(key,Counter()).update(queries=1,present=int(present[qi]),
                            missed_positive_query=int(present[qi] and not good[qi]),
                            canonical_positive_selected=int((mask[qi]&truth[qi]).sum()),
                            canonical_negative_selected=int((mask[qi]&~truth[qi]).sum()))
        result[pair+f'_seed{seed}']=dict(available=available,coverage=dict(coverage),outputs={n:dict(v) for n,v in totals.items()},
            query_types_overlapping_not_additive={n:dict(v) for n,v in by_type.items()},
            labels_reference_pair=canonical[pair],same_target_candidate_and_query_order=True)
    return dict(conditions=result,GT_only_after_all234_sealed=True,
        warning='Canonical full-bank one-to-one IoU.5 membership is a diagnostic, not semantic truth or per-method official matching; no recoverable candidate is not certain scene absence')


def main():
    barrier=json.loads((OUT/'all_predictions_sealed_before_GT.json').read_text());assert barrier['methods']==234
    official=json.loads((OUT/'official.json').read_text());assert official['methods']==234
    allstats={};sufficient={};summary={};max_error=0.
    for record in conditions():
        pair,seed=record['pair'],record['seed'];slug=pair+f'_seed{seed}';root=Path(record['folder'])
        raw=json.loads((root/'completion.json').read_text());count=json.loads((root/'countmatched/completion.json').read_text())
        assert raw['GT_open_attempts']==count['GT_open_attempts']==0
        assert not any(count['CountSelf_changed_query_files'].values())
        paths={r['method']:root/r['method'] for r in raw['methods']}
        paths.update({r['method']:root/'countmatched'/r['method'] for r in count['methods']})
        values={name:read_stats(path) for name,path in paths.items()};allstats[slug]=values
        for name,row in values.items():
            num,den=row['parent_stats'];expected=float(pooled_hota((num,den),np.ones(len(num))))
            max_error=max(max_error,abs(expected-row['HOTA']))
            sufficient[slug+'/'+name]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],numerator=num.tolist(),denominator=den.tolist())
        summary[slug]=dict(raw_HOTA={n:values[n]['HOTA'] for n in METHODS},
            logical_teacher_calls=raw['logical_teacher_calls'],teacher_fit_audit=fit_audit(raw['receipts']),
            reused_neural_fit=pair in ('C1','C4'),CountSelf_changed_query_files=count['CountSelf_changed_query_files'])
    repeated={}
    for pair in PAIRS:
        keys=[pair+f'_seed{s}' for s in SEEDS]
        repeated[pair]=dict(source=PAIRS[pair][0],target=PAIRS[pair][1],neural_seeds=list(SEEDS),QA_seed=0,
            mean_HOTA={name:float(np.mean([summary[k]['raw_HOTA'][name] for k in keys])) for name in METHODS},
            comparisons={name:comparison([allstats[k][a] for k in keys],[allstats[k][b] for k in keys]) for name,(a,b) in CONTRASTS.items()})
    assert max_error<1e-8,max_error
    positive={p:all(r['comparisons'][name]['delta_HOTA']>0 for name in ('JOINT_vs_BASE_PROTOCOL','JOINT_vs_KD_FINAL','JOINT_vs_KD_MEAN','JOINT_vs_HARD_FINAL')) for p,r in repeated.items()}
    result=dict(status='SIX_DIRECTION_TRANSFER_MATRIX_ANALYZED',conditions=summary,repeated=repeated,
        primary='JOINT',per_direction_pointwise_above_all_registered_ordinary=positive,
        six_direction_large_gain_goal_complete=False,goal_status='ACTIVE_RESEARCH_REQUIRES_NEXT_MECHANISM_DECISION',
        all_data_development=True,independent_blind_test=False,old_v8p_rho03_not_same_conditions=True,
        budget_fraction=.1,PCD_observations_per_track=3,same_evidence_all_arms=True,
        current_video_transductive=True,manual_target_GT_in_adaptation=False,
        HOTA_parent_sum_max_error=max_error,query_diagnosis=diagnosis(),
        cost_scope='Four physically measured source/teacher evidence streams, then actual component fits; not independent whole-method deployment wall time',
        limits='No parameters or checkpoints chosen by target GT; fixed predeclared recipe but all datasets already development. No universal proof; KITTI only four original videos.')
    atomic_json(OUT/'analysis.json',result)
    PUBLIC.mkdir(exist_ok=True)
    (PUBLIC/'transfer_analysis.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
    (PUBLIC/'transfer_parent_statistics.json').write_text(json.dumps(sufficient,separators=(',',':'))+'\n')
    lines=['TTA-RMOT：六方向统一迁移诊断（持续研究中）','2026-11-22','',
        '用户目标是多个视觉本能/经典算法机制组合，六方向明确贡献并优于公平普通教师蒸馏；不是把C1/C4通过当整个目标完成。',
        '本研究把已开发的协议/视觉校准配方固定后扩展六方向。C1/C4重放原始真实10%流，其余四方向新真实GPU学生/教师流。',
        '普通蒸馏同输入、同问答、同TrackID传播、充分收敛4000..16000步，保留最终/早晚均值/硬标签和同协议对照，未用目标GT选权重。',
        '主组合和各模块拆除均使用完全相同的收敛软头。234原始及数量输出全部封存后才作本组GT评价。',
        '本轮rho=.1，旧v8P表rho=.3且感知/解码不同，不能把两表差值说成新机制贡献。',
        '所有数据都是DEV；KITTI仅四视频，只给点值与种子敏感性，不造总体置信区间。','',
        '方向 | 普通Final | 普通Mean | 普通Hard | 同协议普通 | 直接老师回填 | 完整组合']
    for p,row in repeated.items():
        h=row['mean_HOTA'];lines.append(p+' '+row['source']+'→'+row['target']+' | '+' | '.join(f'{h[n]:.6f}' for n in ('KD_FINAL','KD_MEAN','HARD_FINAL','BASE_PROTOCOL','RAW_TEACHER_FILL','JOINT')))
        lines.append('  模块/数量效应：'+json.dumps(row['comparisons'],ensure_ascii=False,separators=(',',':')))
    lines+=['','是否六方向均超过注册的普通基线（仅点值，不等于大幅/稳健验收）：'+json.dumps(positive),
        '当前Goal仍ACTIVE。接下来必须逐方向诊断剩余误差，设计有依据的组合机制；未满足全部六方向目标不结项。',
        '真实成本另看各源completion/phase_ledger与拟合收据，不把特征回放说成零成本部署。']
    (ROOT/'docs/TTA_RMOT_六方向统一诊断与组合研究_20261122.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(event='SIX_DIRECTION_MATRIX_ANALYZED',mean_HOTA={p:r['mean_HOTA'] for p,r in repeated.items()},
        pointwise_above_ordinary=positive,goal_status='ACTIVE')),flush=True)


if __name__=='__main__':main()
