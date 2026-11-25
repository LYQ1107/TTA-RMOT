"""Post-seal HOTA and query-retrieval costs of rejecting empty graph sets."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_null_set import NULL,HEADS
from scripts.a100_visual_classical_null_set_launch import conditions
from scripts.a100_visual_classical_rgb_parity import PARITY
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_sparse_observation_analyze import effect
from scripts.a100_visual_classical_sparse_observation_report import describe

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'


def contrasts():
    full='SAFE_MEAN';result={}
    for head in HEADS:
        for other in (f'KD_{head}_PRESENCE',f'GRAPH_{head}_PRESENCE'):
            result[full+'_vs_'+other]=(full,other)
        for other in (f'KD_{head}_PRESENCE',f'GRAPH_{head}_PRESENCE'):
            result[f'SAFE_{head}_vs_'+other]=(f'SAFE_{head}',other)
    for other in ('RAW_KD_MEAN','RAW_GRAPH_MEAN'):
        result[full+'_vs_'+other]=(full,other)
    for other in ('KD_MEAN_PRESENCE','GRAPH_MEAN_PRESENCE'):
        result['COUNT_OWN_'+full+'_vs_'+other]=(full+'__AT_'+full+'_COUNT',other+'__AT_'+full+'_COUNT')
        result['COUNT_BASELINE_'+full+'_vs_'+other]=(full+'__AT_'+other+'_COUNT',other+'__AT_'+other+'_COUNT')
    return result


def retrieval_diagnosis():
    from collections import Counter
    from flex_uda.a100.mechanism import OUT as DIAG,metadata
    m=metadata('SportsMOT');names=('RAW_KD_MEAN','RAW_GRAPH_MEAN','KD_MEAN_PRESENCE','GRAPH_MEAN_PRESENCE','SAFE_MEAN');result={}
    for seed in (0,17,42):
        paths={n:NULL/f'seed{seed}'/'C4'/n for n in names}
        paths.update(RAW_KD_MEAN=PARITY/'C4'/f'seed{seed}'/'RGB_KD_MEAN',RAW_GRAPH_MEAN=PARITY/'C4'/f'seed{seed}'/'RGB_GRAPH_MEAN')
        counts={n:Counter() for n in names};paired={n:Counter() for n in names if n!='KD_MEAN_PRESENCE'}
        for video in sorted(m['videos']):
            truth=np.load(DIAG/'offline_GT/C4'/video/'labels.npy')>=0;present=truth.any(1)
            masks={n:np.load(p/'videos'/video/'mask.npy') for n,p in paths.items()}
            correct={n:(mask&truth).any(1) for n,mask in masks.items()}
            for n,mask in masks.items():
                counts[n].update(queries=len(mask),positive_candidate_queries=int(present.sum()),
                    positive_queries_with_zero_output=int((present&~mask.any(1)).sum()),
                    positive_queries_without_correct_output=int((present&~correct[n]).sum()),
                    TP=int((mask&truth).sum()),FP=int((mask&~truth).sum()),FN=int((~mask&truth).sum()))
            base=correct['KD_MEAN_PRESENCE']
            for n in paired:
                paired[n].update(rescued_queries=int((present&~base&correct[n]).sum()),regressed_queries=int((present&base&~correct[n]).sum()))
        result[str(seed)]=dict(counts={n:dict(v) for n,v in counts.items()},versus_protected_KD={n:dict(v) for n,v in paired.items()})
    return dict(per_seed=result,GT_only_postseal=True,not_additive_HOTA_attribution=True,forced_nonempty_is_not_proof_of_correct_retrieval=True)


def checklist_identity():
    from flex_uda.a100.mechanism import metadata
    from flex_uda.active.task_decode import nonempty_from_reference,retain_nonempty_distillation
    m=metadata('Dance');checked=0
    for seed in (0,17,42):
        root=PARITY/'C1'/f'seed{seed}'
        for video in sorted(m['videos']):
            for kd,graph in HEADS.values():
                base=np.load(root/kd/'videos'/video/'mask.npy');before=np.load(root/graph/'videos'/video/'mask.npy')
                protected,changes=nonempty_from_reference(base,None,None,None,None,'CHECKLIST')
                after,rejected=retain_nonempty_distillation(before,protected,'CHECKLIST')
                assert not changes and not rejected.any() and np.array_equal(base,protected) and np.array_equal(before,after)
                checked+=1
    return dict(status='PASS_CHECKLIST_IDENTITY',matched_head_video_checks=checked,changed_candidate_rows=0,
        existing_C1_predictions_reused=True,no_C1_GT_in_identity_check=True)


def main():
    official=json.loads((OUT/'analysis/null_set_official.json').read_text());assert official['methods']==66
    data={};summary={};sufficient={}
    for record in conditions():
        seed=record['seed'];slug=f'C4_seed{seed}';root=Path(record['folder']);raw=json.loads((root/'completion.json').read_text())
        counts=json.loads((root/'countmatched/completion.json').read_text());assert raw['GT_open_attempts']==raw['new_teacher_calls']==raw['new_head_fits']==0
        assert not any(counts['CountSelf_changed_query_files'].values())
        paths={r['method']:root/r['method'] for r in raw['methods']}
        paths.update({r['method']:root/'countmatched'/r['method'] for r in counts['methods']})
        paths.update(RAW_KD_MEAN=PARITY/'C4'/f'seed{seed}'/'RGB_KD_MEAN',RAW_GRAPH_MEAN=PARITY/'C4'/f'seed{seed}'/'RGB_GRAPH_MEAN')
        data[slug]={n:read_stats(p) for n,p in paths.items()}
        for n,row in data[slug].items():
            sufficient[slug+'/'+n]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],numerator=row['parent_stats'][0].tolist(),denominator=row['parent_stats'][1].tolist())
        summary[slug]=dict(seed=seed,raw_HOTA={n:r['HOTA'] for n,r in data[slug].items() if '__AT_' not in n},
            comparisons={n:effect([data[slug][a]],[data[slug][b]]) for n,(a,b) in contrasts().items()},
            CountSelf_changed_query_files=counts['CountSelf_changed_query_files'],postprocess_output_seconds=sum(r['CPU_seconds'] for r in raw['receipts']),
            additional_teacher_calls=0,additional_head_fits=0)
    slugs=[f'C4_seed{s}' for s in (0,17,42)]
    repeated={'NULL_SET_C4':dict(condition_keys=slugs,neural_seeds=[0,17,42],QA_seed=0,
        mean_HOTA={n:float(np.mean([summary[k]['raw_HOTA'][n] for k in slugs])) for n in summary[slugs[0]]['raw_HOTA']},
        comparisons={n:effect([data[k][a] for k in slugs],[data[k][b] for k in slugs]) for n,(a,b) in contrasts().items()})}
    identity=checklist_identity();diagnosis=retrieval_diagnosis()
    c1=json.loads((PUBLIC/'strict_rgb_analysis.json').read_text())['repeated']['STRICT_RGB_C1']
    result=dict(status='NULL_SET_SAFEGUARD_ANALYZED',conditions=summary,repeated=repeated,query_retrieval_diagnosis=diagnosis,
        C1_CHECKLIST_identity=identity,C1_unchanged_result_reference=c1,all_development_data=True,independent_blind_validation=False,
        additional_teacher_or_encoder_or_neural_updates=0,official_new_methods=66,
        primary='SAFE_MEAN_vs_KD_MEAN_PRESENCE',
        inference_contract_not_estimated_from_GT=True,supplemental_hard_controls_get_extra_strong_soft_reference_head=True,
        compute='Primary uses exactly the same soft head and no new teacher. Hard-control hybrids additionally reuse soft-reference head. Postprocess measurements are not a new cold run.',
        uncertainty='Original-parent and seed descriptive intervals; fixed QA and development iteration; no multiple-testing correction.',goal_status='ACTIVE_REQUIRES_CONTRIBUTION_AND_ERROR_REVIEW')
    atomic_json(PUBLIC/'null_set_analysis.json',result);atomic_json(OUT/'analysis/null_set_analysis.json',result)
    (PUBLIC/'null_set_parent_statistics.json').write_text(json.dumps(sufficient,separators=(',',':'))+'\n')
    lines=['TTA-RMOT：图推断压空目标集合的协议保护对照','2026-11-22','',
        '本轮来自新鲜物理预测封存后的诊断：总HOTA提高仍可能让更多查询失去全部目标。',
        '旧30%静态单例回填和宽泛no-Yes保护已是有限/负结果，本轮不改写它们。',
        '新规则只在原图全空且调用方保证视频存在目标时触发，保留匹配KD的完整集合。',
        '所有五种强教师checkpoint均得到同一个soft均值reference的非空保护；普通KD也享受协议信息。',
        '主比较均值KD和图均只需原一个软头；辅助硬基线得到额外软reference头，不冒充同单头成本。',
        '已提供框/Track ID不改变，源与教师冻结，0额外教师/主干/头训练，GT仅用于全部66输出封存后的官评。',
        'C1任务为CHECKLIST，全部新干预严格不改变已有输出，不能给不存在查询硬塞目标。',
        '全部开发数据；本轮非独立盲测，三个种子共享问答。']
    row=repeated['NULL_SET_C4'];lines += ['', '三种子HOTA：',json.dumps(row['mean_HOTA'],indent=2),'', '完整比较：']
    lines += [n+'：'+describe(e) for n,e in row['comparisons'].items()]
    lines += ['', '检索与误差：',json.dumps(diagnosis,ensure_ascii=False,indent=2),'', 'C1原样性：',json.dumps(identity,indent=2),
        '强制非空只是规则的构造性质，不能当作找对目标。必须看真实正确目标检索与FP/FN和官方HOTA。',
        'SAFE与同reference非空图的主要区别是保留KD集合数量，不把它叫神经语义学习或新的纯排序。',
        '如果模块不胜同等保护的强KD或牺牲大量正确检索，应保留负结果并继续研究，不能通过改指标结项。']
    (ROOT/'docs/TTA_RMOT_目标集合压空_协议保护机制验证_20261122.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(event='NULL_SET_ANALYSIS_COMPLETE',repeated=repeated,identity=identity,goal_status='ACTIVE')),flush=True)


if __name__=='__main__':main()
