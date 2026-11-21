"""Post-seal canonical KD, classic metrics and positive-agreement attribution."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_canonical_pcd_launch import conditions
from scripts.a100_visual_classical_rgb_parity import PARITY
from scripts.a100_visual_classical_zero_colour import ZERO
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_sparse_observation_analyze import effect,fit_audit
from scripts.a100_visual_classical_sparse_observation_report import describe

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'


def contrasts(pair):
    full='GRAPH_GUARD'
    bases=['KD_MEAN','KD400','KD_FINAL','HARD400','HARD_FINAL','GRAPH_PCD','GRAPH_RGB','GRAPH_COLOUR','GRAPH_HIDDEN','ANCHORS',
        'BASE_RGB400','BASE_RGB_FINAL','BASE_RGB_MEAN','BASE_RGB_HARD400','BASE_RGB_HARD_FINAL','BASE_ZERO400','BASE_ZERO_FINAL','BASE_ZERO_MEAN']
    if pair=='C4':bases+=['BASE_ORIG400','BASE_ORIG_FINAL','BASE_ORIG_MEAN']
    result={full+'_vs_'+other:(full,other) for other in bases}
    result.update({a+'_vs_'+b:(a,b) for a,b in [('GRAPH_PCD','KD_MEAN'),('GRAPH_RGB','GRAPH_PCD'),('GRAPH_HIDDEN','GRAPH_PCD'),
        ('GRAPH_HIDDEN','KD_MEAN'),('KD_MEAN','BASE_ZERO_MEAN')]})
    for other in ('KD_MEAN','GRAPH_PCD','GRAPH_RGB','ANCHORS'):
        result['COUNT_OWN_'+full+'_vs_'+other]=(full+'__AT_'+full+'_COUNT',other+'__AT_'+full+'_COUNT')
        result['COUNT_BASELINE_'+full+'_vs_'+other]=(full+'__AT_'+other+'_COUNT',other+'__AT_'+other+'_COUNT')
    for direction,donor in [('OWN','GRAPH_RGB'),('BASELINE','GRAPH_PCD')]:
        result['COUNT_'+direction+'_GRAPH_RGB_vs_GRAPH_PCD']=('GRAPH_RGB__AT_'+donor+'_COUNT','GRAPH_PCD__AT_'+donor+'_COUNT')
    return result


def query_diagnosis(records):
    from collections import Counter
    from flex_uda.a100.mechanism import OUT as DIAG,metadata,PAIRS
    names=['KD_MEAN','HARD400','GRAPH_PCD','GRAPH_GUARD','GRAPH_RGB','GRAPH_HIDDEN'];result={}
    for record in records:
        pair=record['pair'];root=Path(record['folder']);m=metadata(PAIRS[pair][1]);counts={n:Counter() for n in names};paired={n:Counter() for n in names if n!='KD_MEAN'}
        for video in sorted(m['videos']):
            truth=np.load(DIAG/'offline_GT'/pair/video/'labels.npy')>=0;present=truth.any(1)
            masks={n:np.load(root/n/'videos'/video/'mask.npy') for n in names};correct={n:(a&truth).any(1) for n,a in masks.items()}
            for n,a in masks.items():
                counts[n].update(queries=len(a),recoverable_positive_queries=int(present.sum()),
                    positive_queries_empty=int((present&~a.any(1)).sum()),positive_queries_without_correct_output=int((present&~correct[n]).sum()),
                    no_positive_candidate_queries_with_output=int((~present&a.any(1)).sum()),
                    TP=int((a&truth).sum()),FP=int((a&~truth).sum()),FN=int((~a&truth).sum()))
            for n in paired:paired[n].update(rescued=int((present&~correct['KD_MEAN']&correct[n]).sum()),regressed=int((present&correct['KD_MEAN']&~correct[n]).sum()))
        result[pair+f"_seed{record['seed']}"]=dict(candidate_membership={n:dict(v) for n,v in counts.items()},versus_KD_retrieval={n:dict(v) for n,v in paired.items()})
    return dict(conditions=result,postseal_GT_only=True,query_retrieval_is_any_IoU_positive_row_not_full_tracking=True,
        positive_presence_means_recoverable_candidate_not_all_physical_GT=True,not_additive_HOTA_decomposition=True)


def figure(repeated):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    bases=['KD_MEAN','KD400','HARD400','BASE_RGB_MEAN','BASE_ZERO_MEAN','GRAPH_PCD']
    for ax,pair in zip(axes,('C4','C1')):
        row=repeated['CANONICAL_'+pair]
        for i,base in enumerate(bases):
            e=row['comparisons']['GRAPH_GUARD_vs_'+base];lo,hi=e['parent_and_neural_seed_sensitivity_CI95']
            ax.hlines(i,lo,hi,lw=2,color='tab:blue');ax.plot(e['delta_HOTA'],i,'o',color='tab:blue')
        ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(len(bases)),['canonical mean','canonical soft400','canonical hard400','full RGB mean','zero-HSV mean','unguarded PCD graph'])
        ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set(title=pair+': same actual10%QA, three seeds',xlabel='Primary minus baseline HOTA; parent+seed95%')
    fig.suptitle('Canonical768D frozen-PCD teacher learner + classical graph regularization\nNo new teacher/encoder calls; development data; not another cold deployment measurement',fontsize=10)
    dest=PUBLIC/'figures';dest.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest/'canonical_pcd_graph.png',dpi=180,bbox_inches='tight')
    fig.savefig(dest/'canonical_pcd_graph.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)


def main():
    official=json.loads((OUT/'analysis/canonical_pcd_official.json').read_text());assert official['methods']==156
    records=conditions();data={};summary={};sufficient={}
    for record in records:
        pair,seed=record['pair'],record['seed'];slug=pair+f'_seed{seed}';root=Path(record['folder'])
        raw=json.loads((root/'completion.json').read_text());counts=json.loads((root/'countmatched/completion.json').read_text())
        assert raw['GT_open_attempts']==raw['new_teacher_calls']==0
        assert raw['logical_teacher_calls']==(1160 if pair=='C4' else 1103)
        assert not any(counts['CountSelf_changed_query_files'].values())
        paths={r['method']:root/r['method'] for r in raw['methods']};paths.update({r['method']:root/'countmatched'/r['method'] for r in counts['methods']})
        rgb=PARITY/pair/f'seed{seed}';zero=ZERO/pair/f'seed{seed}'
        paths.update({alias:rgb/name for alias,name in [('BASE_RGB400','RGB_KD400'),('BASE_RGB_FINAL','RGB_KD_FINAL'),('BASE_RGB_MEAN','RGB_KD_MEAN'),
            ('BASE_RGB_HARD400','RGB_HARD400'),('BASE_RGB_HARD_FINAL','RGB_HARD_FINAL')]})
        paths.update({alias:zero/name for alias,name in [('BASE_ZERO400','ZERO_400'),('BASE_ZERO_FINAL','ZERO_FINAL'),('BASE_ZERO_MEAN','ZERO_MEAN')]})
        if pair=='C4':paths.update({alias:rgb/name for alias,name in [('BASE_ORIG400','ORIG_KD400'),('BASE_ORIG_FINAL','ORIG_KD_FINAL'),('BASE_ORIG_MEAN','ORIG_KD_MEAN')]})
        data[slug]={n:read_stats(p) for n,p in paths.items()}
        for n,row in data[slug].items():
            sufficient[slug+'/'+n]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],numerator=row['parent_stats'][0].tolist(),denominator=row['parent_stats'][1].tolist())
        if pair=='C1':
            a=json.loads((root/'GRAPH_GUARD/predictions_seal.json').read_text());b=json.loads((root/'GRAPH_PCD/predictions_seal.json').read_text())
            assert [r['prediction_sha256'] for r in a['episodes']]==[r['prediction_sha256'] for r in b['episodes']]
        summary[slug]=dict(pair=pair,seed=seed,raw_HOTA={n:r['HOTA'] for n,r in data[slug].items() if '__AT_' not in n},
            comparisons={n:effect([data[slug][a]],[data[slug][b]]) for n,(a,b) in contrasts(pair).items()},
            teacher_fit_audit=fit_audit(raw['receipts']),CountSelf_changed_query_files=counts['CountSelf_changed_query_files'],
            actual_head_fit_seconds={label:sum(r['fits'][label]['wall_seconds'] for r in raw['receipts']) for label in raw['receipts'][0]['fits']},
            physical_GPU=raw['physical_GPU'],logical_teacher_calls=raw['logical_teacher_calls'],new_teacher_calls=0)
    repeated={}
    for pair in ('C4','C1'):
        slugs=[pair+f'_seed{s}' for s in (0,17,42)]
        repeated['CANONICAL_'+pair]=dict(condition_keys=slugs,neural_seeds=[0,17,42],QA_seed=0,
            mean_HOTA={n:float(np.mean([summary[k]['raw_HOTA'][n] for k in slugs])) for n in summary[slugs[0]]['raw_HOTA']},
            comparisons={n:effect([data[k][a] for k in slugs],[data[k][b] for k in slugs]) for n,(a,b) in contrasts(pair).items()})
    diagnostic=query_diagnosis(records)
    result=dict(status='CANONICAL_PCD_GRAPH_ANALYZED',conditions=summary,repeated=repeated,primary='GRAPH_GUARD_vs_KD_MEAN',
        query_diagnosis=diagnostic,official_new_methods=156,no_new_teacher_or_encoder_calls=True,no_target_GT_in_adaptation=True,
        canonical_teacher_head_dimension=768,no_handcrafted_RGB_in_learned_head=True,
        main_and_baseline_share_exact_same_soft_head=True,all_strong_existing_baselines_retained=True,
        positive_guard_is_heuristic_regularization_not_statistical_confidence=True,C1_CHECKLIST_guard_exact_noop=True,
        old_singleton_and_set_fallback_failures_preserved=True,all_development_data=True,independent_blind_validation=False,
        uncertainty='Original-parent and seed descriptive intervals; same actual QA, no correction for development or multiple comparisons.',
        cost='Paid source/teacher pools reused; actual head/graph component receipts only. Do not attach this quality to an earlier cold method.',
        goal_status='ACTIVE_REQUIRES_SCIENTIFIC_REVIEW_AND_MATCHED_DEPLOYMENT')
    atomic_json(PUBLIC/'canonical_pcd_analysis.json',result);atomic_json(OUT/'analysis/canonical_pcd_analysis.json',result)
    (PUBLIC/'canonical_pcd_parent_statistics.json').write_text(json.dumps(sufficient,separators=(',',':'))+'\n')
    lines=['TTA-RMOT：标准768维蒸馏与经典图的重新分工','2026-11-22','',
        '既有114/66/42核验未完成整个Goal：颜色拼头在C4退步，集合回填增加FP，C1排序与总增益不稳定。负结果保留。',
        '新基线自然768→64GELU→1，不拼零、不把手工HSV学入头；真实软/硬教师标签，充分4000至16000步，保留400/最终/均值。',
        '图分支共用这个软头；源主干和教师不更新，只使用当前/过去图像与同一实际教师问答，不用目标人工标签。',
        'PCD、RGB混合、纯颜色、同头隐藏64维和无图锚点分别对照，不保证所有视觉线索都有效。',
        '主方法在PCD图上按两个checkpoint的共同正判断提高未问节点的保留强度；明确PRESUPPOSED任务才启用。',
        '权重.1+3.9max(0,2min(p400,pfinal)-1)固定，已问节点KD权重仍0；不宣称真实统计置信或贝叶斯独立证据。',
        'CHECKLIST无此正例承诺，所以主方法严格等同原PCD图；不能给这一方向冒领保护模块贡献。',
        '完整RGB软/硬强KD、去HSV1080控制和C4原掩码头仍保留；自然768与1080架构差不冒充纯颜色效应。',
        '66原始+90双向数量全部封存后才进行156官评；teacher预算相同1160/1103，0新教师或源编码器前向。',
        '全部开发数据，种子共享问答，未校正多重尝试；没有独立新域盲测。','']
    for family,row in repeated.items():
        lines += [family,'三种子HOTA：'+json.dumps(row['mean_HOTA'])]
        lines += [n+'：'+describe(e) for n,e in row['comparisons'].items()]
    lines += ['', '真实检索与FP/FN（不是可加HOTA拆账）：',json.dumps(diagnostic,ensure_ascii=False,indent=2),
        '训练不足的max-step拟合记录不剔除：',json.dumps({k:r['teacher_fit_audit'] for k,r in summary.items()},indent=2),
        '新方法精度不能拼接旧3512秒。选择最终固定配置之后还需对应的真实加载/推理/拟合和外部进程账单。',
        '本脚本不根据队列成功、写完报告或某个最高点自动完成Goal。']
    (ROOT/'docs/TTA_RMOT_标准PCD蒸馏与经典图_机制拆解_20261122.txt').write_text('\n'.join(lines)+'\n')
    figure(repeated);print(json.dumps(dict(event='CANONICAL_PCD_ANALYSIS_COMPLETE',repeated=repeated,goal_status='ACTIVE')),flush=True)


if __name__=='__main__':main()
