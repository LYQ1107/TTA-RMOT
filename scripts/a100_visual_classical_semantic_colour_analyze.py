"""Post174-seal semantic feature, monotone shape and strong-KD attribution."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_semantic_colour_launch import conditions
from scripts.a100_visual_classical_semantic_colour import METHODS
from scripts.a100_visual_classical_rgb_parity import PARITY
from scripts.a100_visual_classical_zero_colour import ZERO
from scripts.a100_visual_classical_canonical_pcd import CANON
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_sparse_observation_analyze import effect,fit_audit
from scripts.a100_visual_classical_sparse_observation_report import describe
from scripts.a100_visual_classical_sparse_statistics_reproduce import resolve_record,share_identical_statistics

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'
OLD_RGB={'OLD_RGB400':'RGB_KD400','OLD_RGB_FINAL':'RGB_KD_FINAL','OLD_RGB_MEAN':'RGB_KD_MEAN',
    'OLD_RGB_HARD400':'RGB_HARD400','OLD_RGB_HARD_FINAL':'RGB_HARD_FINAL','OLD_RGB_GRAPH':'RGB_GRAPH_MEAN'}
OLD_PCD={'OLD_PCD400':'KD400','OLD_PCD_FINAL':'KD_FINAL','OLD_PCD_MEAN':'KD_MEAN',
    'OLD_PCD_HARD400':'HARD400','OLD_PCD_HARD_FINAL':'HARD_FINAL'}
OLD_ZERO={'OLD_ZERO400':'ZERO_400','OLD_ZERO_FINAL':'ZERO_FINAL','OLD_ZERO_MEAN':'ZERO_MEAN'}


def contrasts():
    names=[n for n in METHODS if n!='MONO']+list(OLD_RGB)+list(OLD_PCD)+list(OLD_ZERO)
    result={'MONO_vs_'+name:('MONO',name) for name in names}
    result.update({a+'_vs_'+b:(a,b) for a,b in [('SEM_MEAN','ZERO_MEAN'),('SEM400','ZERO400'),('SEM_FINAL','ZERO_FINAL'),
        ('ZERO_MEAN','OLD_RGB_MEAN'),('MONO_ZERO','ZERO_MEAN')]})
    for other in ('SEM_MEAN','ZERO_MEAN','BIN','NAIVE_MEAN'):
        result['COUNT_OWN_MONO_vs_'+other]=('MONO__AT_MONO_COUNT',other+'__AT_MONO_COUNT')
        result['COUNT_BASELINE_MONO_vs_'+other]=('MONO__AT_'+other+'_COUNT',other+'__AT_'+other+'_COUNT')
    for direction,donor in [('OWN','SEM_MEAN'),('BASELINE','ZERO_MEAN')]:
        result['COUNT_'+direction+'_SEM_MEAN_vs_ZERO_MEAN']=('SEM_MEAN__AT_'+donor+'_COUNT','ZERO_MEAN__AT_'+donor+'_COUNT')
    return result


def diagnosis(records):
    from collections import Counter
    from flex_uda.a100.mechanism import OUT as DIAG,metadata,PAIRS
    names=('SEM_MEAN','ZERO_MEAN','MONO','BIN','SHUFFLED','NAIVE_MEAN','MONO_ZERO');result={}
    for record in records:
        pair=record['pair'];root=Path(record['folder']);m=metadata(PAIRS[pair][1]);counts={n:Counter() for n in names};paired={n:Counter() for n in names if n!='SEM_MEAN'}
        for video in sorted(m['videos']):
            truth=np.load(DIAG/'offline_GT'/pair/video/'labels.npy')>=0;present=truth.any(1)
            masks={n:np.load(root/n/'videos'/video/'mask.npy') for n in names};correct={n:(a&truth).any(1) for n,a in masks.items()}
            for n,a in masks.items():
                counts[n].update(queries=len(a),recoverable_positive_queries=int(present.sum()),
                    positive_queries_empty=int((present&~a.any(1)).sum()),positive_queries_without_correct_output=int((present&~correct[n]).sum()),
                    no_positive_candidate_queries_with_output=int((~present&a.any(1)).sum()),TP=int((a&truth).sum()),FP=int((a&~truth).sum()),FN=int((~a&truth).sum()))
            for n in paired:paired[n].update(rescued=int((present&~correct['SEM_MEAN']&correct[n]).sum()),regressed=int((present&correct['SEM_MEAN']&~correct[n]).sum()))
        result[pair+f"_seed{record['seed']}"]=dict(candidate_membership={n:dict(v) for n,v in counts.items()},versus_SEM_KD_retrieval={n:dict(v) for n,v in paired.items()})
    return dict(conditions=result,postseal_GT_only=True,query_retrieval_is_any_IoU_positive_row_not_full_tracking=True,
        positive_presence_means_recoverable_candidate_not_complete_GT=True,not_additive_HOTA_decomposition=True)


def figure(repeated):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),constrained_layout=True)
    bases=['SEM_MEAN','SEM400','SEM_HARD400','ZERO_MEAN','OLD_RGB_MEAN','OLD_ZERO_MEAN','BIN']
    for ax,pair in zip(axes,('C4','C1')):
        row=repeated['SEMANTIC_'+pair]
        for i,base in enumerate(bases):
            e=row['comparisons']['MONO_vs_'+base];lo,hi=e['parent_and_neural_seed_sensitivity_CI95']
            ax.hlines(i,lo,hi,lw=2,color='tab:blue');ax.plot(e['delta_HOTA'],i,'o',color='tab:blue')
        ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(len(bases)),['same-cue KD mean','same-cue soft400','same-cue hard400','zero-cue1081D mean','old RGB1080D mean','old zero-HSV1080D mean','nonmonotone bins'])
        ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set(title=pair+': same actual10%QA, three seeds',xlabel='MONO minus baseline HOTA; parent+seed95%')
    fig.suptitle('Image/query appearance binding + crossfit teacher-only monotone regularization\nAll development data; no additional teacher/encoder calls; no new cold-time claim',fontsize=10)
    dest=PUBLIC/'figures';dest.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest/'semantic_colour.png',dpi=180,bbox_inches='tight')
    fig.savefig(dest/'semantic_colour.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)


def main():
    official=json.loads((OUT/'analysis/semantic_colour_official.json').read_text());assert official['methods']==174
    records=conditions();data={};summary={};sufficient={};reference_checks=0
    public_cache={str((PUBLIC/filename).absolute()):json.loads((PUBLIC/filename).read_text()) for filename in ('strict_rgb_parent_statistics.json','canonical_pcd_parent_statistics.json','zero_colour_parent_statistics.json')}
    for record in records:
        pair,seed=record['pair'],record['seed'];slug=pair+f'_seed{seed}';root=Path(record['folder'])
        raw=json.loads((root/'completion.json').read_text());counts=json.loads((root/'countmatched/completion.json').read_text())
        assert raw['GT_open_attempts']==raw['new_teacher_calls']==0
        assert raw['logical_teacher_calls']==(1160 if pair=='C4' else 1103) and not any(counts['CountSelf_changed_query_files'].values())
        paths={r['method']:root/r['method'] for r in raw['methods']};paths.update({r['method']:root/'countmatched'/r['method'] for r in counts['methods']});aliases={}
        for mapping,bank,filename in [(OLD_RGB,PARITY,'strict_rgb_parent_statistics.json'),(OLD_PCD,CANON,'canonical_pcd_parent_statistics.json'),(OLD_ZERO,ZERO,'zero_colour_parent_statistics.json')]:
            for alias,name in mapping.items():paths[alias]=bank/pair/f'seed{seed}'/name;aliases[alias]=dict(file=filename,record=slug+'/'+name)
        data[slug]={name:read_stats(path) for name,path in paths.items()}
        for name,row in data[slug].items():
            numeric=dict(parents=row['parent_videos'],HOTA=row['HOTA'],numerator=row['parent_stats'][0].tolist(),denominator=row['parent_stats'][1].tolist())
            if name in aliases:
                ref=aliases[name];existing=resolve_record(PUBLIC,ref['file'],ref['record'],public_cache)
                assert numeric==existing,(slug,name,'Published baseline statistic mismatch')
                sufficient[slug+'/'+name]=dict(reference=ref);reference_checks+=1
            else:sufficient[slug+'/'+name]=numeric
        summary[slug]=dict(pair=pair,seed=seed,raw_HOTA={n:r['HOTA'] for n,r in data[slug].items() if '__AT_' not in n},
            comparisons={n:effect([data[slug][a]],[data[slug][b]]) for n,(a,b) in contrasts().items()},
            teacher_fit_audit=fit_audit(raw['receipts']),CountSelf_changed_query_files=counts['CountSelf_changed_query_files'],
            actual_head_fit_seconds={label:sum(r['fits'][label]['wall_seconds'] for r in raw['receipts']) for label in raw['receipts'][0]['fits']},
            active_monotone_nodes=sum(r['active_monotone_nodes'] for r in raw['receipts']),
            actual_all_control_postprocessing_seconds=sum(r['postprocess_seconds'] for r in raw['receipts']),
            postprocessing_wall_includes_all_quality_controls_not_single_deployment=True,
            physical_GPU=raw['physical_GPU'],logical_teacher_calls=raw['logical_teacher_calls'],new_teacher_calls=0)
    repeated={}
    for pair in ('C4','C1'):
        slugs=[pair+f'_seed{s}' for s in (0,17,42)]
        repeated['SEMANTIC_'+pair]=dict(condition_keys=slugs,neural_seeds=[0,17,42],QA_seed=0,
            mean_HOTA={name:float(np.mean([summary[k]['raw_HOTA'][name] for k in slugs])) for name in summary[slugs[0]]['raw_HOTA']},
            comparisons={n:effect([data[k][a] for k in slugs],[data[k][b] for k in slugs]) for n,(a,b) in contrasts().items()})
    diagnostic=diagnosis(records)
    result=dict(status='SEMANTIC_COLOUR_AND_MONOTONE_ANALYZED',conditions=summary,repeated=repeated,primary='MONO_vs_SEM_MEAN',
        query_diagnosis=diagnostic,official_new_methods=174,existing_public_statistic_references_verified=reference_checks,
        no_new_teacher_or_encoder_calls=True,no_target_GT_in_adaptation=True,
        head_dimension=1081,ZERO_and_SEM_same_initialization=True,monotone_map_current_track_group_crossfit=True,
        HSV_joint_from_marginals_is_approximation=True,physical_colour_prototypes_not_fit_to_target_GT=True,
        bounded_odds_are_regularization_not_independent_Bayesian_evidence=True,
        all_strong_existing_baselines_retained=True,all_development_data=True,independent_blind_validation=False,
        uncertainty='Original-parent and shared-QA neural-seed descriptive intervals; no correction for development or multiple comparisons.',
        cost='Paid source/teacher reused. Head and all-control postprocess components only; matching single-method cold measurement still required if retained.',
        goal_status='ACTIVE_REQUIRES_SCIENTIFIC_REVIEW')
    atomic_json(PUBLIC/'semantic_colour_analysis.json',result);atomic_json(OUT/'analysis/semantic_colour_analysis.json',result)
    compact,_=share_identical_statistics(sufficient,'semantic_colour_parent_statistics.json')
    (PUBLIC/'semantic_colour_parent_statistics.json').write_text(json.dumps(compact,separators=(',',':'))+'\n')
    lines=['TTA-RMOT：颜色与衣着部位的语义绑定及经典单调校准','2026-11-22','',
        '用户目标仍是跨域RMOT：官方固定框/ID，目标图片和任务语言，冻结教师指导，适配不读目标人工标签。',
        '本研究来自上一轮保护机制退步及教师高置信错误的开发诊断；不是独立盲测或无调参历史的新域测试。',
        '固定HSV模糊原型连接颜色词与身体部位，深浅队伍用相对上身亮度；HS与V边缘分布相乘是假设，未声称光照不变。',
        'SEM1081普通KD同样得到该标量；ZERO1081同初始化只补0，排除输入维度初始化混杂。全部旧强软/硬KD保留。',
        'PAVA只拟合过去/当前教师软标签，按当前视频TrackID做5折排除；当前轨迹所有查询都不进入自身校准器。',
        '修正=logit(KD)+clip(logit(PAVA)-logit(训练教师平均),-ln3,ln3)。它是有界正则，不是独立贝叶斯证据。',
        '主MONO与SEM_MEAN完全同头/同输入；BIN去掉单调约束，SHUFFLED破坏图像对齐，NAIVE_MEAN简单平均，MONO_ONLY/零头作补充。',
        '84原始+90双向数量全部封存后174官评；0新增教师/源前向，1160/1103同实际回答。',
        '旧基线充分统计采用已有公开记录的精确引用以控制Git<100MB，引用与官评读取数组逐值相等后才公开。',
        '所有描述性区间仍未校正多次开发；固定查询预算、种子、原始GT和指标均未改变。','']
    for family,row in repeated.items():
        lines += [family,'三种子HOTA：'+json.dumps(row['mean_HOTA'])]
        lines += [name+'：'+describe(e) for name,e in row['comparisons'].items()]
    lines += ['', '检索与FP/FN诊断（非HOTA可加分账）：',json.dumps(diagnostic,ensure_ascii=False,indent=2),
        '拟合收据：',json.dumps({k:v['teacher_fit_audit'] for k,v in summary.items()},indent=2),
        '若新模块不优于同输入强KD或仅某个数量投影为正，就保留失败并继续研究。此报告与队列成功不会关闭Goal。',
        '新精度不与旧3512秒拼接；若保留最终方法，仍须对应的真实单方法冷启动测量。']
    (ROOT/'docs/TTA_RMOT_颜色部位语义与单调校准_机制实证_20261122.txt').write_text('\n'.join(lines)+'\n')
    figure(repeated);print(json.dumps(dict(event='SEMANTIC_COLOUR_ANALYSIS_COMPLETE',repeated=repeated,goal_status='ACTIVE')),flush=True)


if __name__=='__main__':main()
