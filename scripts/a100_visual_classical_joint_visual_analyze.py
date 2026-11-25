"""Post-seal graph/calibration factorial and stronger teacher controls."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_joint_visual import METHODS
from scripts.a100_visual_classical_joint_visual_launch import conditions
from scripts.a100_visual_classical_rgb_parity import PARITY
from scripts.a100_visual_classical_canonical_pcd import CANON
from scripts.a100_visual_classical_zero_colour import ZERO
from scripts.a100_visual_classical_semantic_colour import SEMANTIC
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_sparse_observation_analyze import effect
from scripts.a100_visual_classical_sparse_observation_report import describe
from scripts.a100_visual_classical_sparse_statistics_reproduce import resolve_record,share_identical_statistics

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'
RGB={'KD':'RGB_KD_MEAN','GRAPH':'RGB_GRAPH_MEAN','KD400':'RGB_KD400','KD_FINAL':'RGB_KD_FINAL',
    'HARD400':'RGB_HARD400','HARD_FINAL':'RGB_HARD_FINAL','GRAPH400':'RGB_GRAPH400',
    'GRAPH_FINAL':'RGB_GRAPH_FINAL','GRAPH_HARD400':'RGB_GRAPH_HARD400','GRAPH_HARD_FINAL':'RGB_GRAPH_HARD_FINAL'}
PCD={('PCD_'+name):name for name in ('KD400','KD_FINAL','KD_MEAN','HARD400','HARD_FINAL')}
HSV={('ZERO_HSV_'+name):'ZERO_'+name for name in ('400','FINAL','MEAN')}
SEM={name:name for name in ('SEM400','SEM_FINAL','SEM_MEAN','SEM_HARD400','SEM_HARD_FINAL')}
SEM['PREVIOUS_MONO']='MONO'


def external(record):
    pair,seed=record['pair'],record['seed'];slug=pair+f'_seed{seed}';result={}
    for mapping,root,filename in [(RGB,PARITY,'strict_rgb_parent_statistics.json'),(PCD,CANON,'canonical_pcd_parent_statistics.json'),
            (HSV,ZERO,'zero_colour_parent_statistics.json'),(SEM,SEMANTIC,'semantic_colour_parent_statistics.json')]:
        for alias,name in mapping.items():result[alias]=(root/pair/f'seed{seed}'/name,dict(file=filename,record=slug+'/'+name))
    return result


def contrasts():
    others=list(METHODS)+list(RGB)+list(PCD)+list(HSV)+list(SEM)
    result={'JOINT_vs_'+name:('JOINT',name) for name in others if name!='JOINT'}
    for a,b in [('CAL','KD'),('GRAPH','KD'),('EARLY_GRAPH','SEM400'),('EARLY_CAL','SEM400'),
            ('EARLY_JOINT','SEM400'),('EARLY_JOINT','EARLY_GRAPH'),('EARLY_JOINT','EARLY_CAL')]:result[a+'_vs_'+b]=a,b
    for other in ('KD','GRAPH','CAL','ROW_SHIFT'):
        result['COUNT_OWN_JOINT_vs_'+other]=('JOINT__AT_JOINT_COUNT',other+'__AT_JOINT_COUNT')
        result['COUNT_BASELINE_JOINT_vs_'+other]=('JOINT__AT_'+other+'_COUNT',other+'__AT_'+other+'_COUNT')
    return result


def diagnosis(records):
    from collections import Counter
    from flex_uda.a100.mechanism import OUT as DIAG,metadata,PAIRS
    result={};names=('KD','GRAPH','CAL','JOINT','SHUFFLED','ROW_SHIFT','GLOBAL_SHUFFLED','QUERY_TEACHER','ANCHOR_CAL')
    for record in records:
        pair=record['pair'];root=Path(record['folder']);m=metadata(PAIRS[pair][1]);refs=external(record)
        counts={name:Counter() for name in names};tradeoffs={name:Counter() for name in ('GRAPH','CAL','JOINT')}
        for video in sorted(m['videos']):
            truth=np.load(DIAG/'offline_GT'/pair/video/'labels.npy')>=0;present=truth.any(1)
            masks={n:np.load((refs[n][0] if n in refs else root/n)/'videos'/video/'mask.npy') for n in names}
            correct={n:(a&truth).any(1) for n,a in masks.items()}
            for n,a in masks.items():
                counts[n].update(queries=len(a),recoverable_positive_queries=int(present.sum()),
                    positive_queries_empty=int((present&~a.any(1)).sum()),positive_queries_without_correct_output=int((present&~correct[n]).sum()),
                    no_positive_candidate_queries_with_output=int((~present&a.any(1)).sum()),TP=int((a&truth).sum()),FP=int((a&~truth).sum()),FN=int((~a&truth).sum()))
            for n in tradeoffs:tradeoffs[n].update(rescued=int((present&~correct['KD']&correct[n]).sum()),regressed=int((present&correct['KD']&~correct[n]).sum()))
        result[pair+f"_seed{record['seed']}"]=dict(candidate_membership={k:dict(v) for k,v in counts.items()},versus_KD_retrieval={k:dict(v) for k,v in tradeoffs.items()})
    return dict(conditions=result,postseal_GT_only=True,not_additive_HOTA_accounting=True,
        retrieval_is_any_IoU_positive_candidate_row_not_complete_GT=True)


def figure(repeated):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['KD','GRAPH','CAL','ANCHOR_CAL','ROW_SHIFT','SHUFFLED','GLOBAL_SHUFFLED','QUERY_TEACHER','SEM400','ZERO_HSV_MEAN']
    labels=['ordinary RGB KD','graph only','calibration only','anchors + calibration','query-wide shift','within-query shuffled cue',
        'globally shuffled cue','teacher query prior','strong semantic soft400','strong zero-HSV KD mean']
    fig,axes=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
    for ax,pair in zip(axes,('C4','C1')):
        row=repeated['JOINT_'+pair]
        for i,name in enumerate(names):
            e=row['comparisons']['JOINT_vs_'+name];lo,hi=e['parent_and_neural_seed_sensitivity_CI95']
            ax.hlines(i,lo,hi,lw=2);ax.plot(e['delta_HOTA'],i,'o')
        ax.axvline(0,c='black',lw=.8);ax.set_yticks(range(len(names)),labels);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
        ax.set(title=pair,xlabel='JOINT minus control HOTA; fixed-output parent+seed95%')
    fig.suptitle('Graph then appearance regularization: independent effects and base-rate controls\nAll DEV; same paid QA; no additional teacher/encoder/head fits; no new cold-time claim',fontsize=10)
    folder=PUBLIC/'figures';folder.mkdir(exist_ok=True)
    preview=OUT/'analysis/publication_figure_previews';preview.mkdir(parents=True,exist_ok=True)
    fig.savefig(preview/'joint_visual.png',dpi=160,bbox_inches='tight')
    fig.savefig(folder/'joint_visual.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)


def plain_readout(repeated):
    lines=['先看最关键的结果与边界：']
    for pair in ('C4','C1'):
        row=repeated['JOINT_'+pair];values=row['mean_HOTA'];effects=row['comparisons']
        lines += [pair+' 主二乘二：普通KD '+f"{values['KD']:.6f}"+'；只加图 '+f"{values['GRAPH']:.6f}"+
            '；只加校准 '+f"{values['CAL']:.6f}"+'；图+校准 '+f"{values['JOINT']:.6f}"+'。']
        for key,label in [('JOINT_vs_KD','整套对普通蒸馏'),('JOINT_vs_GRAPH','在图之上校准的独立差'),
                ('JOINT_vs_CAL','在校准之上图的独立差'),('JOINT_vs_ANCHOR_CAL','视觉边相对无边教师锚点'),
                ('JOINT_vs_ROW_SHIFT','逐候选判断相对逐句统一偏移'),('JOINT_vs_QUERY_TEACHER','相对纯教师查询基率'),
                ('JOINT_vs_SHUFFLED','真实候选对应相对逐句打乱'),('JOINT_vs_GLOBAL_SHUFFLED','真实对应相对全局打乱'),
                ('EARLY_JOINT_vs_SEM400','较强400步KD上的整套增量'),('EARLY_JOINT_vs_EARLY_GRAPH','400步图上校准独立差')]:
            e=effects[key];lo,hi=e['parent_and_neural_seed_sensitivity_CI95']
            wording='平均差为负，不能据此主张该增量有效。' if e['delta_HOTA']<0 else ('平均差为正，描述性区间仍跨0。' if lo<=0 else '固定输出的开发描述性区间为正。')
            lines += [label+'：'+describe(e)+'；'+wording]
        for direction in ('OWN','BASELINE'):
            lines += [('固定JOINT数量' if direction=='OWN' else '固定KD数量')+'时相对KD排序差：'+describe(effects['COUNT_'+direction+'_JOINT_vs_KD'])+'。']
        baseline_names=['KD','KD400','KD_FINAL','HARD400','HARD_FINAL']+list(PCD)+list(HSV)+[n for n in SEM if n!='PREVIOUS_MONO']
        strongest=max(baseline_names,key=values.get)
        lines += ['全部预先保留的普通蒸馏控制中，本方向事后最高均值为'+strongest+' '+f'{values[strongest]:.6f}'+
            '；主方案对此的差：'+describe(effects['JOINT_vs_'+strongest])+'。这不是按方向换部署头。']
    lines += ['ROW_SHIFT保留逐句轨迹排序，互斥跨句分配仍可能改变；数量投影实际结果优先于直觉解释。',
        '若逐句/全局打乱也取得相近收益，就不能把整套改善全归为正确的逐候选视觉语义。',
        '数量或查询基率校准可成为有效贡献，但必须按真实机制命名，不称神经排序学习。',
        '所有区间基于已封存预测，不重训前缀流程，未修正多次开发；仍无独立盲测与新冷启动成本。','']
    return lines


def main():
    official=json.loads((OUT/'analysis/joint_visual_official.json').read_text());assert official['methods']==138
    data={};summary={};statistics={};verified=0;cache={};records=conditions()
    for record in records:
        pair,seed=record['pair'],record['seed'];slug=pair+f'_seed{seed}';root=Path(record['folder']);refs=external(record)
        raw=json.loads((root/'completion.json').read_text());counts=json.loads((root/'countmatched/completion.json').read_text())
        assert raw['GT_open_attempts']==raw['new_teacher_calls']==raw['new_head_fits']==0
        assert not any(counts['CountSelf_changed_query_files'].values())
        paths={r['method']:root/r['method'] for r in raw['methods']}
        paths.update({r['method']:root/'countmatched'/r['method'] for r in counts['methods']})
        paths.update({name:r[0] for name,r in refs.items()});data[slug]={name:read_stats(folder) for name,folder in paths.items()}
        for name,row in data[slug].items():
            numeric=dict(parents=row['parent_videos'],HOTA=row['HOTA'],numerator=row['parent_stats'][0].tolist(),denominator=row['parent_stats'][1].tolist())
            reference=refs[name][1] if name in refs else (refs[name.split('__AT_')[0]][1] if name in ('KD__AT_KD_COUNT','GRAPH__AT_GRAPH_COUNT') else None)
            if reference is not None:
                assert numeric==resolve_record(PUBLIC,reference['file'],reference['record'],cache),(slug,name,'reference numeric mismatch')
                statistics[slug+'/'+name]=dict(reference=reference);verified+=1
            else:statistics[slug+'/'+name]=numeric
        receipt=raw['receipts'];summary[slug]=dict(pair=pair,seed=seed,raw_HOTA={n:r['HOTA'] for n,r in data[slug].items() if '__AT_' not in n},
            comparisons={n:effect([data[slug][a]],[data[slug][b]]) for n,(a,b) in contrasts().items()},
            CountSelf_changed_query_files=counts['CountSelf_changed_query_files'],
            graph_replay_maximum_probability_error=max(r['graph_max_probability_error'] for r in receipt),
            previous_map_replay_maximum_probability_error=max(r['previous_map_max_probability_error'] for r in receipt),
            row_shift_changed_within_query_order=sum(r['row_shift_changed_within_query_order'] for r in receipt),
            CPU_components={key:sum(r[key] for r in receipt) for key in ('main_graph_cpu_seconds','main_map_cpu_seconds','main_apply_cpu_seconds','all_controls_postprocess_seconds')},
            logical_teacher_calls=raw['logical_teacher_calls'],new_teacher_calls=0,new_encoder_calls=0,new_head_fits=0)
    repeated={}
    for pair in ('C4','C1'):
        slugs=[pair+f'_seed{s}' for s in (0,17,42)]
        repeated['JOINT_'+pair]=dict(condition_keys=slugs,neural_seeds=[0,17,42],QA_seed=0,
            mean_HOTA={name:float(np.mean([summary[k]['raw_HOTA'][name] for k in slugs])) for name in summary[slugs[0]]['raw_HOTA']},
            comparisons={n:effect([data[k][a] for k in slugs],[data[k][b] for k in slugs]) for n,(a,b) in contrasts().items()})
    diagnostic=diagnosis(records)
    result=dict(status='JOINT_VISUAL_GRAPH_AND_APPEARANCE_ANALYZED',conditions=summary,repeated=repeated,query_diagnosis=diagnostic,
        primary='JOINT RGB1080 mean graph then real crossfit appearance regularizer',new_official_methods=138,
        public_baseline_references_verified=verified,new_teacher_calls=0,new_encoder_calls=0,new_head_fits=0,
        graph_and_previous_maps_numerically_reproduced=True,coarse_expansion_exact=True,
        no_target_manual_labels_in_adaptation=True,all_development_data=True,independent_blind_validation=False,
        row_shift_preserves_within_query_order_not_guaranteed_exclusive_joint_assignment=True,
        SHUFFLED_preserves_query_cue_marginals_GLOBAL_SHUFFLED_destroys_them=True,
        query_teacher_prior_uses_only_task_text_teacher_and_shared_availability_not_colour_compatibility_values=True,
        uncertainty='Fixed sealed prediction original-parent/shared-neural-seed bootstrap; no pipeline refit in resamples or multiple-development correction.',
        cost='Paid source/teacher reused; CPU components only, not matching cold single-method time.',goal_status='ACTIVE_REQUIRES_SCIENTIFIC_REVIEW')
    atomic_json(PUBLIC/'joint_visual_analysis.json',result);atomic_json(OUT/'analysis/joint_visual_analysis.json',result)
    compact,_=share_identical_statistics(statistics,'joint_visual_parent_statistics.json')
    (PUBLIC/'joint_visual_parent_statistics.json').write_text(json.dumps(compact,separators=(',',':'))+'\n')
    lines=['TTA-RMOT：视觉图与外观单调校准的联合归因','2026-11-22','',
        '研究范围：官方固定框和TrackID，目标图片/任务查询/冻结教师；适配不读目标人工标签。所有数据都是开发数据。',
        '主头为已有RGB1080软400/充分最终均值，没有新语义维度或新初始化。',
        'JOINT=已有视觉图之后施加固定PAVA有界正则，非独立贝叶斯证据；不继承失败的正信念保护或强制非空。',
        'KD、GRAPH、CAL、JOINT形成主二乘二消融；ANCHOR_CAL区分视觉边与仅恢复已问标签。',
        'ROW_SHIFT只给同一句所有轨迹相同logit偏移，实际检查逐句排序不变；互斥跨句决策仍可能改变。',
        'SHUFFLED保留每句线索分布，GLOBAL_SHUFFLED还破坏跨句分布；QUERY_TEACHER只用教师/查询基率，不读颜色兼容值。',
        '较强SEM400已有头另作GRAPH/CAL/JOINT稳健性控制，不按方向换主头。',
        '6条件60原始+78双向数量输出全部封存后138官评。0新增教师/编码器/头拟合，全部旧强基线保留。',
        '统计只重采样固定预测，不重训前缀流程；未校正开发迭代，不能当作独立新域验证。','']
    lines += plain_readout(repeated)
    for family,row in repeated.items():
        lines += [family,'HOTA均值：'+json.dumps(row['mean_HOTA'])]
        for name,e in row['comparisons'].items():lines.append(name+'：'+describe(e))
    lines += ['', '查询与候选IoU成员诊断（非HOTA可加分账）：',json.dumps(diagnostic,ensure_ascii=False,indent=2),
        '成本与数值复现收据：',json.dumps({k:{n:v[n] for n in ('CPU_components','graph_replay_maximum_probability_error','previous_map_replay_maximum_probability_error','row_shift_changed_within_query_order','CountSelf_changed_query_files')} for k,v in summary.items()},indent=2),
        '完成队列不代表用户科研Goal完成。若保留方法，仍需同配方真实单方法冷启动、成本账本和最终验收。']
    (ROOT/'docs/TTA_RMOT_视觉图与外观校准_联合归因_20261122.txt').write_text('\n'.join(lines)+'\n')
    figure(repeated)
    print(json.dumps(dict(event='JOINT_VISUAL_ANALYSIS_COMPLETE',mean_HOTA={k:v['mean_HOTA'] for k,v in repeated.items()},goal_status='ACTIVE')),flush=True)


if __name__=='__main__':main()
