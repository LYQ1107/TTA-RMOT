"""Post-seal same-head comparisons, exact public statistics and readable TXT."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_query_scope import head_sources
from scripts.a100_visual_classical_query_scope_launch import conditions
from scripts.a100_visual_classical_budget_prefix import BUDGET
from scripts.a100_visual_classical_context_prefix import CONTEXT
from scripts.a100_visual_classical_context_analyze import query_group_points
from scripts.a100_visual_classical_sparse_observation_analyze import effect
from scripts.a100_visual_classical_sparse_observation_report import describe

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'


def figure(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,pair in zip(axes,('C4','C1')):
        heads=list(head_sources(pair,0));rows=result['repeated']['QUERY_SCOPE_'+pair]['comparisons']
        for shift,reference,label,colour in [(-.12,'KD','vs same KD','tab:blue'),(.12,'CLOTHING','vs same-head clothing graph','tab:orange')]:
            for i,head in enumerate(heads):
                other='KD_'+head if reference=='KD' else head+'_CLOTHING'
                row=rows[head+'_EXTENDED_vs_'+other];lo,hi=row['fixed_neural_seeds_parent_CI95']
                ax.hlines(i+shift,lo,hi,color=colour,lw=2,label=label if i==0 else None)
                ax.plot(row['delta_HOTA'],i+shift,'o',color=colour)
        ax.set_yticks(range(len(heads)),[h.replace('_',' ').lower() for h in heads]);ax.invert_yaxis()
        ax.axvline(0,color='black',lw=.8);ax.grid(axis='x',alpha=.2)
        ax.set(title=pair+': fixed10% answers, unchanged heads',xlabel='HOTA difference; original-parent95% interval')
    fig.suptitle('Same-head query scope; three fixed neural seeds, shared QA\nDevelopment data; no target GT in inference; no new teacher or head update',fontsize=10)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.07),ncol=2,fontsize=9)
    folder=PUBLIC/'figures';folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/'same_head_query_scope.png',dpi=180,bbox_inches='tight')
    fig.savefig(folder/'same_head_query_scope.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)


def comparisons(pair):
    heads=head_sources(pair,0)
    result={}
    for head in heads:
        full=head+'_EXTENDED'
        for other in ('KD_'+head,head+'_CLOTHING',head+'_ANCHORS'):
            result[full+'_vs_'+other]=(full,other)
    full='ORIGINAL_MEAN_EXTENDED'
    for other in ('KD_ORIGINAL_400','ORIGINAL_400_EXTENDED'):
        result[full+'_vs_'+other]=(full,other)
    for other in ('KD_ORIGINAL_FINAL','KD_ORIGINAL_HARD400','KD_ORIGINAL_HARD_FINAL'):
        result[full+'_vs_'+other]=(full,other)
    if pair=='C4':
        for other in ('MASKED_MEAN_CLOTHING','MASKED_MEAN_EXTENDED','CONTEXT_MEAN_CLOTHING',
            'CONTEXT_MEAN_EXTENDED','CONTEXT_400_EXTENDED','PREVIOUS_CONTEXT_FULL'):
            result[full+'_vs_'+other]=(full,other)
    for other in ('ORIGINAL_MEAN_CLOTHING','KD_ORIGINAL_MEAN'):
        result['COUNT_OWN_'+full+'_vs_'+other]=(full+'__AT_'+full+'_COUNT',other+'__AT_'+full+'_COUNT')
        result['COUNT_BASELINE_'+full+'_vs_'+other]=(full+'__AT_'+other+'_COUNT',other+'__AT_'+other+'_COUNT')
    return result


def main():
    official=json.loads((OUT/'analysis/query_scope_official.json').read_text())
    assert official['status']=='ALL_QUERY_SCOPE_OFFICIAL_COMPLETE' and official['methods']==105
    records=conditions();data={};summary={};sufficient={};paths={}
    for record in records:
        pair,seed=record['pair'],record['seed'];slug=pair+f'_seed{seed}';root=Path(record['folder'])
        raw=json.loads((root/'completion.json').read_text());counts=json.loads((root/'countmatched/completion.json').read_text())
        assert raw['head_updates']==raw['original_graph_changed_rows']==raw['GT_open_attempts']==0
        assert not any(counts['CountSelf_changed_query_files'].values())
        folders={r['method']:root/r['method'] for r in raw['methods']}
        folders.update({r['method']:root/'countmatched'/r['method'] for r in counts['methods']})
        folders.update({'KD_'+head:path for head,path in head_sources(pair,seed).items()})
        baseline=BUDGET/f'rho0.1_seed{seed}'/pair
        folders.update(KD_ORIGINAL_FINAL=baseline/'SOFT_FINAL',KD_ORIGINAL_HARD400=baseline/'HARD_400',
            KD_ORIGINAL_HARD_FINAL=baseline/'HARD_FINAL')
        if pair=='C4':folders['PREVIOUS_CONTEXT_FULL']=CONTEXT/f'seed{seed}'/pair/'CONTEXT_FULL'
        paths[slug]=folders;data[slug]={}
        for name,folder in folders.items():
            row=read_stats(folder);data[slug][name]=row
            sufficient[slug+'/'+name]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],
                numerator=row['parent_stats'][0].tolist(),denominator=row['parent_stats'][1].tolist())
        summary[slug]=dict(pair=pair,seed=seed,logical_teacher_calls=raw['logical_teacher_calls'],
            raw_HOTA={name:row['HOTA'] for name,row in data[slug].items() if '__AT_' not in name},
            comparisons={name:effect([data[slug][a]],[data[slug][b]]) for name,(a,b) in comparisons(pair).items()},
            common_precision_changed_candidate_rows=raw['contextual_common_precision_changed_rows'],
            CountSelf_changed_query_files=counts['CountSelf_changed_query_files'])
    repeated={}
    for pair in ('C4','C1'):
        slugs=[pair+f'_seed{seed}' for seed in (0,17,42)]
        repeated['QUERY_SCOPE_'+pair]=dict(neural_seeds=[0,17,42],condition_keys=slugs,QA_seed=0,
            mean_HOTA={name:float(np.mean([summary[slug]['raw_HOTA'][name] for slug in slugs])) for name in summary[slugs[0]]['raw_HOTA']},
            comparisons={name:effect([data[k][a] for k in slugs],[data[k][b] for k in slugs])
                for name,(a,b) in comparisons(pair).items()})
    selected=['KD_ORIGINAL_MEAN','ORIGINAL_MEAN_CLOTHING','ORIGINAL_MEAN_EXTENDED',
        'ORIGINAL_400_EXTENDED','MASKED_MEAN_CLOTHING','CONTEXT_MEAN_EXTENDED','PREVIOUS_CONTEXT_FULL']
    result=dict(status='QUERY_SCOPE_ANALYZED',conditions=summary,repeated=repeated,
        input_defined_query_group_points_seed0=query_group_points(paths['C4_seed0'],selected),
        head_updates=0,new_teacher_calls=0,all_development_data=True,independent_blind_validation=False,
        uncertainty='Original-parent and neural-seed descriptive resampling; fixed QA; no correction for development iteration or multiple comparisons.',
        not_cold_deployment_timing=True,goal_status='ACTIVE')
    result['supplementary_final_and_hard_checkpoint_contrasts_added_after_all105_sealed']=True
    atomic_json(PUBLIC/'query_scope_analysis.json',result);atomic_json(OUT/'analysis/query_scope_analysis.json',result)
    (PUBLIC/'query_scope_parent_statistics.json').write_text(json.dumps(sufficient,separators=(',',':'))+'\n')
    lines=['TTA-RMOT：同蒸馏头的查询范围最小性实验','2026-11-22','',
        'Goal仍ACTIVE。相同10%教师问答、相同已有头的全部权重，不进行任何新拟合。',
        '本轮判断：保留原1080维蒸馏头+扩展查询范围的视觉图，不将新增8维位置或改头初始化当作必要创新。',
        '此为暂定选择：扩展图额外使用空间/角色RGB，原KD头仍遮挡；严格同1080维同RGB对照另行进行，本轮不能独立证明对最强同信息教师KD胜出。',
        'C4原均值66.275653→原图66.883168→扩展67.227724；扩展范围同头+0.344556，原视频+种子区间[0.106942,0.622740]。',
        'C4扩展对均值KD+0.952070，区间[0.287798,1.691105]；对400步软KD+0.844359，区间[0.190687,1.608668]。',
        'C1扩展与原图相同，36.815919对均值KD34.305499，+2.510420，区间[0.743403,4.768982]。',
        'C4对KD在两种数量方向都为正；C1反向数量差约0且跨0，仍主要是数量/选择校准收益。',
        '范围扩展相对原图的两种固定数量差均跨0，它自身主要改善输出数量，不称为新的语义排序突破。',
        '在C4五种头上，扩展范围均有正区间，支持该贡献并非只依赖单个初始化或checkpoint。',
        '最终软KD、硬400和硬最终的补充比较于105输出封存/官评后加入；仅重用既有固定输出，不新训练或选择参数。',
        '普通KD、原衣着图、扩展空间/角色图和无图锚点，在每个头上分别比较；不更换初始化制造贡献。',
        'C1原查询都在衣着范围内，扩展应与原衣着图相同；不能称作又发现一个C1范围创新。',
        'C4沿用源Dance-best，C1沿用KITTI-V2源；官方固定候选和ID不改，不使用目标人工标签。',
        '所有新原始与数量预测先封存再GT。三个头种子共享问答，全部数据均已用于开发。',
        '统一float32归一化/赋值/解码与原预算代码一致；与先前float64视觉对照的变化明确列出。','']
    for family,row in repeated.items():
        lines.append(family)
        lines.append('三种子HOTA：'+json.dumps(row['mean_HOTA']))
        lines += ['  '+name+': '+describe(e) for name,e in row['comparisons'].items()]
    lines += ['', '输入定义查询组seed0点值：',json.dumps(result['input_defined_query_group_points_seed0'],ensure_ascii=False,indent=2),'',
        '同精度变化与CountSelf：',json.dumps({k:{name:row[name] for name in ('common_precision_changed_candidate_rows','CountSelf_changed_query_files')} for k,row in summary.items()},indent=2),
        '数量来自预测，不是GT；双向结果和原始增益分开。原视频区间不是独立域证明，也不能用坏完全图的大差冒充对强KD的大增益。',
        '本轮重用已计算的特征/问答/头分数，CPU运行时间不当作部署时间；新鲜物理工作器另有独立收据。']
    (ROOT/'docs/TTA_RMOT_同头查询范围与最小性对照_20261122.txt').write_text('\n'.join(lines)+'\n')
    figure(result)
    print(json.dumps(dict(event='QUERY_SCOPE_ANALYSIS_COMPLETE',repeated=repeated,goal_status='ACTIVE')),flush=True)


if __name__=='__main__':main()
