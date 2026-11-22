"""Analyze sealed native-head mechanisms and reproduce all public effects."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_native_factorial_launch import conditions
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_sparse_observation_analyze import effect
from scripts.a100_visual_classical_sparse_observation_report import describe
from scripts.a100_visual_classical_sparse_statistics_reproduce import resolve_record,share_identical_statistics,methods

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'


def contrasts():
    result={'JOINT_vs_'+name:('JOINT',name) for name in ('KD','GRAPH','CAL','ANCHOR_CAL','ROW_SHIFT','SHUFFLED','GLOBAL_SHUFFLED','QUERY_TEACHER','COMPLETE')}
    result.update(GRAPH_vs_KD=('GRAPH','KD'),CAL_vs_KD=('CAL','KD'))
    for name in ('KD','GRAPH','CAL','ANCHOR_CAL','ROW_SHIFT','COMPLETE'):
        result['COUNT_OWN_JOINT_vs_'+name]=('JOINT__AT_JOINT_COUNT',name+'__AT_JOINT_COUNT')
        result['COUNT_BASELINE_JOINT_vs_'+name]=('JOINT__AT_'+name+'_COUNT',name+'__AT_'+name+'_COUNT')
    assert all(tuple(methods(label))==value for label,value in result.items())
    return result


def figure(repeated):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    groups=[('Module increments',('GRAPH_vs_KD','CAL_vs_KD','JOINT_vs_KD')),
        ('Mechanism specificity',('JOINT_vs_ANCHOR_CAL','JOINT_vs_COMPLETE','JOINT_vs_GLOBAL_SHUFFLED'))]
    for column,pair in enumerate(('C4','C1')):
        row=repeated['NATIVE_'+pair]
        for line,(title,labels) in enumerate(groups):
            ax=axes[line,column]
            for index,label in enumerate(labels):
                effect_=row['comparisons'][label];lo,hi=effect_['parent_and_neural_seed_sensitivity_CI95'];point=effect_['delta_HOTA']
                ax.plot([lo,hi],[index,index],color='tab:blue',lw=2);ax.scatter([point],[index],color='tab:orange',s=30,zorder=3)
            ax.set_yticks(range(len(labels)),[label.replace('_vs_',' vs ').replace('JOINT','J') for label in labels],fontsize=8)
            ax.axvline(0,color='.4',lw=1);ax.set(title=pair+': '+title,xlabel='Delta HOTA');ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.suptitle('Native768 direct teacher head; fixed predictions, 3 neural seeds, all DEV\nOriginal-parent + shared-seed descriptive intervals; no pipeline refit or repeated-development correction',fontsize=10)
    dest=PUBLIC/'figures';dest.mkdir(exist_ok=True);preview=OUT/'analysis/publication_figure_previews';preview.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest/'native_factorial.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(preview/'native_factorial.png',dpi=160,bbox_inches='tight');plt.close(fig)


def main():
    official=json.loads((OUT/'analysis/native_factorial_official.json').read_text());assert official['methods']==132
    data={};summary={};statistics={};cache={};comparisons=contrasts()
    for record in conditions():
        pair,seed=record['pair'],record['seed'];slug=pair+f'_seed{seed}';root=Path(record['folder'])
        raw=json.loads((root/'completion.json').read_text());counts=json.loads((root/'countmatched/completion.json').read_text())
        assert raw['GT_open_attempts']==raw['new_teacher_calls']==raw['new_encoder_calls']==raw['new_head_fits']==0
        assert raw['logical_teacher_calls']==(1160 if pair=='C4' else 1103) and not any(counts['CountSelf_changed_query_files'].values())
        paths={m['method']:root/m['method'] for m in raw['methods']};paths.update({m['method']:root/'countmatched'/m['method'] for m in counts['methods']})
        data[slug]={name:read_stats(folder) for name,folder in paths.items()}
        for name,row in data[slug].items():statistics[slug+'/'+name]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],
            numerator=row['parent_stats'][0].tolist(),denominator=row['parent_stats'][1].tolist())
        references=dict(KD='KD_MEAN',GRAPH='GRAPH',CAL='CAL',JOINT='JOINT',
            KD__AT_JOINT_COUNT='KD_MEAN__AT_JOINT_COUNT',JOINT__AT_KD_COUNT='JOINT__AT_KD_MEAN_COUNT',KD__AT_KD_COUNT='KD_MEAN__AT_KD_MEAN_COUNT')
        for name,original in references.items():
            reference=dict(file='direct_teacher_pcd_parent_statistics.json',record=slug+'/'+original)
            row=resolve_record(PUBLIC,reference['file'],reference['record'],cache)
            data[slug][name]=dict(parent_videos=row['parents'],HOTA=row['HOTA'],parent_stats=(np.asarray(row['numerator']),np.asarray(row['denominator'])))
            statistics[slug+'/'+name]=dict(reference=reference)
        summary[slug]=dict(pair=pair,seed=seed,
            raw_HOTA={name:row['HOTA'] for name,row in data[slug].items() if '__AT_' not in name},
            comparisons={name:effect([data[slug][a]],[data[slug][b]]) for name,(a,b) in comparisons.items()},
            CountSelf_changed_query_files=counts['CountSelf_changed_query_files'],
            frozen_prediction_max_error=max(max(r['frozen_prediction_max_errors'].values()) for r in raw['receipts']),
            complete_total_edge_weight_max_error=max(r['complete_graph_total_edge_weight_error'] for r in raw['receipts']),
            row_shift_changed_within_query_orders=sum(r['row_shift_changed_within_query_order'] for r in raw['receipts']))
    repeated={}
    for pair in ('C4','C1'):
        slugs=[pair+f'_seed{s}' for s in (0,17,42)]
        repeated['NATIVE_'+pair]=dict(condition_keys=slugs,neural_seeds=[0,17,42],QA_seed=0,
            mean_HOTA={name:float(np.mean([summary[k]['raw_HOTA'][name] for k in slugs])) for name in summary[slugs[0]]['raw_HOTA']},
            comparisons={name:effect([data[k][a] for k in slugs],[data[k][b] for k in slugs]) for name,(a,b) in comparisons.items()})
    result=dict(status='NATIVE_DIRECT_FACTORIAL_ANALYZED',conditions=summary,repeated=repeated,new_raw_outputs=36,new_count_outputs=96,
        official_new_methods=132,new_teacher_calls=0,new_encoder_calls=0,new_head_fits=0,
        source='Same sealed native768D direct softmean, no source-logit residual; richRGB/SEM/residual strong baselines remain in direct_teacher_pcd_analysis.',
        no_target_manual_GT_in_adaptation=True,all_development_data=True,independent_blind_validation=False,
        controls='Complete graph matches total edge weight, not individual degrees; rowshift preserves withinqueryorder but exclusive crossqueryassignment may change.',
        uncertainty='Fixed sealed original-parent/shared-neural-seed descriptive bootstrap; no prefix refit or development/multiplicity correction.',
        goal_status='ACTIVE_REQUIRES_MODULE_AND_NATIVE_ACTUAL_COLD_REVIEW')
    atomic_json(PUBLIC/'native_factorial_analysis.json',result);atomic_json(OUT/'analysis/native_factorial_analysis.json',result)
    compact,_=share_identical_statistics(statistics,'native_factorial_parent_statistics.json')
    (PUBLIC/'native_factorial_parent_statistics.json').write_text(json.dumps(compact,separators=(',',':'))+'\n')
    lines=['TTA-RMOT：原生直接教师头的经典视觉机制拆解','2026-11-22','',
        '同一768D直接软均值头，固定源logit为0；所有NN/输入/问答均已封存，0新增教师/编码器调用或小头拟合。',
        '6条件36新原始+96数量共132输出全部封存后才本研究GT；三神经种子、C4原13视频/C1原25视频。',
        '图/PAVA/KD/JOINT逐视频概率与掩码必须复现原生直接头已封存结果0差；没有按方向换头。',
        'COMPLETE只有同总边权，不保持各节点度。若它与真实视觉图打平，不把特定3NN拓扑说成必不可少。',
        'ROW_SHIFT只保护每句内部排序；互斥解码时跨句胜者仍可能变化，所以它不是数学意义上的纯数量对照。','']
    for pair,row in repeated.items():
        lines += [pair,'三种子平均HOTA：'+json.dumps(row['mean_HOTA'])]
        for name,e in row['comparisons'].items():lines.append(name+'：'+describe(e))
    audit_fields=('raw_HOTA','CountSelf_changed_query_files','frozen_prediction_max_error','complete_total_edge_weight_max_error','row_shift_changed_within_query_orders')
    lines += ['', '各神经种子原始分数、输入重放及CountSelf：',json.dumps({k:{n:r[n] for n in audit_fields} for k,r in summary.items()},ensure_ascii=False,indent=2),
        '全部逐种子差值与区间仍完整保留在native_factorial_analysis.json，不在TXT中重复相同的机器字段。',
        '数量收益是合法任务收益，但不等于逐候选语义排序；跨0区间不能改写成稳定正贡献。',
        '本研究复用已付费特征/问答，不提供新冷启动时间。真实四卡另见native_cold_analysis.json；不能贴旧精度/时间。',
        '所有数据为开发数据，没有独立盲测；公开统计的算术复现不等于独立泛化验证，Goal不在脚本中自动关闭。']
    (ROOT/'docs/TTA_RMOT_原生直接头_视觉机制与数量剥离_20261122.txt').write_text('\n'.join(lines)+'\n')
    figure(repeated)
    print(json.dumps(dict(event='NATIVE_FACTORIAL_ANALYSIS_COMPLETE',repeated=repeated,goal_status='ACTIVE')),flush=True)


if __name__=='__main__':main()
