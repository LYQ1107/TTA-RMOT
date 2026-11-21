"""Post-seal equal1080/RGB teacher baselines and actual-pool seed replication."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_rgb_parity_launch import conditions
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_sparse_observation_analyze import effect,fit_audit
from scripts.a100_visual_classical_sparse_observation_report import describe

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'


def contrasts(pair):
    full='RGB_GRAPH_MEAN';bases=['RGB_KD_MEAN','RGB_KD400','RGB_KD_FINAL','RGB_HARD400','RGB_HARD_FINAL','RGB_ANCHORS_MEAN']
    result={full+'_vs_'+base:(full,base) for base in bases}
    for name,base in [('RGB_GRAPH400','RGB_KD400'),('RGB_GRAPH_FINAL','RGB_KD_FINAL'),
        ('RGB_GRAPH_HARD400','RGB_HARD400'),('RGB_GRAPH_HARD_FINAL','RGB_HARD_FINAL')]:
        result[name+'_vs_'+base]=(name,base)
    others=['RGB_KD_MEAN']
    if pair=='C4':
        result.update({a+'_vs_'+b:(a,b) for a,b in [
            ('RGB_KD_MEAN','ORIG_KD_MEAN'),('ORIG_GRAPH_MEAN','ORIG_KD_MEAN'),
            ('ORIG_GRAPH_MEAN','RGB_KD_MEAN'),('ORIG_GRAPH_MEAN','RGB_KD400'),
            ('ORIG_GRAPH_MEAN','RGB_KD_FINAL'),('ORIG_GRAPH_MEAN','RGB_HARD400'),
            ('ORIG_GRAPH_MEAN','RGB_HARD_FINAL'),('RGB_GRAPH_MEAN','ORIG_GRAPH_MEAN')]})
        others.append('ORIG_GRAPH_MEAN')
    for other in others:
        result['COUNT_OWN_'+full+'_vs_'+other]=(full+'__AT_'+full+'_COUNT',other+'__AT_'+full+'_COUNT')
        result['COUNT_BASELINE_'+full+'_vs_'+other]=(full+'__AT_'+other+'_COUNT',other+'__AT_'+other+'_COUNT')
    return result


def figure(repeated):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,pair in zip(axes,('C4','C1')):
        row=repeated['STRICT_RGB_'+pair]
        bases=['RGB_KD_MEAN','RGB_KD400','RGB_KD_FINAL','RGB_HARD400','RGB_HARD_FINAL','RGB_ANCHORS_MEAN']
        for i,base in enumerate(bases):
            e=row['comparisons']['RGB_GRAPH_MEAN_vs_'+base];lo,hi=e['parent_and_neural_seed_sensitivity_CI95']
            ax.hlines(i,lo,hi,color='tab:blue',lw=2);ax.plot(e['delta_HOTA'],i,'o',color='tab:blue')
        ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(len(bases)),[b.removeprefix('RGB_').replace('_',' ').lower() for b in bases])
        ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
        ax.set(title=pair+': same1080D and RGB exposure',xlabel='Graph mean minus baseline HOTA; parent+seed95%')
    fig.suptitle('Actual fresh source/QA pools; same teachers and image information, three neural seeds\nDevelopment data; no target manual labels in adaptation; no blind or cold-time claim',fontsize=10)
    folder=PUBLIC/'figures';folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/'strict_rgb_parity.png',dpi=180,bbox_inches='tight')
    fig.savefig(folder/'strict_rgb_parity.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)


def main():
    official=json.loads((OUT/'analysis/strict_rgb_official.json').read_text())
    assert official['status']=='ALL_STRICT_RGB_OFFICIAL_COMPLETE' and official['methods']==114
    data={};summary={};sufficient={}
    for record in conditions():
        root=Path(record['folder']);pair=record['pair'];slug=pair+f"_seed{record['seed']}"
        raw=json.loads((root/'completion.json').read_text());counts=json.loads((root/'countmatched/completion.json').read_text())
        assert raw['GT_open_attempts']==raw['new_teacher_calls']==0
        assert raw['logical_teacher_calls']==(1160 if pair=='C4' else 1103)
        assert not any(counts['CountSelf_changed_query_files'].values())
        replay=dict(KD=sum(r['physical_seed0_KD_changed_rows'] for r in raw['receipts']),
            graph=sum(r['physical_seed0_graph_changed_rows'] for r in raw['receipts']))
        assert not any(replay.values())
        folders={r['method']:root/r['method'] for r in raw['methods']}
        folders.update({r['method']:root/'countmatched'/r['method'] for r in counts['methods']})
        data[slug]={name:read_stats(folder) for name,folder in folders.items()}
        for name,row in data[slug].items():
            sufficient[slug+'/'+name]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],numerator=row['parent_stats'][0].tolist(),denominator=row['parent_stats'][1].tolist())
        audit=fit_audit(raw['receipts'])
        assert not any(row['videos_exceeding_registered_KL_tolerance'] for row in audit.values())
        summary[slug]=dict(pair=pair,seed=record['seed'],logical_teacher_calls=raw['logical_teacher_calls'],source_pool=raw['source_pool'],
            raw_HOTA={name:row['HOTA'] for name,row in data[slug].items() if '__AT_' not in name},
            comparisons={name:effect([data[slug][a]],[data[slug][b]]) for name,(a,b) in contrasts(pair).items()},
            teacher_fit_audit=audit,physical_seed0_changed_rows=replay,CountSelf_changed_query_files=counts['CountSelf_changed_query_files'],
            head_fit_seconds={label:sum(r['fits'][label].get('wall_seconds',0) for r in raw['receipts']) for label in raw['receipts'][0]['fits']},
            not_cold_deployment_time=True)
    repeated={}
    for pair in ('C4','C1'):
        slugs=[pair+f'_seed{s}' for s in (0,17,42)]
        repeated['STRICT_RGB_'+pair]=dict(condition_keys=slugs,neural_seeds=[0,17,42],QA_seed=0,
            mean_HOTA={name:float(np.mean([summary[k]['raw_HOTA'][name] for k in slugs])) for name in summary[slugs[0]]['raw_HOTA']},
            comparisons={name:effect([data[k][a] for k in slugs],[data[k][b] for k in slugs]) for name,(a,b) in contrasts(pair).items()})
    checks={pair:{name:e['parent_and_neural_seed_sensitivity_CI95'][0]>0 for name,e in repeated['STRICT_RGB_'+pair]['comparisons'].items()
        if name.startswith('RGB_GRAPH_MEAN_vs_')} for pair in ('C4','C1')}
    result=dict(status='STRICT_RGB1080_PARITY_ANALYZED',conditions=summary,repeated=repeated,
        descriptive_positive_CI_checks=checks,official_evaluated_methods=114,
        target_GT_used_only_after_all114_predictions_sealed=True,no_target_manual_labels_in_adaptation=True,
        no_new_teacher_or_encoder_calls=True,all_development_data=True,independent_blind_validation=False,
        same_feature_dimension_and_initialization=True,RGB_information_parity_for_primary_contrast=True,
        older_original_head_candidate_has_different_RGB_exposure=True,
        uncertainty='10000 original-parent resamples plus three neural-seed sensitivity; fixed actual QA; not corrected for iteration or multiple contrasts.',
        cost='Fresh source/teacher were physically measured elsewhere. These head refits reuse paid data; no new cold timing claim and no3seed mean quality attached to seed0 wall.',
        goal_status='ACTIVE_REQUIRES_SCIENTIFIC_ACCEPTANCE')
    atomic_json(PUBLIC/'strict_rgb_analysis.json',result);atomic_json(OUT/'analysis/strict_rgb_analysis.json',result)
    (PUBLIC/'strict_rgb_parent_statistics.json').write_text(json.dumps(sufficient,separators=(',',':'))+'\n')
    lines=['TTA-RMOT：相同1080维、相同RGB信息、真实新鲜源流的严格教师对照','2026-11-22','',
        '本轮解决一处公平性混杂：原扩展图获得空间/角色RGB，原蒸馏头却遮掉这些RGB；此前1088维控制又同时引入位置和初始化变化。',
        '这次RGB普通KD和RGB图共同获得相同1080维PCD/RGB、源先验、当前/历史教师问答及初始化，只有经典图推断不同。',
        '普通软KD保留400步、充分训练最终、两者均值；硬KD也充分拟合，所有失败和更强基线均报告。',
        'C4使用新独立KD物理流的1160次真实新问题；C1使用新物理流1103次。与历史开发池不同，不拼接旧质量和新时间。',
        '三个神经种子共享同一实际问答，不能冒充三次独立教师采样。当前视频是转导适配，历史/当前可用、未来不可用。',
        '源主干/教师冻结，官方候选框和ID不改变。目标人工标签只在114组原始/数量预测封存之后使用。',
        'seed0必须完整复现物理KD决策，C1还复现物理图；数量反事实CountSelf为0。',
        '所有数据都参与过开发，不称独立盲测；区间是描述性不确定性，不证明未见域泛化。','']
    for family,row in repeated.items():
        lines += [family,'三种子HOTA：'+json.dumps(row['mean_HOTA'])]
        lines += ['  '+name+'：'+describe(e) for name,e in row['comparisons'].items()]
        primary=row['comparisons']['RGB_GRAPH_MEAN_vs_RGB_KD_MEAN']
        lines.append('判读：'+('主对照在本开发数据的原视频+种子区间为正。' if primary['parent_and_neural_seed_sensitivity_CI95'][0]>0 else '主对照未得到区间严格为正的证据，不能宣布稳定优于普通教师KD。'))
        lines.append('对各强基线/锚点的区间检查：'+json.dumps(checks[family.rsplit('_',1)[1]]))
        lines.append('')
    lines += ['训练与回归收据：',json.dumps(summary,ensure_ascii=False,indent=2),
        '本轮复用已付费输入，新拟合秒数单独记录，不能当作独立冷启动秒数。',
        '若相同RGB普通KD消除了旧候选优势，应修正旧归因并继续研究；本脚本不自动完成Goal。']
    (ROOT/'docs/TTA_RMOT_同维度同RGB_严格教师对照_20261122.txt').write_text('\n'.join(lines)+'\n')
    figure(repeated)
    print(json.dumps(dict(event='STRICT_RGB_ANALYSIS_COMPLETE',repeated=repeated,goal_status='ACTIVE')),flush=True)


if __name__=='__main__':main()
