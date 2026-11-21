"""Post-seal attribution of explicit HSV channels, not a weaker graph baseline."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_zero_colour_launch import conditions
from scripts.a100_visual_classical_rgb_parity import PARITY
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_sparse_observation_analyze import effect,fit_audit
from scripts.a100_visual_classical_sparse_observation_report import describe

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'


def contrasts():
    result={a+'_vs_'+b:(a,b) for a,b in [('RGB_MEAN','ZERO_MEAN'),('RGB_400','ZERO_400'),
        ('RGB_FINAL','ZERO_FINAL'),('RGB_GRAPH_MEAN','RGB_MEAN'),('RGB_GRAPH_MEAN','ZERO_MEAN')]}
    result['COUNT_OWN_RGB_MEAN_vs_ZERO_MEAN']=('RGB_MEAN__AT_RGB_MEAN_COUNT','ZERO_MEAN__AT_RGB_MEAN_COUNT')
    result['COUNT_BASELINE_RGB_MEAN_vs_ZERO_MEAN']=('RGB_MEAN__AT_ZERO_MEAN_COUNT','ZERO_MEAN__AT_ZERO_MEAN_COUNT')
    return result


def figure(repeated):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),constrained_layout=True)
    for ax,pair in zip(axes,('C4','C1')):
        row=repeated['ZERO_COLOUR_'+pair]
        names=['RGB_400_vs_ZERO_400','RGB_FINAL_vs_ZERO_FINAL','RGB_MEAN_vs_ZERO_MEAN']
        for i,name in enumerate(names):
            e=row['comparisons'][name];lo,hi=e['parent_and_neural_seed_sensitivity_CI95']
            ax.hlines(i,lo,hi,color='tab:blue',lw=2);ax.plot(e['delta_HOTA'],i,'o',color='tab:blue')
        ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(3),['400 steps','final fit','mean400/final']);ax.invert_yaxis()
        ax.set(title=pair+': explicit HSV channels in matched1080D head',xlabel='RGB minus zero-HSV HOTA; parent+seed95%');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Same frozen PCD, initialization, source/QA and soft loss; no additional teacher\nFeature attribution only: strong full-RGB KD remains the graph comparison; all development data',fontsize=10)
    dest=PUBLIC/'figures';dest.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest/'explicit_colour_head_ablation.png',dpi=180,bbox_inches='tight')
    fig.savefig(dest/'explicit_colour_head_ablation.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)


def main():
    official=json.loads((OUT/'analysis/zero_colour_official.json').read_text());assert official['methods']==42
    while True:
        state=json.loads((OUT/'supervisor/strict_rgb_state.json').read_text())
        if state['status'].startswith('FAILED_'):raise RuntimeError('RGB reference official study failed; do not read unfinished reference tables.')
        if state['status']=='PASS_STRICT_RGB_PARITY_RESEARCH_GOAL_ACTIVE':break
        time.sleep(15)
    data={};summary={};sufficient={}
    for record in conditions():
        pair,seed=record['pair'],record['seed'];slug=pair+f'_seed{seed}';root=Path(record['folder'])
        raw=json.loads((root/'completion.json').read_text());counts=json.loads((root/'countmatched/completion.json').read_text())
        assert raw['GT_open_attempts']==raw['new_teacher_calls']==0 and not any(counts['CountSelf_changed_query_files'].values())
        assert raw['logical_teacher_calls']==(1160 if pair=='C4' else 1103)
        folders={r['method']:root/r['method'] for r in raw['methods']}
        folders.update({r['method']:root/'countmatched'/r['method'] for r in counts['methods']})
        ref=PARITY/pair/f'seed{seed}'
        folders.update(RGB_400=ref/'RGB_KD400',RGB_FINAL=ref/'RGB_KD_FINAL',RGB_MEAN=ref/'RGB_KD_MEAN',RGB_GRAPH_MEAN=ref/'RGB_GRAPH_MEAN')
        data[slug]={n:read_stats(p) for n,p in folders.items()}
        for n,row in data[slug].items():
            sufficient[slug+'/'+n]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],numerator=row['parent_stats'][0].tolist(),denominator=row['parent_stats'][1].tolist())
        summary[slug]=dict(pair=pair,seed=seed,raw_HOTA={n:r['HOTA'] for n,r in data[slug].items() if '__AT_' not in n},
            comparisons={n:effect([data[slug][a]],[data[slug][b]]) for n,(a,b) in contrasts().items()},
            physical_GPU=raw['physical_GPU'],teacher_fit_audit=fit_audit(raw['receipts']),
            CountSelf_changed_query_files=counts['CountSelf_changed_query_files'],
            actual_head_fit_seconds=sum(r['fits']['ZERO']['wall_seconds'] for r in raw['receipts']))
    repeated={}
    for pair in ('C4','C1'):
        slugs=[pair+f'_seed{s}' for s in (0,17,42)]
        repeated['ZERO_COLOUR_'+pair]=dict(condition_keys=slugs,neural_seeds=[0,17,42],QA_seed=0,
            mean_HOTA={n:float(np.mean([summary[k]['raw_HOTA'][n] for k in slugs])) for n in summary[slugs[0]]['raw_HOTA']},
            comparisons={n:effect([data[k][a] for k in slugs],[data[k][b] for k in slugs]) for n,(a,b) in contrasts().items()})
    result=dict(status='ZERO_COLOUR_HEAD_ABLATION_ANALYZED',conditions=summary,repeated=repeated,new_official_methods=42,
        identical1080D_architecture_and_initialization=True,no_extra_encoder_or_teacher_calls=True,
        classical_feature_scope='Only explicit312HSV channels are zeroed. Frozen neural PCD still encodes colour, and the common paid acquisition may use RGB. Not a grayscale or end-to-end colour-free baseline.',
        does_not_replace_strong_full_RGB_baselines=True,matched_soft_loss_feature_effect_only=True,
        not_natural768D_architecture_baseline=True,target_GT_only_postseal=True,all_development_data=True,independent_blind_validation=False,
        uncertainty='Original-parent plus neural-seed descriptive resampling; fixed actual QA; no adjustment for development or multiple contrasts.',
        cost='Teacher/features prepaid; measured head refits are not new cold processes. Any max-step fit failures remain visible.',goal_status='ACTIVE_REQUIRES_COMBINED_ACCEPTANCE')
    atomic_json(PUBLIC/'zero_colour_analysis.json',result);atomic_json(OUT/'analysis/zero_colour_analysis.json',result)
    (PUBLIC/'zero_colour_parent_statistics.json').write_text(json.dumps(sufficient,separators=(',',':'))+'\n')
    lines=['TTA-RMOT：手工部位颜色通道的同维度贡献拆解','2026-11-22','',
        '本实验不降低主比较的强KD：全RGB软/硬充分训练基线完整保留。',
        '高分KD本就使用我们加入的经典HSV部位颜色。本轮仅回答这个已有模块对软KD有多少贡献。',
        '1080维和初始化保持，只把312显式HSV通道换成零；固定768源PCD、实际Qwen问答、归一化协议及训练规则。',
        'PCD依然由彩色图像产生，选题仍是原共同RGB策略。不能把它叫灰度模型或整个系统完全不用颜色。',
        '不是独立自然768维网络基线，避免再次混入维度/RNG差异；主图优劣仍由相同RGB信息对照判断。',
        '六条件、18原始+24数量先封存再42官评；种子0/17/42共享实际问答。',
        '0额外教师/源前向，只有小头实际重拟合；不足教师KL的max-step状态不丢弃，训练收据如实报告。',
        '全部参与过开发；不称独立盲测或普遍跨域定律。','']
    for family,row in repeated.items():
        lines += [family,'三种子HOTA：'+json.dumps(row['mean_HOTA'])]
        lines += [n+'：'+describe(e) for n,e in row['comparisons'].items()]
    lines += ['', '拟合与数量自回归：',json.dumps({k:{n:r[n] for n in ('teacher_fit_audit','CountSelf_changed_query_files','actual_head_fit_seconds')} for k,r in summary.items()},indent=2),
        'RGB_GRAPH对ZERO的差同时包含特征与图，不能全部记成图贡献。RGB_GRAPH对RGB_MEAN才是同信息图增量。',
        '只有实际结果支持时才将显式颜色通道称为有效模块；负结果同样保留。']
    (ROOT/'docs/TTA_RMOT_手工颜色通道_同维度贡献拆解_20261122.txt').write_text('\n'.join(lines)+'\n')
    figure(repeated)
    print(json.dumps(dict(event='ZERO_COLOUR_ANALYSIS_COMPLETE',repeated=repeated,goal_status='ACTIVE')),flush=True)


if __name__=='__main__':main()
