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
from scripts.a100_visual_classical_rgb_parity import PARITY,source_spec
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


def teacher_validation(records):
    from flex_uda.a100.mechanism import metadata,PAIRS
    from flex_uda.active.semantic_colour import crossfit_predictions,append_history,shuffled_cues
    cache=OUT/'analysis/semantic_teacher_prediction_audit.json'
    if cache.is_file():return json.loads(cache.read_text())
    result={}
    for record in records:
        if record['seed']!=0:continue
        pair=record['pair'];root=Path(record['folder']);source=source_spec(pair)[0]
        m=metadata(PAIRS[pair][1]);history={f:dict(x=np.empty(0),y=np.empty(0)) for f in (0,1)}
        shuffled_history={f:dict(x=np.empty(0),y=np.empty(0)) for f in (0,1)};count=0;loss={name:0. for name in ('PAVA','BIN','SHUFFLED','CONSTANT')};error=0.
        for video in sorted(m['videos']):
            with np.load(root/'videos'/video/'coarse_scores.npz') as a:
                cue,available,families,tracks,only=a['cue'],a['available'],a['families'],a['tracks'],a['MONO_ONLY']
            teacher=np.full(cue.shape,np.nan)
            for answer in json.loads((source/'source'/video/'answers.json').read_text())['answers']:
                teacher[int(answer['local_query']),int(answer['local_track'])]=answer['yes_probability']
            p,prior,active,_=crossfit_predictions(cue,available,families,teacher,tracks,video,history)
            b,_,ba,_=crossfit_predictions(cue,available,families,teacher,tracks,video,history,monotone=False)
            shuffled=shuffled_cues(cue,available,video);s,_,sa,_=crossfit_predictions(shuffled,available,families,teacher,tracks,video,shuffled_history)
            if active.any():error=max(error,float(np.max(np.abs(p[active].astype('float32')-only[active]))))
            take=np.isfinite(teacher)&active&ba&sa;y=np.clip(teacher[take],.02,.98);count+=len(y)
            for name,value in [('PAVA',p),('BIN',b),('SHUFFLED',s),('CONSTANT',prior)]:
                q=np.clip(value[take],1e-8,1-1e-8);loss[name]+=float(-(y*np.log(q)+(1-y)*np.log1p(-q)).sum())
            append_history(history,cue,available,families,teacher);append_history(shuffled_history,shuffled,available,families,teacher)
        assert error==0,(pair,error)
        result[pair]=dict(current_teacher_rows_with_common_crossfit_models=count,
            heldout_teacher_soft_CE={name:value/count if count else None for name,value in loss.items()},
            replayed_MONO_ONLY_maximum_probability_error=error)
    audit=dict(pairs=result,teacher_targets_not_GT=True,not_a_RMOT_accuracy_or_independent_validation_claim=True,
        identical_teacher_maps_across_neural_seeds_only_seed0_replayed=True,
        BIN_has_fixed_bins_and_smoothing_so_not_a_single_constraint_monotonicity_ablation=True)
    atomic_json(cache,audit)
    return audit


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
    preview=OUT/'analysis/publication_figure_previews';preview.mkdir(parents=True,exist_ok=True)
    fig.savefig(preview/'semantic_colour.png',dpi=180,bbox_inches='tight')
    fig.savefig(dest/'semantic_colour.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)


def plain_readout(repeated,diagnostic):
    """Interpret sealed comparisons; never choose or alter an adaptation rule."""
    lines=['本轮结果应该怎样理解：']
    for pair in ('C4','C1'):
        row=repeated['SEMANTIC_'+pair];comparisons=row['comparisons']
        primary=comparisons['MONO_vs_SEM_MEAN'];lo,hi=primary['parent_and_neural_seed_sensitivity_CI95']
        delta=primary['delta_HOTA'];seeds=primary['per_neural_seed_delta']
        judgement=('平均退步，不能支持保留该校准。' if delta<=0 else
            '平均改善，但描述性区间跨0，仍有明显不确定性。' if lo<=0 else
            '固定预测的开发描述性区间为正；这仍不是独立新域验证。')
        lines += [pair+'：MONO '+str(row['mean_HOTA']['MONO'])+'，同输入普通SEM均值KD '+str(row['mean_HOTA']['SEM_MEAN'])+'。',
            '单调后处理的独立差：'+describe(primary)+'；'+judgement,
            '三个神经种子的差分别为'+str(seeds)+'，共享同一份教师问答。',
            '语义输入对同初始化零输入：'+describe(comparisons['SEM_MEAN_vs_ZERO_MEAN'])+
            '。该对照仍未排除任意非零附加特征的正则作用。',
            '固定MONO数量的排序差：'+describe(comparisons['COUNT_OWN_MONO_vs_SEM_MEAN'])+'。',
            '固定KD数量的排序差：'+describe(comparisons['COUNT_BASELINE_MONO_vs_SEM_MEAN'])+'。']
        baseline_names=[name for name in row['mean_HOTA'] if name!='OLD_RGB_GRAPH' and
            (name in OLD_RGB or name in OLD_PCD or name in OLD_ZERO or name.startswith('SEM') or name.startswith('ZERO'))]
        strongest=max(baseline_names,key=lambda name:row['mean_HOTA'][name])
        lines += ['全部预先保留的普通蒸馏控制中，本方向事后最高均值为'+strongest+' '+str(row['mean_HOTA'][strongest])+
            '。这是保守对照描述，不是按方向挑选部署方法或checkpoint。',
            'MONO对此控制的差：'+describe(comparisons['MONO_vs_'+strongest])+'。']
        for method in ('SEM_MEAN','MONO'):
            records=[v['candidate_membership'][method] for key,v in diagnostic['conditions'].items() if key.startswith(pair+'_')]
            fields=('positive_queries_without_correct_output','positive_queries_empty','no_positive_candidate_queries_with_output','TP','FP','FN')
            means={key:float(np.mean([r[key] for r in records])) for key in fields}
            lines += [method+' 三种子查询/候选成员计数平均：'+json.dumps(means,ensure_ascii=False)+'。']
    lines += ['两种数量投影是机制诊断，不可与原始HOTA作线性加减分账。',
        '教师回答交叉预测的损失只衡量对有噪声教师的拟合，不证明目标语义更准确。',
        '全部数据属于开发数据；区间不重训整个前缀适配流程，也未修正多次开发。',
        '任何本轮队列或报告成功都不自动完成用户的科研Goal。','']
    return lines


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
    diagnostic=diagnosis(records);teacher_audit=teacher_validation(records)
    result=dict(status='SEMANTIC_COLOUR_AND_MONOTONE_ANALYZED',conditions=summary,repeated=repeated,primary='MONO_vs_SEM_MEAN',
        query_diagnosis=diagnostic,official_new_methods=174,existing_public_statistic_references_verified=reference_checks,
        auxiliary_teacher_prediction_audit=teacher_audit,
        no_new_teacher_or_encoder_calls=True,no_target_GT_in_adaptation=True,
        head_dimension=1081,ZERO_and_SEM_same_initialization=True,monotone_map_current_track_group_crossfit=True,
        SHUFFLED_only_changes_auxiliary_cue_not_the_neural_head=True,
        ZERO_controls_dimension_and_initialization_not_generic_nonzero_feature_variance=True,
        HSV_joint_from_marginals_is_approximation=True,physical_colour_prototypes_not_fit_to_target_GT=True,
        bounded_odds_are_regularization_not_independent_Bayesian_evidence=True,
        all_strong_existing_baselines_retained=True,all_development_data=True,independent_blind_validation=False,
        uncertainty='Original-parent and shared-QA seed descriptive intervals on fixed sealed predictions; the adaptation/acquisition pipeline is not refitted in bootstrap. No correction for development or multiple comparisons.',
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
        '主MONO与SEM_MEAN完全同头/同输入；BIN为非单调分箱替代估计器，SHUFFLED破坏辅助线索的图像对齐，NAIVE_MEAN简单平均，MONO_ONLY/零头作补充。',
        'BIN还改变分箱与平滑，不能将MONO与BIN的全部差值归因于单调约束本身。',
        'SHUFFLED只改变辅助校准线索，未重训随机线索头；ZERO控制维度/初始化，未单独排除任意非零额外特征的正则作用。若只保留神经输入增益，须据此限定语义归因。',
        '84原始+90双向数量全部封存后174官评；0新增教师/源前向，1160/1103同实际回答。',
        '旧基线充分统计采用已有公开记录的精确引用以控制Git<100MB，引用与官评读取数组逐值相等后才公开。',
        '所有描述性区间仍未校正多次开发；固定查询预算、种子、原始GT和指标均未改变。','']
    lines += plain_readout(repeated,diagnostic)
    for family,row in repeated.items():
        lines += [family,'三种子HOTA：'+json.dumps(row['mean_HOTA'])]
        lines += [name+'：'+describe(e) for name,e in row['comparisons'].items()]
    lines += ['', '检索与FP/FN诊断（非HOTA可加分账）：',json.dumps(diagnostic,ensure_ascii=False,indent=2),
        '教师回答交叉预测（不是GT准确率，不替代HOTA；BIN另含分箱/平滑差异，不能把差值全归单调约束）：',json.dumps(teacher_audit,ensure_ascii=False,indent=2),
        '拟合收据：',json.dumps({k:v['teacher_fit_audit'] for k,v in summary.items()},indent=2),
        '若新模块不优于同输入强KD或仅某个数量投影为正，就保留失败并继续研究。此报告与队列成功不会关闭Goal。',
        '新精度不与旧3512秒拼接；若保留最终方法，仍须对应的真实单方法冷启动测量。']
    (ROOT/'docs/TTA_RMOT_颜色部位语义与单调校准_机制实证_20261122.txt').write_text('\n'.join(lines)+'\n')
    figure(repeated);print(json.dumps(dict(event='SEMANTIC_COLOUR_ANALYSIS_COMPLETE',repeated=repeated,goal_status='ACTIVE')),flush=True)


if __name__=='__main__':main()
