"""True selected-topology deployment receipts versus optimized ordinary KD."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_rgb_parity import source_spec
from scripts.a100_visual_classical_direct_teacher_pcd import DIRECT
from scripts.a100_visual_classical_stable_topology import PREDICTIVE
from scripts.a100_visual_classical_stable_cold_launch import PLAN,promotion_checks
from scripts.a100_visual_classical_native_cold_analyze import read_run,input_parity,postseal_diagnosis
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_sparse_observation_analyze import effect
from scripts.a100_visual_classical_sparse_observation_report import describe
from scripts.a100_visual_classical_sparse_statistics_reproduce import resolve_record

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'


def prediction_parity(actual,pair,arm,source):
    from flex_uda.a100.mechanism import metadata,PAIRS
    expected=((OUT/'confidence_teacher') if arm=='KD' else PREDICTIVE)/pair/'seed0'/('KD_MEAN' if arm=='KD' else 'SELECTED_JOINT')
    seals=[]
    for root in (actual,expected):
        rows=json.loads((root/'predictions_seal.json').read_text())['episodes']
        seals.append({(r['video'],r['expression']):r['prediction_sha256'] for r in rows})
    assert seals[0].keys()==seals[1].keys();maximum=0.
    for video,bank in sorted(metadata(PAIRS[pair][1])['videos'].items()):
        with np.load(source/'source'/video/'sparse_inputs.npz') as a:tracks=a['tracks']
        lookup={int(t):i for i,t in enumerate(tracks)};columns=np.asarray([lookup[int(t)] for _,t in bank['observations']])
        with np.load(expected.parent/'videos'/video/'coarse_scores.npz') as a:reference=a['KD_MEAN' if arm=='KD' else 'SELECTED_JOINT'][:,columns]
        saved=np.load(actual/'videos'/video/'scores.npy')
        maximum=max(maximum,float(np.max(np.abs(saved.astype(float)-reference.astype(float)))))
    return dict(queries=len(seals[0]),changed_query_prediction_files=sum(seals[0][k]!=seals[1][k] for k in seals[0]),maximum_saved_score_probability_error=maximum,
        GT_open_attempts=0,not_an_additional_independent_neural_seed=True)


def figure(runs,conditions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    previous=json.loads((PUBLIC/'native_cold_analysis.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for ax,pair in zip(axes,('C4','C1')):
        points=[('fresh confidence-weighted KD',runs[pair+'_KD']['worker']['actual_worker_process_wall_seconds'],conditions[pair+'_seed0']['raw_HOTA']['KD']),
            ('previous unweighted KD',previous['runs'][pair+'_KD']['worker']['actual_worker_process_wall_seconds'],previous['conditions'][pair+'_seed0']['raw_HOTA']['KD']),
            ('fixed local + PAVA',previous['runs'][pair+'_JOINT']['worker']['actual_worker_process_wall_seconds'],previous['conditions'][pair+'_seed0']['raw_HOTA']['JOINT']),
            ('teacher-selected + PAVA',runs[pair+'_JOINT']['worker']['actual_worker_process_wall_seconds'],conditions[pair+'_seed0']['raw_HOTA']['SELECTED_JOINT'])]
        for i,(label,x,y) in enumerate(points):
            ax.scatter(x,y,s=65,label=label);ax.annotate(f'{y:.3f}; {x:.0f}s',(x,y),xytext=(6,8 if i!=1 else -15),textcoords='offset points',fontsize=8)
        ax.set(title=pair,xlabel='Actual single-worker Popen-to-exit seconds',ylabel='Actual seed0 HOTA');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Fresh deployment points, not an iso-budget sweep\nAll loads/forwards/fitting/output included; teacher QA identical, private caches\nAll DEV; staggered starts/shared NAS and host load; no timing speedup or all-budget Pareto claim',fontsize=9)
    (PUBLIC/'figures').mkdir(exist_ok=True);preview=OUT/'analysis/publication_figure_previews';preview.mkdir(exist_ok=True,parents=True)
    fig.savefig(preview/'stable_cold.png',dpi=160,bbox_inches='tight')
    fig.savefig(PUBLIC/'figures/stable_cold.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)


def main():
    official=json.loads((OUT/'analysis/stable_cold_official.json').read_text());assert official['all_four_fresh_predictions_sealed_before_followup_GT']
    plan=json.loads(PLAN.read_text());jobs=plan['jobs'];runs={key:read_run(job) for key,job in jobs.items()}
    quality=json.loads((PUBLIC/'stable_topology_analysis.json').read_text());weighted=json.loads((PUBLIC/'confidence_teacher_analysis.json').read_text());checks=promotion_checks(quality,weighted);assert all(checks.values())
    conditions={};statistics={};pair_audits={};history={};predictions={};selectors={};cache={}
    for pair in ('C4','C1'):
        slug=pair+'_seed0';data={};left=Path(runs[pair+'_KD']['root']);right=Path(runs[pair+'_JOINT']['root'])
        pair_audits[pair]=input_parity(left,right,pair,heads=True,ignore_unused_RGB=True)
        history[pair+'_KD']=input_parity(left,source_spec(pair)[0],pair,ignore_unused_RGB=True)
        history[pair+'_SELECTED']=input_parity(right,source_spec(pair)[0],pair)
        for arm,name in (('KD','KD'),('JOINT','SELECTED_JOINT')):
            job=jobs[pair+'_'+arm];source=Path(runs[pair+'_'+arm]['root']);root=source/'methods'/job['methods'][0]
            predictions[pair+'_'+arm]=prediction_parity(root,pair,arm,source);row=read_stats(root);data[name]=row
            filename='confidence_teacher_parent_statistics.json' if arm=='KD' else 'stable_topology_parent_statistics.json'
            reference=slug+'/'+('KD_MEAN' if arm=='KD' else 'SELECTED_JOINT');prior=resolve_record(PUBLIC,filename,reference,cache)
            assert row['parent_videos']==prior['parents'] and row['HOTA']==prior['HOTA']
            assert np.array_equal(row['parent_stats'][0],prior['numerator']) and np.array_equal(row['parent_stats'][1],prior['denominator'])
            statistics[slug+'/'+name]=dict(reference=dict(file=filename,record=reference))
        changed=0;global_count=0;seconds=0.;videos=0
        for file in sorted((right/'source').glob('*/topology_selection.json')):
            value=json.loads(file.read_text());seconds+=value.pop('actual_CPU_seconds');videos+=1;global_count+=value['selected_model']=='GLOBAL'
            reference=json.loads((PREDICTIVE/pair/'seed0/videos'/file.parent.name/'receipt.json').read_text())['selector'];changed+=value!=reference
        assert videos==(45 if pair=='C4' else 25) and not changed
        selectors[pair]=dict(videos=videos,changed_teacher_only_choices_or_losses=changed,global_choices=global_count,actual_selector_CPU_seconds=seconds)
        runs[pair+'_JOINT']['topology_selection_CPU_seconds']=seconds
        conditions[slug]=dict(raw_HOTA={n:r['HOTA'] for n,r in data.items()},comparisons=dict(SELECTED_JOINT_vs_KD=effect([data['SELECTED_JOINT']],[data['KD']])))
    assert all(r['exact_source_questions_and_teacher_answers_equal'] for r in list(pair_audits.values())+list(history.values()))
    # Weighted baseline and unweighted candidate intentionally fit different heads.
    assert not any(r['last_video_head_statistic_max_abs_difference'] for r in pair_audits.values())
    assert not any(r['changed_query_prediction_files'] or r['maximum_saved_score_probability_error'] for r in predictions.values())
    diagnostic=postseal_diagnosis(jobs)
    result=dict(status='STABLE_TOPOLOGY_ACTUAL_ANALYZED_ROOT_REVIEW_REQUIRED',conditions=conditions,repeated={},runs=runs,
        quality_promotion_checks=checks,actual_within_pair_evidence_audit=pair_audits,quality_source_evidence_audit=history,
        expected_seed0_prediction_audit=predictions,actual_teacher_selection_audit=selectors,postseal_candidate_membership_diagnosis=diagnostic,
        diagnostic_JOINT_name_means_selected_topology=True,total_final_record_actual_QA=sum(r['teacher']['new_calls'] for r in runs.values()),
        newly_executed_selected_QA=sum(r['teacher']['new_calls'] for k,r in runs.items() if k.endswith('_JOINT')),newly_executed_all_four_QA=sum(r['teacher']['new_calls'] for r in runs.values()),
        previous_optimized_KD_records_reused_only_for_comparison=False,ordinary_baseline_is_fresh_confidence_weighted_KD=True,head_weights_intentionally_different=True,same_head_causal_controls_are_separate_quality_studies=True,new_selected_workers_read_no_old_features_or_answers=True,
        both_new_selected_predictions_sealed_before_followup_GT=True,previous_KD_GT_evaluated_separately_after_its_original_seal=False,all_four_fresh_predictions_sealed_before_followup_GT=True,
        all_development_data=True,independent_blind_validation=False,actual_reexecution_not_new_neural_seed=True,
        shared_host_load_and_staggered_start_disclosed=True,no_algorithmic_timing_speedup_or_all_budget_Pareto_claim=True,
        upstream_fixed_official_candidate_generation_not_timed=True,old_fixed_local_time_never_pasted_onto_selected_accuracy=True,
        goal_status='ACTIVE_REQUIRES_ROOT_SCIENTIFIC_AND_PUBLICATION_REVIEW')
    atomic_json(OUT/'analysis/stable_cold_analysis.json',result)
    (PUBLIC/'stable_cold_analysis.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    (PUBLIC/'stable_cold_parent_statistics.json').write_text(json.dumps(statistics,separators=(',',':'))+'\n')
    lines=['TTA-RMOT：教师ONE_SE稳定参照结构的真实冷启动实证','2026-11-22','',
        '只有预先登记的质量及加权强基线条件达标后才执行。四个新GPU0/1/2/3单方法进程，各自重载冻结源/教师、重算图像、新SQLite，共新增4526真实回答、预期0缓存。',
        '对照为置信度加权的普通原生KD，跳过不使用的RGB统计；四份都是全新物理执行，不读取历史特征/问答或预付缓存。',
        '四个新预测全部封存后才本组GT。全部数据曾用于开发，不是独立盲测。',
        '当前视频转导，TrackID五折教师留出预测加固定ONE_SE风险正则选局部/全局，随后固定PAVA；无目标人工标签适配，不按域名选图。','']
    for pair,row in conditions.items():lines += [pair,json.dumps(row['raw_HOTA']),describe(row['comparisons']['SELECTED_JOINT_vs_KD'])]
    for key,row in runs.items():lines += [key,'实际外部进程秒='+str(row['worker']['actual_worker_process_wall_seconds']),
        '阶段秒='+json.dumps(row['phase_seconds']),'RGB统计实际计算='+str(row['handcrafted_RGB_statistics_computed']),
        '实际拟合/保存秒='+str(row['actual_head_fit_and_save_seconds'])+'；后处理/输出秒='+str(row['actual_all_method_postprocess_and_output_seconds']),
        '其中PAVA秒='+str(row['appearance_regularization_CPU_seconds'])+'；结构选择秒='+str(row.get('topology_selection_CPU_seconds',0.)),
        '真实教师='+str(row['teacher']['new_calls'])+'；缓存='+str(row['teacher']['cache_hits'])]
    lines += ['输入/问答一致性与最后头差异（加权和未加权拟合权重预期不同）：'+json.dumps(pair_audits,separators=(',',':')),
        '质量源一致性：'+json.dumps(history,separators=(',',':')),'保存分数和预测一致性：'+json.dumps(predictions,separators=(',',':')),
        '实际选择一致性：'+json.dumps(selectors,separators=(',',':')),'封存后检索权衡（JOINT在此表示选择结构）：'+json.dumps(diagnostic,separators=(',',':')),
        '这是真实独立执行，不是新的独立神经种子。三种子因果与数量剥离仍见stable_topology_analysis.json。本组加权KD与未加权头+经典模块的最终方法不同；同头模块贡献由质量消融单独证明。两种方法的精度和时间均分别真实测量，不拼接旧数字。',
        '每个方法自身完整Popen-to-exit计时；OS/NAS缓存未清理、不同启动时刻/共享主机负载不受控，不能从秒差声称算法加速或全预算最优。',
        '已提供官方候选的生成不计入本适配模块成本。人工GT仅封存后评估；全部数据曾用于开发，尚无独立新域盲测。',
        '完整旧固定局部四卡、旧残差四卡、被中止未优化KD科研开销以及所有负结果仍保留。Goal不由报告自动结项。']
    (ROOT/'docs/TTA_RMOT_教师稳定参照与经典校准_真实冷启动_20261122.txt').write_text('\n'.join(lines)+'\n')
    figure(runs,conditions)
    print(json.dumps(dict(event='STABLE_COLD_ANALYSIS_COMPLETE',conditions=conditions,selectors=selectors,goal_status='ACTIVE')),flush=True)


if __name__=='__main__':main()
