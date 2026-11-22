"""Fresh FINAL/protocol runs with complete timing and quality parity."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_rgb_parity import source_spec
from scripts.a100_visual_classical_protocol_calibration import STUDY
from scripts.a100_visual_classical_protocol_calibration_review import quality_checks
from scripts.a100_visual_classical_protocol_cold_launch import PLAN
from scripts.a100_visual_classical_native_cold_analyze import read_run,input_parity
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_sparse_observation_analyze import effect
from scripts.a100_visual_classical_sparse_observation_report import describe
from scripts.a100_visual_classical_sparse_statistics_reproduce import resolve_record

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'


def prediction_parity(actual,pair,arm,source):
    expected=STUDY/pair/'seed0'/('BASE_PROTOCOL' if arm=='KD' else 'PROTOCOL_JOINT')
    seals=[]
    for root in (actual,expected):
        rows=json.loads((root/'predictions_seal.json').read_text())['episodes']
        seals.append({(r['video'],r['expression']):r['prediction_sha256'] for r in rows})
    assert seals[0].keys()==seals[1].keys();maximum=0.
    from flex_uda.a100.mechanism import metadata,PAIRS
    for video,bank in sorted(metadata(PAIRS[pair][1])['videos'].items()):
        with np.load(source/'source'/video/'sparse_inputs.npz') as a:tracks=a['tracks']
        index={int(t):i for i,t in enumerate(tracks)};columns=np.asarray([index[int(t)] for _,t in bank['observations']])
        with np.load(expected.parent/'videos'/video/'coarse_scores.npz') as a:
            reference=a['BASE_PROTOCOL' if arm=='KD' else 'PROTOCOL_JOINT'][:,columns]
        saved=np.load(actual/'videos'/video/'scores.npy')
        maximum=max(maximum,float(np.max(np.abs(saved.astype(float)-reference.astype(float)))))
    return dict(queries=len(seals[0]),changed_query_prediction_files=sum(seals[0][k]!=seals[1][k] for k in seals[0]),
        maximum_saved_score_probability_error=maximum,GT_open_attempts=0,not_a_new_independent_neural_seed=True)


def figure(runs,conditions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.4),constrained_layout=True)
    for ax,pair in zip(axes,('C4','C1')):
        for arm,label in (('KD','ordinary FINAL + task protocol'),('JOINT','protocol pooling + visual PAVA')):
            x=runs[pair+'_'+arm]['worker']['actual_worker_process_wall_seconds']
            y=conditions[pair+'_seed0']['raw_HOTA'][arm]
            ax.scatter(x,y,s=70,label=label);ax.annotate(f'{y:.3f}; {x:.0f}s',(x,y),xytext=(5,8),textcoords='offset points',fontsize=8)
        ax.set(title=pair,xlabel='Actual single-worker process wall seconds',ylabel='Actual seed0 HOTA');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Four fresh deployments: all loads, forwards, fits and outputs\nPrivate QA caches; ordinary skips unused RGB; supplied official candidates\nAll DEV; shared host/NAS load and staggered starts; not an iso-budget sweep',fontsize=9)
    preview=OUT/'analysis/publication_figure_previews';preview.mkdir(exist_ok=True,parents=True)
    (PUBLIC/'figures').mkdir(exist_ok=True)
    fig.savefig(preview/'protocol_cold.png',dpi=160,bbox_inches='tight')
    fig.savefig(PUBLIC/'figures/protocol_cold.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)


def main():
    official=json.loads((OUT/'analysis/protocol_cold_official.json').read_text())
    assert official['all_four_fresh_predictions_sealed_before_followup_GT']
    jobs=json.loads(PLAN.read_text())['jobs'];runs={key:read_run(job) for key,job in jobs.items()}
    quality=json.loads((PUBLIC/'protocol_calibration_analysis.json').read_text());checks,_=quality_checks(quality)
    assert all(checks.values())
    conditions={};statistics={};pair_audits={};history={};predictions={};cache={};operators={}
    for pair in ('C4','C1'):
        slug=pair+'_seed0';data={};left=Path(runs[pair+'_KD']['root']);right=Path(runs[pair+'_JOINT']['root'])
        pair_audits[pair]=input_parity(left,right,pair,heads=True,ignore_unused_RGB=True)
        history[pair+'_KD']=input_parity(left,source_spec(pair)[0],pair,ignore_unused_RGB=True)
        history[pair+'_JOINT']=input_parity(right,source_spec(pair)[0],pair)
        for arm in ('KD','JOINT'):
            job=jobs[pair+'_'+arm];source=Path(runs[pair+'_'+arm]['root']);root=source/'methods'/job['methods'][0]
            predictions[pair+'_'+arm]=prediction_parity(root,pair,arm,source);row=read_stats(root);data[arm]=row
            name='BASE_PROTOCOL' if arm=='KD' else 'PROTOCOL_JOINT';reference=slug+'/'+name
            prior=resolve_record(PUBLIC,'protocol_calibration_parent_statistics.json',reference,cache)
            assert row['parent_videos']==prior['parents'] and row['HOTA']==prior['HOTA']
            assert np.array_equal(row['parent_stats'][0],prior['numerator']) and np.array_equal(row['parent_stats'][1],prior['denominator'])
            statistics[slug+'/'+arm]=dict(reference=dict(file='protocol_calibration_parent_statistics.json',record=reference))
        conditions[slug]=dict(raw_HOTA={n:r['HOTA'] for n,r in data.items()},comparisons=dict(JOINT_vs_KD=effect([data['JOINT']],[data['KD']])))
        operators[pair]=dict(task_protocol=jobs[pair+'_JOINT']['task_protocol'],ordinary='FINAL nonempty-conditioned for PRESUPPOSED; unchanged for CHECKLIST',
            proposed='FINAL identity for PRESUPPOSED; global pool for CHECKLIST; then visual PAVA')
    assert all(r['exact_source_questions_and_teacher_answers_equal'] for r in list(pair_audits.values())+list(history.values()))
    assert not any(r['last_video_head_weight_max_abs_difference'] or r['last_video_head_statistic_max_abs_difference'] for r in pair_audits.values())
    assert not any(r['changed_query_prediction_files'] or r['maximum_saved_score_probability_error'] for r in predictions.values())
    for key,run in runs.items():
        assert run['protocol']['shared_head_readout']=='CONVERGED_FINAL'
        assert run['teacher_fit_audit']['videos_exceeding_registered_KL_tolerance']==0
        root=Path(run['root']);outputs=json.loads((root/'phase_ledger.json').read_text())['method_outputs']
        run['task_contract_pooling_CPU_seconds']=sum(r['task_contract_pooling']['actual_CPU_seconds'] for r in outputs if r.get('task_contract_pooling'))
    result=dict(status='CONVERGED_FINAL_PROTOCOL_ACTUAL_ANALYZED_ROOT_REVIEW_REQUIRED',conditions=conditions,repeated={},runs=runs,
        quality_checks=checks,actual_within_pair_evidence_audit=pair_audits,quality_source_evidence_audit=history,
        expected_seed0_prediction_audit=predictions,operators=operators,total_fresh_actual_QA=sum(r['teacher']['new_calls'] for r in runs.values()),
        actual_ordinary_receives_same_task_contract=True,shared_final_head_and_confidence_rule=True,
        all_four_fresh_predictions_sealed_before_followup_GT=True,new_workers_read_no_old_features_or_QA=True,
        quality_query_diagnosis_reference='protocol_calibration_analysis.json; same masks; canonical IoU membership is not semantic accuracy or official per-method TPFP',
        development_derived_recipe_on_development_data=True,independent_blind_validation=False,actual_reexecution_not_new_neural_seed=True,
        OS_NAS_caches_not_flushed=True,shared_host_load_and_staggered_starts_disclosed=True,
        upstream_official_candidate_generation_not_timed=True,no_speedup_or_global_Pareto_claim=True,
        old_mean_head_times_not_used=True,goal_status='ACTIVE_REQUIRES_ROOT_SCIENTIFIC_AND_PUBLICATION_REVIEW')
    atomic_json(OUT/'analysis/protocol_cold_analysis.json',result)
    (PUBLIC/'protocol_cold_analysis.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    (PUBLIC/'protocol_cold_parent_statistics.json').write_text(json.dumps(statistics,separators=(',',':'))+'\n')
    lines=['TTA-RMOT：共享收敛头、任务协议与视觉校准的四卡真实冷启动','2026-11-22','',
        '仅在预先固定质量门槛通过后执行。四份新单方法0/1/2/3进程，各自加载源/教师、重算特征、独立SQLite，所有新预测封存后本组GT。',
        '普通基线是共享强置信度加权收敛FINAL，并获得同一非空/清单任务协议；跳过不使用的手工RGB。组合承担真实RGB、场景汇聚和PAVA开销。',
        '不是复用旧均值头时间；原始FINAL及完整强软/硬/均值/RGB/SEM控制、双向数量及模块归因见protocol_calibration_analysis.json。','']
    for pair,row in conditions.items():lines += [pair,json.dumps(row['raw_HOTA']),describe(row['comparisons']['JOINT_vs_KD'])]
    for key,row in runs.items():lines += ['',key,'完整外部进程秒='+str(row['worker']['actual_worker_process_wall_seconds']),
        '阶段秒='+json.dumps(row['phase_seconds']),'RGB实际计算='+str(row['handcrafted_RGB_statistics_computed']),
        '拟合/保存秒='+str(row['actual_head_fit_and_save_seconds'])+'；后处理/输出秒='+str(row['actual_all_method_postprocess_and_output_seconds']),
        '其中任务池化CPU秒='+str(row['task_contract_pooling_CPU_seconds'])+'；PAVA秒='+str(row['appearance_regularization_CPU_seconds']),
        '真实问答='+str(row['teacher']['new_calls'])+'；缓存命中='+str(row['teacher']['cache_hits'])]
    lines += ['', '双方输入/教师/最终头：'+json.dumps(pair_audits,separators=(',',':')),
        '质量源一致性：'+json.dumps(history,separators=(',',':')),'质量分数/文件重放：'+json.dumps(predictions,separators=(',',':')),
        '任务协议：'+json.dumps(operators,separators=(',',':')),
        '全部数据为DEV；规则来自多轮开发后的简化，不是独立盲测。真实执行是seed0再执行，不是第四个独立神经种子。',
        '每个独立进程Popen-to-exit包括启动、加载、前向、拟合、校准、输出；官方GT评估不计部署成本。官方固定候选生成成本未测，OS/NAS缓存未清理，共享主机错峰负载不受控。',
        '不把单次秒差写成算法提速，也不把两个端点称为全预算最优成本曲线。数量/基率收益不冒充纯语义排序。',
        'Goal不由执行PASS或报告自动关闭。']
    (ROOT/'docs/TTA_RMOT_收敛头与任务协议视觉校准_四卡真实冷启动_20261122.txt').write_text('\n'.join(lines)+'\n')
    figure(runs,conditions)
    print(json.dumps(dict(event='PROTOCOL_COLD_ANALYZED',conditions=conditions,total_fresh_actual_QA=result['total_fresh_actual_QA'],goal_status='ACTIVE')),flush=True)


if __name__=='__main__':main()
