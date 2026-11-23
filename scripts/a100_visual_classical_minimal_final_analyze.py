"""Provisional minimal graph: independent C4 deployment and fresh C1 audit."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_sparse_observation_analyze import effect
from scripts.a100_visual_classical_sparse_observation_report import describe

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'
SINGLE_KD=OUT/'cold_full_c4/C4_minimal_single_KD_rho0p1_20261122'
SINGLE_GRAPH=OUT/'cold_full_c4/C4_minimal_single_GRAPH_rho0p1_20261122'
C1=OUT/'cold_full_c4/C1_key_rho0p1_mean_pair_20261122'


def read_run(root,questions,single):
    complete=json.loads((root/'completion.json').read_text())
    worker=json.loads((root/'worker_receipt.json').read_text())
    official=json.loads((root/'official_evaluation.json').read_text())
    assert worker['exit_code']==0
    assert complete['GT_open_attempts']==complete['prepaid_artifact_reads']==complete['teacher']['cache_hits']==0
    assert complete['logical_teacher_calls']==complete['teacher']['new_calls']==questions
    assert complete['single_method_execution']==single
    assert worker['independent_single_arm_processes']==single
    assert official['status']==('COMPLETE_FRESH_BUDGET_SINGLE' if single else 'COMPLETE_FRESH_BUDGET_PAIR')
    ledger=json.loads((root/'phase_ledger.json').read_text())
    fits=[r['receipt'] for r in ledger['head_fits']]
    audit=dict(fitted_videos=len(fits),maximum_teacher_KL=max(r['teacher_KL'] for r in fits),
        videos_exceeding_KL_tolerance=sum(not r['teacher_fit_within_0_005'] for r in fits),
        maximum_steps=max(r['steps'] for r in fits))
    phases={}
    for row in ledger['common_phases']:phases[row['phase']]=phases.get(row['phase'],0)+row['seconds']
    receipts=[json.loads(p.read_text()) for p in sorted((root/'source').glob('*/receipt.json'))]
    return dict(root=str(root),protocol=complete['protocol'],worker=worker,source=complete['source'],
        teacher=complete['teacher'],cost_by_method=complete['cost_by_method'],phase_seconds=phases,
        teacher_fit_audit=audit,decoder_frames=sum(r['source_frames'] for r in receipts),
        backbone_image_forwards=sum(r['actual_frozen_encoder_statistics']['backbone_images'] for r in receipts),
        provided_candidates_boundary=True,upstream_detector_tracker_generation_timed=False)


def source_parity():
    result=dict(videos=0,PCD_max_abs_difference=0.,RGB_max_abs_difference=0.,prior_max_abs_difference=0.,
        ordered_question_list_changed_videos=0,question_symmetric_difference=0,
        common_answer_positions=0,teacher_probability_max_abs_difference=0.)
    paths=sorted((SINGLE_KD/'source').glob('*/sparse_inputs.npz'))
    assert len(paths)==45
    for path in paths:
        video=path.parent.name
        with np.load(path) as a,np.load(SINGLE_GRAPH/'source'/video/'sparse_inputs.npz') as b:
            assert np.array_equal(a['tracks'],b['tracks'])
            for field,label in [('pcd','PCD'),('colours','RGB'),('prior','prior')]:
                result[label+'_max_abs_difference']=max(result[label+'_max_abs_difference'],
                    float(np.max(np.abs(a[field].astype(float)-b[field].astype(float)))))
        requests=[];answers=[]
        for root in (SINGLE_KD,SINGLE_GRAPH):
            questions=json.loads((root/'source'/video/'questions_before_answers.json').read_text())['requests']
            key=lambda r:(int(r['local_query']),int(r['local_track']),int(r['frame_id']),r['view'])
            requests.append([key(r) for r in questions])
            returned=json.loads((root/'source'/video/'answers.json').read_text())['answers']
            answers.append({(int(r['local_query']),int(r['local_track'])):float(r['yes_probability']) for r in returned})
        result['ordered_question_list_changed_videos']+=requests[0]!=requests[1]
        result['question_symmetric_difference']+=len(set(requests[0])^set(requests[1]))
        common=set(answers[0])&set(answers[1]);result['common_answer_positions']+=len(common)
        if common:result['teacher_probability_max_abs_difference']=max(result['teacher_probability_max_abs_difference'],
            max(abs(answers[0][k]-answers[1][k]) for k in common))
        result['videos']+=1
    import torch
    heads=[torch.load(root/'CLIP_latest_head.pt',map_location='cpu',weights_only=False) for root in (SINGLE_KD,SINGLE_GRAPH)]
    result['last_video_head_weight_max_abs_difference']=max(float((heads[0]['model'][k]-heads[1]['model'][k]).abs().max()) for k in heads[0]['model'])
    result['last_video_head_statistic_max_abs_difference']=max(float(np.max(np.abs(heads[0][k]-heads[1][k]))) for k in ('mean','std'))
    result['exact_source_questions_and_teacher_answers_equal']=not any(result[k] for k in
        ('PCD_max_abs_difference','RGB_max_abs_difference','prior_max_abs_difference','ordered_question_list_changed_videos',
         'question_symmetric_difference','teacher_probability_max_abs_difference'))
    result['head_weight_check_is_last_video_only']=True
    return result


def c1_scope_equivalence():
    """No GT in this calculation: same actual source/answers/head, both masks."""
    from flex_uda.a100.mechanism import metadata
    from flex_uda.active.acquisition import exclusive_group
    from flex_uda.active.color_cohort import augmented_features,part_mask
    from flex_uda.active.context_cohort import context_part_mask
    from flex_uda.active.visual_classical import refine_distilled_prior
    from scripts.a100_visual_classical_strict_prefix import plain
    m=metadata('Dance');changes=0;queries_checked=0
    for video,bank in sorted(m['videos'].items()):
        with np.load(C1/'source'/video/'sparse_inputs.npz') as a:
            pcd,colours,tracks=a['pcd'],a['colours'],a['tracks']
        queries=[q for q in m['episodes'] if q['video_id']==video];texts=[q['query_text'] for q in queries]
        masks=np.stack([context_part_mask(text) for text in texts])
        assert np.array_equal(masks,np.stack([part_mask(text) for text in texts]))
        geometry,normalized,_=augmented_features(pcd,colours,texts)
        parsed=np.stack([(colours*mask[None,:,None]).reshape(len(colours),-1) for mask in masks])
        parsed/=np.maximum(np.linalg.norm(parsed,axis=2,keepdims=True),1e-8)
        extended=np.concatenate([normalized,parsed],axis=2).astype('float32')
        assert np.array_equal(geometry,extended)
        obs=np.asarray(bank['observations']);lookup={int(t):i for i,t in enumerate(tracks)}
        columns=np.asarray([lookup[int(t)] for _,t in obs]);first=[int(np.flatnonzero(columns==ti)[0]) for ti in range(len(tracks))]
        unary=np.load(C1/'methods/CLIP_MEAN/videos'/video/'scores.npy')[:,first]
        teacher=np.full(unary.shape,np.nan)
        for row in json.loads((C1/'source'/video/'answers.json').read_text())['answers']:
            teacher[int(row['local_query']),int(row['local_track'])]=row['yes_probability']
        score=unary.copy()
        for q,mask in enumerate(masks):
            if mask.any():score[q]=refine_distilled_prior(unary[q],extended[q],teacher[q],source_weight=.1,
                node_mask=np.all(colours[:,mask>0].sum(2)>0,axis=1))
        result=plain(score[:,columns],exclusive_group(texts,'Dance'))
        actual=np.load(C1/'methods/CLIP_MEAN_COHORT/videos'/video/'mask.npy')
        changes+=int(np.count_nonzero(result!=actual));queries_checked+=len(queries)
    assert changes==0,changes
    return dict(status='PASS_FRESH_C1_CLOTHING_EQUALS_FINAL_EXTENDED',videos=25,queries=queries_checked,
        changed_candidate_rows=changes,source_geometry_bitwise_equal=True,teacher_and_head_unchanged=True,
        raw_GT_used_in_this_calculation=False,new_teacher_calls=0,head_updates=0)


def postseal_errors():
    """IoU-membership diagnosis only; these labels never enter adaptation."""
    from collections import Counter
    from flex_uda.a100.mechanism import OUT as DIAG,metadata,PAIRS
    result={}
    for pair in ('C4','C1'):
        m=metadata(PAIRS[pair][1]);counts=Counter();rows={name:Counter() for name in ('KD','OURS')}
        kdroot=SINGLE_KD if pair=='C4' else C1;oursroot=SINGLE_GRAPH if pair=='C4' else C1
        for video,bank in m['videos'].items():
            truth=np.load(DIAG/'offline_GT'/pair/video/'labels.npy')>=0
            kd=np.load(kdroot/'methods/CLIP_MEAN/videos'/video/'mask.npy')
            ours=np.load(oursroot/'methods/CLIP_MEAN_COHORT/videos'/video/'mask.npy')
            assert truth.shape==kd.shape==ours.shape
            obs=np.asarray(bank['observations'])
            tracks=sorted(set(obs[:,1].tolist()));columns=np.searchsorted(tracks,obs[:,1])
            known=np.zeros((len(truth),len(tracks)),bool)
            for row in json.loads((oursroot/'source'/video/'answers.json').read_text())['answers']:
                known[int(row['local_query']),int(row['local_track'])]=True
            asked=known[:,columns];present=truth.any(1)
            correct=dict(KD=(kd&truth).any(1),OURS=(ours&truth).any(1))
            no_correct_asked=present&~(truth&asked).any(1)
            counts.update(queries=len(truth),queries_with_positive_candidate=int(present.sum()),
                queries_without_any_correct_track_asked=int(no_correct_asked.sum()),
                rescued_KD_miss_queries=int((present&~correct['KD']&correct['OURS']).sum()),
                regressed_KD_correct_queries=int((present&correct['KD']&~correct['OURS']).sum()))
            for name,mask in [('KD',kd),('OURS',ours)]:
                rows[name].update(TP=int((truth&mask).sum()),FP=int((~truth&mask).sum()),FN=int((truth&~mask).sum()),
                    positive_candidate_queries_with_zero_output=int((present&~mask.any(1)).sum()),
                    positive_candidate_queries_with_no_correct_output=int((present&~correct[name]).sum()),
                    true_positive_rows_on_unasked_tracks=int((truth&mask&~asked).sum()),
                    recovered_queries_without_correct_track_asked=int((no_correct_asked&correct[name]).sum()))
        result[pair]=dict(query_counts=dict(counts),candidate_membership={name:dict(c) for name,c in rows.items()})
    return dict(pairs=result,target_GT_only_after_all_actual_predictions_sealed=True,
        IoU_membership_counts_not_additive_official_HOTA_attribution=True,
        query_presence_means_positive_candidate_recoverable_not_all_physical_GT=True,
        no_GT_used_in_adaptation=True)


def figure(runs,conditions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    names=['KD','OURS'];values=[runs['C4_'+name]['worker']['actual_worker_process_wall_seconds'] for name in names]
    axes[0].bar(names,values,color=['#7aa6c2','#e59d65'])
    for i,name in enumerate(names):
        axes[0].text(i,values[i]+max(values)*.02,f"HOTA {conditions['C4_seed0']['raw_HOTA'][name]:.3f}",ha='center')
    axes[0].set(title='C4: independent actual single-method processes',ylabel='Actual external worker wall seconds',ylim=(0,max(values)*1.13))
    for i,pair in enumerate(('C4','C1')):
        row=conditions[pair+'_seed0']['comparisons']['OURS_vs_KD'];lo,hi=row['fixed_neural_seeds_parent_CI95']
        axes[1].hlines(i,lo,hi,color='tab:blue',lw=2);axes[1].plot(row['delta_HOTA'],i,'o',color='tab:blue')
    axes[1].set_yticks([0,1],['C4 independent fresh','C1 shared fresh']);axes[1].invert_yaxis()
    axes[1].axvline(0,color='black',lw=.8);axes[1].set(title='Actual fresh quality, fixed seed0',xlabel='HOTA gain; original-parent95% interval')
    for ax in axes:ax.grid(axis='y' if ax==axes[0] else 'x',alpha=.2)
    fig.suptitle('Provisional original-head candidate; equal-RGB and query-retrieval checks pending\nProvided official boxes/IDs; no target GT in adaptation; all development data\nParallel A100 workers share NAS/OS/background load: no algorithmic speedup inferred from wall differences',fontsize=10)
    destination=PUBLIC/'figures';destination.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination/'minimal_final_physical.png',dpi=180)
    fig.savefig(destination/'minimal_final_physical.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)


def main():
    runs=dict(C4_KD=read_run(SINGLE_KD,1160,True),C4_OURS=read_run(SINGLE_GRAPH,1160,True),C1_SHARED=read_run(C1,1103,False))
    parity=source_parity();equivalence=c1_scope_equivalence();errors=postseal_errors()
    data={'C4_seed0':dict(KD=read_stats(SINGLE_KD/'methods/CLIP_MEAN'),OURS=read_stats(SINGLE_GRAPH/'methods/CLIP_MEAN_COHORT')),
        'C1_seed0':dict(KD=read_stats(C1/'methods/CLIP_MEAN'),OURS=read_stats(C1/'methods/CLIP_MEAN_COHORT'))}
    conditions={};sufficient={}
    for label,methods in data.items():
        conditions[label]=dict(raw_HOTA={name:row['HOTA'] for name,row in methods.items()},
            comparisons={'OURS_vs_KD':effect([methods['OURS']],[methods['KD']])})
        for name,row in methods.items():sufficient[label+'/'+name]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],
            numerator=row['parent_stats'][0].tolist(),denominator=row['parent_stats'][1].tolist())
    quality=json.loads((PUBLIC/'query_scope_analysis.json').read_text())
    mechanisms={pair:quality['repeated']['QUERY_SCOPE_'+pair] for pair in ('C4','C1')}
    acceptance=dict(C4_independent_single_method_physical_cost=True,
        C4_actual_source_questions_and_teacher_equal=parity['exact_source_questions_and_teacher_answers_equal'],
        C1_actual_final_scope_equivalence=equivalence['changed_candidate_rows']==0,
        all_actual_heads_meet_registered_fit=not any(row['teacher_fit_audit']['videos_exceeding_KL_tolerance'] for row in runs.values()),
        fresh_point_gains={label:row['comparisons']['OURS_vs_KD']['delta_HOTA']>0 for label,row in conditions.items()},
        fixed_QA_three_seed_mean_KD_gains_positive={pair:mechanisms[pair]['comparisons']['ORIGINAL_MEAN_EXTENDED_vs_KD_ORIGINAL_MEAN']['parent_and_neural_seed_sensitivity_CI95'][0]>0 for pair in ('C4','C1')},
        C4_scope_independent_gain_positive=mechanisms['C4']['comparisons']['ORIGINAL_MEAN_EXTENDED_vs_ORIGINAL_MEAN_CLOTHING']['parent_and_neural_seed_sensitivity_CI95'][0]>0,
        C1_general_pure_ordering_claim=False,independent_blind_validation=False)
    result=dict(status='PROVISIONAL_MINIMAL_FRESH_EVIDENCE_ANALYZED',conditions=conditions,repeated={},runs=runs,
        C4_independent_input_teacher_audit=parity,C1_scope_equivalence=equivalence,acceptance=acceptance,
        postseal_candidate_membership_diagnosis=errors,
        frozen_QA_three_seed_reference='query_scope_analysis.json',all_development_data=True,
        no_target_manual_labels_in_adaptation=True,upstream_official_proposals_shared=True,
        C4_real_singletons_vs_C1_paired_cost_scope_explicit=True,
        RGB1080_information_parity_requires_separate_strict_rgb_study=True,
        query_retrieval_tradeoff_requires_null_set_followup=True,
        candidate_finally_accepted=False,goal_status='ACTIVE_PENDING_STRICT_RGB_PARITY_AND_ROOT_ACCEPTANCE')
    atomic_json(PUBLIC/'minimal_final_analysis.json',result);atomic_json(OUT/'analysis/minimal_final_analysis.json',result)
    (PUBLIC/'minimal_final_parent_statistics.json').write_text(json.dumps(sufficient,separators=(',',':'))+'\n')
    atomic_json(PUBLIC/'minimal_final_C1_equivalence.json',equivalence)
    lines=['TTA-RMOT：暂定最小方案的真实前向、独立进程成本和因果边界','2026-11-22','',
        'Goal待根代理对全部科学验收和发布完成后判断。本脚本不自动结项。',
        '暂定候选：原1080维教师蒸馏头、均值checkpoint、查询条件视觉图；源主干/教师冻结。',
        '公平性尚需严格RGB1080对照：本候选图比原KD额外使用空间/角色RGB，不能仅凭本报告认定最终胜出。',
        '封存后检索诊断：C4正查询空输出19→61，救回4条却损失23条正确检索；总HOTA提高不代表漏目标问题解决，尚待协议保护对照。',
        '不保留LoRA、观察anchor对齐、新8维位置、外观衰减时序或全域不确定性选题作为已证实贡献。',
        '适配只用目标图像、任务语言、官方候选框/ID和教师。人工标签仅用于预测封存后的评估。',
        'C4为Dance-best→SportsMOT，C1为KITTI-V2→Dance。官方前端是共享输入，不宣称完整检测器source-only。','',
        '一、固定问答三种子机制证据（不是本次冷启动时间）']
    for pair,row in mechanisms.items():
        lines += [pair,'原均值KD='+str(row['mean_HOTA']['KD_ORIGINAL_MEAN'])+'；扩展图='+str(row['mean_HOTA']['ORIGINAL_MEAN_EXTENDED']),
            '同头增益：'+describe(row['comparisons']['ORIGINAL_MEAN_EXTENDED_vs_KD_ORIGINAL_MEAN']),
            '范围贡献：'+describe(row['comparisons']['ORIGINAL_MEAN_EXTENDED_vs_ORIGINAL_MEAN_CLOTHING']),
            '按方法数量：'+describe(row['comparisons']['COUNT_OWN_ORIGINAL_MEAN_EXTENDED_vs_KD_ORIGINAL_MEAN']),
            '按KD数量：'+describe(row['comparisons']['COUNT_BASELINE_ORIGINAL_MEAN_EXTENDED_vs_KD_ORIGINAL_MEAN'])]
    lines += ['', '二、这次真正新鲜的精度与各自成本']
    for pair,row in conditions.items():
        lines += [pair,json.dumps(row['raw_HOTA']),'新鲜增益：'+describe(row['comparisons']['OURS_vs_KD'])]
    for name,row in runs.items():
        lines += [name,'真实外部工作进程：'+json.dumps(row['worker'],ensure_ascii=False),
            '阶段秒数：'+json.dumps(row['phase_seconds'],ensure_ascii=False),
            '方法组件分账：'+json.dumps(row['cost_by_method'],ensure_ascii=False),
            '真实教师：'+json.dumps(row['teacher'],ensure_ascii=False),
            '拟合检查：'+json.dumps(row['teacher_fit_audit']),
            'decoder帧='+str(row['decoder_frames'])+'；实际backbone图像前向='+str(row['backbone_image_forwards'])]
    lines += ['', '三、独立运行的输入/问答与实际C1等价',json.dumps(parity,ensure_ascii=False,indent=2),
        json.dumps(equivalence,ensure_ascii=False,indent=2),'',
        '封存后的候选IoU成员误差诊断（不是HOTA损失可加拆账）：',json.dumps(errors,ensure_ascii=False,indent=2),'',
        '四、可以说什么，不能说什么',
        'C4独立单方法Popen-to-exit是实际冷进程端点；C1是共享新鲜流，common+own是实测组件分账，不是独立端点。',
        '源学生加载、实际特征、教师加载/真实推理、小头拟合、输出都计入，没有把逻辑调用数当墙钟。',
        '提供候选前端的检测/物理跟踪生成不在成本边界；所有数据都是参与开发的数据，没有独立盲测。',
        '并行卡和共享NAS/OS可导致总时间差，不能因此说图算法加速了主干或证明全局Pareto最优。',
        'C4图对KD的双向数量结果支持排序增益；C1反向差约0，主要数量/选择校准，不虚构普遍跨域语义排序提升。',
        '范围扩展本身主要改变合适的输出数量；原头和全部权重保持，没有把初始化变化当创新。',
        '前次新场景输入/位置边不独立有效，完整67.495比最小67.228的差也未稳定，选择较简单机制。',
        '原视频区间是固定输出的描述性不确定性，不能消除反复开发或多重比较偏差。',
        '自动验收辅助字段（最终判断仍由根代理完成）：',json.dumps(acceptance,ensure_ascii=False,indent=2)]
    path=ROOT/'docs/TTA_RMOT_最终最小方案_跨域无标签验证与真实成本_20261122.txt'
    path.write_text('\n'.join(lines)+'\n');figure(runs,conditions)
    print(json.dumps(dict(event='FINAL_MINIMAL_PHYSICAL_ANALYSIS_COMPLETE',conditions=conditions,acceptance=acceptance,goal_status=result['goal_status'])),flush=True)


if __name__=='__main__':main()
