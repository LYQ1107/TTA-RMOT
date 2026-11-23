"""Actual independent cold costs and quality, never copied from old timing."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_rgb_parity import source_spec,PARITY
from scripts.a100_visual_classical_joint_visual import JOINT
from scripts.a100_visual_classical_appearance_cold_launch import PLAN
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_sparse_observation_analyze import effect
from scripts.a100_visual_classical_sparse_observation_report import describe
from scripts.a100_visual_classical_sparse_statistics_reproduce import share_identical_statistics

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'
SOURCE_SHA={'C4':'fbb5df7339a1201f34300c105df81e893a5edbb8ef939b7f6eaf8780348eac6c',
    'C1':'71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613'}
TEACHER_SHA='9a13380b4279bc98f4680bd083ff4995fdcfec120cfded459513be90d041a956'


def read_run(job):
    root=OUT/'cold_full_c4'/job['run_name'];complete=json.loads((root/'completion.json').read_text())
    worker=json.loads((root/'worker_receipt.json').read_text());ledger=json.loads((root/'phase_ledger.json').read_text())
    assert worker['exit_code']==0 and worker['independent_single_arm_processes'] and worker['physical_GPU']==job['physical_GPU']
    assert complete['single_method_execution'] and complete['single_uninterrupted_attempt']
    assert complete['GT_open_attempts']==complete['prepaid_artifact_reads']==complete['teacher']['cache_hits']==0
    assert complete['logical_teacher_calls']==complete['teacher']['new_calls']==job['expected_questions']
    assert complete['source']['checkpoint_sha256']==SOURCE_SHA[job['pair']]
    assert complete['teacher']['teacher_weight_sha256']==TEACHER_SHA and complete['teacher']['backend_unmodified']
    assert complete['protocol']['head_query_scope']==complete['protocol']['query_graph_scope']=='extended'
    assert complete['protocol']['appearance_regularization']==job['methods'][0].endswith('_CAL')
    assert complete['protocol']['graph_prior']=='constant' and complete['protocol']['set_decoder']=='original'
    fits=[r['receipt'] for r in ledger['head_fits']];phases={}
    for r in ledger['common_phases']:phases[r['phase']]=phases.get(r['phase'],0.)+r['seconds']
    outputs=ledger['method_outputs'];appearance=[r['appearance_regularization'] for r in outputs if r.get('appearance_regularization')]
    receipts=[json.loads(p.read_text()) for p in sorted((root/'source').glob('*/receipt.json'))]
    return dict(root=str(root),method=job['methods'][0],protocol=complete['protocol'],worker=worker,source=complete['source'],
        teacher=complete['teacher'],cost_by_method=complete['cost_by_method'],phase_seconds=phases,
        teacher_fit_audit=dict(fitted_videos=len(fits),maximum_teacher_KL=max(r['teacher_KL'] for r in fits),
            maximum_steps=max(r['steps'] for r in fits),videos_exceeding_registered_KL_tolerance=sum(not r['teacher_fit_within_0_005'] for r in fits)),
        actual_head_fit_and_save_seconds=sum(r['measured_fit_and_head_save_seconds'] for r in fits),
        actual_all_method_postprocess_and_output_seconds=sum(r['seconds'] for r in outputs),
        appearance_regularization_CPU_seconds=sum(r['actual_CPU_seconds'] for r in appearance),
        appearance_active_nodes=sum(r['active_nodes'] for r in appearance),
        decoder_frames=sum(r['source_frames'] for r in receipts),
        backbone_image_forwards=sum(r['actual_frozen_encoder_statistics']['backbone_images'] for r in receipts),
        provided_official_proposals_boundary=True,upstream_detector_and_tracker_generation_timed=False)


def input_parity(left,right,pair,heads=False):
    """Post-seal actual evidence comparison; no GT or fitted decision used."""
    result=dict(videos=0,PCD_max_abs_difference=0.,RGB_max_abs_difference=0.,prior_max_abs_difference=0.,
        anchor_changed_tracks=0,ordered_question_list_changed_videos=0,question_symmetric_difference=0,
        common_answer_positions=0,teacher_probability_max_abs_difference=0.)
    paths=sorted((left/'source').glob('*/sparse_inputs.npz'));assert len(paths)==(45 if pair=='C4' else 25)
    for path in paths:
        video=path.parent.name
        with np.load(path) as a,np.load(right/'source'/video/'sparse_inputs.npz') as b:
            assert np.array_equal(a['tracks'],b['tracks'])
            for field,label in [('pcd','PCD'),('colours','RGB'),('prior','prior')]:
                assert a[field].shape==b[field].shape
                result[label+'_max_abs_difference']=max(result[label+'_max_abs_difference'],float(np.max(np.abs(a[field].astype(float)-b[field].astype(float)))))
            result['anchor_changed_tracks']+=int(np.count_nonzero(a['anchors']!=b['anchors']))
        requests=[];answers=[]
        for root in (left,right):
            q=json.loads((root/'source'/video/'questions_before_answers.json').read_text())['requests']
            requests.append([json.dumps(r,sort_keys=True) for r in q])
            returned=json.loads((root/'source'/video/'answers.json').read_text())['answers']
            answers.append({(int(r['local_query']),int(r['local_track'])):float(r['yes_probability']) for r in returned})
        result['ordered_question_list_changed_videos']+=requests[0]!=requests[1]
        result['question_symmetric_difference']+=len(set(requests[0])^set(requests[1]))
        common=set(answers[0])&set(answers[1]);result['common_answer_positions']+=len(common)
        if common:result['teacher_probability_max_abs_difference']=max(result['teacher_probability_max_abs_difference'],max(abs(answers[0][k]-answers[1][k]) for k in common))
        result['videos']+=1
    keys=('PCD_max_abs_difference','RGB_max_abs_difference','prior_max_abs_difference','anchor_changed_tracks',
        'ordered_question_list_changed_videos','question_symmetric_difference','teacher_probability_max_abs_difference')
    result['exact_source_questions_and_teacher_answers_equal']=not any(result[k] for k in keys)
    if heads:
        import torch
        a,b=[torch.load(root/'CLIP_latest_head.pt',map_location='cpu',weights_only=False) for root in (left,right)]
        result['last_video_head_weight_max_abs_difference']=max(float((a['model'][k]-b['model'][k]).abs().max()) for k in a['model'])
        result['last_video_head_statistic_max_abs_difference']=max(float(np.max(np.abs(a[k]-b[k]))) for k in ('mean','std'))
        result['head_weight_audit_is_last_video_only']=True
    return result


def prediction_parity(actual,expected,pair,arm,source):
    values=[]
    for root in (actual,expected):
        seal=json.loads((root/'predictions_seal.json').read_text())
        values.append({(r['video'],r['expression']):r['prediction_sha256'] for r in seal['episodes']})
    assert values[0].keys()==values[1].keys()
    from flex_uda.a100.mechanism import metadata,PAIRS
    m=metadata(PAIRS[pair][1]);maximum=0.
    for video,bank in sorted(m['videos'].items()):
        score=np.load(actual/'videos'/video/'scores.npy',mmap_mode='r')
        if arm=='KD':reference=np.load(expected/'videos'/video/'scores.npy',mmap_mode='r')
        else:
            with np.load(source/'source'/video/'sparse_inputs.npz') as a:tracks=a['tracks']
            lookup={int(t):i for i,t in enumerate(tracks)};columns=np.asarray([lookup[int(t)] for _,t in bank['observations']])
            with np.load(expected.parent/'videos'/video/'coarse_scores.npz') as a:reference=a['JOINT'][:,columns]
        maximum=max(maximum,float(np.max(np.abs(score.astype(float)-reference.astype(float)))))
    return dict(queries=len(values[0]),changed_query_prediction_files=sum(values[0][k]!=values[1][k] for k in values[0]),
        maximum_saved_score_probability_error=maximum,
        no_GT_in_comparison=True,not_an_independent_neural_seed=True)


def postseal_diagnosis(jobs):
    from collections import Counter
    from flex_uda.a100.mechanism import OUT as DIAG,metadata,PAIRS
    result={}
    for pair in ('C4','C1'):
        m=metadata(PAIRS[pair][1]);rows={n:Counter() for n in ('KD','JOINT')};trade=Counter()
        for video in sorted(m['videos']):
            truth=np.load(DIAG/'offline_GT'/pair/video/'labels.npy')>=0;present=truth.any(1);correct={}
            for arm in rows:
                job=jobs[pair+'_'+arm];run_root=OUT/'cold_full_c4'/job['run_name'];root=run_root/'methods'/job['methods'][0]
                mask=np.load(root/'videos'/video/'mask.npy');correct[arm]=(mask&truth).any(1)
                with np.load(run_root/'source'/video/'sparse_inputs.npz') as a:tracks=a['tracks']
                lookup={int(t):i for i,t in enumerate(tracks)};columns=np.asarray([lookup[int(t)] for _,t in m['videos'][video]['observations']])
                known=np.zeros((len(mask),len(tracks)),bool)
                for answer in json.loads((run_root/'source'/video/'answers.json').read_text())['answers']:
                    known[int(answer['local_query']),int(answer['local_track'])]=True
                asked=known[:,columns];uncovered=present&~(truth&asked).any(1)
                rows[arm].update(queries=len(mask),recoverable_positive_queries=int(present.sum()),
                    positive_queries_empty=int((present&~mask.any(1)).sum()),positive_queries_without_correct_output=int((present&~correct[arm]).sum()),
                    no_positive_candidate_queries_with_output=int((~present&mask.any(1)).sum()),
                    TP=int((mask&truth).sum()),FP=int((mask&~truth).sum()),FN=int((~mask&truth).sum()),
                    queries_without_any_correct_whole_track_asked=int(uncovered.sum()),
                    recovered_queries_without_correct_whole_track_asked=int((uncovered&correct[arm]).sum()),
                    true_positive_rows_on_unasked_tracks=int((mask&truth&~asked).sum()))
            trade.update(rescued=int((present&~correct['KD']&correct['JOINT']).sum()),regressed=int((present&correct['KD']&~correct['JOINT']).sum()))
        result[pair]=dict(candidate_membership={n:dict(r) for n,r in rows.items()},query_retrieval_tradeoff=dict(trade))
    return dict(pairs=result,GT_only_after_all_four_actual_seals=True,not_additive_HOTA_attribution=True,
        positive_query_means_recoverable_candidate_not_all_physical_GT=True,
        whole_track_asked_coverage_is_not_anchor_frame_teacher_accuracy=True)


def figure(runs,conditions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
    for ax,pair in zip(axes,('C4','C1')):
        xs=[runs[pair+'_'+arm]['worker']['actual_worker_process_wall_seconds'] for arm in ('KD','JOINT')]
        ys=[conditions[pair+'_seed0']['raw_HOTA'][arm] for arm in ('KD','JOINT')]
        ax.bar(['KD','JOINT'],xs,color=['tab:blue','tab:orange'],width=.6)
        for i,(x,y) in enumerate(zip(xs,ys)):
            ax.text(i,x+max(xs)*.025,f'{x:.1f}s\nHOTA {y:.3f}',ha='center',va='bottom',fontsize=10)
        ax.set(title=pair+': actual seed0 cost and quality',ylabel='Independent worker Popen-to-exit seconds',ylim=(0,max(xs)*1.20))
        ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Four actual independent single-method cold processes; all DEV\nAll startup, source/teacher inference, head fitting and output included; GT evaluation excluded\nShared NAS/OS load; no algorithmic speedup or all-budget Pareto optimality inferred',fontsize=9)
    dest=PUBLIC/'figures';dest.mkdir(exist_ok=True)
    preview=OUT/'analysis/publication_figure_previews';preview.mkdir(parents=True,exist_ok=True)
    fig.savefig(preview/'appearance_cold.png',dpi=160,bbox_inches='tight')
    fig.savefig(dest/'appearance_cold.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)


def main():
    official=json.loads((OUT/'analysis/appearance_cold_official.json').read_text());assert official['all_four_predictions_sealed_before_GT']
    plan=json.loads(PLAN.read_text());jobs=plan['jobs'];runs={key:read_run(job) for key,job in jobs.items()}
    data={};conditions={};statistics={};pair_audits={};history_audits={};prediction_audits={}
    for pair in ('C4','C1'):
        key=pair+'_seed0';data[key]={}
        left=Path(runs[pair+'_KD']['root']);right=Path(runs[pair+'_JOINT']['root'])
        pair_audits[pair]=input_parity(left,right,pair,heads=True)
        history_audits[pair]=input_parity(left,source_spec(pair)[0],pair)
        for arm in ('KD','JOINT'):
            job=jobs[pair+'_'+arm];root=Path(runs[pair+'_'+arm]['root'])/'methods'/job['methods'][0]
            expected=PARITY/pair/'seed0/RGB_KD_MEAN' if arm=='KD' else JOINT/pair/'seed0/JOINT'
            prediction_audits[pair+'_'+arm]=prediction_parity(root,expected,pair,arm,Path(runs[pair+'_'+arm]['root']))
            row=read_stats(root);data[key][arm]=row
            statistics[key+'/'+arm]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],numerator=row['parent_stats'][0].tolist(),denominator=row['parent_stats'][1].tolist())
            atomic_json(Path(runs[pair+'_'+arm]['root'])/'official_evaluation.json',dict(status='COMPLETE_FRESH_BUDGET_SINGLE',
                methods={job['methods'][0]:official['results'][str(root)]},all_four_sealed_before_GT=True,
                actual_worker_receipt=runs[pair+'_'+arm]['worker'],no_target_GT_in_adaptation=True,goal_status='ACTIVE'))
        conditions[key]=dict(raw_HOTA={n:r['HOTA'] for n,r in data[key].items()},comparisons=dict(JOINT_vs_KD=effect([data[key]['JOINT']],[data[key]['KD']])))
    quality=json.loads((PUBLIC/'joint_visual_analysis.json').read_text())
    errors=postseal_diagnosis(jobs)
    acceptance=dict(all_four_independent_actual_processes=True,all_four_sealed_before_any_group_GT=True,
        actual_teacher_calls=sum(r['teacher']['new_calls'] for r in runs.values()),actual_cache_hits=sum(r['teacher']['cache_hits'] for r in runs.values()),
        paired_actual_evidence_equal=all(r['exact_source_questions_and_teacher_answers_equal'] for r in pair_audits.values()),
        old_quality_source_equal=all(r['exact_source_questions_and_teacher_answers_equal'] for r in history_audits.values()),
        actual_seed0_predictions_equal=not any(r['changed_query_prediction_files'] for r in prediction_audits.values()),
        actual_seed0_score_probabilities_equal=not any(r['maximum_saved_score_probability_error'] for r in prediction_audits.values()),
        all_heads_meet_registered_teacher_KL=not any(r['teacher_fit_audit']['videos_exceeding_registered_KL_tolerance'] for r in runs.values()),
        actual_point_gains_positive={p:r['comparisons']['JOINT_vs_KD']['delta_HOTA']>0 for p,r in conditions.items()},
        actual_parent_CI_positive={p:r['comparisons']['JOINT_vs_KD']['fixed_neural_seeds_parent_CI95'][0]>0 for p,r in conditions.items()},
        previous_three_seed_primary_intervals_positive=all(r['comparisons']['JOINT_vs_KD']['parent_and_neural_seed_sensitivity_CI95'][0]>0 for r in quality['repeated'].values()),
        C1_all_strong_KD_intervals_positive=False,C1_universal_semantic_ranking_claim=False,C4_no_query_retrieval_regression_claim=False,
        independent_blind_validation=False)
    result=dict(status='FOUR_ACTUAL_APPEARANCE_COLD_ANALYZED_ROOT_REVIEW_REQUIRED',conditions=conditions,repeated={},runs=runs,
        actual_within_pair_evidence_audit=pair_audits,quality_source_evidence_audit=history_audits,expected_seed0_prediction_audit=prediction_audits,
        postseal_candidate_membership_diagnosis=errors,acceptance=acceptance,
        need_matched_followup_if_evidence_differs=not acceptance['paired_actual_evidence_equal'] or not acceptance['old_quality_source_equal'] or not acceptance['actual_seed0_predictions_equal'] or not acceptance['actual_seed0_score_probabilities_equal'],
        all_development_data=True,no_target_manual_labels_in_adaptation=True,
        supplied_official_frontend_is_shared=True,old_3512_second_cost_not_reused=True,
        actual_reexecution_not_an_additional_independent_neural_seed=True,
        conventional_direct_classifier_fairness_requires_separate_nores_study=True,
        uncertainty='Fixed sealed actual seed0 parent-bootstrap only;3seed mechanism statistics remain separate; no development/multiplicity correction or pipeline refit.',
        scope='Actual independent per-worker Popen-to-exit including all model loads, true forwards, calibration/fitting/output; GT excluded. Shared NAS/OS and other host jobs preclude a timing speedup claim.',
        goal_status='ACTIVE_REQUIRES_ROOT_SCIENTIFIC_AND_PUBLICATION_REVIEW')
    atomic_json(PUBLIC/'appearance_cold_analysis.json',result);atomic_json(OUT/'analysis/appearance_cold_analysis.json',result)
    compact,_=share_identical_statistics(statistics,'appearance_cold_parent_statistics.json')
    (PUBLIC/'appearance_cold_parent_statistics.json').write_text(json.dumps(compact,separators=(',',':'))+'\n')
    lines=['TTA-RMOT：图与外观校准的四卡独立真实冷启动验收','2026-11-22','',
        '四个独立单方法进程：C4 KD/JOINT，C1 KD/JOINT；GPU0/1/2/3。各重载模型、重算图像、独立新SQLite。',
        '四份预测全部封存后才统一GT。适配只用目标图像/任务查询/冻结教师，不读人工目标标签。',
        '主模型RGB1080充分软KD的400/最终均值，JOINT增加已有视觉图+固定PAVA；无LoRA、正信念保护或强制非空。',
        '本文精度和时间均来自这次真实运行，不拼接旧3512秒。已有三种子机制证据在joint_visual_analysis.json中单独报告。','']
    for pair,row in conditions.items():
        lines += [pair,'真实本次HOTA：'+json.dumps(row['raw_HOTA']),'真实本次差：'+describe(row['comparisons']['JOINT_vs_KD'])]
    for name,row in runs.items():
        lines += ['',name,'实际外部进程秒：'+str(row['worker']['actual_worker_process_wall_seconds']),
            '阶段秒：'+json.dumps(row['phase_seconds']),'头拟合/保存秒：'+str(row['actual_head_fit_and_save_seconds']),
            '实际后处理/输出秒：'+str(row['actual_all_method_postprocess_and_output_seconds']),
            '其中外观PAVA实测CPU秒：'+str(row['appearance_regularization_CPU_seconds']),
            '教师收据：'+json.dumps(row['teacher']),'拟合收据：'+json.dumps(row['teacher_fit_audit']),
            'decoder帧数='+str(row['decoder_frames'])+'，实际backbone图像前向='+str(row['backbone_image_forwards'])]
    lines += ['', '两独立方法之间的实际输入/问答：',json.dumps(pair_audits,indent=2),
        '与前述质量研究输入的比较：',json.dumps(history_audits,indent=2),
        '与预先固定seed0预测的字节比较：',json.dumps(prediction_audits,indent=2),
        '封存后查询与候选IoU成员诊断（不是HOTA可加分账）：',json.dumps(errors,ensure_ascii=False,indent=2),
        '验收辅助字段：',json.dumps(acceptance,indent=2),
        '科学边界：C1主要数量/查询基率校准，固定KD数量排序为负；部分C1强硬基线区间跨0，不声称全部显著。',
        'C4总体HOTA提高仍伴随部分短查询漏找，不能宣称所有错误已解决。所有数据已用于开发，未做独立盲测。',
        '并行A100共享NAS/OS/主机负载；如耗时不同，不据此声称算法加速或所有计算预算Pareto最优。',
        '前端固定官方候选的生成不在本成本边界。真实重复不是第4个独立神经种子。',
        '本四卡实际比较仍为固定源logit残差KD；无源残差的标准直接教师分类头需另行公平性核验，不能仅据本报告泛称胜过全部普通蒸馏。',
        '完成本分析不会自动关闭Goal，根代理还需核对实际证据、最终报告、轻量发布与验收条件。']
    (ROOT/'docs/TTA_RMOT_图与外观校准_四卡独立真实冷启动_20261122.txt').write_text('\n'.join(lines)+'\n')
    figure(runs,conditions)
    print(json.dumps(dict(event='APPEARANCE_COLD_ANALYSIS_COMPLETE',conditions=conditions,acceptance=acceptance,goal_status='ACTIVE')),flush=True)


if __name__=='__main__':main()
