"""Physical10% paired streams: real quality, cost and replay differences."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_benchmark_scope import FRESH
from scripts.a100_visual_classical_budget_prefix import BUDGET
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_sparse_observation_analyze import effect

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'


def replay_audit(pair,root,receipts):
    result=dict(videos=0,PCD_max_abs_difference=0.,RGB_max_abs_difference=0.,
        prior_max_abs_difference=0.,question_symmetric_difference=0,common_question_answers=0,
        teacher_probability_max_abs_difference_on_common_questions=0.)
    for row in receipts:
        video=row['video'];folder=root/'source'/video
        with np.load(folder/'sparse_inputs.npz') as data:
            fresh={k:data[k] for k in ('pcd','colours','prior','tracks')}
        if pair=='C4':
            with np.load(FRESH/'source'/video/'sparse_inputs.npz') as data:
                old={k:data[k] for k in ('pcd','colours','prior','tracks')}
        else:
            with np.load(OUT/'inputs'/pair/video/'inputs.npz') as data:
                old=dict(pcd=data['features'],prior=data['prior'],tracks=data['tracks'])
            with np.load(OUT/'color_features'/pair/video/'features.npz') as data:
                old['colours']=data['colours']
        assert np.array_equal(fresh['tracks'],old['tracks'])
        for field,label in [('pcd','PCD'),('colours','RGB'),('prior','prior')]:
            assert fresh[field].shape==old[field].shape
            value=float(np.max(np.abs(fresh[field].astype(float)-old[field].astype(float))))
            result[label+'_max_abs_difference']=max(result[label+'_max_abs_difference'],value)
        old_answers=json.loads((BUDGET/'evidence/rho0.1'/pair/video/'answers.json').read_text())['answers']
        new_answers=json.loads((folder/'answers.json').read_text())['answers']
        convert=lambda rows:{(int(r['local_query']),int(r['local_track'])):float(r['yes_probability']) for r in rows}
        previous,current=convert(old_answers),convert(new_answers)
        common=set(previous)&set(current)
        result['question_symmetric_difference']+=len(set(previous)^set(current))
        result['common_question_answers']+=len(common)
        if common:
            result['teacher_probability_max_abs_difference_on_common_questions']=max(
                result['teacher_probability_max_abs_difference_on_common_questions'],
                max(abs(previous[k]-current[k]) for k in common))
        result['videos']+=1
    result['audit_only_not_a_requirement_to_copy_historical_predictions']=True
    result['C1_historical_full_frame_prior_may_differ_from_new_sparse_prior']=pair=='C1'
    return result


def figure(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    colours=['#969696','#72a8d4','#edaf68','#a0bf85','#b695c8','#cbbca2']
    for ax,(pair,row) in zip(axes,rows.items()):
        names=['CLIP_MEAN','CLIP_MEAN_COHORT']
        bottom=np.zeros(2)
        labels=['Startup','Student + image inputs','Teacher service/audit','KD fit','Output','Other common']
        for i,label in enumerate(labels):
            values=[row['cost_plot_components'][name][i] for name in names]
            ax.bar([0,1],values,bottom=bottom,color=colours[i],label=label)
            bottom+=values
        for i,name in enumerate(names):
            ax.text(i,bottom[i]+max(bottom)*.02,f"HOTA {row['raw_HOTA'][name]:.3f}",ha='center',fontsize=9)
        ax.set_xticks([0,1],['KD mean','KD mean + graph'])
        ax.set(title=pair+': actual fresh10% source/teacher',ylabel='Measured common + own component seconds')
        ax.set_ylim(0,max(bottom)*1.12)
        ax.grid(axis='y',alpha=.15)
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.15),ncol=3,fontsize=9)
    fig.suptitle('Physically measured shared two-branch streams; not independent cold-arm repeats\nOfficial supplied proposals/IDs; upstream detector/tracker generation excluded',fontsize=10)
    folder=PUBLIC/'figures'
    folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/'sparse_budget_real_cost.png',dpi=180,bbox_inches='tight')
    fig.savefig(folder/'sparse_budget_real_cost.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)


def main():
    summary,sufficient={},{}
    for pair in ('C4','C1'):
        root=OUT/'cold_full_c4'/f'{pair}_key_rho0p1_mean_pair_20261122'
        complete=json.loads((root/'completion.json').read_text())
        official=json.loads((root/'official_evaluation.json').read_text())
        assert official['status']=='COMPLETE_FRESH_BUDGET_PAIR'
        worker=json.loads((root/'worker_receipt.json').read_text())
        assert worker['exit_code']==0
        assert complete['GT_open_attempts']==complete['prepaid_artifact_reads']==complete['teacher']['cache_hits']==0
        assert complete['logical_teacher_calls']==complete['teacher']['new_calls']==(1160 if pair=='C4' else 1103)
        ledger=json.loads((root/'phase_ledger.json').read_text())
        receipts=[json.loads(p.read_text()) for p in sorted((root/'source').glob('*/receipt.json'))]
        assert len(receipts)==(45 if pair=='C4' else 25)
        stats={name:read_stats(root/'methods'/name) for name in ('CLIP_MEAN','CLIP_MEAN_COHORT')}
        fits=[r['receipt'] for r in ledger['head_fits']]
        fit_audit=dict(videos=len(fits),maximum_teacher_KL=max(r['teacher_KL'] for r in fits),
            maximum_steps=max(r['steps'] for r in fits),
            videos_exceeding_registered_KL_tolerance=sum(not r['teacher_fit_within_0_005'] for r in fits))
        for name,row in stats.items():
            sufficient[pair+'_seed0/'+name]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],
                numerator=row['parent_stats'][0].tolist(),denominator=row['parent_stats'][1].tolist())
        phases={}
        for row in ledger['common_phases']:
            phases[row['phase']]=phases.get(row['phase'],0)+row['seconds']
        startup=sum(phases.get(k,0) for k in ('imports_metadata_and_recovery_setup','student_load_and_integrity_check','teacher_load_and_initial_integrity_check'))
        perception=sum(phases.get(k,0) for k in ('real_sparse_frame_student_forward_and_initial_head','real_RGB_sparse_inputs_and_image_only_acquisition'))
        teacher=sum(phases.get(k,0) for k in ('real_teacher_service_including_image_IO','teacher_final_frozen_weight_audit'))
        components={}
        for name,cost in complete['cost_by_method'].items():
            remainder=cost['common_wall_seconds']-startup-perception-teacher
            assert remainder>=-.01, (pair,remainder)
            components[name]=[startup,perception,teacher,cost['own_head_training_seconds'],cost['own_postprocess_and_output_seconds'],max(remainder,0)]
        summary[pair+'_seed0']=dict(pair=pair,seed=0,raw_HOTA={name:row['HOTA'] for name,row in stats.items()},
            comparisons={'CLIP_MEAN_COHORT_vs_CLIP_MEAN':effect([stats['CLIP_MEAN_COHORT']],[stats['CLIP_MEAN']])},
            actual_two_branch_worker_receipt=worker, common_phase_seconds=phases,
            cost_by_method=complete['cost_by_method'],cost_plot_components=components,
            teacher_fit_audit=fit_audit,
            source=complete['source'],teacher=complete['teacher'],
            logical_teacher_calls=complete['logical_teacher_calls'],
            actual_decoder_frame_positions=sum(r['source_frames'] for r in receipts),
            full_frame_positions=sum(r['full_frames'] for r in receipts),
            actual_backbone_image_forwards=sum(r['actual_frozen_encoder_statistics']['backbone_images'] for r in receipts),
            source_sample_count_not_total_backbone_images=True,
            replay_input_evidence_audit=replay_audit(pair,root,receipts),
            fixed_candidate_boxes_and_IDs=True,target_GT_in_adaptation=False)
    result=dict(status='FRESH_SPARSE_BUDGET_PHYSICAL_ANALYZED',conditions=summary,repeated={},
        independent_single_arm_repeats=False,shared_source_teacher_and_one_head=True,
        all_development_data=True,new_GT_only_post_seal=True,goal_status='ACTIVE',
        cost_scope='Actual external two-branch Popen-to-exit and separately measured common+own components. Provided proposals/IDs; upstream detector/tracker generation excluded. Shared NAS/OS and concurrent jobs; no universal speed or Pareto claim.')
    atomic_json(PUBLIC/'budget_cold_analysis.json',result)
    atomic_json(OUT/'analysis/budget_cold_analysis.json',result)
    (PUBLIC/'budget_cold_parent_statistics.json').write_text(json.dumps(sufficient,separators=(',',':'))+'\n')
    from scripts.a100_visual_classical_sparse_observation_report import describe
    lines=['TTA-RMOT：10%预算真实源模型/教师前向与成本收据','2026-11-22','',
        'Goal仍ACTIVE。本报告使用这次真正新鲜的精度及时间，不把历史精度拼接到新时间。',
        '官方固定候选为共享输入；源主干/教师冻结，当前视频转导，过去+当前教师拟合小头，不使用目标人工标签。',
        '两分支共享一次真实源特征和教师流及同一个充分训练的软KD头；各自400/最终均值，图参数固定。',
        '外部实际进程包含两个分支；common+own只是实测组件分账，不冒充两个独立单方法冷启动。',
        '其它卡/CPU/NAS/OS缓存和任务共享，不能把墙钟偶然差异当算法加速。','']
    for name,row in summary.items():
        lines += [name,'HOTA：'+json.dumps(row['raw_HOTA']),
            '图相对均值KD：'+describe(row['comparisons']['CLIP_MEAN_COHORT_vs_CLIP_MEAN']),
            '真实双分支工作进程收据：'+json.dumps(row['actual_two_branch_worker_receipt'],ensure_ascii=False),
            '共同相位秒数：'+json.dumps(row['common_phase_seconds'],ensure_ascii=False),
            '实际组件分账：'+json.dumps(row['cost_by_method'],ensure_ascii=False),
            '蒸馏拟合审计：'+json.dumps(row['teacher_fit_audit'],ensure_ascii=False),
            '学生模型：'+json.dumps(row['source'],ensure_ascii=False),
            '教师新调用/命中/真实前向：'+json.dumps(row['teacher'],ensure_ascii=False),
            'decoder帧位置='+str(row['actual_decoder_frame_positions'])+'；真实backbone图像前向='+str(row['actual_backbone_image_forwards']),
            '相对历史回放的数值/问答差异：'+json.dumps(row['replay_input_evidence_audit'],ensure_ascii=False),'']
    lines += ['不能声称从原视频检测开始的全系统成本：官方前端候选生成未测。',
        '单次新鲜seed0的原视频区间是描述性结果；三神经种子质量证据与成本分别呈现，没有独立新数据验证。',
        '若新鲜表现与历史不同，应检查上面列出的源先验、PCD和选题差异，不以回填旧预测使数字好看。']
    path=ROOT/'docs/TTA_RMOT_10百分比预算_真实前向与成本收据_20261122.txt'
    path.write_text('\n'.join(lines)+'\n')
    figure({k.split('_',1)[0]:v for k,v in summary.items()})
    print(json.dumps(dict(event='FRESH_BUDGET_PHYSICAL_ANALYSIS_COMPLETE',results={k:v['raw_HOTA'] for k,v in summary.items()},goal_status='ACTIVE')),flush=True)


if __name__=='__main__':
    main()
