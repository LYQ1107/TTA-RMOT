"""Whole-C4 physically measured component costs; fixed-QA quality stays separate."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT


def main():
    directory=ROOT/'docs/a100_visual_classical_20261121/figures';directory.mkdir(exist_ok=True)
    runs={};summaries={}
    for label,name in [('Full perception','C4_fresh_full_stream_seed0'),('Key observations','C4_fresh_track_three_seed0')]:
        root=OUT/'cold_full_c4'/name
        receipt=json.loads((root/'completion.json').read_text());evaluation=json.loads((root/'official_evaluation.json').read_text())
        process=json.loads((root/'worker_process_receipt.json').read_text())
        phases=json.loads((root/'phase_ledger.json').read_text())['common_phases']
        seconds={}
        for p in phases:seconds[p['phase']]=seconds.get(p['phase'],0.)+p['seconds']
        components={
            'Student load':seconds['student_load_and_integrity_check'],
            'Teacher load + audit':seconds['teacher_load_and_initial_integrity_check']+seconds['teacher_final_frozen_weight_audit'],
            'Student forward + initial scores':sum(v for k,v in seconds.items() if k in ('real_all_frame_student_forward_and_initial_head','real_sparse_frame_student_forward_and_initial_head')),
            'RGB + scheduling':seconds['real_RGB_sparse_inputs_and_image_only_acquisition'],
            'Teacher service + I/O':seconds['real_teacher_service_including_image_IO']}
        common=receipt['cost_by_method']['RAW_KD']['common_wall_seconds']
        components['Other common work']=common-sum(components.values())
        assert components['Other common work']>=-.01
        runs[label]=dict(receipt=receipt,evaluation=evaluation,components=components)
        summaries[label]=dict(components_seconds=components,cost_by_method=receipt['cost_by_method'],HOTA=evaluation['HOTA'],
            actual_teacher_new_calls=receipt['teacher']['new_calls'],cache_hits=receipt['teacher']['cache_hits'],
            single_uninterrupted_attempt=receipt['single_uninterrupted_attempt'],generation_GT_open_attempts=receipt['GT_open_attempts'],
            actual_all_arm_experiment_worker_seconds=receipt['actual_multiarm_worker_wall_seconds'],
            external_all_arm_process_seconds=process['wall_seconds'],
            external_process_envelope_seconds=process['wall_seconds']-receipt['actual_multiarm_worker_wall_seconds'])
    full=summaries['Full perception']['components_seconds']['Student forward + initial scores']
    sparse=summaries['Key observations']['components_seconds']['Student forward + initial scores']
    atomic_json(OUT/'analysis/physical_full_vs_key_C4.json',dict(runs=summaries,measured_student_forward_ratio=full/sparse,
        official_candidates_provided_upstream_generation_not_remeasured=True,common_prefix_measured_once_per_condition=True,
        branch_cost_is_component_accounting_not_independent_cold_deployments=True,
        different_A100_GPUs_concurrent_host_NAS_and_OS_cache_retained=True))
    bars=[(condition,method) for condition in runs for method in ('RAW_KD','RAW_COHORT')]
    fig,ax=plt.subplots(figsize=(11,5.3));left=np.zeros(4);colors=plt.cm.tab10(np.arange(8))
    labels=list(next(iter(runs.values()))['components'])+['Actual KD head fit','Own graph/output']
    for index,label in enumerate(labels):
        values=[]
        for condition,method in bars:
            row=runs[condition];cost=row['receipt']['cost_by_method'][method]
            values.append((row['components'][label] if label in row['components'] else cost['own_head_training_seconds'] if label=='Actual KD head fit' else cost['own_postprocess_and_output_seconds'])/60)
        ax.barh(np.arange(4),values,left=left,label=label,color=colors[index]);left+=values
    ax.set_yticks(np.arange(4),[condition+'\n'+method for condition,method in bars]);ax.invert_yaxis()
    ax.set_xlim(0,left.max()*1.22);ax.set_xlabel('Measured common work + own branch (minutes)')
    ax.set_title('Whole C4: actual source/teacher execution, same-evidence KD and graph')
    for y,(condition,method) in enumerate(bars):
        ax.text(left[y]+left.max()*.012,y,f'{left[y]:.2f} min\nHOTA {runs[condition]["evaluation"]["HOTA"][method]:.3f}',va='center',fontsize=9)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=4,fontsize=8,frameon=False)
    fig.tight_layout(rect=(0,.16,1,1));fig.text(.5,.025,'Given frozen candidates. Shared prefix once per condition; own branch measured. Different A100 GPUs, concurrent host/NAS, OS cache retained.',ha='center',fontsize=8)
    for suffix in ['png','pdf']:fig.savefig(directory/('whole_C4_physical_cost.'+suffix),dpi=190)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5.3))
    for condition,marker in [('Full perception','o'),('Key observations','s')]:
        row=runs[condition]
        for method,value in row['evaluation']['HOTA'].items():
            seconds=row['receipt']['cost_by_method'][method]['shared_common_plus_own_branch_seconds']
            color='#168b83' if 'COHORT' in method else '#c48b3a' if 'NO_GRAPH' in method or 'STRONG_PRIOR' in method else '#8092a5'
            ax.scatter(seconds/60,value,marker=marker,color=color,s=35,alpha=.8)
        method='RAW_COHORT';seconds=row['receipt']['cost_by_method'][method]['shared_common_plus_own_branch_seconds']
        ax.annotate(condition+' / RAW_COHORT',(seconds/60,row['evaluation']['HOTA'][method]),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8)
    ax.set_xlabel('Measured shared prefix + own branch (minutes)');ax.set_ylabel('Official HOTA')
    ax.set_title('All measured configurations; no global Pareto-optimality claim');ax.grid(alpha=.15)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([],[],marker='o',linestyle='',color='#555',label='Full perception'),Line2D([],[],marker='s',linestyle='',color='#555',label='Key observations'),Line2D([],[],marker='o',linestyle='',color='#168b83',label='Cohort graph'),Line2D([],[],marker='o',linestyle='',color='#8092a5',label='Actual KD / mean controls')],fontsize=8)
    fig.tight_layout(rect=(0,.055,1,1));fig.text(.5,.015,'One development-data run per condition; identical evidence within condition. QA selection can differ between conditions.',ha='center',fontsize=8)
    for suffix in ['png','pdf']:fig.savefig(directory/('whole_C4_cost_utility.'+suffix),dpi=190)
    plt.close(fig)
    replay=json.loads((OUT/'analysis/fresh_sparse_fixedQA.json').read_text());fig,ax=plt.subplots(figsize=(9,5))
    methods=['RAW_KD_400','RAW_KD','RAW_COHORT','CLIP_KD_400','CLIP_KD','CLIP_COHORT','HARD_KD_400','HARD_KD']
    for y,method in enumerate(methods):
        row=replay['comparisons'][method+'_sparse_source_vs_full_FIXED_QA'];point=row['delta_HOTA'];lo,hi=row['original_parent_video']['CI95']
        ax.errorbar(point,y,xerr=[[point-lo],[hi-point]],fmt='o',capsize=4,color='#168b83' if 'COHORT' in method else '#8092a5')
    ax.axvline(0,linestyle='--',color='#555');ax.set_yticks(range(len(methods)),methods);ax.invert_yaxis()
    ax.set_xlabel('Sparse source prior/gate minus full, with identical QA and reference features')
    ax.set_title('Sampling quality intervention | 13-parent descriptive 95% intervals');ax.grid(axis='x',alpha=.15)
    fig.tight_layout(rect=(0,.055,1,1));fig.text(.5,.015,'Current-video transductive, seed0, development data. Fixed-QA replay is a quality diagnostic, not a cold timing run.',ha='center',fontsize=8)
    for suffix in ['png','pdf']:fig.savefig(directory/('whole_C4_sparse_fixed_QA.'+suffix),dpi=190)
    plt.close(fig);print(json.dumps(dict(status='WHOLE_C4_PHYSICAL_FIGURES_EXPORTED',directory=str(directory))))


if __name__=='__main__':main()
