"""Export mechanism/cost figures without mixing dataset or timing scopes."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.a100_visual_classical_prepare import OUT


def main():
    data=json.loads((OUT/'analysis/strong_prefix_seed0.json').read_text())
    exclusive=json.loads((OUT/'analysis/exclusive_counts_seed0.json').read_text())
    directory=ROOT/'docs/a100_visual_classical_20261121/figures';directory.mkdir(exist_ok=True)
    names=['Soft KD, 400 steps','Hard KD, 400 steps','Soft KD, fully fitted','Hard KD, fully fitted','Same soft KD + cohort graph']
    methods=['SOFT_KD_400','HARD_KD_400','SOFT_KD','HARD_KD','COHORT']
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for ax,pair in zip(axes,['C4','C1']):
        values=[data[pair]['HOTA']['strong_cover_seed0/'+name] for name in methods]
        ax.barh(np.arange(5),values,color=['#728496']*4+['#168b83'])
        ax.set_yticks(np.arange(5),names);ax.invert_yaxis();ax.set_xlim(0,76 if pair=='C4' else 47)
        ax.set_title(pair+' | same questions, features and source');ax.set_xlabel('Official HOTA')
        ax.grid(axis='x',alpha=.15)
        for y,value in enumerate(values):ax.text(value+.2,y,f'{value:.3f}',va='center',fontsize=9)
    fig.suptitle('Actual trained distillation controls | seed 0, development data')
    fig.tight_layout(rect=(0,.04,1,.95));fig.text(.5,.01,'C1 hard-KD 400-step baseline exceeds the proposed graph: retain this result.',ha='center',fontsize=9)
    for suffix in ['png','pdf']:fig.savefig(directory/('strong_same_evidence_seed0.'+suffix),dpi=190)
    plt.close(fig)
    c4=data['C4']['comparisons']['strong_cover_seed0/COUNT_FULL_vs_SOFT_KD'];c1=exclusive['comparisons']['COUNT_FULL_vs_SOFT_KD']
    fig,ax=plt.subplots(figsize=(7,3.6))
    for yi,(pair,value) in enumerate([('C4',c4),('C1, exclusive assignment',c1)]):
        point=value['delta_HOTA'];low,high=value['original_parent_video']['CI95']
        ax.errorbar(point,yi,xerr=[[point-low],[high-point]],fmt='o',capsize=5,color='#168b83' if yi==0 else '#728496')
        ax.text(high+.04,yi,f'{point:+.3f} [{low:+.3f}, {high:+.3f}]',va='center',fontsize=9)
    ax.axvline(0,color='#404040',linestyle='--',linewidth=1);ax.set_yticks([0,1],['C4 (13 parent videos)','C1 (25 videos)'])
    ax.invert_yaxis();ax.set_ylim(1.65,-.65);ax.set_xlim(-1,2.3);ax.set_xlabel('HOTA difference at identical per-query/frame output counts')
    ax.set_title('Cohort graph vs fully fitted soft KD | fixed teacher answers')
    fig.tight_layout(rect=(0,.08,1,1));fig.text(.5,.02,'Descriptive 95% parent bootstrap, seed 0; no independent blind test.',ha='center',fontsize=8)
    for suffix in ['png','pdf']:fig.savefig(directory/('countmatched_ranking_seed0.'+suffix),dpi=190)
    plt.close(fig)
    cold=json.loads((OUT/'cold_live/C4_first_full_clip_seed0/completion.json').read_text())
    evaluation=json.loads((OUT/'cold_live/C4_first_full_clip_seed0/official_evaluation.json').read_text())
    phases=cold['phases'];parts=[('Student load',phases['student_load_including_weight_verification']),
        ('Student forward + I/O',phases['all_source_forward_text_and_initial_head']),
        ('RGB + acquisition',phases['real_RGB_part_features']+phases['image_only_input_assembly_and_acquisition']),
        ('Teacher load',phases['teacher_load_including_weight_verification']),
        ('Teacher service + I/O',phases['real_teacher_service_including_image_IO']),
        ('Frozen weight audit',phases['teacher_final_frozen_weight_audit']),
        ('Other startup',cold['common_wall_seconds']-sum(phases.values()))]
    fig,ax=plt.subplots(figsize=(10,4.5));colors=plt.cm.tab10(np.linspace(0,1,10));bottom=np.zeros(2)
    for (label,seconds),color in zip(parts,colors):
        value=np.full(2,seconds/60);ax.barh([0,1],value,left=bottom,label=label,color=color);bottom+=value
    for key,label,color in [('train_seconds','Actual head fit',colors[7]),('postprocess_including_output_seal_seconds','Postprocess + output',colors[8])]:
        value=np.asarray([cold['arms'][a][key]/60 for a in ['SOFT_KD','COHORT']]);ax.barh([0,1],value,left=bottom,label=label,color=color);bottom+=value
    ax.set_yticks([0,1],['Soft KD','Same KD + cohort']);ax.invert_yaxis();ax.set_xlabel('Measured common cold prefix + own branch (minutes)')
    ax.set_xlim(0,27);ax.set_title('One full fixed C4 clip: 1162 frames, 8 queries, 43 real Qwen calls, 0 cache hits')
    for y,a in enumerate(['SOFT_KD','COHORT']):ax.text(bottom[y]+.2,y,f'{bottom[y]:.2f} min\nHOTA {evaluation["results"][a]["HOTA_unrounded"]:.3f}',va='center',fontsize=9)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=3,fontsize=8,frameon=False)
    fig.tight_layout(rect=(0,.12,1,1));fig.text(.5,.015,'Shared prefix physically measured once; host shared, OS cache retained. Not whole-C4 timing or a Pareto-optimality claim.',ha='center',fontsize=8)
    for suffix in ['png','pdf']:fig.savefig(directory/('real_cold_full_clip_cost.'+suffix),dpi=190)
    plt.close(fig)
    print(json.dumps(dict(status='FIGURES_EXPORTED',directory=str(directory),scope='seed0 development mechanism + one physical cold clip; not goal completion')))


if __name__=='__main__':main()
