"""Plot the completed ordinary-learner I1 benchmark, not unified-model wins."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    public=ROOT/'docs/a100_six_direction_20261122'
    data=json.loads((public/'observation_policy_analysis.json').read_text())
    check=json.loads((public/'observation_policy_statistical_reproduction.json').read_text())
    assert check['HOTA_records']==828 and data['primary']=='DESIGN_MEAN'
    pairs=('C1','C2','C3','C4','C5','C6')
    labels=('V2 → Dance','Dance → KITTI','V2 → Sports','Dance → Sports','LaMOT → KITTI','LaMOT → Dance')
    policies=('DESIGN','COVERAGE_ONLY','PRIOR','UNIFORM')
    colors=('#326a9d','#78a4c8','#c8a463','#747474')
    with plt.rc_context({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
        'savefig.facecolor':'white','svg.fonttype':'none','pdf.fonttype':42}):
        fig,axes=plt.subplots(1,2,figsize=(11.4,4.2),gridspec_kw={'width_ratios':(1.45,1)})
        y=np.arange(6);height=.18
        for i,(policy,color) in enumerate(zip(policies,colors)):
            values=[data['repeated'][p]['mean_HOTA'][policy+'_MEAN'] for p in pairs]
            axes[0].barh(y+(i-1.5)*height,values,height=height,color=color,label=policy.replace('_',' '))
        axes[0].set(yticks=y,yticklabels=labels,xlabel='HOTA (three neural seeds)',xlim=(0,75),
            title='Same ordinary KD learner; different observations')
        axes[0].invert_yaxis();axes[0].grid(axis='x',alpha=.15);axes[0].set_axisbelow(True)
        axes[0].legend(ncol=2,frameon=False,loc='lower right',fontsize=8)
        for index,pair in enumerate(pairs):
            contrast=data['repeated'][pair]['comparisons']['PRIMARY_vs_UNIFORM_MEAN']
            point=contrast['delta_HOTA'];ci=contrast.get('fixed_neural_seeds_parent_CI95')
            if ci is None:
                axes[1].plot(point,index,marker='D',markersize=6,markerfacecolor='white',markeredgecolor='#326a9d')
                axes[1].annotate('4 videos; no CI',(point,index),xytext=(7,6),textcoords='offset points',fontsize=7)
            else:
                axes[1].errorbar(point,index,xerr=[[point-ci[0]],[ci[1]-point]],fmt='o',color='#326a9d',capsize=3,markersize=5)
        axes[1].axvline(0,color='black',lw=.9,alpha=.6)
        axes[1].set(yticks=y,yticklabels=pairs,xlabel='Δ HOTA: DESIGN − UNIFORM',
            title='Paired parent-video 95% intervals')
        axes[1].invert_yaxis();axes[1].grid(axis='x',alpha=.15)
        fig.suptitle('I1 component experiment: ordinary teacher distillation, not the new unified model',fontsize=11)
        fig.text(.5,.015,'DEV data; all 828 outputs sealed before GT. Equal logical budgets; this is not a wall-clock comparison.',
            ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.055,1,.94))
        out=ROOT/'a100_runs/visual_classical_20261121/six_direction_20261122/figures'
        out.mkdir(parents=True,exist_ok=True)
        for ext in ('png','pdf','svg'):fig.savefig(out/('observation_policy_I1_20261123.'+ext),dpi=220)
        plt.close(fig)
    print(str(out/'observation_policy_I1_20261123.png'))


if __name__=='__main__':main()
