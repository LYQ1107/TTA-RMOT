"""Scientific plot of the complete converged-head protocol comparison."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.a100_visual_classical_prepare import OUT

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'


def main():
    result=json.loads((PUBLIC/'protocol_calibration_analysis.json').read_text())
    review=json.loads((PUBLIC/'protocol_calibration_scientific_review.json').read_text())
    assert json.loads((PUBLIC/'protocol_calibration_statistical_reproduction.json').read_text())['status'].startswith('PASS_')
    pure=None
    if (PUBLIC/'pure_components_analysis.json').is_file() and (PUBLIC/'pure_components_statistical_reproduction.json').is_file():
        assert json.loads((PUBLIC/'pure_components_statistical_reproduction.json').read_text())['status'].startswith('PASS_')
        pure=json.loads((PUBLIC/'pure_components_analysis.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(13.5,8.5))
    fig.subplots_adjust(left=.20,right=.98,bottom=.10,top=.80,wspace=.58,hspace=.38)
    order=('KD','BASE_PROTOCOL','FINAL_CAL','FINAL_LOCAL','FINAL_GLOBAL','PROTOCOL_JOINT')
    labels=('ordinary\nFINAL','ordinary +\nprotocol','FINAL +\nPAVA','local +\nPAVA','global +\nPAVA','contract +\nPAVA')
    contrasts=('KD','BASE_PROTOCOL','FINAL_CAL','PROTOCOL_GRAPH')
    colours=['#666666','#8c8c8c','#e69f00','#56b4e9','#0072b2','#b44b87']
    for column,pair in enumerate(('C4','C1')):
        row=result['repeated']['PROTOCOL_CALIBRATION_'+pair];ax=axes[0,column]
        for i,name in enumerate(order):
            values=[result['conditions'][pair+f'_seed{s}']['raw_HOTA'][name] for s in (0,17,42)]
            ax.scatter(i,row['mean_HOTA'][name],s=65,color=colours[i],zorder=3)
            ax.scatter(np.asarray([i-.08,i,i+.08]),values,s=12,color=colours[i],alpha=.5,zorder=2)
            ax.annotate(f"{row['mean_HOTA'][name]:.3f}",(i,row['mean_HOTA'][name]),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
        ax.set(xticks=range(len(order)),xticklabels=labels,ylabel='Official pooled HOTA (0-100)',title=pair+' | exact shared converged head')
        ax.margins(y=.22,x=.1);ax.grid(axis='y',alpha=.2);ax.tick_params(axis='x',labelsize=8)
        ax=axes[1,column]
        effects=[row['comparisons']['PROTOCOL_JOINT_vs_'+name] for name in contrasts]
        effect_labels=('vs ordinary FINAL','vs ordinary + protocol','vs PAVA only','PAVA increment')
        if pure is not None:
            source=pure['repeated']['PURE_COMPONENTS_'+pair]['comparisons']
            effects=[row['comparisons']['PROTOCOL_JOINT_vs_KD'],*[source['PROTOCOL_JOINT_vs_'+name] for name in ('ANCHOR_ONLY','ANCHOR_CAL','SHUFFLED_JOINT')]]
            effect_labels=('vs ordinary FINAL','vs raw teacher overwrite','vs raw anchors + PAVA','vs shuffled visual cues')
        for i,e in enumerate(effects):
            value=e['delta_HOTA'];lo,hi=e['parent_and_neural_seed_sensitivity_CI95']
            ax.plot([lo,hi],[i,i],color='#b44b87',linewidth=2);ax.scatter(value,i,color='#b44b87',s=40,zorder=3)
            ax.text(.98,i-.18,f'{value:+.3f} [{lo:+.3f}, {hi:+.3f}]',transform=ax.get_yaxis_transform(),
                ha='right',va='bottom',fontsize=7)
        ax.axvline(0,color='#555555',linestyle='--',linewidth=1)
        ax.set(yticks=range(len(contrasts)),yticklabels=effect_labels,
            xlabel='Contract + PAVA minus comparator (HOTA)',title='Original-parent + neural-seed sensitivity')
        ax.invert_yaxis();ax.margins(x=.52,y=.2);ax.grid(axis='x',alpha=.2)
    fig.suptitle('Task-contract pooling + visual calibration | '+review['status']+
        '\nSame teacher QA and head; 3 seeds; all DEV; descriptive intervals, no blind or multiplicity-corrected validation'+
        '\n0 new teacher/encoder forwards or neural-head fits in these replays; fresh end-to-end costs reported separately',fontsize=10,y=.985)
    preview=OUT/'analysis/publication_figure_previews';preview.mkdir(parents=True,exist_ok=True);(PUBLIC/'figures').mkdir(exist_ok=True)
    fig.savefig(preview/'protocol_calibration.png',dpi=160,bbox_inches='tight')
    fig.savefig(PUBLIC/'figures/protocol_calibration.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    print('PROTOCOL_CALIBRATION_FIGURE_WRITTEN',flush=True)


if __name__=='__main__':main()
