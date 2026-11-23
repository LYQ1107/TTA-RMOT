"""Scientific figure of matched-head effects; no images, QA or GT are opened."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'
PREVIEW=ROOT/'a100_runs/visual_classical_20261121/analysis/publication_figure_previews'


def main():
    weighted=json.loads((PUBLIC/'weighted_classical_analysis.json').read_text())
    cue=json.loads((PUBLIC/'cue_consensus_analysis.json').read_text())
    for stem in ('weighted_classical','cue_consensus'):
        assert json.loads((PUBLIC/(stem+'_statistical_reproduction.json')).read_text())['status'].startswith('PASS_')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10.8,5.1),layout='constrained')
    for ax,pair in zip(axes,('C4','C1')):
        base=weighted['repeated']['WEIGHTED_CLASSICAL_'+pair];new=cue['repeated']['CUE_CONSENSUS_'+pair]
        rows=[base['comparisons'][name+'_vs_KD'] for name in ('GRAPH','CAL','JOINT')]
        rows.extend([base['comparisons']['JOINT_vs_CAL'],base['comparisons']['JOINT_vs_GRAPH'],new['comparisons']['CONSENSUS_JOINT_vs_FULL_JOINT']])
        for y,row,colour in zip(range(6),rows,('#4675b9','#c38a22','#28856c','#4675b9','#c38a22','#9961a5')):
            point=row['delta_HOTA'];low,high=row['parent_and_neural_seed_sensitivity_CI95']
            ax.errorbar(point,y,xerr=np.asarray([[point-low],[high-point]]),fmt='o',capsize=4,color=colour,lw=1.8,markersize=6)
        ax.axvline(0,color='#555555',lw=1,linestyle='--');ax.set_yticks(range(6),['Graph vs KD','PAVA vs KD','Joint vs KD','Joint vs PAVA','Joint vs Graph','Veto vs Joint'])
        ax.invert_yaxis();ax.set_xlabel('HOTA change in the stated paired comparison')
        ax.set_title(pair+'  |  ordinary KD HOTA '+format(base['mean_HOTA']['KD'],'.3f'))
        ax.grid(axis='x',alpha=.2)
        if pair=='C1':ax.text(.97,.03,'Veto leaves CHECKLIST unchanged',transform=ax.transAxes,ha='right',va='bottom',fontsize=8,color='#666666')
    fig.suptitle('Fair teacher baseline: which classical additions are supported?')
    fig.supxlabel('Development data; 3 neural seeds / 1 QA seed. Bars: original-parent + seed 95% sensitivity intervals.\nFixed-prediction bootstrap; no development/multiplicity correction or full-pipeline refit.',fontsize=8)
    (PUBLIC/'figures').mkdir(exist_ok=True);PREVIEW.mkdir(parents=True,exist_ok=True)
    fig.savefig(PUBLIC/'figures/shared_weighted_evidence.pdf');fig.savefig(PREVIEW/'shared_weighted_evidence.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(pdf=str(PUBLIC/'figures/shared_weighted_evidence.pdf'),preview=str(PREVIEW/'shared_weighted_evidence.png'))))


if __name__=='__main__':main()
