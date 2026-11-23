"""Show whether real visual correspondence matters beyond generic smoothing."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.a100_visual_classical_prepare import OUT


def main():
    records=json.loads((OUT/'analysis/graph_specificity.json').read_text())
    directory=ROOT/'docs/a100_visual_classical_20261121/figures';directory.mkdir(exist_ok=True)
    methods=['PERMUTED_GRAPH','COMPLETE_GRAPH','PERMUTED_AT_FULL_COUNT','COMPLETE_AT_FULL_COUNT']
    labels=['Shuffled correspondence','All-candidate graph','Shuffled, same FULL counts','All-candidate, same FULL counts']
    fig,axes=plt.subplots(2,2,figsize=(12.5,8))
    for row,(variant,title) in enumerate([('raw','Raw soft targets'),('bounded','Matched bounded targets')]):
        for column,pair in enumerate(['C4','C1']):
            ax=axes[row,column]
            for y,method in enumerate(methods):
                record=records[variant][pair]['FULL_vs'][method];point=record['mean_delta_HOTA']
                lo,hi=record['parent_and_seed_resampling_sensitivity_CI95'];ax.plot([lo,hi],[y,y],color='#afbdc9',linewidth=5,solid_capstyle='round')
                lo,hi=record['fixed_seed_parent_CI95'];ax.plot([lo,hi],[y,y],color='#168b83',linewidth=2)
                ax.scatter([point],[y],color='#123c45',s=28,zorder=4);ax.text(point,y-.2,f'{point:+.3f}',ha='center',fontsize=9)
            ax.axvline(0,linestyle='--',color='#555',linewidth=1);ax.set_yticks(range(4),labels);ax.set_ylim(3.5,-.65)
            ax.set_title(pair+' | '+title);ax.set_xlabel('Real visual graph minus control (official HOTA)');ax.grid(axis='x',alpha=.15)
    fig.suptitle('Visual correspondence versus generic smoothing | same head, same teacher answers, three seeds')
    fig.tight_layout(rect=(0,.08,1,.95))
    fig.text(.5,.025,'Green: original-parent bootstrap; pale: parent + seed sensitivity. Edge weight multiset or total precision held fixed. Development data.',ha='center',fontsize=8)
    for suffix in ['png','pdf']:fig.savefig(directory/('graph_specificity.'+suffix),dpi=190)
    plt.close(fig);print(json.dumps(dict(status='GRAPH_SPECIFICITY_FIGURE_EXPORTED',directory=str(directory))))


if __name__=='__main__':main()
