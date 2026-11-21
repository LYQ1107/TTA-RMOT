"""Publication figures for fixed recipes; never select a method by target GT."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.a100_visual_classical_prepare import OUT

DIRECTORY = ROOT / 'docs/a100_visual_classical_20261121/figures'


def save(fig, name, footer):
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.text(.5, .015, footer, ha='center', fontsize=8)
    for extension in ('png', 'pdf'):
        fig.savefig(DIRECTORY / (name + '.' + extension), dpi=190)
    plt.close(fig)


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    studies = {name: json.loads((OUT / 'analysis' / file).read_text()) for name, file in [
        ('Raw soft targets', 'three_seed_task_valid_results.json'),
        ('Matched bounded soft targets', 'three_seed_clipped_results.json')]}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.3))
    methods = ['SOFT_KD_400', 'HARD_KD_400', 'SOFT_KD', 'HARD_KD', 'NO_GRAPH', 'COHORT']
    names = ['Soft KD, 400 steps', 'Hard KD, 400 steps', 'Soft KD, fully fitted',
             'Hard KD, fully fitted', 'Same anchors, no graph', 'Same soft KD + cohort']
    for row, (title, records) in enumerate(studies.items()):
        for column, pair in enumerate(('C4', 'C1')):
            ax = axes[row, column];record = records[pair]
            values = [record['mean_HOTA'][m] for m in methods]
            ax.barh(range(len(methods)), values, color=['#8092a5'] * 5 + ['#168b83'])
            for seed, shift in zip((0, 17, 42), (-.12, 0, .12)):
                ax.scatter([record['per_seed_HOTA'][str(seed)][m] for m in methods],
                           np.arange(len(methods)) + shift, s=15, color='black', alpha=.65)
            for y, value in enumerate(values):
                ax.text(value + .3, y, f'{value:.3f}', va='center', fontsize=8)
            ax.set_yticks(range(len(methods)), names);ax.invert_yaxis()
            ax.set_xlim(0, 77 if pair == 'C4' else 49)
            ax.set_title(pair + ' | ' + title, fontsize=10)
            ax.set_xlabel('Mean of three official HOTA evaluations')
            ax.grid(axis='x', alpha=.15)
    fig.suptitle('Actual hard/soft distillation and matched graph ablations | seeds 0, 17, 42')
    save(fig, 'three_seed_same_evidence', 'Dots are algorithmic seeds on the same development videos, not independent datasets. Same source, inputs and teacher budget.')

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4))
    keys = ['FULL_vs_SOFT_KD', 'FULL_vs_NO_GRAPH', 'COUNT_FULL_vs_SOFT_KD']
    labels = ['Full - soft KD', 'Full - no graph', 'Full - soft KD, fixed counts']
    for row, (title, records) in enumerate(studies.items()):
        for column, pair in enumerate(('C4', 'C1')):
            ax = axes[row, column]
            for y, key in enumerate(keys):
                item = records[pair]['comparisons'][key];point = item['mean_delta_HOTA']
                lo, hi = item['parent_and_seed_resampling_sensitivity_CI95']
                ax.plot([lo, hi], [y, y], color='#afbdc9', linewidth=5, solid_capstyle='round')
                lo, hi = item['fixed_seed_parent_CI95']
                ax.plot([lo, hi], [y, y], color='#168b83', linewidth=2)
                ax.scatter([point], [y], color='#123c45', s=28, zorder=4)
                ax.text(point, y - .18, f'{point:+.3f}', ha='center', fontsize=9)
            ax.axvline(0, linestyle='--', color='#666', linewidth=1)
            ax.set_yticks(range(3), labels);ax.set_ylim(2.5, -.6)
            ax.set_title(pair + ' | ' + title, fontsize=10)
            ax.set_xlabel('Difference in official HOTA; 95% descriptive intervals')
            ax.grid(axis='x', alpha=.15)
    fig.suptitle('Module effect and fixed-count ordering: uncertainty remains visible')
    save(fig, 'three_seed_effect_and_countmatched', 'Green: original-parent bootstrap; pale: parent + seed sensitivity. C4 has 13 original parents; C1 has 25. No blind validation.')

    records = json.loads((OUT / 'analysis/graph_field_search.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    policies = ['COVER', 'UNIFORM', 'FIELD_STATIC', 'FIELD']
    for ax, pair in zip(axes, ('C4', 'C1')):
        record = records[pair];x = np.arange(len(policies))
        for shift, method, label, color in [(-.18, 'SOFT_KD', 'Fully fitted soft KD', '#8092a5'),
                                            (.18, 'COHORT', 'Same KD + cohort', '#168b83')]:
            values = [record['HOTA'][p + '/' + method] for p in policies]
            bars = ax.bar(x + shift, values, width=.36, label=label, color=color)
            ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=2)
        ax.set_xticks(x, ['Stratified\ncoverage', 'Uniform', 'Graph search\nno feedback', 'Graph search\nwith feedback'])
        ax.set_ylim(0, 76 if pair == 'C4' else 47);ax.set_ylabel('Official HOTA');ax.set_title(pair)
        ax.grid(axis='y', alpha=.15);ax.legend(fontsize=8, loc='lower right')
    fig.suptitle('Adaptive questioning is not a universal improvement | fixed total questions, seed 0')
    save(fig, 'adaptive_search_negative_control', 'FIELD vs FIELD_STATIC: same graph, prior and inference; only returned feedback changes selection. Evidence preparation uses an opaque reused cache.')
    print(json.dumps(dict(status='REPEAT_FIGURES_EXPORTED', directory=str(DIRECTORY))))


if __name__ == '__main__':
    main()
