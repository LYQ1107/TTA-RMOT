"""Plot independently observed process walls, without shared-prefix estimates."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.a100_visual_classical_prepare import OUT


def main():
    result = json.loads((OUT / 'analysis/independent_single_cold_C4.json').read_text())
    assert result['status'] == 'PASS_INDEPENDENT_SINGLE_METHOD_COLD_COMPARISON'
    methods = ['CLIP_KD', 'CLIP_COHORT']
    colors = ['#8092a5', '#168b83']
    labels = ['Ordinary bounded-soft KD', 'Same KD + visual cohort graph']
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    walls = [result['methods'][m]['actual_worker_process_seconds'] / 60 for m in methods]
    scores = [result['methods'][m]['HOTA'] for m in methods]
    axes[0].barh(range(2), walls, color=colors)
    axes[0].set_yticks(range(2), labels)
    axes[0].set_xlabel('Actual independent process wall (minutes)')
    axes[0].set_xlim(0, max(walls) * 1.23)
    for y, wall in enumerate(walls):
        axes[0].text(wall + max(walls) * .02, y, f'{wall:.2f}', va='center')
    axes[0].invert_yaxis()
    axes[0].set_title('Both models actually reloaded in each process')
    axes[1].scatter(walls, scores, c=colors, s=75)
    for label, wall, score, offset in zip(labels, walls, scores, [-20, 14]):
        axes[1].annotate(f'{label}\nHOTA {score:.3f}', (wall, score),
                         xytext=(0, offset), textcoords='offset points',
                         ha='center', fontsize=8)
    axes[1].set_xlabel('Actual independent process wall (minutes)')
    axes[1].set_ylabel('Official whole-C4 HOTA')
    axes[1].margins(x=.5, y=.6)
    axes[1].grid(alpha=.15)
    point = result['comparison']['delta_HOTA']
    interval = result['comparison']['original_parent_video']['CI95']
    fig.suptitle(f'Same A100 GPU3 | HOTA difference {point:+.3f} '
                 f'| 13-parent descriptive 95% CI [{interval[0]:+.3f}, {interval[1]:+.3f}]', fontsize=11)
    fig.tight_layout(rect=(0, .14, 1, .92))
    fig.text(.5, .065, '45 videos / 466 queries each; 3508 actual new teacher answers each, zero cache hits. Given frozen candidate boxes/IDs.',
             ha='center', fontsize=8)
    fig.text(.5, .025, 'KD then graph; order not counterbalanced, OS cache retained, concurrent host/NAS. GT evaluation excluded. Development data.',
             ha='center', fontsize=8)
    directory = ROOT / 'docs/a100_visual_classical_20261121/figures'
    for suffix in ('png', 'pdf'):
        fig.savefig(directory / ('whole_C4_independent_single_cold.' + suffix), dpi=190)
    plt.close(fig)
    print(json.dumps(dict(status='INDEPENDENT_COLD_FIGURE_EXPORTED', directory=str(directory))))


if __name__ == '__main__':
    main()
