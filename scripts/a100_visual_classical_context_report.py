"""Describe every contextual-cohort contrast without selecting a winner by GT."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.a100_visual_classical_sparse_observation_report import describe

PUBLIC = ROOT/'docs/a100_visual_classical_20261121'


def figure(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    rows = result['repeated']['CONTEXT_C4']['comparisons']
    contrasts = [('CONTEXT_MEAN_vs_MASKED_MEAN', 'Context inputs vs masked inputs'),
        ('CONTEXT_FULL_vs_CONTEXT_MEAN', 'Graph vs same-feature KD mean'),
        ('CONTEXT_FULL_vs_CONTEXT_ANCHORS', 'Graph vs independent anchors'),
        ('CONTEXT_FULL_vs_CONTEXT_VISUAL', 'Relative-position edges'),
        ('CONTEXT_FULL_vs_ORIGINAL_GRAPH', 'Full vs original mean + graph')]
    for i, (name, label) in enumerate(contrasts):
        row = rows[name]
        lo, hi = row['fixed_neural_seeds_parent_CI95']
        axes[0].hlines(i, lo, hi, color='tab:blue', lw=2)
        axes[0].plot(row['delta_HOTA'], i, 'o', color='tab:blue')
    axes[0].set_yticks(range(len(contrasts)), [label for _, label in contrasts])
    axes[0].set_title('C4: same10% QA, three neural seeds')
    names = [(pair, other) for pair in ('C4', 'C1') for other in ('ORIGINAL_COMPLETE', 'ORIGINAL_PERMUTED')]
    for i, (pair, other) in enumerate(names):
        row = result['repeated']['CONTEXT_'+pair]['comparisons']['ORIGINAL_GRAPH_vs_'+other]
        lo, hi = row['fixed_neural_seeds_parent_CI95']
        axes[1].hlines(i, lo, hi, color='tab:orange', lw=2)
        axes[1].plot(row['delta_HOTA'], i, 'o', color='tab:orange')
    axes[1].set_yticks(range(len(names)), [pair+': visual vs '+other.removeprefix('ORIGINAL_').lower() for pair, other in names])
    axes[1].set_title('Original sparse graph: correspondence controls')
    for ax in axes:
        ax.axvline(0, color='black', lw=.8)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.2)
        ax.set_xlabel('HOTA difference; original-parent95% interval')
    fig.suptitle('Fixed candidate boxes/IDs; development data; no target GT in adaptation\nRecorded features/QA: not a cold deployment cost benchmark', fontsize=11)
    destination = PUBLIC/'figures'
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination/'relative_context_cohort.png', dpi=180)
    fig.savefig(destination/'relative_context_cohort.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def main():
    result = json.loads((PUBLIC/'context_cohort_analysis.json').read_text())
    assert result['status'] == 'CONTEXT_COHORT_ANALYZED'
    lines = ['TTA-RMOT：相对场景关系与稀疏图特异性——持续研究阶段报告', '2026-11-22', '',
        '科研Goal仍ACTIVE。本报告呈现完整预设条件；方法是否保留以实测贡献和成本判断，不以队列完成结项。', '',
        '一、任务与实验缘由',
        '沿用官方候选框及原Track ID；跨域指代适配只用目标图像、查询和冻结教师，不使用目标人工标签。',
        '上一轮三观察对齐和外观衰减时序没有超过轨迹均值基线，故不叠加更多时序网络。',
        'C4空间/角色组较弱且原图不处理这些查询，据此登记新的开发假设：人群相对位置与衣着关系是否有帮助？',
        '这不是独立盲测：相同C4/C1均已参与开发；新预登记明确发生于上一阶段诊断之后。', '',
        '二、输入与公平基线',
        '全部C4条件共享原新鲜PCD/RGB与1160条冻结问答；C1共享1103条，仅检验原稀疏图特异性。',
        '三种子0/17/42只重复神经初始化，问答不变；当前视频允许转导，视频间只用过去与当前反馈。',
        '上下文输入为同帧候选位置排名、相对高度等8维统计，并补充空间/角色查询的原有RGB观察。',
        '普通KD也获得完全相同1088维上下文输入；MASKED头使用原RGB+8零维，控制维度和初始化影响。',
        'soft/hard头实际充分拟合，保留400步、最终和固定均值；不使用GT选checkpoint。',
        '完整关系图与同输入KD、独立锚点、去位置视觉图、同总边权完全图、节点对应打乱图比较。',
        '原10%均值视觉图还须逐观测复现旧输出，之后与完全图/打乱图比较；新增教师为0。', '',
        '三、三种子聚合与全部固定对照']
    for family, row in result['repeated'].items():
        lines.append(family)
        lines.append('均值HOTA：'+', '.join(name+'='+f'{value:.6f}' for name, value in row['mean_HOTA'].items()))
        for name, value in row['comparisons'].items():
            lines.append('  '+name+': '+describe(value))
    lines += ['', '四、每一神经种子与实际头拟合账单']
    for name, row in result['conditions'].items():
        lines.append(name+'; QA='+str(row['logical_teacher_calls']))
        lines.append('  '+json.dumps(row['raw_HOTA'], ensure_ascii=False))
        lines.append('  头拟合秒数（预付特征/问答共享实验，不是冷部署）：'+json.dumps(row['actual_head_fit_seconds']))
        lines.append('  拟合达标：'+json.dumps(row['teacher_fit_audit']))
        lines.append('  CountSelf变化：'+json.dumps(row['CountSelf_changed_query_files']))
    lines += ['', '五、输入定义查询组的seed0点诊断',
        json.dumps(result['input_defined_query_group_points_seed0'], ensure_ascii=False, indent=2),
        '空间/角色组原视频很少，只呈现点数；组间分数不可作为可加HOTA损失，也不是独立域验证。', '',
        '六、归因和证据范围',
        'CONTEXT_MEAN对MASKED_MEAN检验输入；FULL对CONTEXT_MEAN或ANCHORS检验同输入下图推断；FULL对VISUAL检验关系边。',
        '只有总体超过原强均值KD/原图，并且模块自身删除有稳定代价，才支持保留新模块。',
        '同时报告FULL数量、同输入KD数量和原KD数量，数量由预测给出，不读取GT数量。',
        '固定一侧数量的正差不自动意味着普遍语义排序更好；也不能将完全图/打乱图的失败当作对强KD的增益。',
        '所有新预测及数量反事实先封存，再隔离官评；所有失败、负结果与初始化变化均保留。',
        '当前运行使用预付特征和问答；若保留方案，仍需真实加载/学生/教师/拟合/输出的物理成本测量。',
        '原视频bootstrap仅描述固定开发输出，不能消除反复开发、多重比较偏差。',
        '最终Goal验收尚待根代理结合结果及真实部署账单判断，不能自动结项。']
    destination = ROOT/'docs/TTA_RMOT_相对场景关系与稀疏图特异性_阶段报告_20261122.txt'
    destination.write_text('\n'.join(lines)+'\n')
    figure(result)
    print(json.dumps(dict(event='CONTEXT_COHORT_REPORT_SAVED', path=str(destination), goal_status='ACTIVE')), flush=True)


if __name__ == '__main__':
    main()
