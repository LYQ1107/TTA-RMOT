"""Readable phase evidence and scientific plots; never closes the research goal."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.a100_visual_classical_prepare import OUT

PUBLIC = ROOT / 'docs/a100_visual_classical_20261121'


def describe(effect):
    lo, hi = effect['fixed_neural_seeds_parent_CI95']
    value = f"{effect['delta_HOTA']:+.6f}; parent95%[{lo:+.6f}, {hi:+.6f}]"
    if 'parent_and_neural_seed_sensitivity_CI95' in effect:
        lo, hi = effect['parent_and_neural_seed_sensitivity_CI95']
        value += f'; parent+seed95%[{lo:+.6f}, {hi:+.6f}]'
    return value


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    conditions = result['conditions']
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    labels = dict(SOFT_FINAL='Bounded soft KD', SOFT_MEAN_400_FINAL='KD checkpoint mean',
        HARD_400='Hard KD (400 steps)', GRAPH='Same-head visual graph', MEAN_GRAPH='Mean + graph')
    legacy = json.loads((OUT / 'analysis/fresh_whole_C4_sparse.json').read_text())
    legacy_names = dict(SOFT_FINAL='CLIP_KD', SOFT_MEAN_400_FINAL='CLIP_MEAN',
        HARD_400='HARD_KD_400', GRAPH='CLIP_COHORT', MEAN_GRAPH='CLIP_MEAN_COHORT')
    c1_names = dict(SOFT_FINAL='POOLED_SOFT', SOFT_MEAN_400_FINAL='POOLED_MEAN',
        HARD_400='POOLED_HARD400', GRAPH='POOLED_GRAPH', MEAN_GRAPH='POOLED_MEAN_GRAPH')
    for col, pair in enumerate(('C4', 'C1')):
        ax = axes[0, col]
        for name, label in labels.items():
            x, y = [], []
            for rho in (.05, .1, .2):
                row = conditions[f'budget{rho:g}_seed0_{pair}']
                x.append(row['logical_teacher_calls'])
                y.append(row['raw_HOTA'][name])
            if pair == 'C4':
                # Fresh-feature curve uses the ORIGINAL fresh-feature30%
                # result, never the historical observation study's pool.
                x.append(legacy['logical_teacher_calls'])
                y.append(legacy['HOTA'][legacy_names[name]])
            else:
                row = conditions['observation_seed0_C1']
                assert result['observation_input_equivalence']['C1']['pooled_PCD_max_abs_difference'] == 0
                assert result['observation_input_equivalence']['C1']['pooled_RGB_max_abs_difference'] == 0
                x.append(row['logical_teacher_calls'])
                y.append(row['raw_HOTA'][c1_names[name]])
            ax.plot(x, y, 'o-', label=label, linewidth=1.5, markersize=4)
        ax.set(title=pair+': fixed-evidence budget curve (seed0)',
            xlabel='Selected teacher answers (recorded QA)', ylabel='Official HOTA')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        rows = result['repeated']['OBSERVATION_'+pair]['comparisons']
        names = [('ANCHOR_FULL_vs_POOLED_MEAN', 'vs pooled KD mean'),
            ('ANCHOR_FULL_vs_ANCHOR_KD', 'vs aligned KD'),
            ('ANCHOR_FULL_vs_ANCHOR_NO_GRAPH', 'vs anchors only'),
            ('ANCHOR_FULL_vs_ANCHOR_VISUAL_ONLY', 'add temporal edges'),
            ('ANCHOR_FULL_vs_CONSTANT_TEMPORAL', 'vs constant temporal'),
            ('ANCHOR_FULL_vs_MASS_MATCHED_CONSTANT_TEMPORAL', 'vs equal-mass constant')]
        ax = axes[1, col]
        for i, (name, label) in enumerate(names):
            row = rows[name]
            lo, hi = row['fixed_neural_seeds_parent_CI95']
            ax.hlines(i, lo, hi, color='tab:blue', linewidth=2)
            ax.plot(row['delta_HOTA'], i, 'o', color='tab:blue')
        ax.axvline(0, color='black', linewidth=.8)
        ax.set_yticks(np.arange(len(names)), [label for _, label in names])
        ax.invert_yaxis()
        ax.set(title=pair+': observation mechanism, 3 neural seeds',
            xlabel='HOTA difference; original-parent95% interval')
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Fixed official candidates; no target GT in adaptation\nReused features/QA: quality evidence, not a cold wall-time benchmark', fontsize=12)
    destination = PUBLIC / 'figures'
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / 'sparse_evidence_observation_study.png', dpi=180)
    fig.savefig(destination / 'sparse_evidence_observation_study.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def main():
    result = json.loads((PUBLIC / 'sparse_observation_analysis.json').read_text())
    assert result['status'] == 'SPARSE_EVIDENCE_AND_OBSERVATION_ANALYZED'
    diagnostic = json.loads((PUBLIC / 'observation_granularity_diagnosis.json').read_text())
    lines = ['TTA-RMOT：稀疏教师证据与三观察对齐——持续研究阶段报告', '研究批次：2026-11-21；阶段汇总：2026-11-22', '',
        '持续科研Goal仍为ACTIVE。本报告展示全部固定条件，不自动将队列完成视为科研目标达成。', '',
        '一、任务合同与为什么继续实验',
        '用户确认沿用官方固定候选框和Track ID，只研究跨域指代适配。适配只用目标域图像、任务语言及冻结教师，不用目标人工标签。',
        '当前视频允许转导；视频间只用过去和当前的教师反馈，不读取未来答案。全部现有数据均为开发数据。',
        '前次核验：C4直接回填教师44.570，仅当前视频充分拟合KD68.341，历史+当前KD69.050；无缓存官评完整复现。',
        '所以69分是应该保留的强对照；不能通过减少其训练、拿走其合法输入来制造创新优势。', '',
        '本轮结果判断（2026-11-22，全部预测封存并完成404组官评后）：',
        '1. 10%预算下，固定均值KD加视觉图：C4三种子66.883对66.276，+0.608；原视频+种子区间[-0.028,1.269]，仍不能宣布稳健全面胜出。',
        '2. 同条件C1为36.816对34.305，+2.510，原视频+种子区间[0.743,4.769]，支持该方向稀疏证据下的系统收益。',
        '3. C1固定KD数量后该差仅+0.00037且区间跨0，故主要是数量/选择校准收益，不称为普遍纯排序突破。',
        '4. 5%预算C4视觉图63.487低于硬KD400步64.721；不能声称教师越少图越占优。',
        '5. anchor对齐头比同特征POOLED软KD，C4低1.510、C1低2.092；完整观察图也没有超过POOLED图，不保留这条新机制。',
        '6. 外观衰减时序边比恒定边，C4低0.188、C1低0.882；同总边权控制亦无正贡献，不能包装为有效身份切换保护。',
        '7. 教师锚点真值与整轨迹可见多数仅21/3095（C4）、5/2510（C1）不一致；物理串ID不等于衣着语义改变。',
        '8. C4输入定义的空间/角色组seed0 HOTA约49.43/48.05，当前时序机制原样回退这些查询。它们是后续场景关系假设的依据，尚不是关系模块有效的证明。', '',
        '二、两个固定假设',
        '假设A：同样减少教师监督后，图算法是否比普通充分训练的KD更能利用稀疏证据？主预算10%，5%/20%为辅助曲线。',
        '假设B：一次教师只看一帧，却用该标签监督三次观察的均值并广播整轨迹，可能存在粒度错配。用真实anchor训练对齐小头，并按实际外观一致性传播。',
        '保留相同三次观察、源主干、教师、候选框及ID；不增加教师，不引入HMM/LoRA。',
        '对齐机制仅改静态衣着查询，其它查询用同一个POOLED小头。比较视觉边、时序边、恒定时序及同总边权控制。',
        '所有头均真正拟合：残差MLP，至少4000步最多16000步，教师KL停止；保留硬标签400步、充分软蒸馏和固定400/最终均值。', '',
        '三、输入与证据范围',
        json.dumps(result['observation_input_equivalence'], ensure_ascii=False, indent=2),
        'C4历史逐观察特征有数值漂移，观察机制只与同历史特征重新拟合的POOLED KD归因；预算曲线仍用原新鲜特征。',
        'C4曲线30%终点来自原新鲜整流；没有把历史观察研究的30%点拼进该曲线。',
        '三种子共享seed0实际冻结问答，仅重复神经初始化，不是三个测试域，也没有改变选题以挑选答案。',
        '本轮新增教师前向为0，重用神经特征和冻结问答；本轮运行时间不是冷启动部署成本。', '',
        '观察FULL与ANCHOR_KD共享对齐头和提供非衣着回退的POOLED软头；与单POOLED头的训练工作量不同。',
        '不能因教师预算相同就称总算力完全相同；各头实际拟合时间另列，后续保留方案再核算真实新鲜部署成本。', '',
        '四、每一个固定条件的原始HOTA']
    for name, row in result['conditions'].items():
        lines.append(name+'; 问答='+str(row['logical_teacher_calls']))
        lines.append('  实际头拟合秒数（共享实验流，非独立冷部署）：'+json.dumps(row['actual_head_fit_seconds'], ensure_ascii=False))
        lines.append('  '+', '.join(method+'='+f'{value:.6f}' for method, value in row['raw_HOTA'].items()))
    lines += ['', '五、三种子聚合与模块归因（全部对照）']
    for name, row in result['repeated'].items():
        lines.append(name)
        lines.append('均值HOTA：'+', '.join(method+'='+f'{value:.6f}' for method, value in row['mean_HOTA'].items()))
        for label, record in row['comparisons'].items():
            lines.append('  '+label+': '+describe(record))
    lines += ['', '六、数量与排序应怎样区分',
        '所有CountMatched都使用方法自身的逐查询逐帧数量，不使用GT数量。',
        '同时报告以完整方法数量和以基线数量为目标的两种方向；Dance保留联合互斥约束，全部CountSelf必须逐文件复现。',
        '一侧数量投影有正差不等于普遍排序提升；总收益、数量校准和排序必须分别解释。', '',
        '七、物理串ID与语义变化的诊断',
        json.dumps(diagnostic['pairs'], ensure_ascii=False, indent=2),
        '教师同GT的协议一致性是候选IoU成员诊断，还混有定位/标注影响，不称作纯语义准确率。',
        '这些TP/FP/FN不是HOTA损失的可加拆账，也不进入方法的参数、阈值或训练。', '',
        '八、证据边界与继续条件',
        '所有结果均完整保留，包括负结果；没有按目标域逐格选择获胜方法。',
        '区间为固定输出的原视频聚类描述：C4的45片段来自13个原视频；神经种子敏感性另报。',
        '开发迭代与多重比较的偏差不能靠bootstrap消除，没有独立新划分就不能称作独立盲测或泛化定论。',
        '根代理需依据总分、强均值/硬标签对照、模块删除、数量双向与实际新鲜执行逐条判断哪些机制可保留。',
        '尚未完成的验收继续推进，不能因写完本报告或跑完本队列而宣称整个科研Goal完成。']
    path = ROOT / 'docs/TTA_RMOT_稀疏证据与观察对齐_持续研究阶段报告_20261121.txt'
    path.write_text('\n'.join(lines)+'\n')
    plot(result)
    print(json.dumps(dict(event='SPARSE_OBSERVATION_PHASE_REPORT_SAVED', path=str(path), goal_status='ACTIVE')), flush=True)


if __name__ == '__main__':
    main()
