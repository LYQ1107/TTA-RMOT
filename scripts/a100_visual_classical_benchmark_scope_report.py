"""Publish the sealed feedback-scope audit; the research goal remains active."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_benchmark_scope import AUDIT
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_mechanism_analyze import compare_official
from scripts.analyze_pf_video_uncertainty import pooled_hota


def main():
    source = json.loads((OUT / 'analysis/benchmark_scope_audit.json').read_text())
    assert source['status'] == 'PASS_BASELINE_SCOPE_AND_UNCACHED_EVALUATION_AUDIT'
    public = ROOT / 'docs/a100_visual_classical_20261121'
    public.mkdir(parents=True, exist_ok=True)
    sufficient, differences, groups = {}, {}, {}
    names = ('SOURCE_ONLY', 'DIRECT_TEACHER', 'CURRENT_ONLY', 'PAST_ONLY', 'PREFIX_REPRO')
    for pair in ('C4', 'C1'):
        data = {}
        sufficient[pair] = {}
        for name in names:
            scope = 'PAST_ONLY' if name == 'PAST_ONLY' else 'CURRENT_ONLY'
            folder = AUDIT / pair / scope / name
            data[name] = read_stats(folder)
            d = data[name]
            sufficient[pair][name] = dict(videos=d['videos'], parents=d['parent_videos'],
                numerator=d['stats'][0].tolist(), denominator=d['stats'][1].tolist(),
                parent_numerator=d['parent_stats'][0].tolist(),
                parent_denominator=d['parent_stats'][1].tolist(), HOTA=d['HOTA'])
        differences[pair] = {a + '_vs_' + b: compare_official(data[a], data[b])
            for a, b in [('DIRECT_TEACHER', 'SOURCE_ONLY'), ('CURRENT_ONLY', 'DIRECT_TEACHER'),
                         ('PREFIX_REPRO', 'CURRENT_ONLY'), ('PREFIX_REPRO', 'PAST_ONLY')]}
        if pair == 'C4':
            videos = data['SOURCE_ONLY']['videos']
            seen, first = set(), []
            for v in videos:
                parent = v.rsplit('_c', 1)[0]
                first.append(parent not in seen)
                seen.add(parent)
            for label, choice in [('first_clip_per_parent', np.asarray(first)),
                                  ('later_clip_of_seen_parent', ~np.asarray(first))]:
                assert choice.any()
                groups[label] = dict(clips=[v for v, selected in zip(videos, choice) if selected],
                    HOTA={name: float(pooled_hota((d['stats'][0][choice], d['stats'][1][choice]),
                        np.ones(int(choice.sum())))) for name, d in data.items()})
    result = dict(source, descriptive_scope_comparisons=differences,
        original_parent_exposure_groups=groups,
        task_contract='Official fixed candidates accepted by user; cross-domain referring adaptation only.',
        cannot_infer_source_only_frontend=True, different_feedback_scopes_not_equal_budget_competitors=True,
        first_parent_groups_are_descriptive_not_an_independent_split=True,
        ordinary_KD_name_correction='Strong prefix-transductive track KD with query-conditioned features and part colour, not raw teacher replacement.',
        goal_status='ACTIVE; fairness audit is a research step, not completion.')
    atomic_json(public / 'benchmark_scope_audit.json', result)
    atomic_json(public / 'benchmark_scope_exact_statistics.json', sufficient)
    lines = ['TTA-RMOT：普通蒸馏高分来源与公平性核验', '2026-11-21', '',
        '当前科研Goal仍为ACTIVE。本报告只核验基线，不能作为已经完成用户研究目标的结项。', '',
        '一、用户确认的任务范围',
        '沿用官方固定候选框与Track ID，研究跨域指代适配。当前适配仅使用目标图像、任务查询、教师反馈；不使用目标域人工标签。',
        '官方前端的监督来源单独披露，不把上游检测/跟踪能力记作我们的贡献。所有本文方法共享这些候选。', '',
        '二、69分的KD到底是什么',
        '它是充分训练的强转导蒸馏：冻结源模型的查询条件PCD、衣服部位颜色、三次观察聚合、当前及过去视频的教师软答案、轨迹级解码。',
        '1080维输入的小MLP实际训练至少4000步、最多16000步；教师概率裁剪到[0.02,0.98]，不是象征性训练几步的弱基线。',
        'CLIP_KD中的CLIP表示概率裁剪，不是CLIP视觉模型或时序教师视角。',
        '当前视频有教师反馈后再输出，属于当前视频转导；不是严格逐帧在线，也不是不看当前教师的零样本测试。', '',
        '三、固定输入与解码后的五组诊断（HOTA）',
        'SOURCE_ONLY：源轨迹先验；DIRECT_TEACHER：问过的轨迹直接用教师，未问的保留源；',
        'CURRENT_ONLY：只用当前视频答案训练；PAST_ONLY：输出本视频后才读取本视频答案；PREFIX_REPRO：过去加当前答案。',
        '这些信息范围不同，目的是解释高分，不把不同答案预算当成公平胜负。']
    for pair in ('C4', 'C1'):
        lines.append(pair + ': ' + ', '.join(name + '=' + f"{source['HOTA'][pair][name]:.6f}" for name in names))
        volume = source['membership'][pair]['volume']
        lines.append('反馈量：' + json.dumps(volume, ensure_ascii=False))
        repro = source['prefix_reproduction'][pair]
        lines.append('原强KD=' + f"{repro['reference_HOTA']:.6f}" + '，重新拟合=' + f"{repro['refit_HOTA']:.6f}" +
            '，变化的查询预测文件=' + str(repro['changed_query_prediction_files']))
    lines += ['', '四、完全绕过评估缓存复核',
        '独立C4 KD与图方法各466条查询的预测逐文件一致，交给未修改的完整官方TrackEval重新评估。',
        '无缓存官评相对已记录值的差：' + json.dumps(source['uncached_HOTA_differences'], ensure_ascii=False),
        '这只能验证计分复现，不能替代输入/反馈范围审计，也不能证明新机制已经击败强KD。', '',
        '五、同一原视频多片段的影响',
        'C4是13个原视频的45个片段，按时间前缀累计教师答案会包含同场景此前片段的信息。',
        '按事前声明的“每原视频首片段/之后片段”报告两组，不据此挑选有利子集、不称作独立新域验证。',
        json.dumps(groups, ensure_ascii=False, indent=2), '',
        '六、公平比较应怎样做',
        '方法与强KD使用相同候选、图像观察、查询、教师答案和解码约束；分别报告模块增量与全部执行成本。',
        '不能把旧48分和新69分的差全算成图算法贡献；中间同时改变了选题、特征、拟合和解码。',
        '也不能为了赢而撤掉强KD的合理训练、颜色特征或当前视频反馈；如果我们限制信息范围，KD也必须按同一范围重跑。',
        '新方案仍需在强裁剪KD、简单预测均值及数量匹配对照下证明贡献。C4已有小幅收益，C1与强对照的稳定优势仍未解决。',
        '冻结问答与历史特征用于本轮质量诊断；本轮新教师调用为0，诊断运行时间不作为真实冷启动部署成本。',
        '全部现有数据都是开发数据，置信区间仅为已固定输出的描述性原视频聚类分析；不能消除开发迭代偏差。', '',
        '接下来继续研究，不在基线核验处结项。候选框和ID保持固定，优先检查教师anchor、观察聚合与全轨迹传播的粒度是否错配。']
    report = ROOT / 'docs/TTA_RMOT_普通蒸馏高分来源与公平性核验_20261121.txt'
    report.write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(event='BENCHMARK_SCOPE_REPORT_SAVED', report=str(report),
        HOTA=source['HOTA'], groups=groups, goal_status='ACTIVE')), flush=True)


if __name__ == '__main__':
    main()
