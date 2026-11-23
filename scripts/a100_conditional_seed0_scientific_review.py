"""Summarize completed single-seed evidence without selecting a new primary."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json

PUBLIC = ROOT/'docs/a100_six_direction_20261122'
TEXT = ROOT/'docs/TTA_RMOT_六设置单种子负结果与联合观察改进设计_20261123.txt'
LABELS = dict(C1='V2→Dance', C2='Dance→KITTI-V2', C3='V2→SportsMOT',
              C4='Dance→SportsMOT', C5='LaMOT→KITTI-V2', C6='LaMOT→Dance')


def read(name):
    return json.loads((PUBLIC/name).read_text())


def effect(row):
    interval = row.get('fixed_neural_seeds_parent_CI95')
    suffix = f'；父视频描述性95%区间[{interval[0]:+.6f}, {interval[1]:+.6f}]' if interval is not None else '；仅4个原视频，不报告总体区间'
    return f'{row["delta_HOTA"]:+.6f}'+suffix


def main():
    data = read('conditional_seed0_scope_analysis.json')
    proof = read('conditional_seed0_scope_statistical_reproduction.json')
    paid = read('conditional_seed0_evidence_support_analysis.json')
    temporal = read('conditional_seed0_temporal_diagnosis.json')
    alignment = read('conditional_seed0_input_alignment.json')
    assert data['neural_seeds'] == proof['neural_seeds'] == [0]
    assert proof['HOTA_records'] == 876 and proof['paired_effect_records'] == 678
    sections = ['''TTA-RMOT：六个跨域实验设置的单种子机制结论与下一步设计
日期：2026-11-23
状态：当前FULL配方未通过，研究Goal继续ACTIVE；下文区分完成结果与待验证假设。

一、先说结论

遵照用户要求，后续固定seed0，不再追加17/42或神经训练种子网格。
六个跨域设置保留，模块和强对照不能省略。历史非零种子已完成的记录保留。
本轮876份官方统计已全部完成，独立NumPy复算876个HOTA、222个汇总、678项配对效应和6项四臂交互；最大算术误差1.4210854715202004e-14。
这代表评价算对了，不代表方法成功。事先指定的FULL_FINAL在六个设置均低于同一固定普通对照COARSE_SOFT_MEAN。
不能把某个事后更高的消融换成主方法，也不能把对弱逐帧基线的提升写成优于充分蒸馏。

二、研究设定没有改变

固定官方候选和TrackID，只研究指代适配。前端McByte存在SportsMOT预训练背景，不称整套系统source-only。
适配只看目标图片、任务查询和冻结教师；人工GT只在本研究全部输出封存后，由隔离评价/诊断读取。
每个视频可用过去和当前视频的数据，不能用未来视频拟合当前；这是视频级转导，不是逐帧在线。
所有现有数据都是多轮开发集，没有未使用独立验证集。不能拿开发后的区间当盲测或泛化保证。
Dance有25个父视频，Sports的45片段归属13个原视频，KITTI只有4个原视频。区间按父视频而非片段重采样，固定训练seed0，不估计神经训练随机性；KITTI不制造总体置信区间。
Bootstrap只是已完成统计的CPU重采样，不是再训练上万个随机种子。

三、主要精度表：同一个固定普通对照，避免逐设置挑赢家

设置 | FULL_FINAL | COARSE_SOFT_MEAN | FULL−普通对照 | 配对执行COARSE_HARD_FINAL
''']
    summary = {}
    for pair, name in LABELS.items():
        row = data['repeated'][pair]
        scores = row['mean_HOTA']
        delta = scores['FULL_FINAL']-scores['COARSE_SOFT_MEAN']
        assert delta < 0
        sections.append(f"{pair} {name} | {scores['FULL_FINAL']:.6f} | {scores['COARSE_SOFT_MEAN']:.6f} | {delta:+.6f} | {scores['PAIRED_COARSE_HARD_FINAL']:.6f}\n")
        summary[pair] = dict(setting=name, primary_HOTA=scores['FULL_FINAL'],
            fixed_ordinary_HOTA=scores['COARSE_SOFT_MEAN'], delta_HOTA=delta,
            raw_comparison=row['comparisons']['FULL_vs_COARSE_SOFT_MEAN'],
            ranking_at_full_count=row['comparisons']['RANK_AT_FULL_COUNT_vs_COARSE_SOFT_MEAN'],
            ranking_at_baseline_count=row['comparisons']['RANK_AT_BASE_COUNT_vs_COARSE_SOFT_MEAN'])
    sections.append('''
COARSE普通头共享相同三次观察的信息，以轨迹聚合特征拟合教师软目标，优化充分；不是额外使用GT。
旧固定普通参照并非全部与新确定性执行配对，因此另保留4项同执行普通控制。配对COARSE_HARD_FINAL在5/6设置也高于FULL，C3例外。不能把强普通优势全部推给执行版本。
本表不和旧Native/v8P或旧47分C4直接减分归因；输入、训练和输出协议不同。

四、双向固定数量：不能只讲“少选/多选一些人”

下面比较FULL和同一个COARSE_SOFT_MEAN。第一项在FULL的输出数量下，第二项在普通对照的输出数量下。
''')
    for pair in LABELS:
        row = summary[pair]
        sections.append(f"{pair}：FULL数量 {effect(row['ranking_at_full_count'])}；普通数量 {effect(row['ranking_at_baseline_count'])}\n")
    sections.append('''
六个设置在普通对照数量下均为负。不能写成“只差阈值、排序已胜”；两个操作点必须同时报告。
HOTA差的raw−count代数拆分不是唯一数量因果效应，轨迹关联与筛选共同变化。

五、各候选模块和组合并没有自然叠加成优势

每行依次为：profile条件贡献、分解条件贡献、共享范围贡献、曝光平衡贡献。
''')
    for pair in LABELS:
        comparisons = data['repeated'][pair]['comparisons']
        sections.append(pair+'：'+' | '.join(f'{name} {effect(comparisons[name])}' for name in
            ('QUERY_PROFILE', 'CONDITIONAL_STRUCTURE', 'SHARED_SCOPE', 'EXPOSURE_BALANCE'))+'\n')
    sections.append('''
profile在Sports两个设置为负，描述性区间不跨零；不能称其为已验证的相对语义学习贡献。
因子分解在C6有条件正效应，但完整系统仍不及强普通，不能据此宣布整体成功。
共享袋支持在C6相对LOCAL_TWO为负，不能把共享本身说成人类对象恒常性的成功实现。
DESIGN与COVERAGE在C1/C3/C4/C6逐结果完全相同；C2/C5主方法低于COVERAGE。当前证据不支持独立的“状态激励选题”红利。
均匀观察在C3/C4反而更好。覆盖需要有用的证据与合适的学习目标；更多几何维度或问到更多真目标，并不自动转化为HOTA。

四臂交互：FULL−无profile条件头−带profile直接头+平衡直接头。原始交互与固定FULL数量下交互如下。
''')
    for pair, row in data['common_count_four_arm_description'].items():
        sections.append(f"{pair}：raw {row['raw']['interaction']:+.6f}；同FULL数量 {row['at_full_output_count']['interaction']:+.6f}\n")
    sections.append('''
固定FULL数量下交互为正不构成全部模块成功：raw交互正负交叉，也没有对强普通的总体优势。这里没有交互总体置信区间。

六、不是数据接错，也不能简单归咎于“不听老师劝”

全部55,019个已付费锚点、每点1089维，训练与实际打分特征逐值一致，差异元素/行均0。
这排除了本次特征交接错位，不代表历史PCD可由当前冷启动逐值重建，二者是不同问题。
训练中的当前观察总是三锚点之一；绝大多数稠密打分观察不是锚点。这是上下文关系分布差异，尚未证明它是失败的原因。
已付锚点上FULL总体大量保留教师判断；把旧“85%教师准确、融合29%”直接套到新模型不成立。
GT候选匹配统计混有框质量，不是纯教师语义准确率；各帧/查询也相关，不能把几百万行当独立样本。

设置 | 教师对→模型错 | 教师错→模型对 | 可恢复正查询FULL零输出 | 普通对照零输出 | DESIGN未问正袋查询
''')
    full_i = paid['methods'].index('FULL_FINAL')
    coarse_i = paid['methods'].index('COARSE_SOFT_MEAN')
    for pair in LABELS:
        row = paid['conditions'][pair+'_seed0']
        transition = row['teacher_to_method'][full_i]
        # teacher_to_method is one four-column total over paid nominal anchors.
        if isinstance(transition[0], list):
            raise ValueError('Unexpected teacher transition dimensions')
        full_zero = row['query_errors'][full_i][1]
        coarse_zero = row['query_errors'][coarse_i][1]
        missed = paid['DESIGN_support'][pair]['recoverable_positive_queries_no_DESIGN_positive_bag_asked']
        sections.append(f'{pair} | {transition[1]} | {transition[2]} | {full_zero} | {coarse_zero} | {missed}\n')
    sections.append('''
FULL常比粗普通对照少出现“整个查询零输出”，却总体HOTA更低。救回更多查询不是充分条件；还要减少错误轨迹与逐帧漏选。
已封存问答里根本没问到的目标，不能靠重加权凭空补出新证据。后续要换取证方式，并把新反馈同时给普通对照。

七、新模型确有逐帧不稳；不能套用旧v8P的零翻转结论

同TrackID相邻帧、同canonical正GT ID且成员状态稳定：下面是输出翻转数量/稳定正相邻对数。
''')
    for pair in LABELS:
        row = temporal['settings'][pair]
        denom = row['denominators']['stable_same_positive_ID_pairs']
        flips = row['methods']['FULL_FINAL']['positive_ID_stable_output_flips']
        sections.append(f'{pair}：{flips}/{denom} = {100*flips/denom:.4f}%\n')
    sections.append('''
两个KITTI设置约15%–16%的稳定正相邻对翻转；这是新FULL的事实，不与旧v8P零翻转混用。
但粗轨迹常数输出零翻转是构造性质，它也错过真正的状态变化。不能据此盲加EMA/HMM或把所有帧锁成常数。
canonical成员变化还包含几何IoU变化；负负邻对并不保证真实身份相同。该诊断不能独立证明翻转解释了全部HOTA缺口。

八、为什么不能给旧profile和潜在门变量硬贴理论名字

同一标量教师概率构造的profile满足已有KL链分解，不自动增加独立教师信息；逐点完全拟合时该profile项也为零。
只观测q=a*r时，a和r一般不唯一，a不能冒称已经学到真实身份/存在性。共享a(C)还需要正确的共同条件信息与结构假设。
经典思想应决定取什么证据、什么可辨识、什么应共享以及何时更新；不是把旧分数再套命名校准器。

九、下一步候选：联合观察，而不是继续调种子/堆神经模块

人的观察方式是把同一个人的两次观察联系起来，既保留“是同一个候选”的线索，也允许动作/关系改变。
拟测试冻结教师看A/B两组图像、完整原查询，回答：只有A符合、只有B符合、都符合、都不符合。
这支持多目标和目标不存在，不强迫二选一，也不拆坏原句语义。
单次边缘概率(.5,.5)既可能来自独立不确定，也可能来自完全共享不确定；联合概率可以区分二者。
在假设Y_A=Z*S_A、Y_B=Z*S_B且给定共同上下文时局部状态独立的模型里，
  qA=P10+P11，qB=P01+P11，b=P11，
  a=qA*qB/b，rA=b/qB，rB=b/qA。
该参数化只在非零和非负协方差等条件下可辨识；负相关或退化情况不能被静默删除，也不能把a解释成已验证的真实身份概率。
合成核验最大参数误差1.11e-16，只是代数检查，不是新定理、真实语义验证或HOTA提升。
同样，KL(Qjoint || Bern(qA)*Bern(qB)) = I_Q(A;B)+KL(QA||qA)+KL(QB||qB)。
所以把联合目标喂给仍然完全独立的两个输出，本质仍是边缘蒸馏加常数，不能当新学习机制。
后续如果联合证据可用，需要真正建模共享和局部变化，允许模型失配，并给普通对照充分的一般四分类joint-KD。对手不只给两个边缘答案。

物理pilot预先固定24组无标签案例、3个目标图像集、144个真实新输入、六种控制：
JOINT_AB、JOINT_BA、各自原二元提问A/B、相同两组上下文的二元提问A/B。
交换顺序映射回原A/B；报告格式遵循、顺序敏感、与二元边缘差异、共同支持模型适配率。
不依据GT选案例、提示或模型参数。不用格式成功冒充准确率，不把pilot当六设置质量通过。
''')
    pilot = PUBLIC/'joint_observation_teacher_pilot_analysis.json'
    if pilot.is_file():
        report = json.loads(pilot.read_text())
        sections.append('物理pilot实际聚合收据：\n'+json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    else:
        sections.append('报告生成时物理pilot尚未完成；当前没有新的联合反馈质量结论。\n')
    sections.append('''
十、接续研究与成本约束

只有pilot证据支持时，才为统一新方案登记完整六设置、seed0、充分边缘/一般joint-KD、必要模块和交互对照。
同候选、同图像与完整查询、同教师联合回答和实际预算；所有输出封存后再评价。联合输入变贵要真实记账，不能只按“一个逻辑问题”声称省钱。
假设不满足时应修模型或报告不适用，不能删除难例或偷偷用目标GT改阈值。三创新是研究目标，不是已经得到的三个正结果。
当前FULL被否决，不追加数小时冷部署。最终保留方案必须在当前A100同版本真实特征、教师反馈上重拟合和评价实际质量，并记录学生/教师加载、提取、真实前向、拟合、打分和输出总墙钟。
历史三帧重建pilot仍有FP16微差；不把历史质量和新成本拼接，不用容差替代真正输入。NAS只读，历史results不覆盖，权重/原始QA/特征/预测不入Git。
用户的目标尚未完成。本报告如实保留负结果，为下一步设计提供证据，不以队列、报告或pilot完成关闭Goal。

十一、完整37项原始方法表（每列同一seed0，六个设置）

method | C1 | C2 | C3 | C4 | C5 | C6
''')
    methods = data['repeated']['C1']['mean_HOTA']
    for method in methods:
        sections.append(method+' | '+' | '.join(f"{data['repeated'][p]['mean_HOTA'][method]:.9f}" for p in LABELS)+'\n')
    sections.append('''
十二、可追溯入口（仓库相对路径；大资产只在本地/NAS）

docs/a100_six_direction_20261122/conditional_seed0_scope_analysis.json
docs/a100_six_direction_20261122/conditional_seed0_scope_parent_statistics.json
docs/a100_six_direction_20261122/conditional_seed0_scope_statistical_reproduction.json
docs/a100_six_direction_20261122/conditional_seed0_evidence_support_analysis.json
docs/a100_six_direction_20261122/conditional_seed0_input_alignment.json
docs/a100_six_direction_20261122/conditional_seed0_temporal_diagnosis.json
docs/a100_six_direction_20261122/conditional_seed0_fit_cost.json
docs/a100_six_direction_20261122/joint_observation_teacher_pilot_plan.json
docs/a100_six_direction_20261122/joint_observation_moment_audit.json
scripts/a100_conditional_seed0_scientific_review.py（只汇总已有证据，不训练、不读取原始GT）
''')
    TEXT.write_text(''.join(sections))
    report = dict(status='COMPLETE_SINGLE_SEED_SCIENTIFIC_REVIEW_PRIMARY_REJECTED_GOAL_ACTIVE',
        seeds=[0], fixed_ordinary='COARSE_SOFT_MEAN', settings=summary,
        primary_fails_all_six=True, primary_not_replaced_by_best_ablation=True,
        statistical_reproduction=proof, paid_anchor_features_exact=alignment['exact_numeric_equal'],
        all_data_development=True, independent_blind_validation=False,
        no_new_GT_reads_fits_teacher_backbone_or_quality_outputs=True,
        no_full_cold_for_rejected_recipe=True, next_joint_candidate_not_accepted=True,
        report=str(TEXT.relative_to(ROOT)), goal_status='ACTIVE')
    atomic_json(PUBLIC/'conditional_seed0_scientific_review.json', report)
    print(json.dumps(dict(status=report['status'], text=str(TEXT), bytes=TEXT.stat().st_size)))


if __name__ == '__main__':
    main()
