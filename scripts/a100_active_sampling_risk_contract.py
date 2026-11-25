"""Deterministic enumeration of classical active-sampling risk identities.

Fixed synthetic loss vectors only: no fitting, random draws, teacher, models,
images, predictions, or target GT. PURE/LURE are credited to prior work.
"""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'docs/a100_six_direction_20261122'


def proposal(remaining, observed_losses, features, adaptive):
    """Only known features and already revealed losses enter acquisition."""
    if not adaptive:
        return np.full(len(remaining), 1 / len(remaining))
    x = features[list(remaining)]
    past = float(np.mean(observed_losses)) if observed_losses else 0.
    scores = .7*x + .3*past*x*x
    weights = np.exp(scores - scores.max())
    return .8*weights/weights.sum() + .2/len(remaining)


def enumerate_risk(n, budget, adaptive):
    losses = .1 + np.arange(n, dtype=np.float64)**2/(n-1)
    features = np.linspace(-1.2, 1., n)
    outcomes = []

    def walk(remaining, observed, probabilities, path_mass):
        if len(observed) == budget:
            values = np.asarray(observed)
            q = np.asarray(probabilities)
            m = np.arange(1, budget+1, dtype=np.float64)
            pure = (1/q + budget-m)/n
            lure = (np.ones(budget) if budget == n else
                    1+(n-budget)/(n-m)*(1/((n-m+1)*q)-1))
            estimates = [values.mean(), (values/(n*q)).mean(),
                         (pure*values).mean(), (lure*values).mean()]
            outcomes.append((path_mass, estimates))
            return
        q = proposal(remaining, observed, features, adaptive)
        assert np.all(q > 0) and abs(q.sum()-1) < 1e-14
        for at, index in enumerate(remaining):
            walk(tuple(i for i in remaining if i != index), observed+[float(losses[index])],
                 probabilities+[float(q[at])], path_mass*float(q[at]))

    walk(tuple(range(n)), [], [], 1.)
    mass = np.asarray([r[0] for r in outcomes])
    estimates = np.asarray([r[1] for r in outcomes])
    expectation = (mass[:, None]*estimates).sum(0)
    truth = float(losses.mean())
    assert abs(mass.sum()-1) < 1e-12
    assert max(abs(expectation[2:]-truth)) < 1e-11
    return dict(N=n, M=budget, adaptive=adaptive, enumerated_paths=len(outcomes),
                path_probability_sum=float(mass.sum()), fixed_population_risk=truth,
                expected_naive_mean=float(expectation[0]),
                expected_incorrect_inverse_conditional_only=float(expectation[1]),
                expected_PURE=float(expectation[2]), expected_LURE=float(expectation[3]))


def broker_sequence_fields():
    tree = ast.parse((ROOT/'flex_uda/active/native_binary_inputs.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BinaryBroker')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'request')
    rows = []
    for node in ast.walk(method):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and ast.unparse(node.func) == 'self.sequence.append'):
            value = node.args[0]
            assert isinstance(value, ast.Call) and ast.unparse(value.func) == 'dict'
            rows.append([k.arg for k in value.keywords])
    assert rows == [['uid', 'role', 'stage']]
    return rows[0]


def main():
    cases = [enumerate_risk(n, m, adaptive) for n in range(3, 7)
             for m in range(1, n+1) for adaptive in (False, True)]
    error = max(max(abs(r['expected_PURE']-r['fixed_population_risk']),
                    abs(r['expected_LURE']-r['fixed_population_risk'])) for r in cases)
    naive_example = next(r for r in cases if r['N'] == 4 and r['M'] == 2 and not r['adaptive'])
    assert abs(naive_example['expected_incorrect_inverse_conditional_only']
               - naive_example['fixed_population_risk']) > .01
    first = np.array([.2, .8]); second = np.array([.4, .4])
    mean_log_error = abs(float(np.log(first).mean()-np.log(second).mean()))
    assert mean_log_error < 1e-14 and not np.array_equal(1/first, 1/second)
    # An unobservable unit's loss can change without changing the observed law.
    low = np.array([.2, .4, 0., 0.]); high = np.array([.2, .4, 2., 2.])
    assert np.array_equal(low[:2], high[:2]) and low.mean() != high.mean()
    paths = ('flex_uda/active/rsi_curriculum.py', 'flex_uda/active/native_binary_inputs.py',
             'flex_uda/active/native_binary_curriculum.py', 'flex_uda/active/rsi_policy_distribution.py')
    result = dict(status='PASS_CLASSICAL_ACTIVE_SAMPLING_RISK_IDENTITIES_AND_SCOPE',
        fixed_loss_cases=len(cases), enumerated_sample_paths=sum(r['enumerated_paths'] for r in cases),
        maximum_PURE_LURE_expectation_error=error, cases=cases,
        incorrect_inverse_conditional_only_example=naive_example,
        mean_log_probability_insufficiency=dict(first_selected_probabilities=first.tolist(),
            second_selected_probabilities=second.tolist(), mean_log_error=mean_log_error,
            different_inverse_weights=True),
        zero_support_counterexample=dict(observable_indices=[0, 1],
            observable_losses_same=low[:2].tolist(), first_population_risk=float(low.mean()),
            second_population_risk=float(high.mean()), unobserved_loss_changes_not_identifiable=True),
        current_BinaryBroker_sequence_append_fields=broker_sequence_fields(),
        aggregate_log_probability_not_individual_proposal_probability=True,
        teacher_Yes_probability_not_acquisition_probability=True,
        vanilla_LURE_assumptions_not_established_for_current_group_priority_and_probes=True,
        missing_propensity_field_not_proof_reconstruction_is_impossible=True,
        fixed_model_finite_pool_risk_not_data_selected_model_or_semantic_HOTA_guarantee=True,
        no_claim_sampling_bias_caused_current_empirical_HOTA_loss=True,
        PURE_LURE_are_prior_work_not_original_contributions=True,
        no_new_images_teacher_models_fits_predictions_GT_GPU_or_neural_seeds=True,
        source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        primary_sources=[dict(title='On Statistical Bias In Active Learning: How and When To Fix It',
            url='https://arxiv.org/abs/2101.11665', scope_read='abstract and author implementation'),
            dict(title='Active Testing: Sample-Efficient Model Evaluation',
                 url='https://proceedings.mlr.press/v139/kossen21a/kossen21a.pdf',
                 scope_read='PDF; equations3-4 and fixed-evaluated-model scope'),
            dict(url='https://github.com/SebFar/statistical_bias_in_active_learning/blob/main/dataloader.py',
                 file_git_blob_sha1='cc158d0eff31b7eddab6d210fc38233d64679082',
                 file_sha256='83f5f01e7c0af26fa4b8049398f94a5393cf9abfdb15621bbedc516c106c61a9',
                 full_file_blob_verified=True, third_party_code_executed=False)],
        goal_status='ACTIVE')
    receipt = PUBLIC/'active_sampling_risk_contract_20261125.json'
    receipt.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':'))+'\n')
    report = f'''TTA-RMOT：主动取证的采样概率与风险估计边界
2026-11-25；经典理论核对与确定性代数验证；没有新RMOT质量实验

一、与当前问题的联系
原始选题学习的参数变化几乎没有改变实际提问，TRUST虽改变了问题却未胜过充分普通KD。
教师训练拟合达标也不代表未询问候选预测正确。下一轮需要区分学什么、如何选题、如何判断改进。
采样偏差只是一个待验证的因素；本说明没有证明它造成了当前HOTA下降，也没有据此选新主方法。

二、两项原始工作的实际启发
Farquhar等的ICLR2021工作研究主动采样造成的统计偏差，提出PURE/LURE；
其摘要也明确指出偏差有时有利于过参数神经网络，不能据此承诺去偏必然涨点。
本轮核对其摘要与作者完整dataloader.py，文件Git blob与SHA256独立核对通过；未运行第三方代码。
Kossen等ICML2021 Active Testing把高效评估与高效训练区分开，并在固定被评估模型时
用无放回抽样的LURE估计有限池风险。已读其PDF公式3/4及相关评估范围。
这两项是经典统计支撑，不是新的RSI论文或本项目原创贡献。
来源：https://arxiv.org/abs/2101.11665
https://github.com/SebFar/statistical_bias_in_active_learning/blob/main/dataloader.py
https://proceedings.mlr.press/v139/kossen21a/kossen21a.pdf

三、为何不能随手乘一个1/p
先固定N个单位的损失l_i、预算M和被评估模型。第m次从未抽单位中按条件概率q_m选择i_m。
在每个尚未观察单位均有正概率、概率与历史正确记录等条件下：
PURE的样本权重是 (1/q_m + M-m)/N；LURE是
v_m=1+(N-M)/(N-m)*[1/((N-m+1)*q_m)-1]，风险为各v_m*l_i_m的平均。
M=N时直接得到全池平均，脚本显式处理，不做0/0除法。
q_m是给定历史后选中该单位的概率，不是累计纳入概率，也不是教师回答Yes的概率。
仅用l_i_m/(N*q_m)缺少无放回历史修正，连均匀抽样都可能有偏。
本次枚举{len(cases)}组固定总体与预算、共{result['enumerated_sample_paths']}条完整采样路径，
包含仅依赖已观察损失的自适应采样；PURE/LURE最大期望误差{error:.17g}。
这不是随机种子实验，没有抽随机数，也没有用目标数据标签构造损失。

四、旧日志与候选支持的边界
当前BinaryBroker序列写入字段为uid、role、stage；paid记录中的probability是教师Yes概率。
select_round返回选中条件log概率的均值。均值不能代替每个问题的概率：
选中概率(.2,.8)与(.4,.4)的平均log相同，逐例倒数权重却不同。
TRUST的已记录状态平均KL也不是逐问题的采样概率；当前诊断没有证明历史概率绝对无法重建。
在采用相关估计前，应明确总体单位是动作、轨迹、查询还是稠密帧，不能混用其概率。
原动作库之外的候选提问概率为0；库内fresh-track优先、保留probe和逐query配额还构成额外分层约束。
已经观察单位的排除是无放回抽样的正常部分；未观察单位的硬排除需单独处理支持条件。
不能未经证明把上述单一总体LURE公式直接套到当前策略，也不代表其它分层估计一定不可行。
脚本另给出两组总体：可观察损失完全相同，未观察损失不同，全池风险却不同。
缺失支持不能靠重加权补成新证据；应先核定未来采样机制及其支持，不伪造旧日志。

五、对无人工标签RMOT和RSI的约束
即使估计对教师软标签风险无偏，也不意味着对语义真值或HOTA无偏；教师错误仍存在。
模型若用同一批反馈训练或据此选择，固定模型的评估性质不能自动变成泛化保证。
若未来采用此思想，需分别安排学习证据、尚未用于当前更新的评估证据与取证预算，
记录逐次概率、候选集合和选择状态，并明确估计对象。普通KD也应获得相同证据及适用权重。
这只是下一轮方法必须满足的理论与记录条件，尚未设计或运行新的大规模质量方案。
三个实证贡献、六设置可靠KD优势和最终同次fresh冷成本仍未完成。
原native726源、预算、模型和封存顺序不改，继续运行；只用目标图片、查询与冻结教师，Goal ACTIVE。

六、复核
scripts/a100_active_sampling_risk_contract.py
docs/a100_six_direction_20261122/active_sampling_risk_contract_20261125.json
第三方全文代码仅留本地非Git参考目录；公开收据保存来源、文件哈希与全部枚举汇总。
'''
    target = ROOT/'docs/TTA_RMOT_主动取证的采样概率与风险估计边界_20261125.txt'
    target.write_text(report)
    print(json.dumps(dict(status=result['status'], cases=len(cases),
                         paths=result['enumerated_sample_paths'], maximum_error=error,
                         report=str(target), new_GT_or_models=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
