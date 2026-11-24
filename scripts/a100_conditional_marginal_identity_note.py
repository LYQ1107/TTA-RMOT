"""Check probability identities behind the completed utility intervention.

Pure deterministic float64 algebra, not a neural experiment or a new decoder.
No images, labels, checkpoints, teacher answers, Torch or GPU are read.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'docs/a100_six_direction_20261122'
RECEIPT = PUBLIC / 'conditional_pair_marginal_identity_20261125.json'
REPORT = ROOT / 'docs/TTA_RMOT_条件排序与成员概率的区别_20261125.txt'


def softmax(values):
    x = np.asarray(values, dtype=np.float64)
    weights = np.exp(x-x.max())
    return weights/weights.sum()


def family(delta, common, interaction):
    """Exact real-number family of GaugeFixedCrossCandidateHead.forward."""
    ua, ub = delta/2, -delta/2
    count = softmax([0., common+np.logaddexp(ua, ub), 2*common+interaction])
    relative = softmax([ua, ub])
    # Native state order is 10, 01, 11, 00.
    joint = np.array([count[1]*relative[0], count[1]*relative[1], count[2], count[0]])
    marginal = np.array([joint[0]+joint[2], joint[1]+joint[2]])
    return count, relative, joint, marginal


def parameters(q0, q1, q2, delta):
    assert min(q0, q1, q2) > 0 and abs(q0+q1+q2-1) < 1e-12
    common = math.log(q1/q0)-float(np.logaddexp(delta/2, -delta/2))
    interaction = math.log(q2/q0)-2*common
    return common, interaction


def main():
    maximum_identity = maximum_count = maximum_curvature_bound = 0.; cases = 0
    for delta, q1, q2 in itertools.product(
            (-8., -2., -.5, 0., .5, 2., 8.), (.05, .2, .5, .8, .95), (.001, .02, .1, .5, .8)):
        q0 = 1-q1-q2
        if q0 <= 0: continue
        common, interaction = parameters(q0, q1, q2, delta)
        count, relative, joint, marginal = family(delta, common, interaction)
        maximum_count = max(maximum_count, float(np.max(np.abs(count-[q0, q1, q2]))))
        maximum_identity = max(maximum_identity,
            abs(float(marginal[0]-marginal[1])-q1*math.tanh(delta/2)))
        curvature = q1*relative[0]*relative[1]
        maximum_curvature_bound = max(maximum_curvature_bound, float(curvature-q1/4))
        assert abs(float(joint.sum())-1) < 1e-13
        cases += 1
    products = 0; maximum_product = maximum_marginal = 0.
    for pa, pb in itertools.product((.02, .1, .3, .5, .7, .9, .98), repeat=2):
        expected = np.array([pa*(1-pb), (1-pa)*pb, pa*pb, (1-pa)*(1-pb)])
        q1, q2, q0 = float(expected[:2].sum()), float(expected[2]), float(expected[3])
        delta = math.log(pa/(1-pa))-math.log(pb/(1-pb))
        common, interaction = parameters(q0, q1, q2, delta)
        _, _, joint, marginal = family(delta, common, interaction)
        maximum_product = max(maximum_product, float(np.max(np.abs(joint-expected))))
        maximum_marginal = max(maximum_marginal, float(np.max(np.abs(marginal-[pa, pb]))))
        products += 1
    utilities = np.array([2., 1., 0.]); pair_rows = {}; pair_marginal = {}
    for a, b in itertools.combinations(range(3), 2):
        delta = float(utilities[a]-utilities[b]); q1 = .1; q2 = .8 if (a, b) == (1, 2) else .001
        q0 = 1-q1-q2; common, interaction = parameters(q0, q1, q2, delta)
        count, relative, joint, marginal = family(delta, common, interaction)
        pair_marginal[a, b] = marginal
        pair_rows[str(a)+'_'+str(b)] = dict(count_0_1_2=count.tolist(), relative_A_B=relative.tolist(),
            native_joint_10_01_11_00=joint.tolist(), marginals_A_B=marginal.tolist(),
            realizable_common=common, realizable_interaction=interaction)
    def ring(order):
        result = np.zeros(3)
        for position, candidate in enumerate(order):
            partners = (order[position-1], order[(position+1) % 3])
            result[candidate] = sum(pair_marginal[min(candidate, p), max(candidate, p)][int(candidate > p)]
                                    for p in partners)/2
        return result
    reference = ring((0, 1, 2)); ranking = np.argsort(-reference, kind='stable').tolist()
    utility_ranking = np.argsort(-utilities, kind='stable').tolist()
    permutation_error = max(float(np.max(np.abs(ring(order)-reference)))
                            for order in itertools.permutations(range(3)))
    assert ranking != utility_ranking and permutation_error < 1e-13
    assert max(maximum_identity, maximum_count, maximum_product, maximum_marginal) < 1e-13
    assert maximum_curvature_bound < 1e-13
    receipt = dict(status='PASS_DETERMINISTIC_CONDITIONAL_MARGINAL_IDENTITIES_AND_COUNTEREXAMPLE',
        family_grid_cases=cases, maximum_pair_difference_identity_error=maximum_identity,
        maximum_count_reparameterization_error=maximum_count, teacher_product_cases=products,
        maximum_product_reconstruction_error=maximum_product,
        maximum_product_marginal_error=maximum_marginal,
        weighted_relative_curvature_upper_bound_violation=max(maximum_curvature_bound, 0.),
        counterexample=dict(utilities=utilities.tolist(), pairs=pair_rows,
            ring_member_probabilities=reference.tolist(), ring_ranking_0_based=ranking,
            utility_ranking_0_based=utility_ranking, fixed_K=1,
            all_six_global_ID_orders_checked=True, maximum_ID_order_probability_error=permutation_error,
            no_target_truth_assigned=True),
        classical_probability_identity_not_claimed_as_original_theorem=True,
        constructed_algebra_not_actual_model_forward_or_quality_result=True,
        does_not_identify_cause_of_empirical_HOTA_loss=True,
        different_pair_input_conditioning_does_not_alone_prove_incompatible_global_probabilities=True,
        new_models_checkpoints_teacher_calls_images_GT_fits_predictions_GPU_or_neural_seeds=0,
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            'flex_uda/active/cross_candidate_gauge.py',
            'flex_uda/active/cross_candidate_learning.py',
            'scripts/a100_native_binary_six_setting.py')}, goal_status='ACTIVE')
    RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, separators=(',', ':'))+'\n')
    text = f'''TTA-RMOT：条件排序与成员概率的区别
2026-11-25；已完成三套质量研究后的公式诊断；不是新RMOT方案或新的涨点实验

一、问题来自真实代码
cross_candidate_gauge.py 的 GaugeFixedCrossCandidateHead 先产生两候选的utility，
再分别计算数量分布和“恰好一人符合”条件下的身份分布。
原dense_scores把两个邻居对的成员概率取平均，再转换为log odds。
已完成的utility研究只用utility排序，同时沿用原输出人数。
完成报告已证明该读出六设置下降；以下解释为什么理论上没有“同人数就应当等价”的保证。
本说明不把经典概率恒等式当原创理论，也不宣称已经找到新的有效创新点。

二、实际参数族的恒等式
记 q0=P(无人符合), q1=P(恰好一人符合), q2=P(两人都符合)，三者和为1。
这些事件仅指当前候选对，不是整帧总人数；没有将多目标RMOT改成全局单目标任务。
delta=uA-uB，r=sigmoid(delta)=P(A符合 | 恰好一人符合)。
代码四种状态依次是10、01、11、00，对应概率 q1*r、q1*(1-r)、q2、q0。
因此 mA=q2+q1*r，mB=q2+q1*(1-r)。
mA-mB=q1*(2*r-1)=q1*tanh(delta/2)。
在同一个固定候选对内，q1>0时utility和成员概率的相对次序相同；q1=0时成员概率打平。
这不等于跨不同候选对、不同q1/q2时，所有对象的utility次序都等于原聚合成员次序。
CountMatched固定的是最终每query/frame的输出数量，并没有把原神经网络变成一个
在完整候选集合上条件于K的全局联合概率模型；它不会自动消除q1/q2携带的信息。

三、当前参数族内的三对象反例
A/B/C的utility为2/1/0。每一对q1=.1。
AB和AC的q2=.001、q0=.899；BC的q2=.8、q0=.1。
这些都是严格正的合法局部分布，并能由实际gauge参数族的common/interaction精确表示：
common=log(q1/q0)-log(2*cosh(delta/2))；interaction=log(q2/q0)-2*common。
这里验证输出链接函数层面的可实现性，没有构造实际图像、特征或神经权重。
按原代码的双邻居概率平均，对象成员分数为 {reference.tolist()}。
原聚合排序为 {[chr(65+i) for i in ranking]}，utility排序为 {[chr(65+i) for i in utility_ranking]}。
即使固定K=1，两种读出也会选不同对象。本反例没有给任何对象设置人工GT，不评价谁语义正确。
三对象环包含全部无序候选对，因此6种全局ID顺序都给出同一聚合结果。
这说明utility与原读出的差别可以在没有ID顺序变化的情况下发生；
不能把utility替换理解成只删除ID干扰、其它信息完全保留的干净干预。

四、哪些额外条件下utility确实足够
如果同一固定上下文中有对象概率pA,pB，严格位于0和1之间，且以独立Bernoulli乘积
构造目标，那么 q1=pA*(1-pB)+(1-pA)*pB，r=pA*(1-pB)/q1。
此时 logit(r)=logit(pA)-logit(pB)，还原的成员边缘恰为pA、pB。
若同一上下文中的比较图连通、相对意见一致且拟合精确，utility只差一个公共常数，
全局排序可以与这些对象概率一致。有限稀疏观察、跨帧配对、模型误差不自动满足这些条件。
目前代码中的乘积目标是建模假设，不是实际向教师测得的跨对象联合回答，更不是人工真值。
若所有对分布真是同一个固定完整场景条件下的两两边缘，则对象边缘还应随搭档不变。
当前pair网络只输入两个候选的特征，换搭档会换条件信息；不能单凭搭档敏感就宣布
实际概率矛盾。反例证明的是当前局部参数族没有所需的等价保证，不是对真实数据的全局一致性判决。

五、经典条件学习的信息边界
conditional_estimation对数量路径中的utility停止梯度；相对项受目标恰好一人概率加权。
记t1为教师乘积目标的恰好一人质量，t为该目标的条件身份概率；它们区别于模型的q1/r。
相对交叉熵为 -t1*[t*log(sigmoid(delta))+(1-t)*log(1-sigmoid(delta))]。
一阶导数 t1*(sigmoid(delta)-t)，曲率 t1*sigmoid(delta)*(1-sigmoid(delta)) <= t1/4。
恰好一人目标质量很小时，该相对项的梯度和曲率可能很弱；大量“两个都不符合”的证据仍可训练数量分支。
这是条件似然的经典信息权衡，不是当前全部HOTA下降的已证实成因，也不证明应当降低原反馈门槛。

六、与人类视觉方式及下一轮工作的关系
“两者中谁更像”和“这两个人是否真的符合”是不同判断。对应建模时要明确条件与信息集合，
同时保持对象身份和全局输出的一致性，不能只给后处理换个经典算法名字。
utility失败不能单独否定条件学习；当前未完native726的条件/完整似然、局部/参照与身份
模块重训消融仍是必要证据。当前也不能宣称数量分支已经学到正确语义，或所有ID不变读出都无效。
本轮没有新网络、没有新训练或质量矩阵；继续原六设置、目标图片与冻结教师的无人工标签适配设定。
新模型和观察设计必须等剩余完整机制结果后再决定，三个实证贡献及全六设置KD优势仍未完成。

七、实际验证范围
使用纯NumPy float64验证{cases}组参数、{products}组独立乘积目标，检查三对象的6种ID顺序。
最大恒等式误差 {maximum_identity:.17g}；最大乘积还原误差 {maximum_product:.17g}。
这是确定性的代数验证，不是神经随机种子实验，不读取图像/标签/检查点/教师答案，不产生RMOT预测。
可运行 scripts/a100_conditional_marginal_identity_note.py；完整数值收据见
docs/a100_six_direction_20261122/conditional_pair_marginal_identity_20261125.json。
实际质量证据仍以已发布《TTA_RMOT_RSI三项完整质量结果与机制结论_20261125.txt》为准。
Goal保持ACTIVE。
'''
    REPORT.write_text(text)
    print(json.dumps(dict(status=receipt['status'], grid_cases=cases, product_cases=products,
        maximum_identity_error=maximum_identity, counterexample_member_probabilities=reference.tolist(),
        report=str(REPORT), new_models_or_GT=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
