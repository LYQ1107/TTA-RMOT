"""Review three completed seed0 cohorts; optionally pack only new statistics.

Reads sealed aggregate results, never images, teacher answers or target labels.
The registered experiment sources and all already published providers stay fixed.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PUBLIC = ROOT / 'docs/a100_six_direction_20261122'
STEMS = {name: f'native_binary_{name}_quality_seed0_20261124'
         for name in ('trust', 'comparable', 'utility')}
EXPECTED = {'trust': 312, 'comparable': 318, 'utility': 324}
REVIEW = PUBLIC / 'completed_rsi_quality_review_20261125.json'
REPORT = ROOT / 'docs/TTA_RMOT_RSI三项完整质量结果与机制结论_20261125.txt'
BACKUP = Path('/var/tmp/liuyeqiang-tta-rmot-a100-20261122/completed_rsi_quality_publication_20261125/originals')
SOFT = 'COARSE_UNIT_SOFT_LEARNED_FINAL'
MARGINAL = 'MARGINAL_KD_LEARNED_FINAL'
FROZEN = 'FULL_CONDITIONAL_REFERENCE_FROZEN_FINAL'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')


def completed():
    analysis, reproduction = {}, {}
    for name, stem in STEMS.items():
        analysis[name] = read(PUBLIC / (stem + '_analysis.json'))
        reproduction[name] = read(PUBLIC / (stem + '_reproduction.json'))
        proof = reproduction[name]
        assert proof['status'].startswith('PASS_')
        assert proof['statistical_records'] == EXPECTED[name]
        assert proof['maximum_absolute_error'] < 1e-8
        assert analysis[name]['neural_seeds'] == [0]
        assert analysis[name]['all_outputs_sealed_before_GT']
    return analysis, reproduction


def pack_new():
    from scripts.a100_visual_classical_sparse_statistics_reproduce import (
        resolve_record, share_identical_statistics, share_identical_parent_groups)
    completed()
    targets = [PUBLIC / (STEMS[n] + '_parent_statistics.json') for n in ('trust', 'comparable')]
    for path in targets:
        tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', str(path.relative_to(ROOT))],
                                 cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert tracked.returncode != 0, 'Only unpublished providers may be repacked: ' + str(path)
    assert not BACKUP.exists(), 'Original archive already exists; inspect it instead of overwriting it.'
    BACKUP.mkdir(parents=True)
    originals = {}
    for stem in STEMS.values():
        for suffix in ('analysis', 'parent_statistics', 'reproduction'):
            path = PUBLIC / (stem + '_' + suffix + '.json')
            data = path.read_bytes(); (BACKUP / path.name).write_bytes(data)
            originals[path.name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    cache, before = {}, {}
    for name, stem in STEMS.items():
        filename = stem + '_parent_statistics.json'
        before[name] = {key: resolve_record(PUBLIC, filename, key, cache)
                        for key in read(PUBLIC / filename)}
        assert len(before[name]) == EXPECTED[name]
    trust_name = STEMS['trust'] + '_parent_statistics.json'
    trust, trust_shared = share_identical_statistics(copy.deepcopy(before['trust']), trust_name)
    trust, trust_groups = share_identical_parent_groups(trust)
    write(PUBLIC / trust_name, trust)
    owners = {}
    for key, row in before['trust'].items():
        token = json.dumps(row, sort_keys=True, separators=(',', ':'))
        owners.setdefault(token, key)
    corrected, cross = {}, 0
    for key, row in before['comparable'].items():
        token = json.dumps(row, sort_keys=True, separators=(',', ':'))
        if token in owners:
            corrected[key] = dict(reference=dict(file=trust_name, record=owners[token])); cross += 1
        else:
            corrected[key] = copy.deepcopy(row)
    corrected_name = STEMS['comparable'] + '_parent_statistics.json'
    corrected, corrected_shared = share_identical_statistics(corrected, corrected_name)
    corrected, corrected_groups = share_identical_parent_groups(corrected)
    write(PUBLIC / corrected_name, corrected)
    fresh_cache, checked = {}, 0
    for name, stem in STEMS.items():
        filename = stem + '_parent_statistics.json'
        assert set(read(PUBLIC / filename)) == set(before[name])
        for key, original in before[name].items():
            assert resolve_record(PUBLIC, filename, key, fresh_cache) == original, (name, key)
            checked += 1
    assert checked == 954
    # The utility provider already shares its new rows and references the
    # corrected provider; preserve its bytes while checking every nested link.
    assert (PUBLIC / (STEMS['utility'] + '_parent_statistics.json')).read_bytes() == (
        BACKUP / (STEMS['utility'] + '_parent_statistics.json')).read_bytes()
    receipt = dict(status='PASS_EXACT_NEW_ONLY_THREE_COHORT_STATISTIC_PACKING',
        records_restored_exactly=checked, no_float_rounding=True,
        trust_shared_records=trust_shared, trust_shared_parent_lists=trust_groups,
        corrected_cross_provider_equal_records=cross,
        corrected_shared_own_records=corrected_shared,
        corrected_shared_parent_lists=corrected_groups,
        utility_provider_bytes_unchanged=True, already_published_providers_unchanged=True,
        originals=originals,
        packed_bytes={p.name: p.stat().st_size for p in targets},
        full_original_independent_reproductions_required_after_packing=True,
        new_GT_teacher_models_fits_predictions_or_seeds=0)
    write(PUBLIC / 'completed_rsi_quality_publication_pack_20261125.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))


def compact_effect(effect):
    return dict(delta=effect['delta_HOTA'], CI95=effect.get('fixed_neural_seeds_parent_CI95', effect.get('CI95')),
                parent_groups=effect['original_parent_groups'])


def effect_text(effect):
    ci = effect['CI95']
    return f"{effect['delta']:+.6f}" + (f" [{ci[0]:+.6f}, {ci[1]:+.6f}]" if ci is not None else ' [4个原始视频，不报总体CI]')


def review():
    a, reproduction = completed(); corrected = a['comparable']; utility = a['utility']
    rows = {}
    for pair, setting in corrected['settings'].items():
        raw = setting['raw_HOTA']; comparisons = setting['comparisons']; u = utility['settings'][pair]
        ordinary = [n for n in raw if not n.startswith('FULL_')]; assert len(ordinary) == 16
        rows[pair] = dict(source_target=setting['source_target'],
            trust_primary=a['trust']['settings'][pair]['raw_HOTA'][a['trust']['primary']],
            corrected_primary=raw[corrected['primary']],
            same_marginal_final=raw[MARGINAL], same_soft_final=raw[SOFT],
            utility_primary=u['HOTA'][utility['primary']],
            ordinary_wins={label: sum(comparisons[label + '_vs_' + n]['delta_HOTA'] > 0 for n in ordinary)
                           for label in ('RAW', 'AT_PRIMARY_COUNT', 'AT_OTHER_COUNT')},
            fixed_soft={label: compact_effect(comparisons[label + '_vs_' + SOFT])
                        for label in ('RAW', 'AT_PRIMARY_COUNT', 'AT_OTHER_COUNT')},
            learned_vs_frozen={label: compact_effect(comparisons[label + '_vs_' + FROZEN])
                               for label in ('RAW', 'AT_PRIMARY_COUNT', 'AT_OTHER_COUNT')},
            correction={k: compact_effect(v) for k, v in setting['correction_vs_parent'].items()},
            utility_effects={k: compact_effect(u['comparisons'][k]) for k in (
                'READOUT_AT_LEARNED_COUNT', 'READOUT_AT_FROZEN_COUNT', 'RSI_RAW',
                'RSI_AT_LEARNED_COUNT', 'RSI_AT_FROZEN_COUNT')},
            observer=setting['observer'])
    result = dict(status='CURRENT_RSI_RECIPE_NOT_ACCEPTED_NATIVE_FACTORIAL_PENDING',
        neural_seeds=[0], candidate_accepted=False, all_six_ordinary_KD_superiority=False,
        three_evidenced_contributions_complete=False, native_726_factorial_complete=False,
        true_same_cohort_cold_quality_cost_complete=False, all_data_DEV=True,
        nominal_parent_CI_not_blind_generalization_or_multiple_comparison_proof=True,
        records=[312, 318, 324], actual_new_official_evaluations=[312, 108, 120],
        exact_parent_references=[0, 210, 204], reproductions=reproduction, settings=rows,
        numerical_sources={n: STEMS[n] + '_analysis.json' for n in STEMS},
        goal_status='ACTIVE')
    write(REVIEW, result)
    lines = [
        'TTA-RMOT：RSI三项完整质量结果与机制结论', '2026-11-25；训练固定seed0；全部是开发数据', '',
        '一、先给结论',
        '当前这版RSI主方案没有达到六设置优于充分普通蒸馏的目标，不能作为已成立的论文主方法。',
        '同一种MARGINAL_KD_LEARNED_FINAL在六设置全部高于修正后的主方法。',
        '固定soft基线下，主方法原始HOTA五负一微正；双向逐query/frame CountMatched六设置均负。',
        '只换utility读出，在同权重、同特征、同输出数量下六设置全部下降；该读出方案不能采用。',
        '学习观察者确实改变了问答和学生，但其精度贡献混合且随数量条件变化，没有稳定RSI优势。',
        '这否定的是当前具体配方；不否定跨域RMOT设定，也不能外推成所有RSI、人类视觉启发或经典思想都无效。', '',
        '二、研究范围与完整性',
        'TRUST312：108原始输出+204数量对照；全部312项官评完成。',
        '实际视角修正318：108项新增官评+210项精确父引用；四个不受影响设置直接复用，不是独立复验。',
        'utility324：120项新增官评+204项父引用；零新增拟合/教师/骨干前向。',
        '954是三份结果表的记录数；实际新增官评为540项，414项是父引用，不能当954个独立样本。',
        '三套均在本研究所有预测/数量对照封存、GT计数为0后才隔离官评；独立NumPy复算已全通过。',
        '复算覆盖312/318/324条统计、306/306/318个配对效应，以及18个视角修正效应。',
        '独立复算检查的是封存输出的统计；不等于重新训练或独立盲测。',
        '沿用官方候选与Track ID，只研究目标图片/查询/冻结教师指导的跨域指代适配；不宣称检测前端source-only。',
        '所有普通对手按各自原定目标充分拟合；.005门槛含KL和教师logit平方误差，各有原单位，不能统称KL。',
        'SportsMOT的45片段按13个原始视频分组；Dance为25组；KITTI只有4原始视频，不报告总体置信区间。',
        '其它区间是固定神经seed0的开发集父视频bootstrap描述，非额外神经种子、非盲测或多重比较后的保证。', '',
        '三、六设置原始HOTA（每列保持同一方法，未逐设置挑最好基线）',
        '设置(来源→目标)  原TRUST主方法  修正主方法  固定MARGINAL_FINAL  固定SOFT_FINAL  utility主方法']
    for p, r in rows.items():
        label = '→'.join(r['source_target'])
        lines.append(f"{p}({label})  {r['trust_primary']:.6f}  {r['corrected_primary']:.6f}  {r['same_marginal_final']:.6f}  {r['same_soft_final']:.6f}  {r['utility_primary']:.6f}")
    lines += ['', 'MARGINAL和SOFT均是已登记普通对手中的固定例子；完整16种普通输出全部保留在原分析及下方矩阵。',
              '四、固定SOFT参照的双向数量诊断',
              '每行依次为：原始差值；固定主方法数量；固定普通方法数量。区间为描述性95%父视频区间。']
    for p, r in rows.items():
        lines.append(p + '  ' + '；'.join(effect_text(r['fixed_soft'][k]) for k in ('RAW', 'AT_PRIMARY_COUNT', 'AT_OTHER_COUNT')))
    lines += ['', 'C6原始仅+0.115857且区间跨0；两种固定数量后转为负值，不能当语义排序涨点。',
              '五、学习观察者对冻结观察者：正负格子全部保留',
              '原读出每行依次为RAW、固定LEARNED数量、固定FROZEN数量；不能只挑正的一列。']
    for p, r in rows.items():
        lines.append(p + '  ' + '；'.join(effect_text(r['learned_vs_frozen'][k]) for k in ('RAW', 'AT_PRIMARY_COUNT', 'AT_OTHER_COUNT')))
    lines += ['C3在LEARNED数量下+0.419730、区间为正；在FROZEN数量下-0.603287、区间为负。这个正格子保留，但不是双向一致的收益。',
              'C2/C5也出现数量条件改变后符号相反；四视频不给总体CI。C1/C4原始小正值的区间均跨0。',
              '观察者更新及付费预算：设置 / LEARNED实际更新 / FROZEN更新 / 两策略请求数 / 最大记录状态KL']
    for p, r in rows.items():
        l, f = r['observer']['LEARNED'], r['observer']['FROZEN']
        assert l['paid_requests'] == f['paid_requests']
        lines.append(f"{p}  {l['updates']}  {f['updates']}  {l['paid_requests']}/{f['paid_requests']}  {l['max_update_KL']:.8f}")
    lines += ['参数与问题改变证明闭环发生，不证明RMOT改进；局部KL约束也不是任务收益保证。', '',
              '六、utility读出：相同权重与数量下的纯输出机制实验',
              '每行依次为：LEARNED权重/数量下utility替换旧读出的效应；FROZEN权重/数量下的效应。']
    for p, r in rows.items():
        lines.append(p + '  ' + '；'.join(effect_text(r['utility_effects'][k]) for k in ('READOUT_AT_LEARNED_COUNT', 'READOUT_AT_FROZEN_COUNT')))
    lines += ['LEARNED同数量下六设置全负。它是另一个条件集合模型的MAP读出，不是原pair网络的MAP，也不是新的特征学习。',
              '原ring读出的Track ID依赖仍是问题；这次更换读出失败，不能倒过来证明ID依赖合理。',
              '剩余问题是表示、训练目标与全局输出规则是否一致，不能再假设只修输出接线就能涨点。',
              '固定utility读出再比较RSI，依次为RAW、固定LEARNED数量、固定FROZEN数量：']
    for p, r in rows.items():
        lines.append(p + '  ' + '；'.join(effect_text(r['utility_effects'][k]) for k in ('RSI_RAW', 'RSI_AT_LEARNED_COUNT', 'RSI_AT_FROZEN_COUNT')))
    lines += ['', '七、实际视角修正不是新的涨点模块',
              'C2/C5修正仅让critic排除CLIP回退后实际视角不一致的probe对；原付费与普通KD证据均保留。',
              '每行依次为：修正减原方案RAW、固定原数量、固定修正数量。']
    for p, r in rows.items():
        lines.append(p + '  ' + '；'.join(effect_text(r['correction'][k]) for k in ('RAW', 'AT_PARENT_COUNT', 'AT_CORRECTED_COUNT')))
    lines += ['四个未受影响设置的0来自精确复用，不是四次新复验。同视角不自动保证同帧或教师正确。', '',
              '八、完整16种普通输出矩阵（修正研究；差值均为主方法减该普通输出）',
              '列：设置 / 普通输出 / 普通HOTA / RAW差 / 主方法数量下差 / 普通数量下差。完整CI见原analysis.json。']
    for p, s in corrected['settings'].items():
        for n, h in s['raw_HOTA'].items():
            if n.startswith('FULL_'): continue
            e = s['comparisons']
            lines.append(f"{p} {n} {h:.6f} " + ' '.join(f"{e[k+'_vs_'+n]['delta_HOTA']:+.6f}" for k in ('RAW', 'AT_PRIMARY_COUNT', 'AT_OTHER_COUNT')))
    lines += ['', '九、边界与下一步',
              '本轮没有三个已证实的创新点，也没有六设置普通蒸馏优势；原native726完整因子/交互实验仍在运行。',
              '保留全部正负结果，先等该完整消融，不能把某个事后最高消融晋级为主方法或追加随机种子。',
              '下一轮设计需要解决：可比观察的证据覆盖；训练与全局输出的一致性；教师风险反馈与真实任务收益的偏离。',
              '可比观察/完整动作池目前只有接口与元数据支持，不是已验证的新RMOT算法。具体新方案须依据剩余完整机制证据。',
              '继续沿用跨域RMOT、固定候选、无目标人工标签适配与教师指导的设定；不改变研究目标来迁就失败配方。',
              '这些是历史特征/冻结问答回放质量研究，生成耗时覆盖多个方法，不是单方法真实部署成本。',
              '最终仍需同一次当前A100真实特征、教师、拟合和预测的质量与完整冷墙钟账单；不能拼历史质量和新成本。',
              'Goal保持ACTIVE，等待任务不算阻塞或完成。', '', '十、可复核来源']
    lines += [str((PUBLIC/(stem+'_'+suffix+'.json')).relative_to(ROOT)) for stem in STEMS.values()
              for suffix in ('analysis', 'parent_statistics', 'reproduction')]
    REPORT.write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=result['status'], report=str(REPORT), review=str(REVIEW),
        primary_HOTA={p:r['corrected_primary'] for p,r in rows.items()}, goal_status='ACTIVE'), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--pack-new-statistics', action='store_true')
    args = parser.parse_args()
    pack_new() if args.pack_new_statistics else review()
