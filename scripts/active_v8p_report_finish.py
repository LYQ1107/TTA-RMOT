"""CPU-only post-seal report composition; no model or raw GT is opened."""
import argparse
import csv
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha

PUBLIC = ROOT / 'results/presence_active/redesign_v8p'
PAIRS = tuple('C' + str(i) for i in range(1, 7))
METHODS = ('MINUS_SLOW', 'YESTRACK_BUDGET_PRIOR', 'UNIFORM_K')


def read_csv(path):
    with path.open() as source:
        return list(csv.DictReader(source))


def write_csv(path, rows):
    with path.open('w') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(rows, keys):
    return '\n'.join(['| ' + ' | '.join(keys) + ' |',
                      '| ' + ' | '.join('---' for _ in keys) + ' |'] +
                     ['| ' + ' | '.join(str(r.get(k, '')) for k in keys) + ' |' for r in rows])


def slow_details():
    """Read already-closed B3 receipts, not GT or an online model."""
    import numpy as np
    execution = json.loads((PUBLIC / 'execution_B3.json').read_text())
    rows = read_csv(PUBLIC / 'B3_summary.csv')
    baseline = {r['pair']: r for r in read_csv(PUBLIC / 'direction_summary.csv')}
    result = []
    inputs = [dict(path=str(p), sha256=file_sha(p))
              for p in (PUBLIC / 'direction_summary.csv', PUBLIC / 'B3_summary.csv')]
    for row in rows:
        folder = Path(execution['active_root']) / row['pair'] / row['method'] / ('rho_' + row['rho'])
        seal_path = folder / 'predictions_seal.json'
        seal = json.loads(seal_path.read_text())
        inputs.append(dict(path=str(seal_path), sha256=file_sha(seal_path)))
        receipts = sorted((folder / 'videos').glob('*/receipt.json'))
        updates = []
        positive = negative = identity_calls = recheck_calls = 0
        for path in receipts:
            receipt = json.loads(path.read_text())
            assert receipt['protocol'] == 'PRESUPPOSED' and not receipt['presence_inference']
            assert receipt['EB']['after']['iterations'] == 0
            question_path = path.parent / 'questions.json'
            questions = json.loads(question_path.read_text())
            assert all(q['kind'] in ('IDENTITY', 'POSITIVE_RECHECK') for q in questions)
            identity_calls += sum(q['kind'] == 'IDENTITY' for q in questions)
            recheck_calls += sum(q['kind'] == 'POSITIVE_RECHECK' for q in questions)
            belief_path = path.parent / 'belief.npz'
            with np.load(belief_path) as belief:
                assert np.all(belief['presence_probability'] == 1)
                assert not np.any(belief['absent'])
                assert not np.any(belief['low_tentative'])
                assert not np.any(belief['output_absent'])
            rounds = [r['slow'] for r in receipt['rounds']]
            updates.append(sum(r['optimizer_updates'] for r in rounds))
            positive += sum(r['actual_optimized_positive_samples'] for r in rounds)
            negative += sum(r['actual_optimized_negative_samples'] for r in rounds)
            inputs.append(dict(path=str(path), sha256=file_sha(path)))
            inputs.extend(dict(path=str(p), sha256=file_sha(p))
                          for p in (question_path, belief_path))
        assert positive == negative == 16 * sum(updates)
        teacher = seal['teacher']
        result.append(dict(pair=row['pair'], method=row['method'], rho=row['rho'],
            HOTA=row['HOTA'], delta_vs_MINUS=row['delta_vs_MINUS'],
            DetA=row['DetA'], AssA=row['AssA'], IDF1=row['IDF1'],
            prediction_rows=row['prediction_rows'],
            delta_DetA_vs_MINUS=float(row['DetA']) - float(baseline[row['pair']]['DetA']),
            delta_AssA_vs_MINUS=float(row['AssA']) - float(baseline[row['pair']]['AssA']),
            delta_IDF1_vs_MINUS=float(row['IDF1']) - float(baseline[row['pair']]['IDF1']),
            CI95_vs_MINUS=row['CI95_vs_MINUS'], videos=len(receipts),
            videos_with_LoRA_updates=sum(n > 0 for n in updates),
            optimizer_updates=sum(updates), optimized_positive_samples=positive,
            optimized_negative_samples=negative, logical_calls=teacher['logical_calls'],
            identity_calls=identity_calls, positive_recheck_calls=recheck_calls,
            presence_calls=0, presence_EM_iterations=0, fixed_presence_checked=True,
            new_calls=teacher.get('new_calls', 0), cache_hits=teacher.get('cache_hits', 0),
            replayed_answers=teacher.get('replayed_answers', 0),
            LoRA_GPU_hours=seal['LoRA_GPU_seconds'] / 3600,
            rescore_GPU_hours=seal['rescore_GPU_seconds'] / 3600))
    write_csv(PUBLIC / 'slow_actual_updates.csv', result)
    atomic_json(PUBLIC / 'slow_updates_inputs.json', dict(
        status='POST_SEAL_ACTUAL_UPDATE_COUNTS', inputs=inputs,
        actual_B3_PRESUPPOSED_receipts_questions_and_beliefs_checked=True,
        interpretation='FIXED_EVIDENCE_SLOW_PATH_WITH_PRESCRIBED_FIXED32_REFRESH_NOT_WEIGHT_ONLY_FACTORIAL',
        raw_GT_files_opened=0, model_or_teacher_calls=0,
        new_experiments_or_selection_rules=False))
    return result


def budget_supplement():
    rows = read_csv(PUBLIC / 'B2_summary.csv')
    assert len(rows) == 54
    lookup = {(r['pair'], float(r['rho']), r['method']): r for r in rows}
    bootstrap = json.loads((PUBLIC / 'B2_bootstrap.json').read_text())
    coverage = {(r['pair'], float(r['rho']), r['method']): r
                for r in read_csv(PUBLIC / 'budget_coverage.csv')}
    comparisons = []
    for pair in PAIRS:
        for rho in (.1, .3, 1.):
            main, yt, uniform = [lookup[pair, rho, m] for m in METHODS]
            mh, yh, uh = [float(r['HOTA']) for r in (main, yt, uniform)]
            comparisons.append(dict(pair=pair, rho=rho, main_HOTA=mh, YT_HOTA=yh,
                UNIFORM_HOTA=uh, main_minus_YT=mh-yh, main_minus_UNIFORM=mh-uh,
                main_minus_best_control=mh-max(yh, uh),
                main_calls=int(main['logical_calls']), YT_calls=int(yt['logical_calls']),
                UNIFORM_calls=int(uniform['logical_calls'])))
    write_csv(PUBLIC / 'budget_comparisons.csv', comparisons)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = dict(MINUS_SLOW='v8P', YESTRACK_BUDGET_PRIOR='YT prior', UNIFORM_K='Uniform')
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for pair, ax in zip(PAIRS, axes.flat):
        for method in METHODS:
            points = [lookup[pair, rho, method] for rho in (.1, .3, 1.)]
            ax.plot([int(r['logical_calls']) for r in points],
                    [float(r['HOTA']) for r in points], marker='o', label=labels[method])
        ax.set(title=pair, xlabel='Actual logical teacher calls', ylabel='HOTA')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PUBLIC / 'budget_actual_calls.png', dpi=160)
    plt.close(fig)

    historical = ROOT / 'results/multidomain_generality/costs.json'
    preparation = ROOT / 'results/presence_active/redesign_v5/active_map_v1_full/common_preparation_costs.json'
    old = json.loads(historical.read_text())
    common = json.loads(preparation.read_text())
    atomic_json(PUBLIC / 'inherited_costs.json', dict(
        classification='HISTORICAL_MEASURED_PROCESS_TIME_NOT_MATCHED_COLD_START_BENCHMARK',
        inputs=[dict(path=str(p), sha256=file_sha(p)) for p in (historical, preparation)],
        earlier_margin_inference=old['per_new_inference'],
        earlier_reused_inferences=old['reused_inferences'], shared_preparation=common,
        notes=[
            'Do not sum duplicate historical margin/feature passes as minimal required deployment.',
            'Discarded source heads and all PCD exports are not all necessary for frozen MINUS.',
            'Fixed detector/tracker generation and official checkpoint pretraining are unmeasured, not zero.',
            'UNIFORM_K does not consume identity ranking; only MOTION queries optionally use frame_presence.',
            'Equal logical calls are not equal view cost, GPU time, or end-to-end cold-start latency.',
            'New/cache/equivalent teacher phase durations and shared-device worker wall time are separate ledgers.']))

    def main_ci(pair, method):
        value = bootstrap[pair + '__' + method + '__1.0_minus_MINUS']['CI95']
        return None if value is None else [-value[1], -value[0]]

    def hota(pair, method):
        return lookup[pair, 1., method]['HOTA']

    counts = [dict(rho=rho, pointwise_wins=sum(r['main_minus_best_control'] > 0
                 for r in comparisons if r['rho'] == rho), directions=6)
              for rho in (.1, .3, 1.)]
    cov_rows = []
    for pair in ('C3', 'C4'):
        for method in ('MINUS_SLOW', 'UNIFORM_K'):
            r = coverage[pair, 1., method]
            cov_rows.append({k: r[k] for k in ('pair', 'method', 'logical_calls',
                'unique_asked_pairs', 'repeated_pair_calls', 'repeat_pair_fraction')})
    text = '### B2适用边界（事后描述，不新增验收或方法选择）\n\n'
    text += table(counts, ['rho', 'pointwise_wins', 'directions']) + '\n\n'
    text += '胜出指主方法HOTA严格高于两项对照中较好者；不是六个独立数据集的统计显著性。主预算仍是事前固定的ρ=.3，不从本曲线重新选预算。\n\n'
    text += ('高预算C3：主方法' + hota('C3', 'MINUS_SLOW') + '，UNIFORM为' + hota('C3', 'UNIFORM_K') + '，'
             '主方法减UNIFORM的95%区间为' + str(main_ci('C3', 'UNIFORM_K')) + '，跨0。\n\n')
    text += ('高预算C4：主方法' + hota('C4', 'MINUS_SLOW') + '，YT为' + hota('C4', 'YESTRACK_BUDGET_PRIOR') +
             '、UNIFORM为' + hota('C4', 'UNIFORM_K') + '；'
             '主方法减YT区间' + str(main_ci('C4', 'YESTRACK_BUDGET_PRIOR')) +
             '，减UNIFORM区间' + str(main_ci('C4', 'UNIFORM_K')) + '，两者均不跨0。不能宣称全预算支配性。\n\n')
    text += '高预算C2/C5也低于UNIFORM，但KITTI只有4视频，仅报告点值，不造置信区间。Dance两个方向在三档预算点值均胜出；完整18格对比见budget_comparisons.csv。\n\n'
    text += table(cov_rows, ['pair', 'method', 'logical_calls', 'unique_asked_pairs',
                            'repeated_pair_calls', 'repeat_pair_fraction']) + '\n\n'
    text += '覆盖统计只读封存问题与MOT元数据，GT打开0、模型/教师调用0。复问是同一物理对换帧/看法，不是缓存键重复；这是复核与覆盖的观察差异，不能单独证明掉分原因。不新增因果实验，不据此改获取规则。\n\n'
    text += '![实际调用量与HOTA](budget_actual_calls.png)\n\n'
    text += 'ρ=.1/1.0逐视频调用匹配；.3按明确指令保留历史对照，Dance调用数并非完全相等，实际横轴如图。不同视图及不同系统的前置计算不同，不能由同调用数或缓存后的新增GPU耗时宣称端到端算力相等/更低。历史准备见inherited_costs.json，不将复用记成零，也不把废弃研究全部算成最小部署必需。\n'
    (PUBLIC / 'B2_FINDINGS.md').write_text(text)


def finalize_report():
    final = json.loads((PUBLIC / 'final_status.json').read_text())
    snapshot = PUBLIC / 'REPORT_stage_A_snapshot.md'
    if not snapshot.exists():
        snapshot.write_text((PUBLIC / 'REPORT.md').read_text().split('\n<!-- CONDITIONAL_B -->')[0])
    base = snapshot.read_text()
    headings = list(re.finditer(r'(?m)^## ([1-6])\. .*$', base))
    assert len(headings) == 6
    sections = {int(m[1]): base[m.end():headings[i+1].start() if i+1 < len(headings) else len(base)].strip()
                for i, m in enumerate(headings)}
    top = base[:headings[0].start()].strip()
    text = top + '\n\n<!-- ORDERED_V8P_REPORT -->\n\n'
    text += '主预算ρ=.3通过原门槛；ρ=.1也为6/6点值胜出，但ρ=1.0仅Dance两方向胜出。C4高预算落后两对照的区间均不跨0，消融也有反例，不能宣称全预算/全模块成功。\n\n'
    text += '主表全部使用MINUS（神经主干/PCD冻结），但v8快速CPU校准w、a_q、T、β仍只在有教师证据的对上拟合；不是ZERO_UPDATE。教师固定为Qwen3-VL-2B，候选/原始分数不变。CHECKLIST走原v8，PRESUPPOSED假定整视频句子存在，禁用存在性提问/EM；不是假定每一帧都有目标。\n\n'
    text += '方向：C1=V2→Dance，C2=Dance→KITTI-V2，C3=V2→SportsMOT，C4=Dance→SportsMOT，C5=LaMOT→KITTI-V2，C6=LaMOT→Dance。\n\n'
    text += '每格均完整覆盖对应目标：Dance为25视频/425表达，KITTI-V2为4视频/862表达，SportsMOT为45视频/466表达（v4 GT）。\n\n'
    text += '符号：ρ为教师查询预算比例；ρ̂为存在性估计比例，二者不是同一个量。\n\n'
    text += '## 1. 主表与预注册判定\n\n' + sections[1] + '\n\n'
    text += '## 2. 存在性\n\n' + sections[2] + '\n\n'
    text += '## 3. 消融、预算曲线与适用边界\n\n'
    costs = []
    for tier in ('B1', 'B2', 'B3'):
        path = PUBLIC / (tier + '_REPORT.md')
        if not path.exists():
            if tier == 'B3':
                text += '### B3\n\n' + final['status'] + '；未完成项不填数，见B3_deferred.json。\n\n'
            continue
        part, cost = path.read_text().split('\n### 成本\n', 1)
        cost = cost.split('![预算曲线]')[0].strip()
        if tier == 'B1':
            text += (PUBLIC / 'B1_FINDINGS.md').read_text() + '\n\n'
        if tier == 'B2':
            text += (PUBLIC / 'B2_FINDINGS.md').read_text() + '\n\n'
        text += part.replace('## ' + tier, '### ' + tier) + '\n\n'
        if tier == 'B2':
            text += '![名义预算曲线](budget_curves.png)\n\n'
        if tier == 'B3':
            text += '#### B3实际更新与归因范围\n\n'
            slow = slow_details()
            text += table(slow, ['method', 'videos', 'videos_with_LoRA_updates',
                'optimizer_updates', 'optimized_positive_samples',
                'optimized_negative_samples', 'logical_calls']) + '\n\n'
            text += '相对MINUS：' + '；'.join(
                r['method'] + ' ΔDetA=' + format(r['delta_DetA_vs_MINUS'], '+.3f') +
                '、ΔAssA=' + format(r['delta_AssA_vs_MINUS'], '+.3f') +
                '、ΔIDF1=' + format(r['delta_IDF1_vs_MINUS'], '+.3f') for r in slow) + '。\n\n'
            if all(r['delta_AssA_vs_MINUS'] < 0 for r in slow):
                text += '两项HOTA改善都伴随AssA下降，不能说身份关联也全面改善；本轮没有新增等数量排序诊断，不能据此证明语义排序恢复。\n\n'
            text += '两臂门槛通过的视频数和实际优化步数不同，FULL与REPLAY不是等步数训练对照；这些差异原样列出，不通过补步或修改门槛抹平。\n\n'
            text += 'FULL会随LoRA改变后续选题，不能将FULL−MINUS全部解释为固定教师证据下的效应。SLOW_REPLAY固定MINUS的问题、教师回答、DS与存在性快照，单独报告其差值和区间；未过更新门槛的视频不计为实际训练。正负样本数是累计训练抽取次数（可重复），不是不同轨迹数、额外教师标签或独立样本量。逐项成本与计数见slow_actual_updates.csv，冻结回放审计见validation_V8P_B3.json。两项只覆盖预注册C4，不外推为六方向慢分支均有效，也不替换六方向的冻结主表。\n\n'
            text += '归因边界：这是原v6配方既有行为，不是本轮改动。MINUS在全轨迹观测上计算f；慢分支更新后，FrameCacheV6.refresh_sampled会用固定32帧中出现轨迹的子集比例覆盖f，并继续快校准。因此REPLAY−MINUS识别的是固定证据下启用整个慢分支的净变化（含权重更新、既定32帧近似重算和其后快校准），不是纯权重替换的单因素分解。本轮没有新增“零权重更新但执行同样32帧f替换”的控制，不能把净增益全部分配给权重学习本身。零更新时的数值锚定不消除开始更新之后的这项限制；不据此改变原验收或启动额外实验。代码位置见flex_uda/active/v6_frames.py的decode/refresh_sampled与flex_uda/active/v7_frozen.py的refresh。\n\n'
        costs.append('### ' + tier + '成本\n\n' + cost)
    text += '## 4. 行为分析\n\n' + sections[4] + '\n\n'
    text += '该节recovered_GT、lost_GT、new_FP_rows、removed_FP_rows均相对于同方向的NATIVE预测，不是v8→v8P的变化；与v8的召回和输出行数在独立列中并列。新增行误检与不变预测行的匹配重分配仍按离线记录分开。\n\n'
    text += '## 5. 成本\n\n' + sections[5] + '\n\n' + '\n\n'.join(costs) + '\n\n'
    totals = json.loads((PUBLIC / 'cost_totals.json').read_text())
    text += '### 本轮新增实验总账（不重复计入历史引用）\n\n'
    text += table([totals], ['logical_calls', 'new_calls', 'cache_hits', 'replayed_answers',
        'teacher_GPU_hours', 'LoRA_GPU_hours', 'rescore_GPU_hours', 'scheduler_wall_hours']) + '\n\n'
    text += '调度墙钟包含等待GPU与CPU官评；相位时长不是GPU利用率积分或独占预留时间。缓存命中仍消耗逻辑查询预算，回放不新增教师调用；引用的C1/C6和历史对照的原系统成本已另列，不在本轮新增总账里重加。\n\n'
    text += '历史源/目标准备与原完整margin推理账见inherited_costs.json。二者为历史过程时长，不是同硬件冷启动基准，不重复相加为最小部署成本；部分旧成本未精确计时，不记为0。UNIFORM不直接使用身份排序，只有MOTION项可使用原存在性门控，因此不能把所有前置计算一律称为共同必需成本。同调用不等于同视图成本/同GPU算力；本轮新调用、缓存、回放、缓存等价相位时长、实际worker墙钟分别报告。\n\n'
    text += '## 6. 限制\n\n' + sections[6].replace('B1将如实检验模块贡献', 'B1完整反例已列入第3节') + '\n\n'
    text += '协议规则读取全部目标查询元数据一次；整视频分数/轨迹聚合及部分教师看法使用未来帧，属于视频级转导，不是严格逐帧因果部署。六个方向只来自三个目标集合，不能当作六个独立数据集；协议假定尚未在其它查询协议上盲测。外部冻结教师提供额外知识，不等价于无教师的源模型自身恢复。B1的GLANCE必要性及C5的DS/获取收益未获一致支持；高预算四方向落后，正式ρ=.3门槛通过不表示所有模块、所有预算均成立。覆盖统计为事后描述，不进入阈值、验收或主方法选择。\n\n'
    text += 'bootstrap区间仅描述已固定预测的视频聚类变动，不校正多轮Val开发或事后协议设计的选择偏差，也不替代新盲测。连续适应状态会跨视频传递；本次重采样不重排视频、不重跑适应，因此不代表重新运行整个在线流程的全部不确定性。\n\n'
    text += '## 7. 关闭、审计与恢复\n\n'
    text += '状态：' + final['status'] + '；新增官评项' + str(final['official_new_jobs']) + '，累计方法—表达评测条数' + str(final['official_new_episodes']) + '。生成进程目标GT打开尝试均0；独立官评与事后诊断按协议使用GT。24项定向CPU测试通过；两个未改的长合成测试本轮未重复，历史收据保留，不能算作新增通过。覆盖诊断54格GT0、教师/模型调用0。\n\n'
    text += '全部日志和冻结BF16教师原始输出见logs.json、archive_validation.json；预测/数组路径与SHA见prediction_paths.json、B1/B2/B3_prediction_paths.json（存在者）、numerical_artifact_paths.json。大模型/大数组不入git。原v8阶段3独立继续/关闭，未停止其进程、未启动旧阶段4/80项，也未启动新方法。终端observer复接只影响只读传输，不是GPU任务失败或重跑。\n'
    text += '\n冻结分支的优化/回放检查明确标记NOT_EVALUATED_NO_UPDATES/NO_REPLAY，不作为真实LoRA验证。\n'
    b3_validation = PUBLIC / 'validation_V8P_B3.json'
    if b3_validation.exists():
        audit = json.loads(b3_validation.read_text())
        text += '本轮B3的' + str(audit['jobs']) + '项真实优化/固定回放审计为' + audit['status'] + '，实际优化次数' + str(audit['observed_optimizer_updates']) + '；完整收据见validation_V8P_B3.json，额外PRESUPPOSED实测核验及样本数见slow_updates_inputs.json和slow_actual_updates.csv。\n'
    text += '\n报告后处理曾误用旧版main.csv文件名，已修正为v8P的direction_summary.csv；随后90份真实B3收据回归和完整报告编排均通过，没有重跑预测或官评。详细失败/修复及保留判断见research_log.md。\n'
    retained = PUBLIC / 'source_v8_retained/retention_status.json'
    if retained.exists():
        old = json.loads(retained.read_text())
        text += '\n### 原v8关闭审计失败的隔离记录\n\n'
        text += '原v8的11项GPU任务和官评已结束，但其supervisor退出码1不改、全局审计未改判通过。26处冲突均为冻结回放的反事实分层检查；350份六存在性数组、问题/回答/DS与源MINUS一致，真实源判空的覆盖有效。原未改校验器仅在独立目录复查9项非回放方法通过，不能冒充11项全局验收。当前v8P引用的C1/C6 MINUS处于这个严格通过子集；本轮B3为不推断存在性的PRESUPPOSED。\n\n'
        text += '全部旧预测、原关闭traceback和冻结教师原始记录隔离保存于[source_v8_retained/REPORT.md](source_v8_retained/REPORT.md)；具体26处见original_v8_audit_diagnostic.json。原v8代码、验证文件和失败收据均未修改，不开启旧阶段4，也不在本轮擅自修订旧验收口径。归档字节验证状态：' + old['archive_verification'] + '，这不是全局实验PASS。\n'
    (PUBLIC / 'REPORT.md').write_text(text)
    inputs = [snapshot, PUBLIC / 'B2_summary.csv', PUBLIC / 'B2_bootstrap.json',
              PUBLIC / 'budget_coverage.csv', PUBLIC / 'final_status.json']
    if retained.exists():
        inputs += [retained, PUBLIC / 'original_v8_audit_diagnostic.json']
    atomic_json(PUBLIC / 'report_composition.json', dict(
        status='ORDERED_POST_SEAL_REPORT', source_files=[dict(path=str(p), sha256=file_sha(p)) for p in inputs],
        raw_GT_files_opened=0, uses_post_seal_GT_derived_metrics=True,
        model_or_teacher_calls=0, numerical_predictions_or_acceptance_changed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget-only', action='store_true')
    args = parser.parse_args()
    budget_supplement()
    if not args.budget_only:
        finalize_report()
    print(json.dumps(dict(status='BUDGET_SUPPLEMENT_CLOSED' if args.budget_only else
                          'ORDERED_REPORT_CLOSED', model_or_teacher_calls=0,
                          raw_GT_files_opened=0)), flush=True)
