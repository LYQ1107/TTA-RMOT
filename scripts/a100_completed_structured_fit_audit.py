"""Read existing, completed RSI fitting receipts; never train or evaluate GT."""
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / ('a100_runs/visual_classical_20261121/'
                 'six_direction_20261122/native_binary_comparable_quality_20261124')
PUBLIC = ROOT / 'docs/a100_six_direction_20261122'
PRIVATE = ROOT / 'a100_runs/visual_classical_20261121/supervisor'


def summarize(rows):
    values = [r['excess_risk'] for r in rows]
    return dict(records=len(rows), passed=sum(r['passed'] for r in rows),
                maximum_excess_risk=max(values), mean_excess_risk=statistics.mean(values),
                median_excess_risk=statistics.median(values),
                width_histogram=dict(sorted(Counter(r['width'] for r in rows).items())),
                actual_Adam_steps_histogram=dict(sorted(Counter(r['steps'] for r in rows).items())))


def main():
    rows, summaries, inputs = [], [], {}
    for pair, expected in zip(('C1', 'C2', 'C3', 'C4', 'C5', 'C6'), (25, 4, 45, 45, 4, 25)):
        path = COHORT / pair / 'seed0/completion.json'
        data = path.read_bytes()
        inputs[pair] = dict(path=str(path.relative_to(ROOT)),
                            sha256=hashlib.sha256(data).hexdigest())
        raw = json.loads(data)
        receipts = raw['receipts']
        items = list(receipts.items()) if isinstance(receipts, dict) else list(enumerate(receipts))
        assert len(items) == expected
        for policy in ('LEARNED', 'FROZEN'):
            policy_rows = []
            for video, receipt in items:
                info = receipt['fits']['FULL_CONDITIONAL_REFERENCE_'+policy]
                assert len(info['rounds']) == 3
                fits = [r['fit'] for r in info['rounds']] + [info['final_all_paid_fit']]
                for index, fit in enumerate(fits):
                    risk = float(fit['excess_risk'])
                    passed = bool(fit['teacher_fit_gate_passed'])
                    assert passed == (risk <= .005)
                    assert fit['neural_seed'] == 0
                    assert fit['GT_or_task_metric_selection'] is False
                    assert fit['ordinary_required'] is False
                    assert fit['risk_units'] == 'proper teacher KL in nats'
                    row = dict(pair=pair, policy=policy, video=str(video),
                               stage='final' if index == 3 else 'round'+str(index),
                               excess_risk=risk, passed=passed, width=fit['width'],
                               steps=fit['actual_final_Adam_steps'])
                    policy_rows.append(row)
            rows.extend(policy_rows)
            summaries.append(dict(pair=pair, policy=policy,
                                  final=summarize([r for r in policy_rows if r['stage'] == 'final']),
                                  all_stages=summarize(policy_rows)))
    finals = [r for r in rows if r['stage'] == 'final']
    assert len(finals) == 296 and len(rows) == 1184
    private_data = (json.dumps(rows, ensure_ascii=False, separators=(',', ':'))+'\n').encode()
    private_path = PRIVATE / 'completed_structured_fit_audit_rows_20261125.json'
    private_path.write_bytes(private_data)
    report = dict(status='PASS_COMPLETED_STRUCTURED_FIT_RECEIPT_AUDIT',
                  cohort='native_binary_comparable_quality_20261124',
                  scope='two main structured policies, six settings, three round fits plus final',
                  threshold=.005, risk_units='proper teacher KL in nats',
                  final=summarize(finals), all_stages=summarize(rows), by_setting_policy=summaries,
                  inputs=inputs, row_sha256=hashlib.sha256(private_data).hexdigest(),
                  private_complete_rows=str(private_path.relative_to(ROOT)),
                  records_not_independent_models_or_examples=True,
                  four_unaffected_settings_exact_parent_reuse_not_independent_retraining=True,
                  teacher_training_fit_not_task_accuracy_or_unseen_generalization=True,
                  does_not_rule_out_other_optimization_or_capacity_changes=True,
                  does_not_identify_causal_reason_for_HOTA_loss=True,
                  no_new_GT_teacher_calls_models_checkpoints_fits_predictions_GPU_or_seeds=True,
                  source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
                      'flex_uda/active/native_binary_fit.py',
                      'flex_uda/active/native_binary_curriculum.py')},
                  goal_status='ACTIVE')
    receipt_path = PUBLIC / 'completed_structured_fit_audit_20261125.json'
    receipt_path.write_text(json.dumps(report, ensure_ascii=False, separators=(',', ':'))+'\n')
    text = f'''TTA-RMOT：已完成RSI主方法的教师拟合审计
2026-11-25；仅检查已完成修正研究的训练收据；没有新训练或GT读取

目的：核查最新主方法落后于普通蒸馏，是否能简单解释为没有达到既定拟合门槛。
范围：六设置、LEARNED/FROZEN两策略，每视频三个轮次和一次最终全付费证据拟合。
C1/C3/C4/C6精确复用原TRUST，C2/C5为实际视角修正；不是六设置重新独立训练。

最终拟合记录：{len(finals)}，通过原.005教师KL门槛：{report['final']['passed']}。
最大最终训练超额风险：{report['final']['maximum_excess_risk']:.17g} nats。
全部轮次加最终记录：{len(rows)}，通过：{report['all_stages']['passed']}。
最大记录超额风险：{report['all_stages']['maximum_excess_risk']:.17g} nats。
1184是收据记录数，不能当作1184个独立网络、视频或泛化样本。

这些数字说明：已完成主方法确实能按照预先规定的标准拟合收到的教师答案，
不能在没有进一步证据时把其较低HOTA简单归咎于未达到原训练门槛。
它们不说明：教师答案必然正确、未询问候选也预测正确，或更多训练绝不可能改善。
训练目标与稠密任务输出的联系、证据支持和未询问输入泛化，仍需剩余机制消融定位。
本审计没有比较不同损失的数值大小，也没有把联合KL和边缘KL当成同一个目标。
它不是新的正向模块证据，不是HOTA因果分解，不能用来宣布三个创新点或六设置优势成立。

程序：scripts/a100_completed_structured_fit_audit.py
公开摘要：docs/a100_six_direction_20261122/completed_structured_fit_audit_20261125.json
摘要保留六个输入收据和两个训练源文件的SHA256；完整逐条记录保留本地非Git目录。
原所有质量结果、训练源和在途native726保持不变，Goal ACTIVE。
'''
    report_path = ROOT / 'docs/TTA_RMOT_已完成RSI教师拟合审计_20261125.txt'
    report_path.write_text(text)
    print(json.dumps(dict(status=report['status'], final=report['final'],
                          all_stages=report['all_stages'], report=str(report_path)),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
