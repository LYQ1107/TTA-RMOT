"""Close the authorized five-job stage using only already-sealed records.

No model, teacher, raw GT, new evaluator, or scheduler is invoked here.
The generated tables supplement, rather than replace, official TrackEval.
"""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.active.execution import authorize_active
from scripts.report_active import write_table


def fp_residual(row):
    residual = row['unchanged_rows_new_FP'] - row['unchanged_rows_removed_FP']
    assert (row['new_false_positive_rows'] - row['removed_false_positive_rows']
            + residual) == row['net_FP_change']
    return residual


def verify_packaged_records(public, record):
    import gzip
    import hashlib
    import tarfile
    import torch
    from flex_uda.teacher.teacher_cache import request_key
    with tarfile.open(record['archive']['path'], 'r:gz') as archive:
        expected = {row['archive_path']: row for row in record['files']}
        assert set(archive.getnames()) == set(expected)
        for name, item in expected.items():
            value = archive.extractfile(name).read()
            assert len(value) == item['bytes']
            assert hashlib.sha256(value).hexdigest() == item['sha256']
    seen, logits, probabilities, diagnostics = set(), [], [], []
    with gzip.open(record['raw_teacher_responses']['path'], 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            answer, key = row['response'], row['request_id']
            assert key not in seen and key == answer['request_id'] == request_key(answer['input'])
            seen.add(key)
            assert answer['input']['revision'] == '89644892e4d85e24eaac8bacfd4f463576704203'
            assert answer['frozen_weight_sha256'] == answer['input']['weight'] == '9a13380b4279bc98f4680bd083ff4995fdcfec120cfded459513be90d041a956'
            logits.append([answer['yes_logit'], answer['no_logit']])
            probabilities.append(answer['yes_probability'])
            diagnostics.append(answer['yes_probability_fp32'])
    assert len(seen) == record['raw_teacher_response_keys']
    raw = torch.tensor(logits, dtype=torch.bfloat16)
    native = torch.tensor(probabilities, dtype=torch.float32)
    fp32 = torch.tensor(diagnostics, dtype=torch.float32)
    assert torch.isfinite(raw).all() and torch.isfinite(native).all() and torch.isfinite(fp32).all()
    # The released path's BF16 softmax is intentionally NOT replaced by its
    # saved FP32 diagnostic. CPU/CUDA FP32 kernels need not share final bits.
    assert torch.equal(torch.softmax(raw, dim=-1)[:, 0].float(), native)
    torch.testing.assert_close(torch.softmax(raw.float(), dim=-1)[:, 0], fp32, rtol=2e-6, atol=1e-8)
    result = dict(status='ARCHIVE_AND_ORIGINAL_TEACHER_RECORDS_VERIFIED',
        archived_files=len(expected), raw_response_keys=len(seen),
        raw_request_hashes_and_frozen_revision_weights_verified=True,
        native_BF16_softmax_exact=True, FP32_diagnostic_tolerance=dict(rtol=2e-6, atol=1e-8),
        raw_teacher_or_log_bytes_modified=False, model_or_teacher_or_GT_invoked=False,
        original_archive_sha256=record['archive']['sha256'],
        original_teacher_archive_sha256=record['raw_teacher_responses']['sha256'])
    atomic_json(public / 'archive_validation.json', result)
    return result


def close(execution):
    from scripts.presence_redesign_report import table
    public = execution.public
    read = lambda name: json.loads((public / name).read_text())
    queue, validation = read('queue_completion.json'), read('validation.json')
    assert queue['status'] == 'V6_AUTHORIZED_SET_COMPLETE'
    assert validation['status'] == 'V6_AUTHORIZED_SET_VALIDATED'
    assert validation['primary_jobs'] == 5 and validation['conditional_jobs'] == 0
    with (public / 'main.csv').open() as handle:
        main = list(csv.DictReader(handle))
    assert len(main) == 5
    intervals = read('bootstrap.json')
    artifacts = read('artifacts.json')
    raw_logs = read('logs.json')
    assert raw_logs['completed_active_methods'] == 5 and not raw_logs['unpackaged_launcher_logs']
    for name in ('archive', 'raw_teacher_responses'):
        item = raw_logs[name]
        assert file_sha(item['path']) == item['sha256']
    archived = verify_packaged_records(public, raw_logs)
    attribution, final = [], []
    for row in main:
        path = Path(row['path'])
        assert file_sha(path / 'predictions_seal.json') == row['prediction_seal_sha256']
        saved = json.loads((path / 'offline_attribution.json').read_text())
        assert saved['prediction_seal_sha256'] == row['prediction_seal_sha256']
        attribution.append(dict(pair=row['pair'], method=row['method'],
            GT_observations=saved['gt_observations'],
            matched_GT=saved['adapted_matched_gt'],
            GT_recall=saved['adapted_matched_gt'] / saved['gt_observations'],
            output_rows=saved['adapted_selected_rows'],
            Native_matched_GT=saved['baseline_matched_gt'],
            recovered_GT=saved['recovered_gt'], lost_GT=saved['lost_gt'],
            net_GT_vs_Native=saved['net_gt_recovery'],
            FP_rows=saved['adapted_false_positive_rows'],
            new_FP_rows=saved['new_false_positive_rows'],
            removed_FP_rows=saved['removed_false_positive_rows'],
            unchanged_row_new_FP=saved['unchanged_rows_new_FP'],
            unchanged_row_removed_FP=saved['unchanged_rows_removed_FP'],
            matching_reassignment_residual=fp_residual(saved),
            net_FP_vs_Native=saved['net_FP_change'],
            definition=saved['definition']))
        final.append(dict(row, evaluation_scope='OFFICIAL_COMPLETE_TARGET',
                          no_new_evaluation_by_closure=True))
    write_table(public / 'prediction_attribution.csv', attribution)
    write_table(public / 'final_official_results.csv', final)
    atomic_json(public / 'final_official_results.json', dict(
        status='FINAL_FIVE_AUTHORIZED_JOBS_OFFICIALLY_EVALUATED', completed=final,
        replaces_status_snapshot='interim_official_results.json (historical four-job snapshot retained)',
        validation_sha256=file_sha(public / 'validation.json')))
    aindex = {(r['pair'], r['method']): r for r in attribution}
    mindex = {(r['pair'], r['method']): r for r in main}
    for pair in ('C1', 'C3'):
        full, minus = mindex[pair, 'FULL_v6'], mindex[pair, 'MINUS_SLOW_v6']
        afull, aminus = aindex[pair, 'FULL_v6'], aindex[pair, 'MINUS_SLOW_v6']
        interval = intervals[pair + '__FULL_v6_minus_MINUS_SLOW_v6']
        value = dict(pair=pair, comparison='FULL_v6_minus_MINUS_SLOW_v6', **interval,
            delta_Query_Empty_prediction_rows=int(full['query_empty_prediction_rows']) - int(minus['query_empty_prediction_rows']),
            delta_Query_Present_prediction_rows=(int(full['prediction_rows']) - int(full['query_empty_prediction_rows'])
                - int(minus['prediction_rows']) + int(minus['query_empty_prediction_rows'])),
            delta_Present_HOTA=float(full['Present_HOTA']) - float(minus['Present_HOTA']),
            delta_GT_recall=afull['GT_recall'] - aminus['GT_recall'],
            delta_matched_GT=afull['matched_GT'] - aminus['matched_GT'],
            teacher_calls_full=int(full['teacher_calls']), teacher_calls_minus=int(minus['teacher_calls']),
            CI_status='FINAL_VIDEO_CLUSTER_BOOTSTRAP_2000_SEED_0',
            comparison_scope='SLOW closed loop includes LoRA, fixed32 refresh, and subsequent acquisition/fast-calibration feedback; not isolated weight-only effect',
            identical_question_sets_not_claimed=True, runtime_GT_open_attempts=0)
        atomic_json(public / (pair + '_slow_closed_loop.json'), value)
    with (public / 'costs.csv').open() as handle:
        costs = list(csv.DictReader(handle))
    fields = ('logical_calls', 'new_calls', 'cache_hits', 'optimizer_updates')
    totals = {key: sum(int(r[key]) for r in costs) for key in fields}
    for key in ('teacher_forward_GPU_hours', 'committed_teacher_equivalent_forward_GPU_hours',
                'LoRA_GPU_hours', 'rescore_GPU_hours', 'seal_process_wall_hours',
                'assigned_worker_wall_hours_including_failures', 'frame_prior_CPU_hours'):
        totals[key] = sum(float(r[key]) for r in costs)
    assert totals['new_calls'] + totals['cache_hits'] == totals['logical_calls']
    failures = read('failures.json')
    failed_wall = sum(r['worker_wall_seconds'] for r in failures['attempts']) / 3600
    durable = execution.launchers / 'durable_1793787145113785979'
    launched = json.loads((durable / 'launch.json').read_text())
    finished = json.loads((durable / 'completion.json').read_text())
    assert launched['manager_pid'] == finished['manager_pid'] == 1862297 and finished['exit_code'] == 0
    totals.update(status='MEASURED_AUTHORIZED_FIVE_JOB_COSTS',
        resumed_manager_wall_hours=(finished['finished_time_ns'] - launched['launched_time_ns']) / 3.6e12,
        failed_worker_wall_hours_included=failed_wall,
        failed_initial_rescore_phase_GPU_hours=None,
        successful_rescore_plus_LoRA_plus_new_teacher_phase_hours=sum(totals[k] for k in
            ('rescore_GPU_hours', 'LoRA_GPU_hours', 'teacher_forward_GPU_hours')),
        maximum_physical_GPUs=2, GPU_utilization_integral_measured=False,
        teacher_subprocess_wall_not_added_again_to_worker_wall=True,
        smoke_new_teacher_calls_separate=3,
        reused_native_MOT_source_head_and_source_features_cost_not_zero=True,
        reused_baseline_cost_file=str(Path(execution.manifest['prior_v5_public_root']) / 'costs.csv'),
        cache_equivalent_seconds_are_original_shared_hardware_timings_not_new_GPU_usage=True,
        GPU_smoke_and_historical_preparation_excluded_from_this_five_job_sum=True)
    atomic_json(public / 'cost_summary.json', totals)
    # Recompute with precisely the worker's NumPy, retaining the exact-equality
    # requirement. NumPy 1.26 versus 2.2 has different final log() bits.
    import numpy as np
    files = mismatches = 0
    for row in main:
        for path in Path(row['path']).glob('videos/*/belief.npz'):
            with np.load(path) as belief:
                p = np.clip(belief['fractions'], .05, .95)
                exact = np.clip(np.log(p / (1 - p)), -2., 2.)
                mismatches += not np.array_equal(exact, belief['base_logits'])
                files += 1
    assert mismatches == 0
    atomic_json(public / 'postclosure_audit_environment.json', dict(
        status='EXACT_EQUALITY_PASSED_IN_ORIGINAL_WORKER_NUMPY',
        numpy_version=np.__version__, numpy_module=np.__file__,
        checked_video_beliefs=files, unequal_beliefs=mismatches,
        first_attempt=dict(numpy_version='1.26.4', exit_code=1, unequal_beliefs=135,
            example_fraction=.125, stored_logit=-1.9459101490553135,
            recomputed_logit=-1.9459101490553132,
            cause='validator invocation omitted the project pydeps path; log final bits differ'),
        correction='use the identical PYTHONPATH and NumPy 2.2.6 as the original workers',
        exact_equality_acceptance_retained=True, runtime_or_saved_arrays_modified=False,
        GPU_or_teacher_or_GT_called=False))
    status = dict(status='AUTHORIZED_V6_FIVE_JOB_STAGE_COMPLETE_AND_STOPPED',
        primary_jobs=5, conditional_YT_PRIOR_jobs=0, manager_exit_code=0,
        official_episodes=validation['total_episodes'], prediction_rows=validation['prediction_rows'],
        runtime_GT_open_attempts=0, datasets_complete_for_authorized_jobs=True,
        all_predictions_sealed_before_independent_evaluation=True,
        remaining_original_v6_jobs=[dict(pair=p, method=m, rho=.3) for p, m in
            [('C4', 'FULL'), ('C5', 'FULL'), ('C6', 'FULL'), ('C4', 'MINUS_SLOW'), ('C6', 'MINUS_SLOW')]],
        six_direction_four_win_gate_evaluated=False,
        original_v5_remaining80_authorized=False, full_matrix_authorized=False,
        new_method_or_hyperparameter_tuning_started=False,
        waiting_for_explicit_user_direction=True,
        execution_sha256=file_sha(public / 'execution.json'),
        plan_sha256=file_sha(public / 'registered_jobs.json'),
        validation_sha256=file_sha(public / 'validation.json'),
        closure_reads_sealed_evaluation_outputs_only=True,
        new_model_teacher_or_raw_GT_calls_by_closure=0)
    atomic_json(public / 'final_status.json', status)
    comparisons = []
    for key, value in intervals.items():
        comparisons.append(dict(comparison=key, delta_HOTA=value['delta_HOTA'],
            CI95=str(value['CI95']) if value['CI95'] is not None else '4 videos: point only'))
    write_table(public / 'comparison_intervals.csv', comparisons)
    analysis = '''## 最终科研结论与停止范围

五项完整预测和官方评测全部正常结束，不再运行GPU队列。修复不是全面成功：C2、C3相对v5恢复明显，C1显著下降；三个FULL点估计都高于YT_PRIOR/UNIFORM_K中较好者，但C1、C3对最强简单对照的95%区间均跨0，C2只有4视频、按预注册不做显著性判断。因此只能称方向性正结果，尚未证明优于简单教师校验的稳定新方法。

全10项运行前估84.058 GPU小时触发用户20小时回退，只运行C1/C2/C3 FULL及C1/C3 MINUS_SLOW。当前3/3点估计胜两对照不能代替“六方向至少四胜”；该门槛未评估，完整矩阵未授权。尚未运行FULL C4/C5/C6、MINUS_SLOW C4/C6，v5剩余80项继续不获授权。本阶段交付后停止等待，不擅自补项。

'''
    with (public / 'direction_summary.csv').open() as handle:
        directions = list(csv.DictReader(handle))
    analysis += table(['pair', 'source', 'target', 'NATIVE', 'FULL_v5', 'FULL_v6',
        'FULL_v6_minus_v5', 'MINUS_SLOW_v6', 'YT_PRIOR', 'UNIFORM_K'], directions)
    analysis += '''

### 不确定性与慢速分支

下面区间按视频聚类2000次、seed0；KITTI仅点估计。完整区间也见comparison_intervals.csv/bootstrap.json。

'''
    analysis += table(['comparison', 'delta_HOTA', 'CI95'], comparisons)
    analysis += '''

FULL−MINUS_SLOW：C1为−1.808 [−7.832,5.659]，C3为−0.279 [−3.276,2.696]。两个点估计均不支持额外慢速收益，但区间跨0也不证明慢速必然有害。此消融包含LoRA、固定32帧f近似和后续获取/快校准反馈，并非单独权重的因果效应；等调用数也不是等提问集合。

C1的FULL比MINUS_SLOW多88,754行Query-Empty预测，Present预测少1,617行；Present HOTA却高2.530，不能只挑有目标组或AssA宣称成功。C3在相同v4 GT、原418序列ID敏感性下仍是MINUS_SLOW略高（60.058对59.689），不使用旧冲突GT模板。

### 修复到底改善了什么

高置信错负比例（GT正长轨迹对）C1从23.10%降到0.53%，C2从84.45%降到13.71%，C3从19.41%降到1.01%。但置信变软本身不等于恢复：C1 GT观测召回从59.39%掉到31.08%，C2从4.75%升到20.17%，C3从28.65%升到56.52%。C2仍有916个未问过的GT正长轨迹对被最终后验高置信判负；“未问伪标签为UNKNOWN”只保证不用于伪监督，不保证最终输出不会拒绝它们。

C1旧整句缺席标志误删5/40个有目标表达，v6降到0/40；代价是对385个真正空表达，正确打出缺席标志从257降到16（66.75%→4.16%）。这是算法显式absence标志覆盖率，不是整体预测存在性准确率；更保守的整句判空没有自动解决空表达误检。

下表使用已有封存后的离线IoU≥0.5最大基数匹配，绝非官方CLEAR/HOTA匹配替代。FP总变化严格分解为新增行FP−删除行FP+未变行匹配重分配残差，不能把新增行数或新增FP单独当作总FP变化。

'''
    analysis += table(['pair', 'method', 'GT_recall', 'matched_GT', 'output_rows',
        'net_GT_vs_Native', 'new_FP_rows', 'removed_FP_rows',
        'matching_reassignment_residual', 'net_FP_vs_Native'], attribution)
    analysis += '''

C3虽比v5恢复大量GT、比Native的HOTA高8.371，但相对Native仍净丢32,680个已匹配GT观测；其HOTA收益伴随大幅减FP，不能全部解释成新增语义恢复。C2相对Native净恢复73,261个GT观测，同时净增196,943个FP，仍有明显召回/误检折中。

### 仍存在的机制局限（本轮不修改方法）

C1仅2/25视频有LoRA更新，共150步。第五个视频后w变为0.171393，后20视频新a_q=0、ℓ0≤2、单次教师证据≤2，使第一次回答的正后验上界sigmoid(2w+2)≈0.912，达不到原>.95正伪标签门槛；实际这20视频2918个被问过的(q,track)对全部只问一次，无法用重复证据跨过门槛。这是独立GT-free的正监督断供证据，不证明全部指标下降都由此造成。MINUS_SLOW C1的w始终≥0.953612，无该全局上界阻断。

已沿用的存在性头实际只读margin（两斜率Bernoulli），PCD虽按要求FP32缓存但不输入此头。因此LoRA改变margin后仍会经存在性/数量、f、先验和采样反馈影响输出，“身份—存在性解耦”并不完全。它是设计局限与可能归因，不是本批已隔离验证的因果结论。32帧之外的最终缓存仍是视频起始版本，也不等同于最后模型全视频重推理。

DS删除IDENTITY并改Beta(2,2)后，估计仍偏乐观：例如C2 CROP特异度估计0.861、实际已问集合GT为0.726；C3 FRAME_BOX为0.767对0.634。不同视图来自同一冻结模型，不能保证条件独立。v5/v6所问候选和视图混合不同，不能把准确率变化单独归因于Beta先验；完整表同时给关键帧与轨迹多数两种GT口径，仅覆盖实际提问集合。

历史slow日志没保存实际优化输入类别计数，轮末positive_pairs=0不代表优化时只用了负例。C3的2300步属于“轮末零正标签的轮”，真实输入正负比例未知；报告保留此前误用的更正记录，不通过重跑或臆测补造样本统计。

### 可信边界与实现核验

'''
    analysis += (f"五项共{validation['total_episodes']}个episode、{validation['prediction_rows']:,}行预测已逐行核验框/ID/帧号来自固定MOT；"
        "预测封存后才独立评测，运行及教师GT打开尝试均0；旧v5全部seal不变。"
        f"在原worker的NumPy {np.__version__}下，{files}份视频belief的f→截断logit逐位一致，"
        "w范围、证据累加上限、三段判空、DS配置、全视频起始/固定32帧覆盖、FP32 margin/PCD均通过。"
        "首个离线审计误用NumPy1.26导致log末位差异，切回原2.2.6环境即通过原精确门槛，未改数组/预测/数值代码。"
        "原46项定向回归和3项最终FP归因检查共49项通过；同源历史缓存路径并不逐位相同，首视频原生符号无差异仅为局部抽查，不能扩展成全库PRES_TOPK等价。\n\n"
        "本设定为冻结教师辅助的video-transductive适应，有全视频信息且Val已多次开发；不是严格帧因果或盲测。"
        "视频bootstrap条件于固定视频流，不覆盖重排视频后的重新适应与随机seed不确定性。没有按目标GT改参数、阈值或挑checkpoint。\n\n")
    analysis += '### 实测成本、封存日志与恢复\n\n'
    analysis += (f"五任务逻辑教师调用{totals['logical_calls']:,}次，其中新调用{totals['new_calls']:,}、缓存命中{totals['cache_hits']:,}；"
        f"另有真实smoke3次。新教师forward合计{totals['teacher_forward_GPU_hours']:.4f}h、"
        f"LoRA {totals['LoRA_GPU_hours']:.4f}h（{totals['optimizer_updates']}步）、"
        f"成功打分相位{totals['rescore_GPU_hours']:.4f}h。"
        f"已分配共享worker累计wall {totals['assigned_worker_wall_hours_including_failures']:.4f}h，"
        f"含首次失败{failed_wall:.4f}h；恢复后manager wall {totals['resumed_manager_wall_hours']:.4f}h含CPU评测/报告，"
        "这些都不是GPU利用率积分，也不把教师子进程wall再次加到worker总时长。失败首轮未持久化的独立GPU打分相位时间仍未知，不伪造为0。\n\n"
        "逻辑调用相对v5的FULL只变0.695%/0%/0%，均未触发严格>5%的YT_PRIOR补跑规则；"
        "当前两对照按用户要求复用旧结果，不是新逐视频精确预算匹配。YT_PRIOR作为独立系统仍需原生模型/MOT推理，"
        "其部署成本不能写成CPU读缓存耗时；历史源头/候选/特征准备和原完整教师响应成本单列，不计作本轮新GPU时长。"
        "共享卡负载与SKIP/缓存均不同，不能从保守估84小时降到当前实耗推导纯算法加速。\n\n"
        f"完整关闭日志{raw_logs['archive']['bytes']:,}字节，SHA256 `{raw_logs['archive']['sha256']}`；"
        f"原始教师响应{raw_logs['raw_teacher_response_keys']:,}个唯一输入键，"
        f"压缩{raw_logs['raw_teacher_responses']['bytes']:,}字节、SHA256 `{raw_logs['raw_teacher_responses']['sha256']}`。"
        f"实际逐个解码核对{archived['archived_files']}个日志成员哈希及全部教师输入键/冻结权重；"
        "原BF16 Yes/No softmax精确复现，额存FP32概率只作诊断、未替换原监督概率。"
        "所有launcher日志均归档，包括最初类型兼容失败与观察连接重接；原日志/缓存/checkpoint未删。"
        "归档文件名与每项文件SHA见logs.json，若提供.parts按manifest顺序原字节重组即可，不能重新压缩后冒充原SHA。\n\n"
        "运行根目录 `/data2/user/FlexHook-TTA/results/runs/presence_active_20261102/redesign_v6`"
        "链接至已授权data1资产；每个`active/C*/{FULL,MINUS_SLOW}/rho_0.3`内保存预测seal、"
        "逐视频receipt/questions/belief、frame_cache及stream检查点。FULL最后stream分别C1 `stream_024.pt`、"
        "C2 `stream_003.pt`、C3 `stream_044.pt`；完整路径/SHA见artifacts.json和logs.json，"
        "不是启动新任务的命令。execution.json、registered_jobs.json与PREREG保留不变。\n\n"
        "本页为最终结论；interim_official_results.*明确仅保留当时4/5完成的历史快照，"
        "当前状态请读final_status.json、final_official_results.*、queue_completion.json和validation.json。\n\n")
    begin, end = '<!-- V6_FINAL_ANALYSIS_BEGIN -->', '<!-- V6_FINAL_ANALYSIS_END -->'
    text = (public / 'REPORT.md').read_text()
    if begin in text:
        left, rest = text.split(begin, 1)
        _, right = rest.split(end, 1)
        text = left + right.lstrip('\n')
    heading, _, body = text.partition('\n')
    (public / 'REPORT.md').write_text(heading + '\n\n' + begin + '\n' + analysis + end + '\n\n' + body.lstrip('\n'))
    print(json.dumps(dict(status=status['status'], totals=totals, GT_or_model_or_teacher_calls=0), indent=2))


if __name__ == '__main__':
    execution, _ = authorize_active()
    close(execution)
