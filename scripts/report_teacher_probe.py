#!/usr/bin/env python3
"""Generate the independent P1 report from sealed metrics; no GT/model access."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.audit_teacher_leakage import OUTPUT,RUN,METHODS,validate_seal

SHORT={'FROZEN_FLEXHOOK':'Frozen FlexHook','YES_FULL':'A / YesTrack整句',
    'YES_ENSEMBLE_BUDGET_MATCHED':'B / 等预算整句','STRUCTURED_EVIDENCE':'C / 分属性',
    'EVIDENCE_VERIFIED':'D / 证据验证','YES_ENSEMBLE_B_PLUS':'B-plus / 同视图同验证器',
    'YESTRACK_TCP_TRP':'原TCP/TRP / 13帧重置','SOFTMATCH_WEIGHTED':'SoftMatch-inspired权重'}


def pct(value):return 'N/A' if value is None else f'{100*value:.3f}%'
def fnum(value):return 'N/A' if value is None else f'{value:.6f}'


def figures(output,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader((output/'precision_coverage.csv').open()))
    names=['YES_FULL','YES_ENSEMBLE_BUDGET_MATCHED','STRUCTURED_EVIDENCE','EVIDENCE_VERIFIED','YES_ENSEMBLE_B_PLUS']
    fig,axes=plt.subplots(1,3,figsize=(15,4.2))
    for method in names:
        selected=[r for r in rows if r['method']==method and r['scope']=='GT_present_frame_queries' and
            r['selection']=='source_frozen_deployment_POS_only']
        if selected:
            axes[0].plot([float(r['GT_candidate_coverage']) for r in selected],
                [float(r['precision']) for r in selected],label=method.replace('YES_ENSEMBLE_','ENSEMBLE_'),marker='.',ms=3)
    axes[0].set(xlabel='Matched-candidate positive coverage',ylabel='Accepted POS precision',
        xlim=(0,1),ylim=(0,1.03),title='Source-frozen POS selection only')
    if axes[0].lines:axes[0].legend(fontsize=6)
    for ax,key,title in [(axes[1],'AUPRC','Candidate AP (GT-present frames)'),(axes[2],'UNKNOWN_rate','UNKNOWN rate (same primary scope)')]:
        ax.bar(range(len(names)),[metrics[m][key] or 0 for m in names])
        ax.set_xticks(range(len(names)),['A','B','C','D','B+']);ax.set(title=title,ylim=(0,1))
    fig.suptitle('Frozen T0 / fixed 25-video sparse P1: not full RMOT HOTA')
    fig.tight_layout();folder=output/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'teacher_p1_quality.png',dpi=160);fig.savefig(folder/'teacher_p1_quality.pdf');plt.close(fig)


def archive_logs(run,output):
    files=sorted(set(run.rglob('*_stdout.txt'))|set(run.rglob('receipt.json'))|
        set(run.rglob('launcher_status.json'))|set(run.rglob('gpu_admission.json'))|set(run.rglob('phase_completion.json'))|
        set(run.rglob('whole_video_shard_plan.json')))
    folder=output/'logs';folder.mkdir(exist_ok=True);archive=folder/'teacher_p1_complete_text_logs.tar.gz'
    members=[]
    with tarfile.open(archive,'w:gz') as tar:
        for path in files:
            if path.name not in ('receipt.json','launcher_status.json','gpu_admission.json','phase_completion.json','whole_video_shard_plan.json') and not path.name.endswith('_stdout.txt'):
                raise ValueError('unexpected experiment payload in text archive')
            relative=path.relative_to(run);tar.add(path,arcname=str(relative),recursive=False)
            members.append(dict(member=str(relative),bytes=path.stat().st_size,sha256=file_sha(path)))
    with tarfile.open(archive,'r:gz') as tar:
        if sorted(tar.getnames())!=sorted(r['member'] for r in members):raise ValueError('incomplete log archive')
        for item in members:
            if hashlib.sha256(tar.extractfile(item['member']).read()).hexdigest()!=item['sha256']:
                raise ValueError('archived log hash mismatch')
    atomic_json(folder/'archive_index.json',dict(archive=str(archive.relative_to(ROOT)),sha256=file_sha(archive),
        members=members,includes_failed_source_staging_attempt=True,
        excludes=['images','checkpoints','GT','source_truth','progress.sqlite','raw cache']))
    return dict(files=len(members),path=str(archive.relative_to(ROOT)),sha256=file_sha(archive))


def report(output=OUTPUT,run=RUN):
    output,run=Path(output),Path(run);validate_seal(output)
    data=json.loads((output/'evaluation_summary.json').read_text());audit=data['protocol'];gates=data['gates']
    metrics={r['method']:r for r in data['quality'] if r['scope']=='GT_present_frame_queries'}
    all_metrics={r['method']:r for r in data['quality'] if r['scope']=='all425_official_scored'}
    cost={r['method']:r for r in data['per_method_costs']};boot=json.loads((output/'bootstrap.json').read_text())
    recovery=json.loads((output/'GT_recovery_and_FP.json').read_text())
    stage=json.loads((output/'stage_costs_and_attempts.json').read_text());log_archive=archive_logs(run,output)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    table=['| 方法 | AP | AUROC | R@1 | POS(真/总) | POS精度 | GT覆盖 | UNKNOWN |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method,r in metrics.items():
        table.append(f'| {SHORT[method]} | {pct(r["AUPRC"])} | {pct(r["AUROC"])} | {pct(r["Recall_at1"])} | '
            f'{r["POS_true_candidates"]}/{r["POS_count"]} | {pct(r["POS_precision"])} | {pct(r["GT_positive_coverage"])} | {pct(r["UNKNOWN_rate"])} |')
    all_table=['| 方法 | 全425 AP | POS(真/总) | NEG误杀(真/总) | UNKNOWN |',
        '|---|---:|---:|---:|---:|']
    for method,r in all_metrics.items():
        all_table.append(f'| {SHORT[method]} | {pct(r["AUPRC"])} | {r["POS_true_candidates"]}/{r["POS_count"]} | '
            f'{r["NEG_true_candidates_killed"]}/{r["NEG_count"]} | {pct(r["UNKNOWN_rate"])} |')
    costs=['| 方法 | 累计worker秒 | 实际方法wall秒 | 分配GPU小时 | 新前向 | 所需独立raw调用 | 逻辑调用槽 | 逻辑视觉/语言token | 前向峰值GiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method in cost:
        r=cost[method]
        costs.append(f'| {SHORT[method]} | {r["worker_wall_seconds"]:.2f} | {r["actual_method_wall_span_seconds"]:.2f} | {r["GPU_hours_allocated_worker"]:.4f} | '
            f'{r["actual_new_prompt_forwards"]} | {r["unique_required_forwards"]} | {r["logical_request_slots"]} | '
            f'{r["logical_visual_tokens"]}/{r["logical_language_tokens"]} | {r["peak_allocated_bytes"]/(1<<30):.3f} |')
    a,b,c,d=(metrics[m] for m in ['YES_FULL','YES_ENSEMBLE_BUDGET_MATCHED','STRUCTURED_EVIDENCE','EVIDENCE_VERIFIED'])
    def delta(x,y,key):return fnum(x[key]-y[key]) if x[key] is not None and y[key] is not None else 'N/A'
    comparisons=boot['matched_coverage_POST_HOC_comparisons']
    report_text=f'''# P1：V2→Refer-Dance 冻结视觉语义教师可靠性与公平性报告

## 结论与边界

科学验收：`{gates['status']}`。Gate A至少一组通过：**{gates['any_teacher_Gate_A_passed']}**；
Evidence进入P2的Gate：**{gates['evidence_P2_ready']}**。本轮学生更新为0；完成P1后停止，未启动蒸馏长训。
工程完成不等于科研成功，不能用更高置信度、loss下降或少量接受样本代替语义证据。

这是已多次诊断的官方25Val开发性稀疏实验，不是新盲测，不是完整视频RMOT HOTA。
T0为`Qwen/Qwen3-VL-2B-Instruct`，HF revision`89644892e4d85e24eaac8bacfd4f463576704203`。
权重文件SHA256`7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`。
BF16、无量化、冻结、无adapter；YesTrack官方KITTI-V2 LoRA未能正常公开下载/本机核验，用户明确允许缺失，不以V1/Dance/其它项目自训adapter替代。

## 覆盖和官方计分协议

完整probe覆盖25视频、425video-expression、5个预定锚点/视频，共{audit['sampled_frame_queries']}个frame-query、
每方法8381个固定候选—查询配对；独立完整smoke每方法1649对。
抽样完全由GT-free MOT/原句/封存源分数确定；固定manifest SHA为
`a75c2da4453cc279c7560f831a53bb09fcb9f17d71ffbaf5db4885fe0eadee2c`。
官方GT按mark非0、class=1、帧偏移0、IoU≥0.5计分，{audit['query_present_episodes']} Query-Present与
{audit['query_empty_episodes']} Query-Empty。固定锚点有{audit['GT_present_frame_queries']}个GT-present frame-query，
{audit['GT_observations_at_fixed_anchors']}条GT观测、{audit['positive_candidate_opportunities']}次候选正匹配机会，
正候选分布于{len(audit['positive_videos'])}视频。
官方未匹配/空查询不等于画面已被逐人逐属性穷尽核验。主语义Gate在预注册GT-present frame-query上计算；全部425官方计分另列，绝不删除。
同一人多个MOT框可重复形成candidate TP，故GT覆盖使用一对一匹配，不能把候选TP当成不同人。

## 主语义质量（GT-present固定frame-query）

{chr(10).join(table)}

AP/AUROC是candidate级；R@k为每个GT-present frame-query一对一恢复GT比例的宏平均；
GT覆盖是所有锚点官方有效GT中接受POS的一对一匹配比例。最多4个采样候选，R@5不会超出采样库。
AP比较均对同样8381采样库做相同筛选，不混用历史全量Frozen 0.650204 AUROC或HOTA 8.2180。

## 全425官方辅助口径（不能解释为穷尽语义真值）

{chr(10).join(all_table)}

逐episode、逐视频、表达子组、presence及query-swap见同名CSV；不存在独立属性GT，所有17表达均含动作与服饰，
APPEARANCE/ACTION子组重叠，SPATIAL样本为0，不能声称完成空间关系或逐属性准确率验证。
Query-swap只用同一实际候选跨真实表达的采样交集，不补造缺失分数，不把其它可真表达一律当负例。

## 归因、覆盖与不确定性

- B−A AP：{delta(b,a,'AUPRC')}，R@1：{delta(b,a,'Recall_at1')}。
- C−B AP：{delta(c,b,'AUPRC')}，R@1：{delta(c,b,'Recall_at1')}。
- D−C AP：{delta(d,c,'AUPRC')}；D和C同原始分属性分数，验证器只改变三值接受状态，不凭筛选增加原始排序信息。
- D对B-plus同视觉、同调用槽、同验证器比较：`{json.dumps(comparisons['EVIDENCE_VERIFIED_vs_YES_ENSEMBLE_B_PLUS'],ensure_ascii=False)}`。
- D的UNKNOWN原因：`{json.dumps(data['unknown_reasons']['EVIDENCE_VERIFIED'],ensure_ascii=False)}`。
- 完整25视频配对bootstrap，2000次、seed0，相关帧/表达不作为独立样本；有效重采样数和CI保留在bootstrap.json。
- 全425 D恢复/丢失及FP：`{json.dumps(recovery['EVIDENCE_VERIFIED'],ensure_ascii=False)}`。
- 新增/删除FP预测行按新增行是否IoU匹配计；一对一匹配重分配导致的FP变化单列，不相互冒充。
- SoftMatch-inspired仅用源冻结置信度半高斯权重，无额外硬门限；若接受POS权重均1则与A接受集合相同，不能宣称独立监督信号。

## 计算与输入公平性

{chr(10).join(costs)}

A只做原整句当前ROI，成本低；B与C逐候选对应相同视图/调用槽；B-plus逐一匹配D的5类干预视图、调用槽，并使用同验证器。
语言token受问法影响，实际报告，不宣称完全相同token。D/B-plus公平性以逐输入图像SHA和视图类型核验，而非只比参数量。
缓存可以共享，但表中所需raw调用和逻辑槽不抹去缓存前置成本；SoftMatch继承A全部MLLM成本。
分配GPU小时是共享GPU worker存活时间，不是独占利用率或能耗；等价raw秒是原批次均摊观察值，不能冒充另跑的独立实测时长。
全部pipeline阶段含失败/恢复的GPU分配时间{stage['all_pipeline_allocated_GPU_hours']:.4f}小时；
另有首个2对功能检查16.70秒。各阶段wall time、吞吐、峰值、原始请求/图像/token和所有失败记录另列。
TCP/TRP按用户授权固定13个因果活动帧原策略重置，保留6帧TCP、stride4、原TRP，不是从视频开头累积状态，更不是完整视频HOTA。

## 源规则、泄露隔离与风险

固定源4视频64窗口、203对，源POS/NEG门限{data['source_rules']['positive_threshold']}/{data['source_rules']['negative_threshold']}，
仅源开发选择。源验证器接受5个POS、精度1.0，但覆盖仅5/82，样本小且是源整句校准，不能证明Dance服饰教师可靠。
所有六组完整预测、prompt/config/raw缓存索引先SHA封存，独立进程随后才读取目标GT。
所有教师receipt记录0目标GT打开尝试、0参数更新，原始模型完整参数SHA前后相同；Linux Landlock内核限制只读输入。
官方教师MLLM大规模预训练语料无法逐条核验，Refer-Dance潜在重叠**未知**；本轮未加载Dance LoRA或用Dance GT调prompt/阈值。
遮挡可能OOD、临近历史重叠强相关，视觉依赖与身份一致性不是语义真值；不把任何单项证据称为独立oracle。

## 用户16项问题逐项回答

1. 实际Qwen3-VL-2B-Instruct T0；**没有**加载KITTI-V2 LoRA，T1缺失已授权。
2. 教师目标标签输入0；目标GT仅封存后独立评测。
3. 全25个Dance Val视频、425表达均覆盖。
4. A/B/C/D/B-plus/TCP各8381对；完整smoke各1649对，非上限8500/1700虚报。
5. 相同候选/分辨率/预设历史，B=C同视图调用槽、B-plus=D同干预调用槽；A较便宜，语言token实测不同，不能称全组预算完全相等。
6. A主语义AP {pct(a['AUPRC'])}，AUROC {pct(a['AUROC'])}，POS精度 {pct(a['POS_precision'])}，GT覆盖 {pct(a['GT_positive_coverage'])}。
7. B−A AP {delta(b,a,'AUPRC')}，R@1 {delta(b,a,'Recall_at1')}，不把多提示词收益归因于分属性。
8. C−B AP {delta(c,b,'AUPRC')}，R@1 {delta(c,b,'Recall_at1')}，Gate B为{gates['Gate_B']['STRUCTURED_EVIDENCE']['passed']}。
9. D Gate B为{gates['Gate_B']['EVIDENCE_VERIFIED']['passed']}；必须超过强B-plus，不能用双loss或更多调用宣称贡献。
10. POS P–C完整曲线见precision_coverage.csv，部署门限与离线扫分数诊断分列，不回写目标阈值。
11. D候选正机会正确监督{d['POS_true_candidates']}/{d['positive_candidate_opportunities']}；一对一GT监督{d['accepted_matched_GT']}/{d['GT_observations']}，
    未被POS覆盖的GT观测{d['GT_observations']-d['accepted_matched_GT']}；NEG误杀候选{d['NEG_true_candidates_killed']}，不重复计算同一人。
12. D主口径UNKNOWN {pct(d['UNKNOWN_rate'])}，全425为{pct(all_metrics['EVIDENCE_VERIFIED']['UNKNOWN_rate'])}；必须同时看GT覆盖，不能靠拒绝大多数样本报成功。
13. 没有已加载的目标适配权重；通用预训练潜在Refer-Dance重叠无法排除，状态UNKNOWN。
14. 40Train图像/表达已有，匹配官方DETR+NeuralSORT预测MOT仍缺失，不用GT补齐。
15. Evidence P2 Gate为{gates['evidence_P2_ready']}，formal_P2_executable=False；停止P1，不自动蒸馏。若只有简单教师通过，可仅作为以后需授权的基线设计。
16. 分支research/v2dance-evidence-teacher-p1-20261026；评测/报告源提交{commit}。公开映射见publication.json，完整复现命令见下一节。

## 复现与恢复

```bash
cd /data2/user/FlexHook-TTA
# 默认仅打印计划；--execute才执行，已有sealed方法自动保留。
bash scripts/run_teacher_probe_4gpu.sh --phase calibrate --execute
/data2/user/TTA-RMOT-assets/teacher-p1/env/bin/python scripts/calibrate_teacher_reliability.py --predictions results/runs/teacher_probe_p1_20261026_v1/calibrate/T0/SOURCE_RELIABILITY/teacher_outputs.jsonl --output results/teacher_probe/source_calibration_rules_T0.json
bash scripts/run_teacher_probe_4gpu.sh --phase smoke --rules results/teacher_probe/source_calibration_rules_T0.json --execute
bash scripts/run_teacher_probe_4gpu.sh --phase full --rules results/teacher_probe/source_calibration_rules_T0.json --execute
/home/szgx/anaconda3/envs/dfot/bin/python scripts/audit_teacher_leakage.py
/home/szgx/anaconda3/envs/dfot/bin/python scripts/evaluate_teacher_probe.py
/home/szgx/anaconda3/envs/dfot/bin/python scripts/report_teacher_probe.py
```

恢复根目录`{run}`；原始输出/receipt、共享SQLite cache及teacher_cache_index指定的完整raw索引保留服务器。
首个source暂存缺帧失败已修复，38张合法源图补齐，源样本/真值hash不变；失败日志未隐藏。
完整文本日志归档{log_archive['files']}成员，SHA`{log_archive['sha256']}`；不含图像、权重或GT。
历史六TTA/UDA结果未覆盖；本轮不产生新的完整RMOT HOTA，不把开发诊断当独立确认。

![P1 sparse quality](../results/teacher_probe/figures/teacher_p1_quality.png)
'''
    (ROOT/'docs/TEACHER_P1_FINAL_REPORT.md').write_text(report_text)
    readiness=f'''# P2 readiness：证据驱动，不自动开训

本轮P1判决：`{gates['status']}`。任何教师Gate A通过={gates['any_teacher_Gate_A_passed']}；
Evidence Gate C通过={gates['evidence_P2_ready']}。正式P2当前不可执行，不启动学生训练。

完整Gate逐项证据见`results/teacher_probe/final_scientific_verdict.json`、bootstrap及P1报告。
如果Gate A失败/支持不足，保留负结论，停止无依据的蒸馏；不能延长LoRA训练替代伪标签有效性证据。
若仅简单教师通过、B失败，只能保留简单蒸馏作为将来明确授权的效率基线，不宣称Evidence机制成功。

## 即使未来有足够证据，仍须先解决

1. 40个Train视频匹配官方冻结DETR+NeuralSORT的合法预测MOT；不能用GT框/ID，也不能静默换跟踪器。
2. GT无关的预注册随机30适应/10确认研究划分（不是官方Train/Test）；先核实确认10视频有合法指代GT。
3. 原25Val已经多轮诊断，只能作为开发性官方基准，不能再叫盲测。
4. 相同教师、候选、源真实监督、学生参数和训练预算的S0 Source/S1整句/S2置信加权/S3证据蒸馏。
5. C-Hook/PCD保留；若新增属性头，应配参数量匹配对照；不把容量或更大MLLM资源误当方法贡献。
6. 只有同预算优于现成S1/S2，并经完整TrackEval、presence、ranking、Count-Matched及独立视频确认后才讨论贡献。

这些是条件性研究准备，不是已执行的P2实验，也不是本轮新增训练授权。当前遵守STOP_AFTER_P1。
'''
    (ROOT/'docs/TEACHER_P2_READINESS.md').write_text(readiness)
    figures(output,metrics)
    atomic_json(output/'report_generation.json',dict(source_commit=commit,log_archive=log_archive,
        reports=['docs/TEACHER_P1_FINAL_REPORT.md','docs/TEACHER_P2_READINESS.md'],student_training_started=False))
    print(json.dumps(dict(report='docs/TEACHER_P1_FINAL_REPORT.md',verdict=gates['status'],logs=log_archive),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--run-root',type=Path,default=RUN);a=p.parse_args();report(a.output,a.run_root)


if __name__=='__main__':main()
