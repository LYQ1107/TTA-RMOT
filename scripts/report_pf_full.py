#!/usr/bin/env python3
"""Produce the final scientific report only after all six full results exist."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ("ZERO_UPDATE", "RELATIVE_ONLY", "PF_RANK", "PF_RANK_GUARD", "PF_IDENTITY", "PF_HYBRID")


def generate(publish, output):
    reports, attribution = {}, {}
    for method in ORDER:
        folder = Path(publish) / method.lower()
        reports[method] = json.loads((folder / "metrics.json").read_text())
        attribution[method] = json.loads((folder / "offline_attribution.json").read_text())
        if reports[method]["full_audit"]["valid_episodes"] != 425 or len(attribution[method]["episodes"]) != 425:
            raise ValueError("the final report cannot omit incomplete episodes or methods")
    base = reports["ZERO_UPDATE"]["metrics"]
    primary = ("PF_RANK", "PF_IDENTITY", "PF_HYBRID")
    best = max(primary, key=lambda m: reports[m]["metrics"]["HOTA"])
    delta = reports[best]["metrics"]["HOTA"] - base["HOTA"]
    recovered = attribution[best]["recovered_gt"]-attribution[best]["lost_gt"]
    if delta <= 0:
        verdict = "三种主 TTA 方法均未超过匹配 Frozen/ZERO_UPDATE；不宣称方法成功。"
    elif recovered <= 0:
        verdict = "最优主 TTA 的 HOTA 提升，但离线净 GT 恢复不为正；不能将该提升解释为目标召回恢复或新语义学习。"
    else:
        verdict = "最优主 TTA 同时提升 HOTA 与离线净 GT 恢复；这支持本开发集上的选择改善，但不等同于独立盲测泛化或已证明学到新语义。"
    lines = ["# FlexHook V2→Dance Positive-Free TTA：完整实验报告", "",
             f"全部六种方法均已完成 25 视频 / 425 表达。主 TTA 最优：{best}，HOTA {reports[best]['metrics']['HOTA']:.4f}，相对匹配基线 {delta:+.4f} 个百分点。",
             "", verdict, "", "## 固定协议与基线", "",
             "- 执行项目仅为 `LYQ1107/TTA-RMOT`，目录 `/data2/user/FlexHook-TTA`，与 LocateMOT/GroundJEV 无关。",
             "- 固定官方 Refer-KITTI-V2 源 checkpoint，SHA256 `71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613`；不使用 Dance 第二阶段权重。",
             "- Dance 专用官方 DETR+NeuralSORT MOT：25 文件、211,085 条固定框/ID；每视频 17 表达，共 3,588,445 个候选-表达分数，完整时序。",
             "- 默认 rank=4、lr=5e-5、update_interval=4、min_history=3、min_separation=0.6、seed=0、FP32；每 episode 重置可训练状态。没有源端重训或外部教师。",
             "- 用户授权显存足够时共享 GPU，最多同时四张物理卡，优先少进程；24 GiB/worker 的 PyTorch 分配器预算加 8 GiB 入场余量不是跨用户显存预留。",
             "- 原 Frozen 漏掉 692 个轨迹末观测。用户明确批准补齐全候选并重建匹配 Frozen；历史 HOTA 8.2060 不用于本轮增益计算。原文件、补充分数与失败记录均保留。",
             "- 完整 ZERO_UPDATE 的真实 FP32 margin 构成后续全部方法共用的 Frozen 参考。对历史近零翻号逐项比较未缓存原始前向与缓存前向；有实际选择差异则阻止继续。",
             "- 全部 Frozen 分数独立 CPU 回放必须逐行复现 425 个 ZERO_UPDATE 预测；它不是独立第二次 GPU 推理，不能据此夸大模型复现证据。",
             "- 在线过程不读目标 GT、不使用未来帧、不改 MOT；预测完整封存后才由独立 TrackEval 和错误归因读取 GT。这是开发性目标域实验，不是独立盲测。",
             "", "## 官方全量指标", "",
             "指标百分数范围为 0–100；Δ 是相对完整匹配 ZERO_UPDATE 的百分点变化。",
             "", "| 方法 | HOTA | ΔHOTA | DetA | AssA | DetRe | DetPr | IDF1 | IDSW |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method in ORDER:
        m = reports[method]["metrics"]
        lines.append(f"| {method} | {m['HOTA']:.4f} | {m['HOTA']-base['HOTA']:+.4f} | " +
                     " | ".join(f"{m[k]:.4f}" for k in ("DetA", "AssA", "DetRe", "DetPr", "IDF1")) + f" | {m['IDSW']:.0f} |")
    lines += ["", "## 召回、误检与候选排序归因", "",
              "下表仅为固定 IoU≥0.5 的离线逐帧最大基数匹配诊断，不替代官方 HOTA；重复框作为 FP。恢复/丢失均按 GT 观测计数。",
              "", "| 方法 | 选中行 | 选中 ID（episode 累计） | 恢复 GT | 丢失 GT | 新增 FP 行 | 移除 FP 行 | Gate recall | AUROC | Top1 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method in ORDER:
        a, q = attribution[method], reports[method]["full_audit"]
        auc = "N/A" if a["adapted_auroc"] is None else f"{a['adapted_auroc']:.6f}"
        lines.append(f"| {method} | {q['selected_rows']} | {q['selected_ids']} | {a['recovered_gt']} | {a['lost_gt']} | " +
                     f"{a['new_false_positive_rows']} | {a['removed_false_positive_rows']} | {a['adapted_conditional_gate_recall']:.6f} | {auc} | {a['ranking']['adapted']['1']:.6f} |")
    a0 = attribution["ZERO_UPDATE"]
    lines += ["", f"固定候选的可检索 GT 召回上界（逐 GT any-IoU）为 {a0['conditional_candidate_recall']:.6f}；所有方法的候选全集相同。",
              f"Top1 随机选择对照 {a0['ranking']['random_null']['1']:.6f}；Top3/Top5、每表达计数和完整 AUROC 见各方法 `offline_attribution.json`。",
              "GT 仅计官方 nonzero mark 且 pedestrian class=1 的行。新增 FP 指新加入选择集且未匹配的行；完整 FP 净变化另存 JSON，不能假设它恒等于新增减移除 FP（已有行的匹配状态也可能改变）。",
              "", "## 梯度、有效证据与回滚", "",
              "| 方法 | optimizer steps | 非零有限梯度 steps | 无证据跳过 | interval 跳过 | rollback | max adapter Δ L2 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for method in ORDER:
        s = reports[method]["aggregate_stats"]
        lines.append(f"| {method} | {s.get('optimizer_steps',0)} | {s.get('finite_nonzero_gradient_steps',0)} | " +
                     f"{s.get('skipped_no_evidence',0)} | {s.get('skipped_interval_frames',0)} | {s.get('rollback_updates',0)} | {s.get('max_adapter_update_l2',0):.8f} |")
    lines += ["", "`skipped_no_pairs` 历史计数包含间隔跳过，不能将其全部解释为无证据。",
              "RELATIVE_ONLY 不做梯度更新，排序分数与 Frozen 相同；即使门禁选择改善 HOTA，也不是新语义学习。",
              "PF_RANK_GUARD 是 PF_RANK 分数与 Frozen 正分候选取并集的显式保留消融，不是新增语义学习；若采用 CPU 派生，重用的 Rank 更新与额外 GPU 计算分开记录。",
              "", "## 实测计算成本", "",
              "| 方法 | 调度 wall 秒 | episode 计算累计秒 | GPU 小时 | 平均 worker 活跃帧 FPS | 调度 wall 活跃帧 FPS | PyTorch 峰值 GiB |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for method in ORDER:
        c = reports[method]["cost"]
        lines.append(f"| {method} | {c['wall_seconds']:.2f} | {c['total_episode_seconds']:.2f} | {c['gpu_hours']:.4f} | " +
                     f"{c['worker_active_frame_fps']:.3f} | {c['scheduler_wall_active_frame_fps']:.3f} | {c['peak_vram_bytes']/2**30:.3f} |")
    lines += ["", "峰值为 PyTorch max_memory_allocated，不包含 CUDA context 或 allocator reserved；不能解释为整卡峰值。",
              "GPU 小时按实际 worker 起止累计；最初 GPU5 完成 102 表达，后续 GPU6/4/0/1 同时处理剩余表达，所以不同阶段出现过五个 UUID，但从未同时用五张卡。",
              "平均 worker FPS 的分母是各 episode 计算时间之和，不是四卡并行吞吐；调度 wall FPS 的分母包含等待，但不包含前置预检、评测与归因。两者不能混用。",
              "ZERO wall 包含旧空卡策略等待和用户授权后恢复；共享任务会竞争算力，单卡子集与四卡全量并非相同工作负载，不报告未经控制的“4×加速”。",
              "GPU 等价诊断、Guard 单完整 episode 验证、官方评测/归因是额外开销，保留在独立 receipt/log；表内 method wall 不自动包含这些阶段。",
              "", "## 逐方法判断", ""]
    for method in ORDER[1:]:
        m, a = reports[method]["metrics"], attribution[method]
        if m["AssA"] > base["AssA"] and m["DetRe"] < base["DetRe"]:
            judgment = "AssA 上升但 DetRe 下降，不能描述为召回恢复。"
        elif m["HOTA"] <= base["HOTA"]:
            judgment = "未提高官方总体 HOTA；保留完整负面结果。"
        else:
            judgment = "HOTA 有增益，仍需结合召回/FP/排序权衡解释，不据此声称已学到新语义。"
        lines.append(f"- {method}：ΔHOTA {m['HOTA']-base['HOTA']:+.4f}，ΔIDF1 {m['IDF1']-base['IDF1']:+.4f}，净恢复 GT {a['recovered_gt']-a['lost_gt']:+d}。{judgment}")
    lines += ["", "## 可复查材料", "",
              "- GitHub：本分支 `results/positive_free/full/<method>/` 下的官方 summary/detailed CSV、425 条逐表达和 25 个逐视频指标、全量审计摘要与离线归因。逐视频使用官方 combine_sequences 池化，不是宏平均。",
              "- `results/positive_free/full/manifests/`：测试记录、资产预检、原 Frozen 覆盖缺陷、补充 receipt、共享分片、完整边界诊断、匹配 Frozen receipt。",
              "- 服务器原始结果：`/data2/user/FlexHook-TTA/results/runs/pf_*_completebank_full_20261025*`，保留每表达预测、压缩 margin trace、完成/失败收据和运行日志。大体积原始文件不伪装成已上传 GitHub 的模型权重或图像。",
              "- 全量指标由官方 TrackEval 计算，没有缺失表达填空、删视频、改种子或根据目标 GT 选择阈值。",
              "", "研究限制：仅 V2→Dance 单源单目标、固定官方 tracker 和单个预注册 seed；不支持由此泛化到 BDD/AerialMind 或反向迁移。未来扩展需另行明确授权。"]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(methods=6, best_primary_tta=best, delta_hota=delta,
                          net_recovered_gt=recovered, report=str(output)), ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--publish-root", type=Path, default=ROOT / "results/positive_free/full")
    p.add_argument("--output", type=Path, default=ROOT / "docs/PF_FINAL_SCIENTIFIC_REPORT_20261025.md")
    args = p.parse_args()
    generate(args.publish_root, args.output)


if __name__ == "__main__": main()
