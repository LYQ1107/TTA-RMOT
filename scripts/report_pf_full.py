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
        diagnostic = folder / "offline_query_diagnostics.json"
        attribution[method] = json.loads((diagnostic if diagnostic.exists() else folder / "offline_attribution.json").read_text())
        if reports[method]["full_audit"]["valid_episodes"] != 425 or len(attribution[method]["episodes"]) != 425:
            raise ValueError("the final report cannot omit incomplete episodes or methods")
        if "query_presence" not in attribution[method]:
            raise ValueError("final report requires query presence/absence diagnostics for all methods")
    uncertainty = json.loads((Path(publish)/"video_uncertainty.json").read_text())
    if uncertainty["status"] != "COMPLETE_6_METHODS":
        raise ValueError("final report requires the all-six paired video-cluster analysis")
    base = reports["ZERO_UPDATE"]["metrics"]
    primary = ("PF_RANK", "PF_IDENTITY", "PF_HYBRID")
    best = max(primary, key=lambda m: reports[m]["metrics"]["HOTA"])
    best_overall = max(ORDER, key=lambda m: reports[m]["metrics"]["HOTA"])
    delta = reports[best]["metrics"]["HOTA"] - base["HOTA"]
    recovered = attribution[best]["recovered_gt"]-attribution[best]["lost_gt"]
    if delta <= 0:
        verdict = "三种主 TTA 方法均未超过匹配 Frozen/ZERO_UPDATE；不宣称方法成功。"
    elif recovered <= 0:
        verdict = "最优主 TTA 的 HOTA 提升，但离线净 GT 恢复不为正；不能将该提升解释为目标召回恢复或新语义学习。"
    elif not uncertainty["comparisons"][best+"_vs_ZERO_UPDATE"]["CI_excludes_zero"]:
        verdict = "最优主 TTA 的 HOTA 与净 GT 恢复点估计为正，但配对视频聚类95%区间包含0；现有证据不足以支持稳定的正向收益，不能据此宣布科研成功或跨模态语义恢复。"
    else:
        verdict = "最优主 TTA 同时提升 HOTA 与离线净 GT 恢复；这支持本开发集上的选择改善，但不等同于独立盲测泛化或已证明学到新语义。"
    lines = ["# FlexHook V2→Dance Positive-Free TTA：完整实验报告", "",
             f"全部六种方法均已完成 25 视频 / 425 表达。主 TTA 最优：{best}，HOTA {reports[best]['metrics']['HOTA']:.4f}，相对匹配基线 {delta:+.4f} 个百分点。",
             f"含冻结基线和无梯度/保留消融在内，全部方法中最高 HOTA 为 {best_overall}: {reports[best_overall]['metrics']['HOTA']:.4f}。",
             "", verdict, "", "## 固定协议与基线", "",
             "- 执行项目仅为 `LYQ1107/TTA-RMOT`，目录 `/data2/user/FlexHook-TTA`，与 LocateMOT/GroundJEV 无关。",
             "- 固定官方 Refer-KITTI-V2 源 checkpoint，SHA256 `71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613`；不使用 Dance 第二阶段权重。",
             "- Dance 专用官方 DETR+NeuralSORT MOT：25 文件、211,085 条固定框/ID；每视频 17 表达，共 3,588,445 个候选-表达分数，完整时序。",
             "- Source-Free 指第二阶段 FlexHook 适应不读取源数据和目标 GT；候选来自目标域专用官方跟踪器，不能将本实验表述为整套检测跟踪系统完全无目标域监督。",
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
    empty_count = sum(e["gt_observations"] == 0 for e in a0["episodes"])
    lines += ["", f"固定候选的可检索 GT 召回上界（逐 GT any-IoU）为 {a0['conditional_candidate_recall']:.6f}；所有方法的候选全集相同。",
              f"Top1 随机选择对照 {a0['ranking']['random_null']['1']:.6f}；Top3/Top5、每表达计数和完整 AUROC 见各方法 `offline_attribution.json`。",
              "GT 仅计官方 nonzero mark 且 pedestrian class=1 的行。新增 FP 指新加入选择集且未匹配的行；完整 FP 净变化另存 JSON，不能假设它恒等于新增减移除 FP（已有行的匹配状态也可能改变）。",
              "", f"固定 425 个 episode 中 {empty_count} 个 GT 为空（按本协议视为无匹配），其余 {425-empty_count} 个有有效 GT。全部保留，绝不改成仅评正目标的子集。",
              "", "| 方法 | GT 空 episode 上的 FP | 有 GT episode 上的 FP |",
              "|---|---:|---:|"]
    for method in ORDER:
        episodes = attribution[method]["episodes"]
        empty_fp = sum(e["adapted_false_positive_rows"] for e in episodes if not e["gt_observations"])
        present_fp = sum(e["adapted_false_positive_rows"] for e in episodes if e["gt_observations"])
        lines.append(f"| {method} | {empty_fp} | {present_fp} |")
    lines += ["", "GT 空组只用于封存后的错误解释，未反馈到在线阈值、损失或超参数。候选排序在有目标帧上的改善不能单独证明无匹配拒识已改善。",
              "", "## Query-Present / Query-Empty：单独验收", "",
              "下表召回为离线 IoU≥0.5 最大基数匹配的 GT 观测微平均，包含无候选 GT；不是官方跨阈值 DetRe。空组无 GT，HOTA 不具备有效召回解释，不将它改成正目标子集。",
              "", "| 方法 | Present / Empty episodes | Present GT recall | Present FP | Empty FP | Empty 非空预测 episode 比例 |",
              "|---|---:|---:|---:|---:|---:|"]
    for method in ORDER:
        a = attribution[method]
        lines.append(f"| {method} | {a['query_present_episodes']} / {a['query_empty_episodes']} | {100*a['query_present_GT_recall']:.4f}% | " +
                     f"{a['query_present_FP']} | {a['query_empty_FP']} | {100*a['query_empty_nonempty_prediction_rate']:.4f}% |")
    lines += ["", "多目标 Recall@K 与旧 TopK any-hit 分开：前者是在可检索目标帧内 topK 覆盖的正候选比例的帧宏平均；后者只要求 topK 至少命中一个正候选。两者均不衡量空查询拒识。",
              "", "| 方法 | Source Recall@1 / 3 / 5 | Adapted Recall@1 / 3 / 5 |",
              "|---|---:|---:|"]
    for method in ORDER:
        a = attribution[method]
        source = " / ".join(f"{a['source_Recall_at_1_3_5'][str(k)]:.6f}" for k in (1,3,5))
        adapted = " / ".join(f"{a['adapted_Recall_at_1_3_5'][str(k)]:.6f}" for k in (1,3,5))
        lines.append(f"| {method} | {source} | {adapted} |")
    lines += ["", "每方法 JSON 另存四个 episode 活跃帧进度区间的负预测比例，检查是否随时间持续抑制输出；它不是独立帧显著性检验。",
              "身份诊断仅统计 query-GT 可见、间隔≤2帧的物理ID对应GT匹配身份变化及可能更新机会；遮挡/重叠时匹配重分配也可能产生变化，不能当作真实MOT身份错误率。它不覆盖未标注非目标人、GT缺口，也不证明所有正对可靠。在线记忆的 max_gap=2 无法独立识别短间隔 ID 互换；这些 GT 诊断从未参与训练。",
              "", "## 视频级收益分布与不确定性", "",
              "成对 bootstrap 以25个完整视频为聚类单位，每次保留视频内全部表达与帧，2000次重采样、seed0；重新池化官方HOTA充分统计量，不平均视频HOTA。区间只描述本开发集的视频抽样不确定性，不是独立盲测或训练种子显著性。",
              "", "| 比较 | ΔHOTA | 视频聚类95%区间 | 提升 / 下降 / 不变视频 | Query-Present ΔHOTA |",
              "|---|---:|---:|---:|---:|"]
    for name, r in uncertainty["comparisons"].items():
        low,high = r["paired_video_cluster_CI95"]
        lines.append(f"| {name} | {r['delta_hota']:+.6f} | [{low:+.6f}, {high:+.6f}] | " +
                     f"{r['improved_videos']} / {r['declined_videos']} / {r['tied_videos']} | {r['query_present_delta_hota']:+.6f} |")
    rank_pair = uncertainty["comparisons"]["PF_RANK_vs_ZERO_UPDATE"]
    for row in rank_pair["most_influential_videos"][:2]:
        lines.append(f"- Rank影响较大的视频 {row['video']}：单视频ΔHOTA {row['delta_hota']:+.6f}；去掉该视频后总体ΔHOTA {row['delta_hota_without_video']:+.6f}。HOTA非线性，不能把单视频差值当可加贡献。")
    lines += ["", "## 有效证据与计算收益", "",
              "| 方法 | 每副本可训练参数 | 更新机会 | 有证据机会覆盖 | 接受更新 / 机会 |",
              "|---|---:|---:|---:|---:|"]
    for method in ORDER:
        s = reports[method].get("reused_rank_stats", reports[method]["aggregate_stats"])
        opportunities = s.get("update_opportunities",0)
        eligible = (opportunities-s.get("skipped_no_evidence",0)-s.get("skipped_multi_batch",0))/opportunities if opportunities else None
        coverage = f"{100*eligible:.4f}%" if eligible is not None else "N/A"
        accepted = f"{100*s.get('optimizer_steps',0)/opportunities:.4f}%" if opportunities else "N/A"
        lines.append(f"| {method} | {s.get('trainable_parameters',0)} | {opportunities} | {coverage} | {accepted} |")
    rank = reports["PF_RANK"]
    lines += ["", f"Rank 的 {rank['aggregate_stats']['optimizer_steps']:,} 次更新消耗 {rank['cost']['gpu_hours']:.4f} GPU小时、wall {rank['cost']['wall_seconds']:.2f}s，仅得到 +0.0253 个HOTA百分点，且AUROC/AssA下降，当前证据不支持它具有突出的收益成本比。",
              "", "## Hybrid 互补性验收", ""]
    hybrid = reports["PF_HYBRID"]["metrics"]
    stronger = max(("PF_RANK","PF_IDENTITY"), key=lambda m:reports[m]["metrics"]["HOTA"])
    if hybrid["HOTA"] <= reports[stronger]["metrics"]["HOTA"]:
        lines.append(f"Hybrid 未超过较强单模块 {stronger}，未显示整体HOTA互补收益，不因两个loss而选择它为最终方法。")
    else:
        lines.append(f"Hybrid 的HOTA高于较强单模块 {stronger}；仍须结合上表成对视频区间和召回/误检代价，不据小差值认定独立互补或语义恢复。")
    if hybrid["AssA"] > reports[stronger]["metrics"]["AssA"] and hybrid["DetRe"] < reports[stronger]["metrics"]["DetRe"]:
        lines.append("Hybrid相对较强单模块以DetRe下降换取AssA上升，不能描述为所需的目标召回互补。")
    lines += ["", "## 梯度、有效证据与回滚", "",
              "| 方法 | optimizer steps | 非零有限梯度 steps | 无证据跳过 | interval 跳过 | rollback | max adapter Δ L2 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for method in ORDER:
        s = reports[method]["aggregate_stats"]
        if "reused_rank_stats" in reports[method]:
            reused = reports[method]["reused_rank_stats"]
            lines.append(f"| {method} | 复用 Rank {reused['optimizer_steps']}；新增 0 | " +
                         f"复用 Rank {reused.get('finite_nonzero_gradient_steps',0)} | 复用 Rank | 复用 Rank | " +
                         f"复用 Rank {reused.get('rollback_updates',0)} | {reused.get('max_adapter_update_l2',0):.8f}（复用） |")
            continue
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
              "RELATIVE_ONLY 的 CPU 成本同样仅指增量选择，前提是已有完整 ZERO_UPDATE 源分数；不能据此宣称从图像到 RMOT 输出不需要 GPU。",
              "若 Guard 为 CPU 派生，其表内成本仅为增量 CPU 重放，不代表独立从图像运行只需零 GPU；必须同时承担完整 Rank 推理/适应和单独的 GPU 等价验证成本。",
              "GPU 小时按实际 worker 起止累计；ZERO_UPDATE 最初 GPU5 完成 102 表达，后续 GPU6/4/0/1 处理剩余表达，因此该方法累计出现五个 UUID，但从未同时用五张卡。后续各方法的真实 GPU 见独立调度记录。",
              "Rank 使用源推理时序估计的静态整视频分组；其耗时不能代表梯度计算的准确负载预测。Identity/Hybrid 的新计划改用最多四卡领取完整视频的队列，每视频 17 个独立 episode，不跨卡切断在线记忆；重载模型开销计入实测 wall。共享负载仍会变化，调度不保证固定加速倍数。",
              "平均 worker FPS 的分母是各 episode 计算时间之和，不是四卡并行吞吐；调度 wall FPS 的分母包含等待，但不包含前置预检、评测与归因。两者不能混用。",
              "ZERO wall 包含旧空卡策略等待和用户授权后恢复；共享任务会竞争算力，单卡子集与四卡全量并非相同工作负载，不报告未经控制的“4×加速”。",
              "GPU 等价诊断、Guard 单完整 episode 验证、官方评测/归因是额外开销，保留在独立 receipt/log；表内 method wall 不自动包含这些阶段。",
              "", "## 逐方法判断", ""]
    for method in ORDER[1:]:
        m, a = reports[method]["metrics"], attribution[method]
        if reports[method].get("interpretation") == "IDENTITY_CONSISTENCY_GAIN_WITH_REFERRING_RECALL_LOSS":
            judgment = "IDENTITY_CONSISTENCY_GAIN_WITH_REFERRING_RECALL_LOSS：仅身份一致性改善而指代表达召回下降，不是跨模态匹配成功。"
        elif m["AssA"] > base["AssA"] and m["DetRe"] < base["DetRe"]:
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
