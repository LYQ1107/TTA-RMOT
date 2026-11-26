#!/usr/bin/env python3
"""Generate the first-round scientific verdict from immutable offline results."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.pf_diag_common import write_csv
from scripts.diag_pf_query_information import binary_summary

PUBLIC=ROOT/'results/mllm_distillation'
FULL=PUBLIC/'full'


def read(name):return json.loads((FULL/name).read_text())
def number(x):return f'{x:.6f}'


def presence():
    frozen=json.loads((ROOT/'results/diagnostics/pf_recoverability_20261025/query_information/summary.json').read_text())['presence']
    rows=[];summaries={'FROZEN':frozen}
    for method in ('SOURCE_REHEARSAL','MLLM_DISTILL'):
        summaries[method]=json.loads((FULL/method/'presence_quality.json').read_text())
    fields=('mean_margin','p90_margin','mean_frame_max_margin','max_margin','positive_fraction')
    for method,values in summaries.items():
        for field in fields:rows.append(dict(method=method,statistic=field,AUROC=values[field]['AUROC'],AUPRC=values[field]['AUPRC']))
    write_csv(FULL/'presence_comparison.csv',rows)
    return summaries


def figures(summary,presence_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=PUBLIC/'figures';figures.mkdir(exist_ok=True)
    methods=['FROZEN','SOURCE_REHEARSAL','MLLM_DISTILL','FROZEN_AT_MLLM_COUNT','MLLM_AT_FROZEN_COUNT']
    names=['Frozen','Source-only','Naive distill','Frozen @ student count*','Student @ Frozen count*']
    colors=['#45566d','#a3adb9','#327da5','#bcc5ce','#94bfd4']
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    bars=axes[0].barh(names,[summary['official_metrics'][m]['HOTA'] for m in methods],color=colors)
    axes[0].invert_yaxis();axes[0].set_xlim(0,9.8);axes[0].set_xlabel('Official HOTA (25 videos / 425 queries)\n* Post-hoc controls; not deployable.',fontsize=9)
    axes[0].bar_label(bars,fmt='%.4f',padding=4);axes[0].set_title('No overall improvement over Frozen',fontsize=10)
    base=presence_rows['query_present'];empty=presence_rows['query_empty']
    labels=['Frozen','Naive distill'];tp=[base['baseline_matched_gt'],base['adapted_matched_gt']]
    fp_present=[base['baseline_false_positive_rows'],base['adapted_false_positive_rows']]
    fp_empty=[empty['baseline_false_positive_rows'],empty['adapted_false_positive_rows']]
    axes[1].bar(labels,tp,label='Matched GT rows (IoU .5)',color='#347c62')
    axes[1].bar(labels,fp_present,bottom=tp,label='FP: query-present',color='#dfaa68')
    axes[1].bar(labels,fp_empty,bottom=np.asarray(tp)+fp_present,label='FP: official-query-empty',color='#bd6963')
    axes[1].set_ylabel('Prediction rows');axes[1].set_title('Large pruning, only +101 net matched GT',fontsize=10)
    axes[1].legend(fontsize=8)
    fig.suptitle('V2 to Dance: first naive frozen-MLLM distillation (same-Val diagnosis)',fontsize=11)
    fig.savefig(figures/'first_round_attribution.png',dpi=170);fig.savefig(figures/'first_round_attribution.pdf');plt.close(fig)


def report():
    summary=read('summary.json');ranking=read('dense_ranking_transfer.json');text=read('text_signal_offline.json')
    optimization=read('optimization_diagnosis.json');cost=read('compute_cost.json');p=presence()
    if summary['status']!='COMPLETE_25VAL_TRANSDUCTIVE_FIRST_ROUND':raise ValueError('finish all425 first')
    offline=read('MLLM_DISTILL/offline_attribution.json');q=offline['query_presence'];teacher=ranking['teacher']
    comparison=summary['comparisons'];metrics=summary['official_metrics'];lines=[]
    def line(s=''):lines.append(s)
    line('# MLLM Distillation：完整首轮结果与科研验收')
    line();line('2026-10-29，`LYQ1107/TTA-RMOT`；分支`research/v2dance-mllm-naive-distill-20261028`。')
    line();line('结论：**Naive蒸馏HOTA 8.0327，低于Frozen8.2180，未通过进入Ours的科研门槛。**')
    line('不是“教师完全无用”：全量及非教师锚点的分数可分性改善；但同帧语义排序、正目标覆盖和轨迹关联没有形成一致最终收益。')
    line();line('## 1. 实际完成范围与数据隔离')
    line();line('按用户“40Train没有的话先用val”，这是25Val同集无标签离线蒸馏开发诊断，不是40Train→25Val，不是盲测，也不是Source-Free Online TTA。')
    line('40Train图像/表达存在，匹配官方DETR+NeuralSORT的预测MOT缺失；本轮没有用GT框/ID或替代检测器补造。P1及旧UDA/TTA结果未改写。')
    line('目标GT打开尝试在教师、学生训练和推理均为0；全部425预测及GT-free数量对照封存后，独立CPU进程才读取GT。没有GT调阈值、prompt、采样或checkpoint。')
    line('官方协议复核：25视频×17表达=425；40表达序列有官方GT、385无官方GT，212541个GT观测；mark=1/class=1，一基帧号/偏移0，框为top-left xywh。')
    line('“官方无GT”不等于画面中语义上绝对不存在目标；不能用这些离线分组作为在线存在性标签。Qwen完整预训练样本不公开，潜在同源视频重叠未知。')
    line();line('## 2. 完整官方RMOT')
    line();line('| 方法 | HOTA | DetA | AssA | DetRe | DetPr | IDF1 | IDSW | 输出行 |')
    line('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for method in ('FROZEN','SOURCE_REHEARSAL','MLLM_DISTILL','FROZEN_AT_MLLM_COUNT','MLLM_AT_FROZEN_COUNT'):
        m=metrics[method];line('| '+method+' | '+' | '.join(f'{m[k]:.4f}' for k in ('HOTA','DetA','AssA','DetRe','DetPr','IDF1'))+f" | {m['IDSW']:.0f} | {m['Dets']:.0f} |")
    line();line('Source-only经过12项协议逐项复核，可复用同源计划/seed17/初始化/容量/8973步/优化器/源验证选择的既有结果，其历史成本不记为免费。')
    line('历史Mean Teacher HOTA8.2494仅作参考：其3锚点/全候选/EMA损失与本轮5锚点/每次≤4/外部MLLM监督不同，不是公平同预算教师对照。')
    line();line('## 3. 数量匹配和视频级不确定性')
    line();line('两种Count-Matched都不读GT构造：Frozen按学生实际逐帧k取top-k；学生按Frozen实际逐帧k取top-k。它们依赖另一系统已知输出数量，是POST-HOC DIAGNOSTIC，不是可部署TTA或GT阈值优化。')
    line();line('| 配对差值 | ΔHOTA | 视频级95%区间 | 提升/下降/持平视频 |')
    line('|---|---:|---|---|')
    for name,v in comparison.items():
        lo,hi=v['CI95'];line(f"| {name} | {v['delta_HOTA']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {v['improved_videos']}/{v['declined_videos']}/{25-v['improved_videos']-v['declined_videos']} |")
    line();line('完整视频作为聚类单位，2000次配对bootstrap/seed0，使用TrackEval未舍入HOTA充分统计重新合并。不是几十万独立帧，也不是训练种子复核或盲测显著性。')
    line('学生相对Frozen的数量减少与收益/损失不是线性可加的因果比例；不能计算“多少百分比提升来自剪枝”。本轮没有触发预注册的seed0/42和E4–E7扩展。')
    line();line('## 4. 排序、文本敏感性与未标注帧')
    line();line('| 范围 | Frozen AUROC / AP | 学生 AUROC / AP |')
    line('|---|---|---|')
    for scope in ('all425_official','GT_present_frame_queries','direct_teacher_pairs','non_teacher_anchor_frames','non_teacher_anchor_frames_GT_present'):
        a,b=ranking['FROZEN'][scope],ranking['MLLM_DISTILL'][scope]
        line(f"| {scope} | {a['AUROC']:.6f} / {a['AP']:.6f} | {b['AUROC']:.6f} / {b['AP']:.6f} |")
    line();line('这些跨帧/表达池化分数受视频难度和query存在性影响，不等同同帧排序。下表一对一R@K包括没有MOT候选的GT帧，按GT帧表达宏平均：')
    line();line('| 方法 | Recall@1 | Recall@3 | Recall@5 |')
    line('|---|---:|---:|---:|')
    for method in ('FROZEN','SOURCE_REHEARSAL','MLLM_DISTILL'):
        r=ranking[method]['GT_recall_at_K_macro_all_GT_frames'];line('| '+method+' | '+' | '.join(number(r[str(k)]) for k in (1,3,5))+' |')
    line();line('另一个旧版“top1”指标指至少命中一个正候选的hit rate，并非按GT数量归一化的一对一召回，不能与此表混用。')
    line('复用既有真实文本互换/视觉偏置评测器：视觉对照为同候选17表达分数的均值，不是独立视觉模型；其它真实表达未必为负，另报GT不相交及非空不相交对照。')
    line();line('| 方法 | 正确文本top1 hit | 随机hit | 视觉偏置hit | 其它文本hit | 混合正负同帧AUROC |')
    line('|---|---:|---:|---:|---:|---:|')
    for method,t in text['methods'].items():line('| '+method+' | '+' | '.join(number(t[k]) for k in ('correct_top1','random_top1','visual_top1','all_other_top1','mixed_source_AUROC'))+' |')
    delta=text['comparisons']['MLLM_DISTILL_vs_FROZEN']['mixed_source_AUROC_minus_mixed_visual_AUROC']
    line();line(f"同帧query相对视觉偏置AUROC优势的增量{delta['delta']:+.6f}，25视频95%区间{delta['video_cluster_CI95']}，没有稳定独立语义排序提升证据。")
    line('非锚点AP/AUROC提升说明并非仅记住教师样本；但非锚点一对一R@1/3/5与标注帧分开保存在`dense_ranking_transfer.json`，不能靠全局AUC替代未标注帧召回。')
    line();line('## 5. Query-Present/官方-Empty与逐行错误归因')
    line();line('| 组别 | 表达数 | Frozen/学生召回 | Frozen/学生FP行 | Frozen/学生非空预测率 |')
    line('|---|---:|---|---|---|')
    for name,v in q.items():
        recall='N/A (无GT)' if not v['gt_observations'] else f"{v['baseline_GT_recall']:.6f} / {v['adapted_GT_recall']:.6f}"
        line(f"| {name} | {v['episodes']} | {recall} | {v['baseline_false_positive_rows']} / {v['adapted_false_positive_rows']} | {v['baseline_nonempty_prediction_rate']:.6f} / {v['adapted_nonempty_prediction_rate']:.6f} |")
    line();line('恢复GT7797、丢失7696，净+101；新增预测FP行16147、删除85655，另有7个共有行因匹配重分配成为FP、0个反向变化，净FP变化−69501。')
    line('这里为独立离线IoU0.5逐帧匹配；官方CLEAR的跨时序匹配TP为10328→10428/FP89376→19876，数值不同是协议定义差别，不是跳样本或改指标。')
    line();line('存在性为整段封存视频的离线统计，不是因果在线规则，不学习新阈值：')
    line();line('| 固定统计量 | Frozen AUROC / AUPRC | 学生 AUROC / AUPRC |')
    line('|---|---|---|')
    for field in ('mean_margin','p90_margin','mean_frame_max_margin','max_margin','positive_fraction'):
        a,b=p['FROZEN'][field],p['MLLM_DISTILL'][field];line(f"| {field} | {a['AUROC']:.6f} / {a['AUPRC']:.6f} | {b['AUROC']:.6f} / {b['AUPRC']:.6f} |")
    line();line('随机presence AUPRC为40/425=0.094118；存在性信号改善不代表最终空目标决策、语义选择或轨迹连续性已解决。')
    line();line('## 6. 教师覆盖、概率校准与学生拟合')
    line();line(f"8381实际有效配对全部以原始软概率训练，0硬POS/NEG，0无效响应；未询问{teacher['unasked_candidate_query_rows']}个候选-表达观测不自动变NEG。")
    line('P1原阈值UNKNOWN只是旧筛选状态，其中有效原始概率仍保留；当前未做硬阈值筛选。以下p>.5只作独立离线诊断，不是训练规则。')
    line(f"教师实际配对AUROC{teacher['AUROC']:.6f}/AP{teacher['AP']:.6f}；980个p>.5的官方匹配精度{teacher['Accepted_POS_Precision']:.6f}，已采样正候选发现召回{teacher['Positive_Discovery_Recall']:.6f}。")
    line('这些精度以官方不完备指代GT定义，不能断言所有未匹配者语义上都是错误。没有明确硬NEG，因此硬NEG误标率为N/A；低于等于.5的有效soft组里有67个官方正候选。')
    line('1102个固定锚点GT观测分解：97无MOT，523有MOT但未采样，66已采样可恢复却被教师打低分，416被高分教师一对一覆盖；四项恰好1102。')
    line('高分教师覆盖固定锚点37.7495%，覆盖全视频GT0.1957%；两种分母不同。全时域MOT可恢复197385/212541，但稀疏采样不能当全时域教师召回。')
    c=teacher['calibration_against_official_labels'];line(f"原软概率对官方标签固定10bins诊断：Brier{c['Brier']:.6f}、ECE10{c['ECE10']:.6f}、log-loss{c['log_loss']:.6f}。未在目标GT上拟合温度/阈值，排序强不意味着原概率校准正确。")
    line('学生教师锚点BCE1.340922→0.345800/MSE0.129064→0.078107，但对980个教师高分配对预测为正仅4.59%→4.80%；不能用整体负例占优的agreement代替正例拟合。')
    line(f"8973步KD梯度均有限非零；52.00%步骤源/KD方向为负，但余弦10/50/90%为{optimization['source_KD_gradient_cosine_quantiles']}，中位数几乎正交，不能据此认定冲突为主因。")
    line('KD/source梯度范数比中位60.21，KD已占优；不应盲目增加lambda。24,576参数LoRA容量不足尚未被实验证明；本轮不以目标GT结果选择新容量/阈值。')
    line();line('## 7. 真实计算成本与模型来源')
    line();line(f"本轮学生训练{cost['training_actual_wall_seconds']:.2f}s/{cost['training_GPU_hours']:.6f}GPUh，24576参数/8973步；四worker推理累计{cost['inference_allocated_worker_seconds']:.2f}s/{cost['inference_GPU_hours']:.6f}GPUh，实际调度墙钟{cost['inference_scheduler_wall_seconds']:.2f}s。")
    line(f"峰值训练{cost['peak_train_allocated_bytes']/2**30:.3f}GiB、推理{cost['peak_inference_allocated_bytes']/2**30:.3f}GiB；调度query-frameFPS{cost['scheduler_query_frame_FPS']:.3f}，活动原视频frameFPS{cost['scheduler_MOT_active_video_frame_FPS']:.3f}，两者相差17表达因子。")
    line('物理GPU1/0/3/4，先审计显存，按进程数/可放显存共享，不停止其他项目；四worker已正常退出。逐workerFPS、GPU UUID、显存与墙钟见`per_worker_compute.csv`。')
    line('本轮新MLLM调用0不等于教师免费：继承P1的8381次整句单帧判断、483.7811秒分摊forward成本，另依赖既有Frozen候选分数库。原P1完整调用/图像输入/失败预算保留于P1报告，缓存命中不算新标签。')
    line('Source-only历史训练0.255297GPUh+推理0.827119GPUh，MeanTeacher参考0.807551+0.496812GPUh；历史Frozen旧逐表达流水线3.726946GPUh/墙钟7729.72s，含原覆盖和等价性工作。')
    line('不同旧/新query-batch实现及共享负载不构成受控速度基准，不能把这些时间比值冒称算法加速。Count-Matched是CPU派生，但总依赖成本包含两套完整模型输出，不是零GPU的独立RMOT系统。')
    line('训练1544.73s和推理1073.20s不包含准备/失败smoke/CPU评估；所有尝试单独保留。官方五组CPU TrackEval总墙钟24.11s，实际成本边界见compute_cost与日志。')
    line();line('V2初始化SHA `71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613`。')
    line('Qwen revision `89644892e4d85e24eaac8bacfd4f463576704203`，BF16/无量化/无adapter/冻结；YesTrack官方V2 LoRA未取得，按用户允许不使用替代权重。')
    line('学生source-val-only选择step8973、CE0.87920851；checkpoint SHA `eaa6b8477cc314efe813fd56a4bd1f70e64615e152b184ba67774548a5f47954`，不是按Dance指标选出的最优。')
    line();line('## 8. 科研判决与恢复')
    line();line('判决：`NO_GO_FIRST_NAIVE_BASELINE_COMPLETE_FAILURE_DIAGNOSIS`。保留代码、可分性正向信号和完整HOTA负结果；不宣称新算法成立，不推出Positive-Support Recovery，不把研究整体判死。')
    line('后续只注册小规模GT-free固定配对拟合/容量诊断，详见`MLLM_DISTILLATION_NEXT_DIAGNOSTIC.md`；本轮不启动新完整训练，不同时改变容量、损失、采样、阈值。')
    line('所有5组各425预测/收据、8973步完整训练trace、教师原始soft标签与输入manifest压缩副本、失败和恢复日志、完整官方CSV/汇总与离线分析均保留并发布。')
    line('23项相关测试通过，compileall与diff空白检查通过；交付回归第一次的临时目录假设错误单独保留，修正测试隔离后全通过，不修改生产预测或放宽验收。')
    line('大型checkpoint、图像、原始SQLite和完整candidate score库留服务器；每个分数文件SHA/path及恢复命令见`results/mllm_distillation/recovery_index.json`和`full/candidate_score_bank_index.json`。')
    line();line('![完整首轮归因](../results/mllm_distillation/figures/first_round_attribution.png)')
    (ROOT/'docs/MLLM_DISTILLATION_RESULTS.md').write_text('\n'.join(lines)+'\n')
    verdict='''# 最终科研判决：V2→Dance 普通冻结MLLM蒸馏首轮

2026-10-29；只针对用户授权的25Val同集开发阶段，不替代原40Train→25Val研究。

- 完整25视频/425表达，统一MOT/TrackEval/阈值0；Naive8.0327 < Frozen8.2180，Source-only7.4352。没有可信完整RMOT提升。
- 分数中存在可学习信息：全量AUROC0.6502→0.7549，非锚点也改善；不应宣称教师无用。
- 真正同帧top1与双向数量匹配没有稳定优势，AssA下降且IDSW增加；全局AP不等于语义排序或轨迹收益。
- 目标存在性信号改善，但大幅剪枝同时恢复7797/丢失7696个GT观测，净+101不足以补偿关联损失。
- 锚点GT有523个因未采样丢失，97个无MOT，66个教师低分，416得到高分；稀疏覆盖与学生正例拟合均有限。
- 梯度有限不代表学对；KD已强，余弦中位近0，不能简单归因冲突或盲加lambda；容量不足还未经消融证实。
- 不增加长训、目标GT调阈值或复杂Ours；先注册固定小样本GT-free拟合/单变量容量诊断。
- 当前阶段的训练、完整评测和失败归因已完成；原40Train预测MOT仍缺，正式Train→Val泛化未完成。Qwen预训练数据重叠未知。

完整数值、成本、置信区间与恢复记录见[首轮报告](MLLM_DISTILLATION_RESULTS.md)。
'''
    (ROOT/'docs/FINAL_SCIENTIFIC_VERDICT.md').write_text(verdict)
    figures(summary,q)
    print(json.dumps(dict(status='FIRST_ROUND_REPORT_COMPLETE_NO_NEW_GPU_EXPERIMENT',HOTA=metrics['MLLM_DISTILL']['HOTA'],scientific_verdict=summary['scientific_verdict'])))


if __name__=='__main__':report()
