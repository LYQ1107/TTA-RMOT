#!/usr/bin/env python3
"""Render factual Chinese delivery pages from complete, independently evaluated data."""
import argparse
import json
import shutil
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_uda.training import METHODS
from scripts.pf_diag_common import official_metrics,write_csv
from flex_tta.runtime.pf_receipts import atomic_json


def render_interim(root,output):
    root,output=Path(root).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    names=[]
    for method in ('FROZEN',)+METHODS:
        if not(root/method/'results/pedestrian_summary.txt').exists():continue
        audit=json.loads((root/method/'full_audit.json').read_text())
        if audit['status']!='PASS' or audit['valid_episodes']!=425:raise ValueError('no partial-benchmark official table')
        names.append(method)
        cm=method+'_COUNT_MATCHED_FROZEN'
        if (root/cm/'results/pedestrian_summary.txt').exists():names.append(cm)
    result={method:official_metrics(root/method/'results') for method in names}
    for method in names:
        dest=output/method;dest.mkdir(exist_ok=True)
        for name in ('pedestrian_summary.txt','pedestrian_detailed.csv'):shutil.copy2(root/method/'results'/name,dest/name)
        shutil.copy2(root/method/'trackeval.log',dest/'trackeval_stdout.txt')
    atomic_json(output/'summary.json',dict(status='INTERIM_NOT_ALL_FOUR_METHODS',protocol='TRANSDUCTIVE_VAL_DIAGNOSTIC',
        videos_per_method=25,episodes_per_method=425,checkpoint_selection='SOURCE_VAL_ONLY_BEFORE_TARGET_EVALUATION',
        complete_methods=[m for m in METHODS if m in result],metrics=result,
        countmatched_protocol='POST_HOC_DIAGNOSTIC_NOT_DEPLOYABLE; dependent on parent model counts'))
    write_csv(output/'summary.csv',[dict(method=m,HOTA=r['HOTA'],delta_HOTA_vs_Frozen=r['HOTA']-result['FROZEN']['HOTA'],
        **{k:r[k] for k in ('DetA','AssA','DetRe','IDF1','IDSW','CLR_TP','CLR_FP','Dets')}) for m,r in result.items()])
    print(json.dumps(result,indent=2))


def render(public):
    public=Path(public).resolve();summary=json.loads((public/'summary.json').read_text())
    if summary['status']!='COMPLETE_TRANSDUCTIVE_VAL_DIAGNOSTIC':raise ValueError('no report from partial results')
    reports={m:json.loads((public/m/'metrics.json').read_text()) for m in METHODS};metrics=summary['metrics']
    text_signal=json.loads((public/'text_signal_offline.json').read_text())
    pseudo=json.loads((public/'pseudo_label_quality_offline.json').read_text())
    frozen_presence=json.loads((ROOT/'results/diagnostics/pf_recoverability_20261025/query_information/summary.json').read_text())['presence']
    lines=['# FlexHook V2→Refer-Dance：25Val同集适应完整诊断','',
        '日期：2026-10-26。全部25视频、425表达；目标适应和评测是同一25Val。',
        '**TRANSDUCTIVE_VAL_DIAGNOSTIC，不是40Train→25Val UDA、不是独立测试泛化、不是Source-Free Online TTA。**','',
        '本轮完成用户“先25Val看效果”的阶段。原40Train图像/表达已齐，但同配置Train MOT仍缺，原正式Train→Val目标尚未完成。',
        '没有用目标GT选择阈值、checkpoint或训练参数；源开发集固定64窗口CE选模型，全部方法包括step0。','',
        '## 完整官方结果','',
        '| 方法 | HOTA | ΔFrozen | DetA | AssA | DetRe | IDF1 | IDSW | 输出行 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method in ('FROZEN',)+METHODS:
        r=metrics[method]
        lines.append(f"| {method} | {r['HOTA']:.4f} | {r['HOTA']-metrics['FROZEN']['HOTA']:+.4f} | {r['DetA']:.4f} | {r['AssA']:.4f} | {r['DetRe']:.4f} | {r['IDF1']:.4f} | {int(r['IDSW'])} | {int(r['Dets'])} |")
    lines+=['','Frozen仍为完整211085条候选的8.2180；旧8.2060遗漏692末观测，仅保留历史，不混用协议。',
        'SOURCE_REHEARSAL不使用目标loss；B与A源loss相同。C/D另外加入source-presence项，其相对A差异不能全部归于目标约束；本轮未增加source-presence-only消融。',
        '固定前端是Dance专用官方DETR+NeuralSORT；本研究适应第二阶段，不宣称整个检测/跟踪系统无目标域监督。','',
        '## 数量匹配与语言选择','',
        '| 方法 | 原模型AUROC→适应AUROC | 恢复GT / 丢失GT | 新增未匹配预测行 | FP净变化 | Count-Matched HOTA | 适应−Count-Matched |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m,r in reports.items():
        a=r['attribution'];cm=r['countmatched_metrics']
        lines.append(f"| {m} | {a['source_auroc']:.6f} → {a['adapted_auroc']:.6f} | {a['recovered_gt']} / {a['lost_gt']} | {a['new_false_positive_rows']} | {a['net_false_positive_change']:+d} | {cm['HOTA']:.4f} | {r['count_conditioned_delta']:+.4f} |")
    lines+=['','Count-Matched按实际适应模型逐帧输出数量k，在Frozen中取top-k，不使用GT构造预测。',
        '这是依赖适应结果的POST-HOC DIAGNOSTIC，不可独立部署；不能把HOTA差值比例称为因果贡献百分比。',
        'GT恢复/FP行是离线IoU≥0.5最大匹配诊断；新增未匹配预测行与已有行匹配重分配造成的FP净变化不同，官方CLEAR/HOTA仍为主。',
        'Recall@1/3/5见`results/uda/candidate_ranking.csv`；不是只看loss或非零梯度就称语义恢复。','',
        '## Query-Present / Query-Empty','',
        '官方协议40条有GT、385条空GT；仅用于离线分组，未作为训练先验。全部425条均进入主结果。','',
        '| 方法 | Present召回 | Present FP | Empty FP | Empty非空输出率 | 存在性AUROC / AP |',
        '|---|---:|---:|---:|---:|---:|']
    for m,r in reports.items():
        a=r['attribution'];p=a['query_presence']['query_present'];e=a['query_presence']['query_empty'];presence=r['presence']['mean_frame_max_margin']
        lines.append(f"| {m} | {100*p['adapted_GT_recall']:.4f}% | {p['adapted_false_positive_rows']} | {e['adapted_false_positive_rows']} | {100*e['adapted_nonempty_prediction_rate']:.4f}% | {presence['AUROC']:.6f} / {presence['AUPRC']:.6f} |")
    lines+=['','存在性统计固定使用完整episode的平均逐帧max margin，属于离线诊断，不能冒充因果在线presence规则；没有拟合GT阈值。',
        f"同一统计量的Frozen AUROC/AP为{frozen_presence['mean_frame_max_margin']['AUROC']:.6f}/{frozen_presence['mean_frame_max_margin']['AUPRC']:.6f}。",
        f"历史Frozen {frozen_presence['max_margin']['AUROC']:.6f}/{frozen_presence['max_margin']['AUPRC']:.6f}使用整段max margin，不与平均逐帧max混比。",
        'Query-Empty无GT，单独HOTA不用于声称改善；重点检查FP与非空率。','',
        '## 视频级不确定性','',
        '成对bootstrap以25个完整视频为聚类，2000次、bootstrap seed0，使用官方TP加权HOTA充分统计；',
        '不把425表达或几十万相关帧当独立样本。该区间仍是同Val描述性不确定性，不是独立盲测显著性或训练seed不确定性。','',
        '| 比较 | ΔHOTA | 25视频95%区间 | 提升/下降/持平视频 |','|---|---:|---:|---:|']
    for name,r in summary['comparisons'].items():
        lo,hi=r['video_cluster_CI95'];lines.append(f"| {name} | {r['delta_HOTA']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {r['improved_videos']} / {r['declined_videos']} / {r['tied_videos']} |")
    lines+=['','逐视频及移除一个视频后的影响见`results/uda/full/summary.json`；HOTA非线性，视频影响不能机械相加。','',
        '## 成本与模型选择','',
        '| 方法 | source选中step | source-val CE | 更新数 | 训练/推理 GPU-process-h | 完成进程训练峰值GiB | 推理query-frame FPS |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m,r in reports.items():
        c=r['cost'];k=r['checkpoint']
        lines.append(f"| {m} | {k['selected_step']} | {k['source_val_CE']:.8f} | {c['optimizer_updates']} | {c['training_GPU_hours']:.4f} / {c['inference_GPU_hours']:.4f} | {c['peak_training_allocated_bytes']/(1<<30):.3f} | {c['per_worker_query_frame_FPS']:.3f} |")
    lines+=['','每组8973更新、24576个LoRA参数，rank4/最后两层PCD Q/V/lr5e-5/seed17，完整源17视频与目标25视频查询覆盖。',
        '覆盖指视频/表达/源轨迹，不是逐帧优化：源393个不同anchor；目标每视频3个预定anchor，共75个不同anchor、1275个query窗口，重复配对源周期；每个anchor还使用过去4次轨迹观测。完整评测覆盖全部活动帧。',
        f"实际训练scheduler wall {summary['actual_scheduler_training_wall_seconds']:.2f}s；推理scheduler跨度 {summary['actual_scheduler_inference_wall_seconds']:.2f}s；",
        f"重叠流水线实际wall {summary['actual_pipeline_wall_seconds']:.2f}s。训练与先行评测交错，不能将二者wall简单相加。",
        'GPU-process-hours是共享卡worker elapsed的和，不是显卡利用率仪表测量；恢复期曾短暂同卡，不冒充独占四卡。',
        '训练峰值指成功/恢复进程记录的allocated峰值；早期失败进程未单独记录的峰值不伪造。失败尾段额外耗时按最后日志恢复记录保留下界。',
        'Frozen本轮是CPU精确重放历史GPU预测，不能把6秒CPU重放当独立GPU RMOT系统速度；Count-Matched总成本包含父方法。','',
        '## 机制与失败检查','',
        '四组均真实训练、更新LoRA，非骨干完整重训。A是源监督；B为EMA/强弱特征一致性和保守伪标签；',
        'C为源presence及合法文本等价/互斥关系；D再组合源锚点与满足语义条件的短时记忆。',
        '**C/D的目标MIL/presence没有独立正负证据，目标PRESENT/ABSENT覆盖均为0；UNKNOWN跳过，不能宣称已经复现有效目标MIL。**',
        '含dancing+服饰的整句为MIXED，不能硬把整句语言匹配跨所有ID帧固定。纯身份一致性不是服饰匹配真值。',
        '没有外部人物语义教师，也没有原版QCA复现。源码参考固定版本见`UDA_REFERENCE_IMPLEMENTATION_MAP.md`。',
        '原C-Hook的N=1跨表达NaN常量不进入LoRA目标；原v1错误smoke已撤销。AMP溢出采用同样本RNG恢复和scale回退，未跳样本。',
        '全部完成前不以目标GT审查伪标签；完成后的独立结果见`results/uda/full/pseudo_label_quality_offline.json`。','',
        '## 伪标签的独立离线质量','',
        '| 方法 | POS对应GT / POS总暴露 | POS精度 | Query-Empty中的POS | NEG误删GT暴露 | UNKNOWN中的GT暴露 |',
        '|---|---:|---:|---:|---:|---:|']
    for m,p in pseudo['methods'].items():
        e=p['exposures'];positive=e['POS_GT_MATCH']+e['POS_NO_GT_MATCH']
        lines.append(f"| {m} | {e['POS_GT_MATCH']} / {positive} | {100*p['pseudo_POS_precision']:.4f}% | {p['groups']['Query-Empty']['POS_NO_GT_MATCH']} | {e['NEG_GT_MATCH']} | {e['UNKNOWN_GT_MATCH']} |")
    lines+=['','这是全部历史训练证据的重复暴露，不是独立人物数或独立样本；候选按IoU≥0.5对应当前表达GT，非一对一跟踪precision。',
        'Source/EMA一致和同一物理ID稳定并没有保证语言正确；本次POS主要落在空查询，不能把这种共识当作独立语义教师。',
        'UNKNOWN中确有正确候选，但没有足够无标签依据把它们转为正例；继续保持UNKNOWN，不根据这份GT审计补标签。','',
        '## 冻结与适应模型的文本敏感性','',
        '| 方法 | 正确查询Top1命中 | 随机候选Top1 | 视觉偏置Top1 | 互换文本Top1 | 混合候选帧AUROC | 混合帧视觉AUROC |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m,r in text_signal['methods'].items():
        lines.append(f"| {m} | {r['correct_top1']:.6f} | {r['random_top1']:.6f} | {r['visual_top1']:.6f} | {r['all_other_top1']:.6f} | {r['mixed_source_AUROC']:.6f} | {r['mixed_visual_AUROC']:.6f} |")
    tc=text_signal['comparisons']['MEAN_TEACHER_UDA_vs_FROZEN']['mixed_source_AUROC_minus_mixed_visual_AUROC']
    lines+=['','Top1是至少命中一个正确候选的any-hit，不等于全部正确目标Recall@1；纯视觉控制为同一候选跨17真实查询的均值，互换文本在相同候选库中比较。',
        '完整分析含42657个有GT帧-查询、其中26352个正负候选混合组，以及GT不重叠文本互换控制；不把这些相关组作为独立统计样本。',
        f"Teacher相对Frozen的混合帧(查询AUROC−视觉AUROC)变化为{tc['delta']:+.6f}，25视频描述性95%区间[{tc['video_cluster_CI95'][0]:+.6f}, {tc['video_cluster_CI95'][1]:+.6f}]。",
        '所有适应模型的正确查询Top1和候选Recall@1/3/5点估计均低于Frozen；没有观察到一致的语言排序恢复。',
        '本轮没有重做内部特征探针。此前冻结PCD诊断只证明有限可读出信息，不能把带GT探针当作无标签可恢复性或本轮TTA收益。','',
        '## 科研判断与停止边界','',
        '工程方面：四组各8973次真实有限更新、完整25/425只读推理、四组数量匹配和官方评测均完成；旧六组结果和失败现场保留。',
        f"科研方面：Mean Teacher相对Frozen仅{metrics['MEAN_TEACHER_UDA']['HOTA']-metrics['FROZEN']['HOTA']:+.4f} HOTA，相对数量匹配仅{reports['MEAN_TEACHER_UDA']['count_conditioned_delta']:+.4f}；视频区间包含0，尚无可靠正向语义结论。",
        'Source Rehearsal和Weak的AssA提高伴随大幅召回损失；Combined没有超过Frozen或较好单模块，不能作为成功最终方法。',
        '主因证据更支持查询存在性/校准与查询条件排序联合失效，而非单纯身份不稳定；目标伪正例不可靠，合法presence监督与静态语义时间监督覆盖为0。',
        '这不证明所有UDA或所有服饰语义不可恢复：本轮只是一轮固定查询/轨迹覆盖、稀疏anchor预算，C/D还有source-presence混杂，且没有有效目标MIL正负监督。',
        '目前没有足够证据支持直接加长同一EMA/MIL训练；如未来继续，需要先明确独立于源模型错误共识的语义依据，并另获授权，而不是依据Val GT调阈值或叠加loss。',
        '不根据此Val结果再改阈值、轮数或挑checkpoint。本轮结束后不自动开始第五个完整方法、外部教师或40Train训练。',
        '原始40Train→25Val正式UDA仍未完成：缺少同算法/同配置40Train预测MOT，且当前用户要求先不考虑Train。本阶段完成不等于原始完整goal完成。',
        '完整准备/smoke/失败与恢复/资源调度/推理/官方评测文本日志归档于`results/uda/full/logs/runtime_logs.tar.gz`，逐文件SHA索引同目录。',
        '原始checkpoint/分数/全部恢复路径见`results/uda/full/full_recovery_index.json`；完整425预测小压缩包不含GT/图像/模型权重。']
    (ROOT/'docs/UDA_FINAL_SCIENTIFIC_REPORT.md').write_text('\n'.join(lines)+'\n')
    target=['# 完整目标评测与复现','',
        '协议与全部数值见`UDA_FINAL_SCIENTIFIC_REPORT.md`，机器表在`results/uda/*.csv`。',
        '预测：`scripts/evaluate_uda_full.py`；逐视频FP32 query-parity、近零回退，0optimizer/EMA。',
        '封存：`scripts/finalize_uda_evaluation.py`，每组25/425、固定3588445候选-query行。',
        '官方评测：`scripts/evaluate_flexhook_dance_tta.py`；使用固定FlexHook TrackEval fork，HOTA/CLEAR/Identity，非分片均值。',
        '同一gt_template、视频编号及原始1-based帧，无偏移重写；mark非0及class1过滤与既有独立协议审计一致。',
        'Count-Matched构造：`scripts/diag_uda_countmatched.py`，不读取GT；随后独立官方评测。',
        '离线归因：`scripts/analyze_uda_full.py`复用旧归因器；伪标签质量另由`analyze_uda_pseudo_offline.py`读取GT。',
        '训练与推理从未打开目标GT。这里只在完整预测封存之后附加GT并评测，不把离线分组回流模型。','',
        '所有输出保持官方MOT框/ID/置信度原行不变；候选末观测692行已补齐，不再与旧8.206候选覆盖混用。',
        '每个完成episode有prediction/margin SHA256和完整帧receipt；大分数流与权重保留data2/server，索引在full_recovery_index.json。']
    (ROOT/'docs/UDA_TARGET_EVALUATION.md').write_text('\n'.join(target)+'\n')
    print(json.dumps(dict(status='REPORT_RENDERED',methods=list(reports),raw_root=str(public)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--public-root',type=Path,default=ROOT/'results/uda/full')
    p.add_argument('--interim-evaluation-root',type=Path);a=p.parse_args()
    if a.interim_evaluation_root:render_interim(a.interim_evaluation_root,a.public_root)
    else:render(a.public_root)
