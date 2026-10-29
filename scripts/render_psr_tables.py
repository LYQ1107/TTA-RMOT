#!/usr/bin/env python3
"""Render measured PSR results; never supplies or tunes model decisions."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def table(headers,rows):
    def cell(x):
        if x is None:return 'N/A'
        if isinstance(x,float):return f'{x:.4f}'
        return str(x)
    return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',
        *['| '+' | '.join(cell(x) for x in row)+' |' for row in rows],'']


def render(public):
    public=Path(public).resolve();summary=json.loads((public/'summary.json').read_text())
    metrics=summary['official_metrics'];students=summary['student_methods'];methods=['FROZEN','SOURCE_REHEARSAL',*students]
    dense=json.loads((public/'dense_ranking_transfer.json').read_text())['methods']
    support=json.loads((public/'teacher_positive_support.json').read_text())
    lines=['# PSR 全量实测表','',
        '用户授权25Val同集无标签开发研究，非40Train→Val、非盲测、非Source-Free Online TTA。每组完整25视频/425表达。',
        'Fixed MOT为官方Dance DETR+NeuralSORT；目标GT只在预测封存后由独立离线评测读取。','',
        '## 官方完整 RMOT','',
        'DetRe/DetPr是HOTA的多IoU阈值检测召回/精度；不是下表离线IoU≥0.5的匹配计数。','']
    lines+=table(['方法','HOTA','ΔFrozen','DetA','AssA','IDF1','IDSW','DetRe','DetPr','预测行'],[
        [m,metrics[m]['HOTA'],metrics[m]['HOTA']-metrics['FROZEN']['HOTA'],*[metrics[m][k] for k in ('DetA','AssA','IDF1','IDSW','DetRe','DetPr','Dets')]] for m in methods])
    lines+=['## 数量归因（不是可部署方法）','',
        'Frozen@方法数量使用方法逐帧k、原Frozen排序；方法@Frozen数量固定原Frozen逐帧k。两者均不使用GT构建预测。',
        '残差不是因果贡献百分比，亦不能独立于父模型部署。','']
    lines+=table(['方法','原方法HOTA','Frozen@方法数量','方法−同数量Frozen','方法@Frozen数量','其−Frozen'],[
        [m,metrics[m]['HOTA'],metrics['FROZEN_AT_'+m+'_COUNT']['HOTA'],metrics[m]['HOTA']-metrics['FROZEN_AT_'+m+'_COUNT']['HOTA'],
        metrics[m+'_AT_FROZEN_COUNT']['HOTA'],metrics[m+'_AT_FROZEN_COUNT']['HOTA']-metrics['FROZEN']['HOTA']] for m in students])
    lines+=['## 语义排序','',
        '全候选AP/AUROC是分类诊断，不是轨迹关联；R@K为所有GT帧查询的one-to-one GT召回，包含没有可检索MOT候选的GT。',
        '窗口外指没有显式教师/弱标签的时间位置；仍为同视频且可能在轨迹历史上下文出现，不是未见视频/图像盲测。','']
    lines+=table(['方法','全候选AP','AUROC','教师与弱扩展窗口外AP','窗口外AUROC','GT R@1','GT R@3','GT R@5'],[
        [m,dense[m]['all425_official']['AP'],dense[m]['all425_official']['AUROC'],dense[m]['not_direct_or_middle_frames']['AP'],
        dense[m]['not_direct_or_middle_frames']['AUROC'],*[dense[m]['GT_recall_at_K']['all'][str(k)] for k in (1,3,5)]] for m in methods])
    lines+=['没有显式教师或弱扩展标签的时间位置，单独按同帧GT一对一匹配计算：','']
    lines+=table(['方法','窗口外GT R@1','窗口外GT R@3','窗口外GT R@5'],[
        [m,*[dense[m]['GT_recall_at_K']['outside_explicit_window'][str(k)] for k in (1,3,5)]] for m in methods])
    lines+=['## Presence / Empty 与真实恢复','',
        'Query-Present/Empty是官方评测GT的离线episode分组，不是全部现实语义的穷尽存在性标签。',
        'FP与恢复为固定IoU≥0.5逐帧最大匹配诊断；新增FP指新增预测行中的未匹配行，净FP还受旧行匹配重分配影响。','']
    reports={m:json.loads((public/m/'offline_attribution.json').read_text()) for m in methods}
    lines+=table(['方法','Present数','Empty数','Present召回','Present FP','Empty FP','Empty非空率','恢复GT','丢失GT','新增FP行','移除FP行','FP净变化'],[
        [m,*[reports[m][k] for k in ('query_present_episodes','query_empty_episodes','query_present_GT_recall','query_present_FP','query_empty_FP',
        'query_empty_nonempty_prediction_rate','recovered_gt','lost_gt','new_false_positive_rows','removed_false_positive_rows','net_false_positive_change')]] for m in methods])
    lines+=['## 同预算教师发现与局部弱扩展','',
        '各策略8500直接判断。下表Full弱扩展是该缓存可生成的潜在标签；纯Naive/Uniform/Confidence学生不用扩展，普通Temporal使用Naive几何版。',
        'POS资格p≥0.95预先固定；官方匹配精度不等于穷尽语义真值。直接和弱覆盖是不同时间位置的one-to-one GT观测。','']
    lines+=table(['策略','POS数','POS正确','POS精度','比Naive新正确','比Naive丢正确','正确且Frozen拒绝','Full弱数','弱官方错误率','几何弱数','几何错误率'],[
        [p,*[r[k] for k in ('accepted_POS','accepted_POS_correct','accepted_POS_precision','new_correct_candidate_POS_vs_Naive',
        'lost_correct_candidate_POS_vs_Naive','correct_POS_not_source_selected','expanded_weak','expanded_weak_official_error_fraction')],
        r['ordinary_geometry_expansion']['samples'],r['ordinary_geometry_expansion']['official_error_fraction']] for p,r in support.items()])
    lines+=['下表按真实GT ID去重并作一对一分配，不把多个重叠MOT候选重复计为多个被恢复的人物观测。仍是帧观测，不是不同人物数量。','']
    lines+=table(['策略','直接GT观测','比Naive新增GT观测','比Naive丢失GT观测','Full弱GT观测','几何弱GT观测','直接覆盖全部GT比例','弱新增全部GT比例'],[
        [p,r['unique_direct_GT_observations'],r['unique_new_direct_GT_observations_vs_Naive'],r['unique_lost_direct_GT_observations_vs_Naive'],
        r['one_to_one_GT_coverage_counts'].get('expanded_GT_observations',0),r['one_to_one_GT_coverage_counts'].get('geometry_GT_observations',0),
        r['one_to_one_GT_coverage_fraction_of_all'],r['new_weak_GT_coverage_fraction_of_all']] for p,r in support.items()])
    lines+=['## 视频级不确定性','',
        '25完整视频为聚类单位，2000次成对重采样、seed0、官方充分统计量池化。区间不是独立帧显著性，也不是训练随机种子不确定性。','']
    comparisons=summary['comparisons']
    lines+=table(['比较','ΔHOTA','95%区间','提升视频','下降视频'],[
        [name,r['delta_HOTA'],f"[{r['CI95'][0]:+.4f}, {r['CI95'][1]:+.4f}]",r['improved_videos'],r['declined_videos']]
        for name,r in comparisons.items() if name.startswith('PSR_vs_')])
    lines+=['## 实际计算成本','',
        'GPU-hours是占卡进程耗时，不是独占设备利用率；共享GPU负载会影响FPS。每个学生独立运行完整推理，没有跨方法共享backbone计时。',
        '教师等价前向时间包含各方法继承的8500判断成本（跨策略缓存允许复用），不包含模型加载等共同开销；完整实际教师worker耗时见监督审计。','']
    lines+=table(['方法','训练GPUh','推理GPUh','训练秒','推理wall秒','教师判断','等价教师前向秒','可训练参数','更新步','活跃视频帧FPS','查询帧FPS','训练峰值GiB','推理峰值GiB'],[
        [r['method'],r['training_GPU_hours'],r['inference_GPU_hours'],r['actual_training_wall_seconds'],r['actual_inference_scheduler_wall_seconds'],
        r['required_teacher_judgments'],r['required_teacher_amortized_forward_seconds'],r['trainable_parameters'],r['optimizer_updates'],
        r['per_worker_active_video_frame_FPS'],r['per_worker_query_frame_FPS'],r['peak_training_allocated_bytes']/2**30,r['peak_inference_allocated_bytes']/2**30] for r in summary['cost']])
    lines+=['Frozen与Source-only复用封存历史完整结果，当前GPU调用0不等于继承成本为0；历史成本和执行器差异在summary.json的baseline_costs_reused。',
        'FPS仅为固定MOT后的第二阶段指代筛选，非包含检测/跟踪前端的端到端速度；采样复用已有Frozen分数，继承成本另列。',
        'Mean Teacher外部MLLM判断0，不能冒充等MLLM预算语义消融；其容量/源计划/8973步/未标注输入/源val选择匹配。','',
        '## 源域选择的 checkpoint','']
    lines+=table(['方法','选中步','源val CE','SHA256'],[
        [m,r['selected_step'],r['source_val_CE'],r['checkpoint_sha256']] for m,r in summary['selected_checkpoints'].items()])
    lines+=['最新步teacher-fit报告只衡量训练锚点拟合，不能作为GT准确率或改变源val checkpoint选择。',
        '完整分数、逐视频/表达分析、日志、预测压缩包与恢复索引均在同目录；权重、图像、完整原始教师缓存不上传。','']
    (public/'TABLES.md').write_text('\n'.join(lines));print(json.dumps(dict(status='MEASURED_PSR_TABLES_RENDERED',student_methods=len(students))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--public-root',type=Path,default=ROOT/'results/psr/full');a=p.parse_args();render(a.public_root)
