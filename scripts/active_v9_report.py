"""Report only sealed estimates and independent offline GT tables."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.active_v9_inputs import PUBLIC,PAIRS,DATA
from scripts.active_v8_stage0_report import table


def report():
    selection=json.loads((PUBLIC/'selection.json').read_text())
    diagnostic=json.loads((PUBLIC/'D1_D2_diagnostics.json').read_text())
    d3=json.loads((PUBLIC/'inventory_D3.json').read_text())
    rows=selection['rows'];selected=selection['selected'];cost=[];reproductions=[]
    for method in ('E0','E1','E3'):
        for pair in PAIRS:
            path=PUBLIC/f'{method}_{pair}.json'
            if not path.exists():continue
            item=json.loads(path.read_text())
            if method=='E0':
                checks=[r['reproduction'] for r in item['records']]
                reproductions.append(dict(pair=pair,videos=len(checks),
                    parameter_max_error=max(r['parameter_max_error'] for r in checks),
                    probability_max_error=max(r['probability_max_error'] for r in checks),
                    all_output_absence_exact=all(r['output_absence_exact'] for r in checks)))
            cost.append(dict(estimator=method,pair=pair,CPU_process_wall_seconds=item['wall_seconds'],
                videos=len(item['records']),queries=sum(len(r['queries']) for r in item['records']),
                GT_open_attempts=item['GT_open_attempts'],GPU_calls=item['GPU_calls'],teacher_calls=item['teacher_calls']))
    from scripts.active_v9_inputs import write_csv
    write_csv(PUBLIC/'costs.csv',cost)
    text='# v9存在性离线验证\n\n'
    text+=f"选择：{selected or '未选择'}；状态：{selection['status']}。E2={d3['E2_status']}。\n\n"
    if selected:
        text+='E1优先且六方向全部离线门槛通过，才具备后续v9集成与合成恢复测试的条件；尚不能将离线反事实冒充完整RMOT结果。\n\n'
    elif selection['status']=='PENDING':
        text+='E1尚未完成六方向，不能选择或提前判为失败；E2已按GT-free适用性规则排除。\n\n'
    else:
        text+='E1未通过时不以E0/E3替代为新方法；E2因SportsMOT文本复用不分离而不可用。不调整阈值或为单一数据集开先验例外。\n\n'
    text+='本轮门槛使用官方val/test GT，是在val上进行开发，不是盲测。所有估计/生成进程CPU-only、GT打开0；GT只由估计封存后的独立诊断进程读取。原v8阶段3队列继续，阶段4暂停。\n\n'
    text+='## 1. 固定门槛与全部结果\n\n'
    text+=table(rows,['estimator','pair','output_false_absent','output_correct_empty','rho_hat','rho_GT','rho_error',
        'false_absent_gate','correct_empty_gate','rho_gate','ALL_GATES_PASS'])+'\n\n'
    text+='B2两类门槛都用最终输出拒绝（EVIDENCE+PRIOR_BUDGET_EXHAUSTED）；视频末按新估计器重新决定预算耗尽拒绝集合。数值门槛、选择优先级不变。\n\n'
    text+='## 2. 估计器终值、Laplace与截断诊断\n\n'
    text+=table(rows,['estimator','pair','pi_or_mu','tau','rho_laplace_std','uncapped_output_disagreements','uncapped_evidence_disagreements'])+'\n\n'
    text+='E1的mu是logit-normal随机效应的中心，不等于边际平均匹配率；每视频结果另存marginal_pi_strata。Laplace使用不截断光滑目标的全5参数Hessian；奇异/非正定标不可识别，边界处局部高斯近似的限制保留在每个收据。B1不截断判定仅作诊断，不影响选择。\n\n'
    text+='## 3. 先验拒绝与证据拒绝\n\n'
    text+=table(rows,['estimator','pair','evidence_false_absent','evidence_correct_empty','low_tentative','PRIOR_BUDGET_EXHAUSTED',
        'prior_budget_GT_present','prior_budget_correct_empty','correct_empty_majority_from_prior_budget'])+'\n\n'
    text+='最后一列仅标注正确标空是否过半来自预算耗尽先验拒绝，不是新增验收门槛。固定原问题/预算；新估计器即使提前拒绝也继续摄入原流后续回答，不模拟重新选题。\n\n'
    text+='## 4. D1：句子间匹配率与过度离散\n\n'
    text+=table(diagnostic['D1'],['pair','stratum','queries','pi_mean','pooled_pi','variance','homogeneous_binomial_variance','overdispersion','quantiles'])+'\n\n'
    text+='GT标签是每帧IoU>=.5一对一匹配后按轨迹>=50%观测为正；只纳入n>=3。方差为句子率的ddof=1方差，二项参照取共用pooled_pi并保留各句实际分母。共享视频/轨迹/表达具有相关性，过度离散是描述性比较，不作为独立样本显著性。\n\n'
    text+='## 5. D2：教师对正负轨迹的回答\n\n'
    text+=table(diagnostic['D2'],list(diagnostic['D2'][0]))+'\n\n'
    text+='逐句记录见D2_query_teacher_answers.csv，特别保留原C2/C5错误拒绝句。正/负轨迹Yes率按实际回答计数；唯一问过轨迹数按(q,track)去重；全No要求至少有一次回答。原生轨迹多数GT，不冒充单帧裁剪的教师准确率。\n\n'
    text+='## 6. D3：查询复用与E2适用边界\n\n'
    text+=table(d3['D3'],['target','videos','queries','dataset_mean_reuse','query_weighted_reuse','video_reuse_quantiles'])+'\n\n'
    text+='SportsMOT均值.928046>=.5，触发预注册NOT_APPLICABLE；阈值.8已在D3前公开，不调阈值。不能用已知目标存在比例代替这个GT-free协议代理。\n\n'
    text+='## 7. 隔离、复现、成本与恢复\n\n'
    text+=table(reproductions,['pair','videos','parameter_max_error','probability_max_error','all_output_absence_exact'])+'\n\n'
    text+=table(cost,['estimator','pair','CPU_process_wall_seconds','videos','queries','GT_open_attempts','GPU_calls','teacher_calls'])+'\n\n'
    text+='输入路径/SHA见inventory_D3.json和inputs_seal.json。每个E0/E1/E3方向的完整逐视频参数、逐句判定与优化/Laplace收据见同目录E*_C*.json；服务器封存及可恢复CPU checkpoint根目录：`'+str(DATA)+'`。没有新RMOT预测，也没有新TrackEval HOTA；这些是固定已观测回答条件下的存在性反事实。\n\n'
    text+='弱排名固定为v8实际逐轮温度/偏置作用于封存StageA logits；不用视频末排名前视，不按反事实重新拟合快分支。E3仅复现旧估计公式在v8固定问题流上的效果，不冒充历史v7闭环。E0收据逐视频核对原v8参数及拒绝集合。\n\n'
    if selection['status']=='STOP_NO_ESTIMATOR_PASSED':
        text+='## 停止\n\nE1未通过全部固定门槛，E2不可用；本轮停止v9方法实施，不生成live v9.py，不运行MINUS_SLOW_v9，不开启阶段4或任何新方法。原v8阶段3不受此停止影响。\n'
    (PUBLIC/'REPORT.md').write_text(text)
    return selection


if __name__=='__main__':print(json.dumps({k:v for k,v in report().items() if k!='rows'}),flush=True)
