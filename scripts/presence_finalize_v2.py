"""Conclude failed Stage A with corrected Sports labels; never starts B/C."""
import csv
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_uda.presence.runtime import RUN,PUBLIC,IN_DOMAIN
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.diag_attrib_v2_common import write_csv
from scripts.active_plan import jobs
from scripts.presence_corrected_eval import destination
from scripts.presence_correct_sports_labels import HEAD_ROOT


def main():
    selected_gate=json.loads((PUBLIC / 'stage_A_selected_gate_v2.json').read_text())
    if selected_gate['status'] != 'FAIL_SELECTED_SOURCE_HEADS_STOP_BEFORE_BC':
        raise RuntimeError('This failure report must not replace a passed selected-head continuation')
    with (PUBLIC / 'stage_A.csv').open() as stream:old=list(csv.DictReader(stream))
    rows=[];artifacts=[]
    for previous in old:
        row=dict(previous)
        if row['model']=='LaMOT-best' and row['method'].startswith('PRES_TOPK'):
            folder=destination(row['target'],'default') / row['method']
            row.update(json.loads((folder / 'offline_summary.json').read_text()),source_label_version='EXPRESSION_ID_V2')
        else:row['source_label_version']='UNAFFECTED' if row['method'].startswith('PRES_TOPK') else 'FROZEN_OR_ORACLE_REFERENCE'
        rows.append(row)
        if row['method'].startswith('PRES_TOPK'):
            folder=Path(row['path'])
            artifacts.append(dict(target=row['target'],model=row['model'],method=row['method'],path=str(folder),
                prediction_seal_sha256=file_sha(folder / 'predictions_seal.json'),
                official_completion_sha256=file_sha(folder / 'official_completion.json'),
                offline_summary_sha256=file_sha(folder / 'offline_summary.json')))
    if len(rows)!=36:raise ValueError('Stage A must report all9 model-target cells and all4 configurations')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    write_csv(PUBLIC / 'stage_A_v2.csv',[{k:r.get(k) for k in fields} for r in rows])
    own=[];table=['|目标|源模型|Native|PRES_TOPK|Smooth|ORACLE数量|','|---|---|---:|---:|---:|---:|']
    for target,own_model in IN_DOMAIN.items():
        for model in ('V2-best','Dance-best','LaMOT-best'):
            cell={r['method']:r for r in rows if r['target']==target and r['model']==model}
            native,pres,smooth,oracle=[float(cell[k]['HOTA']) for k in ('NATIVE','PRES_TOPK','PRES_TOPK_SMOOTH','R_SxK_ORACLE')]
            table.append(f'|{target}|{model}|{native:.3f}|{pres:.3f}|{smooth:.3f}|{oracle:.3f}|')
            if model==own_model:own.append(dict(target=target,model=model,native_HOTA=native,
                HOTA=pres,delta_HOTA=pres-native,passed=pres>=native-1))
    write_csv(PUBLIC / 'stage_A_default_same_domain_v2.csv',own)
    atomic_json(PUBLIC / 'stage_A_default_gate_v2.json',dict(
        status='PASS_ALL_THREE_SAME_DOMAIN' if all(r['passed'] for r in own) else 'FAIL_SAME_DOMAIN',
        metrics=own,source_label_correction=True,legacy_gate_unchanged=True,published_commit=None))
    status=[dict(job,status='NOT_RUN_STAGE_A_GATE_FAILED',reason='source-loss-selected own-source heads did not retain native HOTA within1') for job in jobs()]
    write_csv(PUBLIC / 'matrix_status.csv',status)
    atomic_json(PUBLIC / 'BC_cost_estimate.json',dict(status='NOT_MEASURED_STAGE_A_GATE_FAILED',
        full_matrix_jobs=len(status),GPU_hour_limit=40,estimated_GPU_hours=None,
        no_real_C1_C3_pilot=True,teacher_calls=0,target_LoRA_optimizer_updates=0,
        note='Do not fabricate a zero-cost or measured40h estimate without actual teacher/LoRA pilots'))
    atomic_json(PUBLIC / 'artifacts_stage_A_v2.json',dict(files=artifacts,
        old_artifacts_preserved=str(PUBLIC / 'artifacts_stage_A.json'),large_files_not_in_git=True))
    selection=json.loads((PUBLIC / 'source_lr_trial_selection_v2.json').read_text())
    trial_table=['|源/同域|配方|源验证loss|HOTA|原生|Δ|源loss选择|','|---|---|---:|---:|---:|---:|---|']
    trial_rows=[]
    for target,model in IN_DOMAIN.items():
        for variant in ('default','source_lr1e5'):
            if model=='LaMOT-best':folder=destination(target,variant) / 'PRES_TOPK'
            else:folder=RUN / ('stage_A' if variant=='default' else 'stage_A_source_lr1e5') / target / model / 'PRES_TOPK'
            summary=json.loads((folder / 'offline_summary.json').read_text())
            choice=selection['models'][model]
            loss=next(c['loss'] for c in choice['candidates'] if c['variant']==variant)
            native=next(r['native_HOTA'] for r in own if r['target']==target)
            selected=variant==choice['selected']
            trial_rows.append(dict(summary,source_variant=variant,source_validation_loss=loss,
                selected_by_source_loss=selected,native_HOTA=native,delta_HOTA=summary['HOTA']-native))
            trial_table.append(f"|{target}|{variant}|{loss:.6f}|{summary['HOTA']:.3f}|{native:.3f}|{summary['HOTA']-native:+.3f}|{'是' if selected else '否'}|")
    fields=list(dict.fromkeys(k for r in trial_rows for k in r))
    write_csv(PUBLIC / 'source_lr_trial_v2.csv',[{k:r.get(k) for k in fields} for r in trial_rows])
    default_cost=list(csv.DictReader((PUBLIC / 'stage_A_costs.csv').open()))
    trial_cost=list(csv.DictReader((PUBLIC / 'source_lr_trial_costs.csv').open()))
    correction_cost=list(csv.DictReader((PUBLIC / 'source_label_fix_v2_costs.csv').open()))
    # Default training appears in both the A and LR tables: count it once.
    additional=[r for r in trial_cost if r['variant']=='source_lr1e5']+correction_cost
    known=sum(float(r['GPU_hours']) for r in default_cost)+sum(float(r['GPU_hours']) for r in additional)
    atomic_json(PUBLIC / 'cost_accounting_v2.json',dict(known_successful_process_hours=known,
        includes_all9_target_exports_and3_source_exports=True,
        includes_6_initial_and2_corrected_source_head_runs=True,
        failed_and_initial_model_load_overhead='not assumed zero; logs retain attempts, not fully timed',
        is_GPU_utilization_integral=False,teacher_new_forward_GPU_hours=0,target_LoRA_GPU_hours=0,
        default_costs=str(PUBLIC / 'stage_A_costs.csv'),trial_costs=str(PUBLIC / 'source_lr_trial_costs.csv'),
        correction_costs=str(PUBLIC / 'source_label_fix_v2_costs.csv')))
    progress=PUBLIC / 'REPORT_PROGRESS_v1.md'
    if not progress.exists():shutil.copy2(PUBLIC / 'REPORT.md',progress)
    report='''# Presence–Identity + Active Verification + Fast/Slow TTA：阶段A未通过

## 结论与执行边界

阶段A实现、完整缓存、默认源头训练、一次固定源学习率缓解对照和官方评测已完成；修正Sports源标签后，按源验证loss选择的头仍未通过“各自同域PRES_TOPK HOTA≥原生−1”门槛。**完整方法科研目标未完成，B/C没有启动。** 不能以18项功能检查、梯度或loss下降宣布科研成功。

本轮新增教师调用=0，目标LoRA更新=0。108项B/C矩阵均为NOT_RUN，而不是性能为0；真实pilot、调用量–HOTA曲线、消融、DS的GT对照及B/C视频bootstrap尚未产生。40GPU小时关卡没有绕过，也未虚构成本估算。继续需要解决给定PresenceHead的源域保持能力，不自行增加超参数搜索或替换方法。

项目为`LYQ1107/TTA-RMOT`，根`/data2/user/FlexHook-TTA`，分支`method/presence-active-20261102`，起点`9b04b070eddc9e80c0c9db1b582a02780689aa16`；不是LocateMOT/GroundJEV任务。

## 阶段A主结果：默认配方，Sports采用修正标签

'''+ '\n'.join(table)+'''

九套完整候选、两种Presence解码、原生及数量ORACLE均全部报告：`stage_A_v2.csv`（36行，含DetA/AssA/IDF1/IDSW、输出行、Query-Empty、存在性/数量误差与原句分组）。ORACLE仅为历史GT诊断，不是可部署方法；没有据跨域HOTA选头。Sports为v4完整466表达，另保留相同v4内容的官方418序列敏感性。

`stage_A.csv`/`source_lr_trial.csv`是保留的v1历史结果，其中LaMOT PresenceHead受源标签合并bug影响，**不作为主结果**。主表使用独立v2预测，没有覆盖历史。

## 唯一学习率缓解对照与固定选择规则

'''+ '\n'.join(trial_table)+'''

替代lr1e-5先公开再运行，其余网络/损失/增强/seed0/批大小/早停不变。只按各自源官方验证 BCE+0.5条件count CE 选头，HOTA前封存选择；Sports标签修复只重做这两个既有配方，不是新学习率点。KITTI/Dance的选择未改。更低源loss不保证更高HOTA：KITTI仍保留loss规则选择，未看HOTA后改回默认。详见`source_lr_trial_v2.csv`、`source_lr_trial_selection_v2.json`和`stage_A_selected_gate_v2.json`。源官方测试用于用户允许的早停，非盲测。

## 失败排查与科学解释

1. 全源核对101视频、10599表达、6241334缓存标签：Sports的9视频/27表达有564个标签因相同文本跨表达文件取并集而错。改为表达文件ID/原始targets，新标签独立保存；修正后全量0不一致。KITTI/Dance从始至终全量一致。旧标签/特征/预测/模型不删改，见`SOURCE_LABEL_FIX.md`及两轮完整审计。
2. 默认KITTI、Dance存在性准确率/数量MAE虽改善，正目标召回却下降；全帧准确率受空目标主导，不能代替RMOT质量。Dance原生TP20136，默认Head15150；FN22646→27632。按训练原句精确匹配，默认Dance Head的225个未见原句序列HOTA0，原生4.292；这是原句分组，不等于所有语义完全未见。
3. Sports验证266786单元中27420个正候选数>10（10.278%）；指定“≥10”类别在解码中取k=10，存在表示上限。默认/替代头均表现出训练loss继续下降、源验证loss恶化；候选GT→tracker支持变化、原句变化和小源数据过拟合是可能因素，不把任一相关统计冒充唯一因果证明。不改类别数、阈值或GT口径绕过验收。
4. 早先源标签检查只覆盖固定首视频，不能推断全源无误；本轮发现问题后补完整核对与最小修复。梯度/测试通过只说明有限工程性质，B/C尚无真实GPU集成结论。
5. 六个跨域组合中，只有LaMOT→Dance的默认PRES_TOPK点值改善（15.476→15.812，+0.336），其他五个均下降。没有据此选择“最好组合”或声称显著改善；本阶段未预注册跨域A的显著性判定。

最早导出parent最终exit1是延迟上报早先KITTI frame101额外全行数值门槛失败；恢复队列最终exit0，九套完整缓存和CPU评测均有独立完成证据。原非零退出及失败日志保留，不伪报原launcher从未失败。数值策略修复和数据标签修复分开记录。

## 数据、模型与数值口径

- 源KITTI按未修改vendor官方16视频Train（历史自定义17含0017，不混用）；Dance40 Train；Sports45 Train/496表达，不含TAO；可选MOT17/VisDrone Train未准备，不额外引入。
- 源单元1099242/164112/30401，候选–句子特征4534302/1371786/335246。九套目标26946084行PCD；FP16特征，身份推理FP32，三个官方checkpoint strict加载、固定RoBERTa分词person=[5970]。
- 预固定三真实帧维持≤1e-3、0翻转。额外全行复算存在微小误差/近零翻转，全部见`stage_A_numerical_audit.csv`；身份输入、Native/排序统一采用原封存完整分数，真实复算margin.npy仅审计，没有改写旧分数或PCD。
- 目标导出/解码GT打开审计均0，预测先封存再由独立TrackEval/分析读GT。合法源GT仅用于各自源训练及源验证；目标运行时拒绝原始和数值GT标签。Sports公开原图存放在旧数据目录，仅作为合法原始数据，不操作另项目代码/进程。

## 成本、资源与实现状态

'''+f'已知成功阶段累计共享GPU进程耗时 **{known:.6f}小时**（非利用率积分/独占预留时长），包括九套目标、三套源特征及八次源头训练。失败尝试及初次载入有未完全计时开销，不能把该数当严格总上限；日志保留，不假定为0。'+'''

源头训练/峰值显存/更新数见三个cost CSV和`cost_accounting_v2.json`。本轮新增教师forward/目标LoRA GPU小时均0。两个Gate-Reassign参照沿用同一批29410条历史教师判断，不是免费独立系统，也不是FULL成绩：C1 HOTA31.682、C6 CPU派生31.126。

最多两张登记物理GPU；GPU6共享显存后来不足，确认无本项目CUDA后替换为GPU5，GPU4健康任务保留。GPU4/5后续继续按显存与进程数准入，未信号其他项目。CPU评测就绪即评，仅替换无子任务的旧CPU等待器，未重跑健康GPU单元。

PresenceHead精确参数1521676；默认结构/损失/AdamW3e-4等及固定单次1e-5见配置/定义。B/C原型包含信念先验与一次证据、Beta(8,2)DS、IG三视图、快校准、最后两块wq/wv LoRA及流恢复，但未以真实完整运行验证。crowd-kit模拟标签一致率1.0、双方准确率.946，不能代替目标科研结果。

B/C仍保留默认A路径和旧失败gate，修复/源loss选择的v2头尚未接入正式FULL运行清单。不能手工只改gate为通过来运行；未来若新头真正过门槛，还须统一选择清单、准备所选头的九套A输出、发布后完成真实pilot及成本关卡。

## 用户补充的YesTrack定义（已实现，未正式运行）

- BUDGET_RANDOM：seed0派生视频子种子均匀覆盖；BUDGET_PRIOR：各句内按z降序、句间字典序逐层覆盖；首帧优先、余量缩Δ、CROP/继承/阈值0.4；预算与同ρ的FULL实际逻辑提问匹配。未覆盖配对严格逐观测回退Native margin>0，不是UNKNOWN，不追加互斥门控。
- FULL_D10：全配对、Δ10、不匹配预算；覆盖率/实际调用/Δ及GPU成本单列，画入未来调用–HOTA曲线。本轮未运行这些教师对照，不能宣称其精度。
- 当前ρ≤1且FULL只问长度≥3轨迹，预算不超过全部配对数，首帧优先通常没有重复调用余额；Δ取超过最长轨迹跨度。所有逻辑调用、新forward、缓存和等价成本分别记录。

## 交付与恢复

主机器表`stage_A_v2.csv`、`source_lr_trial_v2.csv`；失败与修复全记录、训练曲线、官方详细表及失败日志见`experiment_logs.tar.gz/json`。18项定向检查与compileall见`validation.json`，真实六smoke见`smoke.json`。大文件路径/SHA见`artifacts_stage_A_v2.json`、原A清单和修复清单；不把特征/权重/原GT入git。

物理缓存`/data1/user/TTA-RMOT-assets/presence_active_20261102/`，用户授权的逻辑链接`/data2/user/FlexHook-TTA/results/runs/presence_active_20261102/`；不迁移/删除旧数据。源修复标签、head、预测分别在`source_labels_by_expression_v2`、`heads_source_labels_v2`、`stage_A_label_v2`/`stage_A_source_lr1e5_label_v2`。完整B/C计划仍在`BC_plan.json`，108项状态在`matrix_status.csv`；不打开运行开关。

**本阶段结果：A完整评测完成但验收失败；整个最终科研目标未完成。停止新增训练、教师和方法实验，等待是否允许重新设计存在性头的明确方向。**
'''
    (PUBLIC / 'REPORT.md').write_text(report)
    atomic_json(PUBLIC / 'execution_status.json',dict(status='STAGE_A_COMPLETE_GATE_FAILED_BC_NOT_STARTED',
        scientific_final_goal_complete=False,stage_A_primary_metrics=str(PUBLIC / 'stage_A_v2.csv'),
        source_label_fix_verified=True,selected_gate=selected_gate['status'],BC_jobs_not_run=len(status),
        new_teacher_calls=0,target_LoRA_optimizer_updates=0,additional_hyperparameter_search_started=False))
    print(json.dumps(dict(status='FAILED_STAGE_A_REPORTED_TRUTHFULLY',rows=len(rows),BC_not_run=len(status),known_process_hours=known)),flush=True)


if __name__=='__main__':main()
