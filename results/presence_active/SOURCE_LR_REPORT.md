# 单次源域LR缓解对照：真实结果

**历史v1口径：Sports头受源标签按相同文本合并的bug影响，不作为主结果；KITTI/Dance不受影响。修复定义及独立v2结果见SOURCE_LABEL_FIX.md。旧数值完整保留。**

默认3e-4与固定替代1e-5，其他结构/损失/增强/seed/早停规则不变。选择文件在任何替代头HOTA计算前封存，只按各自源验证loss选，不按HOTA选择。源官方测试集用于任务明确允许的早停；不是盲测。全部默认结果保留。

|源/同域测试|变体|源验证loss|Native|PRES_TOPK|Δ|Smooth|源loss选择|
|---|---|---:|---:|---:|---:|---:|---|
|KITTI-V2|default|2.164626|42.752|40.599|-2.153|40.635|否|
|KITTI-V2|source_lr1e5|1.639184|42.752|40.188|-2.564|40.224|是|
|Dance|default|7.434719|32.139|21.605|-10.534|21.921|否|
|Dance|source_lr1e5|4.235332|32.139|25.671|-6.468|25.976|是|
|SportsMOT|default|0.818393|69.924|64.064|-5.860|64.236|否|
|SportsMOT|source_lr1e5|0.745713|69.924|63.937|-5.987|64.045|是|

选择文件SHA256：`665bd1d8f7c152b327dcfae76d064867a6655075c5c8f840584c865ddbff6314`。最终门槛：`FAIL_SELECTED_SOURCE_HEADS_STOP_BEFORE_BC`。失败：['KITTI-V2', 'Dance', 'SportsMOT']。

不能把较低源loss或工程检查通过当成HOTA成功。若失败，不增加学习率搜索，不启动B/C；保留全部默认A结果与一次缓解对照，报告设计/泛化问题。若通过，仍须完成选中头的九套A解码和发布，再进入原任务的真实pilot与40GPU小时成本关卡。

本对照只训练1,521,676参数的PresenceHead；没有新教师调用、目标适应或FlexHook源重训。完整质量/存在性/数量/见过与未见句子分组见`source_lr_trial.csv`；大文件路径和SHA见`source_lr_trial_artifacts.json`。源训练成本从独立训练receipt逐项报告，不与默认训练混算。
