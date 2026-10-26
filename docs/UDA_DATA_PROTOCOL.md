# 数据协议：当前25Val诊断与未来40Train UDA分开

日期2026-10-26，项目`LYQ1107/TTA-RMOT`，工作区`/data2/user/FlexHook-TTA`。
用户最新要求先用25Val看效果，本轮名称固定`TRANSDUCTIVE_VAL_DIAGNOSTIC`。

| 项目 | 当前使用 | 权限 |
|---|---|---|
| Refer-KITTI-V2源训练 | 完整17视频、5939帧、8896原表达，8973固定窗口 | 源GT框、ID、表达成员合法监督 |
| 源模型选择 | 0005/0011/0013/0019，每视频固定16表达，共64窗口 | 仅source-val CE，初始step0也可被选中 |
| Refer-Dance目标适应 | 25Val、425表达，活动帧索引10%/50%/90%共1275窗口 | 仅图像、原句、固定预测MOT；跨episode持续更新LoRA |
| Refer-Dance目标评测 | **同一**25Val、425表达，全部活动帧 | 预测封存后独立官方TrackEval/离线GT归因 |
| Refer-Dance Train | 40视频、41796图像、1560表达已准备 | 当前用户明确暂不使用；同配置预测MOT仍缺 |

“官方25Val”指FlexHook作者发布的DETR+NeuralSORT候选文件夹只有25个Val输出，
不是说Refer-Dance数据只有25视频，也不是用目录`training`推断Train/Val。
分割由固定官方`data/utils.py::VIDEOS['dance']`得到，40Train与25Val视频无重合。

V2完整17Train来自已审计TempRMOT清单；FlexHook默认16Train漏0017。本轮按照用户完整源域要求
显式包含0017，不将其伪称与默认16Train重新训练协议完全一致。四个source-val视频不进入训练。
25条源表达没有当前可检索正框，保留并记录；不制造框或正例，不丢弃难表达。

数据优先保留`/data2/user/TTA-RMOT-assets/flexhook`；已有源资产和官方checkpoint在data3只读复用。
source图片/表达的历史软链接源虽来自其它项目数据区，只有读取，绝不改LocateMOT/GroundJEV文件或进程。

## GT隔离

目标runtime只读取`manifests/dance_val_runtime.jsonl`中白名单字段：视频、表达ID、句子、图片根、
帧范围和预测MOT路径。目标原始表达JSON可能携带标签，因此准备阶段只摘取sentence，运行时禁止回读。
训练/推理审计钩子禁止目标`labels.json`、`gt_template`、`gt.txt`、原始expression目录。
评测路径另放`dance_val_evaluation.jsonl`，训练和推理从不导入它。

训练不使用目标GT、40/385比例、未来目标存在性、目标HOTA、线性探针权重或外部语义教师。
当前输入句子不代表有人符合；同一候选允许多个真实匹配，整体允许空集合。
只有缺少独立target presence证据的事实进入代码：UNKNOWN必须跳过，而不是正例或负例。

## 结果解释

本轮不是Source-Free、不是online TTA、不是40Train→25Val UDA，也不是未见视频泛化测试。
整段目标无标签视频可用于离线适应，但评测单次前向只用当前及过去四个同ID观测。
训练后不再执行optimizer/EMA，checkpoint按源CE选择、预测门限固定0、MOT框和ID逐行不变。
最终必须另列Count-Matched Frozen诊断；它以适应结果数量为条件，不能独立部署、不是GT-free在线方法。
