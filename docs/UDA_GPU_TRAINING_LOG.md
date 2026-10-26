# 实际GPU训练与数值恢复日志

当前阶段为25Val同集无标签适应诊断。所有时间都是实际测量，不把20步smoke当完整训练。

| 阶段 | 物理GPU | 真实状态 |
|---|---|---|
| 四组首次完整覆盖 | A5 / B4 / C6 / D3 | 同一固定rank4/lr5e-5/seed17；均因AMP缩放梯度溢出保留现场，原launcher失败状态不覆盖 |
| A断点恢复 | 5 | **8973次更新完成**；17源视频/8896原表达+77轨迹补充窗口；3次同样本AMP重试、0跳样本 |
| B/C断点恢复 | B6 / C5 | 各自checkpoint恢复，独立RNG/optimizer；C入场曾与A同卡，不谎称全程独占四卡 |
| D断点恢复 | 4 | 原健康进程未被打断，待其实际溢出退出后才恢复 |
| A先行评测 | 3 | 源CE选中7750步；真实17表达批推理parity已通过，完整25/425评测随后执行 |

A完成进程记录：累计训练elapsed919.070289秒（含有效checkpoint前缀与失败尾段下界、恢复），
peak allocated8658199040bytes（约8.06GiB）、24576可训练参数、参数delta L2约6.05048、
8973次有限非零梯度、源参数冻结不变、目标GT访问0。
源开发CE由初始约1.02226下降，选择7750步CE0.9631536889；最终8973步CE0.9647281944未被选中。
这只是源域选择与真实训练证据，尚不是目标域提升。

首先完成的方法立即评测，不等待较慢训练。A的FP32真实首/中/末三帧×17表达批推理，
相对原逐表达decode最大margin误差约1.15e-5，0个选框差异；后续每视频检查parity，近零margin回退原路径。
没有改测试门限、MOT、输入图像、查询语句或模型数值超参数。

后续入场已排除仍健康的本任务GPU worker，同时保留已开始的健康共享进程，不为重新排卡而杀进程。
用户允许与其他任务共享；最多物理GPU3/4/5/6四张，不停止任何其他项目进程。
峰值只指完成/恢复进程记录的allocated峰值，失败进程未单独记录的峰值不冒充已知；每进程分配上限24GiB。
训练与评测交错，其wall可能重叠；最终成本分别报告实际worker elapsed、GPU-process-hours与pipeline wall，不能直接叠加两个重叠阶段的wall。

原始恢复路径：`/data2/user/FlexHook-TTA/results/runs/uda_val_full_20261026_v1`。
每个恢复launcher位于其`launcher_recoveries/`；原四组`launcher_status.json`表示**首次失败尝试**，
最终状态由各方法receipt和全部完成后生成的`full_suite_completion.json`确定。
原始预测位于`results/runs/uda_val_evaluation_20261026_v1`，各独立评测launcher放`evaluation_launchers/`。
GitHub精简记录：`results/uda/full/SOURCE_REHEARSAL/`；其余方法完成后同样归档，不提前填入指标。
