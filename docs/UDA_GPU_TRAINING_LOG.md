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

## 全四组训练完成，首两组官方结果封存

| 方法 | 累计训练worker elapsed秒 | 完成进程peak allocated bytes | optimizer更新 | 同样本AMP重试 | 目标GT模型访问 |
|---|---:|---:|---:|---:|---:|
| SOURCE_REHEARSAL | 919.070289 | 8658199040 | 8973 | 3 | 0 |
| MEAN_TEACHER_UDA | 2907.185327 | 8521772032 | 8973 | 3 | 0 |
| WEAK_MIL_UDA | 2163.689403 | 7819057664 | 8973 | 3 | 0 |
| COMBINED_UDA | 4320.254259 | 8836841984 | 8973 | 3 | 0 |

四组总计2.863944 GPU-process-hours（训练），不是GPU利用率计量；同卡/外国任务共享与失败尾段下界同上。
Source完整推理2977.629690秒，peak allocated5278034432bytes，25视频/425表达，模型GT访问0。
Teacher也已完整封存；Weak继续原GPU6，Combined使用已释放的空GPU5做全25/425推理。
所有训练均结束，没有新增第五种训练。今后入场推理用16GiB分配预算+4GiB余量，
不再用24GiB训练预算排除显存能容纳推理而进程更少的卡；不改batch/FP32/门限或数值超参，不重启健康进程。
本轮Combined仍选中空GPU5，没有为改变卡号而做额外实验。

已封存的Source/Teacher在独立CPU官方评测获得HOTA7.4352/8.2494，匹配Frozen8.2180。
对应数量匹配Frozen7.4207/8.2479；Teacher残差仅+0.0015。其余两组未提前报告指标。

## 最终封存：全部四组评测结束

前述“继续推理”为历史阶段记录。四组25视频/425表达均已封存、官方评测及CPU离线归因结束，无待续GPU训练。

| 方法 | 推理物理GPU | 推理worker elapsed秒 | query-frame FPS | 活动video-frame FPS | 推理GPU-process-hours |
|---|---:|---:|---:|---:|---:|
| SOURCE_REHEARSAL | 3 | 2977.629690 | 145.631 | 8.567 | 0.827119 |
| MEAN_TEACHER_UDA | 5 | 1788.523097 | 242.455 | 14.262 | 0.496812 |
| WEAK_MIL_UDA | 6 | 3735.972314 | 116.070 | 6.828 | 1.037770 |
| COMBINED_UDA | 5 | 1771.851628 | 244.736 | 14.396 | 0.492181 |

每组FP32、0测试时optimizer/EMA更新；checkpoint仅源CE选择；margin>0门限固定，无目标GT模型访问。
主推理peak allocated约4.916GiB（Source/Teacher/Weak5278034432bytes，Combined5276297216bytes）。
训练与推理合计约5.7178 GPU-process-hours；训练scheduler wall4736.92s、推理跨度5533.45s、
重叠pipeline wall7319.48s，不累加重叠wall。共享卡负载导致FPS差异，不宣称方法本身固定有三倍速度差。
每个完成receipt、完整25视频parity、GPU UUID/入场资源快照和失败/恢复历史均保留；没有重跑已完成episode。
最终科研结论为未验证可靠语义提升，不以完成训练或非零梯度代替效果证据。正式40Train→25Val尚未执行。
