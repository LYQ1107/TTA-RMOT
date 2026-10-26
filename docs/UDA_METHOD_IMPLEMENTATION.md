# 首轮25 Val同集无标签适应诊断：预注册

用户追加授权后，本阶段为`TRANSDUCTIVE_VAL_DIAGNOSTIC`，不是40Train→25Val UDA，不是独立盲测。
模型离线接触待评视频的无标签图像/查询/MOT；测试前向仍用当前及过去四观测，不做测试时更新。
目标GT不参与训练、阈值选择或checkpoint选择，40/385不作为训练先验。

## 固定数据与预算

- 初始化：官方V2 SHA`71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613`。
- 源域：完整17Train、8896原始表达、5939帧；每表达预定一个合法窗口，并补足未覆盖的有标注物理轨迹，共8973窗口。
  GT框/ID及表达成员只来自源域。25条表达没有可用源bbox正例，保留表达并记录，不制造缺失框。
- 源模型选择：官方四个source-val视频，每视频固定16个等距表达，共64窗口；只按该固定集合CE选择。
- 目标：25Val、425真实输入表达，每条在MOT活动帧索引10%/50%/90%取一个窗口，共1275窗口。
  窗口位置与目标GT无关。源/目标1:1配对，长度取较长端；一完整覆盖周期8973step，不机械限500step。
  这是25×3=75个不同物理frame-anchor、每个17查询；不是对全部目标帧优化。每个anchor还读取过去四次同ID观测。
  完整覆盖指视频/表达/有标注源轨迹覆盖，不能把8973个重复目标窗口称为8973个独立目标帧；完整评测仍为全部活动帧。
- 四组相同rank4、PCD最后两层Q/V LoRA、24576可训练参数；原视觉/语言/C-Hook/head冻结。
  原始多层decode直接复用，训练CE用所有层，正式决策用最后层二分类margin>0；不对候选维softmax。
- seed17 / AdamW lr5e-5 / wd1e-4 / clip1 / AMP decoder、敏感loss FP32；冻结编码器FP32缓存。
- 每250step保存完整student/EMA/optimizer/constant-LR状态/scaler/RNG/采样进度/统计与记忆；
  跨视频不重置LoRA。初始和预定checkpoint按source-val CE选择，不按目标HOTA选择。
- 先每组真实20step GPU smoke；通过后每组一个完整覆盖周期。不会依据本次Val结果延长轮数或搜参。

## 四组有效监督

| 方法 | 源监督 | 目标约束 | 需要说明的限制 |
|---|---|---|---|
| SOURCE_REHEARSAL | 原多层CE | 无 | 不接收目标无监督loss |
| MEAN_TEACHER_UDA | 同上 | EMA弱/强视图logit一致性、筛选后的平衡伪BCE | 来源相近的模型一致并非独立真值 |
| WEAK_MIL_UDA | 同上+可确认源presence | 合法等价表达一致性、整套纯色穿着的候选内互斥；MIL按证据跳过 | 目标presence当前全UNKNOWN，不能宣称有效MIL正例 |
| COMBINED_UDA | 同上 | MeanTeacher+上述关系+冻结源锚点+语义/时间合法的短时一致性 | 全句同时含动作与服饰时是MIXED，不能强迫全ID时间恒定 |

目标presence没有独立证据，所以SUPPORTED_PRESENT/ABSENT覆盖为0；UNKNOWN不变成负例。
伪标签POS/NEG是相关source/EMA/增强置信代理，不是真实标签；若无高置信POS，不用整批强NEG训练。
保留所有原查询，包括拼写`dacning`，不自动纠正未知语句。训练合法等价文本只用于辅助关系损失，正式测试文本不改。
例如`dancing person dressed all in black`与`a person who is dancing and dressed entirely in black`是等价约束，
它不声称视频有黑衣人；不同人的服饰可以同时匹配，同一候选的明确“全黑/全白”陈述才采用互斥。

源码：`flex_uda/data`、`model/flexhook_uda.py`、`losses/objectives.py`、`adaptation/tracklet_memory.py`、`training.py`。
公开框架仅作思想参考，不将本实现冒充QCA/DSCA/WSRVOS原版复现；完整固定版本参考映射后续单列。
原始C-Hook跨表达point-dispersion regular在单表达N=1时分母N(N-1)=0；C-Hook冻结，
该项对授权LoRA参数没有梯度，因此本轮不将不适用的跨表达regular加入源训练目标。
初次smoke曾把其NaN常量加入有限CE，梯度仍有限但总loss无效；独立loss审计撤销该批smoke通过状态，
保留旧日志/checkpoint，修复后在新目录重做20step。严禁nan_to_num掩盖错误或带NaN开长训。
旧六方法和冻结诊断不改。目标评测完整425封存后由独立进程运行官方TrackEval及Count-Matched/排序/presence归因。
