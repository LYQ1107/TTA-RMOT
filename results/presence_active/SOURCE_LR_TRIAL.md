# 单次源域学习率缓解对照（不是目标GT调参）

默认结构/默认3e-4学习率的三头均由源验证loss选择epoch1，随后训练loss继续下降、源验证loss上升。默认同域HOTA为40.599/21.605/64.064，原生为42.752/32.139/69.924，三者均未过−1门槛。全部默认checkpoint、预测和指标保留。

利用任务原则4已经允许的超参数调整，只试一个预先固定的替代学习率 **1e-5**，三个源模型一致；不是学习率搜索。网络、标签、数据、增强、AdamW/weight decay、batch256、最多20epoch、cosine、patience3、seed0、checkpoint源验证选择规则不变。只训练1.52M参数的PresenceHead，不重训FlexHook。

- 独立输出：`/data2/user/FlexHook-TTA/results/runs/presence_active_20261102/heads_source_lr1e5/<model>/`。
- 每个源模型在默认与该单次缓解对照之间，**仅按自身官方源验证 BCE+0.5×正单元条件count CE 的最小值选择**；相同则选默认。
- `source_lr_trial_selection.json` 在任何新头HOTA计算之前封存，记录两项源loss、路径和SHA。
- 不用其他目标数据集GT，不用任何跨域HOTA选学习率/epoch/seed；不修改0.5解码阈值或同域−1验收线。
- 默认A结果完整报告。选中的三个源头仍须通过同域门槛且先发布，才允许B/C。若本次仍不能通过，停止在源头设计问题上，不自行增加其他搜索或新结构。

完整B/C的40 GPU小时门槛保持原样。本对照无新教师调用、无目标LoRA、至多两张登记物理GPU；全部已封存特征复用。
