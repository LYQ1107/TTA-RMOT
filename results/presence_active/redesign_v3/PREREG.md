# PresenceHead v3：源域监督的残差前缀数量分布

2026-11-02；项目 LYQ1107/TTA-RMOT，分支 method/presence-active-20261102。
用户在 A 失败完整报告后明确允许修改存在性头的结构/损失。原 A、LR 对照、Sports 标签修复、预测和报告全部保留；本轮不是覆盖历史或放宽验收。

## 假设与一次固定改造

旧头从零预测相互独立的存在性/11 类数量，源验证 loss 改善不代表保住 RMOT 召回；Sports 存在数量上限。检验一个更小、以原生决策为起点、没有固定数量上限的头能否保持同域性能。没有充分理由再用目标指标搜索新头。

- FlexHook、身份 margin、候选框/ID/帧号完全冻结，复用已封存的全部特征。Sports 源训练只用 expression_id 修正标签 v2。
- 对一个帧–表达单元，按原生 margin 降序（相等保持原 track_id 顺序）排列 n 个候选。共享候选映射为 LN(768)→Linear(64)→GELU；几何/原生 margin 为 Linear(5,32)→GELU。拼接局部 96 维及集合 mean/max 各 96 维，接 Linear(288,64)→GELU→Dropout(.1)→Linear(64,1)，末层零初始化。没有独立句子 ID、词表、video 参数或句子向量捷径；PCD 本身仍是查询条件特征。
- 增量 u_i = 原生 margin_i + residual_i；数量能量 E(0)=0，E(k)=sum(i≤k)u_i。softmax(E(0..n)) 是联合数量分布，允许空集和所有候选，无“≥10”截断。集合内顺序不被 residual 修改；实际输出始终按原生 margin 取 top-k。
- 零初始化的联合 MAP 数量严格等于原生 margin>0 的数量（能量相等取最小 k）。这是显式原生初始化，不是性能提升；将单列真实缓存的初始化等价检查及学习后变化比例。
- 本轮解码随新输出改为联合数量 MAP（含 k=0），不再把两个不一致头的阈值和条件 argmax 拼接。存在性概率为 1-P(k=0)，条件数量分布由同一个联合分布导出，供后续 B/C 使用。报告同时给概率阈值的存在性质量与实际 MAP 输出质量，避免混淆。
- 损失固定 L=CE(E, 源正候选数)+0.1 KL(P_native || P_head)，其中原生分布是同一冻结 margin 的前缀能量 softmax。KL 是软保留先验，不是禁止改变预测的硬回退；不使用任何目标 GT。
- 只运行这一种新结构/损失，不新增本轮超参搜索。AdamW lr=1e-4、weight_decay=.01、batch=256、seed=0、max_epochs=20、cosine T_max=20、source-validation patience=3。沿用候选 dropout .1、margin 噪声 .1、PCD dropout .1。checkpoint 仅取最小自身源验证 L；不比较旧头 loss（目标函数不同），不按 HOTA 选 epoch/旧头/seed。记录 epoch0 验证仅作原生先验对照，不参与选择。

## 验证与停止条件

1. 定向检查：初始化原生数量/空集/>10、padding 与排列不变性、有限非零梯度、checkpoint 严格加载和 GT guard。不重导 backbone，不调用教师。
2. 先训练三个自身源头；封存各自 source-loss 选择和 SHA 后，GT-free 解码三个完整同域候选，再独立官方 TrackEval。预测先封存，GT 打开尝试须为 0。平滑沿用 ±round(fps/2) 实际帧号窗口，仅作单列对照。
3. **原门槛不变：三个源域的 PRES_TOPK HOTA 均 ≥ 自身 NATIVE complete HOTA−1。** 未通过就报告新的真实失败，不绕过门槛调用 B/C，不继续无界改结构/调参。
4. 三者通过后，完成全部九格新头 A 解码/评测，推送，再用显式 v3 头/解码清单接入 B/C。C1/C3 实际 pilot 先行；保留原 40 GPU 小时成本关卡，超过则先主方法和关键对照后报告等待，不展开 108 项全量矩阵。
5. 最多两张按显存和进程数准入的共享物理 GPU，不停止他人任务。新文件在既有 data1 缓存根的 redesign_v3 下；data2 仅代码/精简记录/既有链接。一个阻塞协调器等待，不做 agent 级轮询。

交付：本目录 REPORT.md、训练/选择/GT 审计/门槛、完整机器表及路径 SHA；源训练/CPU 解码/官方评测成本分开，旧 A 主报告不覆写。先提交并推送本预注册，再运行新训练和新评测。
