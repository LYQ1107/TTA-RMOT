# v8P：按查询协议决定是否推断存在性（最后一轮方法改动）

## 登记与背景

项目 LYQ1107/TTA-RMOT；分支 codex/presence-active-v8p-20261106；起点 f53cbc1751fe4a129bbbdac7636c1599964437d7。
本文件先提交并推送，再执行协议分类、测试和真实实验。v8 文件、历史预测与原阶段 3 队列（FULL C1/C4/C6，REPLAY C1/C4）不改动。原阶段 4 暂停。

### 修订 C1（在看到 D3 之后制定）

只读取运行时查询文本元数据：目标集合中所有视频的查询文本集合完全相同，则 CHECKLIST；其它情况为 PRESUPPOSED。运行开始计算一次，封存各视频集合哈希、唯一集合数与协议。断言 Dance=CHECKLIST，KITTI-V2/SportsMOT=PRESUPPOSED。

这是在看到 v9 D3 后制定的规则，不是事前发现。理由是存在预设理论；在本轮三个数据集上等价于按数据集设定存在性先验。D3 已知 Dance 25 视频/1 集合、SportsMOT 45 视频/41 集合、KITTI 4 视频/4 集合。分类不读取 GT，不以 GT 选择阈值。官方 val 用于方法开发，不宣称盲测。

## 固定实现

新增 flex_uda/active/v8p.py、scripts/active_v8p_run.py，继承 v8，不修改 v8 文件。

CHECKLIST 完全走 v8 代码路径。C1/C6 引用已封存 v8 MINUS 的预测 seal SHA，不重新调用教师、不重新生成。以两个伪造视频验证 CHECKLIST 的状态/问题/解码与 v8 逐字节一致。

PRESUPPOSED：presence_probability 恒为 1，absent 和 low_tentative 恒为 False；不产生 GLANCE、PRESENCE、PRESENCE_COVER，不运行 PresenceEM；预算全部给 IDENTITY/POSITIVE_RECHECK，排序同 v8。DS、路由、快校准、慢分支 BCE/16+16/门槛、输出 mask 沿用 v8；ABSENT_NEG 为空。收据写 protocol、presence_inference=False。教师支持的快校准仍要求实际回答，不把未问对设为伪标签。

其余沿用 v8：教师、候选、看法/关键帧、PRES_TOPK、f/截断先验、DS Beta(2,2) 与单次证据上限 2、伪标签 >.85/<.15 且仅已问对、w∈[0,1]、a_q L2=1、T/β、四轮、seed、预算和输出规则不变。不得由新结果改变。

## 阶段 A（最高优先级）

预算 ρ=.3。运行 MINUS_SLOW_v8P C2/C3/C4/C5；六方向主表 C1/C6 取 v8 MINUS。
同时满足以下才通过：
1. 至少 5/6 方向严格同时高于固定 YT_PRIOR 和 UNIFORM_K；
2. C1≥35.724，C6≥34.469；
3. C3≥63.35。

报告 v8P−v7、v8P−v8 的成对视频聚类 bootstrap（Dance 25、SportsMOT 45 视频，2000 次、seed 0）；KITTI 4 视频只报点估计。未通过即停止新增实验，提交各方向错拒、高置信错负、GT 恢复归因，不运行 B。

## 阶段 B（仅阶段 A 通过后自动顺序执行）

B1，ρ=.3：
- RANDOM_ACQ、−DS、−ROUTE：六方向；
- −GLANCE、−EB（ρ、π 固定 .5，无 EM）、−PRES（同 PRESUPPOSED）：仅 C1、C6；
- −PRESUPPOSITION：直接引用 v8 MINUS C2–C5，不重跑，表中明确标记。

B2，ρ=.1、1.0：MINUS_v8P、YT_PRIOR、UNIFORM_K 六方向，每格匹配实际调用预算；ρ=.3 复用既有结果。

B3，低优先级且有空闲 GPU 时：FULL_v8P C4、REPLAY_v8P C4。MINUS C4 预测封存后才运行；回放冻结问题、回答和存在性快照。

## 资源、隔离、封存

同一项目所有队列合计最多两张物理 GPU；不停止、不抢占其他任务。共享授权仍有效，优先显存充足且进程少的卡。原 v8 阶段 3 健康任务保持不动；新任务只共享原队列已登记两张物理卡上通过显存准入的空间，不足则等待；原队列结束后可重新选择最多两卡，不额外占第三张卡。此资源调度澄清在任何 v8P 真实结果前登记，不改变方法或验收。长任务使用一个阻塞式 shell/内部观察机制，不进行 agent 级进程轮询。

生成进程 GT 打开尝试必须 0。先封存预测，再独立官方 TrackEval、GT 归因和 bootstrap。完整候选库，25/425 Dance、4/862 KITTI、45/466 SportsMOT v4。大文件不进 git；优先 data2，已有获准 data1 缓存不迁移。所有缓存命中与新调用单独记账，不掩盖逻辑教师预算。

## 交付顺序

1. NATIVE、PRES_TOPK、YT_PRIOR、UNIFORM_K、v7、v8、v8P 主表，HOTA 原精度及 v8P DetA/AssA/IDF1。
2. 协议、错判空、正确判空、query_empty 预测行数。
3. B1 消融与 B2 预算曲线（仅 A 通过时）。
4. v8 四类行为图：问题类型占比、提问前后后验直方图、GLANCE 命中率、ρ̂ 曲线；按 CHECKLIST/PRESUPPOSED 分开，禁用项标 N/A，不能虚构。
5. 成本、新调用/缓存命中。
6. 限制：规则 post-D3，KITTI 仅四视频，官方 val 开发。

各独立阶段提交推送。A 不通过就停下讨论；不启动其它方法，不恢复旧 80 项。
