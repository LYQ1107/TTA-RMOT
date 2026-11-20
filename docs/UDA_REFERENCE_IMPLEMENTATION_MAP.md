# UDA参考实现映射（截止2026-10-16）

本轮是自行实现的FlexHook适应对照，不是以下方法原版复现。没有把外部模型权重引入四组实验。
固定源码保留在`/data2/user/TTA-RMOT-assets/uda-references`，原FlexHook保留在本仓库vendor。
所有固定提交的日期均早于截止日。除有MIT的FlexHook外，下面几个源码快照未见顶层LICENSE；
只阅读并独立实现通用思想，不复制其源码或重新分发其数据/模型，也不把“公开可读”称为任意授权。

| 参考 | 固定版本、阅读对象 | 实际采用与未采用 | 许可证 |
|---|---|---|---|
| [R1 FlexHook](https://github.com/buptLwz/FlexHook) | `bd1acc38634b28525d54dc6e0fcb38335f0029f9`；`models/mymodel.py::MyModel.forward_features/decode`、`data/utils.py::VIDEOS`、原CE和轨迹输入 | 直接复用原C-Hook/PCD/head及checkpoint；新增可选PCD特征返回钩子和最后两层Q/V LoRA，不重建骨干 | MIT |
| [R2 WSRVOS](https://github.com/viscom-tongji/WSRVOS)、[论文](https://arxiv.org/abs/2604.17797) | `c00a87750a1e6ad61be51ef739ecd773fef39bc3`（2026-04-21）；`wsrvos/losses.py::classification_loss/temporal_ranking_loss`、`augmentations.py::TextAugmentationBank.get/_fallback_negatives`、模型表达输入 | 借鉴表达一致性与弱聚合。**不移植**原表达默认为正、随机其它句当负、伪掩码和单目标分割监督；本实现显式PRESENT/ABSENT/UNKNOWN，目标MIL无独立证据故为0 | 未见LICENSE |
| [R3 DSCA](https://github.com/whbdmu/DSCA) | `15dd448bbdcbb526cd481af6d837e9811d769b17`（2025-01-14）；`engine.py::train_one_epoch_da`、`train.py`联合source/target loader和DBSCAN流程、`models/cpm.py::ClusterProxyMemory`、`roi_head_da.py` | 借鉴source监督持续存在的训练组织与可靠性审视；不复制检测GT、目标box/ID监督、聚类标签或原memory。EMA是本项目的通用teacher baseline，不冒充DSCA完整自校准 | 未见LICENSE |
| [R4 SCALE](https://github.com/whbdmu/SCALE) | `bf58c3d390e57d35d6fb79605be047b7c5f82720`（2026-09-08）；`engine.py::train_one_epoch_da`、`models/resnet_scale.py::AdaptiveFilter/MultiScaleConv`、ROI训练接口 | 用于检查跨尺度/身份可靠性边界；本轮**不采用**其尺度模块、双重聚类或原检测/ReID训练。预测MOT ID只作物理索引，绝不自动成为语言正标签 | 未见LICENSE |
| [R5 GTR+](https://github.com/Flame-Chasers/GTR) | `d9d51d3446c05a871d966064e029339886a90a01`（2026-08-27）；README及`ps_filter.py`/`train_ps.py`入口检查 | 本轮**NOT_USED**：不下载教师、不增加第五组、不注入额外人物语义；如未来引入需另列Teacher-only控制，不能把教师当GT | 未见LICENSE |
| [R6 CD-RMOT-Bench / QCA](https://arxiv.org/html/2607.25239v1) | arXiv `2607.25239v1`（2026-07-28），§4、附录B可靠性/EMA/query对齐与语言锚定 | 论文区分source监督与无标签target输入；可借鉴teacher一致性/源锚定，但未核实可复现完整官方源码，**不是原版QCA复现**；本轮没有ACDA判别器、原LGCA或其checkpoint | 论文CC BY4.0；代码许可未核实 |

对应新增代码：`flex_uda/training.py`（联合训练、EMA、断点）、`losses/objectives.py`
（CE、UNKNOWN-aware presence、合法文本关系、保守伪标签）、`adaptation/tracklet_memory.py`
（只对静态且短间隔历史提供一致性）。目标增强是原冻结特征dropout，不是新的语义教师。

必须承认的差别：本阶段采用同25Val无标签适应后同集评测，不是上述论文的原训练/测试划分，
不直接比较其已发表数字。该参考表只解释代码来源和机制边界，不能作为实验成功证据。
