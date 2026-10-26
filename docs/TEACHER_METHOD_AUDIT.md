# 教师机制与来源审计

P1不训练学生，也不将分解或视觉扰动本身当成已证明创新。

| 来源 | 实际固定版本 | 本轮实际复用/参考 | 边界 |
|---|---|---|---|
| [YesTrack](https://github.com/ggbondrighthere24/YesTrack) | ed42b8315b28c94dc595d83b0d49bf57002bb281，2026-07-22，MIT | 安全AST只提取模型加载、Yes/No token、裁剪、单帧/视频前向定义；原提示词/ROI/二分类softmax | 不执行其GT dataset、llm_train或评测入口；保留原始分数与TCP/TRP |
| [DINO Teacher](https://github.com/TRAILab/DINO_Teacher) | 44ae126c5d9844752afbd8bdfdd65c92cf713fb1，Apache2.0 | 阅读gen_labels/trainer的离线预测、去除无标签实例、伪标签存储和师生解耦 | 不引入检测器、DINO权重或目标框/ID监督 |
| [DKGTrack](https://github.com/acyddl/DKGTrack) | 197f354443bd1e7b490d204456a7654b7d1e4ccd，未见顶层LICENSE | 阅读fuse_modules、spatial_temporal_reason的语言/历史融合与静态/运动边界 | 不复制网络；分解服饰/动作不是本研究独有贡献 |
| [WSRVOS](https://github.com/viscom-tongji/WSRVOS) | c00a87750a1e6ad61be51ef739ecd773fef39bc3，未见顶层LICENSE | 阅读augmentations/model/losses中的文本构造、聚合和有效mask | 原文默认原句正/随机其它句负不能直接移植；本轮保留UNKNOWN |
| [SoftMatch](https://github.com/Hhhhhhao/SoftMatch) | a7e9da26e3e96962d159b5d8157cbd99b2cb0fdd | 该仓库只有README，明确指向微软USB/TorchSSL；已阅读USB的softmatch/utils.py和softmatch.py | 仅采用截断高斯置信权重的简单筛选对照，不冒充完整半监督分类训练复现 |
| [VES-RFT](https://openaccess.thecvf.com/content/CVPR2026/html/Hou_VES-RFT_Rewarding_Visual_Evidence_Sensitivity_to_Mitigate_Hallucinations_in_Large_CVPR_2026_paper.html) | CVPR2026，Hou等，4168–4177页 | 对照有图/无图判断熵差的动机；增加等面积目标/背景遮挡 | 不执行RL，不声称复现其奖励/GRPO或论文数字 |

YesTrack代码许可全文保留`docs/licenses/YesTrack-MIT.txt`。只读没有明确LICENSE的代码以核对思想，未复制其实现。
官方TCP/TRP采用用户明确同意的13帧因果窗口原策略：stride4，window6，阈值.4，boost.3，
单帧[.2,.8)视频refine。窗口重置是P1采样范围，不伪称完整视频从头状态或完整HOTA。

统一后端实际调用原单帧前向函数，记录原BF16二分类softmax与FP32诊断值；多帧和无图控制
使用相同聊天模板/左padding/末token提取，不导入可能建立GT dataset的上游模块。
原模型的完整参数SHA在加载后和封存时比较，每批检查参数版本；无optimizer或学生更新。
