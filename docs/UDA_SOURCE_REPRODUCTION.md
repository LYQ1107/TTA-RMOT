# 源模型与Source Rehearsal说明

初始化始终为官方`refer-kitti-v2-best.pth`，不是重新完整训练Swin/RoBERTa/FlexHook。
Frozen全25/425重放与官方TrackEval确认HOTA8.2180/DetA3.1937/AssA21.1470/IDF1 4.4414。
重放没有新GPU模型推理，不能把其CPU耗时当原模型GPU速度；真实smoke另检查零LoRA FP32输出与封存源分数。

Source Rehearsal仅训练与其余三组完全相同的24576个LoRA参数；源域CE贯穿全部8973窗口，
四组只用64个固定source-val窗口CE选checkpoint，包括未更新的step0。
如果最终选中step0，必须明确“源域选择没有接受任何更新”，不能改选目标HOTA更高的checkpoint。
SOURCE_REHEARSAL没有目标loss；B与A源CE相同，可作为基本目标一致性对照。
C/D额外包含source-presence项，因此相对A变化不能全部归因于目标约束；本轮没有新增source-presence-only长实验。

完整17Train与官方FlexHook默认16Train的0017差别明确记录，不声称精确复现原论文完整训练。
源监督框、ID、表达成员只来自源GT；不能把source框规范化坐标错当目标预测MOT。
输入归一化、224×672及因果四观测组织直接复用原FlexHook适配。

单表达N=1时原C-Hook跨表达dispersion项分母为0，而C-Hook冻结且该项对LoRA没有梯度。
初次smoke总loss因此NaN，虽有有限CE梯度也已整体撤销。修复不改变原decode或源CE，
只不把不适用的冻结NaN常量作为训练目标；独立审计要求每步实际loss有限。
