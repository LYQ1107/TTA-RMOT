# MLLM 原始软标签审计

当前按2026-10-28用户修订使用25Val同集开发诊断，不是40Train→Val；指定40Train预测MOT仍不存在。
完整官方25个Val文件SHA与原发布receipt一致，合计211085行，完整425表达。

冻结T0为Qwen/Qwen3-VL-2B-Instruct，revision `89644892e4d85e24eaac8bacfd4f463576704203`，
参数SHA `9a13380b4279bc98f4680bd083ff4995fdcfec120cfded459513be90d041a956`。
BF16，无量化/LoRA/更新；只复用P1 A真实单帧整句回答，不使用B/C/D/时序修正。
旧P1 raw输入、prompt、bbox、model和图像SHA均校验；没有新教师GPU调用。

封存缓存：`/data2/user/TTA-RMOT-assets/mllm-distill/val_pseudo_v1`。
seal SHA `73494a2d79ea4560c9f9d6ddb2d48e25027766b9f5e562516af11966d42eeb93`。
含manifest/原始Yes-No logits+BF16 p_yes/单独valid/运行时文本/2125个时间窗口，均逐文件hash，只读。
8381配对全部为有效响应；980个p_yes>0.5，7401个≤0.5。
这不是GT正负计数，也不是训练的硬POS/NEG筛选；直接对全部有效软概率计算BCE。
1113个旧P1低置信度UNKNOWN仍有真实有效原始软响应，保留其概率，不伪造NEG。
未采样/无效候选无损失；独立人物二分类，不在人群间softmax。

缓存导出的identity key显式包括模型revision/prompt哈希/整图和ROI SHA/bbox哈希/视频/表达/帧/预测ID，
另保留parent raw request引用。原始P1 cache不被改写，学生不会在线调用MLLM。
独立MLLMPseudoDataset按sample ID和全部identity字段核对，不能仅按数组行号拼接。

当前已有19个针对性测试通过；GPU接口v6在8组固定窗口/31候选上做真实forward/backward但0step，
KD梯度非零、完整原V2严格加载、RoBERTa person=5970、保存恢复和临时checkpoint清理通过，目标GT打开0。
smoke_v1因源表达读取不完整已撤销；smoke_v2已通过完整源计划检查，6个实际更新/6个非零KD梯度、
参数变化L2=0.027242、冻结参数未动、目标GT打开0、峰值5.247GiB；完整源计划8973窗口，不把6步称全量训练。

机器记录见`results/mllm_distillation/assets/`。P1语义质量仅引用封存的旧报告，
本轮学生指标只有完整预测封存后由独立GT评测生成，不将P1候选AP冒充学生HOTA。
教师完整预训练语料与Refer-Dance潜在重叠仍未知；T1官方V2 LoRA未取得，不用其它权重替代。
