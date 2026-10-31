# 固定执行约定：数量、结构与教师问法（不训练学生）

起点61a388e884f94124ad05e53ba0f21b9f7dbbf2f3；分支research/v2dance-count-structure-20261031。用户附件为完整协议，本文件只明确实现细节；不按Val结果选变体。

1. Phase1全部使用已封存完整211085个MOT观测×17句、425个episode。FROZEN/SAME_SET_D1/T2V_D1_s17/T2V_D3_s17先核验缓存和原margin的SHA。旧结果不改，新的6个oracle和28个选择器全报。
2. GT-free生成与教师进程安装原v2 deny_gt审计，目标GT/原表达JSON禁止读取。oracle、官方评测、探针离线评价是独立进程；预测先逐文件封存。源域GT合法，不能与目标GT计数混淆。
3. Oracle逐帧匹配复用v2最大有效匹配数优先、IoU≥.5。P(q,track)=匹配观测数/全部该轨迹观测数≥.5。平均GT数round-half-up；逐帧预算截到[0,候选数]，排序/ID tie复用v2并与topk_ids核对。F_FRAME必须逐行等于旧M_FROZEN__ORACLE_GT，201009行/HOTA45.794，否则停止。
4. C1/C1c/C2/C3/C4/C5/C6原定义不变；C2/C3/C4查询tie按表达文件名字典序，C4的目标全局均值按各margin库分别算，源均值固定旧源Frozen统计。仅C1c/C2不使用未来观测；其余明确离线。
5. C5仅执行用户明确要求的两参数源域Platt校准，不训练或更新学生/教师。用原2355528行源Frozen候选，源原标注几何做IoU一对一匹配；无正例权重/目标标签，最小化源域二元NLL。四margin库共用该源Frozen校准，不新增学生源域推理。标签移位假设对适应后分数未必成立，必须披露。Saerens EM逐q初始pi_s，最多100次，容差1e-6；log-odds稳定实现，边界仅1e-12数值截断，记录未收敛项，不弃样本。C6复用C5 posterior后轨迹求均值。
6. 探针视频为排序位置1/6/11/16/21。每视频取完整运行时首末帧之间10个等距帧，round-half-up、含端点；无候选帧保留。全17句按文件名排序。探针只有5视频50帧的描述性评价，不将未观察的其余帧补NEG，也不报告伪装的425序列全量HOTA。
7. T1使用原320方形/.2 margin/整句YesNo代码和prompt。T2同一crop、归一化中心，固定A–Q为17句，R=none；18个单letter token logits的FP32 softmax，保存原logits、argmax及所有概率；二元比较/评价严格p>.5，R比例另用18类argmax。T3原始整帧、不额外裁剪；红框2px、白底黑字DejaVuSans24px，1-based序号按track_id。固定用户给定prompt，greedy最多64 tokens，新的生成路径允许KV cache，不改旧YesNo路径。
8. T3只接受逗号分隔合法编号或none（忽略外围空白/末尾句点），重复编号去重并计数，越界/其它文字/截断格式失败均保留为UNKNOWN，不当NEG。逐候选p为解析后的0/1，因此其AUROC是二值分数AUROC而非概率校准。失败逐项保留，报告有效覆盖、条件指标及将失败计错的全请求准确率，不默默丢弃。
9. Qwen3-VL-2B固定旧revision89644892e4d85e24eaac8bacfd4f463576704203，BF16，无量化/adapter/更新。只在已有可核验原版8B权重时加跑T2/T3，不下载、不借用其它项目微调模型。单GPU审计准入，允许既有共享授权，优先进程少且显存足够的卡，不kill外部进程。先少量功能检查，再同一缓存补齐固定探针，不改问法。
10. 全部官方评价复用v2的TrackEval、25视频成对bootstrap2000次seed0。6个oracle全部标ORACLE/GT仅诊断；GT-free选择器只说明当前选择进程不读GT，SAME_SET_D1输入仍是历史同Val教师蒸馏分数，不冒称Train→Val。三个描述性判定逐字按用户规则，不能推荐胜者。
11. 新目录results/count_structure和results/runs/count_structure_20261031；大文件只记路径与SHA。每阶段发布精简记录；Q1–Q4各不超过五句话。完成后停止，不补训、不加方法、不选问法。
