# D1 Train→Val：冻结执行约定

起点619390d，分支research/v2dance-train2val-20261031。完整用户任务为准；本文件只固定执行要点，不覆盖用户约束。

1. 已完成P0：40Train/41,796帧/348,930原始MOT几何行/419轨迹；每视频自己的39表达，共1,560组合。不使用跨视频文本并集。Val25×17=425，视频互斥，8个表达文件名重合，因此并非全部未见语言测试。
2. 原始MOT仅frame/id/xywh合法，独立转换10列，无过滤/插值/改ID，不读mark/class/visibility。它是Train的GT候选几何，而Val使用预测NeuralSORT；明确披露此差异。
3. 文本独立进程只解码sentence；训练/教师不读混合表达容器或任何目标referringGT。Train-only输入审计先于cache加载，kernel whitelist逐Train视频授权，不授权整个Dance图片根目录/整个manifests目录。旧Val教师库及D-FIT输出禁止读取。
4. 固定Qwen3-VL-2B revision89644892e4d85e24eaac8bacfd4f463576704203，BF16，无量化/adapter/更新；不改yestrack_backend.py或visual_input_builder.py。
5. 原Naive sample_episode/seed17原样重用：五分位锚点，2条轨迹，±1端点，每query最多20判断。源模型分数只在所需锚点与端点计算；不是新全量Frozen或TTA评测。先真实4样本smoke，再全Train。未采样/无效/UNKNOWN不变NEG。
6. D1=diag_fit_opt，seeds17/23/42；D3=diag_fit_cap，seed17（其原配置auto-pos_weight上限10保留，不能错误说D3也不加权）。除split/数据路径/隔离设置/注册seed外原数值不改。每臂8973步、K4、source计划8896+77、source-dev64、final_step，不用GT或源best选最终模型。仅允许原6.2数值回退。
7. 全部表A真实Train教师拟合和source_val_CE先推送，核验公开提交再启动四final-step学生完整Val推理。211085候选×17=3588445 margin/425表达，margin>0，逐文件SHA先封存，再独立CPU官方TrackEval/离线GT。
8. 复用v2历史参照Frozen8.218/D129.702/D327.768/B1_GLOBAL13.848/完整官方32.139（不是旧32.134）。全部表C/PresentHOTA/EmptyFP/GT帧k、同帧R@1/AUROC、指定数量×排序、25视频paired bootstrap2000次seed0全报；每臂成本单列。
9. 判定逐字沿用用户：s17−B1的95%CI下界>0且三个seed都>13.848，记“MLLM 存在性知识可跨视频、跨表达泛化”；CI包含0或任一seed≤13.848，记“同集增益未能泛化到未见视频和表达”。未覆盖情况如实描述，不新增规则；同时披露文本重合限制，不把字面判定当全表达未见证明。
10. G=三D1seedHOTA均值，保留率(G−8.218)/(29.702−8.218)，≥.7大部分保留，[.3,.7)部分保留，<.3基本未保留。不能据Val结果改模型/seed/阈值/上报臂。
11. 最多四物理GPU，显存准入、优先进程少的卡、允许共享；不停止/抢占外部任务。完成后停止，不做新方法/密集标注/调参。

历史标签计数更正：Naive8500缓存实际p≥.95为656；657来自PSR自己的8500。后续教师分布对比同时注明用户所给657/8500及这两个不同缓存，不能改原缓存来消除差异。
