# v6关键集合结果

<!-- V6_FINAL_ANALYSIS_BEGIN -->
## 最终科研结论与停止范围

五项完整预测和官方评测全部正常结束，不再运行GPU队列。修复不是全面成功：C2、C3相对v5恢复明显，C1显著下降；三个FULL点估计都高于YT_PRIOR/UNIFORM_K中较好者，但C1、C3对最强简单对照的95%区间均跨0，C2只有4视频、按预注册不做显著性判断。因此只能称方向性正结果，尚未证明优于简单教师校验的稳定新方法。

全10项运行前估84.058 GPU小时触发用户20小时回退，只运行C1/C2/C3 FULL及C1/C3 MINUS_SLOW。当前3/3点估计胜两对照不能代替“六方向至少四胜”；该门槛未评估，完整矩阵未授权。尚未运行FULL C4/C5/C6、MINUS_SLOW C4/C6，v5剩余80项继续不获授权。本阶段交付后停止等待，不擅自补项。

| pair | source | target | NATIVE | FULL_v5 | FULL_v6 | FULL_v6_minus_v5 | MINUS_SLOW_v6 | YT_PRIOR | UNIFORM_K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | V2-best | Dance | 8.218023 | 37.723955 | 26.367793 | -11.356162 | 28.175406 | 22.710562 | 24.107948 |
| C2 | Dance-best | KITTI-V2 | 13.501503 | 16.099630 | 26.818212 | 10.718582 |  | 24.285695 | 24.413313 |
| C3 | V2-best | SportsMOT | 50.970478 | 43.914165 | 59.341507 | 15.427342 | 59.620586 | 58.600791 | 40.804245 |


### 不确定性与慢速分支

下面区间按视频聚类2000次、seed0；KITTI仅点估计。完整区间也见comparison_intervals.csv/bootstrap.json。

| comparison | delta_HOTA | CI95 |
| --- | --- | --- |
| C1__FULL_v6_minus_NATIVE | 18.149770 | [9.99587381867082, 27.796862789381073] |
| C1__FULL_v6_minus_PRES_TOPK | 16.046910 | [9.122892608444, 23.73727376871764] |
| C1__FULL_v6_minus_FULL_v5 | -11.356162 | [-19.12888328471565, -2.4812255556611893] |
| C1__FULL_v6_minus_UNIFORM_K | 2.259845 | [-2.685933139567676, 7.9389498852924385] |
| C1__FULL_v6_minus_YT_PRIOR | 3.657231 | [-1.4957773018997378, 9.386628212789203] |
| C1__FULL_v6_minus_MINUS_SLOW_v6 | -1.807613 | [-7.831863421276841, 5.658636824891629] |
| C2__FULL_v6_minus_NATIVE | 13.316709 | 4 videos: point only |
| C2__FULL_v6_minus_PRES_TOPK | 13.036825 | 4 videos: point only |
| C2__FULL_v6_minus_FULL_v5 | 10.718582 | 4 videos: point only |
| C2__FULL_v6_minus_UNIFORM_K | 2.404899 | 4 videos: point only |
| C2__FULL_v6_minus_YT_PRIOR | 2.532517 | 4 videos: point only |
| C3__FULL_v6_minus_NATIVE | 8.371029 | [3.6666289142914543, 11.965309534678347] |
| C3__FULL_v6_minus_PRES_TOPK | 6.414046 | [1.7925254471625856, 9.965659484174443] |
| C3__FULL_v6_minus_FULL_v5 | 15.427342 | [12.001315448303222, 18.852638679511777] |
| C3__FULL_v6_minus_UNIFORM_K | 18.537262 | [13.292457193011732, 22.9237470106457] |
| C3__FULL_v6_minus_YT_PRIOR | 0.740716 | [-3.475928997705552, 3.89863587065353] |
| C3__FULL_v6_minus_MINUS_SLOW_v6 | -0.279079 | [-3.27629526949165, 2.696420626116388] |


FULL−MINUS_SLOW：C1为−1.808 [−7.832,5.659]，C3为−0.279 [−3.276,2.696]。两个点估计均不支持额外慢速收益，但区间跨0也不证明慢速必然有害。此消融包含LoRA、固定32帧f近似和后续获取/快校准反馈，并非单独权重的因果效应；等调用数也不是等提问集合。

C1的FULL比MINUS_SLOW多88,754行Query-Empty预测，Present预测少1,617行；Present HOTA却高2.530，不能只挑有目标组或AssA宣称成功。C3在相同v4 GT、原418序列ID敏感性下仍是MINUS_SLOW略高（60.058对59.689），不使用旧冲突GT模板。

### 修复到底改善了什么

高置信错负比例（GT正长轨迹对）C1从23.10%降到0.53%，C2从84.45%降到13.71%，C3从19.41%降到1.01%。但置信变软本身不等于恢复：C1 GT观测召回从59.39%掉到31.08%，C2从4.75%升到20.17%，C3从28.65%升到56.52%。C2仍有916个未问过的GT正长轨迹对被最终后验高置信判负；“未问伪标签为UNKNOWN”只保证不用于伪监督，不保证最终输出不会拒绝它们。

C1旧整句缺席标志误删5/40个有目标表达，v6降到0/40；代价是对385个真正空表达，正确打出缺席标志从257降到16（66.75%→4.16%）。这是算法显式absence标志覆盖率，不是整体预测存在性准确率；更保守的整句判空没有自动解决空表达误检。

下表使用已有封存后的离线IoU≥0.5最大基数匹配，绝非官方CLEAR/HOTA匹配替代。FP总变化严格分解为新增行FP−删除行FP+未变行匹配重分配残差，不能把新增行数或新增FP单独当作总FP变化。

| pair | method | GT_recall | matched_GT | output_rows | net_GT_vs_Native | new_FP_rows | removed_FP_rows | matching_reassignment_residual | net_FP_vs_Native |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | FULL_v6 | 0.310820 | 66062 | 200166 | 55725 | 121396 | 76666 | 7 | 44737 |
| C2 | FULL_v6 | 0.201720 | 99102 | 370093 | 73261 | 234289 | 37337 | -9 | 196943 |
| C3 | FULL_v6 | 0.565152 | 866760 | 1004405 | -32680 | 44094 | 221161 | -272 | -177339 |
| C1 | MINUS_SLOW_v6 | 0.303508 | 64508 | 113029 | 54171 | 32805 | 73657 | 6 | -40846 |
| C3 | MINUS_SLOW_v6 | 0.568968 | 872612 | 1012866 | -26828 | 46023 | 220422 | -331 | -174730 |


C3虽比v5恢复大量GT、比Native的HOTA高8.371，但相对Native仍净丢32,680个已匹配GT观测；其HOTA收益伴随大幅减FP，不能全部解释成新增语义恢复。C2相对Native净恢复73,261个GT观测，同时净增196,943个FP，仍有明显召回/误检折中。

### 仍存在的机制局限（本轮不修改方法）

C1仅2/25视频有LoRA更新，共150步。第五个视频后w变为0.171393，后20视频新a_q=0、ℓ0≤2、单次教师证据≤2，使第一次回答的正后验上界sigmoid(2w+2)≈0.912，达不到原>.95正伪标签门槛；实际这20视频2918个被问过的(q,track)对全部只问一次，无法用重复证据跨过门槛。这是独立GT-free的正监督断供证据，不证明全部指标下降都由此造成。MINUS_SLOW C1的w始终≥0.953612，无该全局上界阻断。

已沿用的存在性头实际只读margin（两斜率Bernoulli），PCD虽按要求FP32缓存但不输入此头。因此LoRA改变margin后仍会经存在性/数量、f、先验和采样反馈影响输出，“身份—存在性解耦”并不完全。它是设计局限与可能归因，不是本批已隔离验证的因果结论。32帧之外的最终缓存仍是视频起始版本，也不等同于最后模型全视频重推理。

DS删除IDENTITY并改Beta(2,2)后，估计仍偏乐观：例如C2 CROP特异度估计0.861、实际已问集合GT为0.726；C3 FRAME_BOX为0.767对0.634。不同视图来自同一冻结模型，不能保证条件独立。v5/v6所问候选和视图混合不同，不能把准确率变化单独归因于Beta先验；完整表同时给关键帧与轨迹多数两种GT口径，仅覆盖实际提问集合。

历史slow日志没保存实际优化输入类别计数，轮末positive_pairs=0不代表优化时只用了负例。C3的2300步属于“轮末零正标签的轮”，真实输入正负比例未知；报告保留此前误用的更正记录，不通过重跑或臆测补造样本统计。

### 可信边界与实现核验

五项共2644个episode、2,700,559行预测已逐行核验框/ID/帧号来自固定MOT；预测封存后才独立评测，运行及教师GT打开尝试均0；旧v5全部seal不变。在原worker的NumPy 2.2.6下，144份视频belief的f→截断logit逐位一致，w范围、证据累加上限、三段判空、DS配置、全视频起始/固定32帧覆盖、FP32 margin/PCD均通过。首个离线审计误用NumPy1.26导致log末位差异，切回原2.2.6环境即通过原精确门槛，未改数组/预测/数值代码。原46项定向回归和3项最终FP归因检查共49项通过；同源历史缓存路径并不逐位相同，首视频原生符号无差异仅为局部抽查，不能扩展成全库PRES_TOPK等价。

本设定为冻结教师辅助的video-transductive适应，有全视频信息且Val已多次开发；不是严格帧因果或盲测。视频bootstrap条件于固定视频流，不覆盖重排视频后的重新适应与随机seed不确定性。没有按目标GT改参数、阈值或挑checkpoint。

### 实测成本、封存日志与恢复

五任务逻辑教师调用34,639次，其中新调用18,231、缓存命中16,408；另有真实smoke3次。新教师forward合计0.7390h、LoRA 1.3237h（3600步）、成功打分相位10.5749h。已分配共享worker累计wall 13.1743h，含首次失败0.2485h；恢复后manager wall 7.3274h含CPU评测/报告，这些都不是GPU利用率积分，也不把教师子进程wall再次加到worker总时长。失败首轮未持久化的独立GPU打分相位时间仍未知，不伪造为0。

逻辑调用相对v5的FULL只变0.695%/0%/0%，均未触发严格>5%的YT_PRIOR补跑规则；当前两对照按用户要求复用旧结果，不是新逐视频精确预算匹配。YT_PRIOR作为独立系统仍需原生模型/MOT推理，其部署成本不能写成CPU读缓存耗时；历史源头/候选/特征准备和原完整教师响应成本单列，不计作本轮新GPU时长。共享卡负载与SKIP/缓存均不同，不能从保守估84小时降到当前实耗推导纯算法加速。

完整关闭日志26,299,938字节，SHA256 `38465803d5d698580c4d1590e142640be0a68808c6a1e41fbcda171550faabe5`；原始教师响应32,082个唯一输入键，压缩8,276,188字节、SHA256 `bbac8af937567d564540e2b886b77ec2172352ab77e02b11a58f2f198a066b73`。实际逐个解码核对840个日志成员哈希及全部教师输入键/冻结权重；原BF16 Yes/No softmax精确复现，额存FP32概率只作诊断、未替换原监督概率。所有launcher日志均归档，包括最初类型兼容失败与观察连接重接；原日志/缓存/checkpoint未删。归档文件名与每项文件SHA见logs.json，若提供.parts按manifest顺序原字节重组即可，不能重新压缩后冒充原SHA。

运行根目录 `/data2/user/FlexHook-TTA/results/runs/presence_active_20261102/redesign_v6`链接至已授权data1资产；每个`active/C*/{FULL,MINUS_SLOW}/rho_0.3`内保存预测seal、逐视频receipt/questions/belief、frame_cache及stream检查点。FULL最后stream分别C1 `stream_024.pt`、C2 `stream_003.pt`、C3 `stream_044.pt`；完整路径/SHA见artifacts.json和logs.json，不是启动新任务的命令。execution.json、registered_jobs.json与PREREG保留不变。

本页为最终结论；interim_official_results.*明确仅保留当时4/5完成的历史快照，当前状态请读final_status.json、final_official_results.*、queue_completion.json和validation.json。

<!-- V6_FINAL_ANALYSIS_END -->

FIVE_JOB_COST_GATE_STAGE_ONLY。只报告实际完成方向，不以3方向替代6方向验收。

原10项保守估算84.058 GPU小时>20，故按用户指令只运行C1/C2/C3 FULL和C1/C3 MINUS_SLOW。
v5剩余80项及本轮未授权方向未启动；不根据目标GT选择参数、checkpoint或扩展范围。

| pair | source | target | NATIVE | PRES_TOPK | FULL_v5 | FULL_v6 | FULL_v6_minus_v5 | MINUS_SLOW_v6 | YT_PRIOR | UNIFORM_K | above_both_controls | teacher_calls | YT_PRIOR_rematched |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | V2-best | Dance | 8.218023 | 10.320883 | 37.723955 | 26.367793 | -11.356162 | 28.175406 | 22.710562 | 24.107948 | True | 3334 | False |
| C2 | Dance-best | KITTI-V2 | 13.501503 | 13.781387 | 16.099630 | 26.818212 | 10.718582 | None | 24.285695 | 24.413313 | True | 20955 | False |
| C3 | V2-best | SportsMOT | 50.970478 | 52.927461 | 43.914165 | 59.341507 | 15.427342 | 59.620586 | 58.600791 | 40.804245 | True | 3508 | False |


## 完整指标

| pair | method | HOTA | DetA | AssA | IDF1 | IDSW | prediction_rows | query_empty_prediction_rows | Present_HOTA | teacher_calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | FULL_v6 | 26.368000 | 17.461000 | 39.849000 | 20.446000 | 528.000000 | 200166 | 121762 | 32.645537 | 3334 |
| C2 | FULL_v6 | 26.818000 | 11.858000 | 60.794000 | 21.548000 | 574.000000 | 370093 | 0 | 26.818212 | 20955 |
| C3 | FULL_v6 | 59.342000 | 48.646000 | 72.392000 | 58.532000 | 1578.000000 | 1004405 | 0 | 59.341507 | 3508 |
| C1 | MINUS_SLOW_v6 | 28.175000 | 22.490000 | 35.314000 | 23.597000 | 403.000000 | 113029 | 33008 | 30.115621 | 3334 |
| C3 | MINUS_SLOW_v6 | 59.621000 | 48.928000 | 72.654000 | 59.956000 | 1480.000000 | 1012866 | 0 | 59.620586 | 3508 |


## 高置信错负（最终posterior，非伪标签）

| pair | version | GT_positive_long_track_pairs | high_confident_negative_GT_positive_pairs | high_negative_rate_among_GT_positive | high_negative_unasked_GT_positive_pairs | unasked_fraction_within_high_negative_GT_positive | GT_present_queries | inferred_absent_queries | GT_present_but_inferred_absent | retrievable_GT_but_inferred_absent | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | v5 | 567 | 131 | 0.231041 | 106 | 0.809160 | 40 | 262 | 5 | 5 | posterior diagnosis, not masked optimizer pseudo labels |
| C1 | v6 | 567 | 3 | 0.005291 | 0 | 0.000000 | 40 | 16 | 0 | 0 | posterior diagnosis, not masked optimizer pseudo labels |
| C2 | v5 | 8441 | 7128 | 0.844450 | 6219 | 0.872475 | 862 | 382 | 382 | 376 | posterior diagnosis, not masked optimizer pseudo labels |
| C2 | v6 | 8441 | 1157 | 0.137069 | 916 | 0.791703 | 862 | 0 | 0 | 0 | posterior diagnosis, not masked optimizer pseudo labels |
| C3 | v5 | 5528 | 1073 | 0.194103 | 1019 | 0.949674 | 466 | 2 | 2 | 2 | posterior diagnosis, not masked optimizer pseudo labels |
| C3 | v6 | 5528 | 56 | 0.010130 | 0 | 0.000000 | 466 | 0 | 0 | 0 | posterior diagnosis, not masked optimizer pseudo labels |


## GT召回与预测行数

| pair | version | GT_recall | GT_observations | matched_GT_observations | output_rows | new_FP | removed_FP | net_GT_recovery | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | v5 | 0.593881 | 212541 | 126224 | 163608 | 36455 | 88449 | 115887 | offline IoU>=.5 max-cardinality, not official CLEAR |
| C1 | v6 | 0.310820 | 212541 | 66062 | 200166 | 121396 | 76666 | 55725 | offline IoU>=.5 max-cardinality, not official CLEAR |
| C2 | v5 | 0.047530 | 491286 | 23351 | 38433 | 9560 | 68526 | -2490 | offline IoU>=.5 max-cardinality, not official CLEAR |
| C2 | v6 | 0.201720 | 491286 | 99102 | 370093 | 234289 | 37337 | 73261 | offline IoU>=.5 max-cardinality, not official CLEAR |
| C3 | v5 | 0.286546 | 1533675 | 439469 | 485840 | 12615 | 280696 | -459971 | offline IoU>=.5 max-cardinality, not official CLEAR |
| C3 | v6 | 0.565152 | 1533675 | 866760 | 1004405 | 44094 | 221161 | -32680 | offline IoU>=.5 max-cardinality, not official CLEAR |


## 教师与适应成本

| pair | method | rho | logical_calls | new_calls | cache_hits | teacher_forward_GPU_hours | committed_teacher_equivalent_forward_GPU_hours | LoRA_GPU_hours | rescore_GPU_hours | seal_process_wall_hours | assigned_worker_wall_hours_including_failures | failed_worker_attempts | unfinished_attempt_wall_unknown | wall_not_GPU_utilization_integral | incomplete_failed_score_phase_not_fabricated_as_zero | peak_teacher_allocated_bytes | peak_parent_allocated_bytes | trainable_parameters_observed_in_updates | optimizer_updates | frame_prior_CPU_hours |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | FULL_v6 | 0.300000 | 3334 | 1705 | 1629 | 0.036914 | 0.053967 | 0.087951 | 2.251601 | 2.410984 | 2.498493 | 1 | 0 | True | True | 4678529024 | 4576561664 | [24576] | 150 | 0.009536 |
| C2 | FULL_v6 | 0.300000 | 20955 | 12199 | 8756 | 0.534667 | 0.736938 | 0.279548 | 1.743927 | 2.693393 | 2.854713 | 1 | 0 | True | True | 5471136256 | 4589635072 | [24576] | 450 | 0.010468 |
| C3 | FULL_v6 | 0.300000 | 3508 | 2294 | 1214 | 0.084732 | 0.108390 | 0.956160 | 1.908212 | 2.991270 | 2.991437 | 0 | 0 | True | True | 5906143232 | 4580624896 | [24576] | 3000 | 0.008722 |
| C1 | MINUS_SLOW_v6 | 0.300000 | 3334 | 653 | 2681 | 0.025412 | 0.050683 | 0.000000 | 2.952326 | 3.012107 | 3.012212 | 0 | 0 | True | True | 4678529024 | 4566034432 | [] | 0 | 0.009078 |
| C3 | MINUS_SLOW_v6 | 0.300000 | 3508 | 1380 | 2128 | 0.057277 | 0.114456 | 0.000000 | 1.718845 | 1.817305 | 1.817479 | 0 | 0 | True | True | 5906143232 | 4573455360 | [] | 0 | 0.007034 |


## 教师回答到后验的证据利用（GT-free，不等于训练标签）

| pair | method | teacher_yes | teacher_no | immediate_confident_positive_after_yes | immediate_confident_negative_after_no | teacher_yes_still_posterior_nonpositive | teacher_yes_still_nonpositive_fraction | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | FULL_v6 | 464 | 2870 | 5 | 203 | 27 | 0.058190 | GT-free immediate post-answer posterior; not final labels, training samples, or semantic correctness |
| C2 | FULL_v6 | 7211 | 13744 | 650 | 2211 | 549 | 0.076134 | GT-free immediate post-answer posterior; not final labels, training samples, or semantic correctness |
| C3 | FULL_v6 | 2526 | 982 | 72 | 256 | 358 | 0.141726 | GT-free immediate post-answer posterior; not final labels, training samples, or semantic correctness |
| C1 | MINUS_SLOW_v6 | 673 | 2661 | 48 | 1976 | 193 | 0.286776 | GT-free immediate post-answer posterior; not final labels, training samples, or semantic correctness |
| C3 | MINUS_SLOW_v6 | 2547 | 961 | 5 | 19 | 50 | 0.019631 | GT-free immediate post-answer posterior; not final labels, training samples, or semantic correctness |


## 证据与更新覆盖（GT-free，不是学习有效性的指标）

| pair | method | videos | zero_slow_update_videos | optimizer_updates | rounds | rounds_without_final_positive_pseudo_pairs | zero_final_positive_update_rounds | optimizer_updates_in_zero_final_positive_rounds | optimizer_input_class_counts_recorded | min_final_w | max_final_w | new_videos_single_answer_cannot_certify_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | FULL_v6 | 25 | 23 | 150 | 100 | 97 | 0 | 0 | False | 0.171393 | 1.000000 | 20 |
| C2 | FULL_v6 | 4 | 0 | 450 | 16 | 7 | 0 | 0 | False | 1.000000 | 1.000000 | 0 |
| C3 | FULL_v6 | 45 | 26 | 3000 | 180 | 166 | 46 | 2300 | False | 1.000000 | 1.000000 | 0 |
| C1 | MINUS_SLOW_v6 | 25 | 25 | 0 | 100 | 77 | 0 | 0 | False | 0.953612 | 1.000000 | 0 |
| C3 | MINUS_SLOW_v6 | 45 | 45 | 0 | 180 | 178 | 0 | 0 | False | 0.000000 | 1.000000 | 16 |


新视频a_q初始为0、ℓ0≤2且单次证据≤2，因此身份重拟合之前，首次回答的正后验上界为sigmoid(2w+2)。此上界若≤0.95，单次回答无法生成正伪标签，需要重复证据；它不是教师正确率或新的在线阈值。optimizer_updates_in_zero_final_positive_rounds只按轮末零正标签分组，不能解释为优化时只用了负例。旧slow记录未保存实际输入类别计数，真实正负优化样本比例未知；轮末f/后验可能已在LoRA及重校准后变化，不用最终belief冒充训练标签。原二元损失确实允许只用已问负例，但本批不据此倒推哪些实际步骤为负例独占，不临时改损失或补跑。

## SportsMOT 418条敏感性（v4内容不变，仅限制序列ID）

| pair | method | full466_HOTA | HOTA_unrounded | DetA | AssA | IDF1 | IDSW | episodes | prediction_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C3 | FULL_v6 | 59.341507 | 59.688924 | 49.275000 | 72.308000 | 58.993000 | 1576.000000 | 418 | 983814 |
| C3 | MINUS_SLOW_v6 | 59.620586 | 60.057707 | 49.698000 | 72.580000 | 60.569000 | 1466.000000 | 418 | 976166 |


## DS估计与事后GT（实际提问集合，轨迹多数口径）

| pair | version | worker | specificity_DS | specificity_GT | sensitivity_DS | sensitivity_GT | observations | GT_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | v5 | CROP | 0.964476 | 0.769485 | 0.809584 | 0.849887 | 3311.000000 | TRACK_MAJORITY_50_PERCENT |
| C1 | v6 | CROP | 0.969884 | 0.916208 | 0.870032 | 0.843621 | 3334 | TRACK_MAJORITY_50_PERCENT |
| C2 | v5 | CROP | 0.867061 | 0.557696 | 0.872403 | 0.809536 | 15249.000000 | TRACK_MAJORITY_50_PERCENT |
| C2 | v5 | CLIP | 0.824673 | 0.698885 | 0.694090 | 0.680328 | 1589.000000 | TRACK_MAJORITY_50_PERCENT |
| C2 | v5 | FRAME_BOX | 0.760280 | 0.618625 | 0.769493 | 0.834971 | 4117.000000 | TRACK_MAJORITY_50_PERCENT |
| C2 | v6 | CROP | 0.860645 | 0.726312 | 0.880608 | 0.690346 | 14597 | TRACK_MAJORITY_50_PERCENT |
| C2 | v6 | CLIP | 0.829825 | 0.759551 | 0.744104 | 0.721116 | 2031 | TRACK_MAJORITY_50_PERCENT |
| C2 | v6 | FRAME_BOX | 0.719965 | 0.675432 | 0.700002 | 0.836299 | 4327 | TRACK_MAJORITY_50_PERCENT |
| C3 | v5 | CROP | 0.735102 | 0.506329 | 0.850267 | 0.896047 | 2523.000000 | TRACK_MAJORITY_50_PERCENT |
| C3 | v5 | CLIP | 0.772332 | 0.785714 | 0.675636 | 0.863636 | 36.000000 | TRACK_MAJORITY_50_PERCENT |
| C3 | v5 | FRAME_BOX | 0.342434 | 0.549261 | 0.658493 | 0.819521 | 949.000000 | TRACK_MAJORITY_50_PERCENT |
| C3 | v6 | CROP | 0.776345 | 0.696839 | 0.848266 | 0.910339 | 1934 | TRACK_MAJORITY_50_PERCENT |
| C3 | v6 | FRAME_BOX | 0.767175 | 0.634096 | 0.896921 | 0.925892 | 1574 | TRACK_MAJORITY_50_PERCENT |


## 首视频未更新缓存与历史源分数（CPU抽查）

| pair | unchanged_initial_model_rows | excluded_LoRA_refreshed_observations | bitwise_equal | max_absolute_error | mean_absolute_error | margin_positive_disagreements |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | 72301 | 118 | False | 0.000922 | 0.000099 | 0 |
| C2 | 186362 | 118 | False | 0.000008 | 0.000000 | 0 |


当前与历史评分路径不逐位一致，原因未单独归因；抽查原生符号判定无差异不代表全库或PRES_TOPK等价。

DS完整估计/实测特异度与敏感度、两种GT口径见reliability_vs_GT.csv；v5相同表保留于../redesign_v5/active_map_v1_critical28/reliability_vs_GT.csv。
视频聚类bootstrap2000次seed0，KITTI仅4视频点估计；条件于既定有序流，不覆盖重新适应/训练seed不确定性。
本设定冻结教师辅助、video-transductive，Val历史开发复用，非盲测/严格帧因果。
缓存含视频起始全帧与LoRA后固定32帧版本，非所有帧最后模型；PCD为FP32，但原两标量存在性头只读margin。
GT仅在完整预测封存后独立CPU评测；未问伪标签UNKNOWN是规则，不是语义正确率。
teacher_absent是算法依据若干候选No与模型存在性推断的缺席标志，并非教师审阅完整视频后的全局不存在结论。
调用变化>5%才按v6逐视频预算补跑YT_PRIOR，未触发者复用v5且明确非新逐视频精确匹配。
