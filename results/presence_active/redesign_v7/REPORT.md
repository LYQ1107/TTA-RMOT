# v7：在线EB存在性与慢速分支公平检验

本轮成本回退授权的12项已全部完成，科研验收**未通过**。MINUS_SLOW在5/6方向点胜两简单对照，但C6未满足Dance保留条件；FULL只获授权运行3方向，六方向为NOT_EVALUATED，且已测C1也低于必要保留门槛。不跨方法逐格挑最优，不申请预算曲线或消融。

当前已封存并完成官方评测 12/12 个主项；原授权成本回退集合全部完成；未运行的FULL C2/C5/C6仍为NA，不能当作六方向FULL验收。

| method | six_directions_complete | directions_above_both_controls | Dance_both_within_two_of_v5 | result |
| --- | --- | --- | --- | --- |
| FULL_v7 | False | 3 | False | NOT_EVALUATED |
| MINUS_SLOW_v7 | True | 5 | False | FAIL |

## 0. 最终科学判读

最强的真实正向结果是C4 FULL：HOTA61.378753，相对YT_PRIOR的+15.279814，45视频聚类95% CI[12.244116,18.021758]；相对MINUS+15.721841，CI[11.158589,20.752579]。它说明完整反馈路径在这个方向有收益，不等于普遍解决跨域RMOT。

慢速分支的固定证据检验没有支持额外收益：C1回放−MINUS为−1.200113（CI跨0）；C3为−3.242482，CI[−4.950690,−1.412725]；C4为−40.544277，CI[−48.254527,−34.905244]。FULL同时改变重打分数值路径、f、选题和快校准，不能把C4 FULL的全部收益称为独立LoRA增益；REPLAY冻结问答，但仍包含LoRA引起的f/快速/EB下游变化。

C4回放不是漏跑或恢复损坏：466条序列完整封存，45视频均有实际优化，9000步使用0个正/288000个负BCE样本；预测仅6537行，IoU≥.5召回.004206。365/466个有目标句被判为空，其中235个没有存在性问题回答（不等于从未问过任何身份问题）；GT正长轨迹高置信错负4735/5528，含未问3581。未问对的训练标签仍为UNKNOWN；整句拒绝会把输出后验置零，不能将这些数值误写成未问负监督。

该回放从第13视频起，ρ̂已使最高排名弱证据也无法越过.1，后33视频/334表达全部零输出；期间固定回放仍摄入2905条原回答，但不能重开终止判空。此状态早于第37视频的磁盘中断；恢复前36视频的72个receipt/checkpoint哈希全部不变。存在性拒绝自强化、正例断供和负例独占更新共同出现，未做单因素消融，不把最初触发或全部差值单因归于其中某一项。

回放与MINUS的原始问答、每轮DS状态完全一致，不能把两臂差异归因为换了教师答案或另一套DS数值。事后DS也未充分校准：例如C4 FRAME_BOX特异度DS=.826275，实际提问集合的轨迹多数GT口径=.583208（MINUS与回放相同）。相关看法和有选择的抽样是共同限制，不把此事后准确率推广到全候选库。

MINUS的C6 HOTA30.204325，比v5 FULL36.468777低6.264451，低于34.469；5个点胜不能抵消预注册失败。FULL的C1为35.166028，亦低于35.724。这里的失败结论不抹掉C4正结果，也不以该正结果覆盖C3/C4回放的负结果。

本批新增已测GPU相位合计27.727046小时，是包含一个未测完失败打分相位的下界；调度壁钟15.348615小时。教师实际新调用30253、缓存命中34381、固定回放10280；继承冻结打分/Stage A的成本没有重测，不等于零。20小时用于选择预先固定的12项成本回退范围，不将其误称为这12项的硬时间上限。

## 1. 六方向主表（未运行格为NA，不是0）

| pair | source | target | NATIVE | PRES_TOPK | FULL_v5 | FULL_v6 | FULL_v7 | MINUS_SLOW_v7 | SLOW_REPLAY_v7 | YT_PRIOR | UNIFORM_K | YT_PRIOR_rematch_pending |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | V2-best | Dance | 8.218023 | 10.320883 | 37.723955 | 26.367793 | 35.166028 | 38.849551 | 37.649438 | 22.710562 | 24.107948 | False |
| C2 | Dance-best | KITTI-V2 | 13.501503 | 13.781387 | 16.099630 | 26.818212 | NA | 25.313445 | NA | 24.285695 | 24.413313 | False |
| C3 | V2-best | SportsMOT | 50.970478 | 52.927461 | 43.914165 | 59.341507 | 65.133068 | 65.345262 | 62.102780 | 58.600791 | 40.804245 | False |
| C4 | Dance-best | SportsMOT | 20.559603 | 21.010887 | 32.191307 | NA | 61.378753 | 45.656912 | 5.112636 | 46.098939 | 40.804245 | False |
| C5 | LaMOT-best | KITTI-V2 | 22.398306 | 23.875533 | 18.315360 | NA | NA | 30.331540 | NA | 27.588195 | 25.674444 | False |
| C6 | LaMOT-best | Dance | 15.475743 | 15.151263 | 36.468777 | NA | NA | 30.204325 | NA | 20.821630 | 24.107948 | False |

## 2. 完整官方指标

| pair | method | HOTA_unrounded | DetA | AssA | IDF1 | IDSW | prediction_rows | query_empty_prediction_rows | Present_HOTA | teacher_calls | GT_open_attempts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | MINUS_SLOW_v7 | 38.849551 | 44.840000 | 33.728000 | 37.332000 | 1386.000000 | 183380 | 18477 | 40.248855 | 3264 | 0 |
| C1 | FULL_v7 | 35.166028 | 37.435000 | 33.111000 | 32.586000 | 1327.000000 | 189470 | 35767 | 37.559187 | 3195 | 0 |
| C1 | SLOW_REPLAY_v7 | 37.649438 | 43.592000 | 32.597000 | 35.317000 | 1372.000000 | 177705 | 24573 | 39.495566 | 3264 | 0 |
| C2 | MINUS_SLOW_v7 | 25.313445 | 11.013000 | 58.264000 | 19.629000 | 701.000000 | 155932 | 0 | 25.313445 | 20955 | 0 |
| C3 | MINUS_SLOW_v7 | 65.345262 | 60.290000 | 70.837000 | 65.297000 | 2848.000000 | 1517412 | 0 | 65.345262 | 3508 | 0 |
| C3 | FULL_v7 | 65.133068 | 59.881000 | 70.858000 | 64.876000 | 3195.000000 | 1635162 | 0 | 65.133068 | 3508 | 0 |
| C3 | SLOW_REPLAY_v7 | 62.102780 | 54.194000 | 71.175000 | 60.913000 | 4114.000000 | 2299972 | 0 | 62.102780 | 3508 | 0 |
| C4 | MINUS_SLOW_v7 | 45.656912 | 30.011000 | 69.461000 | 40.131000 | 1480.000000 | 815058 | 0 | 45.656912 | 3508 | 0 |
| C4 | FULL_v7 | 61.378753 | 53.603000 | 70.290000 | 60.107000 | 2777.000000 | 1328506 | 0 | 61.378753 | 3508 | 0 |
| C4 | SLOW_REPLAY_v7 | 5.112636 | 0.400590 | 65.278000 | 0.706270 | 6.000000 | 6537 | 0 | 5.112636 | 3508 | 0 |
| C5 | MINUS_SLOW_v7 | 30.331540 | 16.362000 | 56.330000 | 27.606000 | 2746.000000 | 1125042 | 0 | 30.331540 | 20955 | 0 |
| C6 | MINUS_SLOW_v7 | 30.204325 | 27.982000 | 32.653000 | 26.355000 | 840.000000 | 111717 | 13144 | 31.024575 | 2233 | 0 |

## 3. 存在性与在线比例

| pair | method | query_present_episodes | query_empty_episodes | correctly_inferred_empty | present_incorrectly_inferred_empty | empty_rejected_without_presence_answers | present_rejected_without_presence_answers | query_empty_prediction_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | FULL_v7 | 40 | 385 | 302 | 9 | 95 | 1 | 35767 |
| C1 | MINUS_SLOW_v7 | 40 | 385 | 293 | 6 | 96 | 1 | 18477 |
| C1 | SLOW_REPLAY_v7 | 40 | 385 | 302 | 11 | 93 | 0 | 24573 |
| C2 | MINUS_SLOW_v7 | 862 | 0 | 0 | 296 | 0 | 0 | 0 |
| C3 | FULL_v7 | 466 | 0 | 0 | 52 | 0 | 0 | 0 |
| C3 | MINUS_SLOW_v7 | 466 | 0 | 0 | 2 | 0 | 0 | 0 |
| C3 | SLOW_REPLAY_v7 | 466 | 0 | 0 | 0 | 0 | 0 | 0 |
| C4 | FULL_v7 | 466 | 0 | 0 | 67 | 0 | 0 | 0 |
| C4 | MINUS_SLOW_v7 | 466 | 0 | 0 | 4 | 0 | 0 | 0 |
| C4 | SLOW_REPLAY_v7 | 466 | 0 | 0 | 365 | 0 | 235 | 0 |
| C5 | MINUS_SLOW_v7 | 862 | 0 | 0 | 12 | 0 | 0 | 0 |
| C6 | MINUS_SLOW_v7 | 40 | 385 | 274 | 13 | 218 | 12 | 13144 |

![rho stream](rho_hat_stream.png)

![pi stream](pi_hat_stream.png)

逐视频先验与软计数见eb_stream.csv/原presence.json；逐句未截断/截断LL和触发次数见presence_likelihood.csv。停止概率与按最后DS重算概率分别记录，不冒充同一个时点。
presence_decisions.csv中的teacher_absent/teacher_confirmed_present是继承的字段名，v7实际表示EB的终止判空/判存在，不是教师直接审阅完整视频的标注；无分层回答而仅由先验触发的拒绝另行列出。

## 4. 高置信错负与真实恢复

| pair | method | matched_candidate_positive_tracks_majority | confident_negative_GT_FN | confident_negative_GT_FN_unqueried | GTpositive_high_negative_rate |
| --- | --- | --- | --- | --- | --- |
| C1 | FULL_v7 | 567 | 135 | 121 | 0.238095 |
| C1 | MINUS_SLOW_v7 | 567 | 86 | 76 | 0.151675 |
| C1 | SLOW_REPLAY_v7 | 567 | 129 | 49 | 0.227513 |
| C2 | MINUS_SLOW_v7 | 8441 | 2508 | 2122 | 0.297121 |
| C3 | FULL_v7 | 5528 | 291 | 221 | 0.052641 |
| C3 | MINUS_SLOW_v7 | 5528 | 27 | 1 | 0.004884 |
| C3 | SLOW_REPLAY_v7 | 5528 | 25 | 0 | 0.004522 |
| C4 | FULL_v7 | 5528 | 331 | 243 | 0.059877 |
| C4 | MINUS_SLOW_v7 | 5528 | 47 | 18 | 0.008502 |
| C4 | SLOW_REPLAY_v7 | 5528 | 4735 | 3581 | 0.856548 |
| C5 | MINUS_SLOW_v7 | 8441 | 145 | 50 | 0.017178 |
| C6 | MINUS_SLOW_v7 | 567 | 271 | 271 | 0.477954 |

此表高置信错负仍用v6的p<.05以可比；实际伪监督阈值为.15/.85，且未问始终UNKNOWN。最终posterior不是实际优化输入。

v7组合prior截断到[-2,2]且未问对evidence=0，因此未判空句子的未问对posterior至少为sigmoid(-2)≈.1192；本表未问p<.05来自整句判空置零，不是未问对被赋了训练负标签。

| pair | method | GT_observations | matched_GT | GT_recall | output_rows | recovered_GT | lost_GT | net_GT_vs_Native | new_FP_rows | removed_FP_rows | unchanged_rows_new_FP | unchanged_rows_removed_FP | net_FP_vs_Native | definition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | MINUS_SLOW | 212541 | 133456 | 0.627907 | 183380 | 124690 | 1571 | 123119 | 46430 | 85885 | 12 | 0 | -39443 | offline max-cardinality framewise IoU>=0.5 assignment, not official HOTA matching |
| C1 | FULL | 212541 | 119390 | 0.561727 | 189470 | 115259 | 6206 | 109053 | 69648 | 88945 | 10 | 0 | -19287 | offline max-cardinality framewise IoU>=0.5 assignment, not official HOTA matching |
| C1 | SLOW_REPLAY | 212541 | 128714 | 0.605596 | 177705 | 123491 | 5114 | 118377 | 48838 | 89225 | 11 | 0 | -40376 | offline max-cardinality framewise IoU>=0.5 assignment, not official HOTA matching |
| C2 | MINUS_SLOW | 491286 | 70575 | 0.143654 | 155932 | 53656 | 8922 | 44734 | 63808 | 52499 | 6 | 6 | 11309 | offline max-cardinality framewise IoU>=0.5 assignment, not official HOTA matching |
| C3 | MINUS_SLOW | 1533675 | 1197247 | 0.780639 | 1517412 | 348657 | 50850 | 297807 | 170541 | 165288 | 221 | 293 | 5181 | offline max-cardinality framewise IoU>=0.5 assignment, not official HOTA matching |
| C3 | FULL | 1533675 | 1239667 | 0.808298 | 1635162 | 438007 | 97780 | 340227 | 303144 | 222971 | 504 | 166 | 80511 | offline max-cardinality framewise IoU>=0.5 assignment, not official HOTA matching |
| C3 | SLOW_REPLAY | 1533675 | 1408416 | 0.918328 | 2299972 | 549072 | 40096 | 508976 | 738717 | 162874 | 802 | 73 | 576572 | offline max-cardinality framewise IoU>=0.5 assignment, not official HOTA matching |
| C4 | MINUS_SLOW | 1533675 | 567570 | 0.370072 | 815058 | 458089 | 37612 | 420477 | 230396 | 48344 | 78 | 124 | 182006 | offline max-cardinality framewise IoU>=0.5 assignment, not official HOTA matching |
| C4 | FULL | 1533675 | 1044451 | 0.681012 | 1328506 | 925113 | 27755 | 897358 | 266873 | 48378 | 97 | 19 | 218573 | offline max-cardinality framewise IoU>=0.5 assignment, not official HOTA matching |
| C4 | SLOW_REPLAY | 1533675 | 6450 | 0.004206 | 6537 | 1441 | 142084 | -140643 | 29 | 65423 | 0 | 1 | -65395 | offline max-cardinality framewise IoU>=0.5 assignment, not official HOTA matching |
| C5 | MINUS_SLOW | 491286 | 247921 | 0.504637 | 1125042 | 165312 | 9922 | 155390 | 721812 | 85035 | 95 | 42 | 636830 | offline max-cardinality framewise IoU>=0.5 assignment, not official HOTA matching |
| C6 | MINUS_SLOW | 212541 | 76869 | 0.361667 | 111717 | 38637 | 29062 | 9575 | 21341 | 429175 | 78 | 28 | -407784 | offline max-cardinality framewise IoU>=0.5 assignment, not official HOTA matching |

GT召回是独立IoU≥.5最大基数匹配，不替代官方TrackEval；新增/删除预测行的FP与不变行匹配重分配分列。

## 5. 慢速更新、成本与成对差值

| pair | method | rho | logical_questions | new_teacher_calls | cache_hits | replayed_answers | teacher_forward_GPU_hours | teacher_equivalent_GPU_hours | replay_teacher_cost_in_MINUS_dependency | LoRA_GPU_hours | rescore_GPU_hours | assigned_worker_wall_hours | failed_worker_attempts | worker_wall_includes_estimated_attempts | incomplete_failed_rescore_GPU_phases | peak_parent_allocated_bytes | peak_teacher_allocated_bytes | trainable_parameters_observed_in_updates | optimizer_updates | videos_with_LoRA_updates | wall_is_not_GPU_utilization_integral |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | MINUS_SLOW | 0.300000 | 3264 | 595 | 2669 | 0 | 0.012809 | 0.042194 | False | 0.000000 | 0.000000 | 0.040861 | 0 | False | 0 | None | 4472145920 | [] | 0 | 0 | True |
| C1 | FULL | 0.300000 | 3195 | 572 | 2623 | 0 | 0.034324 | 0.064342 | False | 2.415356 | 2.544065 | 5.037107 | 0 | False | 0 | 4578221056 | 4522808832 | [24576] | 3900 | 23 | True |
| C1 | SLOW_REPLAY | 0.300000 | 3264 | 0 | 0 | 3264 | 0.000000 | 0.042194 | True | 1.659143 | 2.147413 | 3.830350 | 0 | False | 0 | 4575358464 | None | [24576] | 3050 | 18 | True |
| C2 | MINUS_SLOW | 0.300000 | 20955 | 9477 | 11478 | 0 | 0.318490 | 0.646320 | False | 0.000000 | 0.000000 | 0.428484 | 0 | False | 0 | None | 5469817344 | [] | 0 | 0 | True |
| C3 | MINUS_SLOW | 0.300000 | 3508 | 1720 | 1788 | 0 | 0.039901 | 0.090459 | False | 0.000000 | 0.000000 | 0.075097 | 0 | False | 0 | None | 5899776000 | [] | 0 | 0 | True |
| C3 | FULL | 0.300000 | 3508 | 1761 | 1747 | 0 | 0.043687 | 0.089526 | False | 3.070007 | 1.315566 | 4.477327 | 0 | False | 0 | 4582467584 | 5904013312 | [24576] | 7000 | 40 | True |
| C3 | SLOW_REPLAY | 0.300000 | 3508 | 0 | 0 | 3508 | 0.000000 | 0.090459 | True | 2.191930 | 1.776948 | 3.988956 | 0 | False | 0 | 4582467584 | None | [24576] | 4550 | 26 | True |
| C4 | MINUS_SLOW | 0.300000 | 3508 | 1234 | 2274 | 0 | 0.027750 | 0.081677 | False | 0.000000 | 0.000000 | 0.061267 | 0 | False | 0 | None | 5896557568 | [] | 0 | 0 | True |
| C4 | FULL | 0.300000 | 3508 | 1258 | 2250 | 0 | 0.027714 | 0.072779 | False | 3.510564 | 1.604821 | 5.185544 | 0 | False | 0 | 4583483392 | 5255084032 | [24576] | 8400 | 44 | True |
| C4 | SLOW_REPLAY | 0.300000 | 3508 | 0 | 0 | 3508 | 0.000000 | 0.081677 | True | 3.225285 | 1.488657 | 4.749047 | 1 | True | 1 | 4576369664 | None | [24576] | 9000 | 45 | True |
| C5 | MINUS_SLOW | 0.300000 | 20955 | 13042 | 7913 | 0 | 0.247136 | 0.465638 | False | 0.000000 | 0.000000 | 0.355100 | 0 | False | 0 | None | 5474210304 | [] | 0 | 0 | True |
| C6 | MINUS_SLOW | 0.300000 | 2233 | 594 | 1639 | 0 | 0.025479 | 0.046053 | False | 0.000000 | 0.000000 | 0.050530 | 0 | False | 0 | None | 4676065280 | [] | 0 | 0 | True |

| pair | method | loss_type | sample_count_unit | videos_with_updates | optimizer_updates | actual_target_or_positive_samples | actual_NONE_or_negative_samples | NONE_reweighted_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | FULL | EXCLUSIVE_LIST_CE | sampled track-frame target/none | 23 | 3900 | 124800 | 0 | 0 |
| C1 | SLOW_REPLAY | EXCLUSIVE_LIST_CE | sampled track-frame target/none | 18 | 3050 | 97600 | 0 | 0 |
| C3 | FULL | BINARY_BCE | sampled query-track-frame | 40 | 7000 | 115407 | 108593 | 0 |
| C3 | SLOW_REPLAY | BINARY_BCE | sampled query-track-frame | 26 | 4550 | 115562 | 30038 | 0 |
| C4 | FULL | BINARY_BCE | sampled query-track-frame | 44 | 8400 | 122509 | 146291 | 0 |
| C4 | SLOW_REPLAY | BINARY_BCE | sampled query-track-frame | 45 | 9000 | 0 | 288000 | 0 |

| comparison | delta_HOTA | CI95 | clusters |
| --- | --- | --- | --- |
| C1__FULL_v7_minus_MINUS_SLOW_v7 | -3.683522 | [-9.777914, 1.400499] | 25 |
| C1__SLOW_REPLAY_v7_minus_MINUS_SLOW_v7 | -1.200113 | [-7.271403, 5.171619] | 25 |
| C3__FULL_v7_minus_MINUS_SLOW_v7 | -0.212194 | [-1.672129, 1.316876] | 45 |
| C3__SLOW_REPLAY_v7_minus_MINUS_SLOW_v7 | -3.242482 | [-4.950690, -1.412725] | 45 |
| C4__FULL_v7_minus_MINUS_SLOW_v7 | 15.721841 | [11.158589, 20.752579] | 45 |
| C4__SLOW_REPLAY_v7_minus_MINUS_SLOW_v7 | -40.544277 | [-48.254527, -34.905244] | 45 |

逐轮真实优化输入正负暴露、训练池计数、两种权重、更新步数见slow_rounds.csv与原逐步records；不由轮末标签倒推。SLOW_REPLAY复用MINUS实际问题/帧/看法/原始回答，新增教师调用为0；独立部署总成本仍包含MINUS教师依赖，不能说完全不用教师。
互斥CE的负样本计数指NONE类训练样本，不是所有负类梯度；正例列表中已知负类仍可受CE约束。BCE的负样本则是负(query,track,frame)实例，两个单位不混加。没有NONE样本时，NONE频率降权未实际触发，不能宣称该机制已获实验验证。

FULL−MINUS包括选题反馈；REPLAY−MINUS固定教师内容但包含LoRA及模型引起的f/快校准/EB下游效应，并非将派生状态也固定。零adapter输入已用相同冻结bank锚定，后续FP32打分路径及32帧近似的限制仍需披露。

## 6. 不确定性与边界

完整视频bootstrap2000次seed0见bootstrap.json；KITTI四视频仅点估计，条件于当前固定流，未覆盖重新适应后的顺序/seed不确定性。

视频转导式/冻结教师辅助，Val历史开发复用，非严格帧因果或盲测。所有GT仅封存后独立CPU读取。DS事后值仅实际提问集合，不代表全候选教师准确率。worker/phase elapsed非GPU利用率积分；历史候选、冻结推理、源头成本不是零。

成本关卡见budget_preflight.json；本批结束后停止，不自动开预算曲线/消融、未选v7 FULL、v6扩展或旧80项。

## SportsMOT 418条敏感性（v4内容不变，仅限制序列ID）

| pair | method | full466_HOTA | HOTA_unrounded | DetA | AssA | IDF1 | IDSW | episodes | prediction_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C3 | MINUS_SLOW_v7 | 65.345262 | 66.270536 | 62.067000 | 70.773000 | 66.436000 | 2834.000000 | 418 | 1457601 |
| C3 | FULL_v7 | 65.133068 | 66.164138 | 61.837000 | 70.807000 | 66.142000 | 3175.000000 | 418 | 1567967 |
| C3 | SLOW_REPLAY_v7 | 62.102780 | 63.387637 | 56.518000 | 71.103000 | 62.532000 | 4086.000000 | 418 | 2187514 |
| C4 | MINUS_SLOW_v7 | 45.656912 | 45.919286 | 30.352000 | 69.472000 | 40.478000 | 1480.000000 | 418 | 809852 |
| C4 | FULL_v7 | 61.378753 | 61.840624 | 54.414000 | 70.289000 | 60.690000 | 2772.000000 | 418 | 1313103 |
| C4 | SLOW_REPLAY_v7 | 5.112636 | 5.138552 | 0.404660 | 65.278000 | 0.713420 | 6.000000 | 418 | 6537 |

## 固定证据回放与主慢速分支的最终差值

| pair | comparison | delta_HOTA | CI95 | delta_GT_recall | delta_prediction_rows | delta_Query_Empty_rows | teacher_content_fixed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | FULL_v7_minus_MINUS_SLOW_v7 | -3.683522 | [-9.777914, 1.400499] | -0.066180 | 6090 | 17290 | False |
| C1 | SLOW_REPLAY_v7_minus_MINUS_SLOW_v7 | -1.200113 | [-7.271403, 5.171619] | -0.022311 | -5675 | 6096 | True |
| C3 | FULL_v7_minus_MINUS_SLOW_v7 | -0.212194 | [-1.672129, 1.316876] | 0.027659 | 117750 | 0 | False |
| C3 | SLOW_REPLAY_v7_minus_MINUS_SLOW_v7 | -3.242482 | [-4.950690, -1.412725] | 0.137688 | 782560 | 0 | True |
| C4 | FULL_v7_minus_MINUS_SLOW_v7 | 15.721841 | [11.158589, 20.752579] | 0.310940 | 513448 | 0 | False |
| C4 | SLOW_REPLAY_v7_minus_MINUS_SLOW_v7 | -40.544277 | [-48.254527, -34.905244] | -0.365866 | -808521 | 0 | True |

固定回放保持MINUS的问题、帧、视图和原始回答不变；复核资格按原MINUS选题状态审计，而不是依据回放改变后的后验重选。LoRA引起的f/快速校准/EB变化仍是下游效应。

## DS最终估计与事后GT（只限实际提问集合）

| pair | method | worker | observations | specificity_DS | specificity_GT | sensitivity_DS | sensitivity_GT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | FULL_v7 | CROP | 3195.000000 | 0.925204 | 0.736909 | 0.802696 | 0.885127 |
| C1 | MINUS_SLOW_v7 | CROP | 3264.000000 | 0.949822 | 0.778666 | 0.817097 | 0.897503 |
| C1 | SLOW_REPLAY_v7 | CROP | 3264.000000 | 0.949822 | 0.778666 | 0.817097 | 0.897503 |
| C2 | MINUS_SLOW_v7 | CLIP | 2328.000000 | 0.858087 | 0.707003 | 0.613129 | 0.686528 |
| C2 | MINUS_SLOW_v7 | CROP | 8748.000000 | 0.797425 | 0.652011 | 0.655694 | 0.738818 |
| C2 | MINUS_SLOW_v7 | FRAME_BOX | 9879.000000 | 0.759826 | 0.467442 | 0.884780 | 0.855222 |
| C3 | FULL_v7 | CROP | 1450.000000 | 0.602611 | 0.677122 | 0.851314 | 0.897577 |
| C3 | FULL_v7 | FRAME_BOX | 2058.000000 | 0.621782 | 0.552009 | 0.945972 | 0.941419 |
| C3 | MINUS_SLOW_v7 | CROP | 1580.000000 | 0.721560 | 0.701176 | 0.835795 | 0.890909 |
| C3 | MINUS_SLOW_v7 | FRAME_BOX | 1928.000000 | 0.943674 | 0.661731 | 0.892468 | 0.943810 |
| C3 | SLOW_REPLAY_v7 | CROP | 1580.000000 | 0.721560 | 0.701176 | 0.835795 | 0.890909 |
| C3 | SLOW_REPLAY_v7 | FRAME_BOX | 1928.000000 | 0.943674 | 0.661731 | 0.892468 | 0.943810 |
| C4 | FULL_v7 | CROP | 3065.000000 | 0.647299 | 0.641527 | 0.906353 | 0.938724 |
| C4 | FULL_v7 | FRAME_BOX | 443.000000 | 0.494773 | 0.596923 | 0.498848 | 0.771186 |
| C4 | MINUS_SLOW_v7 | CROP | 2037.000000 | 0.588424 | 0.566993 | 0.851502 | 0.896140 |
| C4 | MINUS_SLOW_v7 | FRAME_BOX | 1471.000000 | 0.826275 | 0.583208 | 0.838473 | 0.917910 |
| C4 | SLOW_REPLAY_v7 | CROP | 2037.000000 | 0.588424 | 0.566993 | 0.851502 | 0.896140 |
| C4 | SLOW_REPLAY_v7 | FRAME_BOX | 1471.000000 | 0.826275 | 0.583208 | 0.838473 | 0.917910 |
| C5 | MINUS_SLOW_v7 | CLIP | 7926.000000 | 0.898837 | 0.664092 | 0.700220 | 0.820489 |
| C5 | MINUS_SLOW_v7 | CROP | 8688.000000 | 0.882761 | 0.685566 | 0.642348 | 0.699252 |
| C5 | MINUS_SLOW_v7 | FRAME_BOX | 4341.000000 | 0.836002 | 0.612217 | 0.716720 | 0.822960 |
| C6 | MINUS_SLOW_v7 | CROP | 2233.000000 | 0.965171 | 0.876988 | 0.674622 | 0.864553 |

轨迹多数标签口径仅作诊断；同一轨迹的重复帧/视图高度相关，不能把这些行当独立样本作显著性检验，也不把选题后的正确率推广到全库。

## 最终成本、日志与恢复路径

| metric | value |
| --- | --- |
| primary_jobs | 12 |
| conditional_YT_PRIOR_jobs | 0 |
| logical_questions_including_fixed_replay | 74914 |
| new_teacher_calls | 30253 |
| cache_hits | 34381 |
| replayed_answers | 10280 |
| teacher_forward_GPU_phase_hours | 0.777291 |
| LoRA_GPU_phase_hours | 16.072285 |
| rescore_GPU_phase_hours | 10.877470 |
| measured_new_GPU_phase_hours | 27.727046 |
| assigned_worker_wall_hours | 28.279670 |
| scheduler_wall_hours | 15.348615 |
| failed_worker_attempts | 1 |
| optimizer_updates | 35900 |
| worker_wall_includes_estimated_failed_attempt | True |
| incomplete_failed_rescore_GPU_phases | 1 |
| measured_GPU_phase_total_is_lower_bound | True |
| GPU_phase_and_worker_times_are_not_utilization_integrals | True |
| scheduler_wall_includes_waiting_loading_CPU_evaluation_and_reports | True |
| teacher_subprocess_wall_not_double_added | True |
| replay_teacher_dependency_cost_in_MINUS_not_added_twice | True |
| inherited_official_model_and_Stage_A_cost_not_zero_or_remeasured | True |

主集合为六MINUS、三FULL、三REPLAY；额外YT仅按FULL调用变化严格>5%规则触发。MINUS与FULL调用量不同，复用对照不伪称逐臂精确等预算。
预算匹配的是教师调用次数，不是图像张数、token数或GPU时间；不同看法的实测forward成本分别保留。表中是本批方法实验的新增成本，smoke见gpu_smoke.json，历史冻结打分/Stage A输入的成本未重测，不等于零。

GT审计及完整候选/EB更新/实际优化输入核验见validation.json；逐句截断前后似然和生效次数见presence_likelihood.csv。
完整关闭日志与实际用到的唯一教师原响应见logs.json、archive_validation.json及其列出的归档；checkpoint仅发布路径/大小/SHA，预测与PCD留服务器。

运行根目录：`/data2/user/FlexHook-TTA/results/runs/presence_active_20261102/redesign_v7`。原durable恢复记录：`/data2/user/FlexHook-TTA/results/runs/presence_active_20261102/redesign_v7/launchers/durable_1793863666149837309`。封存方法不会重跑已完成episode。

ENOSPC恢复见storage_recovery.json与storage_copy_manifest.json：原始文件保留，仅C4工作副本及日志换盘；前36视频和7200步从checkpoint恢复。失败worker壁钟由文件时间估算并明确标记；中断的全视频打分GPU时间未知，新增GPU相位合计为已测下界，不把缺失成本记为0。最终durable：`/data2/user/FlexHook-TTA/results/runs/presence_active_20261102/redesign_v7/launchers/durable_1793915191037718367`。

## 最终研究边界与停止

不以loss下降、梯度非零、代码通过或某个单方向提升代替六方向验收；不跨FULL/MINUS逐格挑最优。v7复合改动包含EB、伪标签、复核和损失平衡，未做单因素消融，不能把全部差值仅归因EB。
DS各视图来自同一MLLM且相关；分层内按价值选题会影响π估计，未证明无偏。数据流先验可以导致无新教师证据的拒绝，存在/空句误拒及未问错负均完整报告。固定32帧不是末模型全视频重打分。

事后代数诊断（不参与预注册验收）：排名弱证据最多+1，故视频初ρ̂<0.039270300550050576时，全部句子的最高初始存在概率也<.1。终止判空阻断新提问，视频末又将它们计入ρ的负计数，可形成无法自行恢复的拒绝状态。见absorbing_prior.csv/json中的实际日志，不把所有HOTA差值都单因归于此。

| pair | method | video | rho_before | highest_possible_prior_presence | observed_questions | inferred_absent | prediction_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C4 | SLOW_REPLAY | v_2QhNRucNC7E_c017 | 0.030303 | 0.078295 | 76 | 8 | 0 |
| C4 | SLOW_REPLAY | v_4-EmEtrturE_c009 | 0.024390 | 0.063633 | 50 | 14 | 0 |
| C4 | SLOW_REPLAY | v_4r8QL_wglzQ_c001 | 0.018182 | 0.047926 | 43 | 8 | 0 |
| C4 | SLOW_REPLAY | v_5ekaksddqrc_c001 | 0.015873 | 0.042002 | 36 | 8 | 0 |
| C4 | SLOW_REPLAY | v_5ekaksddqrc_c002 | 0.014085 | 0.037381 | 36 | 8 | 0 |
| C4 | SLOW_REPLAY | v_5ekaksddqrc_c003 | 0.012658 | 0.033676 | 38 | 8 | 0 |
| C4 | SLOW_REPLAY | v_5ekaksddqrc_c004 | 0.011494 | 0.030639 | 38 | 8 | 0 |
| C4 | SLOW_REPLAY | v_5ekaksddqrc_c005 | 0.010526 | 0.028105 | 43 | 8 | 0 |
| C4 | SLOW_REPLAY | v_9MHDmAMxO5I_c002 | 0.009709 | 0.025958 | 45 | 10 | 0 |
| C4 | SLOW_REPLAY | v_9MHDmAMxO5I_c003 | 0.008850 | 0.023695 | 45 | 10 | 0 |
| C4 | SLOW_REPLAY | v_9MHDmAMxO5I_c004 | 0.008130 | 0.021795 | 57 | 10 | 0 |
| C4 | SLOW_REPLAY | v_9MHDmAMxO5I_c006 | 0.007519 | 0.020178 | 78 | 13 | 0 |
| C4 | SLOW_REPLAY | v_9MHDmAMxO5I_c009 | 0.006849 | 0.018402 | 63 | 10 | 0 |
| C4 | SLOW_REPLAY | v_BgwzTUxJaeU_c008 | 0.006410 | 0.017235 | 51 | 10 | 0 |
| C4 | SLOW_REPLAY | v_BgwzTUxJaeU_c012 | 0.006024 | 0.016207 | 45 | 10 | 0 |
| C4 | SLOW_REPLAY | v_BgwzTUxJaeU_c014 | 0.005682 | 0.015295 | 48 | 10 | 0 |
| C4 | SLOW_REPLAY | v_G-vNjfx1GGc_c004 | 0.005376 | 0.014481 | 138 | 10 | 0 |
| C4 | SLOW_REPLAY | v_G-vNjfx1GGc_c008 | 0.005102 | 0.013748 | 96 | 10 | 0 |
| C4 | SLOW_REPLAY | v_G-vNjfx1GGc_c600 | 0.004854 | 0.013086 | 88 | 7 | 0 |
| C4 | SLOW_REPLAY | v_G-vNjfx1GGc_c601 | 0.004695 | 0.012660 | 117 | 10 | 0 |
| C4 | SLOW_REPLAY | v_ITo3sCnpw_k_c007 | 0.004484 | 0.012096 | 190 | 12 | 0 |
| C4 | SLOW_REPLAY | v_ITo3sCnpw_k_c010 | 0.004255 | 0.011483 | 102 | 9 | 0 |
| C4 | SLOW_REPLAY | v_ITo3sCnpw_k_c011 | 0.004098 | 0.011063 | 168 | 14 | 0 |
| C4 | SLOW_REPLAY | v_ITo3sCnpw_k_c012 | 0.003876 | 0.010466 | 270 | 12 | 0 |
| C4 | SLOW_REPLAY | v_cC2mHWqMcjk_c007 | 0.003704 | 0.010004 | 79 | 14 | 0 |
| C4 | SLOW_REPLAY | v_cC2mHWqMcjk_c008 | 0.003521 | 0.009514 | 51 | 10 | 0 |
| C4 | SLOW_REPLAY | v_cC2mHWqMcjk_c009 | 0.003401 | 0.009192 | 50 | 12 | 0 |
| C4 | SLOW_REPLAY | v_dw7LOz17Omg_c053 | 0.003268 | 0.008834 | 135 | 10 | 0 |
| C4 | SLOW_REPLAY | v_dw7LOz17Omg_c067 | 0.003165 | 0.008556 | 108 | 11 | 0 |
| C4 | SLOW_REPLAY | v_i2_L4qquVg0_c006 | 0.003058 | 0.008269 | 158 | 11 | 0 |
| C4 | SLOW_REPLAY | v_i2_L4qquVg0_c007 | 0.002959 | 0.008002 | 70 | 9 | 0 |
| C4 | SLOW_REPLAY | v_i2_L4qquVg0_c009 | 0.002882 | 0.007795 | 86 | 9 | 0 |
| C4 | SLOW_REPLAY | v_i2_L4qquVg0_c010 | 0.002809 | 0.007599 | 207 | 11 | 0 |
| C6 | MINUS_SLOW | dancetrack0077 | 0.036269 | 0.092807 | 0 | 17 | 0 |
| C6 | MINUS_SLOW | dancetrack0079 | 0.033333 | 0.085701 | 0 | 17 | 0 |
| C6 | MINUS_SLOW | dancetrack0081 | 0.030837 | 0.079606 | 0 | 17 | 0 |
| C6 | MINUS_SLOW | dancetrack0090 | 0.028689 | 0.074320 | 0 | 17 | 0 |
| C6 | MINUS_SLOW | dancetrack0094 | 0.026820 | 0.069692 | 0 | 17 | 0 |
| C6 | MINUS_SLOW | dancetrack0097 | 0.025180 | 0.065607 | 0 | 17 | 0 |
视频聚类CI条件于本次有序适应流，不覆盖流顺序或重新训练的不确定性；历史开发Val不是盲测集。本设置是MLLM辅助、视频转导式，不是严格帧因果或完全不调用教师的Source-Free在线系统。

本批结束并停止。不运行v6 C4–C6、旧80项、未选v7 FULL、预算曲线、消融或新方法；即便满足申请门槛，也必须等待用户下一条明确授权。
