# v7：在线EB存在性与慢速分支公平检验

当前已封存并完成官方评测 7/12 个主项；只对实际完成集合统计，不能把阶段结果当最终验收。

最新C3 FULL：HOTA 65.133068，对MINUS 65.345262为−0.212194，视频聚类95% CI[−1.672129,1.316876]跨0；未证明慢速增益，也不能断言它必然有害。40/45视频实际发生更新，共7000步；成功优化的正/负样本暴露分别115407/108593，不能用“没训练或纯负例训练”解释这一结果。LoRA与重打分分别3.070007/1.315566小时，教师调用仍3508，但FULL与MINUS选题内容不相同；固定证据回放尚待完成。

C3 FULL的独立IoU≥.5 GT召回78.0639%→80.8298%，匹配GT观测净增42420，但FP总数净增75330（不是新增预测行误检的同义词）。有目标句误判空2→52，高置信错负27→291，其中221个未问对由整句判空置零；未问对仍UNKNOWN、不参与训练。FULL对YT_PRIOR +6.532277，CI[4.827158,8.233172]，但这一总体改善不能归因于慢速分支。

六MINUS仍为5/6点胜两简单对照较好者，但C6=30.204325低于固定34.469保留门槛，故该臂FAIL。C6最后6视频的102句先验全拒绝另见absorbing_prior.csv/json；这不是未完成或漏评。FULL只计划三个方向，六方向门槛未评估；当前C1/C4 FULL与三回放继续，不跨臂拼最优，也不开旧80项或未选FULL。

| method | six_directions_complete | directions_above_both_controls | Dance_both_within_two_of_v5 | result |
| --- | --- | --- | --- | --- |
| FULL_v7 | False | 1 | False | NOT_EVALUATED |
| MINUS_SLOW_v7 | True | 5 | False | FAIL |

## 1. 六方向主表（未运行格为NA，不是0）

| pair | source | target | NATIVE | PRES_TOPK | FULL_v5 | FULL_v6 | FULL_v7 | MINUS_SLOW_v7 | SLOW_REPLAY_v7 | YT_PRIOR | UNIFORM_K | YT_PRIOR_rematch_pending |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | V2-best | Dance | 8.218023 | 10.320883 | 37.723955 | 26.367793 | NA | 38.849551 | NA | 22.710562 | 24.107948 | False |
| C2 | Dance-best | KITTI-V2 | 13.501503 | 13.781387 | 16.099630 | 26.818212 | NA | 25.313445 | NA | 24.285695 | 24.413313 | False |
| C3 | V2-best | SportsMOT | 50.970478 | 52.927461 | 43.914165 | 59.341507 | 65.133068 | 65.345262 | NA | 58.600791 | 40.804245 | False |
| C4 | Dance-best | SportsMOT | 20.559603 | 21.010887 | 32.191307 | NA | NA | 45.656912 | NA | 46.098939 | 40.804245 | False |
| C5 | LaMOT-best | KITTI-V2 | 22.398306 | 23.875533 | 18.315360 | NA | NA | 30.331540 | NA | 27.588195 | 25.674444 | False |
| C6 | LaMOT-best | Dance | 15.475743 | 15.151263 | 36.468777 | NA | NA | 30.204325 | NA | 20.821630 | 24.107948 | False |

## 2. 完整官方指标

| pair | method | HOTA_unrounded | DetA | AssA | IDF1 | IDSW | prediction_rows | query_empty_prediction_rows | Present_HOTA | teacher_calls | GT_open_attempts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | MINUS_SLOW_v7 | 38.849551 | 44.840000 | 33.728000 | 37.332000 | 1386.000000 | 183380 | 18477 | 40.248855 | 3264 | 0 |
| C2 | MINUS_SLOW_v7 | 25.313445 | 11.013000 | 58.264000 | 19.629000 | 701.000000 | 155932 | 0 | 25.313445 | 20955 | 0 |
| C3 | MINUS_SLOW_v7 | 65.345262 | 60.290000 | 70.837000 | 65.297000 | 2848.000000 | 1517412 | 0 | 65.345262 | 3508 | 0 |
| C3 | FULL_v7 | 65.133068 | 59.881000 | 70.858000 | 64.876000 | 3195.000000 | 1635162 | 0 | 65.133068 | 3508 | 0 |
| C4 | MINUS_SLOW_v7 | 45.656912 | 30.011000 | 69.461000 | 40.131000 | 1480.000000 | 815058 | 0 | 45.656912 | 3508 | 0 |
| C5 | MINUS_SLOW_v7 | 30.331540 | 16.362000 | 56.330000 | 27.606000 | 2746.000000 | 1125042 | 0 | 30.331540 | 20955 | 0 |
| C6 | MINUS_SLOW_v7 | 30.204325 | 27.982000 | 32.653000 | 26.355000 | 840.000000 | 111717 | 13144 | 31.024575 | 2233 | 0 |

## 3. 存在性与在线比例

| pair | method | query_present_episodes | query_empty_episodes | correctly_inferred_empty | present_incorrectly_inferred_empty | empty_rejected_without_presence_answers | present_rejected_without_presence_answers | query_empty_prediction_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | MINUS_SLOW_v7 | 40 | 385 | 293 | 6 | 96 | 1 | 18477 |
| C2 | MINUS_SLOW_v7 | 862 | 0 | 0 | 296 | 0 | 0 | 0 |
| C3 | FULL_v7 | 466 | 0 | 0 | 52 | 0 | 0 | 0 |
| C3 | MINUS_SLOW_v7 | 466 | 0 | 0 | 2 | 0 | 0 | 0 |
| C4 | MINUS_SLOW_v7 | 466 | 0 | 0 | 4 | 0 | 0 | 0 |
| C5 | MINUS_SLOW_v7 | 862 | 0 | 0 | 12 | 0 | 0 | 0 |
| C6 | MINUS_SLOW_v7 | 40 | 385 | 274 | 13 | 218 | 12 | 13144 |

![rho stream](rho_hat_stream.png)

![pi stream](pi_hat_stream.png)

逐视频先验与软计数见eb_stream.csv/原presence.json；逐句未截断/截断LL和触发次数见presence_likelihood.csv。停止概率与按最后DS重算概率分别记录，不冒充同一个时点。
presence_decisions.csv中的teacher_absent/teacher_confirmed_present是继承的字段名，v7实际表示EB的终止判空/判存在，不是教师直接审阅完整视频的标注；无分层回答而仅由先验触发的拒绝另行列出。

## 4. 高置信错负与真实恢复

| pair | method | matched_candidate_positive_tracks_majority | confident_negative_GT_FN | confident_negative_GT_FN_unqueried | GTpositive_high_negative_rate |
| --- | --- | --- | --- | --- | --- |
| C1 | MINUS_SLOW_v7 | 567 | 86 | 76 | 0.151675 |
| C2 | MINUS_SLOW_v7 | 8441 | 2508 | 2122 | 0.297121 |
| C3 | FULL_v7 | 5528 | 291 | 221 | 0.052641 |
| C3 | MINUS_SLOW_v7 | 5528 | 27 | 1 | 0.004884 |
| C4 | MINUS_SLOW_v7 | 5528 | 47 | 18 | 0.008502 |
| C5 | MINUS_SLOW_v7 | 8441 | 145 | 50 | 0.017178 |
| C6 | MINUS_SLOW_v7 | 567 | 271 | 271 | 0.477954 |

此表高置信错负仍用v6的p<.05以可比；实际伪监督阈值为.15/.85，且未问始终UNKNOWN。最终posterior不是实际优化输入。

v7组合prior截断到[-2,2]且未问对evidence=0，因此未判空句子的未问对posterior至少为sigmoid(-2)≈.1192；本表未问p<.05来自整句判空置零，不是未问对被赋了训练负标签。

| pair | method | GT_observations | matched_GT | GT_recall | output_rows | recovered_GT | lost_GT | net_GT_vs_Native | new_FP_rows | removed_FP_rows | unchanged_rows_new_FP | unchanged_rows_removed_FP | net_FP_vs_Native | definition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | MINUS_SLOW | 212541 | 133456 | 0.627907 | 183380 | 124690 | 1571 | 123119 | 46430 | 85885 | 12 | 0 | -39443 | offline max-cardinality framewise IoU>=0.5 assignment, not official HOTA matching |
| C2 | MINUS_SLOW | 491286 | 70575 | 0.143654 | 155932 | 53656 | 8922 | 44734 | 63808 | 52499 | 6 | 6 | 11309 | offline max-cardinality framewise IoU>=0.5 assignment, not official HOTA matching |
| C3 | MINUS_SLOW | 1533675 | 1197247 | 0.780639 | 1517412 | 348657 | 50850 | 297807 | 170541 | 165288 | 221 | 293 | 5181 | offline max-cardinality framewise IoU>=0.5 assignment, not official HOTA matching |
| C3 | FULL | 1533675 | 1239667 | 0.808298 | 1635162 | 438007 | 97780 | 340227 | 303144 | 222971 | 504 | 166 | 80511 | offline max-cardinality framewise IoU>=0.5 assignment, not official HOTA matching |
| C4 | MINUS_SLOW | 1533675 | 567570 | 0.370072 | 815058 | 458089 | 37612 | 420477 | 230396 | 48344 | 78 | 124 | 182006 | offline max-cardinality framewise IoU>=0.5 assignment, not official HOTA matching |
| C5 | MINUS_SLOW | 491286 | 247921 | 0.504637 | 1125042 | 165312 | 9922 | 155390 | 721812 | 85035 | 95 | 42 | 636830 | offline max-cardinality framewise IoU>=0.5 assignment, not official HOTA matching |
| C6 | MINUS_SLOW | 212541 | 76869 | 0.361667 | 111717 | 38637 | 29062 | 9575 | 21341 | 429175 | 78 | 28 | -407784 | offline max-cardinality framewise IoU>=0.5 assignment, not official HOTA matching |

GT召回是独立IoU≥.5最大基数匹配，不替代官方TrackEval；新增/删除预测行的FP与不变行匹配重分配分列。

## 5. 慢速更新、成本与成对差值

| pair | method | rho | logical_questions | new_teacher_calls | cache_hits | replayed_answers | teacher_forward_GPU_hours | teacher_equivalent_GPU_hours | replay_teacher_cost_in_MINUS_dependency | LoRA_GPU_hours | rescore_GPU_hours | assigned_worker_wall_hours | failed_worker_attempts | peak_parent_allocated_bytes | peak_teacher_allocated_bytes | trainable_parameters_observed_in_updates | optimizer_updates | videos_with_LoRA_updates | wall_is_not_GPU_utilization_integral |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | MINUS_SLOW | 0.300000 | 3264 | 595 | 2669 | 0 | 0.012809 | 0.042194 | False | 0.000000 | 0.000000 | 0.040861 | 0 | None | 4472145920 | [] | 0 | 0 | True |
| C2 | MINUS_SLOW | 0.300000 | 20955 | 9477 | 11478 | 0 | 0.318490 | 0.646320 | False | 0.000000 | 0.000000 | 0.428484 | 0 | None | 5469817344 | [] | 0 | 0 | True |
| C3 | MINUS_SLOW | 0.300000 | 3508 | 1720 | 1788 | 0 | 0.039901 | 0.090459 | False | 0.000000 | 0.000000 | 0.075097 | 0 | None | 5899776000 | [] | 0 | 0 | True |
| C3 | FULL | 0.300000 | 3508 | 1761 | 1747 | 0 | 0.043687 | 0.089526 | False | 3.070007 | 1.315566 | 4.477327 | 0 | 4582467584 | 5904013312 | [24576] | 7000 | 40 | True |
| C4 | MINUS_SLOW | 0.300000 | 3508 | 1234 | 2274 | 0 | 0.027750 | 0.081677 | False | 0.000000 | 0.000000 | 0.061267 | 0 | None | 5896557568 | [] | 0 | 0 | True |
| C5 | MINUS_SLOW | 0.300000 | 20955 | 13042 | 7913 | 0 | 0.247136 | 0.465638 | False | 0.000000 | 0.000000 | 0.355100 | 0 | None | 5474210304 | [] | 0 | 0 | True |
| C6 | MINUS_SLOW | 0.300000 | 2233 | 594 | 1639 | 0 | 0.025479 | 0.046053 | False | 0.000000 | 0.000000 | 0.050530 | 0 | None | 4676065280 | [] | 0 | 0 | True |

| comparison | delta_HOTA | CI95 | clusters |
| --- | --- | --- | --- |
| C3__FULL_v7_minus_MINUS_SLOW_v7 | -0.212194 | [-1.672129, 1.316876] | 45 |

逐轮真实优化输入正负暴露、训练池计数、两种权重、更新步数见slow_rounds.csv与原逐步records；不由轮末标签倒推。SLOW_REPLAY复用MINUS实际问题/帧/看法/原始回答，新增教师调用为0；独立部署总成本仍包含MINUS教师依赖，不能说完全不用教师。

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
| C4 | MINUS_SLOW_v7 | 45.656912 | 45.919286 | 30.352000 | 69.472000 | 40.478000 | 1480.000000 | 418 | 809852 |
