# v7：在线EB存在性与慢速分支公平检验

当前仅完成3/12个主项，原队列仍运行，以下不是最终科研验收。

已封存三项的视频聚类2000次/seed0结果：C1 MINUS−YT_PRIOR +16.139，CI[10.477,21.522]；C3 +6.744，CI[5.694,7.841]；C4 −.442，CI[−5.228,3.551]。C1相对v5 FULL +1.126的CI[−2.475,5.483]跨0，不能把点增益当作稳定胜过v5。

C1 EB正确判空293/385，但也误拒6/40个有目标句子；高置信错负86/567（含76个未问对），仍有明确代价。C3相对Native净恢复297807个GT观测、FP净增5181，并非只靠删输出；C4虽净恢复420477个GT，仍未超过最强简单对照。完整统计见下表，GT只在这三项完整封存后独立读取，不反馈运行中的模型。

{
  "status": "REGISTERED_TIER_REPORTED",
  "arms": [
    {
      "method": "FULL_v7",
      "six_directions_complete": false,
      "point_estimate_wins": [],
      "directions_above_both_controls": 0,
      "Dance_both_within_two_of_v5": false,
      "result": "NOT_EVALUATED",
      "may_request_curves_and_ablations": false,
      "automatically_authorized": false
    },
    {
      "method": "MINUS_SLOW_v7",
      "six_directions_complete": false,
      "point_estimate_wins": [
        "C1",
        "C3"
      ],
      "directions_above_both_controls": 2,
      "Dance_both_within_two_of_v5": false,
      "result": "NOT_EVALUATED",
      "may_request_curves_and_ablations": false,
      "automatically_authorized": false
    }
  ],
  "completed_jobs": 3,
  "expected_primary_jobs": 12,
  "conditional_controls_pending": false,
  "may_request_curves_and_ablations": false,
  "automatically_authorized": false,
  "remaining_v5_80_denied": true,
  "v6_C4_C6_denied": true
}

## 1. 六方向主表（未运行格为NA，不是0）

| pair | source | target | NATIVE | PRES_TOPK | FULL_v5 | FULL_v6 | FULL_v7 | MINUS_SLOW_v7 | SLOW_REPLAY_v7 | YT_PRIOR | UNIFORM_K | YT_PRIOR_rematch_pending |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | V2-best | Dance | 8.218023 | 10.320883 | 37.723955 | 26.367793 | None | 38.849551 | None | 22.710562 | 24.107948 | False |
| C2 | Dance-best | KITTI-V2 | 13.501503 | 13.781387 | 16.099630 | 26.818212 | None | None | None | 24.285695 | 24.413313 | False |
| C3 | V2-best | SportsMOT | 50.970478 | 52.927461 | 43.914165 | 59.341507 | None | 65.345262 | None | 58.600791 | 40.804245 | False |
| C4 | Dance-best | SportsMOT | 20.559603 | 21.010887 | 32.191307 | None | None | 45.656912 | None | 46.098939 | 40.804245 | False |
| C5 | LaMOT-best | KITTI-V2 | 22.398306 | 23.875533 | 18.315360 | None | None | None | None | 27.588195 | 25.674444 | False |
| C6 | LaMOT-best | Dance | 15.475743 | 15.151263 | 36.468777 | None | None | None | None | 20.821630 | 24.107948 | False |

## 2. 完整官方指标

| pair | method | HOTA_unrounded | DetA | AssA | IDF1 | IDSW | prediction_rows | query_empty_prediction_rows | Present_HOTA | teacher_calls | GT_open_attempts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | MINUS_SLOW_v7 | 38.849551 | 44.840000 | 33.728000 | 37.332000 | 1386.000000 | 183380 | 18477 | 40.248855 | 3264 | 0 |
| C3 | MINUS_SLOW_v7 | 65.345262 | 60.290000 | 70.837000 | 65.297000 | 2848.000000 | 1517412 | 0 | 65.345262 | 3508 | 0 |
| C4 | MINUS_SLOW_v7 | 45.656912 | 30.011000 | 69.461000 | 40.131000 | 1480.000000 | 815058 | 0 | 45.656912 | 3508 | 0 |

## 3. 存在性与在线比例

| pair | method | query_present_episodes | query_empty_episodes | correctly_inferred_empty | present_incorrectly_inferred_empty | query_empty_prediction_rows |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | MINUS_SLOW_v7 | 40 | 385 | 293 | 6 | 18477 |
| C3 | MINUS_SLOW_v7 | 466 | 0 | 0 | 2 | 0 |
| C4 | MINUS_SLOW_v7 | 466 | 0 | 0 | 4 | 0 |

![rho stream](rho_hat_stream.png)

![pi stream](pi_hat_stream.png)

逐视频先验与软计数见eb_stream.csv/原presence.json；逐句未截断/截断LL和触发次数见presence_likelihood.csv。停止概率与按最后DS重算概率分别记录，不冒充同一个时点。

## 4. 高置信错负与真实恢复

| pair | method | matched_candidate_positive_tracks_majority | confident_negative_GT_FN | confident_negative_GT_FN_unqueried | GTpositive_high_negative_rate |
| --- | --- | --- | --- | --- | --- |
| C1 | MINUS_SLOW_v7 | 567 | 86 | 76 | 0.151675 |
| C3 | MINUS_SLOW_v7 | 5528 | 27 | 1 | 0.004884 |
| C4 | MINUS_SLOW_v7 | 5528 | 47 | 18 | 0.008502 |

此表高置信错负仍用v6的p<.05以可比；实际伪监督阈值为.15/.85，且未问始终UNKNOWN。最终posterior不是实际优化输入。

| pair | method | GT_observations | matched_GT | GT_recall | output_rows | recovered_GT | lost_GT | net_GT_vs_Native | new_FP_rows | removed_FP_rows | unchanged_rows_new_FP | unchanged_rows_removed_FP | net_FP_vs_Native | definition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | MINUS_SLOW | 212541 | 133456 | 0.627907 | 183380 | 124690 | 1571 | 123119 | 46430 | 85885 | 12 | 0 | -39443 | offline max-cardinality framewise IoU>=0.5 assignment, not official HOTA matching |
| C3 | MINUS_SLOW | 1533675 | 1197247 | 0.780639 | 1517412 | 348657 | 50850 | 297807 | 170541 | 165288 | 221 | 293 | 5181 | offline max-cardinality framewise IoU>=0.5 assignment, not official HOTA matching |
| C4 | MINUS_SLOW | 1533675 | 567570 | 0.370072 | 815058 | 458089 | 37612 | 420477 | 230396 | 48344 | 78 | 124 | 182006 | offline max-cardinality framewise IoU>=0.5 assignment, not official HOTA matching |

GT召回是独立IoU≥.5最大基数匹配，不替代官方TrackEval；新增/删除预测行的FP与不变行匹配重分配分列。

## 5. 慢速更新、成本与成对差值

| pair | method | rho | logical_questions | new_teacher_calls | cache_hits | replayed_answers | teacher_forward_GPU_hours | teacher_equivalent_GPU_hours | replay_teacher_cost_in_MINUS_dependency | LoRA_GPU_hours | rescore_GPU_hours | assigned_worker_wall_hours | optimizer_updates | videos_with_LoRA_updates | wall_is_not_GPU_utilization_integral |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | MINUS_SLOW | 0.300000 | 3264 | 595 | 2669 | 0 | 0.012809 | 0.042194 | False | 0.000000 | 0.000000 | 0.040861 | 0 | 0 | True |
| C3 | MINUS_SLOW | 0.300000 | 3508 | 1720 | 1788 | 0 | 0.039901 | 0.090459 | False | 0.000000 | 0.000000 | 0.075097 | 0 | 0 | True |
| C4 | MINUS_SLOW | 0.300000 | 3508 | 1234 | 2274 | 0 | 0.027750 | 0.081677 | False | 0.000000 | 0.000000 | 0.061267 | 0 | 0 | True |

逐轮真实优化输入正负暴露、训练池计数、两种权重、更新步数见slow_rounds.csv与原逐步records；不由轮末标签倒推。SLOW_REPLAY复用MINUS实际问题/帧/看法/原始回答，新增教师调用为0；独立部署总成本仍包含MINUS教师依赖，不能说完全不用教师。

FULL−MINUS包括选题反馈；REPLAY−MINUS固定教师内容但包含LoRA及模型引起的f/快校准/EB下游效应，并非将派生状态也固定。零adapter输入已用相同冻结bank锚定，后续FP32打分路径及32帧近似的限制仍需披露。

## 6. 不确定性与边界

完整视频bootstrap2000次seed0见bootstrap.json；KITTI四视频仅点估计，条件于当前固定流，未覆盖重新适应后的顺序/seed不确定性。

视频转导式/冻结教师辅助，Val历史开发复用，非严格帧因果或盲测。所有GT仅封存后独立CPU读取。DS事后值仅实际提问集合，不代表全候选教师准确率。worker/phase elapsed非GPU利用率积分；历史候选、冻结推理、源头成本不是零。

成本关卡见budget_preflight.json；本批结束后停止，不自动开预算曲线/消融、未选v7 FULL、v6扩展或旧80项。
