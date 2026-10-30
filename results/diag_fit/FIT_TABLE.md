# 表 A — D-FIT 教师拟合（不使用目标 GT）

全部主拟合列使用相同Naive8500判断、POS≥0.95，NEG为有效p<0.95；300窗口子集的SHA相同。
旧PSR在自身8500标签的历史拟合另列summary，不混入这个共同标签主表。
这是训练锚点拟合，不是GT精度或完整RMOT；新臂固定最后8973步，不选择source-best作目标评价。

| 指标 | Frozen | Naive(s17) | PSR(s17) | D1_OPT | D2_OPT_POS | D3_CAP |
|---|---:|---:|---:|---:|---:|---:|
| 可训练参数量 | 0 | 24576 | 24576 | 24576 | 24576 | 2758656 |
| 实际 pos_weight（未加权=1） | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 10.000000 | 10.000000 |
| teacher_POS_selected_rate | 0.083841 | 0.060976 | 0.082317 | 0.795732 | 0.940549 | 0.977134 |
| teacher_NEG_selected_rate | 0.063870 | 0.004335 | 0.008287 | 0.055584 | 0.226160 | 0.126849 |
| mean_student_prob_on_POS | 0.090603 | 0.207832 | 0.222300 | 0.719190 | 0.901210 | 0.944540 |
| mean_student_prob_on_NEG | 0.068301 | 0.085495 | 0.092968 | 0.102303 | 0.271442 | 0.158571 |
| pair_agreement | 0.570896 | 0.611940 | 0.593284 | 0.761194 | 0.716418 | 0.865672 |
| 结束 source_val_CE | 1.022258 | 0.883511 | 0.882947 | 0.919990 | 0.872234 | 0.725210 |
| step0 source_val_CE | 1.022258 | 1.022258 | 1.022258 | 1.022258 | 1.022258 | 1.022258 |
| 裁剪前总梯度范数中位数 | N/A | 0.224535 | 0.220825 | 0.988961 | 3.055050 | 4.102582 |
| 裁剪前总梯度范数 p90 | N/A | 1.911308 | 2.036309 | 3.426016 | 17.861844 | 26.604391 |
| 被裁剪步数比例 | N/A | 0.197816 | 0.202942 | 0.038560 | 0.316728 | 0.447565 |

新臂状态与source-selected记录（只记录，不对其source-best做Val评测）：
- D1_OPT: COMPLETE; fallback=False; selection={'step': 8973, 'sha256': 'b58568b37e4074369d11d672ac0d0bc1aa2814fe52451bba7000af79a4b96d49', 'selection': 'FINAL_STEP_ONLY_NO_TARGET_GT', 'source_selected_step': 2750}
- D2_OPT_POS: COMPLETE; fallback=False; selection={'step': 8973, 'sha256': '9e149179814fd9a3b669b9d2e858ccdff41ff3edfedd64f24beb639ef4cfab15', 'selection': 'FINAL_STEP_ONLY_NO_TARGET_GT', 'source_selected_step': 7750}
- D3_CAP: COMPLETE; fallback=False; selection={'step': 8973, 'sha256': 'a95b57fc6fe231a358d5259ba8d9b569807d3b41b15a1fc87124caab27020214', 'selection': 'FINAL_STEP_ONLY_NO_TARGET_GT', 'source_selected_step': 8250}
