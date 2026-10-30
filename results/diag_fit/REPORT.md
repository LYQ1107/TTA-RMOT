# D-FIT：学生能否拟合冻结 MLLM 标签

状态：FIT_COMPLETE_VAL_PENDING；从公开965e71fe5055a4299ae6f46f89dbe4e993cfad11开始，独立分支research/v2dance-diag-fit-20261030。
用户允许显存足够时共享、优先进程少的最多4卡，不停止其他任务。25Val同集诊断，不是论文主结果或Train→Val。

## 1. 修改前只读核对

- LoRALinear原scaling=1/rank，rank4为0.25，lora_b全零初始化；只替换最后两个PCD块wq/wv。
- 官方checkpoint有4个output_ca_qkv块；wq/wk/wv/wo均768→768、无bias。
- head有4个MLP，各Linear(768,768,bias=False)→GELU→Linear(768,2,bias=False)，dropout0；共2,365,440参数。
- 原mllm_training.run在attach、容量检查、provenance、最终receipt处固定rank4/24576；每步1目标window，unscale后clip1，在step前。
- 旧含POS步裁剪前direct梯度：Naive中位2.246040/p90 4.884213，PSR中位2.227479/p90 4.746833。PSR含weak的target总范数与direct分开。
- PSR含weak的实际target_gradient_norm在原逐步trace中为中位2.739018/p90 5.960116；原training_dynamics.json没有单独汇总该combined字段，未将direct误称combined。
- **与提示预期不同：657/374的POS门是p≥0.95，不是0.5**，见psr_supervision.direct_evidence与封存state。按“以代码为准”沿用0.95；D-FIT NEG按指定auto公式为有效p<0.95，含历史低置信UNKNOWN，与旧审计NEG≤0.05区分。
- 原load_selected固定rank4；两matrix脚本固定checkpoint_best_source_selected.pt。
- D3理论LoRA393,216+head2,365,440=2,758,656；最终以真实smoke核验填回配置。
- 执行源码与公开快照映射见[source_publication_map.json](source_publication_map.json)：准备执行快照a67f4ada对应可检出的2259228b，同Git tree为98831e0d223e34e0f520cc3f32006ad153ebc904。训练核心/compat/三配置/训练入口与当前6dc3d52的tracked diff为空；后续改变只涉及报告、测试与记录，不把不同Git对象冒充同一提交。

## 执行计划与隔离

实现/必要回归→三臂200步smoke→旧checkpoint同300窗口子集与全8500判断拟合→三臂8973步。
拟合表A先实际push并核验，随后才允许final-step全425推理；完整预测SHA封存后单独GT评测。
三臂只用相同Naive8500封存标签；PSR基线同时报告其原8500拟合（解释旧38/374）与相同Naive标签拟合（公平比较）。
仅NaN/Inf或源val CE连续2次超过初始3倍时，允许该臂lr=1e-4从头重跑一次，head_lr不变；原失败保留。
最终checkpoint固定最后8973步；source-selected只记录，不对三新臂做source-selected Val评测。

## 2. 表A：拟合教师

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

完整8500判断上的拟合合格判定：D1/D2/D3均False。D1 POS差3个；D2和D3的NEG选中率分别22.62%/12.68%，均未达到≤10%。不放宽标准，也不以300窗口曲线子集替代全8500判定。

## 3. 表B：封存后Val结果

完整表A先推送并核验GitHub，再启动三个final8973模型的25视频/425表达推理；预测全部封存后才执行独立GT评测。当前尚无新Val结果。

## 4. 预注册判定

拟合合格：POS selected≥0.80且NEG selected≤0.10。原样应用用户四项分支判定，不另调规则。
规则中的下一阶段只写建议，本轮不自动做密集GT标注、列表蒸馏或正式Train→Val训练。

## 5. 成本与provenance

checkpoint/完整预测保留data2，不进git；保存路径/SHA与每run的目标GT打开次数，必须为0。

### 兼容性执行记录

CPU相关29项测试通过；真实图像alpha1/8/32的初始输出均与未attach逐位相同。
原trainer第一次回归在任何优化步之前被内核输入白名单阻止读取其归档源码（用于自身provenance哈希）。
仅给legacy兼容wrapper加入该单个源码文件读权限，未开放任何目标GT、未改旧训练运算；原失败目录/日志保留，重跑该失败单元。
修复后真实旧/新默认训练均完成6步，前5步source_loss和target_KD_loss的最大差异**均为0**，严格通过atol1e-4。
两次默认真实训练底座未变、目标GT打开0；兼容性不是用合成toy loss替代，另有CPU结构/损失与strict-loader回归。
三臂200步smoke首轮：CAP遇到launch配置JSON数字5e-05被YAML1.1解析成字符串的代码bug，第一次optimizer.step报TypeError。
修复仅为JSON配置走JSON解析，保留head_lr=0.00005和所有预注册超参；补真实数值类型回归，保留失败后从头重跑CAP。
OPT_POS及CAP的早期尝试曾因FP16梯度Inf提前退出，并执行过一次LR回退；这些尝试均保留但撤销为正式结果。
随后代码审查发现集成时错误删除了旧trainer的AMP动态loss-scale恢复（最多8次、每次减半、原样重放同一批次、无优化步）。
恢复原有机制，不更改lr/AMP/rank/层数/clip；按6.3代码bug处理，从头重跑受影响两臂。只有旧机制仍不能恢复的非有限信号才触发6.2唯一LR回退。
此修复没有跳样本、没有关闭AMP、没有新增数值超参；D1已通过的200步不重复运行。不能把无效提前退出当学生容量/教师矛盾的证据。
恢复原AMP后，三臂有效200步smoke全部通过，lr均保持0.0002，无正式LR回退；D3为2,758,656参数且head实际变化，已填回expected_trainable。
三臂fit_subset SHA相同：369cabeb214b2d296bc1a4effc5ebd1d4f22cc8f7f68f2740861a3a9649c30b6；GT打开均0，冻结参数版本均未变。相关CPU32项通过。
D3真实smoke final-step checkpoint经load_selected严格重建后，保存前/重载后完整logits逐位相同，GT打开0。
已有Frozen/Naive/PSR在共同Naive8500上的完整教师拟合基线已完成；三新臂8973步已在GPU1/5/4开始，不复用smoke权重、不重跑D1已完成smoke。
这只是训练运行状态，不是完整拟合或Val改善结论。原命令一次阻塞等待全部训练；没有agent级ps/tail轮询或干预其它任务。

### 已完成的共同教师标签基线（不是完整表A）

完整记录见[baselines/completion.json](baselines/completion.json)。三模型均在相同Naive8500判断上统计：656个teacher POS、7844个有效非POS；目标GT打开0。
Frozen选中55/656（0.083841）、旧Naive(s17)选中40/656（0.060976）、旧PSR(s17)选中54/656（0.082317）。对应teacher NEG selected rate分别0.063870、0.004335、0.008287。
PSR原先自身8500标签有657个POS，在该不同采样集合上选中47/657（0.071537），单独存于own_PSR_labels，不能混入共同标签主比较。
此阶段复用Frozen分数和已有学生，不重新训练，也不重跑完整RMOT；三新臂全部结束后才生成并发布完整FIT_TABLE.md。

### D2已完成：仅教师拟合的阶段记录，不是完整表A或Val结果

DIAG_FIT_OPT_POS完成8973个更新、17源视频/425目标表达/8500教师判断；原lr=0.0002、pos_weight=10，无数值回退，冻结参数版本未变，目标GT打开0。
共同Naive8500上的POS selected=617/656（0.940549），NEG selected=1774/7844（0.226160），pair agreement=0.716418；POS/NEG平均学生概率分别0.901210/0.271442。
按预注册双条件，NEG selected>0.10，故D2尚不具备“拟合合格”资格；不降低阈值，也不以POS单项改善或loss下降替代完整判定。
唯一待评checkpoint为final8973，SHA256=9e149179814fd9a3b669b9d2e858ccdff41ff3edfedd64f24beb639ef4cfab15；source-selected7750只记录、不做目标评测。
D1/D3继续原配置，表A发布门仍未打开。students/DIAG_FIT_OPT_POS保留完整小型训练记录；运行中其它attempt记录只是阶段快照，不是最终成本或完整结果。

### D1随后完成：严格保留80%边界，不事后放宽

DIAG_FIT_OPT原lr=0.0002完成8973更新，同8500判断、17源视频/425目标表达，GT打开0/冻结版本不变。
POS selected=522/656（0.795732），NEG selected=436/7844（0.055584），pair agreement=0.761194；POS/NEG平均学生概率0.719190/0.102303。
80%至少需525/656，D1尚差3个POS，因此严格记为拟合不合格，不将79.57%四舍五入为80%，不挑选中途checkpoint。
final8973 SHA256=b58568b37e4074369d11d672ac0d0bc1aa2814fe52451bba7000af79a4b96d49；source-best2750只记录，不做目标评测。
D1/D2已完成，但D3尚未结束，FIT_TABLE.md及其发布门仍未生成，三新模型的完整Val推理均未开始。
