# D-FIT：学生能否拟合冻结 MLLM 标签

状态：COMPLETE_STOP；从公开965e71fe5055a4299ae6f46f89dbe4e993cfad11开始，独立分支research/v2dance-diag-fit-20261030。
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

## 3. 表B：封存后Val结果

# 表 B — 固定final-step完整25Val/425表达

Frozen与D0旧Naive直接复用完整预测和官方CSV，未重跑；不可训练臂记N/A，不用smoke或source-best冒充final8973。

| 指标 | Frozen | D0 Naive(s17) | D1_OPT | D2_OPT_POS | D3_CAP |
|---|---:|---:|---:|---:|---:|
| HOTA | 8.218000 | 7.449900 | 29.702000 | 22.844000 | 27.768000 |
| DetA | 3.193700 | 3.746900 | 28.983000 | 15.155000 | 21.355000 |
| AssA | 21.147000 | 14.834000 | 30.515000 | 34.476000 | 36.157000 |
| IDF1 | 4.441400 | 4.683300 | 27.621000 | 16.194000 | 22.109000 |
| IDSW | 269.000000 | 283.000000 | 2119.000000 | 2160.000000 | 2101.000000 |
| 预测行数 | 99704.000000 | 21100.000000 | 345302.000000 | 996330.000000 | 695148.000000 |
| Present HOTA | 9.745608 | 7.635013 | 37.271181 | 38.947697 | 43.876648 |
| Present FP（离线行匹配） | 1996 | 1234 | 52186 | 136371 | 74879 |
| Empty FP | 87371 | 10783 | 156758 | 686166 | 446071 |
| 同帧 GT R@1 全部GT帧 | 0.156803 | 0.153885 | 0.161229 | 0.161959 | 0.237572 |
| R@1 无教师标签帧（旧outside-window口径） | 0.156615 | 0.153624 | 0.160755 | 0.161810 | 0.237412 |
| R@1 只排除实际教师端点 | 0.156704 | 0.153711 | 0.160814 | 0.161830 | 0.237431 |
| 全候选 AUROC | 0.650204 | 0.755405 | 0.937848 | 0.913927 | 0.940641 |
| Frozen@方法计数 HOTA | 8.218000 | 7.262100 | 29.400000 | 22.854000 | 27.103000 |
| 方法@Frozen计数 HOTA | 8.218000 | 8.027400 | 7.897800 | 7.838000 | 8.158900 |
| 固定 PSR correct-POS374 选中数 | 30 | 32 | 322 | 349 | 366 |
| 视频95% CI: 方法−Frozen | [+0.0000, +0.0000] | [-7.0827, +4.7416] | [+14.1088, +30.5131] | [+9.1929, +21.6902] | [+13.2182, +28.1607] |
| 视频95% CI: 方法−D0 | [-4.7416, +7.0827] | [+0.0000, +0.0000] | [+16.8499, +29.7091] | [+11.2199, +21.4259] | [+15.0536, +28.0418] |

outside-window沿旧口径排除anchor±1三帧；本轮仅端点有直接标签，因此另报只排除实际端点的R@1，不混淆分母。
Present FP沿旧IoU≥0.5离线匹配（Frozen1996），不同于官方CLEAR的2005；官方CLEAR随完整CSV保留。
374是历史PSR教师正确POS固定集合，并非三新臂实际使用的Naive363正确POS；全部GT只由此封存后独立进程读取。
Count-matched是继承父模型成本的后验诊断，不是独立部署系统；25视频聚类2000次/seed0，不以帧作为独立样本。

## 4. 按预注册规则判定

预注册规则标签：`教师和学生的输入不一致，或者教师标签自相矛盾`。

按用户预注册分支原样归类，三臂拟合资格均为False。本轮三臂实际均可正常训练且有完整最终指标；这个标签不能当作输入差异或标签矛盾已经被因果证实。端点一致率与真实RMOT结果一并呈现，不改验收门槛。

教师相邻端点一致率：

```json
{
  "pairs": 4250,
  "endpoint_offset": 1,
  "binary_POS_agreement": 0.9745882352941176,
  "YesNo_half_agreement": 0.9534117647058824,
  "original_three_state_agreement": 0.9049411764705882,
  "abs_probability_difference": 0.044810425253475415,
  "primary_POS_threshold": 0.95,
  "target_GT_used": false,
  "used_for_training_or_tuning": false
}
```

### 实证结果与归因（不改变上述预注册判定）

同Val诊断中确有大幅整体改善：D1/D2/D3 HOTA为29.7020/22.8440/27.7680，相对匹配Frozen8.2180分别+21.4840/+14.6260/+19.5500；25视频聚类bootstrap的差值95%区间均为正。这是开发集同集适应结果，不是盲测或Train→Val泛化结论。

数量对照解释了大部分HOTA变化：Frozen采用各模型逐帧k后分别为29.4000/22.8540/27.1030，与模型本身仅差+0.3020/−0.0100/+0.6650。反向固定Frozen输出数量时，新模型HOTA为7.8978/7.8380/8.1589，均未超过8.2180。因此不能把整体增益全部归为候选语义排序恢复。此对照本身不读GT，但需要适应模型提供每帧、每表达的k，包含学到的数量/存在性信息，并继承其计算成本；不是免费、独立可部署的Frozen系统，也没有做GT选阈值。

CAP仍有独立可见的排序信号：无教师标签帧R@1从0.156615到0.237412（约+8.08百分点），D1/D2为0.160755/0.161810，提升较小。全候选AUROC分别0.937848/0.913927/0.940641；全局跨帧/跨表达AUROC与同帧R@1不是同一个问题，不能混用。

误检和身份问题尚未解决。Empty FP从87,371增加到156,758/686,166/446,071；更多参数或更高POS选中率没有自动得到最佳整体方法。原始IDSW也从269增加到2119/2160/2101，不能只凭AssA上升宣称身份稳定性问题已解决。计数增加与覆盖率变化会影响这些原始计数，完整CSV同时保留。

预注册门槛仍逐字执行：D1差3个POS，D2/D3的NEG选中率超过10%，三臂都未合格，不重新挑checkpoint或放宽阈值。上面的规则归类不是已经证明“输入不一致/教师自相矛盾”的因果结论；真实RMOT改善、正确POS恢复与排序数据也不支持写成“学生完全学不进去”。本轮到此停止，不自行启动密集GT、列表式蒸馏或正式Train→Val训练。

## 5. 成本、checkpoint与GT隔离

三新臂训练、教师拟合与完整推理的目标GT打开数实际均为0；全部425条预测封存后，才由独立CPU评价进程读取目标GT。评测后对1275个episode的预测/margin文件、三份原始run_metadata和最终checkpoint再次核验通过，见[post_eval_seal_audit.json](post_eval_seal_audit.json)。

| 方法 | 训练GPU / 推理GPU | 成功full训练worker-body小时 | 完整推理GPU-worker小时 | 单方法实际wall秒 | 推理峰值allocated GiB |
|---|---|---:|---:|---:|---:|
| D1_OPT | 1 / 4 | 4.4000 | 0.8464 | 3051.88 | 4.9146 |
| D2_OPT_POS | 5 / 1 | 4.2320 | 0.8488 | 3060.25 | 4.9146 |
| D3_CAP | 4 / 4 | 5.9683 | 0.8729 | 3147.21 | 4.9161 |

训练尝试总计15.3862 GPU-process小时，包含有效smoke与保留的实现失败尝试；另有单列的基线拟合/兼容验证和继承的教师/Frozen成本，不能当作免费。三模型完整推理合计2.5681 GPU-worker小时；推理矩阵实际wall为6199.17秒，真实使用GPU1/4两条lane（不是宣称四卡加速），排队计入矩阵wall，CPU文件锁等待不计GPU推理。GPU-worker/process小时不是独占GPU利用率积分；未单独测量的验证开销在costs.json中明确标记。

运行清单、各checkpoint路径/SHA、全部成本和源版本分别见[server_artifacts.json](server_artifacts.json)、[costs.json](costs.json)、[execution_roles.json](execution_roles.json)及各provenance。完整预测/大checkpoint不进入git，封存路径和SHA在[prediction_manifest.json](prediction_manifest.json)。

离线父命令曾在最后打包阶段因派生count-control元数据被误认成GPU方法而退出1；正式9个评测、密集归因、bootstrap和表格均已完成。保留原stdout，补18项回归后只恢复package并成功退出0，没有重跑GPU、TrackEval或密集归因，见[packaging_recovery.json](packaging_recovery.json)和runtime_logs.tar.gz。未隐藏该失败，不把工程修复当作科研成功。

以下保留历史阶段记录，其中的运行状态指当时，不代表最终状态；最终结果以表A/B为准。

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
