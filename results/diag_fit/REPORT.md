# D-FIT：学生能否拟合冻结 MLLM 标签

状态：FULL_TRAINING_RUNNING；从公开965e71fe5055a4299ae6f46f89dbe4e993cfad11开始，独立分支research/v2dance-diag-fit-20261030。
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

## 执行计划与隔离

实现/必要回归→三臂200步smoke→旧checkpoint同300窗口子集与全8500判断拟合→三臂8973步。
拟合表A先实际push并核验，随后才允许final-step全425推理；完整预测SHA封存后单独GT评测。
三臂只用相同Naive8500封存标签；PSR基线同时报告其原8500拟合（解释旧38/374）与相同Naive标签拟合（公平比较）。
仅NaN/Inf或源val CE连续2次超过初始3倍时，允许该臂lr=1e-4从头重跑一次，head_lr不变；原失败保留。
最终checkpoint固定最后8973步；source-selected只记录，不对三新臂做source-selected Val评测。

## 2. 表A：拟合教师

待真实实验，不填写占位指标。

## 3. 表B：封存后Val结果

表A发布前不执行。

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
