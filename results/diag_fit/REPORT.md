# D-FIT：学生能否拟合冻结 MLLM 标签

状态：IMPLEMENTATION；从公开965e71fe5055a4299ae6f46f89dbe4e993cfad11开始，独立分支research/v2dance-diag-fit-20261030。
用户允许显存足够时共享、优先进程少的最多4卡，不停止其他任务。25Val同集诊断，不是论文主结果或Train→Val。

## 1. 修改前只读核对

- LoRALinear原scaling=1/rank，rank4为0.25，lora_b全零初始化；只替换最后两个PCD块wq/wv。
- 官方checkpoint有4个output_ca_qkv块；wq/wk/wv/wo均768→768、无bias。
- head有4个MLP，各Linear(768,768,bias=False)→GELU→Linear(768,2,bias=False)，dropout0；共2,365,440参数。
- 原mllm_training.run在attach、容量检查、provenance、最终receipt处固定rank4/24576；每步1目标window，unscale后clip1，在step前。
- 旧含POS步裁剪前direct梯度：Naive中位2.246040/p90 4.884213，PSR中位2.227479/p90 4.746833。PSR含weak的target总范数与direct分开。
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
