# 最终验证记录

```text
CUDA_VISIBLE_DEVICES='' /home/szgx/anaconda3/envs/dfot/bin/python -m pytest -q tests/test_count_structure.py tests/test_diag_attrib_v2.py
17 passed in 0.47s
```

8项新必要回归＋9项既有v2回归，不把6/7项阶段重跑累加。新回归覆盖原topk_ids的tie/越界、轨迹及因果累计均值、单句分配、平移拒绝、Saerens固定点/趋空先验、SOM失败为UNKNOWN、固定17+none选项、源预测MOT帧/ID offset、用户粒度判定规则。

所有新增脚本及structure_prompts.py的compileall通过，`git diff --check`通过。相对61a388e8，原vendor、yestrack_backend.py、visual_input_builder.py、历史train2val和diag_attrib结果diff为空，旧源码路径未patch；新增问法走新函数。

实际全量/真实检查：

- 四库原margin共1700文件＋四个float64缓存SHA核验；每库3588445候选–表达观测，25视频×17表达，包含完整211085个原MOT观测。
- 源原始预测MOT与源GT分开，frame=raw_frame−1、ped ID offset=max(car ID)，完整2355528行/862query均对齐；IoU一对一标签354658正例，Platt14次两参数迭代收敛，无学生/教师更新。
- 28套C预测各425episode，生成进程Dance GT打开0、教师调用0；C2/C3每库输出211085行，C4的三个全空结果也保留425个空文件，不跳样本。
- 单GPU0的真实smoke＋完整固定probe退出0；T1/T2/T3分别6426/378/850逻辑判断，smoke缓存51/3/17复用，既无重复收费也无遗漏；模型冻结参数完整SHA前后一致，目标GT打开0，更新0。
- T3解析失败9个，保留原文和UNKNOWN；候选有效6278/6426，帧query有效841/850，条件指标附覆盖，全请求准确率将失败计错，不当NEG。原版8B本地未找到，明确跳过，不下载/不借其他项目微调权重。
- 全部GT-free预测先封存，之后独立oracle读取425个官方GT；F_FRAME全部425个原始文件与旧v2逐SHA相同，201009行、HOTA45.794，停止门通过。
- 6 oracle＋28 C＋2严格等价参照＝36套×425＝15300个封存预测文件；33套新官方TrackEval首次成功，3套逐行等价复用。完整GT212541/40 Present/385 Empty口径不改。
- 每套官方CSV以25视频TP加权重组均与全精度COMBINED一致；2000次、seed0的60个配对比较全部报告，相关帧/表达不作为独立样本。
- 说明“任一选择器”的全部28范围及Frozen子集时，60组比较的规范JSON SHA保持`f1db65a37d08f2e3fba093c4f5d0b8e4e04614dd0ed3f5e1da537de40d0cc130`不变，未改预测、区间、超参或seed；初稿路径/SHA留档。完整scope说明不是新增阈值或选择赢家。

最终report/completion、官方归档及原预测SHA分别复核。成本诚实标注：GPU进程elapsed0.131604909小时，包含smoke/full加载与CPU段；33个CPU job wall合计869.035943秒，不冒充实际并行wall或CPU core-hours；最初CPU失败耗时未单独记录，不伪填0。

无任何新学生checkpoint、完整训练、密集标注或额外模型试验。完成后停止。
