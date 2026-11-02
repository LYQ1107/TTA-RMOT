# v4：源校准多伯努利数量分支

同域门槛失败，未启动B/C。

## 1. 同域验收

各官方模型在自身源域验证，不是V2→Dance跨域TTA。原Native−1门槛不变，主结果只用源Train固定拟合解，不能改选Native、未校准头、平滑版或其他温度。

| source | model | Native | v3 | v4 | delta |
| --- | --- | --- | --- | --- | --- |
| KITTI-V2 | V2-best | 42.752000 | 39.130000 | 42.593000 | -0.159000 |
| Dance | Dance-best | 32.139000 | 25.942000 | 28.358000 | -3.781000 |
| SportsMOT | LaMOT-best | 69.924000 | 70.360000 | 69.902000 | -0.022000 |

## 2. 从原始文献得到的构造

[Guo et al.](https://proceedings.mlr.press/v70/guo17a.html)支持低维正温度概率校准的思路；[Reuter et al.](https://ba-ngu.vo-au.com/vo/RVVD_LMB_TSP14.pdf)给出多伯努利集合建模。本轮只借用候选概率及数量聚合，不复现该跟踪器、也不把论文的无偏结论套在本方法上。

v3监督单个排序前缀的能量；v4使用候选级源GT拟合两个正斜率，正/负margin各一参数，无偏置、不读PCD/句子/图像。按候选独立近似计算Poisson-binomial数量分布，包含全部候选子集；P(存在)=1−prod(1−p_i)，数量解码为0..n后验的MAP，再用未变的源排序top-k。没有Native数量下界或失败回退。单候选分类/排序保持不保证集合数量保持，本轮HOTA是真实解码输出，不是复制Native。

两个参数只由完整自身源Train候选BCE的一维凸优化决定，范围[.05,20]、xatol1e-6、maxiter128；源验证仅诊断，不选参数。selection.json中的best_source_loss兼容字段仅表示该固定解的源验证NLL，不是按验证选择的最小值。预注册与参数SHA先公开，再预测封存、独立GT评测。

| source | alpha_neg | alpha_pos | train_native_NLL | train_calibrated_NLL | validation_native_NLL | validation_calibrated_NLL |
| --- | --- | --- | --- | --- | --- | --- |
| KITTI-V2 | 0.433742 | 0.391016 | 0.117760 | 0.068893 | 0.892742 | 0.431664 |
| Dance | 3.330125 | 0.164293 | 0.062177 | 0.024604 | 0.406140 | 0.764006 |
| SportsMOT | 0.453220 | 0.680155 | 0.059547 | 0.044988 | 0.741813 | 0.447451 |

| source | validation_native_Brier | validation_calibrated_Brier | source_train_rows | source_validation_rows |
| --- | --- | --- | --- | --- |
| KITTI-V2 | 0.182604 | 0.145499 | 4534302 | 2425382 |
| Dance | 0.057888 | 0.044643 | 1371786 | 3588445 |
| SportsMOT | 0.118539 | 0.110655 | 335246 | 2968201 |

Dance源Train NLL下降而源验证NLL变差，已在HOTA前记录且未换参数。Brier变好也不能消除错误高置信概率的尾部风险；源GT候选到tracker候选支持偏移仍存在。

## 3. 全部已计算指标

| target | model | method | native_HOTA | HOTA | delta_HOTA | DetA | AssA | IDF1 | IDSW | prediction_rows | query_empty_prediction_rows | Present_HOTA | GT_open_attempts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KITTI-V2 | V2-best | PRES_TOPK | 42.752000 | 42.593000 | -0.159000 | 30.870000 | 58.910000 | 46.574000 | 3185.000000 | 739222.000000 | 0.000000 | 42.592855 | 0.000000 |
| KITTI-V2 | V2-best | PRES_TOPK_SMOOTH | 42.752000 | 42.448000 | -0.304000 | 30.781000 | 58.682000 | 46.493000 | 3186.000000 | 730254.000000 | 0.000000 | 42.448265 | 0.000000 |
| Dance | Dance-best | PRES_TOPK | 32.139000 | 28.358000 | -3.781000 | 25.446000 | 31.656000 | 25.892000 | 1236.000000 | 188707.000000 | 77024.000000 | 32.571013 | 0.000000 |
| Dance | Dance-best | PRES_TOPK_SMOOTH | 32.139000 | 28.597000 | -3.542000 | 25.709000 | 31.862000 | 26.078000 | 1265.000000 | 187796.000000 | 75373.000000 | 32.755972 | 0.000000 |
| SportsMOT | LaMOT-best | PRES_TOPK | 69.924000 | 69.902000 | -0.022000 | 70.925000 | 68.918000 | 71.497000 | 4462.000000 | 1564882.000000 | 0.000000 | 69.902130 | 0.000000 |
| SportsMOT | LaMOT-best | PRES_TOPK_SMOOTH | 69.924000 | 69.912000 | -0.012000 | 70.994000 | 68.871000 | 71.530000 | 4826.000000 | 1558117.000000 | 0.000000 | 69.912225 | 0.000000 |

## 4. 召回、数量及局限

| target | native_presence_recall | decoded_presence_recall | native_presence_false_positive_rate | decoded_presence_false_positive_rate | native_count_MAE | decoded_count_MAE | changed_count_units_vs_native | added_prediction_rows_vs_native | removed_prediction_rows_vs_native |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KITTI-V2 | 0.872746 | 0.884842 | 0.374954 | 0.385402 | 0.985386 | 0.916130 | 50940.000000 | 9673.000000 | 46328.000000 |
| Dance | 0.470665 | 0.470969 | 0.055885 | 0.056423 | 0.547706 | 0.501430 | 34849.000000 | 256.000000 | 66459.000000 |
| SportsMOT | 0.910980 | 0.921217 | 0.119864 | 0.135727 | 1.060921 | 1.032202 | 27344.000000 | 16493.000000 | 12029.000000 |

该分支没有生成新视觉语言语义。所有输出仍按原身份排序，HOTA变化只能来自数量改变及后续匹配/关联变化。大量弱候选的noisy-or可放大存在性，条件独立假设也可能使真实相关目标集合的数量被平均化；这些是模型假设和待验证解释，不以一次结果断言唯一因果。不根据本表改变任何温度/阈值/主结果。

后验描述性视频聚类bootstrap（不是确认性盲测/参数选择）：

| target | delta_HOTA | CI95 | clusters |
| --- | --- | --- | --- |
| KITTI-V2 | -0.158868 | None | 4 |
| Dance | -3.781038 | [-6.019440, -1.808414] | 25 |
| SportsMOT | -0.021633 | [-0.064212, 0.023596] | 45 |

## 5. Sports418

主结果全部466序列采用v4 GT，敏感性仅限制原418ID，内容不换回旧冲突模板。

| model | method | HOTA | DetA | AssA | IDF1 | IDSW | Dets |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LaMOT-best | PRES_TOPK | 70.250000 | 71.585000 | 68.963000 | 71.851000 | 4442.000000 | 1545242.000000 |
| LaMOT-best | PRES_TOPK_SMOOTH | 70.250000 | 71.647000 | 68.904000 | 71.878000 | 4809.000000 | 1538780.000000 |

## 6. 成本、隔离与恢复

| source | parameters | function_evaluations | CPU_fit_wall_seconds | GPU_hours | GT_open_attempts |
| --- | --- | --- | --- | --- | --- |
| KITTI-V2 | 2 | 36 | 2.733048 | 0.000000 | 0 |
| Dance | 2 | 33 | 1.703571 | 0.000000 | 0 |
| SportsMOT | 2 | 37 | 0.938848 | 0.000000 | 0 |

本阶段新增GPU小时=0、教师调用=0、目标LoRA更新=0；源凸拟合、头解码和官方评测均CPU。源训练仅允许自身域缓存标签，其他域数值GT打开被拒绝；解码启用runtime_guard并在独立进程封存预测后才评价。冻结分数与全部MOT候选复用，没有图像/PCD重导；原主报告、v3和所有历史预测不覆写。26项定向测试和compileall通过；每行原MOT/完整episode覆盖与进程收尾见validation.json。动态规划小集合精确枚举通过，保存后概率归一化/存在性/均值核对见probability_audit.json。

**本次固定构造未通过：Dance。未生成六跨域新头成绩，未启动B/C，不把未运行写为0 HOTA。**

新缓存/参数/预测在data1既有presence_active_20261102/redesign_v4，经原data2 results/runs链接访问。机器路径与SHA在artifacts.json、selection.json；大文件不入Git。独立日志包logs.tar.gz包含源拟合、全部官方详细表和失败尝试，旧归档不改。
