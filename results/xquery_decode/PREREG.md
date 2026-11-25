# 跨句竞争解码：固定协议，2026-11-01

项目 LYQ1107/TTA-RMOT，起点000e196d3609b3fa09ed04b0f469754be00f5ead；新分支research/v2dance-xquery-decode-20261101。
先发布本协议，再执行新预测或评测。既有结果、checkpoint、vendor代码不改。

## 研究计划与输入

1. 核验六套已封存Dance margin及checkpoint；X0对FROZEN、T2V_D1_s17、T2V_D3_s17、SAME_SET_D1必须425文件逐字节等于上轮C3，否则停止。
2. 在同一完整候选库211085个MOT观测、25视频×17句=425序列上，全部报告六模型×六规则，无新增规则，无选择性上报。
3. 唯一新GPU工作为KITTI-V2官方验证候选上的学生checkpoint打分，最多一张显存准入的共享GPU，不干预他人进程；无训练、优化或Dance教师调用。
4. 每个模型需要自己的源域拒绝阈值：目前只有FROZEN有完整源域分数，因此除第5节的T2V_D1_s17外，其他四个学生也必须在同一KITTI验证候选上补分数才能执行第4节。此为源域打分，不是新增解码变体；只有FROZEN与T2V_D1_s17做第5节的完整源域边界评测。
5. GT-free解码/教师缓存组合沿用v2 deny_gt，目标GT打开尝试必须0；源域GT仅由独立阈值/源域评价进程读取。所有目标预测和教师组合先封存，再由独立进程评测。

分数顺序：FROZEN、T2V_D1_s17、T2V_D1_s23、T2V_D1_s42、T2V_D3_s17、SAME_SET_D1；最后一个明确为“同集测试时蒸馏参照”。保留原float64分数，不重新量化。

## 六个固定解码器

观测按帧号、数字track ID排序；句子按文件名字典序，argmax并列取第一个。轨迹均值直接复用count_structure.track_mean/causal_track_mean。

- X0_C3：原始margin整轨迹均值argmax，输出获胜轨迹全部观测，离线。
- X1_C3_CAUSAL：原始margin截至当前帧的轨迹均值argmax，仅输出当前观测，因果。
- X2_C3_Z：每(v,q)全部观测上总体均值/标准差(ddof=0)标准化后，执行X0；sigma=0时z=0，离线。
- X3_C3_Z_CAUSAL：每帧先纳入当前帧全部观测，统计截至当前帧的句子总体矩；观测数<50时该帧用原margin，否则sigma=0取0、sigma>0取z。每个历史观测保留其发生时的因果z，再对同一track累计均值argmax，仅输出当前观测，因果。
- X4_C3_Z_REJECT：X2分配后，获胜句子的原始整轨迹均值<theta_M则整条拒绝，等于阈值保留，离线。
- X5_C3_Z_REJECT_CAUSAL：X3分配后，获胜句子的原始因果轨迹均值<theta_M则拒绝当前观测，因果。

因果统计不读取未来值，且不使用未来观测数来决定fallback。没有任何强制正例补救；拒绝导致空输出也原样保留。

## 源域阈值与适用边界

FROZEN源域result.json SHA256=d6fdc6dff6d6bece1961e3b888ba15a31a88bfddf8557f8a34fc0eae44de5e46，固定2355528行、862表达、4验证视频。所有学生严格使用相同候选/表达/历史窗口覆盖；不补齐源域历史dataloader跳过的末观测，不混用完整Dance协议。
源域Temp-NeuralSORT合并car/pedestrian，ped ID加max(car ID)，图像帧=原MOT帧−1，输出评测MOT帧=图像帧+1，框与ID不变。学生FP32/eval/no_grad，加载原source checkpoint+学生最终步state，核查冻结编码器相同；无参数更新。
每帧候选与该表达GT按IoU>=.5最大基数优先匹配，轨迹>=50%评分覆盖观测匹配则为正例，不用预测ID代替GT身份。
每M独立枚举原始轨迹均值的全部不同值（保留>=阈值）以及最大值之后的空集边界，精确最大化源域轨迹F1；并列取最小阈值。不用Dance指标或标签。
FROZEN/T2V_D1_s17各评测原生margin>0、X0、X2、X4，原官方KITTI-V2 seqmap、gt_template与TrackEval，报告源域0/1/>=2句正例轨迹分布，不参与目标判定。

## 教师组合，仅复用封存probe

沿用count_structure的5视频50帧、378候选、17句。T1_ARGMAX对17个p_yes取最大为唯一正例；T1_ARGMAX_REJECT仅在最大p_yes>=.5保留；T2_ARGMAX取全部18选项argmax，R为空。并列按固定选项顺序；输出硬解码0/1，AUROC若列出仅为硬分数描述，不能冒充新概率。不重新调用教师；所有组合全部报告、分seen/unseen，不选问法。GT由独立进程读取，指标口径复用上轮IoU匹配/帧GT数量。

## 评测、bootstrap、五项结果

36套完整目标预测：HOTA/DetA/AssA/IDF1/IDSW/行数/Query-Empty行数/Present HOTA/seen及unseen HOTA/是否未来帧/GT打开次数，另三seed均值、样本标准差(ddof=1)。
复用官方TP加权池化与25视频配对bootstrap，2000次、seed0；每套分别−Frozen原生、−B1_SRC_RATE_GLOBAL(13.848)、−OFFICIAL_DANCE_COMPLETE(32.139)。另对每个M报告X1−X0、X3−X2、X5−X4、X4−X2及其CI，共132比较；不做多重比较选择，不把帧/表达当独立样本。

1. 确认：s23、s42的X0−B1之95%CI下界均>0，记“跨句竞争解码在Train→Val下可复现”；否则“未能复现”。
2. 去偏：D1三seed的X2在未见句子上的HOTA均值>14.229记“句子标准化恢复了没见过的句子”；否则“未恢复”。
3. 因果：各M三组差值与CI，只报告不判定。
4. 拒绝：各M X4−X2差值、CI、Query-Empty行数变化，只报告不判定。
5. 后续默认：去偏成立用X4/X5；否则X0/X1；本轮不启动任何后续实验。

交付results/xquery_decode/REPORT.md、decoders.csv、thresholds.json、source_boundary.csv、teacher_combo.csv、bootstrap.json、预测SHA与服务器路径、精简日志和完整评测日志。完成push后停止。
