# D1 Train→Val 泛化实验

分支：`research/v2dance-train2val-20261031`；起点：`619390d44a09f362eac5cce0f4b1df62c20ec07a`。

状态：P3四臂固定8973步训练进行中；P0–P2与D1/D3 smoke已完成，P4/P5未启动。不得将历史同集诊断数值当作本轮结果。

## 1. P0：数据与配对盘点

40 个 Train 视频的图像与原始 DanceTrack `gt.txt` 均齐全：41,796 帧、348,930 行框、419 个视频内物理轨迹 ID。
逐视频帧号从 1 连续到 `seqinfo.ini` 的 `seqLength`，无缺帧；GT 六列的框宽高有效，帧号与图像对齐。
逐视频路径、帧数、轨迹数、框行数见 `p0_videos.csv`，机器可读盘点见 `p0_inventory.json`。
补充跨归档图像检查：每视频固定首/中/尾，共120帧，原始DanceTrack JPG与运行时Refer-Dance PNG的解码RGB、尺寸全部逐点相等；详见`train_image_alignment.json`，可用`prepare_train2val.py rgb-check`复现，无Val/指代GT输入。

- 运行时 PNG：`/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance/DanceTrack/training/image_02/<Train video>`。
- 原始 JPG 与 GT 的已有只读数据入口：`/data2/user/LocateMOT/data/DanceTrack/train/<video>/{img1,gt/gt.txt}`。
  该入口解析到 `/data2/user/LocateMOT/data/downloads/DanceTrack/extracted/{train1,train2}/<video>`。
  这里只复用合法原始数据，未修改 LocateMOT/GroundJEV 代码、结果或进程。
- 本地 `Refer-Dance.zip`（11,147,254,419 bytes）没有原始 DanceTrack `gt.txt`；其中 `labels_with_ids` 不读取、不替代原始 GT 几何。

Train 每视频 39 个表达文件，40×39＝1,560 个视频–表达组合，共 39 个不同句子。
文件结构为顶层 `sentence: string` 与 `label: object`。独立文本导出进程只解码句子与结构键名，词法跳过 `label` 对象，不解析/导出目标 ID、帧列表或其他指代标签。
训练与教师进程只读取导出的 runtime manifest，不读取原表达 JSON 容器。

Val 的官方 `vendor/FlexHook/seqmaps/dance.txt` 和数据包 `seqmap.txt` 都是 25×17＝425 个序列。
依据 `vendor/FlexHook/data/utils.py:VIDEOS`、`data/mydataloader.py:_parse_data` 的逐视频表达目录枚举、`test_utils.py` 的逐视频输出及 seqmap 文件名，全部表达来自本视频目录，未发现跨视频取文本再配对的生成逻辑。
因此 Train 同样枚举各自目录，不额外生成别的视频表达。
本阶段不打开 Val 图像、表达内容、GT 内容。

这里“本视频表达”仅指文件目录归属，不声称该句在视频中一定有目标、也不推断句子的最初来源。Val每个目录都有相同17个文件名，425等效于25视频×17文本的目录级组合；Train每个目录都有相同39个句子，按同构规则得到40×39=1560。不使用指代标签筛掉空目标表达，也不根据标签重新构造正/负组合。

**划分限制：Train/Val 视频无交集，但有 8 个表达文件名重合（Val 17 类中的 8 类）；它不是严格的全表达未见划分。**
按用户指定原划分继续，不能将最终结果无条件解读为“全部未见表达泛化”。不依据 GT 修改表达集合。

## 2. 固定实验协议

D1：`diag_fit_opt.yaml` 原数值配置，seed 17/23/42；D3：`diag_fit_cap.yaml` 原数值配置，seed 17。
每臂 8,973 次更新、K=4、源 KITTI-V2 计划不变、final-step 唯一评测模型；不使用历史 D-FIT checkpoint 初始化。
Teacher 为相同 revision 的冻结 Qwen3-VL-2B，Naive 配对端点采样 seed 17，每视频–表达最多 20 个判断。
不读取目标 referring GT，不使用 Val 指标决策，不将未判断样本变成负例。

Train 候选按明确授权只使用原始 MOT 的 frame/id/box，conf=1；它们是 **GT 几何/ID，不是预测 NeuralSORT**。
Val 则固定完整预测 DETR+NeuralSORT 211,085 观测。该候选来源差异必须披露，不冒称无 oracle 几何的目标 Train。

## 3. 表 A：Train 教师拟合

NOT_RUN。完成并推送本表前，不开始 Val 学生推理。

三个D1已完成8973步与全Train拟合；阶段性结果见`D1_TRAIN_FIT_INTERIM.md/json`，D3仍运行，尚不打开最终表A发布门。
三seed教师POS选中率仅8.95%/8.67%/7.65%（冻结10.20%），非POS选中率降至约0.40%–0.46%，存在负预测偏置；完整Val尚未运行，不能把这等同GT结果或纯记忆的因果证明。
源dev CE为0.8941/0.9010/0.8982，未数值发散，故不触发6.2学习率回退，也不因正例拟合差改参。

D1/D3 真实smoke各完成200优化步（原LR2e-4、无回退），目标非零梯度均200/200，源模型初始预测max error=0/selection flips=0。
真实训练目标GT与禁止输入打开尝试均0，冻结参数版本不变，D3 head实际更新；仅为接口/数值检查，不是正式8973步结果。
小型记录见 `smoke_summary.json` 与 `smoke/`；重复打包只读seal的问题及修复见 `EXECUTION_NOTES.md`，未重跑已完成smoke。

正式训练从官方V2重新初始化，D1 seed17/23/42分别GPU3/4/5（启动PID680048/680050/680052），D3 seed17在GPU7（PID680054）。
启动数值源码private `a159741`对应公开`5c9fc345`同一Git tree；未使用smoke或旧D-FIT权重。最新stdout报告点为D1三seed4000/D3 3000，不是实时轮询进度；完整结果仍待8973步与全量Train拟合完成。

P1 转换保留全部 348,930 行的 frame、id、box 原始文本和行序，419 条轨迹；候选–表达组合共 13,608,270。
新候选与 runtime 位于 `/data2/user/TTA-RMOT-assets/train2val/20261031`，逐文件 SHA 见 `candidate_receipt.json`；没有修改任何原数据。

Train 的 7,800 个 Naive 窗口、31,200 个判断已冻结，manifest SHA256：`c09a161246ce2635f1307ec0e13d6605f026b2f0a3280cb5a68efc2775961ed9`。
原 sampler 在独立测试进程重放 Val，425 个 query 的全部 8,500 位置、ID、框、selection reason 和 frozen margin 逐条一致（`naive_sampling_replay.json`）；这些旧缓存绝不进入训练进程。
本轮源模型只计算 Train 锚点±1和首帧，40视频全部完成，wall 344.61s；各进程与路径/SHA见 `sparse_source_scores.json`。
教师4样本真实smoke通过，冻结Qwen/单帧YesNo/输入ID/缓存恢复/GT0，见 `teacher_smoke_check.json` 和原始receipt；P1 backend与visual builder均未改。

全Train教师完成：31,200/31,200有效软标签，POS(p≥.95)=1,765（5.657051%）；历史Naive实际656/8500=7.717647%，用户给出的657/8500属于PSR自身缓存。
4个全视频分片GPU进程wall合计788.278秒（0.218966 GPU-process h），总wall310.098秒；Teacher内部body计763.876秒，forward621.275秒，含smoke4个缓存命中后全量新增31,196 prompt forwards。
这些是已判断软概率，不是GT精度，比例不用于选阈值。完整raw logits与框/图片/输入哈希留服务器，公开 `teacher_receipt.json` / `teacher_summary.json`。
封存SHA：`c3939408e541157937088c489b293a3dbddc10caf5de24bae6d243689a4e62cf`；实际PSRPseudoDataset在split=train加载40视频、1560组合、7800组、31200判断通过。
原封存器同时导出兼容用weak文件，但新训练明确expanded_kind=none、expanded loss=0，不读取其标签用于学习。

CPU缓存参照：同一批31200判断中，冻结V2仅选中教师POS的180/1765（10.1983%），非POS统计桶选中率6.9407%，同帧教师pair agreement0.521338。
见`frozen_train_fit.json`与逐视频/逐表达的`teacher_by_*.csv`。这不是GT准确率；统计的“NEG”桶沿D-FIT定义含所有有效p<.95，UNKNOWN仍用原始软概率，不转成硬负例。
固定总更新数意味着每个新窗口曝光少于旧同集实验；若泛化下降，不能仅凭此实验将全部原因归为记忆，还须结合Train拟合与候选来源差异解释。协议不因此增加步数或改参。

## 4. 表 B / C、归因与 bootstrap

NOT_RUN。历史参照仅用于后续相同口径比较：Frozen 8.218、同集 D1 29.702、同集 D3 27.768、B1_GLOBAL 13.848、完整官方 32.139。
预测先逐文件 SHA256 封存，再由独立离线进程打开 Val GT。

## 5. 预注册判定

设 G 为 T2V_D1 三 seed HOTA 均值。若 s17−B1 的视频聚类 bootstrap 95% CI 下界>0 且三 seed 均>13.848，记为“MLLM 存在性知识可跨视频、跨表达泛化”；若 CI 包含0或任一 seed≤13.848，记为“同集增益未能泛化到未见视频和表达”。其他情况如实描述，不新增规则。
上述字面判定须与第1节的 8 类表达文件名重合限制同时解读，不能证明全表达未见。
保留率=(G−8.218)/(29.702−8.218)，≥0.7“大部分保留”，[0.3,0.7)“部分保留”，<0.3“基本未保留”。

## 6. GT 隔离、成本、失败与恢复

P0/P1 新 GPU 成本0；P2 准备的真实源打分与教师 smoke 单列，尚无学生优化或新 GT 评测。
24项不同的相关CPU测试通过，包括全部40文件六列逐行转换、split=train缓存加载、Python审计和kernel-only白名单实际阻止Val图片/MOT/表达manifest/旧标签/旧D-FIT checkpoint（各5类）。故意违规仅发生在独立测试子进程，不能计入真实训练GT0。
一次隔离测试发现Val MOT文件名的video ID是stem而非目录组件：kernel已阻止读取，但Python audit漏记该次尝试；补充stem识别后两层均5/5拦截，未发生真实训练或GT读取。尝试计数现即时落盘，避免失败退出丢失记录。
一次基础 `python` 命令不存在，已使用系统 `python3` 完成盘点；不影响任何实验数值。
原始数据、旧教师缓存、旧 checkpoint 和历史报告不覆盖。
