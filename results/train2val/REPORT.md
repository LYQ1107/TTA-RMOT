# D1 Train→Val 泛化实验

分支：`research/v2dance-train2val-20261031`；起点：`619390d44a09f362eac5cce0f4b1df62c20ec07a`。

状态：P0 只读盘点通过；P1 候选转换完成，P2–P5 尚未运行。不得将历史同集诊断数值当作本轮结果。

## 1. P0：数据与配对盘点

40 个 Train 视频的图像与原始 DanceTrack `gt.txt` 均齐全：41,796 帧、348,930 行框、419 个视频内物理轨迹 ID。
逐视频帧号从 1 连续到 `seqinfo.ini` 的 `seqLength`，无缺帧；GT 六列的框宽高有效，帧号与图像对齐。
逐视频路径、帧数、轨迹数、框行数见 `p0_videos.csv`，机器可读盘点见 `p0_inventory.json`。

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

P1 转换保留全部 348,930 行的 frame、id、box 原始文本和行序，419 条轨迹；候选–表达组合共 13,608,270。
新候选与 runtime 位于 `/data2/user/TTA-RMOT-assets/train2val/20261031`，逐文件 SHA 见 `candidate_receipt.json`；没有修改任何原数据。

## 4. 表 B / C、归因与 bootstrap

NOT_RUN。历史参照仅用于后续相同口径比较：Frozen 8.218、同集 D1 29.702、同集 D3 27.768、B1_GLOBAL 13.848、完整官方 32.139。
预测先逐文件 SHA256 封存，再由独立离线进程打开 Val GT。

## 5. 预注册判定

设 G 为 T2V_D1 三 seed HOTA 均值。若 s17−B1 的视频聚类 bootstrap 95% CI 下界>0 且三 seed 均>13.848，记为“MLLM 存在性知识可跨视频、跨表达泛化”；若 CI 包含0或任一 seed≤13.848，记为“同集增益未能泛化到未见视频和表达”。其他情况如实描述，不新增规则。
上述字面判定须与第1节的 8 类表达文件名重合限制同时解读，不能证明全表达未见。
保留率=(G−8.218)/(29.702−8.218)，≥0.7“大部分保留”，[0.3,0.7)“部分保留”，<0.3“基本未保留”。

## 6. GT 隔离、成本、失败与恢复

P0 未运行模型、教师、优化或 GT 评测，新 GPU 成本0。
训练白名单、故意违规访问测试、采样复现、split=train 加载测试尚待后续集成，不提前标为通过。
一次基础 `python` 命令不存在，已使用系统 `python3` 完成盘点；不影响任何实验数值。
原始数据、旧教师缓存、旧 checkpoint 和历史报告不覆盖。
