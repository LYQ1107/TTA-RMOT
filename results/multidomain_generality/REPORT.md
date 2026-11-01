# 三数据集多域普遍性：补充授权后继续执行

## 最新阶段（2026-11-01，优先于下方历史盘点）

补充PREREG已推送0f9c61c66f26a72ace8f044864b8b0f71a8ecc0a。完整C1–C6范围保留，Sports466序列已获授权。最新恢复盘点、strict实机smoke与成本门槛见[P0_RESUMED.md](P0_RESUMED.md)，公开下载逐文件清单见[downloads.json](downloads.json)。

C1复用、C2新推理、KITTI补分在新授权范围内继续；LaMOT权重与McByte官方文件尚未成功下载，其相关组合仍受资产门槛阻塞。以下原始P0停止记录仅为历史，不表示忽略新授权；原始机器盘点另存results/runs/multidomain_generality_20261101/history_p0_stop，旧提交8e16d1bc可完整恢复。最终结果未出前不判三域普遍性。

---

# 首次P0盘点与停止（修订前历史记录）

## 1. 判定结果汇总

状态：**STOPPED_P0_MISSING_ASSETS_AND_SEQMAP_MISMATCH**。研究问题尚未回答，不判“普遍成立 / 部分成立 / 不普遍”。

项目是TTA-RMOT（不是LocateMOT/GroundJEV）。起点321ec548，新分支research/multidomain-generality-20261101。用户完整任务已先以PREREG.md提交并推送，远端提交98e7f84efbe43493761063e7832de7bfe9d4e18d。发现缺权重和数据包后遵守停止条件，未启动任何冒烟、P1或P2；未只挑现成的C1或先跑C2。

| 源模型 / 目标 | KITTI-V2 | Dance | SportsMOT |
|---|---|---|---|
| V2-best | 同域参照：历史已核验；完整覆盖待补 | C1：历史可复用，本轮ρ未重算 | C3：未判定，P0停止 |
| Dance-best | C2：未判定，P0停止 | 同域参照：历史及完整结果已核验 | C4：未判定，P0停止 |
| LaMOT-best | C5：未判定，缺权重 | C6：未判定，缺权重 | 同域参照：缺权重及数据包 |

机器文件rho_matrix.csv中ρ/gap/Δ_rank均为空，不能把尚未执行的组合当ρ=0或“排序主导”，也不能对不完整的组合集合套用2/3规则。已知C1现象没有被取消，但不能据单个历史组合得出普遍性结论。

## 2. P0：资产、冒烟、自检

### Checkpoint

| 模型 | 实际路径 | SHA256 |
|---|---|---|
| V2-best | `/data3/user/TTA-RMOT-assets/flexhook/SOTA_ckpts/refer-kitti-v2-best.pth` | `71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613` |
| Dance-best | `/data3/user/TTA-RMOT-assets/flexhook/SOTA_ckpts/refer-dance-best.pth` | `fbb5df7339a1201f34300c105df81e893a5edbb8ef939b7f6eaf8780348eac6c` |
| LaMOT-best | 预期 `/data2/user/FlexHook-TTA/vendor/FlexHook/SOTA_ckpts/LaMOT-best.pth` | **缺失，未产生SHA** |

已按文件名扫描/data2/user、/data3/user、/home/user，未找到LaMOT-best.pth或LaMOT-best.pt。未下载、未用其它模型或其它项目权重代替。两份已存在checkpoint只计算文件哈希，未加载模型；不能将SHA核验当作strict加载通过。

### 三个目标的候选库

| 目标 | 视频 | seqmap表达序列 | 完整MOT观测 | 完整候选–表达对 |
|---|---:|---:|---:|---:|
| KITTI-V2 | 4 | 862 | 10822 | 2425382 |
| Dance | 25 | 425 | 211085 | 3588445 |
| SportsMOT | 45 | **实际466，提示写465** | 缺McByte，未知 | 未知 |

KITTI tracker：`/data2/user/FlexHook-TTA/vendor/FlexHook/tracker_outputs/Temp-NeuralSORT-kitti2`；Dance tracker：`/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs`。两者所有候选对应图像均存在；逐视频行数、文件哈希、表达数及源图像计数见inventory.json。KITTI保留官方car/pedestrian合并规则及图像帧=原MOT帧−1；这里没有写出或改动任何MOT预测。

**KITTI完整覆盖不是旧的2,355,528分数。** 现有历史10508个物理观测，完整10822个，缺314条轨迹的末观测；按每视频197/228/207/230条表达展开，历史2355528对、完整2425382对，尚缺**69854**个分数。后续应复用历史，按完整候选仅补这些缺分，不重跑健康的旧观测。历史HOTA42.526不能冒充未测的完整覆盖HOTA。

### SportsMOT已有原图，但官方LaMOT封装和前端缺失

只读找到原始SportsMOT Val图像：`/data2/user/LocateMOT_FullDataV2_spill/datasets/SportsMOT/val`，45个官方seqmap视频全部匹配seqinfo.ini的seqLength，共**26970**个有效6位数字jpg帧。未把macOS资源分叉文件当图像；未读取gt.txt。这只是其它目录下的原始图像盘点，没有借用LocateMOT的代码、标签、预测或checkpoint，没有修改该目录或建立链接。

TTA官方预期布局缺：

- `/data2/user/FlexHook-TTA/vendor/FlexHook/datasets/LaMOT-main/sequences/val/SportsMOT-sequences`
- `/data2/user/FlexHook-TTA/vendor/FlexHook/datasets/LaMOT-main/annotations_v1/val/SportsMOT`
- `/data2/user/FlexHook-TTA/vendor/FlexHook/datasets/LaMOT-main/sequences/val/SportsMOT_labels.json`
- `/data2/user/FlexHook-TTA/vendor/FlexHook/datasets/LaMOT-main/sequences/val/SportsMOT_RESOLUTION.json`
- `/data2/user/FlexHook-TTA/vendor/FlexHook/datasets/LaMOT-main/sequences/val/TAO_frame2file.json`
- `/data2/user/FlexHook-TTA/vendor/FlexHook/tracker_outputs/McByte-SportsMOT`

其中SportsMOT-sequences是缺链接而非服务器没有原图；annotations_v1及SportsMOT_labels/RESOLUTION是未准备的LaMOT/FlexHook数据封装。TAO_frame2file.json是当前mix构造函数无条件读取的辅助元数据，不代表要运行TAO子集或下载TAO图像。McByte-SportsMOT必须是真正的官方预测，不能用原始SportsMOT gt.txt或其它项目轨迹替代。

### 465与466的原因已定位

本地官方seqmaps/SportsMOT.txt有466条有效、唯一的(video,expression)记录，45视频，每视频7–14条；**最后一行无换行符**，因此wc -l打印465。官方TrackEval的首行跳过逻辑已被注释，不会把第一条当表头；按现有代码会读取466条，不是465。

原文件SHA256为`cc59ba459690d7225e2b546f441e46c1ab3c2a9c91b9842b18dc0327de2a2e7a`，与/data2/user/reference_repos/FlexHook中的副本一致。没有删首行、去重、补表头或修改vendor来凑465。恢复前需要明确按完整官方466条继续，或提供经核验的不同465条官方seqmap；原预注册保持不改。

### 已有自检结果：只核验，不重跑

| 已封存官方结果 | 观测覆盖 | HOTA | 预测行数 | 本轮动作 |
|---|---|---:|---:|---|
| V2-best→KITTI-V2 | 历史 | 42.526 | 750834 | summary/detailed及862份预测SHA核验 |
| Dance-best→Dance | 历史210393观测 | 32.134 | 254118 | summary/detailed及425份预测SHA核验 |
| Dance-best→Dance | 完整211085观测 | 32.139 | 254910 | summary/detailed及425份预测SHA核验 |
| V2-best→Dance | 完整211085观测 | 8.218 | 99704 | summary/detailed及425份预测SHA核验 |

前两个规定数值门槛通过的是**历史结果复用验证**，不是本轮新冒烟。reproduction.json保存完整路径、SHA和实际GT_Dets。官方infer.sh注释的源域GT_Dets=491222，而既有封存源域结果为491286，HOTA同为42.526；未改历史结果或GT来消除这个差异，后续新评测需维持可追溯的统一过滤协议。

LaMOT-best→SportsMOT尚不能自检。仓库infer.sh的56.756确为四子集合并值，不能当SportsMOT单独门槛。因为更早的资产停止条件已触发，论文/补充材料检索尚未进入；这里**不声称已证明“无官方子集数字可核对”**。

### 冒烟、预处理、类别token与GT隔离的已知情况

9个(source,target)冒烟全部NOT_RUN，strict加载NOT_ATTEMPTED，GPU小时估算为空。没有用虚构耗时绕过6 GPU小时停止门槛。

目标配置、变换和dataloader已静态审读：KITTI/Dance用RMOT_Dataset；SportsMOT用RMOT_Dataset_mix；统一224×672及目标配置的norm_mean/std。kitti2.yaml和dance.yaml里的VISUAL=resnet34不是官方best命令的最终值，infer.sh明确覆盖为rope-swin-tiny、text roberta。不会拿未覆盖的YAML去加载best后“修补”结构。

静态数据类别字段：KITTI car=0、pedestrian=1；Dance与SportsMOT候选为0。类别文本token的实际路径与跨checkpoint一致性尚未运行验证，不能标为通过；不按源模型改类别词或文本处理。已存在的compat/sitecustomize.py通过原vocab.json+merges.txt重建ByteLevel BPE、RobertaProcessing，不新改tokenizer，不以字符切分替代正确BPE。

mix dataloader存在同样的exclusive range末观测遗漏；但缺McByte文件，尚不能量化Sports缺分数。另一个实际代码问题是mix test的_parse_data仍无条件open SportsMOT_labels.json（mydataloader_mix.py:270），不能直接运行官方main.py而假称GT打开0。普通loader已有的test不读labels.json补丁是旧工作，保持不动。恢复后需先使mix运行时只接收合法图像/轨迹/表达文本元数据，再执行规定smoke；这不授权修补checkpoint结构或改变类别token。

## 3. 六个跨域组合的诊断

C1–C6均未进入本轮P2。diagnostics.csv已列全6×7个固定配置及ORACLE标记，所有新指标为空、状态NOT_RUN_P0_HARD_STOP。没有把缺失值写成0，没有重跑C1，没有用D1/PSR等非官方checkpoint代替官方LaMOT。

## 4. KITTI-V2类别拆分

per_category.csv列明C2/C5×car/pedestrian×NATIVE/IN_DOMAIN/R_S×K_I的12个待执行项；HOTA为空。没有读取目标类别GT或据此构建输入。

## 5. 三目标结构统计

seqmap及每视频表达数已在inventory.json中盘点；Dance为17句跨25视频共用。GT定义的0/1/≥2正例表达数、空表达率和有GT帧比例都尚未执行。structure_stats.csv明确标记ORACLE_NOT_RUN，不拿旧覆盖下的结构数字冒充完整新协议。

## 6. Bootstrap

bootstrap.json状态NOT_RUN，planned_samples=2000、samples_executed=0、seed=0；没有置信区间或普遍性判定。KITTI未来只报4视频点估计的规则不变。

## 7. 成本

本轮新GPU推理0小时，模型加载0次，训练/微调/MLLM调用均0，下载0，链接修改0，新增TrackEval作业0。只运行CPU盘点、文件校验和静态代码检查；smoke_and_costs.json中估算为null，不把0实际消耗误说成全量推理只需0小时。

## 8. 审计、停止与恢复条件

P0进程安装GT payload打开审计，GT打开尝试0；只检查缺失标注的路径存在性，未打开标注内容。已有预测在历史封存之后生成的summary/detailed允许只读核验；margin_paths.json列三份可复用官方分数库及SHA。旧结果、checkpoint、原始数据、vendor和其它项目进程都未改动。

没有发生模型兼容性报错，因为没有尝试加载模型；不能把缺资产说成模型不兼容。静态发现的mix读取GT和末帧问题、seqmap末行问题均原样记录，没有绕过或偷偷降低验收条件。scripts/multidomain_inventory.py可在CPU上复查这一P0，未实现或启动后续方法实验。

恢复需要：提供官方LaMOT-best及完整LaMOT/FlexHook Sports封装、官方McByte-SportsMOT预测路径（或用户另行明确授权下载），并明确466条seqmap口径。之后先完成GT-free兼容性smoke及成本估算，再推P0更新、进入P1；若任何估算>6 GPU小时或strict不兼容，仍按原规则停止。

**当前已停止。没有把这份阻塞报告标作完整三域实验成功；不训练、不调用MLLM、不自行开始新方法。**
