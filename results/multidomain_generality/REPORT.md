# 三数据集多域普遍性：授权恢复后的真实结果

## 1. 判定汇总

状态：C1、C2独立授权部分中C1已完成、C2进行中；C3–C6仍等待官方LaMOT权重和McByte/FlexHook数据包。**完整六组合未完成，不计算全局2/3判定，不宣称普遍成立/部分成立/不普遍。**

| 源 / 目标 | KITTI-V2 | Dance | SportsMOT |
|---|---|---|---|
| V2-best | 同域参照 | 0.984627：数量主导 | 待官方资产 |
| Dance-best | 推理进行中 | 同域参照 | 待官方资产 |
| LaMOT-best | 待官方资产 | 待官方资产 | 同域参照 |

| 组合 | gap | ρ | Δrank | 固定规则判定 |
|---|---|---|---|---|
| C1 | 23.920809 | 0.984627 | 0.367741 | 数量主导 |
| C2 | — | — | — | 推理进行中 |
| C3 | — | — | — | 待官方资产 |
| C4 | — | — | — | 待官方资产 |
| C5 | — | — | — | 待官方资产 |
| C6 | — | — | — | 待官方资产 |

ρ使用官方detailed CSV未舍入HOTA计算；表内HOTA展示summary数值。缺失单元不是ρ=0，不删组合来做普遍性结论。

## 2. P0、下载、冒烟和自检

补充预注册先推送0f9c61c66f26a72ace8f044864b8b0f71a8ecc0a；恢复P0先推送e7bf73cf81a1ed6afc775815581e0564a77cbea3，随后才继续P1。原93行PREREG正文和首次P0封存保留。详见[P0_RESUMED.md](P0_RESUMED.md)、[downloads.json](downloads.json)。

Sports seqmap实际466条、45视频；原465为文件末行无换行符造成wc -l少计。原图45视频26970有效帧已链接，没有重新下载。官方LaMOT GitHub原始Sports标注508份、45视频已下载，数字ID名不同于FlexHook文本名，未冒充通过466条一一映射。

百度提取码验证成功且可列官方LaMOT-best（995768578字节）和45个McByte预测；匿名sharedownload errno2且无dlink，缺签名/参数。未把errno2谎称登录错误；公开BaiduPCS-Go客户端规定账号登录为必做，无账号凭据故停止该路径，不索取密码、不借他人登录、不自跑tracker。

需要手动下载的精确网盘路径（分享根实际为/FlexHook）：

- /FlexHook/SOTA_ckpts/LaMOT-best.pth
- /FlexHook/tracker_outputs/McByte-SportsMOT/（45文件逐个列在downloads.json）
- /FlexHook/datasets/LaMOT-main/annotations_v1/val/SportsMOT/
- /FlexHook/datasets/LaMOT-main/sequences/val/SportsMOT_labels.json
- /FlexHook/datasets/LaMOT-main/sequences/val/SportsMOT_RESOLUTION.json
- /FlexHook/datasets/LaMOT-main/sequences/val/TAO_frame2file.json（仅mix构造器辅助元数据，不运行TAO）

目的目录：/data2/user/TTA-RMOT-assets/flexhook/lamot_20261101/flexhook_baidu/；不覆盖已有文件。LaMOT公开仓库无独立数据LICENSE，不能假定MIT；FlexHook代码MIT但单独checkpoint/数据授权未写。

两套KITTI实机smoke strict全部key匹配，原生getitem/forward与冻结编码缓存数值误差≤2.1e-5、无符号翻转；box/grid完全一致。使用官方kitti2.yaml、infer.sh显式rope-swin-tiny/roberta覆盖、FP32、规范RoBERTa BPE、官方car/ped合并/帧偏移；没有额外类别token，所有候选同等打分。1视频估算V2 0.572601、Dance 0.630457 GPU小时，均未触发6小时停止门槛。

V2 KITTI历史2355528行原值保留，只新增69854尾观测分数，完整2425382行已封存，历史npy SHA保持cb0a935c…1051687。完整覆盖原生HOTA如实见下表，不假定等于42.526。

Dance历史32.134/254118行与完整32.139/254910行已核对SHA，新诊断统一完整范围。论文最新版正文只有LaMOT合并56.77，infer.sh合并56.756；未找到Sports子集官方数字，仓库无补充PDF、CVF补充入口403。只有官方资产核验全过后才允许Sports smoke和后续正式推理。

## 3. 所有组合和固定配置

| 组合 | 配置 | HOTA | DetA | AssA | IDF1 | IDSW | 预测行 | Empty行 | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| C1 | NATIVE | 8.218000 | 3.193700 | 21.147000 | 4.441400 | 269.000000 | 99704 | 87371 | COMPLETE |
| C1 | IN_DOMAIN | 32.139000 | 28.354000 | 36.486000 | 27.816000 | 1287.000000 | 254910 | 108123 | COMPLETE |
| C1 | R_SxK_I | 31.771000 | 28.208000 | 35.846000 | 27.598000 | 1379.000000 | 254910 | 108123 | COMPLETE |
| C1 | R_IxK_S | 8.136100 | 3.110300 | 21.286000 | 4.300500 | 274.000000 | 99704 | 87371 | COMPLETE |
| C1 | R_SxK_ORACLE | 45.794000 | 62.668000 | 33.600000 | 46.020000 | 2226.000000 | 201009 | 0 | COMPLETE |
| C1 | R_IxK_ORACLE | 46.757000 | 63.503000 | 34.559000 | 46.702000 | 2148.000000 | 201009 | 0 | COMPLETE |
| C1 | B0_GLOBAL | 11.845000 | 6.403900 | 21.914000 | 7.700100 | 577.000000 | 254910 | 215145 | COMPLETE |
| C2 | NATIVE | — | — | — | — | — | — | — | INFERENCE_RUNNING |
| C2 | IN_DOMAIN | — | — | — | — | — | — | — | INFERENCE_RUNNING |
| C2 | R_SxK_I | — | — | — | — | — | — | — | INFERENCE_RUNNING |
| C2 | R_IxK_S | — | — | — | — | — | — | — | INFERENCE_RUNNING |
| C2 | R_SxK_ORACLE | — | — | — | — | — | — | — | INFERENCE_RUNNING |
| C2 | R_IxK_ORACLE | — | — | — | — | — | — | — | INFERENCE_RUNNING |
| C2 | B0_GLOBAL | — | — | — | — | — | — | — | INFERENCE_RUNNING |
| C3 | NATIVE | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C3 | IN_DOMAIN | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C3 | R_SxK_I | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C3 | R_IxK_S | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C3 | R_SxK_ORACLE | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C3 | R_IxK_ORACLE | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C3 | B0_GLOBAL | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C4 | NATIVE | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C4 | IN_DOMAIN | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C4 | R_SxK_I | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C4 | R_IxK_S | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C4 | R_SxK_ORACLE | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C4 | R_IxK_ORACLE | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C4 | B0_GLOBAL | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C5 | NATIVE | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C5 | IN_DOMAIN | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C5 | R_SxK_I | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C5 | R_IxK_S | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C5 | R_SxK_ORACLE | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C5 | R_IxK_ORACLE | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C5 | B0_GLOBAL | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C6 | NATIVE | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C6 | IN_DOMAIN | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C6 | R_SxK_I | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C6 | R_IxK_S | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C6 | R_SxK_ORACLE | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C6 | R_IxK_ORACLE | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C6 | B0_GLOBAL | — | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |

R_S×K_I和逆向数量交换是事后诊断；I(T)是目标域官方模型，不是可免费获得的Source-Free TTA门控。ORACLE两臂显式使用目标GT数量，仅诊断，不是可部署结果。B0_GLOBAL保持同域总行数，但不含逐帧逐表达的数量信息。

| 目标 | 排序来源 | R@1(原候选召回口径) | Hit@1 | 有效帧 | 全候选AUROC |
|---|---|---|---|---|---|
| Dance | source | 0.157999 | 0.614788 | 42657 | 0.650204 |
| Dance | in_domain | 0.157938 | 0.619945 | 42657 | 0.828769 |

R@1复用旧离线脚本的条件候选召回：在存在IoU≥0.5候选的帧，top1命中的正候选数/该帧正候选数，再对帧求均值；正候选按任一GT IoU≥0.5，tie=(-margin,numeric ID)。不是全局AP，也不是top1命中率（Hit@1另列）。GT轨迹结构统计使用一对一最大基数匹配，二者分母不混用。

## 4. KITTI按表达类别

| 组合 | 类别 | 配置 | HOTA | DetA | AssA | 表达数 | 状态 |
|---|---|---|---|---|---|---|---|
| C2 | car | NATIVE | — | — | — | — | INFERENCE_RUNNING |
| C2 | car | IN_DOMAIN | — | — | — | — | INFERENCE_RUNNING |
| C2 | car | R_SxK_I | — | — | — | — | INFERENCE_RUNNING |
| C2 | pedestrian | NATIVE | — | — | — | — | INFERENCE_RUNNING |
| C2 | pedestrian | IN_DOMAIN | — | — | — | — | INFERENCE_RUNNING |
| C2 | pedestrian | R_SxK_I | — | — | — | — | INFERENCE_RUNNING |
| C5 | car | NATIVE | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C5 | car | IN_DOMAIN | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C5 | car | R_SxK_I | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C5 | pedestrian | NATIVE | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C5 | pedestrian | IN_DOMAIN | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |
| C5 | pedestrian | R_SxK_I | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |

按已导出的句子car/people目标类型分组，保留该类表达的全部候选（不事后按预测类别删框）；官方HOTA.combine_sequences聚合。只有4个验证视频，不做bootstrap。

## 5. 结构统计（ORACLE，仅描述）

| 目标 | 轨迹0表达 | 轨迹1表达 | 轨迹≥2表达 | 空表达率 | Present GT帧率均值 | p50 | 状态 |
|---|---|---|---|---|---|---|---|
| Dance | 0.166185 | 0.832370 | 0.001445 | 0.905882 | 0.973752 | 1.000000 | COMPLETE_OFFLINE_ORACLE |
| KITTI-V2 | — | — | — | — | — | — | NOT_YET_EVALUATED |
| SportsMOT | — | — | — | — | — | — | BLOCKED_OFFICIAL_ASSETS |

轨迹正例：一对一IoU≥0.5匹配观测达到该物理预测轨迹≥50%；不是预测ID=GT ID。逐视频表达数、共享文本见structure_stats.csv；Dance17句跨25视频共享。Sports未知项留空，不能以现成图像替代官方McByte轨迹统计。

## 6. 不确定性

| 比较 | ΔHOTA | 95%视频聚类CI |
|---|---|---|
| C1__R_SxK_I_minus_NATIVE | 23.553068 | [14.685326142991581, 33.6424827678502] |
| C1__IN_DOMAIN_minus_R_SxK_I | 0.367741 | [0.09069506709065518, 0.7193641061779391] |

每次2000抽样、seed0，整视频为聚类单位，复用官方TP加权HOTA充分统计且对COMBINED自检；不是对相关表达/帧独立重采样。KITTI仅4视频，点估计。Sports45视频未运行，不伪造CI。

## 7. 成本

已记账GPU进程时间共0.165124小时（含本轮失败环境尝试和smoke；若C2仍运行，其未完成时长暂不计）。每次job详见costs.json。最多2物理GPU，入场按进程少/显存足够共享，不终止别人。训练、微调、MLLM调用均0。

C1七套官方CSV全部在425份预测逐行一致和原SHA核验后复用，未重跑健康推理/评测。新KITTI评测记录在official_results.json；原始日志、margin、预测保留在服务器，Git只存小型表/清单/压缩指标日志。

## 8. GT审计、失败和恢复

模型/GT-free解码的GT打开尝试为0；官方评价和oracle在独立进程、预测SHA封存后执行，未向模型写回标签、阈值或参数。纯文本runtime从旧白名单元数据导出，不打开原表达标签。下载的JSON只复制/解压/哈希，未作为模型GT输入。

首次选错vllmedit环境缺yacs，两子进程6秒退出；切回原dfot（torch2.7.0+cu128/transformers5.4.0），配置回归通过，未修改共享环境。smoke后P1发布锁按要求阻止提前全量运行，远端盘点发布后恢复，0005完成单元未重跑。V2 smoke对旧分数最大重算差0.0010823保留记录，不覆盖旧分数；实际官方forward缓存误差≤2.1e-5且无符号翻转。

所有新代码和run均独立；vendor已有mydataloader旧test补丁原样保留，本轮未改vendor代码；仅新增授权数据软链接。历史结果、源checkpoint和其他项目进程未覆盖。完整范围继续保留，当前仅因缺官方二进制资产停止C3–C6。

恢复路径：results/runs/multidomain_generality_20261101；大文件SHA见margin_paths.json、all_prediction_files.csv.gz、official_results.json和downloads.json。手动资产到位后先strict/45视频/466序列/smoke门槛，再按同一PREREG继续；不生成替代tracker，不开展方法或训练。

官方来源：[FlexHook](https://github.com/buptLwz/FlexHook)、[DATASET.md](https://github.com/buptLwz/FlexHook/blob/main/datasets/DATASET.md)、[LaMOT](https://github.com/Nathan-Li123/LaMOT)、[论文v5](https://arxiv.org/html/2503.07516v5)。
