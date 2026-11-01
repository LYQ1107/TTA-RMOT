# 三数据集多域普遍性：授权恢复后的真实结果

## 1. 判定汇总

状态：六组正式指标已齐，正在最终一致性验收；全部九套分数和GT-free预测已封存，不再跑GPU。C3见[C3_REPORT.md](C3_REPORT.md)：原生50.970、同域数量62.830、同域69.924，ρ=0.625732，为“数量与排序共同作用”。C4见[C4_REPORT.md](C4_REPORT.md)：原生20.560、同域数量61.351、同域69.924，ρ=0.826331，为“数量主导”。Sports v3精确门410/418通过、8失败并停止；用户随后授权并先公开v4，466条统一按官方原始标注重建，共1533675行、0空序列，旧418原样保留。C3/C4统一标注PREREG_V4_ORIGINAL_ANNOTATION_GT，不冒称旧418模板协议复现；其他四组GT和结果不改。详见[SPORTS_GT_CONFLICT_REPORT.md](SPORTS_GT_CONFLICT_REPORT.md)。最终判定和全套机器文件将在验收后统一更新。

| 源 / 目标 | KITTI-V2 | Dance | SportsMOT |
|---|---|---|---|
| V2-best | 同域参照 | 0.984627：数量主导 | 0.625732：数量与排序共同作用 |
| Dance-best | 0.535040：数量与排序共同作用 | 同域参照 | 0.826331：数量主导 |
| LaMOT-best | 0.560254：数量与排序共同作用 | 0.961155：数量主导 | 同域参照 |

| 组合 | gap | ρ | Δrank | 固定规则判定 |
|---|---|---|---|---|
| C1 | 23.920809 | 0.984627 | 0.367741 | 数量主导 |
| C2 | 29.250220 | 0.535040 | 13.600183 | 数量与排序共同作用 |
| C3 | 18.953284 | 0.625732 | 7.093613 | 数量与排序共同作用 |
| C4 | 49.364159 | 0.826331 | 8.573034 | 数量主导 |
| C5 | 20.353418 | 0.560254 | 8.950338 | 数量与排序共同作用 |
| C6 | 16.663089 | 0.961155 | 0.647284 | 数量主导 |

ρ使用官方detailed CSV未舍入HOTA计算；表内HOTA展示summary数值。缺失单元不是ρ=0，不删组合来做普遍性结论。

研究边界：这里衡量固定候选、原生margin>0协议下的数量/排序损失，不证明目标域数量可以无监督估计。C1虽然ρ=98.46%，同域排序仍比源排序高约0.368 HOTA，25视频CI为正；“数量主导”不等于排序完全无损，也不等于新TTA成功。

## 2. P0、下载、冒烟和自检

2026-11-01 当前下载恢复：用户扫描服务器独立Firefox原生网盘界面的官方二维码后，浏览器实际进入`/disk/`、保存私有BDUSS/STOKEN。真实小文件smoke返回HTTP200/210064字节；515项逐文件任务保留290份有效资产，后段225项API失败原样记录。随后官方batch接口下载权重、GT模板、全部表达和辅助JSON；长权重传输后旧签名过期112，按每批刷新签名补齐，不重下载权重。最终933份官方payload共1144635019字节，全部实际SHA核对通过；不是继续停在匿名403。详见[QR_LOGIN_STATUS.md](QR_LOGIN_STATUS.md)、[authenticated_downloads.json](authenticated_downloads.json)和[downloads.json](downloads.json)。

LaMOT-best实际SHA256=`400980a1c92eacaa10206f1f26f2ffff20055b8a3925499afb6c6e7d9ba4213f`；45份McByte共295779观测；466表达与GitHub原始文件逐字节相同、seqmap无缺项；2968201个候选–表达对，26970帧有效原图。Sports只建数字JPG链接视图，原始macOS资源分叉不进入官方目录计帧，不改图像或vendor代码。

LaMOT→Sports/KITTI/Dance及V2/Dance→Sports五套P0已通过strict、原native loader/forward及隔离query对齐，GT打开0；实机GPU2/0，峰值约4.3GiB。对应全量估计0.790154/0.827651/2.119336/0.792368/0.787743 GPU小时，见remaining_p0.json。只发布P0，不把其当完整HOTA结果。

GT资产差异单列：模板包的418条全部属于466条seqmap，缺48条、无额外条目，覆盖视频数仍为45。网盘抽查`v_0kUtTtmLaJA_c004`只有10个表达目录，缺失长句目录请求errno4，证明至少该例不是本地解压漏文件。完整清单见[sports_template_inventory.json](sports_template_inventory.json)。用户已授权补建，v3预注册先发布0a7477bf。此时仅检查包/路径，尚未打开GT值；必须等完整预测封存，再从官方原始标注和原始框逐行复现418，全部通过才接入48。任何失败停Sports评测，不删序列、不改门槛。

下述匿名下载错误和手工清单是此前历史，已经不再代表当前服务器未登录；C1/C2科研结果和原有封存保持不变。

补充预注册先推送0f9c61c66f26a72ace8f044864b8b0f71a8ecc0a；恢复P0先推送e7bf73cf81a1ed6afc775815581e0564a77cbea3，随后才继续P1。原93行PREREG正文和首次P0封存保留。详见[P0_RESUMED.md](P0_RESUMED.md)、[downloads.json](downloads.json)。

Sports seqmap实际466条、45视频；原465为文件末行无换行符造成wc -l少计。原图45视频26970有效帧已链接，没有重新下载。官方LaMOT GitHub原始Sports标注508份、45视频已下载，数字ID名不同于FlexHook文本名，未冒充通过466条一一映射。

百度提取码验证成功且可列官方LaMOT-best（995768578字节）和45个McByte预测；初始匿名sharedownload errno2、无dlink；补齐官网签名接口后errno0，但list是不透明字符串、不是可下载链接，encrypt=0也相同。未把参数错误或不透明响应谎称登录失败；未执行网页JS/外部解析代理。公开BaiduPCS-Go客户端规定账号登录为必做，无账号凭据故停止该路径，不索取密码、不借他人登录、不自跑tracker。

早期手工恢复清单（现已自动下载补齐，保留来源追踪；分享根为/FlexHook）：

- /FlexHook/SOTA_ckpts/LaMOT-best.pth
- /FlexHook/tracker_outputs/McByte-SportsMOT/（45文件逐个列在downloads.json）
- /FlexHook/datasets/LaMOT-main/annotations_v1/val/SportsMOT/
- /FlexHook/datasets/LaMOT-main/sequences/val/SportsMOT_labels.json
- /FlexHook/datasets/LaMOT-main/sequences/val/SportsMOT_RESOLUTION.json
- /FlexHook/datasets/LaMOT-main/sequences/val/TAO_frame2file.json（仅mix构造器辅助元数据，不运行TAO）

目的目录：/data2/user/TTA-RMOT-assets/flexhook/lamot_20261101/flexhook_baidu/；不覆盖已有文件。LaMOT公开仓库无独立数据LICENSE，不能假定MIT；FlexHook代码MIT但单独checkpoint/数据授权未写。

两套KITTI实机smoke strict全部key匹配，原生getitem/forward与冻结编码缓存数值误差≤2.1e-5、无符号翻转；box/grid完全一致。使用官方kitti2.yaml、infer.sh显式rope-swin-tiny/roberta覆盖、FP32、规范RoBERTa BPE、官方car/ped合并/帧偏移；没有额外类别token，所有候选同等打分。1视频估算V2 0.572601、Dance 0.630457 GPU小时，均未触发6小时停止门槛。

逐目标文本/类别处理：KITTI保留car=0/pedestrian=1的官方候选解析和合并ID，但模型没有新加类别token；Dance复用已封存的规范BPE分数库，原17句不改写、候选class=0、无新类别前缀；Sports官方mix静态代码class=0/roberta，实际加载/tokenizer/末观测补分尚未运行，不能标为兼容性通过。Sports原图中存在macOS资源分叉._jpg，盘点仅计有效数字文件名；未来官方链接视图/seqLength也须核对，当前不据此制造Sports结果。

V2 KITTI历史2355528行原值保留，只新增69854尾观测分数，完整2425382行已封存，历史npy SHA保持cb0a935c…1051687。完整覆盖原生HOTA如实见下表，不假定等于42.526。

| KITTI覆盖/模型 | HOTA | 行数 | 逐行相等后复用旧评测 |
|---|---|---|---|
| V2_COMPLETE_REFERENCE | 42.752000 | 775877 | False |
| HIST_V2-best | 42.526000 | 750834 | True |
| NATIVE | 13.502000 | 99889 | False |
| HIST_Dance-best | 13.343000 | 97276 | False |

历史V2分数逐位相同，862份原生预测逐行数值相同，HOTA42.526自检通过。新完整覆盖与历史覆盖均保留，不混用参照。

源域官方封存GT_Dets=491286，infer.sh注释为491222，差64。沿用已通过42.526自检的原GT模板/过滤规则，没有修改GT来凑注释数字；所有本轮KITTI比较使用同一491286协议。

关键反向组合C2并不支持“排序近乎无损”：补同域数量后仍落后13.600 HOTA，GT R@1为0.197195对0.424417。Oracle同量时32.576对47.593，排序损失也没有消失。C2只有4视频，这些差距为点估计，不作统计显著性声明。C2已不满足汇总中要求的“C2也数量主导”必要条件；C5现同样为数量与排序共同作用，其余三组合继续，不提前作六组全局判定。

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
| C2 | NATIVE | 13.502000 | 4.141000 | 44.145000 | 7.914700 | 382.000000 | 99889 | 0 | COMPLETE |
| C2 | IN_DOMAIN | 42.752000 | 30.688000 | 59.696000 | 46.392000 | 3203.000000 | 775877 | 0 | COMPLETE |
| C2 | R_SxK_I | 29.152000 | 18.129000 | 47.001000 | 30.094000 | 3264.000000 | 775877 | 0 | COMPLETE |
| C2 | R_IxK_S | 16.263000 | 6.173300 | 42.890000 | 11.597000 | 482.000000 | 99889 | 0 | COMPLETE |
| C2 | R_SxK_ORACLE | 32.576000 | 22.072000 | 48.273000 | 35.670000 | 2828.000000 | 480848 | 0 | COMPLETE |
| C2 | R_IxK_ORACLE | 47.593000 | 38.588000 | 58.934000 | 54.845000 | 3491.000000 | 480848 | 0 | COMPLETE |
| C2 | B0_GLOBAL | 24.353000 | 11.872000 | 50.103000 | 20.782000 | 1772.000000 | 775877 | 0 | COMPLETE |
| C3 | NATIVE | — | — | — | — | — | — | — | P1_RUNNING_AWAIT_EVALUATION |
| C3 | IN_DOMAIN | — | — | — | — | — | — | — | P1_RUNNING_AWAIT_EVALUATION |
| C3 | R_SxK_I | — | — | — | — | — | — | — | P1_RUNNING_AWAIT_EVALUATION |
| C3 | R_IxK_S | — | — | — | — | — | — | — | P1_RUNNING_AWAIT_EVALUATION |
| C3 | R_SxK_ORACLE | — | — | — | — | — | — | — | P1_RUNNING_AWAIT_EVALUATION |
| C3 | R_IxK_ORACLE | — | — | — | — | — | — | — | P1_RUNNING_AWAIT_EVALUATION |
| C3 | B0_GLOBAL | — | — | — | — | — | — | — | P1_RUNNING_AWAIT_EVALUATION |
| C4 | NATIVE | — | — | — | — | — | — | — | P1_RUNNING_AWAIT_EVALUATION |
| C4 | IN_DOMAIN | — | — | — | — | — | — | — | P1_RUNNING_AWAIT_EVALUATION |
| C4 | R_SxK_I | — | — | — | — | — | — | — | P1_RUNNING_AWAIT_EVALUATION |
| C4 | R_IxK_S | — | — | — | — | — | — | — | P1_RUNNING_AWAIT_EVALUATION |
| C4 | R_SxK_ORACLE | — | — | — | — | — | — | — | P1_RUNNING_AWAIT_EVALUATION |
| C4 | R_IxK_ORACLE | — | — | — | — | — | — | — | P1_RUNNING_AWAIT_EVALUATION |
| C4 | B0_GLOBAL | — | — | — | — | — | — | — | P1_RUNNING_AWAIT_EVALUATION |
| C5 | NATIVE | 22.398000 | 11.567000 | 43.483000 | 20.449000 | 1270.000000 | 332822 | 0 | COMPLETE |
| C5 | IN_DOMAIN | 42.752000 | 30.688000 | 59.696000 | 46.392000 | 3203.000000 | 775877 | 0 | COMPLETE |
| C5 | R_SxK_I | 33.801000 | 22.264000 | 51.433000 | 35.799000 | 3074.000000 | 775877 | 0 | COMPLETE |
| C5 | R_IxK_S | 25.534000 | 15.139000 | 43.188000 | 26.085000 | 1645.000000 | 332822 | 0 | COMPLETE |
| C5 | R_SxK_ORACLE | 37.595000 | 27.691000 | 51.230000 | 42.693000 | 3230.000000 | 480848 | 0 | COMPLETE |
| C5 | R_IxK_ORACLE | 47.593000 | 38.588000 | 58.934000 | 54.845000 | 3491.000000 | 480848 | 0 | COMPLETE |
| C5 | B0_GLOBAL | 26.577000 | 15.058000 | 47.052000 | 25.808000 | 2404.000000 | 775877 | 0 | COMPLETE |
| C6 | NATIVE | — | — | — | — | — | — | — | P1_RUNNING_AWAIT_EVALUATION |
| C6 | IN_DOMAIN | — | — | — | — | — | — | — | P1_RUNNING_AWAIT_EVALUATION |
| C6 | R_SxK_I | — | — | — | — | — | — | — | P1_RUNNING_AWAIT_EVALUATION |
| C6 | R_IxK_S | — | — | — | — | — | — | — | P1_RUNNING_AWAIT_EVALUATION |
| C6 | R_SxK_ORACLE | — | — | — | — | — | — | — | P1_RUNNING_AWAIT_EVALUATION |
| C6 | R_IxK_ORACLE | — | — | — | — | — | — | — | P1_RUNNING_AWAIT_EVALUATION |
| C6 | B0_GLOBAL | — | — | — | — | — | — | — | P1_RUNNING_AWAIT_EVALUATION |

R_S×K_I和逆向数量交换是事后诊断；I(T)是目标域官方模型，不是可免费获得的Source-Free TTA门控。ORACLE两臂显式使用目标GT数量，仅诊断，不是可部署结果。B0_GLOBAL保持同域总行数，但不含逐帧逐表达的数量信息。

| 目标 | 排序来源 | GT R@1 | 条件候选R@1 | Hit@1 | 可检索帧 | 全候选AUROC |
|---|---|---|---|---|---|---|
| Dance | source | 0.156803 | 0.157999 | 0.614788 | 42657 | 0.650204 |
| Dance | in_domain | 0.155387 | 0.157938 | 0.619945 | 42657 | 0.828769 |
| KITTI-V2 | source | 0.197195 | 0.243035 | 0.347478 | 195480 | 0.552911 |
| KITTI-V2 | in_domain | 0.424417 | 0.525893 | 0.708625 | 195480 | 0.901302 |

上表的source分别指C1(V2)和C2(Dance)；新增C5(LaMOT)的GT R@1=0.277423、AUROC=0.670094，完整字段见C5_RESULTS.json，未混用两个不同源模型。

同帧GT R@1：对官方GT非空帧，top1候选以IoU≥0.5命中的GT数/该帧GT数，按帧均值（无可匹配候选的帧记0）。另外保留旧离线脚本的条件候选召回：在有IoU≥0.5候选的帧，top1命中的正候选数/该帧正候选数再均值。正候选可有重复框，故两个分母不等，不能混称。tie=(-margin,numeric ID)；Hit@1另列，不是AP。GT轨迹结构用一对一最大基数匹配。新增GT分母诊断未改变任何预测、HOTA或预注册判定。

## 4. KITTI按表达类别

| 组合 | 类别 | 配置 | HOTA | DetA | AssA | 表达数 | 状态 |
|---|---|---|---|---|---|---|---|
| C2 | car | NATIVE | 13.190555 | 3.593490 | 48.448768 | 481 | COMPLETE |
| C2 | pedestrian | NATIVE | 13.616864 | 4.429253 | 41.976949 | 381 | COMPLETE |
| C2 | car | IN_DOMAIN | 52.702969 | 42.228716 | 65.815273 | 481 | COMPLETE |
| C2 | pedestrian | IN_DOMAIN | 38.811844 | 26.924673 | 56.037477 | 381 | COMPLETE |
| C2 | car | R_SxK_I | 30.773127 | 18.926806 | 50.063527 | 481 | COMPLETE |
| C2 | pedestrian | R_SxK_I | 28.408893 | 17.855415 | 45.278945 | 381 | COMPLETE |
| C5 | car | NATIVE | 21.142895 | 10.464572 | 42.763867 | 481 | COMPLETE |
| C5 | car | IN_DOMAIN | 52.702969 | 42.228716 | 65.815273 | 481 | COMPLETE |
| C5 | car | R_SxK_I | 33.782533 | 22.637578 | 50.455240 | 481 | COMPLETE |
| C5 | pedestrian | NATIVE | 22.948632 | 12.177309 | 43.322060 | 381 | COMPLETE |
| C5 | pedestrian | IN_DOMAIN | 38.811844 | 26.924673 | 56.037477 | 381 | COMPLETE |
| C5 | pedestrian | R_SxK_I | 33.716089 | 22.145489 | 51.432362 | 381 | COMPLETE |

评测分组使用官方表达canonical raw_sentence（缺失时原sentence），匹配car/auto与pedestrian/person/gender词；全部862条唯一分类，映射和原文件SHA见source_query_categories.csv。不能只在改写后的句子里搜索car：automobiles/autos等改写会被错误归成行人。映射只在V2参照预测封存后的离线进程生成，从未进入模型；不读referring label值来调预测。保留该类表达的全部候选（不事后按预测类别删框），官方HOTA.combine_sequences聚合。只有4个验证视频，不做bootstrap。

## 5. 结构统计（ORACLE，仅描述）

| 目标 | 轨迹0表达 | 轨迹1表达 | 轨迹≥2表达 | 空表达率 | Present GT帧率均值 | p50 | 状态 |
|---|---|---|---|---|---|---|---|
| Dance | 0.166185 | 0.832370 | 0.001445 | 0.905882 | 0.973752 | 1.000000 | COMPLETE_OFFLINE_ORACLE |
| KITTI-V2 | 0.318471 | 0.000000 | 0.681529 | 0.000000 | 0.464424 | 0.461279 | COMPLETE_OFFLINE_ORACLE |
| SportsMOT | — | — | — | — | — | — | P1_RUNNING_AWAIT_EVALUATION |

轨迹正例：一对一IoU≥0.5匹配观测达到该物理预测轨迹≥50%；不是预测ID=GT ID。逐视频表达数、共享文本见structure_stats.csv；Dance17句跨25视频共享。Sports未知项留空，不能以现成图像替代官方McByte轨迹统计。

## 6. 不确定性

| 比较 | ΔHOTA | 95%视频聚类CI |
|---|---|---|
| C1__R_SxK_I_minus_NATIVE | 23.553068 | [14.685326142991581, 33.6424827678502] |
| C1__IN_DOMAIN_minus_R_SxK_I | 0.367741 | [0.09069506709065518, 0.7193641061779391] |

每次2000抽样、seed0，整视频为聚类单位，复用官方TP加权HOTA充分统计且对COMBINED自检；不是对相关表达/帧独立重采样。KITTI仅4视频，点估计。Sports45视频未运行，不伪造CI。

## 7. 成本

有完整退出记录的GPU进程时间共0.443612小时；加上外部中断作业，已发生总成本落在[0.677329, 0.731916] GPU进程小时。外部SIGTERM未提供精确退出时刻，使用已完成视频计算耗时为下界、发现退出时刻为上界，不能伪造精确成本。若C2仍运行，其未完成时长暂不计。每次job/区间详见costs.json。最多2物理GPU，入场按进程少/显存足够共享，不终止别人。训练、微调、MLLM调用均0。

C1七套官方CSV全部在425份预测逐行一致和原SHA核验后复用，未重跑健康推理/评测。新KITTI评测记录在official_results.json；完整detailed CSV、原始日志、margin、预测保留在服务器，Git存summary/清单/压缩日志，detailed CSV不重复装入压缩包，其绝对路径及SHA可由official_results.json定位校验。复用旧推理不是一个“零GPU成本独立系统”，继承成本未重测、不能冒报为0。

## 8. GT审计、失败和恢复

模型/GT-free解码的GT打开尝试为0；官方评价和oracle在独立进程、预测SHA封存后执行，未向模型写回标签、阈值或参数。纯文本runtime从旧白名单元数据导出，不打开原表达标签。下载的JSON只复制/解压/哈希，未作为模型GT输入。

首次选错vllmedit环境缺yacs，两子进程6秒退出；切回原dfot（torch2.7.0+cu128/transformers5.4.0），配置回归通过，未修改共享环境。smoke后P1发布锁按要求阻止提前全量运行，远端盘点发布后恢复，0005完成单元未重跑。V2 smoke对旧分数最大重算差0.0010823保留记录，不覆盖旧分数；实际官方forward缓存误差≤2.1e-5且无符号翻转。

C2首个full作业发生外部SIGTERM（143），没有模型traceback，来源未确定。原进程确认已退出，保留0005/0011/0013完成回执和0019的72个完整帧/43240个已有分数，仅恢复未完成帧；新增逐帧flush和独立进程会话，未改checkpoint/精度/阈值/采样，未停止他人进程。成本记录保留中断区间，不把失败计算免费抹去。

所有新代码和run均独立；vendor已有mydataloader旧test补丁原样保留，本轮未改vendor代码；仅新增授权数据软链接。历史结果、源checkpoint和其他项目进程未覆盖。完整范围继续保留，当前仅因缺官方二进制资产停止C3–C6。

恢复路径：results/runs/multidomain_generality_20261101；大文件SHA见margin_paths.json、all_prediction_files.csv.gz、official_results.json和downloads.json。手动资产到位后先strict/45视频/466序列/smoke门槛，再按同一PREREG继续；不生成替代tracker，不开展方法或训练。

官方来源：[FlexHook](https://github.com/buptLwz/FlexHook)、[DATASET.md](https://github.com/buptLwz/FlexHook/blob/main/datasets/DATASET.md)、[LaMOT](https://github.com/Nathan-Li123/LaMOT)、[论文v5](https://arxiv.org/html/2503.07516v5)。

针对性回归3项通过，compileall通过；14套主预测、11595份完整/历史/源参照文件索引及逐帧数量等价自检通过。详见validation.json。当前任务GPU进程已全部结束，不再启动新实验。
