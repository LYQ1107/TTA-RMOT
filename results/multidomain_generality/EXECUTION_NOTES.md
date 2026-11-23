# 执行与恢复记录

项目仅为 `/data2/user/FlexHook-TTA` / `LYQ1107/TTA-RMOT`，不是 LocateMOT。

## 已发布门槛

- 原始预注册：`98e7f84efbe43493761063e7832de7bfe9d4e18d`，原93行保持不变。
- 用户补充授权先发布：`0f9c61c66f26a72ace8f044864b8b0f71a8ecc0a`，之后才下载和推理。
- 恢复P0、strict实机smoke和成本先发布：`e7bf73cf81a1ed6afc775815581e0564a77cbea3`，之后才恢复P1。
- C1七套固定诊断及逐行复用：`1188a377ce4e1168a116edda680db7044523d38a`。

## 环境与运行路径

使用已有 `/home/szgx/anaconda3/envs/dfot/bin/python`。

```bash
cd /data2/user/FlexHook-TTA
export PYTHONPATH=/data2/user/FlexHook-TTA/compat:/data2/user/FlexHook-TTA:/data2/user/TTA-RMOT-assets/flexhook/pydeps
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
```

运行时目录：`results/runs/multidomain_generality_20261101`。大分数数组、逐表达预测不进Git，按公开清单的绝对路径与SHA恢复。

GPU任务由 `scripts/multidomain_kitti_scores.py` 管理，最多两卡、共享须显存准入，进程少优先。不要在已有工作进程存活时重新运行launcher。

完成后的CPU阶段（三个独立进程，GT只在后两者读）：

```bash
/home/szgx/anaconda3/envs/dfot/bin/python scripts/multidomain_diagnostics.py predict --target KITTI-V2
/home/szgx/anaconda3/envs/dfot/bin/python scripts/multidomain_diagnostics.py oracle --target KITTI-V2
/home/szgx/anaconda3/envs/dfot/bin/python scripts/multidomain_diagnostics.py evaluate --target KITTI-V2
/home/szgx/anaconda3/envs/dfot/bin/python scripts/report_multidomain_generality.py
```

源域完整参照独立先封存，再评测，后续IN_DOMAIN逐行复用该评测，不重复跑：

```bash
/home/szgx/anaconda3/envs/dfot/bin/python scripts/multidomain_diagnostics.py reference_predict --target KITTI-V2
/home/szgx/anaconda3/envs/dfot/bin/python scripts/multidomain_diagnostics.py reference_evaluate --target KITTI-V2
/home/szgx/anaconda3/envs/dfot/bin/python scripts/multidomain_source_categories.py
```

类别映射在该参照封存后，从原表达的canonical raw_sentence（不存在则sentence）唯一识别481 car /381 pedestrian；不在模型中使用，不从改写句子是否含car猜测类别。

## 实际失败和最小修复

1. 最初使用vllmedit，缺yacs。两smoke都在模型构建前退出。回原dfot，配置/依赖回归通过，不安装到共享环境。
2. 第一版百度verify的surl错误带了分享URL前导1，errno2。按官方页面的surl修正后提取码验证errno0；不能将参数错误称为账号错误。
3. 百度页面公开uk=0是访问者，实际share_uk=1714442791。按share_uk可以公开列出目标权重和45个McByte文件。
4. 未签名sharedownload errno2；正确调用tplconfig得到公开签名后errno0，但list为不透明字符串。最初假定为数组导致AttributeError，尝试JSON解析又报JSONDecodeError；修正为识别真实响应类型并停止，不执行网页JS或下载内容、不使用外部解析代理。query encrypt=0仍相同。支持的BaiduPCS客户端文档要求登录，无账号凭据，因此列精确手动文件，不伪报下载成功。
5. smoke成功后P1发布锁拒绝提前运行；P0远端确认后解除。健康0005回执和分数保留。
6. C2第一轮full监督进程收到SIGTERM，外层退出143，子进程确认已退出，模型日志没有traceback。保留0005/0011/0013，0019有72个完整帧、43240个已写分数。恢复只跳过完整帧，未完成帧重算；新增逐帧flush、独立会话，不改精度/超参/候选/权重。
7. 第二轮监督shell也收到143，但隔离后的worker 2981499仍在GPU2运行，PPID=1；没有重启它。用一个阻塞 `multidomain_wait_owned_worker.py` 等其封存，不做agent级ps/tail轮询。后续若等待shell又被外部结束，先检查既有封存/原worker，禁止启动重复推理。
8. 首次full中断的精确退出时刻不可用。costs.json明确给上下界：已完成视频计算时长下界、发现退出时刻上界；不能把失败GPU计算抹掉，也不能制造精确总时长。

所有原始GPU尝试在 `kitti/launchers/<时间戳>/`；未完成与失败记录保留。官方TrackEval原始日志及summary/detailed CSV在 `diagnostics/<目标>/<配置>/`，压缩精简副本随Git发布。

## 资产阻塞不是缩小预注册

完整C1–C6继续保留。C3–C6尚无合法官方LaMOT-best/McByte/FlexHook Sports封装，不运行，也不拿空值当0。官方GitHub已下载508个原始Sports JSON，但不是466个FlexHook表达文件名的现成封装。

下载目的地：`/data2/user/TTA-RMOT-assets/flexhook/lamot_20261101`。精确 `/FlexHook/SOTA_ckpts/LaMOT-best.pth`、45个 `/FlexHook/tracker_outputs/McByte-SportsMOT/*.txt` 和数据JSON路径见downloads.json。原始Sports图像只建链接，原图与其它项目代码/模型/标签均未改。

收到官方文件后还必须核验strict、45视频覆盖、466表达一一映射、官方mix loader真实视频smoke及≤6 GPU小时估算，再恢复相关组合。当前不生成替代tracker、不训练、不调用MLLM、不开始新方法。

## 本轮结束状态

C1与C2各7固定配置已完成，另外保留KITTI两模型历史覆盖和独立完整V2参照。C2恢复worker自行封存退出，当前任务没有GPU进程存活。14套主预测与11595个完整/历史/参照文件索引、逐帧同量自检通过，3项回归通过，compileall通过。

C1 ρ=0.984626733（数量主导）；C2 ρ=0.535039998（数量与排序共同作用）。C2原生13.502、源排序同域量29.152、同域完整42.752；不是只报正向Dance结果。完整三域六组合未完成，C3–C6在精确资产清单处停止，最终汇总标签不计算。
