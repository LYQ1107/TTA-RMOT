# Positive-Free TTA 服务器集成验证（2026-10-25）

源码基于 `research/v2dance-positivefree-prep-20261025` 的固定提交
`896d67fe92d44e18c33980a7b26c2d3da1f8b7f7`。本地验证分支为
`codex/pf-server-smoke-20261025`，原研究分支保持在
`0a4377c1363c32605ce8d6305cb49c42300ed2d0`。

本轮完成代码集成、资产预检和单卡 smoke，未启动源训练、完整视频推理、
完整 25 视频评测或四卡正式运行。两个完整运行授权变量均未打开。

## 代码测试和必要修复

指定四个测试文件最终 **26 passed / 0 failed**；`compileall`、shell 语法检查
和源码/文档的 `git diff --check` 均通过。原始第三方 stdout 日志保留其行尾空格，
未为格式检查改写实验记录。新增的四项回归覆盖相对路径、分片元数据合并、
当前/未来身份特征不可作为历史证据、两次 episode 重置后的源模型前向一致性。
合并回归使用合成的空预测文件，没有真实数据推理或 TrackEval。

服务器集成发现并修复的问题：

1. 首轮是 21 passed / 1 failed。CPU smoke fixture 只有一个视频，但复用的
   `dance_videos()` 正式集检查要求 25 个。测试显式传入自己的 smoke 视频，
   保留正式数据加载器的覆盖检查。
2. FlexHook 配置函数切换进程 cwd，导致调用者相对输出/vendor 路径失效。
   runner 在加载模型前解析所有输入输出的绝对路径。
3. 原 helper 使用 `strict=False`。Positive-Free runner 改用显式
   `load_state_dict(..., strict=True)`；旧实验 helper 的默认行为保留。
4. 新 runner 原本只输出聚合统计，缺少现有分片合并器需要的 `episodes`。
   补齐逐 episode 统计、实际视频列表、序列数和超参数，并通过合成合并回归。
5. 原预检只检查部分文件数量；扩展为 tracker 下载收据核验、全部引用图像、
   Frozen 各表达的 score 覆盖、源训练图像/表达/标注视频和 tokenizer 回归。
   预检开发时一次误写 token ID 被独立 tokenizer 回归发现；现按发布 vocab
   读取 `person` 对应 ID（5970）验证。

运行环境为 `/home/szgx/anaconda3/envs/dfot/bin/python`（Python 3.10）。
原环境缺少 pytest，本轮以用户安装方式补齐 pytest 9.1.1。运行时使用仓库
`compat`、仓库根目录和现有 `flexhook/pydeps` 的 PYTHONPATH。

## 资产预检

`results/positive_free/server_smoke_20261025/preflight.json` 的
`server_assets_pass=true`。检查过程没有打开 Dance GT。

| 资产 | 验证结果 |
|---|---|
| 官方 V2 checkpoint | 995,768,450 bytes，SHA256 与固定值一致 |
| 官方 Dance tracker | 25 文件均与原始下载收据 SHA256 一致 |
| tracker 覆盖 | 211,085 行，25,508 视频帧，引用图像全部存在 |
| 目标表达 | 25 视频 × 17 表达 = 425 episodes |
| Frozen scores | 393,731,645 bytes，各视频的 17 条表达均有 score |
| 源训练清单 | 17 个视频、5,172 个引用图像，缺失 0 |
| 源训练目录 | 5,939 张图像、8,896 个表达 JSON，标注视频覆盖已核对 |
| tokenizer | `person` → `[5970]`，独立 ByteLevel-BPE 回归通过 |

checkpoint SHA256：
`71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613`。
Frozen score SHA256：
`fbf498a24b73d6b2db6d9c757d313f660c0d5b09a7dbfec5c2635b8ca517ee69`。

17 视频清单包含 `0017`；但当前 vendor 的 `data/utils.py` 中
`VIDEOS['kitti-2']['train']` 仅列了 16 视频，漏掉 `0017`。数据是齐全的，
未来若另行授权“全部 17 视频源训练”，需先修正并单独验证训练配置。
本轮使用发布 checkpoint，未修改 vendor 训练划分、重训或替换权重。

## 单卡 smoke

启动前两次检查 GPU5：无计算进程、4 MiB / 81,920 MiB、利用率 0%。
所有 GPU smoke 均依次使用物理 GPU5，UUID 为
`GPU-c6523d98-cb2d-95ed-4cb2-a84936acceb6`，未与其他任务共享该卡。

每次运行仅 `dancetrack0004` 的第一条表达
`dacning-person-with-white-Tshirt-and-purple-pants`、前 12 帧、36 个候选。

| 方法/探针 | 结果 | 优化步数 | 非零有限梯度步数 | 回滚次数 |
|---|---|---:|---:|---:|
| ZERO_UPDATE | PASS | 0 | 0 | 0 |
| PF_RANK | PASS | 3 | 3 | 0 |
| PF_IDENTITY | PASS | 3 | 3 | 0 |
| PF_HYBRID | PASS | 3 | 3 | 0 |
| RELATIVE_ONLY（CPU score replay） | PASS | 0 | — | 0 |
| PF_RANK_GUARD | PASS | 3 | 3 | 0 |
| PF_RANK 回滚探针 | PASS | 0 | 3 | 3 |
| PF_RANK 无证据探针 | PASS | 0 | 0 | 0 |

主方法使用默认 `lr=5e-5`、rank 4、每 4 帧更新、seed 0。回滚探针单独将
`max_adapter_delta_norm` 设置为 `1e-12`；无证据探针单独将
`min_separation` 设置为 `100000`，用于核对更新门控，均不作为方法收益结果。

ZERO_UPDATE 的 36 个候选符号/选择与 Frozen 完全一致，margin MAE 为
`6.33134e-5`，最大误差 `1.64032e-4`；本轮没有检验完整集 HOTA 的复现。
PCD 输出通道实测为 768；4 个 Q/V LoRA 投影均为 768→768、rank 4，
共 24,576 个可训练参数。身份方法读取到的特征也是 768 维。

更新只发生在帧 4/8/12，排序证据来自更早的帧，身份记忆来自前一帧。
所有原模型冻结参数均无梯度；梯度有限且非零；接受的 adapter L2 更新约
0.0114–0.0136。回滚恢复参数并清空被拒绝的 Adam 状态。episode 开始和
结束的 adapter/身份记忆/排序记忆重置检查均通过；两次独立 episode 的
源前向恢复另有 CPU 回归验证。

文件访问审计记录 Dance GT 打开尝试为 0；图像历史上限不超过预测帧。
当前片段源模型全部输出负 margin，六种方法均输出空目标集，因此直接验证了
没有强制正例。该真实 GPU 片段的非空 MOT 输出行检查数量为 0；非空输出
原始行一致性由 CPU smoke fixture 的两条输出验证，不能把空输出检查夸大为
已覆盖非空真实预测。完整正式集之前仍需完整 ZERO_UPDATE 对照。

峰值 PyTorch allocated 显存约 1.58 GiB。实际 tracker 全集每帧最大候选数为
22，不超过当前 batch size 24；未触发多 batch 帧跳过优化的分支。

原始运行记录保留在 `gpu_smoke_v1/`（补齐合并元数据前的首轮）、
`gpu_smoke_v2/`（最终核心 runner）和 `ablation_smoke_v1/`。
每轮都是新的独立 smoke 目录，均未扩大单次 episode/帧数限制。

## 四卡预案与资源状态

`run_pf_4gpu.sh` 默认 dry-run 已验证，只打印分片，不创建 GPU 进程。
全量开关缺失时拒绝启动的测试也通过。

| 槽位 | 预案示例 GPU | 视频数 | episode 数 |
|---|---:|---:|---:|
| 0 | 5 | 7 | 119 |
| 1 | 4 | 6 | 102 |
| 2 | 6 | 6 | 102 |
| 3 | 1 | 6 | 102 |

精确视频清单见 `validation_summary.json` 的 `planned_shards`。分片按整个视频
分配，视频内每条表达完整运行，在线记忆不跨 GPU 切断。每种方法合计
25 视频 / 425 episodes，不能用分片或 smoke 指标代替全量结果。

收尾时 GPU5 已无进程、显存回到 4 MiB；GPU4/6 虽占用较低，仍分别有
1 个计算进程，GPU1 有 2 个。当前只有 1 张真正空闲卡，不能宣称四卡资源
已就绪。表中物理卡号只是 dry-run 示例，未来必须重新选取 4 张可用卡。
本轮未停止或改变其他任务的进程。

未来经明确授权后的顺序：
`ZERO_UPDATE → RELATIVE_ONLY → PF_RANK → PF_RANK_GUARD → PF_IDENTITY → PF_HYBRID`。
先让完整 ZERO_UPDATE 复现已有 Frozen HOTA 8.206 及主要指标，再比较其余
方法；一次只运行一种方法，每种方法使用新的输出目录。

## 运行边界和待解决问题

六种方法的服务器集成条件已通过，效果仍未知。当前 smoke 仅覆盖一个
三候选、零目标输出片段；不能据此推断 HOTA 提升、复杂 ID 切换行为或长序列
稳定性。排序目标仍可能保持错误的源排序，身份对比也可能仅改善身份稳定性。
Dance 集此前已用于 GT 诊断，应保留 development 集解释。

本轮停止于代码、预检和 smoke 结果交付。完整源训练、目标 TTA 和正式评测
均未启动，等待用户下一条明确指令。
