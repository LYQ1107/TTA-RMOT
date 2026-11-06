# v8 阶段 0：按预注册停止（EM 未收敛）

## 结论与执行边界

**状态：STOP_EM_NONCONVERGENCE。** 没有启动阶段 1、阶段 2/3 或阶段 4，没有新的真实 RMOT 指标；v7 的全部代码、结果、checkpoint 和教师缓存均未改动。

预注册及 A1 修订在任何 v8 结果产生前公开：[4d44d479](https://github.com/LYQ1107/TTA-RMOT/commit/4d44d479e5d462b16eb5c4b164ec73284d20f32e)。A1 仅把 rho=1.0 用例改为 `rho_hat >= 0.95 - 1e-9`；这是 [0.05,0.95] 截断设计的结果，不是估计误差。rho=0.1、每档 pi、旧估计器回归及主/慢分支判据都未改变。

## 1. 实际测试结果

正式完整测试命令：

```bash
CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=/data2/user/FlexHook-TTA/compat:/data2/user/FlexHook-TTA:/data2/user/TTA-RMOT-assets/flexhook/pydeps \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
/home/szgx/anaconda3/envs/dfot/bin/python -m pytest -q --maxfail=1 tests/test_active_v8.py
```

结果：**9 passed、1 failed，7.59 秒**；rho=1.0 参数化用例因首个失败而未执行，不能算通过。此前只运行基础检查的结果是 9 passed、2 deselected，5.33 秒，与本次 9 项重叠，不累加。

已通过的是 CPU 单元契约：

- 无回答句子不进入 EM；标量单步公式与向量实现一致。
- 覆盖不足不因低先验判空，覆盖完成后判空保持。
- 预算足够时首轮 GLANCE 覆盖全部句子；low_tentative 优先补覆盖。
- 慢分支单类训练池跳过；采样批次为 16 正/16 负，区分直接与派生负例。
- 两个伪造视频、五个快照位置的回放存在性数组逐字节一致。
- GT guard 的正常分支打开尝试为 0，故意打开 GT 的测试被拒绝并单独计为 1。

这些不是实际 GPU 优化或真实回放验证。没有执行学生/教师模型 forward；合成实验没有读取真实数据或 GT。

## 2. 首个失败：不是 rho=1.0 边界问题

固定 fixture 已在实验前写入 manifest：seed 0、200 视频、每视频 17 句、30 条轨迹、每轨迹 8 个可问位置、4 轮、预算 rho=.3；真值 rho=.1、pi=(.8,.3,.05)、教师 s/c=.8。用实际 `questions_v8` 自适应选题；逐视频拟合，不能按结果更换 fixture、seed 或问题预算。

在**第 5 个视频结束**时，固定上限 200 次 EM 迭代耗尽：

| 字段 | 实际值 |
| --- | --- |
| completed_videos | 5 / 200 |
| iterations | 200 / 200 |
| converged | false |
| last_delta | 8.198819538840496e-7 |
| registered tolerance | 1e-7 |
| rho_hat | 0.23273205643681252 |
| pi_0 | 0.43335323219726707 |
| pi_1 | 0.15119714931447706 |
| pi_2 | 0.5567154949779427 |
| 有回答句子记录数 | 85 |
| 各档已回答轨迹对数 | 108 / 477 / 48 |
| 逻辑合成回答数 | 765 |
| GLANCE / PRESENCE / RECHECK / IDENTITY / COVER | 85 / 104 / 87 / 486 / 3 |

`last_delta` 是本轮候选参数相对上一迭代的最大绝对变化，约为容差的 8.20 倍。因此不能把达到 200 次上限写成已收敛，也不能仅把 `converged` 改为 true。

这些是仅五个视频后的中间估计，**不作为完成 200 视频后的恢复精度结论**。rho=.1 的最终精度、每档 pi 的最终精度、旧估计器 pi_0 偏差及 rho=1.0 边界恢复均未完成验收。不能由此宣称 EM 永远无法恢复真值，或宣称 v8 已改善/恶化 HOTA。

## 3. 实现核对与未验证部分

`PresenceEM` 按注册公式使用全部已回答句子、三档 pi、Beta(2,2) MAP、每轨迹 LLR clip ±2、200 次上限及 1e-7 容差；没有使用按“已确认存在”筛选的记录，也没有把判定结果作为 EM 观测。单步标量/向量对照通过，当前没有证据表明这次非收敛只是边界断言或公式转写错误。

该截断 E 步是注册的稳健近似，MAR 与条件独立是建模假设；本轮没有验证一般统计一致性，不把预注册的理论动机写成已证明的结果。

新增文件只涉及 v8：

- `flex_uda/active/v8.py`：PresenceEM、覆盖约束的 BeliefStateV8、快照和选题。
- `flex_uda/active/v8_slow.py`：BCE 训练池、平衡门槛及 16/16 采样。
- `scripts/active_v8_run.py`：v8 运行器草稿与存在性冻结回放路径，**未完成真实 GPU 集成验收，不能直接声称可正式运行**。
- `scripts/active_v8_stage0.py`、`tests/test_active_v8.py`：固定合成 fixture、离线 v7 重放入口及 CPU 契约测试。

由于触发停止条件，配套调度/关闭脚本与阶段 0b–0e 的结果尚未交付，不以空表或历史数字冒充。

## 4. 阶段状态

| 阶段 | 状态 |
| --- | --- |
| 预注册、manifest.amendments A1 | 已先提交并推送 |
| 0a CPU 测试 | 9 passed / 1 failed；触发 EM 非收敛停止 |
| 0b v7 逐值复现与六方向 EM 离线流 | 未执行 |
| 0c GT 估计器诊断 | 未执行 |
| 0d 反事实存在性 | 未执行 |
| 0e KITTI 专项 | 未执行 |
| 1 C1/C2 真实冒烟 | 未启动 |
| 2 六方向 MINUS | 未启动 |
| 3 FULL / REPLAY | 未启动 |
| 4 曲线 / 消融 | 未启动，主判据未计算 |

## 5. 成本、保护与恢复条件

本轮新增真实教师调用 0、GPU 调用 0、学生优化步数 0；没有打开目标真实 GT。基础 guard 单测中故意触发的 1 次拒绝不是生成进程的真实 GT 读取。合成教师回答不计作 Qwen 调用。

未改迭代上限、收敛容差、seed、伪标签阈值或验收条件。没有执行 v5 剩余 80 项、v6 扩展或任何新方法。当前磁盘审计 data2 约 20 GiB、data1 不足 1 GiB；尚未分配 GPU 特征缓存。

失败原始机器收据：`stage0_synthetic_rho_0.1.json`；测试记录：`stage0_pytest.txt`；完整状态、路径和 SHA：`stage0_status.json` / `stage0_artifacts.json`。

按预注册“EM 不收敛即停止”已停止。继续需要用户明确决定如何修订收敛处理；本轮不会自行增加迭代、忽略早期视频失败或直接跳到 GPU。
