# MAP v1 主方法与关键对照：完整视频续跑

状态：预注册/费用清单已公开，10项队列已真实启动，尚未产生本阶段完整 HOTA。不能将首视频数值稳定性解释为跟踪精度改善。

预注册公开提交：`2c4e0a319e3f5d39aeecf72dca0da0cbad495c58`。定向回归34项通过，compileall通过。
首次准入：C1 FULL在物理GPU5（PID3578621），C3 FULL在物理GPU7（PID3578622），
这只是启动时快照，不声称进程以后永不退出。共享准入、不停止他人进程、最多两卡。
阻塞协调器和可恢复路径见 execution_started.json；首视频已复用，两个方向都已进入后续视频。

真实 MAP 两个首视频计时外推：108项保守估算292.886 GPU小时；本阶段10项11.030 GPU小时。
其中C1 FULL 4.084、C3 FULL 6.491，其他8项合计0.454。不是已经消耗的时间；共享卡、全部轮均训练、
不计缓存且不扣已完成首视频的保守外推，KITTI尚无单位测速。实际开销与估算差异将如实报告。

本阶段为 C1/C3、ρ=0.3 的 FULL、MINUS_SLOW、UNIFORM_K、两种预算匹配 YesTrack，共10项。
完整108项目标保留，其余项目不伪报完成。协议见 PREREG.md，真实估算与运行状态随后追加。

FULL 复用首视频检查点和随机状态，恢复目录为：

- `/data2/user/FlexHook-TTA/results/runs/presence_active_20261102/redesign_v5/active_map_v1/C1/FULL/rho_0.3`
- `/data2/user/FlexHook-TTA/results/runs/presence_active_20261102/redesign_v5/active_map_v1/C3/FULL/rho_0.3`

上述目录实际位于已授权的 data1。旧 pilot 报告、原失稳结果及 vendor 教师代码保持不动。
