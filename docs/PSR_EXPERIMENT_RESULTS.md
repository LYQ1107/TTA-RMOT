# PSR 实验状态

2026-10-29，用户允许的25Val同集适应，非Train→Val/非盲测。

已完成匹配预算教师缓存及真实学生smoke；Naive/PSR正在独立目录按固定8973步训练。
两组均保持24576参数、seed17、lr5e-5和源val CE选checkpoint，不读取目标GT。

完整RMOT新指标：**尚未产生**。Frozen8.2180和Source-only7.4352为已核验可复用参考，
旧Naive8.0327使用另一采样协议，仅保留历史，不能填入本轮新Naive的空位。

后续：完成E2/E3→封存完整425预测→独立官方TrackEval→完成必需Uniform/Confidence/Discovery/Geometry/普通Temporal对照。
所有GT派生正确率、覆盖、传播错误和HOTA增益均以实际封存后计算为准，不使用loss或高分标签数代替。
