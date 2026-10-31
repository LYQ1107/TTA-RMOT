# 执行问题与修复记录

1. P0：基础shell没有`python`；改为系统`python3`只读盘点，模型与测试使用既有`/home/szgx/anaconda3/envs/dfot/bin/python`，教师沿用独立teacher-p1环境。未触发模型、GT或实验数值变化。
2. 输入隔离的故意违规测试：Val MOT文件名是`dancetrackXXXX.txt`，初版audit仅检查路径目录组件，kernel已拒绝5/5，但audit只记4/5。补充stem识别，Python审计和kernel-only两次独立子进程都拒绝5/5。尝试记录同时即时写出，失败退出仍可追踪。发生在准备测试，不是真实训练读到了Val。
3. P0发现Train/Val有8个表达文件名重合；按用户给定官方划分继续、不修改query集合，报告明确不是全语言未见测试。视频完全不重叠。
4. 历史标签计数纠正：Naive8500为656个p≥.95，PSR自身8500为657。只读取已封存的教师输出比较，不读GT；新训练只读Train31200，不把旧Val缓存作为输入。
5. 原PSR封存器产生兼容weak文件，但本轮所有臂expanded_kind=none、expanded=0，weak不作为任何训练项；没有增加教师视图或新方法。
6. 两个学生200步smoke都正常完成；重复发布元数据时发现copy2会继承seal的0444权限，下一次重复复制同一公开seal报PermissionError。仅修复打包器：已有相同SHA的不可变元数据直接复用，不改原seal、不重跑任何模型/训练；添加真实只读文件复现回归。

26项不同相关CPU测试通过（`test_train2val.py` 8项、`test_diag_fit.py` 18项；合并运行15.38秒），compileall及diff检查通过。
Naive8500采样重放由独立审核进程执行，与训练白名单隔离。
真实教师4样本smoke和全40Train的31200判断、split=train加载均已完成，GT尝试0，原后端/裁剪代码不变。
学生smoke/完整训练/预测/GT评价的真实状态以后续独立receipt和最终报告为准，不把上述准备检查当科研结果。
