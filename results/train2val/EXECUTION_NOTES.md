# 执行问题与修复记录

1. P0：基础shell没有`python`；改为系统`python3`只读盘点，模型与测试使用既有`/home/szgx/anaconda3/envs/dfot/bin/python`，教师沿用独立teacher-p1环境。未触发模型、GT或实验数值变化。
2. 输入隔离的故意违规测试：Val MOT文件名是`dancetrackXXXX.txt`，初版audit仅检查路径目录组件，kernel已拒绝5/5，但audit只记4/5。补充stem识别，Python审计和kernel-only两次独立子进程都拒绝5/5。尝试记录同时即时写出，失败退出仍可追踪。发生在准备测试，不是真实训练读到了Val。
3. P0发现Train/Val有8个表达文件名重合；按用户给定官方划分继续、不修改query集合，报告明确不是全语言未见测试。视频完全不重叠。
4. 历史标签计数纠正：Naive8500为656个p≥.95，PSR自身8500为657。只读取已封存的教师输出比较，不读GT；新训练只读Train31200，不把旧Val缓存作为输入。
5. 原PSR封存器产生兼容weak文件，但本轮所有臂expanded_kind=none、expanded=0，weak不作为任何训练项；没有增加教师视图或新方法。
6. 两个学生200步smoke都正常完成；重复发布元数据时发现copy2会继承seal的0444权限，下一次重复复制同一公开seal报PermissionError。仅修复打包器：已有相同SHA的不可变元数据直接复用，不改原seal、不重跑任何模型/训练；添加真实只读文件复现回归。

完整训练完成后再次合并运行，27项不同相关CPU测试通过（`test_train2val.py` 9项、`test_diag_fit.py` 18项；15.57秒），compileall及diff检查通过。此前26项/15.38秒为添加只读seal回归前的阶段记录，不累加重复测试。
Naive8500采样重放由独立审核进程执行，与训练白名单隔离。
真实教师4样本smoke和全40Train的31200判断、split=train加载均已完成，GT尝试0，原后端/裁剪代码不变。
学生smoke/完整训练/预测/GT评价的真实状态以后续独立receipt和最终报告为准，不把上述准备检查当科研结果。

## 最终执行结果

四个8973步正式训练、四个完整425表达推理、16套新官方TrackEval均首次正常完成，没有GPU失败、缺失episode或6.2数值回退；健康episode没有重跑，6套历史官方结果由逐行等价证明后复用。此前准备问题的修复与失败记录保留，不覆盖旧P1/v1/v2/D-FIT结果。

最终CPU回归27项通过（13.75秒），compileall/diff检查通过；原teacher后端/裁剪、原两个数值YAML与vendor相对619390d不变。训练数值代码相对实际启动snapshot也未改，详见VALIDATION.md。

首次生成完整报告后，澄清公开Table C的来源说明：原先统一描述过于笼统，现分开标注历史同Val训练、同域官方监督、本轮Train40→Val25、Frozen和GT-free校准；只重新生成CPU汇总与文档，预测、官方CSV、bootstrap数值不改变。INTERPRETATION.md给出所有负面结果、成本和限制；不是补训或用GT重新选结果。

最终D1主验收失败：三seed HOTA8.4978/8.8393/7.7520，G8.363011、SD0.556033，净增益保留0.674973%。D3点估计12.2860，但与Frozen的配对视频CI包含0。完整记录保留后停止，不启动新方法或补训。
