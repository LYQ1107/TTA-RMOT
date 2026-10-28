# MLLM Distillation：真实失败与修复记录

2026-10-28；当前25Val同集开发诊断，不是Train→Val。

1. 首次无step接口检查在GPU1的CUDA驱动初始化阶段失败，`cudaGetDeviceCount Error304`。
   此时模型/图像尚未加载，也没有优化器更新。原因是dfot驱动初始化在Landlock白名单之后。
   最小修复：先初始化空CUDA上下文，再安装同样输入白名单，之后才加载模型与数据。
   保留`interface_stdout.txt`、原入场审计与失败目录；新输出`interface_v2`验证，不覆盖失败。
   这不是梯度/损失问题，不改变BF16教师、FP32学生接口、LoRA参数、任何输入或GT协议。

2. 第二次检查已通过CUDA初始化，但Transformers的延迟导入需要本机用户site-packages里的psutil，
   原白名单只有conda前缀，因此在读取该模块时明确失败。新增`site.getusersitepackages()`只读运行库路径，
   不放宽目标数据访问，不安装或修改公共环境。保留`interface_v2`，使用`interface_v3`真实回归。

3. 第三次已经严格加载V2全部参数，但新入口漏导入旧训练器使用的`compat.sitecustomize`，
   本机Transformers5.4把person切为6个字符token。独立CPU比对证明原shim恢复正确ID5970。
   新入口显式复用原shim（不重写tokenizer/词表），记录shim与官方模型源码hash，保留v3失败并用v4回归。
   该检查在任何loss/step前失败，没有用错误tokenizer产生训练或指标。

4. v4的模型与tokenizer均通过，但新增provenance需要读取shim源码hash；白名单没有该代码目录，
   因而在写训练来源记录前失败。仅加入本仓库`compat`只读代码路径，不改变数据权限，v5真实回归。

5. v5核对原Frozen分数时遇到已封存merged文件指向原worker的符号链接；Landlock按真实目标路径授权。
   改为白名单加入这一个分数文件的resolved路径，同时显式允许合法源KITTI共享图像根的resolved只读路径。
   不授权其它项目代码/模型/进程，也不扩大目标GT权限；v5失败与最初五次零更新尝试完整保留。

6. 首次6步smoke确有有限KD梯度和参数更新，但回读receipt发现所有source `provided_expressions=0`，
   总周期错误地成为2125而非8973，源验证空计划导致NaN。Python glob在Landlock不允许的源表达符号链接上返回空列表。
   该smoke **撤销科研有效性，不用于正式初始化或任何指标**，原结果保留并另写invalidation。
   仅补充合法源`expression`的resolved只读根，并在任何step前强制核验8896源表达、8973源窗口、64源验证窗口，
   源val CE非有限立即失败。使用新`smoke_v2`独立输出，既不跳过样本，也不通过修改步数掩盖缺失。

7. 修正后的smoke_v2完整源计划/6步均通过，随后真实checkpoint恢复也成功，0新增更新；
   但新CLI finally写运行耗时时再次解析相对路径，而官方model loader会chdir到vendor，导致仅末尾日志写入失败。
   在入口一次性规范为绝对输出路径；保留原resume日志，再做0新增更新的恢复回归，不重跑已完成6步。

正式结果尚未产生；后续归因由完整学生预测与独立评测填写，不能把工程修复当作科研提升。
