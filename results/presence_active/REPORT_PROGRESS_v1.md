# Presence–Active：进行中的独立方法实验

**源标签修复中：**全源核对发现Sports Train相同文本的不同表达文件被合并标签（27表达、564观测），旧LaMOT PresenceHead结果仅作历史，不作为主结果。KITTI/Dance全量源标签通过。按表达ID独立修复、原配方重训受影响头，旧数据和结果保留，见`SOURCE_LABEL_FIX.md`。B/C没有启动。

本报告当前为阶段A进度，不是完整方法的科研验收结论。默认三头训练及九套完整候选的A评测已完成；阶段B/C尚未调用教师或启动LoRA适应。不得把下面的smoke或测试通过当作HOTA提升。

三个默认源头及同域官方评测已经完成：KITTI 40.599 / 原生42.752；Dance 21.605 / 32.139；SportsMOT 64.064 / 69.924。全部未过−1门槛，B/C保持锁定。其余跨域A缓存/评测继续，已授权的单次源学习率缓解对照定义见`SOURCE_LR_TRIAL.md`；默认结果不覆盖。

## 项目、范围与数据隔离

- 分支 `method/presence-active-20261102`，起点 `9b04b070eddc9e80c0c9db1b582a02780689aa16`，根目录 `/data2/user/FlexHook-TTA`。
- 不操作LocateMOT代码或进程；原始Sports图像/标注来自盘上既有公共数据存放位置，不是LocateMOT输出。
- 六个跨域组合和三个同域模型全部保留；不修改历史结果、vendor代码或冻结教师的原后端/裁剪器。
- 源标签只用于各自PresenceHead的源训练及用户允许的源官方测试早停；目标特征、主动提问和适应进程启用打开审计，评价GT不进入它们。
- 按完整轨迹聚合/全视频存在性构造先验是视频转导式，不是严格逐帧因果；四帧视觉历史和CLIP输入仍不读取相应观测的未来图像。LoRA按视频ID顺序跨视频延续。

## 阶段A：盘点及规模（全量导出前报告）

|源训练集|视频|表达文件|帧–句子单元|候选–句子特征行|PCD FP16字节|
|---|---:|---:|---:|---:|---:|
|KITTI-V2|16|8543|1099242|4534302|6964687872|
|Dance|40|1560|164112|1371786|2107063296|
|SportsMOT|45|496|30401|335246|514937856|

KITTI按未修改vendor `VIDEOS['kitti-2']['train']` 的16视频；旧源监督清单有17视频，额外含0017。本轮不直接沿用该历史自定义清单。所有源GT框保留，包括没有本句指代标签的合法候选；没有候选的采样图像也保留为空集合单元。KITTI每3帧、另两域每10帧抽样。

LaMOT此前下载的官方GitHub压缩包已含496条Sports Train标注，45个Train视频图像及原框也齐全，无需新下载。MOT17/VisDrone Train未在既有LaMOT布局准备，本轮采用主要求的Sports Train部分，不引入TAO。

九套目标完整特征26946084行，PCD本身41389185024字节；加源缓存、margin、索引和单元记录，初估51544164190字节。`/data2`起始仅余约5.7GB；用户授权新缓存放 `/data1/user/TTA-RMOT-assets/presence_active_20261102/`，指定 `results/runs/presence_active_20261102/` 仅新建链接，不迁移/删除旧数据。盘点时data1余270506663936字节。

## 已完成的真实GPU检查

三个目标均strict加载官方checkpoint、FP32、修正后的RoBERTa BPE，在第一个视频的首/中/末MOT候选帧与已封存完整margin比较：

|目标/同域模型|候选–句子行|最大margin绝对误差|符号翻转|目标GT打开|
|---|---:|---:|---:|---:|
|KITTI-V2 / V2-best|1182|0.0005518794|0|0|
|Dance / Dance-best|153|0.0002222061|0|0|
|SportsMOT / LaMOT-best|224|0|0|0|

三个源Train真实smoke也已通过：KITTI0000为4768行、Dance dancetrack0001为819行、Sports v_-6Os86HzwCs_c001为203行；仅合法自身源GT生成监督，不读取其他目标标签。

16项定向CPU功能检查已通过：最终PCD与margin、句子mask池化、padding不变性、空集合梯度、候选dropout/count同步、教师证据不重复累加、两条不同轨迹的No规则、dancing路由修正、模拟DS、流状态序列化、快速校准、原始及数值GT拒绝、YesTrack原生回退、短轨迹先验保持、目标身份输入使用封存分数且不改写复算文件，以及源验证损失的正单元归一化/JSON原生标量。尚不能据此宣称完整方法通过。

三域合法源缓存各做了独立的3步、batch256真实GPU训练功能检查，loss/总梯度有限且非零，没有覆盖正式Head checkpoint。按指定结构精确实现的参数量为1,521,676（约1.52M）。Sports的三帧smoke子集恰好全正，按该小子集公式其presence权重为0，梯度来自count项；正式Sports源Train有28,271正单元/2,130负单元，默认pos_weight=0.0753422，不是0，正式训练仍按完整源集比率计算，不采用smoke权重。

crowd-kit 1.4.2仅安装在本任务独立目录，2000条模拟任务上，本实现与其标签一致率1.0，两者准确率均0.946；灵敏度/特异度详见`ds_reference_comparison.json`。本实现带Beta(8,2)先验，参考实现不带该先验，不要求参数逐位一致。没有使用任何目标数据或GT。

## 当前实现与未完成项

已实现PresenceHead、完整候选特征导出、源缓存、源验证标签准备、固定默认Head训练、TOPK/中值平滑解码和独立TrackEval接口。三套源和九套目标特征、默认头及全部A官方HOTA已完成，见下方结果。

主动校验代码原型包含独立先验/证据、Beta(8,2)二分类DS、信息增益、三视图路由、原教师包装与跨环境服务、快校准、D1式LoRA、按视频恢复。crowd-kit模拟对照已完成；尚需真实集成验证、完整基线/消融及成本审计。40个Dance源Train视频的完整特征已导出，耗时3379.13秒；不会绕过当前失败的A门槛启动B/C。

源验证损失以匹配到的正候选数为count，和源训练/候选dropout后的定义一致；另保存官方场景GT数，用于独立离线报告场景存在性/数量误差。两种数量不混用；没有候选支持时不靠GT要求模型多选错误框。

### 用户补充的YesTrack对照定义

- `YESTRACK_BUDGET_RANDOM`：seed0派生视频子种子，均匀覆盖部分(句子,轨迹)对，优先首帧覆盖，余款缩小Δ；实际调用量严格匹配相同ρ的FULL。未覆盖的对逐行保留原生margin>0，**不是UNKNOWN**。
- `YESTRACK_BUDGET_PRIOR`：同样预算/回退；各句内按轨迹z降序覆盖，句间按文件名排序逐层轮转，不按目标GT分配预算。
- `YESTRACK_FULL_D10`：完整覆盖全部对、首帧加Δ=10关键帧、CROP、阈值0.4，不匹配预算；单独报告调用量/GPU时间并画入曲线。

按本轮ρ≤1和FULL预算定义，实际FULL调用数不超过全部候选–句子对数，故预算匹配版在“优先首帧覆盖”下通常没有重复调用余额；Δ取超过该视频最大轨迹跨度的整数。若预算恰足全部对，自然完整覆盖。关键帧为首观测、其后距离上次调用≥Δ的首个实际观测，按本次补充定义；不冒充上游全局帧取模逐位复现。每视频保存覆盖率、实际调用数和Δ。

用户已确认每轮提问上限以 `(句子, track)` 对为单位，不是跨句共用的物理track。规则未明示的实现约定会在正式运行前单列：不确定存在性范围0.1–0.9；存在性Yes取消低存在性时以至少1个目标求正常截距；预算总数向下取整；L2为0.1×mean(a²)。不会根据目标GT调整。

只有完成阶段A、发布三域同域HOTA门槛后，才启动B/C真实教师/适应。完整B/C矩阵须先实测估算；超过40 GPU小时则只做用户要求的主方法/关键对照并报告等待，不自行扩展。

完整B/C运行清单见`BC_plan.json`（108项，含用户补充的两个预算匹配YesTrack，以及三预算UNIFORM曲线点）。先做C1/C3的可恢复首视频pilot，状态保留在正式运行目录；后续从下一视频继续，不重跑健康pilot视频。成本估算以真实教师、非零LoRA更新和重打分计时为依据，不能把缓存命中或零更新当独立系统零成本。

### 已封存的Gate-Reassign参照（不是本轮新方法成绩）

|组合|HOTA|DetA|AssA|IDF1|输出行|历史教师判断|本轮新教师调用|
|---|---:|---:|---:|---:|---:|---:|---:|
|C1 V2→Dance|31.682|29.781|33.734|27.505|211085|29410|0|
|C6 LaMOT→Dance|31.126|28.046|34.577|26.638|211085|29410|0|

C1逐行复现并复用旧官方结果；C6用同一批既有封存门控和LaMOT margin做CPU派生后新评测。两者共享同29410次门控，不把其历史成本当成零，不把这里的HOTA当作FULL的结果。完整封存/指标SHA见`reference_controls.json`。

## 失败与修复

1. 首次目标smoke误用教师环境，缺`yacs`，在构建模型前失败。改用既有dfot FlexHook环境；不改共享环境，重做后通过。
2. KITTI源表达目录链接到既有合法源数据，初版白名单只允许外层根路径，误拦一个合法源表达。仅补入官方Train表达目录的真实路径；不放行其他项目代码或目标GT。失败尝试日志与成功smoke均保留。
3. KITTI额外全行误差门槛在frame101触发0.0010823011（符号翻转0），与既有公开`multidomain_generality/smoke_V2-best.json`的历史整批forward复算误差完全相同。额外全帧审计还发现KITTI0011/frame298和Dance0030/frame962近零分数的符号差异（分别误差0.0008719、0.0002027）。原验收范围是预固定三帧≤1e-3、0翻转，保持不变；其余全行误差和翻转如实记录，不把它们扩成额外停止条件。**目标/源验证的Head身份输入及Native/排序始终读已封存官方完整margin**；本轮`margin.npy`保留真实复算输出，仅作数值审计，PCD与旧文件不改写。源GT候选没有历史库，仍用其正常冻结decode margin。该读取策略在正式Head优化前落实并增加回归。未封存视频恢复，健康缓存不重算。
4. C6参照首次CPU TrackEval因调用传入相对路径、vendor工作目录解析错误失败。入口统一resolve绝对路径；保留失败日志后重评通过，预测SHA和评测规则未改。
5. GPU6共享环境后来出现约56.75GiB的新占用，一次KITTI恢复进程OOM（本进程约2.30GiB、卡仅余18.25MiB）。未停止其他项目。确认GPU6已无本项目CUDA进程后，登记集合由4/6改为4/5，GPU5入场余约77.9GB、1个进程；GPU4两个健康导出原样保留。新CUDA任务读取任务级登记，退休GPU6不能再启动任务；源头训练、标签准备、CPU解码/评测均支持锁内复用。两名无科学子进程的旧CPU等待协调器已退出，新队列继续，未停止健康GPU进程。
6. KITTI正式Head首epoch完成计算后，在写曲线时遇到NumPy bool不能JSON序列化，尚未保存任何有效Head checkpoint。将epoch指标返回Python原生float/int；同时按全部正单元汇总条件count CE，避免顺序验证的全负batch改变CE归一化。训练每个minibatch的BCE+0.5CE、权重、lr、seed等不变；checkpoint改为原子落盘后写日志。已增加最小数值回归并恢复该未封存epoch，失败日志保留，不伪造损失曲线或跳过epoch。

## 记录与恢复入口

- 盘点：`results/presence_active/inventory.json`。
- 大文件：`/data1/user/TTA-RMOT-assets/presence_active_20261102/`；逻辑路径仍为 `/data2/user/FlexHook-TTA/results/runs/presence_active_20261102/`。
- launcher日志、准入快照和逐视频特征seal均在上述目录；已封存视频不重算，只恢复未完成视频。
- 完整导出：`/home/szgx/anaconda3/envs/dfot/bin/python /data2/user/FlexHook-TTA/scripts/presence_launch.py --phase A-export`。

参考实现只作结构/接口参考，未加载其额外权重：[SAM3](https://github.com/facebookresearch/sam3)、[crowd-kit](https://github.com/Toloka/crowd-kit)、[YesTrack](https://github.com/ggbondrighthere24/YesTrack)。

## 阶段A完整结果

|目标|模型|Native HOTA|PRES_TOPK|Smooth|同域Δ|
|---|---|---:|---:|---:|---:|
|KITTI-V2|V2-best|42.752|40.599|40.635|-2.153|
|KITTI-V2|Dance-best|13.502|5.933|5.990|跨域|
|KITTI-V2|LaMOT-best|22.398|21.045|21.274|跨域|
|Dance|V2-best|8.218|4.849|4.981|跨域|
|Dance|Dance-best|32.139|21.605|21.921|-10.534|
|Dance|LaMOT-best|15.476|15.652|15.916|跨域|
|SportsMOT|V2-best|50.970|33.999|34.264|跨域|
|SportsMOT|Dance-best|20.560|10.618|10.497|跨域|
|SportsMOT|LaMOT-best|69.924|64.064|64.236|-5.860|

固定同域门槛：`FAIL_SAME_DOMAIN_INVESTIGATE_BEFORE_BC`。失败目标：['KITTI-V2', 'Dance', 'SportsMOT']。必须先定位实现/泛化原因；当前不启动B/C。

全部指标（含DetA/AssA/IDF1/IDSW、输出、空表达误检、场景及可观测存在性/数量误差）见`stage_A.csv`；源特征、训练及目标特征成本见`stage_A_costs.csv`；大文件路径和SHA见`artifacts_stage_A.json`。ORACLE只引用既有独立GT诊断，不是可部署方法。Sports主评测为v4全466序列，418敏感性保留相同v4内容。

数值口径详见`stage_A_numerical_audit.csv`：预固定三真实帧维持≤1e-3、0符号翻转；额外全行复算的微小误差/近零翻转不隐藏。PCD保留本轮真实计算，身份margin、原生判断和排序统一用原封存完整分数，复算margin.npy只作审计，没有改写旧分数或健康特征。GPU小时为共享卡上的成功阶段进程耗时，不是利用率积分；失败/初次载入额外开销见完整日志，未假定为零。
