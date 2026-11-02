# Presence–Active：进行中的独立方法实验

本报告当前为工程与资产进度，不是科研验收结论。阶段A尚未完成训练/官方评测；阶段B/C尚未调用教师或启动LoRA适应。不得把下面的smoke或测试通过当作HOTA提升。

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

9项定向CPU功能检查已通过：最终PCD与margin、句子mask池化、padding不变性、空集合梯度、候选dropout/count同步、教师证据不重复累加、两条不同轨迹的No规则、dancing路由修正、模拟DS。尚不能据此宣称完整方法通过。

## 当前实现与未完成项

已实现PresenceHead、完整候选特征导出、源缓存、源验证标签准备、固定默认Head训练、TOPK/中值平滑解码和独立TrackEval接口。九套目标和三套源特征正在最多两张经准入GPU上全量导出；尚无新的完整HOTA。

主动校验代码原型包含独立先验/证据、Beta(8,2)二分类DS、信息增益、三视图路由、原教师包装与跨环境服务、快校准、D1式LoRA、按视频恢复。尚需真实集成验证、crowd-kit模拟对照、完整基线/消融及成本审计。

用户已确认每轮提问上限以 `(句子, track)` 对为单位，不是跨句共用的物理track。规则未明示的实现约定会在正式运行前单列：不确定存在性范围0.1–0.9；存在性Yes取消低存在性时以至少1个目标求正常截距；预算总数向下取整；L2为0.1×mean(a²)。不会根据目标GT调整。

只有完成阶段A、发布三域同域HOTA门槛后，才启动B/C真实教师/适应。完整B/C矩阵须先实测估算；超过40 GPU小时则只做用户要求的主方法/关键对照并报告等待，不自行扩展。

## 失败与修复

1. 首次目标smoke误用教师环境，缺`yacs`，在构建模型前失败。改用既有dfot FlexHook环境；不改共享环境，重做后通过。
2. KITTI源表达目录链接到既有合法源数据，初版白名单只允许外层根路径，误拦一个合法源表达。仅补入官方Train表达目录的真实路径；不放行其他项目代码或目标GT。失败尝试日志与成功smoke均保留。

## 记录与恢复入口

- 盘点：`results/presence_active/inventory.json`。
- 大文件：`/data1/user/TTA-RMOT-assets/presence_active_20261102/`；逻辑路径仍为 `/data2/user/FlexHook-TTA/results/runs/presence_active_20261102/`。
- launcher日志、准入快照和逐视频特征seal均在上述目录；已封存视频不重算，只恢复未完成视频。
- 完整导出：`/home/szgx/anaconda3/envs/dfot/bin/python /data2/user/FlexHook-TTA/scripts/presence_launch.py --phase A-export`。

参考实现只作结构/接口参考，未加载其额外权重：[SAM3](https://github.com/facebookresearch/sam3)、[crowd-kit](https://github.com/Toloka/crowd-kit)、[YesTrack](https://github.com/ggbondrighthere24/YesTrack)。
