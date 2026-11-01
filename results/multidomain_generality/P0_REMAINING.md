# 剩余四组合恢复 P0（全量推理前发布）

项目：TTA-RMOT。完整C1–C6范围不变，C1/C2不重跑；不训练、不调用MLLM，最多两卡。

LaMOT-best SHA256：`400980a1c92eacaa10206f1f26f2ffff20055b8a3925499afb6c6e7d9ba4213f`；大小995768578字节；strict=True全部key一致。

SportsMOT：45视频、466表达、26970原始有效帧、295779个McByte观测、2968201个候选–表达对。
466官方表达与GitHub原始文件逐字节一致；只独立导出language，原始targets/labels不进模型。
Stock mix loader读取空对象形式的视频索引（不是伪GT）和纯文本；GT审计0。
原图仅建数字JPG软链接视图，排除macOS ._资源分叉，官方TrackEval的目录计帧与seqLength一致；原图不修改。

| 源模型 | 目标 | 实机卡 | smoke秒 | 全量估计GPU小时 | GT打开 |
|---|---|---|---|---|---|
|LaMOT-best|SportsMOT|2|97.141|0.790154|0|
|LaMOT-best|KITTI-V2|0|267.839|0.827651|0|
|V2-best|SportsMOT|2|97.101|0.792368|0|
|LaMOT-best|Dance|0|171.376|2.119336|0|
|Dance-best|SportsMOT|2|96.349|0.787743|0|

所有smoke覆盖预先固定的第一个完整视频，原生getitem/forward、缓存特征与隔离query均通过既有1e-3/无符号翻转门。
Sports使用原mcbyte-sports-infer-tiny.yaml/mix loader/4帧/25token/224×672预处理，静态class=0、不加类别token。
KITTI/Dance按各自原yaml和infer.sh的rope-swin-tiny/roberta，FP32/TF32禁用；源模型仅换官方权重。

无Sports子集官方HOTA可核对：v5正文和infer.sh只给LaMOT四子集合并值，分享根目录无额外补充材料。
执行方式为单进程冻结encoder缓存、原始masked decoder，逐视频真实forward对齐；不宣称实际运行了DDP main.py自动GT评测入口。

GT模板原包仅418/466。v3先发布后，预测完整封存后才允许独立读取原始targets与Sports官方框；逐行复现418才接入48，任何失败停止Sports评测。
本阶段未打开GT值，未进行新完整评测，不把smoke通过当科研结论。完整日志和恢复路径见remaining_p0.json。
