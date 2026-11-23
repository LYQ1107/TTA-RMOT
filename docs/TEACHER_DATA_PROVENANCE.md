# P1 数据与权重来源

实验为`DEVELOPMENT_TEACHER_PROBE`：已多次研究的官方25Val/425原句，不是盲测，不是Train→Val UDA。
仅用固定DETR+NeuralSORT预测MOT、图像、过去四观测与Frozen分数采样，0帧号偏移；不改变框/ID。

- Smoke实际1649配对（上限1700），25视频/425查询，459张去重合法输入帧。
- 完整probe实际8381配对（上限8500），每视频固定5锚点，25视频/425查询，2218张输入帧。
- 清单分别`smoke_manifest.jsonl`/`manifest.jsonl`；receipt记SHA、规则和实际数量。无空候选删除。
- 只读staging位于`/data2/user/TTA-RMOT-assets/teacher-p1/staging_smoke`和`staging_full`。
  图像硬链接不chmod源文件；teacher内核Landlock只允许读stage，写仅限自身结果/cache等必要运行目录。
  独立单元检查已证明未staging的标签读与stage写均被内核拒绝；Python审计是第二道检查。
- 源开发固定0005/0011/0013/0019各16query窗口，源GT框/表达成员仅用于合法源校准准备与独立评价。
  源truth不放teacher staging，不上传源GT、原始图像或教师权重。

T0：`Qwen/Qwen3-VL-2B-Instruct`，revision `89644892e4d85e24eaac8bacfd4f463576704203`，Apache2.0。
官方LFS声明model.safetensors为4255140312bytes，SHA256 `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`。
下载目录在data2；必须下载完成并核对实际文件后才声明权重就绪。T0无YesTrack LoRA、无Dance微调。

T1：官方README指定`v2bestweight`，作者称在Refer-KITTI-V2训练并下采样负例；不能使用默认V1。
公开分享实际是178142422bytes的`yestrack.zip`。正常公开API验证/目录访问成功，但匿名二进制下载未提供签名。
全盘检查未找到可核验官方V2副本；检出的其它项目自训YesTrack adapter不是T1，未使用。
用户已明确“找不到就算了”：可记录T1缺失，不强行替换或把T0冒称T1。来源状态见assets审计。

Qwen公开模型卡不能提供完整预训练样本清单，无法排除Refer-Dance或同源人物视频重叠。
公开V2 LoRA的来源依作者声明，缺少可独立核验的完整训练账本；不作绝对无重叠保证。

官方40Train图像/表达已准备，但匹配的预测MOT仍缺；本轮不生成GT候选、不重新训练MOT。
P1图像、MOT、输入不含目标标签。所有新teacher预测封存后，独立评测器才读取目标GT。
