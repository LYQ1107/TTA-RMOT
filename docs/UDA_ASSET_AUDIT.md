# 实际资产审计

机器可读记录：`results/uda/assets/asset_audit.json`、`dance_manifest_audit.json`、
`train_image_preparation.json`、`official_tracker_release.json`。这些保留首次审计时间，不改写成当前训练状态。

- 官方V2权重实际重新计算SHA256：`71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613`；
  路径`/data3/user/TTA-RMOT-assets/flexhook/SOTA_ckpts/refer-kitti-v2-best.pth`。每次真实运行严格加载，
  不能用Dance插件权重初始化V2→Dance方法。
- 源训练完整17视频、8896表达、5939图像齐全；源GT合法，四个source-val独立。
- 原Refer-Dance ZIP含全部40Train图像。已从既有合法ZIP只提取41796张训练PNG并校验ZIP CRC；
  未读取/提取其GT payload。25Val既有图像及425表达齐全。
- FlexHook公开作者分享中的`tracker_outputs/DETR-NeuralSORT-Dance`实际列出25个Val文本+2CSV，
  其它候选文件夹也没有匹配的40Train输出。原始请求的40Train→25Val UDA因此未启动。
  首次访客uk字段错误及修正后的正常公开分享目录访问记录分别保留；不保存cookie、密码、签名下载URL。
- 当前25Val官方候选完整211085行，425查询共3588445候选-query记录。旧文件漏692末观测的8.206
  仅作历史；匹配基线8.2180。新CPU重放425预测哈希全部一致，官方TrackEval重新得到相同数值。
- tokenizer person→5970的修复在真实GPU每组重验；PCD768、LoRA Q/V rank4最后两层，严格加载全部匹配。
- 首次GPU快照与每次新任务入场均记录UUID、进程数与显存；共享已获用户授权，不误称独占。
  每worker24GiB分配上限、8GiB入场余量，最多四张本任务物理卡。

40Train缺的是匹配前端的预测MOT/完整复现资产，不是图像未下载，不用GT生成假预测，不偷换YOLO/ByteTrack。
用户已明确当前先25Val；本轮不再因Train资产卡住该诊断，也不自行重训检测器。
