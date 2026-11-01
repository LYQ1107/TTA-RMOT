# P0恢复盘点（补充预注册后，P1前）

补充预注册远端：0f9c61c66f26a72ace8f044864b8b0f71a8ecc0a。原始PREREG正文未改，完整C1–C6范围恢复；Sports按466条。

| KITTI目标/官方源权重 | strict | 完整一视频margin数 | 耗时(s) | 全量估算GPU h | GT打开 |
|---|---|---:|---:|---:|---:|
| V2-best | 全部key一致 | 209608 | 188.671 | 0.572601 | 0 |
| Dance-best | 全部key一致 | 209608 | 205.907 | 0.630457 | 0 |

两套均用官方kitti2.yaml+infer.sh的rope-swin-tiny/roberta覆盖、FP32、同一官方RMOT_Dataset test；实际getitem/forward对照无符号翻转，box/grid一致。类别不添加新token、不替换car/ped语义。
缓存只复用同一个冻结checkpoint的编码；两模型缓存独立。源历史分数不被重算覆盖，源V2 smoke重算的历史最大差0.0010823如实记录（历史库原值保留）。

公开下载已有515文件（含官方LaMOT原始Sports JSON508份、README、树和压缩包），磁盘下载前余18268487680字节。原始JSON数字ID命名与FlexHook表达名seqmap并不直接相等，不能把下载成功宣称为466条封装核验通过。
百度提取码已通过，匿名能列出995768578字节的LaMOT-best和45个McByte预测文件；匿名sharedownload返回errno=2（参数/签名未满足），没有返回下载链接。未将errno=2谎称为已证实的登录错误。公开BaiduPCS-Go文档将登录列为必做；无可用账号凭据，不安装需登录工具/不索取密码，不绕过签名。该路径停止，精确手动文件见downloads.json。
FlexHook代码MIT；单独checkpoint/数据授权未列；LaMOT仓库无独立LICENSE，不假定其数据继承MIT。未执行下载内容代码，未改vendor代码，未重下载Sports图像，未自跑tracker。

C1复用路径和SHA已核验，C2及KITTI补分可按用户授权继续。Sports/LaMOT所有组合仍受资产核验门槛约束；不能把2/6完成当三域普遍性成功。

首次启动环境选错（vllmedit缺yacs），切回既有dfot且目标配置依赖回归通过。没有改动共享环境。smoke成功后先暂停P1到此记录远端发布；健康0005视频保留，不重跑。

论文最新版arXiv v5正文仅LaMOT汇总56.77，infer.sh汇总56.756；未找到Sports子集官方数字。仓库无补充PDF，CVF补充页面403；不将不存在的子集数字充作自检门槛。

论文：https://arxiv.org/html/2503.07516v5；官方代码：https://github.com/buptLwz/FlexHook；工具登录说明：https://github.com/qjfoidnh/BaiduPCS-Go#初级使用教程。
