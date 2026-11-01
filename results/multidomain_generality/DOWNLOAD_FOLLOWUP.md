# 公开资产下载续办：真实传输仍被 CDN 拒绝

> 历史记录：本文记录匿名下载阶段的失败。之后用户通过服务器原生官网二维码登录成功，已获得真实 HTTP 200 资产传输，当前状态见 [QR_LOGIN_STATUS.md](QR_LOGIN_STATUS.md) 和 downloads.json；下文“新增 0”仅指该次匿名续办，不是当前累计下载量。

2026-11-01；承接 `fb484d1e60d65506d2c8fe85fc748f984db9c08a`。本轮用户要求“你可以继续下载啊”，未改变预注册科学协议。

## 结果

实际继续尝试下载，**新增成功落盘的模型、轨迹、标注文件为 0，新增有效 payload 为 0 字节**。不是因为没有下载授权，也不是磁盘不足。已有 GitHub 原始 SportsMOT 标注 508 份和现成 45 视频图像保留。C1/C2 的预测、分数、指标与封存不变；C3–C6 仍未通过资产准入，未启动推理。

网盘公开提取码验证、目录列表、签名接口均成功。不透明的大文件响应不能被简单解释成登录失败；小文件确实能返回链接，但该链接不能实际传输。

| 实际检查 | API 返回 | 真实传输 |
|---|---|---|
| 小 McByte 文件，`type=nolimit` | errno 0、明文 dlink | HTTP 403 / 31360，`expire time out error` |
| 同一个小文件，不传 type | errno 0、明文 dlink | 同样 HTTP 403 / 31360 |
| LaMOT-best，`type=nolimit` | errno 0、不透明字符串 list | 没有可直接使用的链接 |
| LaMOT-best，`type=batch` | errno 0、打包 dlink | HTTP 403 / 31064，`file is not authorized` |

最新有限检查的小文件：API `server_time=1793530766`，链接 `dstime=1793527165`、`expires=1h`，差 3601 秒；下一次不传 type 的请求同样差 3601 秒。未修改带签名的下载 URL 或时间戳以绕过服务端校验。此前已尝试重新获取匿名提取会话、加新请求时间参数和 `Cache-Control: no-cache`，以及浏览器/`pan.baidu.com` User-Agent，仍收到同样的过期错误。clienttype 5/12 返回 errno 9019，也没有可用链接；没有把该错误解释成确认的登录失败。

第一轮实际逐文件请求了 149 项：147 项 HTTP 403，2 项大文件没有可用 dlink，下载 0。其历史日志状态曾用 `PARTIAL_PRESERVED`，但失败发生在打开本地文件之前；实查没有产生 partial 文件。该记录保留，并在 downloads.json 的续办摘要中明确纠正措辞。

来源及每次结果见 [downloads.json](downloads.json)。没有记录 Cookie、签名、临时下载链接；没有搜寻或索取账号凭据，没有使用第三方解析代理、转存账号或执行下载内容中的代码。

## 完整目录与本地资产核查

初次调用请求 `num=1000`，服务端实际最多返回 100 项，因此发现并修复分页截断：改为每页 100，持续分页到末页。此前 149 项的尝试不冒充全量资产下载。

完整清单现在覆盖 515 个选定文件：45 个官方 McByte txt、466 个官方 SportsMOT 表达 JSON、3 个 mix 元数据 JSON、1 个 LaMOT checkpoint。列表不是已下载文件。

| 资产 | 官方列表 | 本地状态 |
|---|---:|---|
| LaMOT-best.pth | 995,768,578 字节 | 缺失，strict 加载未运行 |
| McByte-SportsMOT | 45 文件，总计 13,923,805 字节 | 本轮下载 0；不自行生成 tracker |
| FlexHook SportsMOT 表达 | 466 JSON | 466 文件名/大小均与已有 GitHub 下载相符；未证明字节一致 |
| SportsMOT_labels.json | 86,531,476 字节 | 缺失 |
| SportsMOT_RESOLUTION.json | 1,937 字节 | 缺失 |
| TAO_frame2file.json | 2,136,244 字节 | 缺失，仅 mix 构造辅助元数据 |

共享接口的 `md5` 字段含非十六进制字符，例如权重字段 `3d9fe4de8t192b80831450c3dcb3310b`，不是可直接比较的明文 MD5。保留为不透明元数据；下载器以实际字节数验证完整性并记录实际 SHA256，不把该字段当成文件损坏证据。所有 466 个已有 GitHub 文件虽然同名同大小，仍不据此宣布与网盘版本内容相同。没有为判断等价性解析 referring 标签，未创建模型运行时标签输入。

本轮也查看了 [FlexHook 官方仓库](https://github.com/buptLwz/FlexHook)、[发布页](https://github.com/buptLwz/FlexHook/releases) 和 [Tracker Output 议题](https://github.com/buptLwz/FlexHook/issues/4)，以及 [LaMOT 官方仓库](https://github.com/Nathan-Li123/LaMOT)。本次未找到作者提供的可用非百度权重/轨迹镜像；不能将“本次没找到”宣称为证明所有镜像不存在。原始 LaMOT JSON 不能替代官方 McByte 预测及 FlexHook 处理后的 labels JSON。

## 恢复所需文件

无需再次授权公开下载。需要能实际传输的官方链接，或将这些官方文件手动下载到服务器：

- `/FlexHook/SOTA_ckpts/LaMOT-best.pth`
- `/FlexHook/tracker_outputs/McByte-SportsMOT/`：45 个文件名已在 downloads.json 枚举。
- `/FlexHook/datasets/LaMOT-main/annotations_v1/val/SportsMOT/`：需核验网盘版本与已有 GitHub 版本是否一致。
- `/FlexHook/datasets/LaMOT-main/sequences/val/SportsMOT_labels.json`
- `/FlexHook/datasets/LaMOT-main/sequences/val/SportsMOT_RESOLUTION.json`
- `/FlexHook/datasets/LaMOT-main/sequences/val/TAO_frame2file.json`
- 官方评测布局 `/FlexHook/datasets/LaMOT-main/gt_templates/SportsMOT/`；只供后续独立评测进程，不能进入模型。

分享：[FlexHook_best 官方网盘](https://pan.baidu.com/s/1L-43y9SFDKmgl3dJNRlvNA)，提取码 `d3qj`。保存根目录：`/data2/user/TTA-RMOT-assets/flexhook/lamot_20261101/flexhook_baidu/`，不要覆盖已有文件。对账号密码没有需求。到位后仍须逐项完成原来的 strict、45 视频、466 序列、官方 mix 冒烟、成本及复现门槛，不因下载授权而跳过。

## 实现、测试与资源

`scripts/multidomain_baidu_resume.py` 提供公开文件下载、完整分页清单、有限传输诊断与离线摘要模式。没有改 vendor 代码、模型、协议、阈值或历史指标。3 个下载回归检验分页、不透明响应和实际字节/SHA256 记录；它们使用模拟 HTTP，不代表真实远端下载已经成功。现有 3 个多域检查一并重跑，compileall 检查通过。

新增 GPU 使用、训练、微调、MLLM 调用均为 0。实际下载失败被保留，不启动无效重试循环，不将状态写为后台下载中。
