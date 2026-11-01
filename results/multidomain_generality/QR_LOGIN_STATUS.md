# 百度网盘登录与服务器下载会话核查

2026-11-01；承接 `e93b0f95ccfe21092a6fa71295eabee5fedb0db6`。

用户明确授权扫码登录，随后在 Codex 内嵌浏览器打开的百度网盘官网自行完成登录，并告知“登录成功”。这是用户报告的浏览器登录成功，不是服务器下载器已登录的证明。

## 最新补充：用户明确授权读取 Cookie

用户随后明确表示“你可以去读取cookie登录”，故读取本次网盘会话的授权已足够，不再以缺少授权作为阻塞。限定检查当前执行用户的浏览器位置，不读取其他用户或其他网站的数据，不打印 Cookie 值。

实际执行主机为远程 Linux `szgx-c10u14-r5500g5-a800-23`。当前用户的 Codex/ChatGPT/Chrome/Chromium 浏览器资料目录均不存在；唯一找到的 Firefox Cookie 数据库是 10 月初的旧文件。对数据库只做只读、Baidu 域名限定的 host/name/time 查询，仅发现 BAIDUID、HOSUPPORT、pplogid，没有 BDUSS 或 STOKEN，没有读取明文值。因此未取得可用于下载的账号会话；不能把允许读取 Cookie 等同于服务器已经有 Cookie。

已准备服务器私有接收目录 `/data2/user/TTA-RMOT-assets/flexhook/lamot_20261101/auth_sessions/browser-import/`（权限 700，在 Git 仓库之外）。需要用户提供仅本次百度网盘 Cookie 的可访问文件路径，或把导出文件放入此目录；文件权限应为 600，不应将 Cookie 值粘贴进聊天或报告。尚未收到该文件，不反复匿名请求、不读取无关浏览器数据。

## 服务器扫码尝试

- 只请求 `passport.baidu.com` 的官方二维码与状态接口，未执行第三方下载的登录客户端。
- 原始官方 PNG 的二维码没有留白边界，OpenCV 首次未识别；只给内存中的识别输入增加白边后，解码得到 HTTPS `wappass.baidu.com`、`cmd=login`、`tpl=netdisk`，sign 与本次请求一致。存储和展示的原始图像没有被改写，也没有把图片下载 URL 编造成登录二维码。
- 但官方响应的应用授权提示是“登录后威马将获得百度账号的公开信息（用户名、头像）”，与预期应用不一致。官网当前 SDK 确实配置 `product=netdisk`、`subpro=netdisk_web`，也调用同一个 getqrcode 接口；按其参数重查仍有该提示。无法据此确认授权对象，未把这张码作为可核验的网盘登录码交给用户确认。
- 唯一已开始等待扫码的自有进程已终止；状态改为取消，不继续显示 WAITING。未终止其他任务。
- 两个本任务私有扫码目录中，`session.cookies.json` 均不存在；没有服务器登录成功记录。未索取密码或 Cookie，未导出浏览器凭据，未向 Git 提交 QR 图片、二维码 sign 或凭据。

## 用户在浏览器登录之后

当前可用工具能请求打开浏览器页面，但没有读取或操作该已登录页面、复用其下载会话的接口。查找现有工具与连接能力后，未找到能接管本次 Codex 本机浏览器会话的已连接接口；外部云浏览器不是这个会话，不能宣称会自动继承登录。没有读取本机浏览器 profile 或 Cookie 文件。

已请求在同一 Codex 窗口打开官方资产分享页：

[FlexHook_best](https://pan.baidu.com/s/1L-43y9SFDKmgl3dJNRlvNA?pwd=d3qj)，提取码 `d3qj`。工具只返回打开请求已排队，不等于文件已下载。

因此现在可由用户在已登录的官方页面下载以下精确资产，再提供文件所在机器和路径；无需发送账号凭据：

1. `/FlexHook/SOTA_ckpts/LaMOT-best.pth`
2. `/FlexHook/tracker_outputs/McByte-SportsMOT/`
3. `/FlexHook/datasets/LaMOT-main/annotations_v1/val/SportsMOT/`
4. `/FlexHook/datasets/LaMOT-main/sequences/val/SportsMOT_labels.json`
5. `/FlexHook/datasets/LaMOT-main/sequences/val/SportsMOT_RESOLUTION.json`
6. `/FlexHook/datasets/LaMOT-main/sequences/val/TAO_frame2file.json`
7. `/FlexHook/datasets/LaMOT-main/gt_templates/SportsMOT/`，仅供后续独立评测。

服务器最终接收根目录仍为 `/data2/user/TTA-RMOT-assets/flexhook/lamot_20261101/`，保留来源、字节数、SHA256，不覆盖现有文件。到位后按原协议核验及继续，不绕过资产准入。

本阶段下载新增成功 0；没有新 GPU 推理、训练、微调或 MLLM 调用；C1/C2 和所有历史结果不变，C3–C6 仍等待官方资产。新增登录脚本只通过语法编译；实际登录和凭据交换尚未通过端到端验证，不能标成可用的已登录下载器。
