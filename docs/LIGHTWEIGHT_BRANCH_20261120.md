# 轻量分支与 NAS 实验资产

2026-11-20，按用户“小于100 MB，只保留重要代码和实验记录”的要求，重建
codex/presence-active-v8p-20261106。此分支从一个无父提交的精简快照开始，避免继续
携带旧的大文件历史。旧版本的完整原件保留在 NAS；其他 GitHub 分支没有修改。

## 保留内容

- 全部原有代码、配置、测试、研究文档、恢复脚本和完整研究历程 TXT。
- 各阶段正式报告、预注册、最终状态、主结果、消融、预算、成本和统计区间。
- 小型 JSON/CSV/TXT、图表，以及明确保留的 v7/v8/v8P 存在性明细。
- 失败记录、负结果和旧 v8 审计失败说明，不改实验数值或验收结论。

一般实验记录采用单文件256 KiB上限；关键存在性记录原样保留。未将大型记录截断
成看似完整的文件：未纳入的原文件统一进入 NAS-only 清单。

## 放在 NAS 的内容

checkpoint、数据集、原始教师回答、完整日志归档、全量预测/特征缓存及二进制分片。
此前 Git 快照约1.42 GB，主要由归档和分片构成，没有直接追踪的 .ckpt/.pth 权重。
本次从 GitHub 快照移出756个文件、1,360,747,940字节；没有删除 NAS 原件。

完整路径、原大小和 Git blob SHA 见
[NAS-only清单](NAS_ONLY_ARTIFACTS_20261120.json)。原代码快照为
4294b7d269d4a9116de1f4b98b23485e832ddbcf。A100 保留本地备份引用
refs/backup/v8p-before-slim-20261120-4294b7d；旧 A800 镜像历史也继续保留。

NAS 挂载根目录：

    /data3/liuyeqiang/nas/A800

NAS 原仓库对象存储：

    /data3/liuyeqiang/nas/A800/data2/user/TTA-RMOT/.git

如果需要恢复清单中某一个文件，可以按记录的 SHA 读取原始字节：

    git --git-dir=/data3/liuyeqiang/nas/A800/data2/user/TTA-RMOT/.git cat-file blob <sha> > <destination>

也可以沿当前 A100 的 assets/、results/ 入口访问完整实验资产；历史结果目录仍只读。
NAS-only 文件不应再次提交到轻量分支。

## 下载和体积检查

    git clone --single-branch --branch codex/presence-active-v8p-20261106 https://github.com/LYQ1107/TTA-RMOT.git
    cd TTA-RMOT
    python tools/check_publication_size.py

检查器要求当前快照和该分支可达 Git 对象的原始字节均小于100,000,000，并拒绝
权重、缓存、归档或分片。发布时另用全新单分支克隆核对实际下载结果。

必须区分“当前轻量分支”和“整个远端仓库”：其他分支仍可能引用旧历史，
GitHub 的总体 size 统计与未回收旧对象也不会因此立即变小。普通全分支 clone
可能下载其他分支历史，应使用上面的单分支命令。

本轮只做发布整理，没有启动训练、教师查询或 GPU 评测。
