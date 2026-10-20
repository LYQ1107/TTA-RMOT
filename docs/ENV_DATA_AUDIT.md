# LocateMOT QCS-TTA 环境与数据审计

更新时间：2026-10-20。项目身份确认：`/data2/user/LocateMOT`。

## 资产

| 资产 | 实际路径 | 状态 |
|---|---|---|
| TempRMOT 源码 | `/data2/user/reference_envs/temprmot/source` | commit `6a65640d849fdee4a32bb055945ee34c3b0edeb1` |
| V2 源权重 | `/data2/user/LocateMOT/data/checkpoints/checkpoint_rk2.pth` | 1,096,316,824 bytes；SHA256 `4953bd2b2966f752b7d5dced94216f0ddacd04158b1c85c3fc3534a333e9581b` |
| 本地文本编码器 | `/data2/user/reference_envs/temprmot/roberta-base` | 已存在，离线加载 |
| AerialMind 完整 release | `/data2/user/LocateMOT/data/AerialMind` | 93 视频、24,617 expressions；完整引用帧审计通过 |
| KITTI 图像 | `/data2/user/LocateMOT/data/kitti_tracking` | 约 6.1 GB |
| Refer-KITTI-V1 标注/表达 | `/data2/user/LocateMOT/data/refer_kitti_v1` | 已存在 |
| Refer-KITTI-V2 标注/表达 | `/data2/user/LocateMOT/data/refer_kitti_v2` | 已存在，使用修正版 `labels_with_ids` |

V2 官方测试视频 `0005,0011,0013,0019` 的 runtime/evaluation manifest 已生成：各 457,503 行；runtime manifest 不含 GT 字段，audit 结果为 `complete`。V1 开发/源端校准 manifest 为 5,320 行。

AerialMind 已形成可核验的 expression+ID 接口：官方 17 个 VisDrone 和 13 个 UAVDT 测试视频均在 `/data2`，标签格式和 1-based frame convention 已纳入 manifest/evaluator。完整数据审计 receipt 为 `results/audits/aerialmind_dataset_audit.json`；官方 63 个训练视频的独立性审计为 `results/audits/aerialmind_training_split_audit.json`。Refer-BDD、Refer-Dance 仍未作为本轮目标域使用。

## 环境与 GPU

使用项目独立 `.venv`；TempRMOT 的 CUDA ops、兼容依赖和本地文本资源来自 `/data2/user/reference_envs/temprmot`，没有改写其他项目环境。V2 source sanity 已在单卡逻辑 `cuda:0` 上通过，物理卡由 `scripts/select_gpu.py` 按“显存足够后进程少优先”选择；一次核查选择物理 GPU 2（80 GB，总空闲约 81 GB、compute process 为 0）。

下载器默认禁用 HTTP(S) 代理，支持 `.partial` 断点文件，并在原子改名后核对文件大小和 SHA256。
