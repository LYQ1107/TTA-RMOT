# PSR 复现与恢复入口

项目仅为 `LYQ1107/TTA-RMOT`，服务器根 `/data2/user/FlexHook-TTA`。
本轮用户明确授权25Val/425表达同集无标签开发研究，不是40Train→Val、盲测或Source-Free Online TTA。
以下是恢复/复现说明，不是让健康队列重复启动的指令。

## 外部代码与环境

FlexHook官方源代码固定 `bd1acc38634b28525d54dc6e0fcb38335f0029f9`：

```bash
git clone https://github.com/buptLwz/FlexHook vendor/FlexHook
git -C vendor/FlexHook checkout bd1acc38634b28525d54dc6e0fcb38335f0029f9
git -C vendor/FlexHook apply ../../results/positive_free/full/manifests/vendor_tracked_source.patch
```

仅在新clone使用上面的patch命令；当前服务器已经应用，不要重复应用或覆盖vendor改动。
该补丁只让test loader不再无条件打开目标labels.json；本轮轨迹末观测完整覆盖由自己的DanceEpisode执行，不复用旧exclusive-range候选缺口。
LoRA、PCD返回钩子、训练器和推理器都在本仓库，不依赖另一项目的模型代码。

实际环境与交付核验见 [delivery_audit_primary.json](../results/psr/assets/delivery_audit_primary.json)：

| 用途 | Python入口 | torch | transformers | numpy |
|---|---|---|---|---|
| 学生/CPU评测 | `/home/szgx/anaconda3/envs/dfot/bin/python` | 2.7.0+cu128 | 5.4.0 | 1.26.4 |
| 冻结MLLM | `/data2/user/TTA-RMOT-assets/teacher-p1/env/bin/python` | 2.8.0+cu128 | 4.57.6 | 2.2.6 |

两者均Python3.10.20；完整相关包版本在JSON，不要求将两个环境混成同一个。
学生环境必须保留 `compat/sitecustomize.py` 的规范RoBERTa BPE修复，运行时检查 `person -> [5970]`。
教师固定Qwen3-VL-2B-Instruct revision `89644892e4d85e24eaac8bacfd4f463576704203`，BF16、无量化/adapter/更新；不是YesTrack未取得的官方V2 adapter。

## 固定资产与合法权限

- V2 checkpoint：`/data3/user/TTA-RMOT-assets/flexhook/SOTA_ckpts/refer-kitti-v2-best.pth`，SHA256 `71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613`。
- Source：`/data3/user/TTA-RMOT-assets/flexhook/refer-kitti-v2`，17Train视频/8896表达及预注册77覆盖窗口；source-val固定64例选择checkpoint。
- Target图像：`/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance`。
- 固定MOT：`/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs`，25文件/211085观测。
- 完整Frozen：`results/runs/pf_zero_completebank_full_20261025_merged/ZERO_UPDATE`，3588445候选-查询分数。
- 已封存监督：`/data2/user/TTA-RMOT-assets/psr/val_v1/{NAIVE,PSR,UNIFORM,CONFIDENCE}`；raw SQLite `/data2/user/TTA-RMOT-assets/psr/cache/raw.sqlite`。
- 40Train图像41796/表达1560存在，但相同前端预测0/40；不得用GT轨迹或替换检测器补造。详见formal_train_audit.json及恢复条件。

学生仅获源GT、目标图像/原句/预测MOT与封存伪标签白名单；教师输入独立隔离；目标GT仅由预测封存后的CPU评价进程读取。
Source图像/表达的旧LocateMOT目录别名只是只读数据来源，不使用其代码、checkpoint或GPU任务。
图像、权重和完整原始教师缓存不上传；所有checkpoint路径与SHA在各训练收据，逐episode分数路径与SHA在episode_recovery.json。

## 从已有缓存恢复

先确认原调度器已经结束；健康作业继续原等待句柄，不重新启动第二个调度器。
完整单位会按真实收据保留，失败单位从最新有效checkpoint恢复，不重跑健康425预测。

```bash
/home/szgx/anaconda3/envs/dfot/bin/python scripts/run_psr_matrix.py \
  --root results/runs/psr_val_20261029_v1
/home/szgx/anaconda3/envs/dfot/bin/python scripts/finish_psr_study.py \
  --run-root results/runs/psr_val_20261029_v1
```

条件确认使用相同8500教师标签、相同采样seed17，只改变学生seed0/42：

```bash
/home/szgx/anaconda3/envs/dfot/bin/python scripts/run_psr_confirmations.py
/home/szgx/anaconda3/envs/dfot/bin/python scripts/run_psr_parent_control.py
/home/szgx/anaconda3/envs/dfot/bin/python scripts/finish_psr_confirmations.py
```

Parent控制是POST_HOC机会匹配正例权重对照，不是预注册新算法；263机会/526原端点项，新增弱标签0。
这三条属于同一实验队列，当前已运行，不能在未退出前重复启动。最多4物理卡，16GiB allocator+4GiB准入余量，授权共享时优先进程少的卡。
首次复现到不同机器需显式调整配置/manifest资产路径并重新审计；不要假装本机绝对路径可直接跨机器使用。

## 从头生成监督（仅新独立目录）

`prepare_psr_manifests.py --split val --val-authorized --output NEW_ROOT --frozen-root EXISTING_FROZEN` 先固定四策略清单；
`launch_psr_teachers.py --pseudo-root NEW_ROOT --output NEW_TEACHER_ROOT --smoke` 做4真实判断，正常后去掉smoke完成四策略；
逐策略 `seal_psr_supervision.py --pseudo-directory NEW_ROOT/POLICY --raw-directory NEW_TEACHER_ROOT/POLICY`，
再 `audit_psr_supervision.py --pseudo-root NEW_ROOT --output NEW_AUDIT_ROOT`。
既有目录已经只读封存，原manifest/seal不要改写；首次教师/学生流程均必须通过实际接口和梯度检查。

## CPU重建评价与交付核验

现有 `finish_psr_study.py` 保留已通过CPU阶段；不需要重跑GPU。
官方结果、双向数量控制和排序/覆盖/Query-Present/Empty归因均已在首轮完成。
数量控制不是独立可部署模型，不能把其CPU时间当整系统成本。
主表由官方四位汇总相减，视频CI由未舍入CSV充分统计池化；差值末位可因四舍五入相差0.0001，不是协议改变。

```bash
CUDA_VISIBLE_DEVICES='' /home/szgx/anaconda3/envs/dfot/bin/python scripts/audit_psr_delivery.py
```

该命令实际打开全部预测压缩包、核对425预测/25视频/官方425行、8学生最终收据和补丁，读取版本而不加载GPU模型/目标GT。
它验证交付完整性，不把工程通过判为PSR科研成功。最终结论必须看完整RMOT、普通对照、数量残差和不确定性。
