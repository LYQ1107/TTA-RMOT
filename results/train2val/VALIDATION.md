# 最终验证

最终运行（CPU，`CUDA_VISIBLE_DEVICES=''`）：

```text
/home/szgx/anaconda3/envs/dfot/bin/python -m pytest -q tests/test_train2val.py tests/test_diag_fit.py
...........................                                              [100%]
27 passed in 13.75s
```

27是不同测试的总数（9个Train→Val、18个D-FIT既有测试），不把此前重复通过的24/26/27次阶段运行累加。

- `python -m compileall -q flex_uda scripts`：exit0。
- `git diff --check`：exit0。
- 相对起点619390d，vendor/FlexHook、两个原D1/D3 YAML、yestrack_backend.py、visual_input_builder.py：diff为空。
- 相对实际训练源码a159741（公开同tree为5c9fc345），mllm_training.py、diag_fit.py、models与UDA配置：diff为空；运行中未更换数值实现。
- 相对619390d，历史results/diag_attrib及P1两个正式报告：diff为空。

所需真实检查：

1. Python audit和kernel-only独立子进程故意打开既有Val图片、Val MOT、Val文本manifest、旧Val教师缓存、旧D-FIT checkpoint，各5类均拒绝；这些是测试尝试，不计为真实训练GT读取。
2. 原始40个MOT文件共348930行，frame/id/box六列与转换文件逐行原文相等，ID/顺序保留；无过滤/插值。
3. 旧Val Naive的425组合、8500条采样位置/ID/框/原因/分数逐条重放一致，见naive_sampling_replay.json；只在独立审计进程读取，不进入训练。
4. 真实新缓存以split=train加载40视频/1560组合/7800窗口/31200有效软判断通过；另有对应CPU回归。
5. 两个200步真实Train-only学生smoke通过，随后正式训练都从官方V2重新初始化；四臂8973步完整覆盖31,200个唯一判断，目标GT/禁止输入尝试0，frozen不变，D3 head更新。
6. 四个final-step权重各完整25视频/425表达，3588445候选–表达分数，推理更新0；MOT框/ID不改。1700对预测/margin文件封存并核验SHA后才开GT。
7. 22套预测、9350个输出文件全部核验；16套新官方TrackEval首次完成，6套严格逐行等价复用。官方CSV的25视频TP加权重组与COMBINED全精度HOTA一致，sealed输出行数与官方Dets一致。
8. 完整官方GT为212541观测，40 Query-Present/385 Query-Empty；42782 GT有目标query-frame、390854 GT空query-frame，无帧/ID平移，官方mark!=0/class1过滤不变。

最终公开报告元数据的`information`按来源区分Source-only、历史同Val训练、GT-free校准、同域官方监督和本轮Train40→Val25；不会把同域官方权重误标为无目标监督，也不把旧同集学生当新Train→Val结果。

官方评测后仅重新生成CPU汇总/公开说明，没有新TrackEval、GPU推理、训练、模型选择或阈值搜索。大文件路径/SHA、完整原始官方summary/detailed、阶段stdout和恢复路径均随本轮结果提交。
