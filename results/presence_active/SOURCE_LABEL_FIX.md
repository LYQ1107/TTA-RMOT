# Sports源标签修复（不是新增方法/超参数搜索）

全源CPU核对发现Sports Train中9个视频、27条表达文件共564/335246个缓存候选标签不一致：相同句子文本出现在多个表达文件，但原始逐帧targets不同，旧代码以language文本为键取并集。KITTI和Dance全量标签/缓存逐项通过，早先首视频检查没有覆盖该Sports问题。失败完整记录保留在`source_label_full_audit.csv/json`。

修复改用**表达文件ID**，逐条原始targets为准；不改原官方GT、不把GT引入目标运行时、不改冻结PCD/margin/框/ID。新标签独立放在`source_labels_by_expression_v2/SportsMOT/LaMOT-best/`，旧labels.npy、seal、checkpoint和所有旧结果保留。

重新运行两个**既有**Sports源头配方：默认lr3e-4和已固定的单次缓解lr1e-5。网络、损失、增强、seed0、batch256、cosine20epoch/patience3全部不变；不是增加第三个学习率。每项独立输出到`heads_source_labels_v2/<variant>/LaMOT-best/`。

在修正头任何新HOTA前，按相同源验证loss规则封存`source_lr_trial_selection_v2.json`；KITTI/Dance的选择和checkpoint不变。修正后的默认LaMOT头补做三个目标的A评测，缓解头做Sports同域评测；全部预测先封存再独立评测。

旧Sports头及其跨域A结果标为**源标签合并bug的历史结果，不是主结果**；新结果独立v2表呈现，不覆盖。修复规模很小，不能预先断言它能解释全部HOTA下降。仍须各自满足原生−1门槛，当前B/C与教师调用继续锁定。
