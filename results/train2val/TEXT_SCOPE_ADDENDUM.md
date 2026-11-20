# P0引出的报告范围补充（在本轮Val预测前固定）

P0发现Train与Val共享8个表达文件名。原用户规则的425序列主表、全部对照、bootstrap、seed阈值与保留率分类一字不改；原PREREG.md也不改。

为避免把所有Val表达都称作未见，预测全部封存后，单独报告整句已见/未见分组：

- Train的独立sentence-only runtime文本集合，小写并合并空白；与已封存Val运行时原始文本作精确匹配，不使用任何GT标签定义分组。
- 对每个主模型复用同一次官方TrackEval detailed CSV，用原HOTA.combine_sequences汇总各组的HOTA与CLEAR FP，并给出组内GT帧宏平均R@1、序列/GT数量。
- 它只描述语言覆盖范围，不参与预注册判定、不决定任何训练/阈值/checkpoint/seed。GT只在独立封存后评价进程读取。
- 这是整句精确匹配，不是词汇、颜色属性或组合概念全部未见的证明；Val仍是历史开发集。

该补充不启动新GPU工作，不增加教师判断、不改变40×39的Train采样、不删除425条评测序列。
完整官方CSV用压缩归档与逐文件SHA公开，避免把重复未压缩大文件放进git。
