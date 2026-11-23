# 原v8已完成预测的隔离保留

11项GPU工作与官方TrackEval已完成，但原supervisor退出码为1：旧审计对冻结回放使用了LoRA分支的反事实分层。原失败状态、原文件及全部预测不改，不声称11项全局审计通过，也不启动旧阶段4。

原封不动的旧校验器在独立目录复查9项非回放方法通过；不是删除两项回放来宣称全局通过。回放26处当前分层冲突全部记录，350份六存在性数组快照、问题/回答/DS与源MINUS一致，实际源决策的三档覆盖有效；完整证据见../original_v8_audit_diagnostic.json。

| pair | method | HOTA | episodes | global audit |
| --- | --- | --- | --- | --- |
| C1 | MINUS_SLOW | 38.738938585641804 | 425 | FAILED_NOT_OVERRIDDEN |
| C2 | MINUS_SLOW | 24.0898042336293 | 862 | FAILED_NOT_OVERRIDDEN |
| C3 | MINUS_SLOW | 61.47714552132973 | 466 | FAILED_NOT_OVERRIDDEN |
| C4 | MINUS_SLOW | 48.149705641172716 | 466 | FAILED_NOT_OVERRIDDEN |
| C5 | MINUS_SLOW | 27.95353889860121 | 862 | FAILED_NOT_OVERRIDDEN |
| C6 | MINUS_SLOW | 37.225905280090466 | 425 | FAILED_NOT_OVERRIDDEN |
| C1 | FULL | 37.640994989853404 | 425 | FAILED_NOT_OVERRIDDEN |
| C4 | FULL | 52.981674901766 | 466 | FAILED_NOT_OVERRIDDEN |
| C6 | FULL | 38.478534376501074 | 425 | FAILED_NOT_OVERRIDDEN |
| C1 | SLOW_REPLAY | 38.49159678807762 | 425 | FAILED_NOT_OVERRIDDEN |
| C4 | SLOW_REPLAY | 55.287959495718795 | 466 | FAILED_NOT_OVERRIDDEN |

原始关闭traceback包含在日志归档中；logs.json列出全部已关闭日志、原始冻结教师输出与服务器大文件路径。archive_validation.json只证明归档字节及教师revision/BF16概率，不能取代失败的全局实验审计。未打包路径如非空保留在retention_status.json，不冒称已关闭。

## 原生成表图的字节快照

原v8根目录的MD/JSON/CSV/PNG以及offline_diagnostics中的派生GT诊断，按原字节保留在original_public_outputs.tar.gz；每个源文件和归档的SHA见public_outputs_manifest.json。这些文件由原程序在全局审计失败前生成；其中原verdict/PASS字段只按原样保留，不覆盖supervisor退出1，不构成11项全局验收。未纳入旧stage4或其它未授权文件，旧目录不改。
