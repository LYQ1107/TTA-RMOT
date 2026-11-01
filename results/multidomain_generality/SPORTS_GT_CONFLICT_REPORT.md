# SportsMOT GT：v3精确门失败与v4授权

九套官方模型×目标分数已封存，C1/C2/C5/C6已完成；C3/C4尚未进行任何SportsMOT正式评测。预测先封存提交为03cf5de30e16b6e396754112fcc3208106e22d55。

v3实际结果：410/418逐行全字段与顺序一致，8条失败，涉及5个视频。按原规则立即停止，未生成原缺48条GT，未覆盖原模板。CPU重建/核对8.756秒，新增GPU0小时。

| 序列 | 旧模板行数 | 原始标注重建行数 | 原始标注额外行数 |
|---|---:|---:|---:|
| v_G-vNjfx1GGc_c008 + player | 4534 | 4573 | 39 |
| v_ITo3sCnpw_k_c007 + football-player | 62 | 9780 | 9718 |
| v_ITo3sCnpw_k_c007 + player | 62 | 9780 | 9718 |
| v_ITo3sCnpw_k_c007 + soccer-player | 62 | 9780 | 9718 |
| v_ITo3sCnpw_k_c011 + player | 225 | 9736 | 9511 |
| v_ITo3sCnpw_k_c011 + soccer-player | 225 | 9736 | 9511 |
| v_ITo3sCnpw_k_c012 + player | 359 | 10832 | 10473 |
| v_i2_L4qquVg0_c007 + player | 4619 | 4828 | 209 |
| 合计 | 10148 | 69045 | 58897 |

只读差异分析证明：8个旧模板均为原始标注重建结果的严格子集；共有(frame,id)的全部字段相同，非舍入、框坐标、帧偏移或行顺序差异。例如c007的三个通用表达旧模板只有ID20的62行，原始targets包含ID0–20。此证据说明两份官方资产的行收录不一致，不能据此无证据断言具体上游代码错误。

原始JSON逐字节与LaMOT官方GitHub版本一致。全部输入SHA和首个差异见[sports_GT_completion.json](sports_GT_completion.json)；完整八条的ID/帧覆盖及集合比较见[sports_GT_mismatch_analysis.json](sports_GT_mismatch_analysis.json)、[CSV](sports_GT_mismatch_analysis.csv)；实际失败stdout/成本见[sports_GT_completion.log](sports_GT_completion.log)。不将这次失败包装为通过。

用户随后明确授权：“另行预注册：用官方原始标注统一重建全部466条GT，保留旧模板不覆盖”。执行PREREG v4：全部466条采用同一原始targets×原框规则，另存新目录，保留旧418和v3失败记录。新GT在本修订推送后才生成；SportsMOT所有后续指标明确标记PREREG_V4_ORIGINAL_ANNOTATION_GT，不冒称旧模板协议的官方复现。

不修改任何模型、候选、分数、预测、阈值或六组合判定规则。没有SportsMOT HOTA被用于选择GT协议；其他四组合不重跑。
