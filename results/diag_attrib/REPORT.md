# D-FIT 事后归因诊断：协议自检失败，按指令停止

状态：`STOP_OFFICIAL_COMPLETEBANK_CANNOT_REPRODUCE_HISTORICAL_NATIVE`。
这不是任务全部完成，不是新的方法结果；矩阵、校准基线与bootstrap均未运行。

## 1. 结论摘要（Q1–Q4）

1. Q1：已有行数中D1约为历史官方1.3588倍，D2/D3分别约3.9207/2.7355倍，但官方旧分数不完整，不能把这张历史描述表当作本轮已对齐的正式表C或判定C验收。
2. Q2：完整的排序×数量归因未开始，不能新增数量/存在性与排序的归因结论；上一轮封存结论和数值不改。
3. Q3：B1–B4未开始，不能判断无学习校准能否达到D1；判定A未执行。
4. Q4：D3在合理数量下的可转化HOTA与视频CI未计算，判定B未执行。
5. 停止原因是同域官方遗漏692条MOT末观测；真实补分数新增792条正预测、影响56条序列，使254,118变为254,910行，与必须逐行复现历史官方输出的硬条件不相容。

## 2. 表C：仅保留已有结果，不冒充新对齐

下列HOTA直接读取旧官方summary；没有新GT评价。OFFICIAL_DANCE_HISTORICAL的分数覆盖仅210,393条MOT，其余模型采用完整211,085条候选。**新补齐官方HOTA未知，绝不沿用32.134。**

| 历史/已封存模型 | HOTA（旧结果） | 原生预测行数 | /历史官方254,118 | 覆盖状态 |
|---|---:|---:|---:|---|
| OFFICIAL_DANCE_HISTORICAL | 32.1340 | 254,118 | 1.0000 | INCOMPLETE_210393_MOT |
| FROZEN | 8.2180 | 99,704 | 0.3924 | COMPLETE_211085_MOT |
| D1 | 29.7020 | 345,302 | 1.3588 | COMPLETE_211085_MOT |
| D2 | 22.8440 | 996,330 | 3.9207 | COMPLETE_211085_MOT |
| D3 | 27.7680 | 695,148 | 2.7355 | COMPLETE_211085_MOT |
| D0_NAIVE | 7.4499 | 21,100 | 0.0830 | COMPLETE_211085_MOT |

完整原有DetA/AssA/DetRe/DetPr/AssRe/AssPr/IDF1/IDSW/CLEAR FP/FN/Dets/IDs等在alignment.json中；有输出帧/GT分组k未在停止后补做。历史行数比只供解释影响，不是已通过的匹配评测。

## 3. 5×6矩阵

HOTA矩阵、行数矩阵和同列差值矩阵均未启动；attrib_matrix.csv为30个明确NOT_RUN的格子，空值不是0。未把792条新增正例删去、压成负分或只对旧覆盖集合评分以强行通过。

## 4. 校准基线

B0和B1–B4全部未启动。baselines.csv列明8个GT-free版本的NOT_RUN状态，不把未启动进程的GT审计伪报成已经执行。

## 5. 视频bootstrap

未运行。bootstrap.json保留预注册25视频/2000次/seed0计划，实际comparisons为空；不使用旧区间冒充新比较，不执行判定A/B/C。

## 6. B0曲线

未生成：停止后不继续新的阈值评测或GT-informed曲线，亦不绘制虚构图。

## 7. 自检与失败影响

- FROZEN、D1、D2、D3各425表达、3,588,445行分数全部通过旧封存SHA核对；margin>0逐候选ID/坐标复现原生预测99,704 / 345,302 / 996,330 / 695,148行。官方旧输出按数值MOT字段规范化比较，原文件不改。
- 官方旧result.json SHA为de8ff46428818cbd1ee0dd2b6ea1a7692eae2b7e0133c521c1470b67b58da110；其中3,576,681条已覆盖分数能复现全部425条历史预测，但比完整要求少11,764条。
- 原因：vendor/FlexHook/data/mydataloader.py:354将sample_end设为obj_last−sample_frame_num，359行range的右端不包含，跳过每条轨迹末观测；没有修改vendor或历史结果。
- 仅使用官方Dance checkpoint，SHA fbb5df7339a1201f34300c105df81e893a5edbb8ef939b7f6eaf8780348eac6c；FP32，参数冻结，tokenizer person=[5970]。
- 8个真实已覆盖帧对照：缓存解码vs原始forward最大margin误差1.0491e−5；vs历史margin最大误差1.3257e−4；选择翻转0。新增正例有明确大正margin，不能用浮点误差解释。
- 25视频692条漏观测的11,764组双类logits完整补在新目录；其中792组margin>0，影响56条表达，完整模型按同规则必然254,910行。例：dancetrack0004 / dancing-person-dressed-all-in-white / frame138 / track4，margin=7.9885921478，历史没有该候选。
- 因行集合已经不等，完整官方HOTA不再运行；32.134只保留历史，不赋给254,910行版本。新官方margin也未供给矩阵。
- 每视频漏项与新增正例见per_video_protocol_impact.csv；完整候选明细见official_missing_observations.json；25个补分数文件路径和SHA见prediction_manifest.json及official_reproduction.json。
- 旧6个count_controls的数值不改，但本轮未启动相应矩阵等价性运行，不能标记为本轮已完成自检。
- 补分数进程退出码3是刻意触发的协议停止，不是OOM、loss发散或训练失败。3项必要单元检查通过；不以此代替科研验收。

## 8. GT隔离与旧结果完整性

CPU输入核验、单GPU补分数、停止归档均启用目标GT打开审计，实际尝试0。没有打开Dance表达原始注释、gt.txt或labels文件；只用已有白名单runtime文本manifest。未运行ORACLE或TrackEval，无新教师调用、无优化步。
停止后复核全部1,700对原生预测/margin、3个final checkpoint、原aggregate seal对应metadata和历史官方logits，均未改变。详见gt_audit.json、post_stop_integrity.json；大文件未放入Git。

## 9. 成本

CPU-only输入审计wall=51.533s；GPU 4补分数worker wall=186.155s（0.051710 GPU-process h），含准备总wall=196.431s；峰值allocated=3.977GiB。
入场时GPU4无计算进程、81,154MiB空闲，最多1张卡，没有停止他人任务。GPU-process小时不是独占GPU利用率小时；CPU-core小时未测，不伪造。另有少量归档/测试CPU成本，归档wall在post_stop_integrity.json。

## 10. 失败与修复记录

- 真实协议失败：历史官方分数遗漏候选，而任务同时要求完整库与逐行复现254,118；补分数证明二者不相容。**未通过改定义、删除正例、换权重或改阈值绕过。**
- 一次compileall命令误用不存在的裸python，shell报/bin/bash: python: command not found；随后用既定dfot绝对解释器重新compileall并通过。该调用不执行模型，后续补分数本来已用正确解释器；不是科研改进。原报错单独保留于logs/compile_attempt_1.txt。
- pytest三项通过、compileall通过、代码diff检查通过；生成CSV默认CRLF曾触发空白检查，已仅转LF且不改字段数值；原始日志尾随空格原样保留，不为格式检查改写证据。没有训练、没有完整GPU重推理、没有新的官方评测或参数更新。

## 建议（只建议，不执行）

请用户确认是否允许预注册v2：保留32.134/254,118为历史，接受同一checkpoint补齐后的254,910行官方原生输出，单独封存并重新计算其官方HOTA，再以该匹配官方数量列继续原定CPU矩阵与校准基线。需更新OFFICIAL_DANCE数量参照及复现门槛并先推送v2，不覆盖v1和历史结果；不能把新HOTA预先假定为32.134。
在明确确认前不运行任何剩余矩阵、阈值扫描、bootstrap、新学生或Train→Val实验。
