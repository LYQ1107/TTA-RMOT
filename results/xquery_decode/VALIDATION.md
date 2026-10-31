# 最终验收：2026-11-01

状态：PASS；研究结果按固定规则报告，不以工程PASS替代科学结论。

- CPU回归：`tests/test_xquery_decode.py tests/test_count_structure.py tests/test_diag_attrib_v2.py`，23项不同测试通过，最终一次0.53秒；9个新脚本compileall及git diff --check通过。
- 四套历史X0各425文件逐SHA复现旧C3，合计1700文件；六套输入共2550原margin文件和6套float64缓存在生成前核验。
- 源域5个学生各2355528有限分数，实际forward计数逐模型恰好相等，数组SHA及形状核验；源域初始化checkpoint SHA实测`71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613`，与pin相同。所有学生final-step8973，编码器逐tensor等于源模型，参数版本守卫通过。
- 全36套目标方法生成时目标GT打开尝试0；旧teacher三组合生成时0；源域worker为0；学生/教师更新0，新增Dance教师调用0。独立GT进程只在各预测完整封存后运行。
- 39套完整目标结果（32新+7严格等价复用），8套源域完整结果：共47套正式官方CSV、94份summary/detailed文件，归档条目与原文件SHA逐一相同；40份新TrackEval完整stdout归档。
- 23471个预测文件逐一重核SHA，且(domain,method,video,expression)无重复；预测/GT原始行不进日志归档。全部服务器大文件及日志索引SHA重核一致。
- 132个delta/CI在补充解释/报告打包前后保持不变，规范JSON SHA：`d50a0a726a2c4953236b1e4382b94d34381c6fc46a348eeed63e7fe06989dd2d`。报告SHA：`7b37fb523c77714444441f35870d230c1a72e339f872bc3e009a6f70d0df8187`，与completion.json一致。
- 历史count_structure/train2val/diag_attrib、已有flex_uda模型/训练/teacher实现、旧v2公共评测与配置相对000e196d无tracked差异；本轮没有编辑vendor代码或旧数据/checkpoint。新增代码与记录均位于本轮目录和脚本，research_log保留历史内容并追加本轮记录。
- 全部源域GPU子进程和CPU评测/报告进程均正常退出，无实际失败或数值fallback，不留下一轮任务。后处理链wall287.426秒，额外CPU报告刷新15.109秒（只加解释链接/历史teacher指标复制与归档，不重新推理/TrackEval）；GPU总1.934139 process-hours包含smoke和full。

初次完整报告SHA4edd3272...与最终报告的数值和132比较均相同；变化仅为补充解释链接，历史teacher对照CSV由已有评价原样复制。所有预注册方法与失败结果全报，默认X0/X1只是规则产物，不是启动下一实验的授权。
