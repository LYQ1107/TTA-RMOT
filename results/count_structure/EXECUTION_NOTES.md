# 失败与修复记录

1. 源Platt校准首次尝试在拟合前因`KeyError:1`停止：实现错误地把官方源预测轨迹ID当作源GT ID。检查旧正式运行config与vendor测试路径后，确认候选是Temp-NeuralSORT car/pedestrian输出，pedestrian须加max(car ID)，原tracker帧号须减1。
   最小修复：复用已有merge_kitti_class_splits，用真实预测MOT几何与源query GT做IoU匹配；不造框、不借GT ID替代预测ID、不丢弃缺失样本。新增几何/帧偏移/ID offset真实格式的针对性回归。该次尚未开始Platt拟合，学生/教师更新0、Dance GT打开0；原失败源码保留在首个准备提交中。

教师单卡真实smoke与完整探针由独立父命令运行，不受上述CPU校准失败影响；健康调用保留缓存，不重跑。所有最终状态以后续receipt为准。

修复后源候选2355528行全部对齐，862个query，实际匹配正例354658，Platt迭代14次收敛；没有用预测正例率750834/2355528充当GT先验pi_s。

2. 教师真实结果：T3有9/850个原始输出不符合预注册解析规则，全部留在原始缓存、输出与UNKNOWN指标中；这属于教师行为诊断，不据此改prompt/解析器重跑。完整三问法与smoke均成功退出，冻结权重SHA一致，GT打开0。未找到可核验原版8B，按用户规则不下载、不用其它项目的微调权重替代。
3. C4在T2V_D1、T2V_D3、SAME_SET_D1上全空；其它选择器仍照常完成425表达，不人为补正例，也不根据结果修改全局平移。
