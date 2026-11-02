# v5：RankIoU 式的候选集合贝叶斯决策

2026-11-02，延续用户授权的构造探索。v4已完整封存：KITTI42.593 / Dance28.358 / Sports69.902，Dance未过原门槛；不覆盖或改写v4。

## 文献、问题与固定变化

[Dai & Li, RankSEG, JMLR 2023, Lemma 13](https://jmlr.org/papers/volume24/22-0712/22-0712.pdf)研究概率排序和预期IoU的集合决策。这里将元素从像素换成固定MOT候选；不宣称其分割理论保证RMOT HOTA。

v4的数量后验众数对应“猜对整个数量”的0/1损失，不是集合重叠损失；Dance有目标帧召回已保持，但帧内仍少选。本轮不重新拟合概率、不搜索温度，只检验改变决策风险是否改善。

- 精确复用v4源Train拟合的三个checkpoint及SHA、正/负斜率，源/目标数据、所有候选框/ID/帧号/排序全部不变。无新训练、无GPU/MLLM调用。
- 校准概率p_i仍与源margin严格同序。S_k为原margin排序前k个候选，k=0..n。
- 空集效用U(0)=prod_i(1-p_i)，约定预测与真实均空时集合IoU=1。
- k>=1时，F_out表示S_k以外的正例数，按同一个独立伯努利后验精确求分布：
  U(k)=[sum_{i<=k} p_i] * sum_{j=0}^{n-k} P(F_out=j)/(k+j)。
- 选择argmax_k U(k)，并列取最小k；没有额外阈值、Native数量下界或失败回退。保存U(k)供审计；概率输出仍是原v4多伯努利后验，不能把U当成概率。平滑对照窗口不变，不选其过关。
- 这只是每帧候选集合IoU代理，不是HOTA优化：匹配、轨迹关联、候选相关性和跨域失准均可能导致失败。

## 执行与门槛

预注册先commit/push；小集合完全枚举核查U、padding/空集/排序tie及无Native硬下界。随后封存复用权重清单，三个完整同域GT-free预测先SHA封存再官方TrackEval，Sports v4全466与原418ID敏感性。

原三域PRES_TOPK HOTA>=完整Native−1门槛不变，不按HOTA回选v4或未校准概率。通过后完成九格A并发布，再实际C1/C3教师/TTA pilot和40GPU小时关卡；失败如实封存，不把代理效用当真实指标。所有新记录仅写redesign_v5，旧记录不改。没有使用跨域GT调任何参数。
