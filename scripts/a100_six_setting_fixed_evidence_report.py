"""Summarize complete fixed-evidence controls without promoting a new method."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'docs/a100_six_direction_20261122'
DEST=ROOT/'docs/TTA_RMOT_六个跨域设置_多视角证据统筹诊断_20261122.txt'
SETTINGS={'C1':'V2→Dance','C2':'Dance→KITTI-V2','C3':'V2→SportsMOT',
          'C4':'Dance→SportsMOT','C5':'LaMOT→KITTI-V2','C6':'LaMOT→Dance'}


def interval(row):
    ci=row.get('parent_and_neural_seed_sensitivity_CI95',row.get('CI95'))
    return '不报告总体CI：只有4个原始视频' if ci is None else '['+', '.join(f'{x:.6f}' for x in ci)+']'


def main():
    studies={stem:json.loads((PUBLIC/(stem+'_analysis.json')).read_text()) for stem in
        ('view_fusion','view_fusion_flat_rgb','view_fusion_direct_fill','lame_reference')}
    receipts={stem:json.loads((PUBLIC/(stem+'_statistical_reproduction.json')).read_text()) for stem in studies}
    assert all(r['status'].startswith('PASS_') for r in receipts.values())
    core=studies['view_fusion'];rgb=studies['view_fusion_flat_rgb'];direct=studies['view_fusion_direct_fill'];lame=studies['lame_reference']
    lines=['TTA-RMOT：六个跨域实验设置——完整固定证据与强对照诊断',
        '2026-11-22；全为开发数据，Goal仍ACTIVE；当前组合不是已完成的三创新方案','',
        '最关键结论：固定证据组合在两个SportsMOT设置有实测收益，但在两个KITTI设置落后于普通教师头；Dance上的优势也经不起所有强对照与数量条件。不能把DS、外观规则和查询约束叠起来就宣布跨域问题解决。',
        '学习经典思想应体现在任务建模、可辨识性与机制设计；经典算法名、优化唯一解或收敛检查不是论文创新本身。','',
        '一、比较范围与公平边界',
        '四项研究各自全部输出封存后才读取本项GT，适配不读取人工目标标签。固定官方候选框/TrackID；McByte前端有SportsMOT预训练，不声称完整前端source-only。',
        '这四组共有810份已官评的方法条件输出；统计档包含共享对照，共900条HOTA记录，不是900个独立试验。每设置3个神经种子0/17/42；SportsMOT按13个原始视频组重采样，不把45切片当45独立视频；Dance25组；KITTI只有4组，不给总体置信区间。',
        '本组使用相同旧实际多视角QA、相同候选和新冻结特征流；旧问答帧不一定与新三帧特征锚点相同，所有方法共享这一限制。不能将它与此前新CROP/10%流的差值，单独归因于更多问答或多视角。',
        'RAW_FINAL/RAW_MEAN/HARD_FINAL/RAW_SRC_FINAL、同1081D信息RAW_RGB_FINAL、教师直接回填、两个LAME移植对照全部保留。LAME有原实现与真实数据核验，但这是RMOT移植，不声称原论文在此基准有同样设定。',
        '下文“普通对照最高值”只是开发结果的描述性上包络，不能当作按GT选择的部署策略；主方法仍为预登记DS_FULL，不晋级更高的RAW_FULL或DS_CONTEXT。','',
        '二、三种子平均HOTA（显示值取3位，完整值保存在JSON）',
        '设置 | 普通KD/直接教师对照最高已观察值 | RAW-KD+LAME | DS_FULL | FULL减普通最高 | FULL减LAME']
    best={}
    for pair,title in SETTINGS.items():
        h=core['repeated'][pair]['mean_HOTA']
        ordinary={k:h[k] for k in ('RAW_FINAL','RAW_MEAN','HARD_FINAL','RAW_SRC_FINAL')}
        ordinary.update(RAW_RGB_FINAL=rgb['repeated'][pair]['mean_HOTA']['RAW_RGB_FINAL'],
                        RAW_TEACHER_FILL=direct['repeated'][pair]['mean_HOTA']['RAW_TEACHER_FILL'])
        name=max(ordinary,key=ordinary.get);value=ordinary[name];best[pair]=(name,value)
        lm=lame['repeated'][pair]['mean_HOTA']['RAW_LAME_KNN5']
        lines.append(f"{title} | {value:.3f} ({name}) | {lm:.3f} | {h['DS_FULL']:.3f} | {h['DS_FULL']-value:+.3f} | {h['DS_FULL']-lm:+.3f}")
    lines+=['','三、区间与模块的条件贡献','下列CI为开发数据原始视频和神经种子敏感性区间，不是独立盲测证明。']
    names=('LEARNED_vs_FIXED_RELIABILITY','LEARNED_vs_RAW_PROBABILITY','VISUAL_CONDITIONAL','QUERY_CONSISTENCY_CONDITIONAL')
    for pair,title in SETTINGS.items():
        row=core['repeated'][pair];lines.append('\n'+title)
        for name in names:
            effect=row['comparisons'][name]
            lines.append(f"  {name}: {effect['delta_HOTA']:+.6f}, CI {interval(effect)}")
        effect=lame['repeated'][pair]['comparisons']['FULL_vs_ORDINARY_KD_LAME']
        lines.append(f"  FULL_vs_ORDINARY_KD_LAME: {effect['delta_HOTA']:+.6f}, CI {interval(effect)}")
    lines+=['',
        'C1/C6的DS等于固定可靠性回退；这些输入只有CROP，不能据零差声称学会了视角可靠性。SportsMOT学得可靠性相对固定0.8只有约0.2–0.3点且区间跨零；KITTI相对固定与原概率更差，不支持DS普适有效。',
        'C1的外观条件效应为-2.487797，区间[-5.026075,-0.294214]；C3/C4分别约+0.701/+1.005。外观规则的前提在不同数据中未必成立，不能按GT在每个设置手工开关后宣称统一方法。',
        'C4查询一致性的条件效应+0.819066且区间为正；C3约+0.233095但本组区间跨零，其它设置也未普遍证实。该关系图依赖语言范围假设，不是语义GT。','',
        '四、数量控制揭示的排序边界',
        'FULL相对RAW-KD+LAME，在FULL输出数量与LAME输出数量两个操作点上分别比较：']
    for pair,title in SETTINGS.items():
        row=lame['repeated'][pair]['comparisons']
        a=row['RANK_AT_FULL_COUNT_vs_RAW_LAME_KNN5'];b=row['RANK_AT_BASE_COUNT_vs_RAW_LAME_KNN5']
        lines.append(f"{title}: FULL数量 {a['delta_HOTA']:+.6f}, CI {interval(a)}; LAME数量 {b['delta_HOTA']:+.6f}, CI {interval(b)}")
    lines+=['',
        '两个SportsMOT设置在FULL数量下排序优于LAME，在LAME数量下却更差，且两种差异的区间方向相反。排序效用随输出数量变化，不能说语义排序全面支配对手；原始HOTA收益既包含数量选择，也包含特定操作点上的排序变化。',
        'Dance相对强LAME的原始差值和两种数量下的差值均缺少稳定正区间。两个KITTI对原普通头仍落后，即使对较弱的LAME偶尔为正，也不能据此包装成已经获胜。','',
        '五、已实施的下一步，而不是预先宣布成功',
        '对象参照学习：在教师监督进入头训练之前移除状态项，学习统一参照位置的对象判断，再恢复当前位置；与未分解目标、时间均值、置乱、同回退普通头及双向数量控制共同评测。当前左右门控只适用两个KITTI设置，其它四项精确保留DS_FULL；它本身不是全六设置的完整解法。',
        '普通头训练充分性：首C2前缀16000步仍KL0.0063925；无精确输入碰撞。额外输出层优化仍为0.0050518，继续同容量全头优化后达到0.00410485。原0.005门槛不放宽。',
        '参照450与同容量普通补充162已完成，详情见TTA_RMOT_对象关系学习与普通蒸馏充分性_20261122.txt。C2仍有5个前缀拟合未达门槛，不能称全部充分；teacher-only容量诊断已使最难前缀达到0.0049274，完整容量补充正在独立执行，旧主方法不变。不能用最终视频头修复过去视频，也不能用单个成功前缀替代完整比较。',
        '后续要从任务描述与观察可辨识性检查静态对象属性、当前关系及不同视角的证据范围。不能把完整描述的No自动拆成每个属性都No，也不能未经诊断就增加时序网络。全新的主动观察策略只有清单与小规模真实输入验证，尚无其全六设置效用证明。','',
        '六、成本与复现',
        '这些是固定问答、预付特征的质量回放，0新增教师前向不等于0部署成本。CPU优化诊断的几秒也不包括冷启动、特征提取与原始拟合。最终权重/配方确定后仍需配套真实端到端执行与精度-时间曲线。',
        '原始统计文件在本机保留可读备份；Git中的新统计只作无损float64编码和相同父视频行复用。900条记录全部值与数值类型保持；压缩后再次复算全部210项配对比较，最大误差不超过2.85e-14。']
    for stem in studies:
        lines.append('OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/a100_six_setting_statistics_reproduce.py --stem '+stem)
    lines+=['',
        '所有代码/收据在docs/a100_six_direction_20261122，源代码与原始输出在a100_runs/visual_classical_20261121/six_direction_20261122。原始QA、图片、模型和预测不进入Git。',
        '目前没有证明三项创新均成立、没有实现六设置一致可靠超过全部强普通对手，也没有完成对应的新部署成本验收。Goal保持ACTIVE。']
    DEST.write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='COMPLETE_FIXED_EVIDENCE_REPORT_WRITTEN',path=str(DEST),
        HOTA_statistic_records=sum(r['HOTA_records'] for r in receipts.values()),
        paired_effect_records=sum(r['paired_effect_records'] for r in receipts.values()),research_goal_complete=False),ensure_ascii=False))


if __name__=='__main__':main()
