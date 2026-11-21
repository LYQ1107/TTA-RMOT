"""Readable research history and acceptance evidence; never closes the goal."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.a100_visual_classical_sparse_observation_report import describe

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'


def load(stem):
    return json.loads((PUBLIC/(stem+'.json')).read_text())


def summary(lines,title,row,contrasts):
    lines += ['',title,'三神经种子平均HOTA：']
    for name,value in row['mean_HOTA'].items():lines.append('  '+name+' = '+f'{value:.6f}')
    lines.append('对应效应（完整结果和全部失败均见JSON）：')
    for name in contrasts:
        if name in row['comparisons']:lines.append('  '+name+'：'+describe(row['comparisons'][name]))


def main():
    strict=load('strict_rgb_analysis');physical=load('minimal_final_analysis');null=load('null_set_analysis');colour=load('zero_colour_analysis')
    assert strict['status']=='STRICT_RGB1080_PARITY_ANALYZED'
    assert physical['status']=='PROVISIONAL_MINIMAL_FRESH_EVIDENCE_ANALYZED'
    for stem in ('strict_rgb','minimal_final','query_scope','context_cohort','budget_cold','sparse_observation','null_set','zero_colour'):
        assert load(stem+'_statistical_reproduction')['status'].startswith('PASS_')
    sparse=load('sparse_observation_analysis');context=load('context_cohort_analysis');scope=load('query_scope_analysis')
    lines=['TTA-RMOT：人类视觉启发与经典算法的持续研究综合实证',
        '2026-11-22；研究根目录 /data3/liuyeqiang/FlexHook-TTA','',
        '本文汇总已完成的实证及其适用范围，不自动将科研Goal标记完成。',
        '后续Goal状态以线程Goal工具为准；所有数字均来自封存收据及官方TrackEval，不承诺必然正结果。','',
        '一、始终没有改变的最终目标',
        '跨域RMOT：在与源训练场景不同的目标视频中，依据语言描述选出对应的人或多名人的轨迹。',
        '用户明确接受官方固定候选框与Track ID，把研究范围限定在跨域指代适配模块。',
        '适配使用目标图像、任务语言、冻结源模型输出、冻结教师反馈；不读取目标人工标签。',
        '源主干与Qwen教师冻结，允许用教师伪标签训练一个小残差头。不是宣称所有权重都不更新。',
        'GT只能在预测封存后用于独立评估/诊断；不能依据GT自动选题、拟合、选阈值或挑checkpoint。',
        '所有现有数据都参与过开发，用户目前没有未接触的新划分，所以不存在独立盲测结论。',
        'C4为Dance-best→SportsMOT：45剪辑、466查询、13个原视频；C1为KITTI-V2→Dance：25视频、425查询。',
        '官方候选的上游监督来源另行披露，既不重建前端，也不把前端性能记作我们的贡献。',
        '当前视频内是转导推断，按视频前缀使用过去/当前反馈；三次观察包含当前视频未来帧，不叫逐帧在线。','',
        '二、为什么最初的快慢叙事发生了修正',
        'A800/NAS恢复以后，A100真实一图一句跑通了学生PCD、Qwen三视角27次真实调用/0缓存和快校准。',
        '旧SLOW_REPLAY虽固定问答，仍同时包含LoRA更新、固定32帧重算与重新快拟合。它的整体改善不能全归LoRA。',
        '匹配重算后：MINUS48.847077、RECALC_ONLY50.787438、SLOW_REPLAY52.151672。',
        'LoRA相对匹配重算+1.364233，但原先区间[-2.306727,4.863342]跨0；后续原视频聚类也没有建立独立正贡献。',
        '固定RECALC数量时神经margin排序差-0.609531；固定LoRA数量差-0.084175且跨0。',
        '这些是本配置的归因限制，不证明所有LoRA或所有神经适配永远无效。',
        '旧600秒冷预算下三策略均没有提交真实教师调用，冷启动已耗尽窗口；不能称主动策略全预算最省时。',
        '完整C4旧主动47.478与均匀40.833及其耗时有条件差异，也不能把它们拼接成新KD/图的增量。','',
        '三、先做错误诊断，为什么没有继续盲加HMM',
        'C4候选可恢复约98.193%的GT观测，但旧系统119/466查询未问到正确轨迹；122条有目标零输出中82条属于漏查。',
        '一句至少问一次不足以覆盖RMOT的多名目标；覆盖应在查询—候选层面检查。',
        '诊断Oracle约80.224只是使用GT的参照，既不可部署，也不是保证语义模块可达到的严格上界。',
        '已有最终v8P稳定片段输出翻转0，包含整轨迹传播；它也可能稳定地传播错误。不能由此推断动态语义永远不变。',
        '无教师RAW加EMA仅约+0.275 HOTA且区间跨0；没有依据把复杂时序模块列为首要修复。',
        '强制所有查询清单式判空会错拒真实目标，任务契约必须明确。',
        '旧DS完整系统对固定可靠性约+8.798，但点概率质量并非各域一致，多视角路由也没有独立稳定收益。',
        '新保留方法没有强行继承全部历史模块，旧DS大差更不能当作新图相对强KD的涨幅。','',
        '四、普通蒸馏为什么突然这么高，是否合理',
        '所谓普通KD不是逐框直接抄教师结果，而是充分训练的小头，使用同样轨迹结构、颜色和当前视频教师反馈。',
        'C4同解码SOURCE24.578、直接教师回填44.570、仅当前视频KD68.341、过去+当前KD69.050。',
        'C1对应10.804、23.599、39.040、39.030。主要高分来自当前教师监督与轨迹传播，不是历史反馈泄漏。',
        '两个方向重新拟合前缀KD均0个查询文件变化；C4各466查询独立绕过缓存官评，KD/图分数差均0。',
        'SOURCE同解码分数不是未修改的官方学生原生成绩；直接回填也不是完整发表方法的复现。',
        '标准稀疏关键帧和沿ID传播本就应给予合理强基线，不能拿掉这些能力来制造我们的优势。',
        '强KD：1080→64 GELU→1残差MLP，AdamW0.001/weight_decay0.001，batch512。',
        '至少4000步、最多16000步，按教师KL<=0.005停止；保留400步、充分最终和400/最终概率均值。',
        '教师软概率裁剪[0.02,0.98]，并保留硬伪标签充分训练；CLIP表示概率裁剪，不是CLIP视觉模型。',
        '同一输入/问答下新模块必须击败这些强对照，不能只击败没训练充分或没轨迹传播的弱教师。','',
        '五、真实尝试过什么，哪些失败被保留下来',
        '5%/10%/20%同预算和30%三观察对齐：16条件、176原始加228数量反事实，404官评全部完成。',
        '10%历史固定问答，C4均值KD66.275653→衣着图66.883168，+0.607515，原视频+种子区间跨0。',
        'C1为34.305499→36.815919，+2.510420，区间[0.743403,4.768982]。',
        '5%C4图不及硬KD400步，不能讲预算越稀疏图越好的单调故事。',
        '将三次观察分别用anchor训练，C4/C1均低于匹配的聚合KD；完整观察图也不及聚合图。',
        '外观衰减时序不及常数连接，未作为有效模块保留。物理ID混杂不等于衣着语义反复变化。',
        '随后1088维场景位置研究96官评：整体改善存在，但8位置边的独立增量和位置输入本身未稳定成立。',
        '仅多8个零维的MASKED头已对原头+0.386791，暴露维度/初始化混杂，不能说零维学到空间。',
        '同头查询范围105官评：不改头权重而扩展空间/角色图，C4原均值66.275653→67.227724，+0.952070。',
        '扩展范围自身+0.344556，在五个已有头上都正；C1原查询已在衣着范围，扩展严格等同原图。',
        '但原扩展图额外使用空间/角色RGB，原KD头仍遮挡。此前1088输入控制又混入位置和初始化。',
        '因此上述证据仍不能单独完成最强同信息基线验收，才有本次严格RGB1080研究。','',
        '六、这次严格相同RGB与新鲜源流的实证',
        '维度固定1080，初始化相同，普通KD与图都拿到相同查询解析RGB，没有8位置维度。',
        'C4只用这次真实独立KD流，1160条新教师答案；C1只用真实V2→Dance流，1103条。',
        '新复核不再产生教师或主干前向；每头的拟合时间是真实测量，但这不是又一次冷启动。',
        '种子0/17/42共享同一实际问答，只衡量神经训练敏感性，不能称三次独立教师/域重复。',
        '81原始加33数量输出全部封存，之后114官评；seed0逐决策复现实际物理KD，C1图也0变化。']
    for pair in ('C4','C1'):
        row=strict['repeated']['STRICT_RGB_'+pair]
        summary(lines,'严格RGB / '+pair,row,[name for name in row['comparisons'] if name.startswith('RGB_GRAPH_MEAN_vs_') or name.startswith('COUNT_') or name in ('RGB_KD_MEAN_vs_ORIG_KD_MEAN','ORIG_GRAPH_MEAN_vs_RGB_KD_MEAN')])
        e=row['comparisons']['RGB_GRAPH_MEAN_vs_RGB_KD_MEAN']
        lo,hi=e['parent_and_neural_seed_sensitivity_CI95']
        lines.append('主对照判读：'+('该开发方向的原视频+种子区间为正。' if lo>0 else '区间含0或负值，尚不能宣布稳定超过公平普通KD。'))
    for pair in ('C4','C1'):
        summary(lines,'显式手工颜色通道归因 / '+pair,colour['repeated']['ZERO_COLOUR_'+pair],[
            'RGB_MEAN_vs_ZERO_MEAN','RGB_400_vs_ZERO_400','RGB_FINAL_vs_ZERO_FINAL',
            'COUNT_OWN_RGB_MEAN_vs_ZERO_MEAN','COUNT_BASELINE_RGB_MEAN_vs_ZERO_MEAN'])
    lines += ['这只检验同1080维显式HSV通道，不是灰度PCD或自然768维架构，也不改变原共同RGB选题。',
        '颜色归因不能替换完整RGB软/硬强KD主比较，图相对强KD的额外贡献仍单独报告。']
    summary(lines,'图压空的协议保护复核',null['repeated']['NULL_SET_C4'],[
        'SAFE_MEAN_vs_KD_MEAN_PRESENCE','SAFE_MEAN_vs_GRAPH_MEAN_PRESENCE',
        'SAFE_MEAN_vs_KD_400_PRESENCE','SAFE_MEAN_vs_KD_FINAL_PRESENCE',
        'SAFE_MEAN_vs_KD_HARD400_PRESENCE','SAFE_MEAN_vs_KD_HARD_FINAL_PRESENCE',
        'COUNT_OWN_SAFE_MEAN_vs_KD_MEAN_PRESENCE','COUNT_BASELINE_SAFE_MEAN_vs_KD_MEAN_PRESENCE'])
    lines += ['新鲜原头图曾让正查询空输出19→61，救回4却损失23条正确检索。该副作用是继续研究的理由。',
        '保护规则仅在图全空且显式任务保证存在时保留KD完整集合；所有强KD同样得到非空协议控制。',
        '不新增教师/主干/头训练；辅助硬基线得到额外软reference头，不称同单头成本。',
        '这与旧单例回填/宽泛no-Yes保护有区别，但必须靠下列结果判断，不能预设有效。',
        json.dumps(null['query_retrieval_diagnosis'],ensure_ascii=False,indent=2),
        '强制非空是构造性质，真实找对目标和FP/FN才说明问题是否改善；模块可能是数量修复，不能冒充新语义学习。']
    lines += ['', '七、保留候选到底如何把视觉启发变成经典计算',
        '先扫视再类比：每条固定轨迹保留三次真实观察，解析查询所需身体部位，形成源PCD与HSV外观描述。',
        'COVER高/中/低分层扫描和多样性帮助降低只看高分候选的风险，但所有KD/图共享，尚非独立最优选题贡献。',
        '图传播的是是否符合这句描述，不把外观相似的不同人合并成同一物理身份。',
        '对称自适应尺度3近邻，度归一化；缺少所需可见颜色部位的节点不强连。',
        '求解二次能量：sum_i a_i(z_i-z_KD_i)^2 + sum_i b_i(z_i-z_T_i)^2 + 1/2 sum_ij W_ij(z_i-z_j)^2。',
        '已问节点a_i=0以避免教师标签和拟合该标签的KD被误当作两份独立证据；未问节点a_i=0.1。',
        'b_i=1+3|2p_i-1|，logit边界4。这些是固定强度常数，不声称它们是由GT校准的真实可靠性。',
        '去图仍保留教师锚点；同头、同信息对照才隔离图的独立作用。',
        '最终是否保留RGB扩展头或原头，必须同时面对全部强KD，而不是按域挑胜者或只报最高checkpoint。',
        '图贡献若主要是数量校准就按数量校准报告；固定数量收益才支持额外排序判断。',
        'C1完全图与稀疏图没有稳定差，不能将C4视觉拓扑特异性推广到全部目标域。','',
        '八、这次实际物理成本与精度，禁止拿旧精度配新时间']
    for label,row in physical['conditions'].items():
        lines += [label,'实际本次HOTA：'+json.dumps(row['raw_HOTA']),'本次差：'+describe(row['comparisons']['OURS_vs_KD'])]
    for label,row in physical['runs'].items():
        lines += [label,'外部进程秒：'+str(row['worker']['actual_worker_process_wall_seconds']),
            '相位秒：'+json.dumps(row['phase_seconds']),
            '组件分账：'+json.dumps(row['cost_by_method']),
            '真实教师收据：'+json.dumps(row['teacher']),
            '拟合收据：'+json.dumps(row['teacher_fit_audit']),
            'decoder观察帧='+str(row['decoder_frames'])+'；实际backbone图像前向='+str(row['backbone_image_forwards'])]
    lines += ['独立C4输入/教师一致性：'+json.dumps(physical['C4_independent_input_teacher_audit']),
        'C4是两个独立单方法进程，包含导入、元数据、学生/教师加载、特征、真实Qwen、头拟合和输出。',
        '并行A100共享NAS/OS及后台负载，进程wall差不能直接当图的算法提速。',
        'C1是同一真实流的两分支，common+own是实测组件和，不是两次独立冷进程。',
        '上游官方候选检测/跟踪生成在输入边界之外，没有免费抹掉未测成本。',
        '新鲜C1与历史池先验最大差0.603、问题对称差458，所以不能拿历史36.816配本次wall。',
        '早期全帧到三观察把学生相位7686.8秒降到1858.4秒；PCD/RGB在匹配问答中相同，但先验/门控有变化，非严格无损。',
        '这属于执行调度节省，不称LoRA或神经特征学习收益，也不是所有GPU利用率的积分。','',
        '九、GT仅在封存后做的错误回看',json.dumps(physical['postseal_candidate_membership_diagnosis'],ensure_ascii=False,indent=2),
        '这些是候选IoU成员计数，不是HOTA损失的可加分账；有正候选不代表原始全部GT都可恢复。','',
        '十、仍需准确回答的科学问题与结论边界',
        '严格同RGB主对照是否在两个方向都支持正增益，是否胜过所有充分软/硬checkpoint，详见上面的全部实际结果。',
        '即使均值提升，也应检查三个种子的方向、按原视频不确定性和双向数量投影，不凭某个单点封神。',
        '开发数据上的描述性区间没有修正多次开发或多重比较，也不能证明未见域总体性能。',
        '没有固定全预算Pareto最优结论，没有所有视觉直觉都有效、所有LoRA都无效的结论。',
        '不借用Oracle、固定坏图的大差或直接回填弱基线，冒充对强普通KD的独立增益。',
        '如果严格同信息对照消除了旧优势，研究必须继续，不能弱化基线或以写完报告当作完成。','',
        '十一、完整证据索引（仓库仅存精选统计，原始结果不上传）']
    for stem in ('benchmark_scope_audit','sparse_observation_analysis','context_cohort_analysis','query_scope_analysis','budget_cold_analysis','minimal_final_analysis','strict_rgb_analysis','null_set_analysis','zero_colour_analysis'):
        lines.append(str(PUBLIC/(stem+'.json')))
    lines += [str(PUBLIC/'strict_rgb_parent_statistics.json'),str(PUBLIC/'strict_rgb_statistical_reproduction.json'),
        '公开充分统计可在无GPU/无原始GT条件下用NumPy复算HOTA和效应；算术复现不是独立实验。',
        '原始新运行：/data3/liuyeqiang/FlexHook-TTA/a100_runs/visual_classical_20261121',
        '大型历史资产只读：/data3/liuyeqiang/nas/A800；新产物不覆盖NAS历史results。',
        'GitHub仅代码/实验TXT/精选JSON与PNG/PDF；snapshot和可达历史均必须小于100,000,000字节。',
        'Goal由根代理核验真实目标后单独判断；此报告不会自动结项。']
    path=ROOT/'docs/TTA_RMOT_视觉本能与经典算法_持续研究综合实证_20261122.txt'
    path.write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='RESEARCH_SYNTHESIS_WRITTEN',path=str(path),goal_auto_closed=False),ensure_ascii=False))


if __name__=='__main__':main()
