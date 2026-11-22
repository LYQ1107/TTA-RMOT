"""Readable research history and acceptance evidence; never closes the goal."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.a100_visual_classical_sparse_observation_report import describe

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'
FULL_DIAGNOSTICS=False


def diagnostic_record(stem,key,value):
    if FULL_DIAGNOSTICS:return json.dumps(value,ensure_ascii=False,indent=2)
    return '完整原始记录：'+str(PUBLIC/(stem+'.json'))+'，字段 '+key+'。对应专题TXT亦保留完整记录；本报告保留历程、全部HOTA/效应与结论。'


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
    canonical=None
    if (PUBLIC/'canonical_pcd_analysis.json').is_file() and (PUBLIC/'canonical_pcd_statistical_reproduction.json').is_file():
        canonical=load('canonical_pcd_analysis')
        assert load('canonical_pcd_statistical_reproduction')['status'].startswith('PASS_')
    semantic=None
    if (PUBLIC/'semantic_colour_analysis.json').is_file() and (PUBLIC/'semantic_colour_statistical_reproduction.json').is_file():
        semantic=load('semantic_colour_analysis')
        assert load('semantic_colour_statistical_reproduction')['status'].startswith('PASS_')
    joint=None
    if (PUBLIC/'joint_visual_analysis.json').is_file() and (PUBLIC/'joint_visual_statistical_reproduction.json').is_file():
        joint=load('joint_visual_analysis')
        assert load('joint_visual_statistical_reproduction')['status'].startswith('PASS_')
    appearance_cold=None
    if (PUBLIC/'appearance_cold_analysis.json').is_file() and (PUBLIC/'appearance_cold_statistical_reproduction.json').is_file():
        appearance_cold=load('appearance_cold_analysis')
        assert load('appearance_cold_statistical_reproduction')['status'].startswith('PASS_')
    direct=None
    if (PUBLIC/'direct_teacher_analysis.json').is_file() and (PUBLIC/'direct_teacher_statistical_reproduction.json').is_file():
        direct=load('direct_teacher_analysis')
        assert load('direct_teacher_statistical_reproduction')['status'].startswith('PASS_')
    direct_pcd=None
    if (PUBLIC/'direct_teacher_pcd_analysis.json').is_file() and (PUBLIC/'direct_teacher_pcd_statistical_reproduction.json').is_file():
        direct_pcd=load('direct_teacher_pcd_analysis')
        assert load('direct_teacher_pcd_statistical_reproduction')['status'].startswith('PASS_')
    native_factorial=None
    if (PUBLIC/'native_factorial_analysis.json').is_file() and (PUBLIC/'native_factorial_statistical_reproduction.json').is_file():
        native_factorial=load('native_factorial_analysis')
        assert load('native_factorial_statistical_reproduction')['status'].startswith('PASS_')
    native_cold=None
    if (PUBLIC/'native_cold_analysis.json').is_file() and (PUBLIC/'native_cold_statistical_reproduction.json').is_file():
        native_cold=load('native_cold_analysis')
        assert load('native_cold_statistical_reproduction')['status'].startswith('PASS_')
    predictive=None
    if (PUBLIC/'predictive_topology_analysis.json').is_file() and (PUBLIC/'predictive_topology_statistical_reproduction.json').is_file():
        predictive=load('predictive_topology_analysis')
        assert load('predictive_topology_statistical_reproduction')['status'].startswith('PASS_')
    stable=None
    if (PUBLIC/'stable_topology_analysis.json').is_file() and (PUBLIC/'stable_topology_statistical_reproduction.json').is_file():
        stable=load('stable_topology_analysis')
        assert load('stable_topology_statistical_reproduction')['status'].startswith('PASS_')
    weighted=None
    if (PUBLIC/'confidence_teacher_analysis.json').is_file() and (PUBLIC/'confidence_teacher_statistical_reproduction.json').is_file():
        weighted=load('confidence_teacher_analysis')
        assert load('confidence_teacher_statistical_reproduction')['status'].startswith('PASS_')
    contract=None
    if (PUBLIC/'contract_graph_analysis.json').is_file() and (PUBLIC/'contract_graph_statistical_reproduction.json').is_file():
        contract=load('contract_graph_analysis')
        assert load('contract_graph_statistical_reproduction')['status'].startswith('PASS_')
    weighted_classical=None
    if (PUBLIC/'weighted_classical_analysis.json').is_file() and (PUBLIC/'weighted_classical_statistical_reproduction.json').is_file():
        weighted_classical=load('weighted_classical_analysis')
        assert load('weighted_classical_statistical_reproduction')['status'].startswith('PASS_')
    lines=['TTA-RMOT：人类视觉启发与经典算法的持续研究综合实证',
        '2026-11-22；研究根目录 /data3/liuyeqiang/FlexHook-TTA','',
        '本文汇总已完成的实证及其适用范围，不自动将科研Goal标记完成。',
        '后续Goal状态以线程Goal工具为准；所有数字均来自封存收据及官方TrackEval，不承诺必然正结果。','',
        '当前方法状态：'+load('minimal_retained_method_card')['status'],
        ('最新共享收敛头方案：C4 67.151609→67.629222（+.477613），C1 32.721677→38.429592（+5.707914），三个种子均正，原视频+种子描述区间均正。'
            if (PUBLIC/'protocol_calibration_scientific_review.json').is_file() else ''),
        ('精度与既定模块门槛已过，四卡真实冷启动及教师回填/纯图边/视觉对应补查另行验收；尚不代表Goal全部完成，也非独立盲测。'
            if (PUBLIC/'protocol_calibration_scientific_review.json').is_file() else ''),
        '本文区分已完成的开发数据实证、真实执行成本和独立泛化；一组显著差值不意味着所有跨域RMOT错误已经解决。',
        ('已验证的残差配方：原RGB1080均值KD+视觉图+固定PAVA，C4 66.169708→67.692534（+1.522827），C1 32.199208→36.618032（+4.418824）。'
            if joint is not None else '联合主配方尚待完整官评和统计复现。'),
        ('三个神经种子的主差均正，两个方向的主描述性区间正。C1固定KD数量排序为负，仍是数量/查询基率校准，不作普遍排序突破宣称。'
            if joint is not None else ''),
        ('四卡独立真实冷启已完成，准确率与各自新墙钟详见后文。' if appearance_cold is not None else
            '四卡新独立冷启仍待完整收据验收，旧3512秒不能配新方法精度。'),
        ('RGB1080和原生PCD768两种无源残差直接教师头均已完成，完整公平性结果详见后文。' if direct is not None and direct_pcd is not None else
            '标准无源残差直接分类头的RGB1080/原生PCD768补查尚未全部完成，不能据残差家族结果泛称胜过所有一般蒸馏。'),
        ('强原生PCD普通均值C4为67.082937，旧残差组合对它+.609597的区间跨0；因此继续统一原生768直接头的机制与真实成本验收，不能在旧优势上结项。' if direct_pcd is not None else ''),
        '', '先用人话解释普通蒸馏为什么高，以及我们究竟要证明什么：',
        '强普通KD不是把教师问过的几个框直接复制出去。它用同样的少量教师回答训练共享小头，再预测没问过的查询-轨迹组合，并沿官方轨迹输出整段结果。',
        '最新强置信度加权普通头：均值C4 67.102356/C1 32.259906，充分收敛FINAL为67.151609/32.721677；后者两个方向都更高。只胜过均值不能自动算赢完整强基线家族。',
        '双方沿用相同官方候选，冻结源编码器本身也已有视觉语义能力；因此强KD高分有合理来源，不能为了突出新模块而限制它正常推广、充分训练或使用已有视觉特征。',
        '旧47.478、早期68/69和当前66/67来自不同预算、感知/先验与协议条件，不能把跨设置分数差全算成新模块增益。',
        '当前人类视觉启发落在两处：用相似衣着与视觉特征构建局部参照关系；把语言中的颜色与衣着部位绑定，利用教师反馈校准这种视觉线索。',
        '对应经典算法是自适应近邻图正则与单调回归PAVA。源主干和教师不更新；教师小头是共享的强基础，图与校准只有在同头消融中额外赢得的部分才能算模块贡献。',
        'C1数量/查询基率校准的改善也是任务收益，但不能改称神经元学会了更好的逐候选语义排序；C4长轨迹总体收益也不意味着每条查询都更少漏找。',
        '',
        '下面首先保留此前各次阶段判断，包括所有已经被后续研究修正的限制与失败。',
        '严格同RGB1080：C4图对充分均值KD +1.186539、对软400步 +0.807277，开发区间为正。',
        '但同架构HSV置零的C4普通KD更强，图对它仅+0.501837，区间[-0.292293,1.207896]跨0。',
        'C1图对均值KD +1.971637，原视频+神经种子区间[-0.069997,4.626153]跨0；固定KD数量排序差为负。',
        '强制非空/恢复KD集合让空查询变少，但SAFE对原图平均-0.258289，不能保留为已证实的有效贡献。',
        ('自然768维研究已完成，详见后文全部强对照；完成队列仍不等于完整Goal验收。'
            if canonical is not None else '继续自然768维强蒸馏与独立图度量/正信念保护研究；未完成前不宣布新模块有效。'),'',
        ('最新自然768主方案GRAPH_GUARD：C4比均值KD -0.460804，三个种子都负；比无保护图 -0.809855，区间负。该保护不保留。'
            if canonical is not None else '自然768主方案尚未完成，暂不作机制成功判断。'),
        ('C1无保护PCD图比768均值 +3.260991，但对更强RGB均值的区间仍跨0；CHECKLIST下保护严格no-op。'
            if canonical is not None else ''),
        ('下一轮颜色—衣着部位语义绑定与教师单调校准已完成，全部强对照详见后文。' if semantic is not None else
            '下一轮颜色—衣着部位语义绑定与教师单调校准正在进行，尚无本轮成功结论。'),'',
        '一、始终没有改变的最终目标',
        '跨域RMOT：在与源训练场景不同的目标视频中，依据语言描述选出对应的人或多名人的轨迹。',
        '用户明确接受官方固定候选框与Track ID，把研究范围限定在跨域指代适配模块。',
        '适配使用目标图像、任务语言、冻结源模型输出、冻结教师反馈；不读取目标人工标签。',
        '源主干与Qwen教师冻结，允许用教师伪标签训练共享小分类头；最新直接头没有固定源logit残差。不是宣称所有权重都不更新。',
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
    lines += ['这只检验同1080维显式HSV通道，不是灰度PCD或自然768维架构，也不改变原共同选题。',
        '代码复核澄清：共同COVER实际只用冻结原始PCD距离，不用额外手工HSV；PCD和教师仍来自RGB图像。',
        '早期元数据中“固定RGB选题”的宽泛表述不能理解为手工HSV曾参与此次选题；封存结果本身未改。',
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
        diagnostic_record('null_set_analysis','query_retrieval_diagnosis',null['query_retrieval_diagnosis']),
        '强制非空是构造性质，真实找对目标和FP/FN才说明问题是否改善；模块可能是数量修复，不能冒充新语义学习。']
    lines += ['', '七、候选方案如何把视觉启发变成经典计算',
        '先扫视再类比：每条固定轨迹保留三次真实观察，解析查询所需身体部位，形成源PCD与HSV外观描述。',
        'COVER高/中/低分层扫描和多样性帮助降低只看高分候选的风险，但所有KD/图共享，尚非独立最优选题贡献。',
        '图传播的是是否符合这句描述，不把外观相似的不同人合并成同一物理身份。',
        '对称自适应尺度3近邻，度归一化；缺少所需可见颜色部位的节点不强连。',
        '求解二次能量：sum_i a_i(z_i-z_KD_i)^2 + sum_i b_i(z_i-z_T_i)^2 + 1/2 sum_ij W_ij(z_i-z_j)^2。',
        '已问节点a_i=0以避免教师标签和拟合该标签的KD被误当作两份独立证据；未问节点a_i=0.1。',
        'b_i=1+3|2p_i-1|，logit边界4。这些是固定强度常数，不声称它们是由GT校准的真实可靠性。',
        '去图仍保留教师锚点；同头、同信息对照才隔离图的独立作用。',
        '最终是否保留RGB扩展头、原头或自然768维头，必须同时面对全部强KD，而不是按域挑胜者或只报最高checkpoint。',
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
    lines += ['独立C4输入/教师一致性：'+diagnostic_record('minimal_final_analysis','C4_independent_input_teacher_audit',physical['C4_independent_input_teacher_audit']),
        'C4是两个独立单方法进程，包含导入、元数据、学生/教师加载、特征、真实Qwen、头拟合和输出。',
        '并行A100共享NAS/OS及后台负载，进程wall差不能直接当图的算法提速。',
        'C1是同一真实流的两分支，common+own是实测组件和，不是两次独立冷进程。',
        '上游官方候选检测/跟踪生成在输入边界之外，没有免费抹掉未测成本。',
        '新鲜C1与历史池先验最大差0.603、问题对称差458，所以不能拿历史36.816配本次wall。',
        '早期全帧到三观察把学生相位7686.8秒降到1858.4秒；PCD/RGB在匹配问答中相同，但先验/门控有变化，非严格无损。',
        '这属于执行调度节省，不称LoRA或神经特征学习收益，也不是所有GPU利用率的积分。','',
        '九、GT仅在封存后做的错误回看',diagnostic_record('minimal_final_analysis','postseal_candidate_membership_diagnosis',physical['postseal_candidate_membership_diagnosis']),
        '这些是候选IoU成员计数，不是HOTA损失的可加分账；有正候选不代表原始全部GT都可恢复。']
    if (PUBLIC/'fresh_teacher_anchor_diagnosis.json').is_file():
        teacher=load('fresh_teacher_anchor_diagnosis')
        lines += ['', '九补、实际被询问锚点的证据质量',
            '该诊断在自然768研究全部156输出封存后执行，不改变本轮任何适配预测或参数。',
            'Yes以p>=0.5计，沿用硬KD标签约定；0.5并不意味着肯定、可靠的证据。',
            'GT比较的是所问候选在锚帧是否匹配该查询目标；不是纯教师语义准确率，也不代表全部未问候选。',
            'C4全部466句都问过，但133句未问到GT正锚点；76句无Yes，其中59句没问到正锚点、17句问到了却仍无Yes。',
            '此口径为当前10%实际问答的锚点匹配，与旧119条整轨迹漏查口径/预算不同，不直接作升降比较。',
            'C1有31句完全未问，但其中仅1句有可恢复目标；不能盲目给所有未问或无Yes查询加预算。',
            'C1共40条有可恢复目标的查询，10条无Yes；352条无Yes中其余342条并无正候选。',
            'C4教师Yes在已问锚点的正匹配比例71.616%；C1仅26.087%。C1整体89.755%一致率受大量负例主导。',
            'C1概率>=0.9的88个样本也只有30个正匹配，34.091%；仅调高置信阈值不能保证正确。',
            'C1错误Yes常见衣着类别混淆，C4常见左右/深浅队伍；框定位、邻人入裁剪及可观察性也是混杂。',
            '已查看的原图只作定性诊断，不据少量样例声称这些错误的全局原因。',
            '这些GT诊断精确率不能作为适配时的教师权重；后续方案仍须只用图像、语言和教师反馈。']
        for pair,row in teacher['pairs'].items():lines += [pair,diagnostic_record('fresh_teacher_anchor_diagnosis','pairs.'+pair,row)]
    if canonical is not None:
        lines += ['', '十、自然768维教师头与经典图的独立拆解',
            '自然768->64 GELU->1，无312零填充或手工HSV学入头；软/硬教师均充分训练。',
            '主方法为PCD图+显式PRESUPPOSED共同正信念保护，CHECKLIST严格等同未保护PCD图。',
            '图与KD共享完全相同软头；PCD/RGB/纯颜色/同头隐藏特征及无图锚点均报告，未按域挑胜者。',
            '保护系数.1+3.9max(0,2min(p400,pfinal)-1)是固定启发式正则，不是GT校准概率或统计置信区间。',
            '66原始+90数量输出封存后统一156官评，独立NumPy充分统计复现PASS。',
            '复用既有1160/1103真实答案，0新增教师/主干前向；该质量不能与旧3512秒拼接。']
        for pair in ('C4','C1'):
            row=canonical['repeated']['CANONICAL_'+pair]
            summary(lines,'自然768 / '+pair,row,list(row['comparisons']))
        lines += ['本轮查询检索/FP/FN诊断：',diagnostic_record('canonical_pcd_analysis','query_diagnosis',canonical['query_diagnosis'])]
    else:lines += ['', '十、自然768维研究仍在进行；未读取未完成的局部分数作结论。']
    if (PUBLIC/'preregistration_semantic_colour_v1.json').is_file():
        lines += ['', '十一、颜色与衣着部位的语义绑定及单调校准',
            '固定11颜色HSV原型，不用目标GT拟合；依据语言绑定上衣/下装，dressed-all同时看两部分。',
            'dark/light-side用上身相对亮度，按轨迹长度加权；不是新神经网络或目标GT规则表。',
            '规则支持C4的225/466句、C1的425/425句；HS和V独立近似、光照偏移、衣着部位遮挡等局限公开。',
            '普通SEM1081蒸馏也获得完全相同物理兼容标量，ZERO1081同初始化只补0；保留全部旧768/RGB/零HSV强软硬KD。',
            'PAVA只用过去/当前教师概率，当前视频5个TrackID组交叉排除该组全部查询；目标GT诊断精确率不进入权重。',
            '修正是logit(KD)+clip(logit(PAVA)-logit(训练教师平均),-ln3,ln3)，有界正则而非独立贝叶斯证据。',
            'BIN为非单调分箱替代估计器，另有分箱与平滑差异，不是只去掉单调约束的纯净对照。',
            'SHUFFLED只破坏辅助校准线索的图像对应，神经头仍使用真实线索；NAIVE_MEAN简单平均，同头与零头归因均报告。',
            'ZERO控制维度与初始化，未单独排除任意非零附加特征的正则作用；不据此过度归因语义绑定。',
            '6条件84原始+90双向数量=174官评，全部输出封存后才GT；源/教师前向新增0，不是新的冷启动。',
            '后处理未触发行等于自身SEM头；新语义头的重新训练仍可改变相对历史1080头的其它输出，不宣称全系统局部无损。']
        if semantic is None:lines += ['本轮仍在运行。上述内容是预登记设计，不是已得到的效果。']
        else:
            from scripts.a100_visual_classical_semantic_colour_analyze import plain_readout
            lines += plain_readout(semantic['repeated'],semantic['query_diagnosis'])
            for pair in ('C4','C1'):
                row=semantic['repeated']['SEMANTIC_'+pair]
                summary(lines,'语义与单调 / '+pair,row,list(row['comparisons']))
            lines += ['检索/FP/FN：',diagnostic_record('semantic_colour_analysis','query_diagnosis',semantic['query_diagnosis'])]
    if (PUBLIC/'preregistration_joint_visual_v1.json').is_file():
        lines += ['', '十一补、视觉图与外观校准的联合机制拆解',
            'SEM神经输入缺乏独立增益，主方案回到旧RGB1080均值头；显式物理线索只进入经典PAVA正则。',
            'KD/GRAPH/CAL/JOINT主二乘二，无边锚点、逐句偏移、逐句/全局打乱、纯教师查询基率控制均保留。',
            '较强SEM400已有头另作稳健性检查，不依据各方向GT切换部署头。',
            '6条件60原始+78数量共138官评，全部先封存再本研究GT；0新教师/编码器/头拟合，CPU质量重放。']
        if joint is None:lines += ['联合研究仍在运行，尚无本轮联合贡献结论。']
        else:
            from scripts.a100_visual_classical_joint_visual_analyze import plain_readout as joint_readout
            lines += joint_readout(joint['repeated'])
            for pair in ('C4','C1'):
                row=joint['repeated']['JOINT_'+pair]
                summary(lines,'图与校准联合 / '+pair,row,list(row['comparisons']))
            lines += ['联合检索/FP/FN：',diagnostic_record('joint_visual_analysis','query_diagnosis',joint['query_diagnosis'])]
    if (PUBLIC/'preregistration_appearance_joint_cold_v1.json').is_file():
        lines += ['', '十一再补、四卡独立真实冷启动',
            'GPU0 C4KD、1 C4JOINT、2 C1KD、3 C1JOINT；各自模型重载、新源前向、新SQLite，预计4526真实教师问答。',
            '四份全部封存后统一GT，真实外部Popen-to-exit含启动/权重/前向/头拟合/图/PAVA/输出，GT评估除外。']
        if appearance_cold is None:lines += ['本轮仍在运行，尚未取得完整实际成本结论。']
        else:
            for pair,row in appearance_cold['conditions'].items():
                lines += [pair,'本次实际HOTA：'+json.dumps(row['raw_HOTA']),
                    '本次实际差：'+describe(row['comparisons']['JOINT_vs_KD'])]
            for name,row in appearance_cold['runs'].items():
                lines += [name,'实际单方法外部进程秒='+str(row['worker']['actual_worker_process_wall_seconds']),
                    '相位秒='+json.dumps(row['phase_seconds']),
                    '实际头拟合/保存秒='+str(row['actual_head_fit_and_save_seconds']),
                    '实际后处理/输出秒='+str(row['actual_all_method_postprocess_and_output_seconds']),
                    '其中外观校准CPU秒='+str(row['appearance_regularization_CPU_seconds']),
                    '真实教师='+json.dumps(row['teacher']),'充分拟合='+json.dumps(row['teacher_fit_audit'])]
            lines += ['实际两臂输入比较：',diagnostic_record('appearance_cold_analysis','actual_within_pair_evidence_audit',appearance_cold['actual_within_pair_evidence_audit']),
                '与开发质量源比较：',diagnostic_record('appearance_cold_analysis','quality_source_evidence_audit',appearance_cold['quality_source_evidence_audit']),
                '预先固定seed0预测比较：',diagnostic_record('appearance_cold_analysis','expected_seed0_prediction_audit',appearance_cold['expected_seed0_prediction_audit']),
                '实际查询权衡：',diagnostic_record('appearance_cold_analysis','postseal_candidate_membership_diagnosis',appearance_cold['postseal_candidate_membership_diagnosis']),
                '验收与边界辅助字段：',diagnostic_record('appearance_cold_analysis','acceptance',appearance_cold['acceptance']),
                '这些是实际新运行各自的精度/时间，不是第四个独立神经种子；并行NAS/OS负载不支持算法加速或全预算Pareto最优宣称。']
    if (PUBLIC/'preregistration_direct_teacher_nores_v1.json').is_file():
        lines += ['', '十一末、标准直接教师分类头：去掉固定源先验的公平性补查',
            '普通直接蒸馏未必应叠加一个已跨域失准的源logit；新模块可能仅补救固定先验，所以必须排除。',
            '保持RGB1080/初始化/教师证据/归一化/优化与充分软硬训练；训练与预测z同时为0。源信息仍共同用于选题和tie-break。',
            '5种直接KD检查点加同头GRAPH/CAL/JOINT，6条件420小头拟合；48原始+24数量共72输出全封存后GT。',
            'GPU0..3等此前四份物理worker全部退出后才启动，不影响原计时；新增源/教师前向0，小头组件时间不冒充新冷启。']
        if direct is None:lines += ['此项尚未完成，当前Goal继续ACTIVE。']
        else:
            for pair in ('C4','C1'):
                row=direct['repeated']['DIRECT_'+pair]
                summary(lines,'标准直接教师 / '+pair,row,list(row['comparisons']))
            lines += ['直接头的检索/FP/FN：',diagnostic_record('direct_teacher_analysis','query_diagnosis',direct['query_diagnosis']),
                '若此项消除了此前残差优势，就必须继续修正方法和论文叙事。保留配方/最强主对照改变需对应真实物理成本验证。']
    if (PUBLIC/'preregistration_direct_teacher_pcd_nores_v1.json').is_file():
        lines += ['', '十一附、原生PCD768标准直接教师分类头',
            '前面的同维度零HSV仍是1080架构，不能代替自然768架构；RGB有时拖累基线，常规直接头必须覆盖原生输入。',
            '本方案在RGB直接头任何新GT结果之前登记；768→64 GELU→1，不拼接HSV，不叠加源logit，保留同5种充分软/硬检查点。',
            '同样48原始+24数量全部先封存后GT；PCD/RGB和教师问答复用相同物理源流，不按方向换最终头。',
            '图/外观后处理仍可用RGB；更丰富的RGB1080常规直接头及全部原有残差对照亦保留，不能仅胜较弱输入就宣布完成。']
        if direct_pcd is None:lines += ['此项尚未完成，当前Goal继续ACTIVE。']
        else:
            for pair in ('C4','C1'):
                row=direct_pcd['repeated']['DIRECT_'+pair]
                summary(lines,'原生PCD直接教师 / '+pair,row,list(row['comparisons']))
            lines += ['原生直接头检索/FP/FN：',diagnostic_record('direct_teacher_pcd_analysis','query_diagnosis',direct_pcd['query_diagnosis']),
                '这些是固定输入和问答的小头质量补查，不是新的真实冷启动时间点；变更配方/主对照须完成匹配物理验证。']
    if (PUBLIC/'preregistration_native_direct_cold_v1.json').is_file():
        lines += ['', '十一续、强原生基线后的统一最小化：原生直接头＋经典校准',
            '原生PCD直接头是合理且更强的常规基线。不能因旧方案绝对分数67.69较高，就忽略它对原生67.08基线的区间跨0。',
            '新的全局候选固定为768D直接充分软均值，不叠加源logit，保留视觉图后PAVA；两个方向使用同一规则，不按域选最高头。',
            '神经输入去掉312HSV；RGB仍用于明确声明的经典图/外观模块。旧丰富RGB直接、残差、语义标量与零HSV强普通头全部保留。',
            '已有三种子：C4 67.082937→67.638744，+.555807，描述性CI[.126540,1.020584]；C1 32.074099→35.731536，+3.657437，CI[.908978,7.208643]。',
            '旧残差组合绝对值稍高，差异区间跨0；选择更少输入、无固定先验、同头归因更清楚的候选属于已披露开发迭代，不是独立盲测。',
            'C4图独立及图在已校准头上的增量仍跨0；C1图在校准头上+2.238068区间正，校准在图上+1.861499区间正。不能把条件贡献改写为每域每模块都稳定。',
            '因此追加132封存控制：无边锚点、同总边权完全图、逐句偏移、逐句/全局打乱和纯教师查询基率；0新增头拟合或源/教师前向。']
        if native_factorial is None:lines += ['原生头特异性控制尚在运行，不能提前宣称视觉拓扑不可替代。']
        else:
            for pair in ('C4','C1'):
                row=native_factorial['repeated']['NATIVE_'+pair]
                summary(lines,'原生直接头机制 / '+pair,row,list(row['comparisons']))
            lines += ['同总边权完全图不保持每节点度，ROW_SHIFT仍可改变互斥跨句分配；上述控制需要按其真正干预解释。']
            lines += ['关键反证：局部关系不是普适最优。原生局部组合对同总边权完全图，C4 +3.772352、区间正，但C1 -1.844362、区间负。',
                '这排除了把固定局部邻域作为两方向均已证实的视觉本能；不能按数据集名称或人工GT选择不同拓扑。']
        if native_cold is None:
            lines += ['四张卡新的独立原生KD/JOINT真实冷启动仍在运行：全部重新加载/图像前向/私有SQLite，预期4526真实问答，全部四份先封存后GT。',
                '旧3738/3029秒不与新的原生准确率拼接，待native_cold收据和数值一致性核验。']
        else:
            for pair,row in native_cold['conditions'].items():
                lines += ['原生独立本次 '+pair,'HOTA：'+json.dumps(row['raw_HOTA']),describe(row['comparisons']['JOINT_vs_KD'])]
            for name,row in native_cold['runs'].items():
                lines += [name+' 实际Popen-to-exit秒='+str(row['worker']['actual_worker_process_wall_seconds']),
                    '阶段秒='+json.dumps(row['phase_seconds']),
                    '头拟合/保存秒='+str(row['actual_head_fit_and_save_seconds'])+'；后处理/输出秒='+str(row['actual_all_method_postprocess_and_output_seconds']),
                    '其中PAVA CPU秒='+str(row['appearance_regularization_CPU_seconds'])+'；真实教师='+str(row['teacher']['new_calls'])+'；缓存命中='+str(row['teacher']['cache_hits'])]
            lines += ['本次实际两臂输入：',diagnostic_record('native_cold_analysis','actual_within_pair_evidence_audit',native_cold['actual_within_pair_evidence_audit']),
                '与原生质量源的输入：',diagnostic_record('native_cold_analysis','quality_source_evidence_audit',native_cold['quality_source_evidence_audit']),
                '全部保存分数/预测的重放：',diagnostic_record('native_cold_analysis','expected_seed0_prediction_audit',native_cold['expected_seed0_prediction_audit']),
                '本次查询权衡：',diagnostic_record('native_cold_analysis','postseal_candidate_membership_diagnosis',native_cold['postseal_candidate_membership_diagnosis']),
                '本次验收字段：',diagnostic_record('native_cold_analysis','acceptance',native_cold['acceptance']),
                '这是新实测精度和新实测成本，不是第4个独立神经种子；不声称算法提速或所有预算最优。']
        if (PUBLIC/'preregistration_native_optimized_cold_v2.json').is_file():
            lines += ['原生真实成本V2另做了必要纠正：常规768D KD不使用手工RGB统计，旧共同实现却仍为它计算HSV和重复图像IO，会高估基线成本。',
                'V2仅让普通KD跳过无用RGB/几何，保留同一PCD、源先验、最大面积教师锚点和全部问答；全70视频锚点已验证相同，无需手工RGB图像解码。',
                'GPU0/2重新运行优化KD，GPU1/3的组合方案与各自计时launcher从本轮V1持续运行；没有重启或改动其数值。',
                '已中止的旧KD各3视频及已完成40/75回答（如批次在途则为下界），原始部分输出和中止耗时全部保留，作为科研修正开销单列，不能算成最终部署预付缓存。',
                'KD的RGB零占位与组合非零统计属于预期差异，必须明示。PCD/先验/轨迹/锚点/有序问答/教师概率/小头分数与掩码仍要求一致；组合RGB也须与质量源一致。',
                '组合方案使用RGB就支付其真实图像统计费用。旧同RGB比较的两秒后处理增量，不能代表当前原生KD与组合之间的全部成本差。',
                '模型VRAM都重新加载，初始化错峰、OS/NAS缓存未清理和主机共享负载明确披露；不以单次墙钟差宣称算法提速或所有预算最优。']
            if native_cold is not None:lines += ['旧KD中止的科研开销：',json.dumps(native_cold.get('aborted_unoptimized_KD_research_costs',{}),indent=2)]
    if (PUBLIC/'preregistration_predictive_topology_v1.json').is_file():
        lines += ['', '十一新、仅用教师留轨迹预测选择局部与整体参照',
            '先前固定局部拓扑的跨域矛盾促成此开发实验。不同于旧Gaussian边际证据BMA，直接比较两个实际图算子的教师留出预测风险。',
            '同TrackID全部查询统一五折留出，验证使用中性0.5一元项，避免已经见过留出教师回答的神经头把答案泄漏回交叉验证。',
            '每查询同权后汇总当前视频；给定PRESUPPOSED契约使用正负均衡softCE，CHECKLIST使用BernoulliCE。严格更小才选全局，同分选局部。',
            '选择器不读取域名/人工GT/神经头，所有种子的选择须一致。全BCE、全平衡损失、固定等权混合和反向选择为同批对照。',
            '6条件42原始及114数量共156输出全部封存后才本轮GT；0新增教师、编码器调用或神经头拟合。旧BMA失败仍完整保留。']
        if predictive is None:
            lines += ['本轮仍在运行。教师预测一致性不保证GT准确率，不能提前宣称解决了拓扑选择。']
        else:
            for pair,row in predictive['repeated'].items():summary(lines,'教师预测结构选择 / '+pair,row,list(row['comparisons']))
            lines += ['逐视频全局选择计数：',json.dumps({k:r['selector_global_video_counts'] for k,r in predictive['conditions'].items()}),
                '两个开发方向等权点均值（无新总体泛化区间）：'+json.dumps(predictive['equal_domain_macro_points_only']),
                '若没有胜过固定图、混合或强普通KD，应保留失败，不得按GT把选择器改成域名开关。']
            lines += ['V1实际晋级失败：C4均值67.413321对强nativeKD仅+.330384，原视频+种子CI[-.239241,.939313]跨0；C1均值37.493610、+5.419510虽显著，不能替代C4验收。',
                '所以为V1准备的额外两个GPU真实冷启动并未启动；70视频部署glue零差也不代表算法质量条件已过。']
        lines += ['本轮是CPU复用质量实验；若保留该选择器，必须有匹配实际执行/成本记录，不能将当前固定局部图的真实时间配到新精度上。']
    if (PUBLIC/'preregistration_stable_topology_v2.json').is_file():
        lines += ['', '十一又补、教师预测风险的稳定性控制',
            '严格最小值对很小的风险差也切换整体结构，V1的两个C4切换总计损失.225423 HOTA；这是诊断，不是给新选择器提供GT。',
            'V2主方法在任何新预测/GT之前固定ONE_SE：配对逐查询风险的均值改善必须大于一倍SE才选全局，否则局部。',
            '这些查询及交叉验证预测共享轨迹，SE只作敏感性正则，绝不是校准置信区间，也不证明教师一致性必然等于语义准确。',
            'TWO_SE、MEDIAN和DELETE_ONE均为预先固定的敏感性对照，不能在看到GT后把最高分的对照换成预登记主方法。',
            '同中性一元项、TrackID五折、冻结头与PAVA，0新增教师/编码器/小头拟合，30原始+114数量=144输出全部封存后本研究GT。']
        if stable is None:lines += ['本轮仍在运行；没有提前宣称更保守的规则就必然有效。']
        else:
            for pair,row in stable['repeated'].items():summary(lines,'ONE_SE稳定选择 / '+pair,row,list(row['comparisons']))
            lines += ['逐视频选择计数：'+json.dumps({k:r['stability_global_video_counts'] for k,r in stable['conditions'].items()}),
                '封存后检索诊断：'+diagnostic_record('stable_topology_analysis','query_diagnosis',stable['query_diagnosis'])]
            lines += ['V2主方法也未晋级：C4 +.481959，parent+seedCI[-.042041,.993548]仍跨0；C1相比完全图-.819848且区间为负。',
                '没有把TWO_SE/MEDIAN的探索性结果改称ONE_SE主检验通过；其预备额外GPU冷流没有启动。']
    if (PUBLIC/'preregistration_confidence_teacher_v1.json').is_file():
        lines += ['', '十一最后补、把教师置信度权重也交给普通蒸馏',
            '本方法图锚点使用1+3|2p-1|可靠性权重；补查普通软/硬KD也使用同一权重，不能把该信息当作本模块独占。',
            '除加权CE/熵/KL外，原生768D无源残差头、前缀归一化、初始化、充分训练、教师问答和全部快照不变。',
            '6条件420拟合、30原始+24数量=54输出全部封存后本组GT，0新增教师/主干前向；拟合组件时间不等于独立冷启动。']
        if weighted is None:lines += ['该普通基线仍在运行；最终科研Goal继续ACTIVE，不能据先前未加权基线结果提前结项。']
        else:
            for pair,row in weighted['repeated'].items():summary(lines,'教师置信度加权公平性 / '+pair,row,list(row['comparisons']))
            lines += ['加权普通均值C4 67.102356、C1 32.259906。旧未加权组合相对它C4平均+.536388，固定种子均值parentCI为正，但包含神经种子敏感性的CI[-.010093,1.127430]跨0；不能改用更窄的区间结项。',
                'C1 +3.471630区间仍正。420拟合全部4000步内满足加权KL<=.005，不能把普通基线说成训练不足。',
                '如果强加权普通头消除了此前增益，应保留该结果继续研究，不能挑回较弱基线。']
    if (PUBLIC/'preregistration_contract_graph_v1.json').is_file():
        lines += ['', '十一后续、只改变No证据可以传播到哪里',
            '全图总体HOTA提高却会误删部分短查询。新假设是：个别No通过相似关系传给尚未查证的人，形成额外误删。此处是待验证机制，不是已经证实的原因。',
            'PRESUPPOSED只断开教师No节点的图边，其局部No锚点保留；CHECKLIST保留原完整传播。源先验精度、教师权重、头、PAVA和>.5解码均不变，不强制非空。',
            'WEAK控制逐query总边权与剪边图相同，避免只减弱传播强度就冒称No边界特异性；ALL_POSITIVE在CHECKLIST也剪No边，检验协议必要性。',
            '24原始+114数量=138新输出全封存后本组GT；只用同一图像/教师/查询，无新编码器、教师调用或头拟合。']
        if contract is None:lines += ['本轮仍在运行，尚不能宣称改善正确检索或减少误删。']
        else:
            for pair,row in contract['repeated'].items():summary(lines,'负证据边界 / '+pair,row,list(row['comparisons']))
            lines += ['正确检索、空输出及误报漏报：'+diagnostic_record('contract_graph_analysis','query_diagnosis',contract['query_diagnosis']),
                '主方案失败：C4比完整J下降.384532，原视频+种子CI[-.670867,-.074881]为负；seed0虽然净救回5查询，但FP增加51,935。全面限制No传播不可保留。']
    if (PUBLIC/'preregistration_weighted_classical_v1.json').is_file():
        lines += ['', '十一更后续、强加权基础上的完全同头归因',
            '加权普通KD已成为必要强对照。让两边都共享它同一个已封存的原生768D直接均值头，只增减原有图/PAVA。',
            '置信度加权是共享普通训练基础，不记作我方创新；图参数、锚点精度、颜色线索、校准和解码也不按结果调整。',
            '6条件18原始+60数量=78输出，全封存后本组GT，0新增源/教师前向或头拟合。所有旧软/硬和丰富输入普通头仍在比较中。']
        if weighted_classical is None:lines += ['本轮仍在运行；旧未加权真实时间不能套到这个新配方精度。']
        else:
            for pair,row in weighted_classical['repeated'].items():summary(lines,'共享强加权头的经典模块 / '+pair,row,list(row['comparisons']))
            lines += ['查询权衡：'+diagnostic_record('weighted_classical_analysis','query_diagnosis',weighted_classical['query_diagnosis']),
                '主验收未通过：C4共享加权J67.580033对强加权KD+.477677，parent+seedCI[-.069102,1.060519]跨0；C1+3.467750区间正。不能把一域稳定、一域不确定写成全面完成。',
                '如保留此共享基础，须新的对应真实执行/成本与来源核验；没有因复用质量队列完成自动结束Goal。']
    if (PUBLIC/'preregistration_cue_consensus_v1.json').is_file():
        lines += ['', '十一继续、视觉冲突只约束特定No传播',
            '全面No隔离失败后，保留同一强加权普通头，只在排除当前轨迹全部教师监督后的视觉校准正修正达到已有ln3上限时，限制这条No向外传播；局部No锚点仍保留。',
            'CHECKLIST完整图不变；随机同数量No隔离及同总边权均匀弱化为预登记对照，不能事后从对照挑最高分。',
            '6条件30原始+114数量=144新输出，全封存后本研究GT，0新增教师/编码器/头拟合。门槛继承冻结校准上限，没有用目标GT扫描阈值。']
        if (PUBLIC/'cue_consensus_analysis.json').is_file() and (PUBLIC/'cue_consensus_statistical_reproduction.json').is_file():
            consensus=load('cue_consensus_analysis')
            assert load('cue_consensus_statistical_reproduction')['status'].startswith('PASS_')
            for pair,row in consensus['repeated'].items():summary(lines,'选择性视觉冲突 / '+pair,row,list(row['comparisons']))
            lines += ['查询权衡：'+diagnostic_record('cue_consensus_analysis','query_diagnosis',consensus['query_diagnosis']),
                '完整根代理复核：主假设失败，不保留。C4 67.547497相对原完整共享加权J-.032536，区间[-.126173,.085614]跨0；相对强加权KD+.445142，区间[-.091387,1.028821]仍跨0。',
                '三个种子都没有救回新的漏找query（rescue/regress均0），却FP分别增加4990/5347/7190。对随机同量隔离没有稳定优势。C1完全不变不是新贡献。',
                '294HOTA/336效应独立复算PASS只是算术可信，不等于新机制有效；没有启动该配方额外实际冷流。其成本不沿用旧方法时间。']
        else:lines += ['本轮待完整预测、官评和独立算术复现，当前没有新方案已胜过普通KD的结论。']
    if (PUBLIC/'preregistration_coverage_search_v1.json').is_file():
        lines += ['', '十一新证据、把有限检查次数分给尚未找到人的查询',
            '冻结旧问答不能凭空提供从未问过的候选证据。新COVERAGE_SEARCH保留原COVER逐query候选顺序和每视频10%总预算，仅在PRESUPPOSED按已观察No概率乘积/(query已花次数+1)重新分配余下预算。',
            '先每query看一次，预算不足时遵循共同查询顺序；CHECKLIST关闭该反馈分配，问题集合与原COVER完全一致。70视频无反馈集合核验差0、GT0。',
            'No乘积只是相关教师回答下的搜索优先级代理，不是校准存在性后验。未见回答不参与选题，也不读取GT；新增缺失问题由真实GPU教师回答。',
            '私有冻结COVER缓存用于质量证据复用，不能把缓存采集时间冒充冷启动。普通软/硬KD同样取得全部新SEARCH问题及回答，不能独占信息。',
            '三种子270头拟合，27原始+57双向数量=84输出全封存后本研究GT；拆解普通KD下的选题贡献与同问答图/PAVA贡献。C1条件原样不是新增独立验证。']
        if (PUBLIC/'coverage_search_analysis.json').is_file() and (PUBLIC/'coverage_search_statistical_reproduction.json').is_file():
            coverage=load('coverage_search_analysis')
            assert load('coverage_search_statistical_reproduction')['status'].startswith('PASS_')
            for pair,row in coverage['repeated'].items():summary(lines,'同预算覆盖搜索 / '+pair,row,list(row['comparisons']))
            lines += ['封存后覆盖/检索诊断：'+diagnostic_record('coverage_search_analysis','query_diagnosis',coverage['query_diagnosis']),
                '根代理完整复核：主门槛失败，不保留。新KD66.982569对旧强KD67.102356 -.119787；新J67.239436低于旧J67.580033，差-.340597，三个种子均负。',
                '新J对自己的软KD、旧强软KD、旧完整收敛FINAL均无可靠正增益。270拟合全4000步达标，不能归成普通头欠拟合。',
                '实际正确候选覆盖改善：未问到正确candidate的查询122→113，未得到正确Yes的查询147→134。与旧J相比，新J净救回13/8/4条query，却多出34603/33914/31842个FP观测；覆盖改善不是HOTA必然提高。',
                '图比坏完全图+4.15不证明优于强KD；新J比CAL还低.172124。只读单一CountMatched正差也不能宣称普遍排序突破。',
                '没有启动失败SEARCH配方额外真实冷流，质量复用时间不冒充实际部署成本。Goal继续ACTIVE。']
        else:lines += ['本轮已启动真实缺失回答采集及后续质量队列，尚无完整评分结果。Goal继续ACTIVE。']
    if (PUBLIC/'preregistration_optional_transfer_v1.json').is_file():
        lines += ['', '十一进一步、图选择器必须允许不作跨人类推',
            '旧选择器只能在局部/全局两种非零传播图之间选。只读教师收据、不读GT的审计发现：C4 36/45视频两图都不及0.5中性教师预测，C1仅1/25；这本身不是HOTA结论。',
            '新候选集加入IDENTITY：部署时严格保留同一个强加权KD，再接原PAVA；不把它混同成已问节点仍强制替换原始教师概率的零边图。',
            '留TrackID教师预测时，最佳图相对中性IDENTITY的风险下降须超过一倍配对query SE才使用，否则原KD不作传播。PRESUPPOSED用平衡softCE，CHECKLIST用BCE。',
            'SE只是共享轨迹/交叉拟合依赖下的敏感性正则，不声称校准置信度或GT最优性。严格最小风险、被迫用图、固定全局、原局部/CAL及局部原始教师锚点回填控制均保留。',
            '6条件36原始+114数量=150输出，全封存后本研究GT，0新增教师/主干/头拟合。70视频旧CV风险、完整图、IDENTITY→CAL重放均0差、GT0；教师风险主规则在C4选择44IDENTITY/1LOCAL，C1选择5IDENTITY/20GLOBAL。']
        if (PUBLIC/'optional_transfer_analysis.json').is_file() and (PUBLIC/'optional_transfer_statistical_reproduction.json').is_file():
            optional=load('optional_transfer_analysis')
            assert load('optional_transfer_statistical_reproduction')['status'].startswith('PASS_')
            for pair,row in optional['repeated'].items():summary(lines,'允许不传播的教师预测选择 / '+pair,row,list(row['comparisons']))
            lines += ['查询检索及误报漏报：'+diagnostic_record('optional_transfer_analysis','query_diagnosis',optional['query_diagnosis']),
                '根代理复核：primary仍未接受。虽然对均值KD，C4+.360045、C1+5.019675的区间均正，但C4对更强充分收敛FINAL仅+.310792，区间[-.063214,.762190]跨0。',
                '新门控对被迫传播C4+.084777区间跨0、C1-.266020且区间上界近0，没有稳定独立收益；不把更高的STRICT对照事后改称primary成功。',
                '没有启动该失败配方新实际冷流。是否只是取消已问节点的强制教师锚点，须读ANCHOR_OPTIONAL控制。与SEARCH叠加必须另测，不能把两组涨幅相加。']
        else:lines += ['本轮已启动，尚无完整新HOTA实证。若保留须相应新真实物理执行，不沿用旧固定图wall。Goal继续ACTIVE。']
    if (PUBLIC/'preregistration_protocol_calibration_v1.json').is_file():
        lines += ['', '十一之五、回到共享收敛头：任务协议与视觉校准的最简组合',
            '旧OPTIONAL、SEARCH以及选择性No规则均未通过，不把其中最高对照事后晋级。',
            '本轮双方统一使用强加权充分收敛FINAL，而不是只与略弱均值比较；不按目标GT选择checkpoint。',
            '任务提供PRESUPPOSED时原样保留FINAL，CHECKLIST时应用原同总边权全局图，然后统一执行原PAVA。函数只看任务协议，不看域名或GT。',
            '这里全局是已有候选之间的证据汇聚，不是新增FRAME_BOX教师视角；双方仍使用同一冻结CROP教师接口、相同回答和预算。',
            '普通基线也取得同样协议，对独立Bernoulli候选边缘条件化为至少一个候选；清单原样。该对照假设候选覆盖场景，不强制单人，也不提供真实存在性。',
            '固定局部/全局、仅校准、仅协议池化和两方向数量投影全部保留；36原始+78数量共114个输出全封存后本研究GT。',
            '旧均值头的CAL/全局组件结果已经看过，当前规则是开发结果驱动的简化，不是独立确认性验证。',
            '全局秩一求解已在70视频与原稠密算子/旧组件核验，概率与决策不变；仅求解阶段线性，不将建图成本隐藏。']
        if (PUBLIC/'protocol_calibration_analysis.json').is_file() and (PUBLIC/'protocol_calibration_statistical_reproduction.json').is_file():
            assert load('protocol_calibration_statistical_reproduction')['status'].startswith('PASS_')
            result=load('protocol_calibration_analysis')
            for pair,row in result['repeated'].items():
                summary(lines,pair,row,[name for name in row['comparisons'] if name.startswith('PROTOCOL_JOINT_vs_') or name.startswith('COUNT_') or name in ('FINAL_CAL_vs_KD','PROTOCOL_GRAPH_vs_KD')])
            review=PUBLIC/'protocol_calibration_scientific_review.json'
            if review.is_file():
                checks=json.loads(review.read_text())
                lines += ['主科学验收状态：'+checks['status'],'逐项门槛：'+json.dumps(checks['checks'],ensure_ascii=False),
                    '完整验收及效应见：'+str(review)]
            lines += ['配套真实成本必须来自对应新配方和FINAL读出，不引用旧均值头wall；算术复算PASS并不自动完成Goal。']
            lines += ['C4相对强Final双数量投影均有小幅正差；C1两投影区间跨0，主要是数量/查询基率校准，不作普遍语义排序宣称。',
                'C4三个种子相对Final的canonical query净救回0/1/-3，C1为0/2/-2；总体HOTA正不等于漏查已全部消除。',
                '当前两个方向中，协议和数据集重合；代码接收任务协议而非域名，但尚无同域跨协议或未见域验证，不能据此证明普遍的协议泛化。',
                diagnostic_record('protocol_calibration_analysis','query_diagnosis',result['query_diagnosis'])]
        else:lines += ['本轮已启动，结果尚未完整核验；不提前宣布有效。']
    if (PUBLIC/'preregistration_protocol_cold_v1.json').is_file():
        lines += ['', '十一之六、共享收敛头与同任务协议的四卡真实冷启动',
            'GPU0/1分别C4普通/组合，GPU2/3分别C1普通/组合；每份独立加载源与Qwen、真实提特征、独立新SQLite、重训相同强加权头，预期4526真实回答/0缓存。',
            '普通基线也获得非空/清单协议，使用收敛FINAL并跳过未使用RGB；组合支付RGB、汇聚、PAVA成本。全部四份预测封存后本组GT。']
        if (PUBLIC/'protocol_cold_analysis.json').is_file() and (PUBLIC/'protocol_cold_statistical_reproduction.json').is_file():
            assert load('protocol_cold_statistical_reproduction')['status'].startswith('PASS_')
            actual=load('protocol_cold_analysis')
            for pair,row in actual['conditions'].items():
                lines += [pair,'本次真实HOTA：'+json.dumps(row['raw_HOTA']),describe(row['comparisons']['JOINT_vs_KD'])]
            for name,row in actual['runs'].items():
                lines += [name,'完整外部进程wall秒='+str(row['worker']['actual_worker_process_wall_seconds']),
                    '阶段秒='+json.dumps(row['phase_seconds']),
                    '拟合保存秒='+str(row['actual_head_fit_and_save_seconds'])+'；后处理输出秒='+str(row['actual_all_method_postprocess_and_output_seconds']),
                    '真实教师='+str(row['teacher']['new_calls'])+'；缓存='+str(row['teacher']['cache_hits'])]
            lines += ['实际与质量预测/输入一致性：'+json.dumps(actual['expected_seed0_prediction_audit']),
                '各方法真实Popen-to-exit，不套旧均值头时间。OS/NAS缓存未清、主机共享与错峰负载不受控，不作算法提速或全预算最优宣称。']
        else:lines += ['已按通过的质量门槛启动，物理收据尚待完整验收。尚无新的完成时长，不引用旧时间。']
    if (PUBLIC/'preregistration_pure_components_v1.json').is_file():
        lines += ['', '十一之七、回填老师答案、跨候选图边和视觉对应的纯净拆解',
            '主质量分已知后补做机制审计，主配方与正在执行的物理源码不改；不是独立盲测，也不按最高控制重新选配方。',
            '零边图保留相同局部教师锚点；ANCHOR_CAL随后原PAVA。置乱控制按固定hash打乱线索与候选对应，再用原样跨TrackID前缀重拟合PAVA。',
            '6条件24原始+42数量=66份新控制，全部封存后本组GT，0新增教师/编码器/神经头拟合。不是旧封存v8P66。']
        if (PUBLIC/'pure_components_analysis.json').is_file() and (PUBLIC/'pure_components_statistical_reproduction.json').is_file():
            assert load('pure_components_statistical_reproduction')['status'].startswith('PASS_')
            pure=load('pure_components_analysis')
            for pair,row in pure['repeated'].items():summary(lines,pair,row,list(row['comparisons']))
            lines += ['内部成分归因检查：'+json.dumps(pure['attribution_checks']),
                '如果成分证据不成立，不能把汇聚整体差全部算成图边或语义对应；不得事后晋级表现更高的控制。']
        else:lines += ['已登记并执行，成分归因尚无新结论。']
    lines += ['', '十二、仍需准确回答的科学问题与结论边界',
        '候选诊断标签来自全候选库的逐query逐帧IoU.5一对一匹配。canonical正/负选中数不是教师纯语义正确率，也不是各方法TrackEval重新匹配后的实际TP/FP。',
        '定性诊断可见框定位、重叠邻人、模糊与光照歧义；例如一个视觉上all-black的TeacherYes候选IoU仅.405，不能直接叫教师语义答错。样例用于解释，非随机频率估计，未参与参数选择。',
        '严格同RGB主对照是否在两个方向都支持正增益，是否胜过所有充分软/硬checkpoint，详见上面的全部实际结果。',
        '即使均值提升，也应检查三个种子的方向、按原视频不确定性和双向数量投影，不凭某个单点封神。',
        '开发数据上的描述性区间没有修正多次开发或多重比较，也不能证明未见域总体性能。',
        'bootstrap只重组封存预测的统计，不在每个抽样中重训适配/选题全流程；视频前缀适配依赖不会因此消失。',
        '没有固定全预算Pareto最优结论，没有所有视觉直觉都有效、所有LoRA都无效的结论。',
        '不借用Oracle、固定坏图的大差或直接回填弱基线，冒充对强普通KD的独立增益。',
        '如果严格同信息对照消除了旧优势，研究必须继续，不能弱化基线或以写完报告当作完成。','',
        '十三、完整证据索引（仓库仅存精选统计，原始结果不上传）']
    for stem in ('benchmark_scope_audit','sparse_observation_analysis','context_cohort_analysis','query_scope_analysis','budget_cold_analysis','minimal_final_analysis','strict_rgb_analysis','null_set_analysis','zero_colour_analysis','fresh_teacher_anchor_diagnosis','canonical_pcd_analysis','preregistration_semantic_colour_v1','semantic_colour_analysis','joint_visual_analysis','appearance_cold_analysis','direct_teacher_analysis','direct_teacher_pcd_analysis','native_factorial_analysis','native_cold_analysis','preregistration_predictive_topology_v1','predictive_topology_analysis','preregistration_stable_topology_v2','stable_topology_analysis','preregistration_confidence_teacher_v1','confidence_teacher_analysis','preregistration_contract_graph_v1','contract_graph_analysis','preregistration_weighted_classical_v1','weighted_classical_analysis'):
        if not (PUBLIC/(stem+'.json')).is_file():continue
        lines.append(str(PUBLIC/(stem+'.json')))
    for stem in ('cue_consensus_analysis','cue_consensus_scientific_review','preregistration_coverage_search_v1','coverage_search_evidence_receipt','coverage_search_analysis','coverage_search_scientific_review','teacher_transfer_null_audit','preregistration_optional_transfer_v1','optional_transfer_sanity','optional_transfer_analysis','optional_transfer_scientific_review','preregistration_protocol_calibration_v1','protocol_calibration_sanity','protocol_calibration_analysis','protocol_calibration_scientific_review','preregistration_protocol_cold_v1','protocol_cold_analysis','preregistration_pure_components_v1','pure_components_analysis'):
        if (PUBLIC/(stem+'.json')).is_file():lines.append(str(PUBLIC/(stem+'.json')))
    lines += [str(PUBLIC/'strict_rgb_parent_statistics.json'),str(PUBLIC/'strict_rgb_statistical_reproduction.json'),
        '公开充分统计可在无GPU/无原始GT条件下用NumPy复算HOTA和效应；算术复现不是独立实验。',
        '原始新运行：/data3/liuyeqiang/FlexHook-TTA/a100_runs/visual_classical_20261121',
        '大型历史资产只读：/data3/liuyeqiang/nas/A800；新产物不覆盖NAS历史results。',
        'GitHub仅代码/实验TXT/精选JSON与PNG/PDF；snapshot和可达历史均必须小于100,000,000字节。',
        'Goal由根代理核验真实目标后单独判断；此报告不会自动结项。']
    path=ROOT/'docs/TTA_RMOT_视觉本能与经典算法_持续研究综合实证_20261122.txt'
    if FULL_DIAGNOSTICS:
        path=ROOT/'a100_runs/visual_classical_20261121/analysis/TTA_RMOT_持续研究完整逐查询计数版_20261122.txt'
    path.write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='RESEARCH_SYNTHESIS_WRITTEN',path=str(path),goal_auto_closed=False),ensure_ascii=False))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--full-diagnostics',action='store_true')
    FULL_DIAGNOSTICS=parser.parse_args().full_diagnostics
    main()
