"""Publish only small, completed A100 evidence and a detailed Chinese TXT."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import compat.sitecustomize
from flex_uda.a100.runtime import OUTPUT


def load(p):return json.loads(p.read_text())


def save_official_evidence(source,destination):
    """Keep exact video pooling inputs without publishing large raw predictions."""
    from scripts.active_eval import official_stats
    destination.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source/'official_completion.json',destination/'official_completion.json')
    stats=official_stats(source)
    result=dict(videos=stats['videos'],HOTA_unrounded=stats['HOTA_unrounded'],
        numerator_AssA_times_TP=stats['stats'][0].tolist(),
        denominator_TP_plus_FN_plus_FP=stats['stats'][1].tolist(),
        pooling='100 * mean_alpha(sqrt(sum_video(numerator) / max(sum_video(denominator), 1)))',
        role='Exact sufficient statistics from official TrackEval; supports independent paired-video bootstrap without GT or predictions.')
    (destination/'official_video_pooling.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


def plot_causal(result,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=('MINUS','MINUS_RECALC_ONLY','SLOW_REPLAY_MATCHED')
    labels=('Fast only','Fast + recalc','LoRA + matched recalc')
    values=[result['metrics'][m]['HOTA_unrounded'] for m in methods]
    fig,(left,right)=plt.subplots(1,2,figsize=(11,4.7),gridspec_kw={'width_ratios':[1,1.65]})
    left.barh(labels,values,color=['#7b8794','#2369bd','#e48a22'])
    left.set(xlim=(0,60),xlabel='Official full C4 HOTA (%)',title='Observed system results')
    left.invert_yaxis()
    for i,value in enumerate(values):left.text(value+.4,i,f'{value:.3f}',va='center',fontsize=9)
    contrasts=[('MINUS_RECALC_ONLY_vs_MINUS','Recalc - fast only'),
        ('SLOW_REPLAY_MATCHED_vs_MINUS_RECALC_ONLY','LoRA - matched recalc'),
        ('LORA_RANK__RECALC_COUNT_vs_FROZEN_RANK__RECALC_COUNT','LoRA rank, recalc counts'),
        ('LORA_RANK__LORA_COUNT_vs_FROZEN_RANK__LORA_COUNT','LoRA rank, LoRA counts')]
    for i,(key,label) in enumerate(contrasts):
        row=result['comparisons'][key];point=row['delta_HOTA'];low,high=row['paired_video_CI95']
        right.errorbar(point,i,xerr=[[point-low],[high-point]],fmt='o',capsize=4,color='#b94141' if i>=2 else '#2369bd')
    right.set_yticks(range(len(contrasts)),[x[1] for x in contrasts]);right.invert_yaxis()
    right.axvline(0,color='#777777',linestyle='--',linewidth=1)
    right.set(xlabel='Delta HOTA (percentage points), paired video 95% CI',title='Isolated contributions and ranking diagnostics')
    right.grid(axis='x',alpha=.2)
    fig.suptitle('C4: fixed Q&A and matched 32-frame recalculation',fontsize=13)
    fig.tight_layout();fig.savefig(destination/'causal_attribution.png',dpi=220);fig.savefig(destination/'causal_attribution.pdf');plt.close(fig)


def main():
    s1=load(OUTPUT/'stage1/receipt.json');s2=load(OUTPUT/'stage2/analysis.json');s3=load(OUTPUT/'stage3/analysis.json')
    assert s1['status']=='PASS' and s1['cost_receipt_audit']=='PASS'
    assert len(s3['rows'])==75
    assert s2['actual_optimizer_updates']['MINUS_RECALC_ONLY']==0
    assert s2['recalc_events']['MINUS_RECALC_ONLY']==s2['recalc_events']['SLOW_REPLAY_MATCHED']
    assert load(OUTPUT/'verification/regression.json')['status']=='PASS'
    assert load(OUTPUT/'verification/budget_receipt_audit.json')['status']=='PASS'
    from scripts.a100_evaluate import plot_short,plot_full
    plot_short(s3['rows'])
    plot_full(load(OUTPUT/'stage3/full_curve_analysis.json')['rows'])
    dst=ROOT/'docs/a100_three_stage_20261120';dst.mkdir(parents=True,exist_ok=True)
    sources=['environment.json','nas_mount.json','image_io_audit_probe.json','compute_integrity.json','compute_complete.json','cache_handoff_complete.json',
        'stage1/receipt.json','stage1/receipt_before_cost_audit.json','stage1/cuda_ops.json',
        'stage2/preregistration.json','stage2/preregistration_v1_before_A100_numerics.json','stage2/runtime_cache_amendment.json','stage2/analysis.json',
        'stage3/preregistration.json','stage3/scheduling_amendment.json','stage3/startup_accounting_amendment.json',
        'stage3/deadline_accounting_amendment.json','stage3/analysis.json','stage3/cost_utility.csv',
        'stage3/cost_utility.png','stage3/cost_utility.pdf','stage3/full_curve_registration.json',
        'stage3/full_cost_utility.csv','stage3/full_curve_analysis.json','stage3/full_cost_utility.png','stage3/full_cost_utility.pdf',
        'verification/initial_regression_attempt.json','verification/regression.json','verification/regression.log','verification/budget_receipt_audit.json']
    for relative in sources:
        p=OUTPUT/relative
        if p.exists():
            target=dst/relative
            if relative=='verification/regression.log':target=target.with_suffix('.txt')
            target.parent.mkdir(parents=True,exist_ok=True)
            if p.suffix=='.csv':target.write_text(p.read_text())
            else:shutil.copy2(p,target)
    for p in (OUTPUT/'stage2/canonical').glob('*/resume_numerical_equivalence.json'):
        target=dst/'stage2/cache_equivalence'/p.parent.name/'receipt.json';target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for method in ('MINUS','MINUS_RECALC_ONLY','SLOW_REPLAY_MATCHED'):
        save_official_evidence(OUTPUT/'stage2'/method,dst/'stage2'/method)
    for method in ('FROZEN_RANK__RECALC_COUNT','LORA_RANK__RECALC_COUNT','FROZEN_RANK__LORA_COUNT','LORA_RANK__LORA_COUNT'):
        relative=Path('stage2/countmatched')/method
        save_official_evidence(OUTPUT/relative,dst/relative)
    for method in ('SLOW_REPLAY_MATCHED','MINUS_RECALC_ONLY'):
        for p in (OUTPUT/'stage2'/method/'attempts').glob('*/supervisor_receipt.json'):
            target=dst/'stage2'/method/'attempts'/p.parent.name/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for scope,repeats in [('short',range(3)),('full',range(1))]:
        for repeat in repeats:
            for method in ('MINUS_v8P','YT_PRIOR','UNIFORM_K'):
                relative=Path('stage3')/scope/f'repeat_{repeat}'/method
                for name in ('supervisor_receipt.json','worker_receipt.json','cost_ledger.json','implementation_equivalence_audit.json'):
                    p=OUTPUT/relative/name
                    if p.exists():
                        target=dst/relative/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
                budget=600 if scope=='short' else int(load(OUTPUT/relative/'supervisor_receipt.json')['wall_seconds'])
                save_official_evidence(OUTPUT/relative/'at_budget'/str(budget),dst/relative/'at_final_budget')
    plot_causal(s2,dst/'stage2')
    weight=s2['comparisons']['SLOW_REPLAY_MATCHED_vs_MINUS_RECALC_ONLY']
    recalc=s2['comparisons']['MINUS_RECALC_ONLY_vs_MINUS']
    total=s2['comparisons']['SLOW_REPLAY_MATCHED_vs_MINUS']
    cm=s2['comparisons']['LORA_RANK__RECALC_COUNT_vs_FROZEN_RANK__RECALC_COUNT']
    cm_lora=s2['comparisons']['LORA_RANK__LORA_COUNT_vs_FROZEN_RANK__LORA_COUNT']
    positive=weight['delta_HOTA']>0 and weight['paired_video_CI95'][0]>0
    semantic=cm['delta_HOTA']>0 and cm['paired_video_CI95'][0]>0
    lines=[]
    def add(text=''):lines.extend(str(text).splitlines());lines.append('')
    add('TTA-RMOT：A100三阶段实证报告与研究交接（2026-11-20）')
    add('报告生成时间：'+datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')+'；文件名日期为本轮实验登记日期。')
    add('一、这轮工作的出发点')
    add('此前已经完成A800对话、NAS资产和代码恢复，以及小于100MB的GitHub轻量分支整理。旧v8P六方向66项已经封存。本轮用户明确要求推进三个新阶段：真实GPU最小闭环；剥离32帧近似重算后检验LoRA贡献；按真实墙钟时间比较主动选题与基准策略。我们没有把“代码已恢复”“模型能导入”或“逻辑调用少”当成这三个目标已经实现。')
    add('总约束：物理GPU仅0、1、2、3；NAS历史results只读；新结果在a100_runs；权重/大数据不上传GitHub；不启动旧80项队列、v9探针或重做旧66项矩阵。所有实验不以涨点为前提。')
    add('二、阶段一：真实一图一句闭环')
    add('实际样本：排序第一个SportsMOT视频，第10帧，第一条包含clothes的真实查询。一个学生锚帧、一个查询、9个官方固定候选。学生直接执行官方视觉骨干、RoBERTa文本编码和解码身份打分，PCD形状[9,1,768]。候选是原官方跟踪前端输出，不能把读候选说成重跑了检测和关联。学生历史按episode起始规则重复锚帧；教师CLIP另取真实过去时刻，没有用CROP回退冒充时序。')
    add(f"学生严格加载：{s1['student']['strict_load']}；官方checkpoint SHA256：{s1['student']['checkpoint_sha256']}。")
    add('环境为Python3.10、PyTorch2.6.0+cu118。Qwen所需transformers4.57.1、accelerate1.11、peft0.17.1及配套依赖已安装到项目.venv，共享环境未改动。实跑先暴露checkpoint实际文件名、fvcore、wordninja问题，修正后继续；失败日志保留，未把失败包装成成功。')
    add('GPU2/3上的torchvision CUDA NMS和ROI Align前向/梯度都通过。旧CP311 fused-window扩展未用于本次执行。另发现旧配置解析器把存在的False参数也标为True；官方build_model没有把该标志传给RoPE骨干，实际块采用False默认。本机入口现明确置False，使配置记录和实际执行一致。没有声称重编译了未使用的扩展。')
    add(f"冻结Qwen：真实请求{s1['teacher']['new_calls']}次，缓存命中{s1['teacher']['cache_hits']}；CROP、CLIP、FRAME_BOX各9次，4批真实GPU前向。固定revision={s1['teacher']['revision']}；权重参数指纹={s1['teacher']['teacher_weight_sha256']}。初末冻结校验通过，GT打开次数0。较晚出现的候选轨迹只有两个可用历史时刻，其余有四个；均保留实际时刻。")
    add(f"快分支身份MAP：目标{s1['fast']['identity']['objective_before']:.9f}→{s1['fast']['identity']['objective_after']:.9f}；真实教师支持正例{s1['fast']['identity']['positive']}、负例{s1['fast']['identity']['negative']}；参数{s1['params_before']}→{s1['params_after']}。这是校准参数和选择信念更新，不是LoRA神经网络训练。单查询没有presence双类，因此T/beta按照原规则跳过，没有补造负标签。")
    add('阶段一成本（同步墙钟秒）：')
    for name,value in [('学生加载与校验',s1['student']['model_load_seconds']),('学生真实特征前向',s1['student_feature_seconds']),
        ('教师初始化含原始参数指纹',s1['teacher_initialization_seconds']),('教师实际backend.forward相位',s1['teacher']['forward_seconds']),
        ('教师完整请求服务含图片准备',s1['teacher_request_wall_seconds']),('教师最终冻结封签',s1['teacher_final_seal_seconds']),
        ('快拟合',s1['fast_fit_seconds']),('端到端冷进程',s1['process_wall_seconds'])]:add(f'  {name}：{value:.6f}')
    add('这些相位有包含关系，不能机械相加。教师backend.forward相位包含必要的处理和同步，不是GPU内核利用率积分。收据还修正了默认CUDA设备导致的显存记录问题：教师GPU1实际峰值5,854,893,056字节，原GPU0默认设备数值另存。')
    add('三、阶段二：为什么旧REPLAY不足以完成归因')
    add('旧SLOW_REPLAY固定了MINUS选题和回答，但仍执行LoRA更新、固定32帧近似重算、按32帧重新估计轨迹选择频率，以及再一次快拟合。因此它相对MINUS的涨点是联合效果，不能直接称为纯LoRA学习。近似重算即使权重不变，也可能因采样统计替代完整统计而改变输出。')
    add('本轮在新GT评估前固定45视频、466表达、3508原始问答；原概率、视图、顺序、逐轮presence快照与DS规则保留。LoRA作为匹配调度的生产者，零权重组在相同轮次执行同32帧重算和再快拟合。零组LoRA从不执行optimizer.step，最终参数逐项核验未变。')
    add('运行前V2修订专门消除了机器数值混杂：A100零权重组先实时产生canonical完整视频输入，纯快分支和零更新LoRA前缀共享该基准。A800旧数值仅作历史参考，避免只给控制组回填旧缓存、给学习组使用当前机器重算的非对称处理。V1文件仍保留，可追溯修订发生在GT之前。')
    add('模型与优化保持原慢分支：最后两个PCD块的Q/V投影，rank4、alpha8；AdamW学习率2e-4、weight_decay1e-4；每次50步、每批16正16负，梯度裁剪5，FP16 AMP和原回退规则；固定seed0和排序视频流，跨视频携带适配器、优化器、DS及快分支状态。未按GT挑视频、调参数或挑checkpoint。')
    add('初始与32帧重算仍正常计算PCD，但不再落盘下游从未消费的大PCD数组。margin、重算记录、预测、信念及状态保留。这个工程节省单独说明，不当作特征学习贡献。')
    add('另一次运行时工程修订发生在两组共同完成11视频之后：旧512MiB缓存仅容纳79帧冻结Swin四级特征，第9视频50步LoRA耗时742.102秒。两因果GPU组从共同检查点将精确冻结编码器LRU改为8192MiB，历史样本、优化步数、超参数和32帧规则不变；遇到旧canonical须对margin、presence、utility、raw_k逐位回归。旧运行、中断与重载成本保留并计入归因研究总wall。阶段三的独立计时仍用原缓存配置，未把缓存工程速度收益当作LoRA精度收益。')
    add('旧主监督器记录了这次有意中断，最终退出码为1；没有删除该失败记录。恢复监督器分别完成剩余因果流与UNIFORM完整计时。后续compute_integrity.json逐项验证3个因果组、9个预算任务、3个完整计时任务，共15个完成单元；compute_complete.json来自该验证，不把旧监督器伪装成exit0。')
    add('本轮官方完整C4结果：')
    for method in ('MINUS','MINUS_RECALC_ONLY','SLOW_REPLAY_MATCHED'):
        r=s2['metrics'][method];v=r['metrics']
        add(f"  {method}: HOTA={r['HOTA_unrounded']:.9f}, DetA={v['DetA']}, AssA={v['AssA']}, IDF1={v['IDF1']}, updates={s2['actual_optimizer_updates'][method]}, recalc_events={s2['recalc_events'][method]}")
    add('两个关键增量：')
    for name,r in [('重算与再拟合：RECALC_ONLY−MINUS',recalc),('匹配重算后的权重分支：LoRA−RECALC_ONLY',weight)]:
        add(f"  {name}: ΔHOTA={r['delta_HOTA']:.9f}, 视频成对bootstrap95%区间={r['paired_video_CI95']}, 改善视频={r['improved_videos']}, 下降视频={r['declined_videos']}")
    add(f"联合分支LoRA−MINUS：ΔHOTA={total['delta_HOTA']:.9f}，95%区间={total['paired_video_CI95']}。联合增益的区间不跨0，也不能替代权重与重算分别归因的检验。")
    add('CountMatched使用另一组已经预测出来的逐查询/逐帧数量，在相同候选中按真实模型margin排序取top-k。没有使用GT数量，也没有用教师posterior冒充神经特征排序。四个组合全部逐帧验证数量相等。')
    add('即便“自身rank＋自身count”也不必复现主方法HOTA：主方法按融合后的后验信念选轨迹，这个诊断则强制按神经margin取top-k。两者分数定义不同；诊断专门回答固定输出数量时神经排序是否改善。')
    for method in ('FROZEN_RANK__RECALC_COUNT','LORA_RANK__RECALC_COUNT','FROZEN_RANK__LORA_COUNT','LORA_RANK__LORA_COUNT'):
        add(f"  {method}: HOTA={s2['metrics'][method]['HOTA_unrounded']:.9f}")
    add(f"固定RECALC数量时的排序差：ΔHOTA={cm['delta_HOTA']:.9f}, 视频bootstrap95%区间={cm['paired_video_CI95']}。")
    add(f"固定LoRA数量时的排序差：ΔHOTA={cm_lora['delta_HOTA']:.9f}, 视频bootstrap95%区间={cm_lora['paired_video_CI95']}。两种固定数量口径都没有给出稳定的正排序增益。")
    add('本轮支持固定问答和匹配重算条件下的权重分支正贡献。' if positive else '本轮没有建立固定问答和匹配重算条件下的LoRA独立正贡献；不能保留“旧REPLAY涨点就证明LoRA学到了语义”的说法。')
    add('CountMatched进一步支持这些输出中模型排序的改善。' if semantic else 'CountMatched没有建立相应的稳定正排序增益；不能把整体HOTA变化直接改写成纯语义特征学习。')
    if cm['paired_video_CI95'][1]<0:
        add('在固定重算组数量的诊断中，排序差的整个95%区间都低于0；这是反对本配置“改善了跨域语义排序”叙事的证据。该结论并不等于证明所有LoRA配置永远无用。')
    add('这里的权重差包含权重变化带来的下游再快拟合反应；它不是冻结所有校准参数后的孤立神经元实验。bootstrap按视频聚类、10000次、seed0，反映固定已拟合流下的评测视频不确定性，不含重新训练、改变视频顺序和训练seed的变动。单一固定视频流也不足以支持所有跨域语义的一般性结论。')
    add('C4在旧v8P阶段已经用于开发与分析，本轮是既有方向的机制审计，不能包装成第一次独立盲测。当前新生成过程不读取GT，与整个研究历史从未接触该验证集GT是不同的陈述。')
    add('归因组进程会等待其他组生成canonical输入或匹配重算调度，其总wall包含同步等待，不能拿来代表部署速度。真实自足的冷启动成本只取第三阶段。')
    for method,r in s2['costs'].items():
        add(f"  归因研究账本 {method}: process_wall={r['process_wall_seconds']:.6f}s, LoRA_phase_wall={r['LoRA_wall_seconds']:.6f}s, initial_plus_recalc_wall={r['initial_and_recalc_wall_seconds']:.6f}s, matched_schedule_wait={r['matched_schedule_wait_seconds']:.6f}s, new_teacher_calls={r['new_teacher_calls']}, replayed_answers={r['replayed_answers']}。总进程wall保留中断尝试；有效视频相位小计不冒充全部失败成本。")
    for method,r in s2['mechanism_diagnostics'].items():
        add(f"  {method}机制诊断：45视频中快分支w=0的个数={r['zero_slope_videos']}，身份MAP轮数={r['fast_identity_MAP_rounds']}，实际优化正/负样本={r['optimized_positive_samples']}/{r['optimized_negative_samples']}，AMP回退记录={r['AMP_retry_records']}。")
    for capacity,r in s2.get('frozen_cache_runtime',{}).get('SLOW_REPLAY_MATCHED',{}).items():
        add(f"  精确冻结缓存{capacity}MiB：有效视频{r['videos']}、优化器步{r['optimizer_steps']}、慢更新相位总wall={r['slow_update_phase_wall_seconds']:.6f}s、每步相位wall={r['slow_phase_wall_seconds_per_optimizer_step']}s。这是缓存/数据准备条件下的实际更新相位，不是纯GPU内核时间。")
    add('A800历史参考（不作为匹配A100因果组）：')
    for method,r in s2['historical_references'].items():add(f"  {method}: HOTA={r['HOTA_unrounded']:.9f}")
    add(f"A100纯快分支与A800原预测逐字节相同：{s2['MINUS_original_prediction_bytes_equal']}；差异序列={s2['MINUS_A100_vs_A800_different_sequences']}。")
    add('四、阶段三：真正按时间计费')
    add('按用户选择，先固定排序最前两个C4视频各前128个真实观测帧及全部查询，600秒包含冷进程启动和实际推理；完整45视频另测总成本。短切片选择不看GT，GT仅在预测封签后的独立进程中裁剪和评估。所有未处理样本保留为空预测，不从评测分母移除。')
    add('每臂一张A100 40GB，学生和教师共享该卡，三个新进程重复；不用旧教师缓存。模型必须实际从NAS加载。冷启动指模型和进程冷启动，没有清空共享服务器OS页缓存。GPU0/1先跑有顺序依赖的因果流，GPU2/3跑预算任务，空闲后复用。实际物理卡和并发时序均在监督收据。NAS为共享存储，曲线是本次并行负载下的实际成本，不能冒充独占NAS的无噪声速度或全局最优证明。')
    add('实际采用空闲卡队列调度，未实现完整随机交叉设计；Uniform三次短切片都在物理GPU2，完整运行在GPU0。三次短切片和单次完整运行的加载时长也有区别，比较时保留这类实际条件差异，不能把不同范围的600秒点视为同一次启动的直接重复。')
    add('MINUS保留四轮主动信息增益、路由、重查和证据快拟合；YT_PRIOR按学生均分排序并保留未问Native回退，教师阈值0.4；UNIFORM_K在同一名义总问题上限内不放回均匀抽取合格query-track对，在轨迹中点按查询类型路由视图，阈值0.5，未问保持UNKNOWN并省略。每个查询实际抽中数量可不同。纯外观UNIFORM不需要学生，未给它强加无用计算；遇到motion gate才计算必要学生特征。')
    add('为提供可提交的时间前缀，主动轮次内每次最多8个问题返回后写快照；该轮候选问题池保持固定，正式快拟合只在轮末执行。三个策略还包含不同的解码和未问回退规则，所以这里衡量完整已定义策略的效用；不能把系统间全部精度差归因给“主动选题”一个模块。')
    add('短切片名义总问答上限为rho=1，完整C4为rho=.3；时间预算是上限，不延长原方法定义去无限重复提问。因此比较的是这三个已定义策略在相同墙钟上限下的轨迹，不能由此证明所有可能计算分配中的数学最优。')
    add('每次输出先压缩写出完整候选mask并落盘，再由父进程观测提交完成时间。600秒后未返回的Qwen调用仍计费，答案不计入预测；父进程到时终止未完成工作。相位BEGIN/END日志能识别正在加载或在途推理，实际结束和截止清理开销单列。GPU秒在这里是占用设备的真实进程墙钟，不是纯核函数积分。')
    add('600秒短切片结果（均值与三次范围，不是置信区间）：')
    add('三次均固定算法seed0，重复的是新进程启动与运行条件；它们不是三个独立训练seed，也不是对所有随机选题路径的稳定性证明。')
    final=[]
    for method in ('MINUS_v8P','YT_PRIOR','UNIFORM_K'):
        rows=[r for r in s3['rows'] if r['scope']=='short' and r['method']==method and r['budget_seconds']==600]
        vals=[r['HOTA'] for r in rows];mean=sum(vals)/len(vals);final.append((mean,method))
        add(f"  {method}: meanHOTA={mean:.9f}, range=[{min(vals):.9f},{max(vals):.9f}], 已提交调用={[r['teacher_calls'] for r in rows]}, GPUs={[r['physical_GPU'] for r in rows]}")
    best=max(x[0] for x in final);winners=[method for mean,method in final if abs(mean-best)<1e-9]
    add('本轮600秒最高均值策略（允许并列）：'+', '.join(winners)+'。该结论仅指上述固定切片和真实启动条件。')
    active_calls=[r['teacher_calls'] for r in s3['rows'] if r['scope']=='short' and r['method']=='MINUS_v8P' and r['budget_seconds']==600]
    if not any(active_calls):add('关键限制：三次600秒主动策略均尚未提交教师反馈。因此这些点主要衡量冷启动前缀和初始解码，不能拿来证明主动选题本身更有效。选题进入工作状态后的成本效用另看完整C4补充曲线。')
    add('完整C4实际总成本与精度（每法一次，不放进600秒短切片曲线）：')
    for r in s3['rows']:
        if r['scope']=='full':add(f"  {r['method']}: wall={r['total_process_wall_seconds']:.6f}s ({r['total_process_wall_seconds']/3600:.6f}h), HOTA={r['HOTA']:.9f}, submitted_calls={r['teacher_calls']}, GPU={r['physical_GPU']}")
    full_rows=[r for r in s3['rows'] if r['scope']=='full']
    active=next(r for r in full_rows if r['method']=='MINUS_v8P')
    uniform=next(r for r in full_rows if r['method']=='UNIFORM_K')
    add(f"主动完整运行耗时是Uniform的{active['total_process_wall_seconds']/uniform['total_process_wall_seconds']:.6f}倍，终点HOTA差为{active['HOTA']-uniform['HOTA']:.9f}。这组数字描述本次固定调用上限下的精度与耗时取舍。")
    dominating=[r['method'] for r in full_rows if r['method']!='MINUS_v8P' and r['HOTA']>=active['HOTA'] and r['total_process_wall_seconds']<=active['total_process_wall_seconds'] and (r['HOTA']>active['HOTA'] or r['total_process_wall_seconds']<active['total_process_wall_seconds'])]
    if dominating:add('在这三种策略的本次完整运行中，主动策略同时在精度与时间上受到以下策略支配：'+', '.join(dominating)+'。本轮不能声称主动统筹实际成本最优。')
    else:add('在这三种策略的本次完整运行中，主动策略属于精度—耗时非支配集合。它仍不是所有计算分配下的全局最优证明；完整C4每方法仅一次，且NAS并发条件并非严格配对。')
    add('阶段二的MINUS采用固定历史回答；阶段三的MINUS_v8P重新执行Qwen前向，回答来自本次独立教师缓存。两个结果用于不同问题，不将其精度差当成单个变量的因果贡献。')
    add('各次短切片完整成本账本（秒；相位不重叠，other包括导入、CPU准备、解码、快照和未单列工作）：')
    for repeat in range(3):
        for method in ('MINUS_v8P','YT_PRIOR','UNIFORM_K'):
            folder=OUTPUT/'stage3/short'/f'repeat_{repeat}'/method
            ledger=load(folder/'cost_ledger.json')
            add(f"  repeat={repeat}, {method}: billed={ledger['accounted_wall_seconds']:.6f}, actual={ledger['actual_total_process_wall_seconds']:.6f}, stop_overhead={ledger['deadline_stop_overhead_seconds']:.6f}, returned_requests={ledger['returned_requests']}, completed_new_calls={ledger['completed_new_teacher_calls']}, cache_hits={ledger['completed_cache_hits']}, attempted_requests={ledger['attempted_logical_requests']}")
            add('    phases='+json.dumps(ledger['phase_wall_seconds'],ensure_ascii=False)+'; other='+str(ledger['import_setup_snapshot_and_other_seconds'])+'; unfinished='+json.dumps(ledger['unfinished_phases'],ensure_ascii=False))
    add('完整C4成本相位（秒；包含模型冷启动、实际服务与最终封签，GT评测另行执行）：')
    for method in ('MINUS_v8P','YT_PRIOR','UNIFORM_K'):
        ledger=load(OUTPUT/'stage3/full/repeat_0'/method/'cost_ledger.json')
        add(f"  {method}: phases="+json.dumps(ledger['phase_wall_seconds'],ensure_ascii=False)+f"; other={ledger['import_setup_snapshot_and_other_seconds']:.6f}; actual={ledger['actual_total_process_wall_seconds']:.6f}; new_teacher_calls={ledger['completed_new_teacher_calls']}; cache_hits={ledger['completed_cache_hits']}; backend_forward_wall={ledger['synchronized_backend_forward_wall_seconds']:.6f}")
    add('该端到端范围从原始图像、真实文本和固定官方跟踪候选开始，包含TTA系统实际需要的模型与特征、教师、拟合、写出；不含已经作为共享输入的上游检测/跟踪资产生成，也不含事后GT评估时间。不能称为从裸视频开始重跑检测器/关联器的全部成本。')
    add('完整C4补充曲线使用预先封存的600秒倍增网格以及全部实际完成时刻，直接评估已经提交的完整C4预测前缀，未新增GPU实验、未按GT挑时间点；未处理样本仍保留。主600秒短切片结果不因此改变。')
    add('图中的点是实际评估预算，阶梯线仅保持上一测量点；预算点之间仍有更多原始提交记录，不把这条稀疏展示线称为连续精确测量。完整C4固定倍增网格HOTA如下：')
    curve_rows=load(OUTPUT/'stage3/full_curve_analysis.json')['rows']
    for budget in (600,1200,2400,4800,9600,19200):
        at=[r for r in curve_rows if r['budget_seconds']==budget]
        add(f"  budget={budget}s: "+'; '.join(f"{r['method']}={r['HOTA']:.9f}" for r in at))
    add('五、当前形成的方法与下一步判断')
    add('已经形成可运行的A100真实数据流、三个固定证据归因组、四个CountMatched组合、带在途任务账本的严格时间截止、完整C4成本收据和可导出的PNG/PDF图。当前结论必须由上面的数值约束，不能把工程重算和神经学习合并讲成一种贡献，也不能用教师几秒前向去掩盖数百秒冷加载。')
    add('这轮判断经历三次收紧：先用真实GPU请求检验恢复环境；随后加入零权重同重算和固定数量排序，拆开联合涨点；最后把冷启动、初始学生特征和在途工作计入同一时间账本。结果保留了可运行的快慢框架，也要求收缩此前对语义学习和部署最优性的解释。工程打通与科研假设得到支持分别验收，出现负结果后继续完成报告，而不改门槛或挑选更好看的子集。')
    add('当前应分别保留观察到的工程、数量和先验效应，并验证其稳定性；不要因点估计为正就写成已经证明。慢分支需要重新检查监督质量、排序目标、适配位置、梯度与源数量头的相互作用，再用独立预注册验证；当前GT不能反过来用于挑参。若冷启动下主动策略不是最好，应如实报告。常驻模型、共享编码器或NAS加载优化属于另一个部署协议，不改写本轮600秒结果。')
    add('后续建议先做监督审计：分别检查DIRECT_POS、DIRECT_NEG、EXCLUSIVE_NEG和ABSENT_NEG的真实覆盖及冲突，区分教师回答有误、正负采样目标与排序任务不一致、以及数量头对分数尺度变化敏感。已有GT可以做注明为事后诊断的误差分析；不能把按这批GT挑出的最好配置再称作本轮独立验证。')
    add('慢分支下一轮若要继续，应先在独立开发数据上明确排序目标或源模型保留约束，再封存新的验证协议；继续保留固定问答、零LoRA匹配重算和CountMatched四组合。验收应同时关注LoRA−重算的区间、固定数量的排序差、样本覆盖及视频流/训练seed重复，而非只看训练loss下降或联合HOTA上涨。当前实测没有授权写成LoRA一般无用，也没有授权写成跨域语义学习已成立。')
    add('工程下一轮首先针对原始图像读取与冻结视觉特征成本，而非只减少教师调用。精确缓存的收益必须单列；如果采用常驻教师、共享视觉特征或预热资产，应明确摊销周期、首请求延迟和新增部署条件，再另报温启动预算曲线，同时保留这次冷启动曲线。用已经加载好的模型开始计时，不能替代用户这次要求的全部启动包含在600秒内。')
    add('六、复现入口与交接')
    add('本轮既有4个回归文件共28项通过，221.19秒。首次运行的旧合成测试默认向NAS历史目录写临时结果，Errno30阻止写入；13项通过、1项失败后中断剩余合成案例。设置V8_SYNTHETIC_RESULT_ROOT到本轮本地verification/synthetic，再完整重跑同4文件，没有排除案例或修改指标/seed/通过条件。初次失败与最终通过收据都保留。')
    add('75个主评预算点还逐项核对父进程实际观测的提交时间、纳入的全部查询及官方指标；12个成本账本核对相位＋other＝计费时长、计费＋停止清理＝实际进程总wall。结果见verification/budget_receipt_audit.json。')
    add('  source scripts/env_a100.sh\n  export TTA_RMOT_A100_RUN_ROOT=/data3/liuyeqiang/FlexHook-TTA/a100_runs/NEW_EXPERIMENT_ID\n  .venv/bin/python scripts/a100_stage1.py\n  .venv/bin/python scripts/a100_causal.py --prepare\n  .venv/bin/python scripts/a100_wallclock.py --prepare\n  .venv/bin/python scripts/a100_supervisor.py\n  .venv/bin/python scripts/a100_evaluate.py --phase all\n  .venv/bin/python tools/finalize_a100_three_stage.py')
    add('入口监督器会跳过已完成任务。不要在原阶段一cache上把旧回答当成新的冷测；复现实验需指定全新的运行目录或保留当前目录后另建版本。不要启动旧启动器或覆盖NAS封存结果。当前新运行根目录：'+str(OUTPUT))
    add('轻量证据目录：'+str(dst)+'；大型预测/状态/原始回答仅在本地a100_runs，来源权重与数据继续留NAS。发布前需运行tools/check_publication_size.py，确认当前Git分支快照及全部可达对象均小于100,000,000字节。')
    add('轻量目录还保存官方逐视频HOTA精确汇总所需的分子/分母数组，可以在不读取GT和完整预测的情况下复算总体HOTA及本轮视频成对bootstrap。原始逐表达官方明细CSV仍留本地，校验值保存在对应official_completion.json。')
    add('来源代码和软件依据：项目官方FlexHook vendor、固定YesTrack MIT上游函数及Qwen3-VL官方README（https://github.com/QwenLM/Qwen3-VL/blob/main/README.md）；实际包版本详见environment.json。本文所有新性能数字均取本次A100原始收据与官方TrackEval，未用历史A800速度代替。')
    add('附录A：短切片全部预注册预算点（HOTA为百分数，未处理部分仍在分母）')
    add('repeat | method | budget_s | HOTA | submitted_teacher_calls | prediction_rows | physical_GPU')
    for r in s3['rows']:
        if r['scope']=='short':
            add(f"{r['repeat']} | {r['method']} | {r['budget_seconds']} | {r['HOTA']:.9f} | {r['teacher_calls']} | {r['prediction_rows']} | {r['physical_GPU']}")
    add('附录B：完整C4补充时间曲线（完成后以原方法输出保持平台，不额外增加问答）')
    add('method | budget_s | full_C4_HOTA | submitted_teacher_calls | task_finished')
    full=load(OUTPUT/'stage3/full_curve_analysis.json')
    for r in full['rows']:
        add(f"{r['method']} | {r['budget_seconds']:.6f} | {r['HOTA']:.9f} | {r['teacher_calls']} | {r['task_finished_by_budget']}")
    add('附录C：逐视频因果诊断（完整视频为聚类单位，不按单帧独立假设放大显著性）')
    add('video | LoRA_minus_recalc_HOTA | LoRA_rank_at_recalc_count_minus_frozen_rank_HOTA')
    ranked={r['video']:r['delta_HOTA'] for r in cm['per_video']}
    for r in weight['per_video']:
        add(f"{r['video']} | {r['delta_HOTA']:.9f} | {ranked[r['video']]:.9f}")
    report=ROOT/'docs/A100_三阶段实验完整报告_20261120.txt';report.write_text('\n'.join(lines).rstrip()+'\n')
    history=ROOT/'docs/TTA_RMOT_研究心路历程与完整交接_20261120.txt'
    combined=ROOT/'docs/TTA_RMOT_完整历程与A100三阶段实证_20261120.txt'
    combined.write_text('阅读说明：第一部分是A100恢复时保存的完整研究历程快照；第二部分追加本轮三阶段新实证，最新状态以第二部分为准。\n\n'+history.read_text()+'\n\n'+'='*76+'\n本轮A100三阶段实证追加\n'+'='*76+'\n\n'+report.read_text())
    print(json.dumps(dict(status='FINAL_EVIDENCE_SAVED',report=str(report),complete_history=str(combined),published_bytes=sum(p.stat().st_size for p in dst.rglob('*') if p.is_file())),ensure_ascii=False))


if __name__=='__main__':main()
