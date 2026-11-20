"""Rebuild the Chinese TXT report and small scientific figures from receipts."""
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_uda.a100.mechanism import OUT
from flex_tta.runtime.pf_receipts import atomic_json, file_sha

PUBLIC=ROOT/'docs/a100_mechanism_20261121'
REPORT=ROOT/'docs/TTA_RMOT_视觉本能与经典算法_机制诊断完整报告_20261121.txt'


def read(relative):
    return json.loads((OUT/relative).read_text())


def number(value,digits=4):
    return '不适用' if value is None else f'{value:.{digits}f}'


def figures(official,bank,profile,audits,coverage,temporal):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
    destination=PUBLIC/'figures';destination.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.savefig(destination/(name+'.png'),bbox_inches='tight')
        fig.savefig(destination/(name+'.pdf'),bbox_inches='tight')
        plt.close(fig)
    methods=official['methods']
    fig,ax=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    names=['MINUS','TRACK_MAJORITY_ORACLE','FRAME_ORACLE']
    bars=ax[0].bar(range(3),[methods[n]['HOTA'] for n in names],color=['#3875a5','#e4a649','#8e616b'])
    ax[0].set_xticks(range(3),['Frozen fast','Track-majority\noracle','Frame oracle'])
    ax[0].set_ylabel('HOTA');ax[0].set_title('C4: fixed candidate boxes and IDs')
    ax[0].bar_label(bars,fmt='%.2f',padding=3);ax[0].set_ylim(0,100)
    c=audits['C4']['counts'];v=coverage['pairs']['C4'];k=v['counts']
    values=[100*c['candidate_retrievable_recall'],100*c['v8p_matched_GT']/c['GT_observations'],
        100*v['positive_track_coverage'],100*v['missed_positive_unasked_fraction']]
    bars=ax[1].barh(range(4),values,color=['#6e9f76','#3875a5','#e4a649','#b96d65'])
    ax[1].set_yticks(range(4),['GT retrievable\n(IoU >= .5)','GT matched by\nfinal output','Positive observations\non queried tracks','Missed positive observations\non unqueried tracks'])
    ax[1].invert_yaxis();ax[1].set_xlim(0,112);ax[1].bar_label(bars,fmt='%.1f%%',padding=3)
    ax[1].set_title('C4: coverage and selection diagnosis')
    ax[1].set_xlabel('Percent (different denominators; not additive)')
    save(fig,'coverage_and_oracles')
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True,constrained_layout=True)
    core=['CROP_FIXED','ROUTED_FIXED','CROP_DS','ROUTED_DS']
    for ax,pair in zip(axes,('C1','C4','C5')):
        values=bank['pairs'][pair]['splits']['evaluation']
        bars=ax.bar(range(4),[values['methods'][m]['Brier'] for m in core],color=['#8299ac','#3875a5','#e3c28f','#d6983b'])
        ax.set_xticks(range(4),['Crop\nfixed','Routed\nfixed','Crop\nDS','Routed\nDS'])
        ax.set_title(f'{pair}: n={values["samples"]}, parents={values["original_parent_groups"]}')
        ax.bar_label(bars,fmt='%.3f',padding=3)
    axes[0].set_ylabel('Brier score (lower is better)')
    fig.suptitle('Held-out parent groups in existing development data; one answer per test item')
    save(fig,'paired_view_bank')
    fig,ax=plt.subplots(1,2,figsize=(11.5,4.5),constrained_layout=True)
    arms=['FIXED_BANK_CROP','FIXED_BANK_ORIGINAL','DS_BANK_CROP','DS_BANK_ORIGINAL']
    bars=ax[0].bar(range(4),[methods[n]['HOTA'] for n in arms],color=['#8299ac','#3875a5','#e3c28f','#d6983b'])
    ax[0].set_xticks(range(4),['Crop\nfixed','Original route\nfixed','Crop\nDS','Original route\nDS'])
    ax[0].bar_label(bars,fmt='%.3f',padding=3);ax[0].set_ylabel('HOTA')
    ax[0].axhline(methods['MINUS']['HOTA'],color='#777777',ls='--',lw=1,label='Original MINUS')
    ax[0].legend(fontsize=8);ax[0].set_title('C4:3508 fixed QA\nView intervention:100 slots; DS across full stream')
    pairs=[('DS_BANK_ORIGINAL_vs_FIXED_BANK_ORIGINAL','DS vs fixed\n(original route)'),
        ('DS_BANK_ORIGINAL_vs_DS_BANK_CROP','Original route vs crop\n(DS)'),
        ('FIXED_BANK_ORIGINAL_vs_FIXED_BANK_CROP','Original route vs crop\n(fixed)')]
    for i,(key,label) in enumerate(pairs):
        item=official['comparisons'][key];lo,hi=item['original_parent_video']['CI95'];delta=item['delta_HOTA']
        ax[1].plot([lo,hi],[i,i],color='#3875a5',lw=2);ax[1].plot(delta,i,'o',color='#d6983b')
    ax[1].set_yticks(range(len(pairs)),[label for _,label in pairs]);ax[1].invert_yaxis();ax[1].axvline(0,color='#777777',ls='--')
    ax[1].set_xlabel('Delta HOTA and 95% interval');ax[1].set_title('Paired bootstrap: 13 original parent videos')
    save(fig,'full_c4_controls')
    fig,ax=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    names=['RAW_MARGIN','EMA_MARGIN','HYSTERESIS_MARGIN','MINUS']
    labels=['Raw margin','EMA alpha=.2','Hysteresis .6/.4','Existing MINUS']
    bars=ax[0].bar(range(4),[methods[n]['HOTA'] for n in names],color=['#8299ac','#6e9f76','#e4a649','#3875a5'])
    ax[0].set_xticks(range(4),labels,rotation=12);ax[0].bar_label(bars,fmt='%.2f',padding=3)
    ax[0].set_ylabel('HOTA');ax[0].set_title('Teacher-free raw filters vs existing fast method')
    values=[100*temporal['methods'][n]['stable_identity_membership_flips']/max(1,temporal['methods'][n]['stable_identity_membership_transitions']) for n in names]
    bars=ax[1].bar(range(4),values,color=['#8299ac','#6e9f76','#e4a649','#3875a5'])
    ax[1].set_xticks(range(4),labels,rotation=12);ax[1].bar_label(bars,fmt='%.3f',padding=3)
    ax[1].set_ylabel('Output flips per 100 stable transitions')
    ax[1].set_title('Less flicker does not by itself prove better semantics')
    save(fig,'temporal_diagnosis')
    fig,ax=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
    keys=['image_IO_and_transform','backbone','PCD_decode','text_cache_and_encoder','CPU_cardinality','other_seconds']
    labels=['Image IO/transform','Backbone','PCD','Text/cache','Cardinality','Other']
    colors=['#3875a5','#6e9f76','#d6983b','#916fa1','#bd7773','#a5a5a5']
    positions=np.arange(2);bottom=np.zeros(2)
    for key,label,color in zip(keys,labels,colors):
        vals=np.array([r.get(key,r['phase_wall_seconds'].get(key,0)) for r in profile['records']])
        ax[0].bar(positions,vals,bottom=bottom,label=label,color=color);bottom+=vals
    ax[0].set_xticks(positions,['512 MiB','8192 MiB']);ax[0].set_ylabel('Synchronized elapsed seconds')
    ax[0].set_title('Fixed 16-anchor feature flow, model resident')
    ax[0].legend(fontsize=8,loc='upper left',bbox_to_anchor=(1,1))
    bars=ax[1].bar(['Weight load +\nverification','Both feature\npasses'],[profile['student_load_wall_seconds'],sum(r['feature_flow_wall_seconds'] for r in profile['records'])],color=['#b96d65','#3875a5'])
    ax[1].bar_label(bars,fmt='%.1f',padding=3);ax[1].set_ylabel('Elapsed seconds')
    ax[1].set_title('One cold profile process, shared read-only NAS')
    fig.suptitle('All four output arrays bitwise equal; 62 backbone images per capacity; no cache-capacity speedup')
    save(fig,'feature_profile')
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,pair in zip(axes,('C1','C4','C5')):
        rows=[r for r in bank['acquisition'] if r['pair']==pair]
        for method,style in zip(('UNIFORM','INFORMATION_GAIN','UNCERTAINTY','UNCERTAINTY_EXPLORE'),('o-','s--','^-','d:')):
            selected=[r for r in rows if r['method']==method]
            budgets=sorted({r['teacher_calls'] for r in selected})
            vals=[[r['metrics']['Brier'] for r in selected if r['teacher_calls']==b] for b in budgets]
            ax.plot(budgets,[np.mean(v) for v in vals],style,label=method)
            ax.fill_between(budgets,[np.min(v) for v in vals],[np.max(v) for v in vals],alpha=.10)
        ax.set_title(pair);ax.set_xlabel('Answers consulted from prepaid evaluation bank')
    axes[0].set_ylabel('Brier score (lower is better)');axes[-1].legend(fontsize=7)
    fig.suptitle('Historical prepaid pool; shading=min-max over 3 tie/exploration seeds\nNo live cost or full-candidate coverage claim')
    save(fig,'finite_bank_acquisition')


def main():
    assert read('cpu_supervisor/completion.json')['status']=='ALL_REGISTERED_CPU_CONTROLS_COMPLETE'
    if not (OUT/'analysis/study_cost_ledger.json').is_file():
        from scripts.a100_mechanism_analyze import costs
        costs()
    official=read('analysis/official_results.json');bank=read('analysis/bank_results.json')
    profile=read('profile/completion.json');coverage=read('analysis/coverage.json')
    temporal=read('analysis/temporal_details.json');absence=read('analysis/absence.json')
    audits={p:read(f'audit/{p}/completion.json') for p in ('C1','C4','C5')}
    evidence=read('evidence/completion.json');registration=read('preregistration.json')
    PUBLIC.mkdir(parents=True,exist_ok=True)
    files=['preregistration.json','evidence_manifest.json','user_scope_amendment.json','route_definition_amendment.json',
        'full_replay_intervention_registration.json','teacher_label_definition.json','initial_audit_failure.json',
        'protocol/batch_invariance.json','protocol/contract_invariance.json','profile/completion.json',
        'profile/supervisor_receipt.json','evidence/completion.json','evidence/supervisor_receipt.json',
        'evidence/attempt00_failed_supervisor_receipt.json','evidence/image_path_repair.json',
        'fixed_replay/C4/reference_equivalence.json','analysis/coverage.json','analysis/absence.json',
        'fixed_replay/C4/new_original_view_equivalence.json',
        'analysis/bank_results.json','analysis/official_results.json','analysis/official_sufficient_statistics.json',
        'analysis/prior_parent_sensitivity.json',
        'analysis/study_cost_ledger.json',
        'analysis/sampling_limitations.json',
        'validation_receipt.json',
        'analysis/temporal_details.json','cpu_supervisor/completion.json']
    for relative in files:
        source=OUT/relative;dest=PUBLIC/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
    for pair,data in audits.items():
        compact={k:v for k,v in data.items() if k not in ('episodes','videos')}
        atomic_json(PUBLIC/f'audit/{pair}/completion.json',compact)
    # Keep every completed CPU cost receipt, without raw outputs/logs.
    for source in sorted((OUT/'cpu_supervisor').glob('*.json')):
        dest=PUBLIC/'cpu_supervisor'/source.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    #300 decisions are enough to recompute point metrics;900 raw teacher rows
    # and the SQLite cache stay local. Truth attachment is explicitly offline.
    atomic_json(PUBLIC/'analysis/bank_labeled_rows.json',read('analysis/bank_labeled_rows.json'))
    figures(official,bank,profile,audits,coverage,temporal)
    methods=official['methods'];comparisons=official['comparisons'];lines=[]
    def add(text=''):lines.append(text)
    add('TTA-RMOT：人类视觉本能与经典算法——A100机制诊断完整报告')
    add('日期：2026-11-21。研究基线：f915e49490bcddd7661d9650cca373e1c453a172。')
    add('原始结果：/data3/liuyeqiang/FlexHook-TTA/a100_runs/mechanism_20261121')
    add('本报告为已有开发数据的机制诊断；用户明确目前没有独立数据，本轮不做独立盲测。')
    add('')
    add('先看已经测清的结果')
    add(f'• C4框候选可恢复约98.193%的GT观测；原快分支HOTA={number(methods["MINUS"]["HOTA"],6)}，GT语义逐帧参照={number(methods["FRAME_ORACLE"]["HOTA"],6)}。语义选择仍有大空间，物理关联也不完美。')
    add('• C4有119/466查询从未问到正确候选；122条真实有目标却零输出的查询里82条属于这种漏查。覆盖比盲加网络更新更值得优先研究，但有限历史池尚未证明新的覆盖策略。')
    add(f'• 已有最终v8P稳定片段翻转0次；无教师RAW→EMA仅{number(methods["RAW_MARGIN"]["HOTA"],4)}→{number(methods["EMA_MARGIN"]["HOTA"],4)}。不能由原分数抖动推出现有快分支缺少记忆。')
    add(f'• C4强制清单式后HOTA={number(methods["UNIFIED_CHECKLIST"]["HOTA"],6)}，错误拒绝{methods["UNIFIED_CHECKLIST"]["false_absence_query_count"]}条；Dance实际清单式则错拒11/40条有目标查询。存在性推断既能删空，也会错杀。')
    for key,label in (('DS_BANK_ORIGINAL_vs_FIXED_BANK_ORIGINAL','完整C4原视角下DS−固定可靠性'),('DS_BANK_ORIGINAL_vs_DS_BANK_CROP','完整C4的原路由−CROP（DS）')):
        r=comparisons[key];add(f'• {label}: ΔHOTA={number(r["delta_HOTA"],6)}，13原视频95%区间={r["original_parent_video"]["CI95"]}。视角干预仅100/3508回答位置。')
    for pair,p in bank['pairs'].items():
        mm=p['splits']['evaluation']['methods'];add(f'• {pair}配对点诊断Brier（越低越好）：CROP固定={number(mm["CROP_FIXED"]["Brier"],5)}，原路由固定={number(mm["ROUTED_FIXED"]["Brier"],5)}，CROP+DS={number(mm["CROP_DS"]["Brier"],5)}，原路由+DS={number(mm["ROUTED_DS"]["Brier"],5)}。')
    add('• 精确缓存数值一致，但16锚帧本就能装进512MiB，扩大到8192MiB未获提速；真实瓶颈指向IO与加载，不能声称已有全C4加速。')
    add('• Sports45片段实际13原视频；新主区间按原视频聚类，旧LoRA联合增益在此口径下也跨0。所有结论限已有开发数据。')
    add('')
    add('一、我们究竟想实现什么')
    add('最终目标仍是跨域RMOT：给一段视频和一句描述，在新场景中持续找对目标，并保持身份。人的视觉直觉在这里是可检验的设计启发：先知道任务是否保证有目标；按属性选择局部、时序或全景；有疑点再看；综合有偏差的证据；可靠的信息保留，不在稀疏反馈上盲目更新网络。经典算法负责把这些直觉变成明确的概率、预算和决策规则。')
    add('本轮目的不是证明每个模块都先进，而是查清当前成绩从哪里来、失分在哪里、下一步该改哪里。LoRA全程冻结，未运行旧66项、旧80队列或v9旧探针。')
    add('')
    add('二、本轮实际完成的实验及其边界')
    add('1. C1、C4完整候选/语义/物理ID/输出翻转审计；C5沿用原封存数值GT做成员与空输出诊断。')
    add('2. C4两个保留原框和原ID的离线oracle，明确含GT，不是可部署方法，也不是严格理论上界。')
    add('3. C1/C4/C5各100个历史真实查证位置的三视角冻结Qwen证据；所有位置、分组、参数在新证据GT分析前登记。')
    add('4. 单视角CROP/原路由 × 固定.8可靠性/无监督DS四组点诊断；补CLIP、FRAME_BOX、平均、投票等。DS只在校准原视频拟合，机制评估用不重叠的原视频。')
    add('5. C4四组完整固定问答回放：3508个回答位置不变，其中预注册100个位置替换为新配对回答；其他3408个回答一致。四组全部官评。原路由保留历史文本路由与后期信息增益选视角，不是只看关键词。')
    add('6. 原集合协议启发式与显式任务协议的分批不变性；C4统一CHECKLIST固定问答官评；三个方向的实际错误拒绝审计。')
    add('7. RAW/EMA/滞回三个无教师、无权重更新的因果时序对照，完整C4官评；不将它们说成给v8P添加了新记忆模块。')
    add('8. 有限证据库中的均匀/信息增益/不确定性/不确定性+探索，3个预注册随机种子；这是付过教师费用后的历史池诊断，不能替代真实全候选覆盖实验。')
    add('9. GPU1真实16锚帧精确缓存剖析，检查IO、backbone、文本、PCD、数量头与剩余耗时。GPU0采集证据；GPU2/3当时已有其他任务，没有打断。物理GPU未超出0/1/2/3。')
    add('NAS始终只读，历史results不改；新问答、SQLite、完整预测和日志留本地a100_runs，Git只放代码、TXT和精选收据。')
    add('')
    add('三、离线错误诊断：候选有没有、选没选对、轨迹可靠吗')
    for pair in ('C1','C4','C5'):
        a=audits[pair]['counts'];c=coverage['pairs'][pair];k=c['counts']
        add(f'{pair}: '+('候选可恢复GT比例='+number(100*a['candidate_retrievable_recall'],3)+'%；' if pair!='C5' else '原KITTI gt_template未备份，不能声称已检查原始GT身份或一对一候选覆盖上限；')+
            f'有目标查询{k["query_present"]}条，零输出{k["present_zero_output"]}条；有正确候选但从未问到正确轨迹{k["query_positive_candidate_but_no_positive_track_asked"]}条。')
        add(f'  正例候选观测中位于问过轨迹上的比例={number(100*c["positive_track_coverage"],3)}%；漏选正例中位于从未问过轨迹上的比例={number(100*c["missed_positive_unasked_fraction"],3)}%。有目标零输出且没问过正确轨迹={k["present_zero_and_no_positive_track_asked"]}条。')
        add('  计数详情：'+json.dumps(k,ensure_ascii=False,sort_keys=True))
        qs=c['queries'];add(f'  完全未查证的查询={sum(q["asked_tracks"]==0 for q in qs)}；查证过但从未问到正确轨迹={sum(q["query_positive_candidate_but_no_positive_track_asked"] and q["asked_tracks"]>0 for q in qs)}。')
        if pair!='C5':
            p=c['physical'];add(f'  物理轨迹匹配观测加权纯度={number(p["weighted_track_purity"],6)}；{p["candidate_tracks"]}条候选中，{p["tracks_with_multiple_visible_GT_ids"]}条曾对应多个可见GT身份；{p["GT_ids_observed"]}个匹配到的GT身份中，{p["GT_ids_with_multiple_candidate_IDs"]}个对应多条候选ID。')
            add(f'  每条物理轨迹只计一次的相邻可见身份变更代理={p["visible_identity_change_proxy"]}。这是IoU匹配代理，受遮挡/重叠/匹配影响，不等同官方IDSW。原审计中逐查询重复的变更次数不拿来当独立物理ID错误。')
    add('注意：GT观测可恢复率、选择错误、关联错误相互作用，不能强行拆成加总100%的HOTA失分占比。候选覆盖高不等于物理ID完美。候选标签上的漏选数与每个选择子集重新做最大匹配的漏检数可略不同，详见两份口径收据。')
    add('C4只有5条查询完全没问过；119条没问到正确轨迹中114条其实问过别的轨迹。因此每句至少问一次仍不够。RMOT一句话可能对应一整队或多名目标，确认其中一人存在也不等于把所有相关人找齐；覆盖问题必须在查询—候选轨迹与多目标数量层面检查。')
    add('')
    add('四、C4离线oracle与真实官评')
    for name in ('MINUS','TRACK_MAJORITY_ORACLE','FRAME_ORACLE'):
        r=methods[name];m=r['metrics'];add(f'{name}: HOTA={number(r["HOTA"],8)}, DetA={number(m["DetA"])}, AssA={number(m["AssA"])}, IDF1={number(m["IDF1"])}, IDSW={number(m["IDSW"],0)}, 输出行={r["prediction_rows"]}。')
    add('FRAME_ORACLE每帧按GT做IoU>=.5一对一候选选择；TRACK_MAJORITY_ORACLE只在正例匹配超过一半时保留整条候选轨迹。两者保留原框和原ID；多数规则并未求解使HOTA最大的组合，逐帧IoU=.5的oracle也不是HOTA全部阈值下的最优选择。它们是诊断参照，不是任何语义模块能够保证达到的分数。')
    add('')
    add('五、时序：有没有“看清一次却马上忘记”的证据')
    for pair in ('C1','C4'):
        c=audits[pair]['counts'];add(f'{pair}真实身份与语义成员稳定片段：原top-k翻转{c["topk_stable_identity_membership_flips"]}次，最终v8P翻转{c["v8p_stable_identity_membership_flips"]}次。')
    for name in ('RAW_MARGIN','EMA_MARGIN','HYSTERESIS_MARGIN','MINUS'):
        r=methods[name];t=temporal['methods'][name];add(f'{name}: HOTA={number(r["HOTA"],8)}, AssA={number(r["metrics"]["AssA"])}, 稳定片段翻转={t["stable_identity_membership_flips"]}/{t["stable_identity_membership_transitions"]}, 选择正例行={t["selected_positive_candidate_rows"]}, 选择负例行={t["selected_negative_candidate_rows"]}。')
    for key in ('EMA_MARGIN_vs_RAW_MARGIN','HYSTERESIS_MARGIN_vs_RAW_MARGIN'):
        r=comparisons[key];add(f'{key}: ΔHOTA={number(r["delta_HOTA"],8)}；13原视频95%区间={r["original_parent_video"]["CI95"]}。')
    add('结论边界：最终v8P在这些稳定片段已经没有输出翻转。它有整轨迹传播式记忆，但可能把错判断稳定地传播出去，或沿身份混杂的物理轨迹传播。不能因为原分数抖动就说给v8P再加EMA/HMM一定有用；无教师RAW平滑的任何收益只针对RAW这一基线。动态动作或真实身份变化需要分段，不能无条件冻结整轨迹语义。')
    add('')
    add('六、怎么看、信多少：三视角配对证据')
    add(f'教师实际返回{evidence["responses"]}个回答，对应{evidence["locations"]}个位置；新调用{evidence["teacher"]["new_calls"]}，缓存命中{evidence["teacher"]["cache_hits"]}；CLIP回退{evidence["clip_fallback_responses"]}次。冻结权重不变，生成GT打开0。')
    add('四组在同一个候选位置上只消费一个视角回答。ROUTED采用预先登记的原实际视角（含后期信息增益）；TEXT_ROUTE单列为文本规则诊断。DS在校准原视频上用三视角无标签回答拟合Beta(2,2)/20次EM；固定可靠性设(.8,.8)。两组共用来源先验。校准成本没有免费消失，部署时必须另计或摊销。')
    add('点诊断的Brier/准确率不是HOTA；不能拿一个点指标的提升宣称完整视频跟踪提高。三视角平均/累加消费三个回答，必须与同为三个回答的融合方法比较，不能与单视角假装等成本。')
    core=('CROP_FIXED','ROUTED_FIXED','CROP_DS','ROUTED_DS','CLIP_FIXED','FRAME_BOX_FIXED','TEXT_ROUTE_FIXED','ALL_MEAN','ALL_VOTE_FIXED','ALL_DS','SOURCE_PRIOR')
    for pair,p in bank['pairs'].items():
        s=p['splits']['evaluation'];add(f'{pair}机制评估：{s["samples"]}个位置，{s["original_parent_groups"]}个原视频；路由视角={s["routing_views"]}；查询类型={s["query_types"]}。')
        for method in core:
            r=s['methods'][method];add(f'  {method}: accuracy={number(r["accuracy"])}, balanced_accuracy={number(r["balanced_accuracy"])}, Brier={number(r["Brier"],6)}, NLL={number(r["NLL"],6)}, AUROC={number(r["AUROC"])}, TP/TN/FP/FN={r["TP"]}/{r["TN"]}/{r["FP"]}/{r["FN"]}。')
        for r in p['comparisons']:
            add(f'  {r["first"]}−{r["second"]}: Δaccuracy={number(r["delta_accuracy"])}, CI95={r["accuracy_CI95"]}; Brier降低={number(r["Brier_reduction"],6)}, CI95={r["Brier_reduction_CI95"]}。')
        for view,r in p['teacher'].items():
            add(f'  教师{view}: accuracy={number(r["accuracy"])}, sensitivity={number(r["sensitivity"])}, specificity={number(r["specificity"])}, Brier={number(r["Brier"],6)}；DS校准估计={r["unsupervised_calibration_estimate"]}。')
        for r in p['view_error_correlation']:
            add(f'  {r["views"]}错误相关={number(r["error_correlation"])}, 同错{r["both_wrong"]}, 错误指示一致率={number(r["error_agreement"])}。按GT正负条件的细分见JSON。')
        add(f'  标签定义：{p["numerical_GT_definition"]}；主标签与一对一敏感性标签不同的位置={p["primary_vs_one_to_one_label_disagreements"]}。')
        add('  阈值诊断：'+json.dumps(p['threshold_diagnostics'],ensure_ascii=False,sort_keys=True))
    add('所有视角来自同一个Qwen，并非相互独立的三名专家。DS的条件独立、样本覆盖和可识别性都是实质假设；无标签估计不等于真实正确率。教师误差按候选与GT的几何成员标签诊断，不是脱离定位问题的纯语义真值。原视频分组只隔离本轮DS拟合与评估，不把已参与开发的数据变成盲测。少于5个原视频的点诊断只报点数，不造置信区间。')
    add('本轮还暴露了抽样设计的局限：按(video,type,stratum)字典序轮转，并在100位置截断，不保证每个原视频等量或全部覆盖。预注册输入表中的13原视频是完整C4资产数，不是点诊断覆盖数；实际校准/评估原视频数以上面实测为准，C4评估仅3个、C5仅2个，均不报点诊断bootstrap区间。已看GT后不改样本追求更好结果；下一轮应先分配每个原视频名额，再在视频内分层。')
    add('C1的64个评估位置原路由全部选择CROP，因此CROP与ROUTED在这里是相同干预，零差不能证明路由有效或无效。C4评估仅6个位置原路由非CROP，其中仅2个位置的教师二值回答与CROP不同，路由比较的实质信息量很小。')
    add('C4点诊断中教师CROP原始判断准确率约85.25%，但叠加注册的来源先验与固定二值证据后只有29.51%；52/61位置先验logit已经在±2截断边界，单次受限反馈常跨不过决策阈值。这说明该简化点推断中的“先验—证据幅度—阈值”耦合需要检查，不能直接归咎教师看不懂，也不能等同完整v8P快校准已被同一实验修复。三视角组合的效果同时改变调用数和证据幅度，不能因此宣布增加视角必然必要；同预算单视角原始概率/软证据是下一轮应保留的简单对照。')
    add('DS点结果并不一致：C1概率误差Brier下降而准确率略降；C4/C5本批Brier变差；同三回答预算下，ALL_DS并未在三个方向的Brier上超过简单ALL_MEAN。准确率与概率质量可能不同向，不能只选有利指标。')
    add('')
    add('七、完整C4：固定3508问答的四组反事实')
    for name in ('FIXED_BANK_CROP','FIXED_BANK_ORIGINAL','DS_BANK_CROP','DS_BANK_ORIGINAL'):
        r=methods[name];add(f'{name}: HOTA={number(r["HOTA"],8)}, DetA={number(r["metrics"]["DetA"])}, AssA={number(r["metrics"]["AssA"])}, IDF1={number(r["metrics"]["IDF1"])}, 输出行={r["prediction_rows"]}, 有目标零输出={r["present_zero_output"]}。')
    for key in ('DS_BANK_ORIGINAL_vs_FIXED_BANK_ORIGINAL','DS_BANK_CROP_vs_FIXED_BANK_CROP',
                'DS_BANK_ORIGINAL_vs_DS_BANK_CROP','FIXED_BANK_ORIGINAL_vs_FIXED_BANK_CROP','DS_BANK_ORIGINAL_vs_MINUS'):
        r=comparisons[key];add(f'{key}: ΔHOTA={number(r["delta_HOTA"],8)}, 13原视频95%区间={r["original_parent_video"]["CI95"]}；45剪辑补充区间={r["evaluation_clip"]["CI95"]}。')
    add('原MINUS的466条predict文件与新参考回放逐字节一致，因此复用其原官评而没有重复运行相同预测。新视角反事实只覆盖100/3508≈2.85%的回答位置，不是完整线上选视角或选题政策的全面消融。DS/固定可靠性比较作用于整个回放流，也会影响后续快校准；点诊断DS则是独立校准子集拟合，两者不是同一个实验。')
    add('本轮完整系统结果支持保留DS：原视角48.847077对固定可靠性40.048707，ΔHOTA=+8.798371，13原视频95%区间[2.560713,15.826144]；CROP位置替换下DS也提高8.279670。不能因简化点诊断Brier变差就把DS一刀切删除；反过来，完整系统收益包含累计证据、后续快拟合、阈值与输出数量变化，并未证明DS学到了真实无偏可靠性或获得神经语义学习收益。要主张独立排序改善仍需进一步固定数量诊断。')
    add('原路由−CROP在DS下−0.305243，13原视频区间[-3.989475,3.968693]，不支持当前部分干预范围内的稳定正路由收益。不能说人类视觉启发被推翻，也不能说现有路由已经不可替代。DS_BANK_ORIGINAL与原MINUS的HOTA及输出完全相同，100个新原视角回答没有改变最终基准结果。')
    add('C4的100个替换位置实际集中于15/45剪辑、5/13原视频；其中校准2原视频/39位置，评估3原视频/61位置。完整官评仍包括全部45剪辑/466查询，后续DS/快校准流可能传播早期干预的影响。不能把全套输出官评等同于在13个原视频上均匀验证了新视角。')
    add('置信区间以Sports的13个原视频为主，将同一视频剪出的全部c片段保持在一起；45片段结果只作与旧表可比的补充。都是固定视频流与模型下的发展数据描述性区间，不是重训稳定性或盲域显著性，不对多重比较作正式确证性推断。')
    add('补充复核上一轮已完成LoRA/重算结果的原视频聚类敏感性（本轮新增的事后统计复核，不改原报告的预注册口径，也没有重跑GPU）：')
    for key,r in read('analysis/prior_parent_sensitivity.json')['comparisons'].items():
        add(f'  {key}: ΔHOTA={number(r["delta_HOTA"],8)}；13原视频CI95={r["original_parent_video"]["CI95"]}；原45片段CI95={r["evaluation_clip"]["CI95"]}。')
    add('联合LoRA＋重算−MINUS在13原视频下的区间也跨0；所以连联合收益的稳定性都不能只靠原45片段正区间下强结论。固定RECALC数量的LoRA排序负差在13原视频下仍为负区间，但另一数量口径仍跨0。没有稳定正权重归因，亦不能扩大为所有LoRA普遍有害。')
    add('')
    add('八、先判断有没有：协议是任务契约，不能由批次凑出来')
    legacy=read('protocol/batch_invariance.json');fixed=read('protocol/contract_invariance.json')
    for old,new in zip(legacy['rows'],fixed['rows']):
        add(f'{old["pair"]}: 原整批={old["full_batch"]}，单视频相对整批翻转{old["same_video_protocol_flips_due_to_batching"]}次；显式{new["declared_contract"]}协议翻转0次。')
    add('旧正式v8P实际在prepare阶段对完整目标域计算一次协议表，运行时读取封存表；这里发现的是分类规则随批构成变化的可移植性缺陷，并非证明旧正式运行在视频中途错误切换。新增classify_protocol(..., task_protocol=...)使用调用者给定的任务契约，默认旧行为兼容。该工程修复不冒充视觉学习能力。')
    for pair,r in absence['pairs'].items():add(f'{pair}旧实际输出拒绝：'+json.dumps(r['counts'],ensure_ascii=False,sort_keys=True))
    r=methods['UNIFIED_CHECKLIST'];d=comparisons['UNIFIED_CHECKLIST_vs_MINUS']
    add(f'C4强制统一CHECKLIST固定问答：HOTA={number(r["HOTA"],8)}，ΔHOTA={number(d["delta_HOTA"],8)}，13原视频95%区间={d["original_parent_video"]["CI95"]}；错误拒绝{r["false_absence_query_count"]}条，有目标零输出{r["present_zero_output"]}条。原MINUS相应为{methods["MINUS"]["false_absence_query_count"]}条/{methods["MINUS"]["present_zero_output"]}条。')
    add('强制CHECKLIST使用通用.5存在先验，未重新拟合跨视频的存在性EB流；固定历史问答仍全部回放，即使反事实会提前拒绝也不删回答。因此只诊断固定证据与此通用先验下的协议/存在性决策，不能外推完整自适应CHECKLIST政策一定损失相同分数，也不宣称真实重选题后的时间收益。视频保证存在不代表每一帧必然存在，身份阈值也仍可能使整句无输出。')
    add('')
    add('九、主动查证：有限池预算诊断能说明多少')
    for pair in ('C1','C4','C5'):
        add(pair+'（以下准确率/Brier为3个tie/exploration种子均值，不是3次新教师运行）：')
        by=defaultdict(list)
        for r in bank['acquisition']:
            if r['pair']==pair:by[r['method'],r['teacher_calls']].append(r)
        for (method,calls),rr in sorted(by.items()):
            add(f'  {method}, 查{calls}个回答: accuracy={number(np.mean([r["metrics"]["accuracy"] for r in rr]))}, Brier={number(np.mean([r["metrics"]["Brier"] for r in rr]),6)}。')
    add('已确认存在大量未查证正确轨迹，因此值得做覆盖研究。但本轮证据库只取历史上问过的位置，它没有提供从未问过轨迹的新回答。有限池比较只能诊断这些位置中的排序，不能证明20%探索已修复全候选覆盖缺口，更不能用预付900回答后的回放时间声称部署更便宜。未查位置回退原先验，全部回答时各策略必然汇合也不是主动策略的胜利。')
    add('该有限池的C4在10%与30%回答预算下，均匀抽样Brier均低于信息增益/纯不确定性；C1则低预算主动方法更好，C5随指标和预算变化。过度自信但错误的来源先验可能不会触发纯不确定性查证。这里只是已选历史池上的诊断，不能外推全域探索收益，但足以否定“主动必胜”的预设。')
    add('')
    add('十、真实工程剖析和成本')
    add(f'GPU1学生加载及权重校验={number(profile["student_load_wall_seconds"],3)}秒；完整profile进程={number(profile["process_wall_seconds"],3)}秒。加载包含NAS读取和完整权重校验，不等于GPU核执行时间。')
    for r in profile['records']:
        add(f'{r["cache_mib"]}MiB精确缓存：16锚帧，1208候选×查询行；特征流={number(r["feature_flow_wall_seconds"],3)}秒；phase={json.dumps(r["phase_wall_seconds"],ensure_ascii=False)}；cache={r["frozen_cache_stats"]}；四类数值逐位一致={r["bitwise_equal_to_first"]}。')
    add('62张历史帧对应峰值419,973,120字节，小缓存已经容纳，两个容量的backbone调用完全相同。本切片没有证实扩大缓存会提速；8192MiB的第二次IO更慢也不能当成容量导致变慢的因果证据。先512后8192、OS缓存未清、模型已驻留且NAS与其他实验共享，不做整域倍速外推。')
    add('阶段一已验证真实GPU前向；上一轮完整C4主动47.478/HOTA、11229.134秒及其IO成本仍是上一轮的独立收据，本次不能把固定回答回放或这个小profile替换为新的部署成本曲线。')
    add(f'GPU0证据准备最终续跑进程={number(evidence["process_wall_seconds"],3)}秒；连同失败尝试约={number(evidence["all_attempts_wall_seconds"],3)}秒；续跑加载/服务/最终冻结校验={evidence["phase_wall_seconds"]}；教师全部回答对应真正前向累计={number(evidence["teacher"]["forward_seconds"],3)}秒。900回答准备成本单列，不与任一单视角部署等同。')
    add('完整成本收据保留GPU监督器端到端wall和各CPU生成/官评时间。并发任务的wall不能简单相加当作历时；CPU回放与GT官评不是方法在线GPU成本。初次失败的CPU审计是额外研究开销，不从记录中删除。')
    add('第一次失败CPU审计没有可恢复的精确wall账本，因此不声称已精确加总本研究的全部CPU总开销；该缺口在成本JSON中单列。GPU证据准备的失败与续跑两次父进程wall均有完整收据。')
    add('300个登记位置的原视角输入与历史记录逐项一致（原图、视图像素、提示、框、模型、代码）；重新回答中有1个二值判断改变，具体数值成因未单变量归因。相同输入和冻结权重不在本报告中冒充跨环境逐数值确定性；C4原视角回放最终输出未变另有独立核验。')
    add('')
    add('十一、对“视觉本能＋经典算法”创新点的逐项解释')
    add('1. 有疑点才看：方向仍然合理，但当前更大的疑点是看错了候选或根本没看正确轨迹。需要覆盖感知的选题设计；不能仅把旧信息增益改个名字称新主动查证。')
    add('2. 根据问题切换视角：这是可测假说，不能用直觉保证必胜。以上三视角同位置结果与完整C4部分干预给出实测证据，支持范围仅限这些属性、位置和开发方向。')
    add('3. 合理融合有偏证据：DS是经典工具，不是正确性证明。必须看它相对固定可靠性/同预算平均的实测变化，并考虑同一教师相关错误、无标签可识别性和校准覆盖。')
    add('   本轮C4完整固定问答有约+8.8 HOTA系统级贡献，支持保留；而小型点诊断没有一致的概率质量改善，故“DS在系统里有用”和“已准确估计真实可靠性”是两种强度不同的结论。')
    add('4. 先判断有没有，再判断是谁：必须服从任务是否保证存在的契约。C4的存在性EM本来关闭；该方向好成绩不能归功于经验贝叶斯存在性学习。CHECKLIST中的稀疏No与错误拒绝需同时报告。')
    add('5. 看清一次持续记住：当前已按轨迹保留信念，最终稳定片段翻转为0。下一步若做时序，应研究何时因遮挡、真实语义变化或ID混杂而分段/复核，而非先堆一个平滑模块。')
    add('6. 学习新域语义：上一轮LoRA独立贡献区间跨0，数量控制下无稳定正排序收益；本轮没有重新训练LoRA。不能扩大为所有TTA/所有LoRA无效，也不能把本轮冻结推断的收益说成神经权重学会了跨域语义。')
    add('')
    add('十二、失败、修改、验证和未声称完成的事')
    add('首轮C4审计完成45视频后，在oracle封签处因重复status参数失败；修复为独立offline_oracle_uses_GT标志，保留原日志/失败收据，CPU审计完整重跑，不跳视频、不改变阈值和seed。')
    add('首轮GPU0采集返回C1/C4的600个回答后，在第一个KITTI图像路径处失败。原因是新入口只替换A800路径前缀，漏用了既有configs/a100_paths.json中的KITTI共享图像映射；并非NAS缺失这批图像。修复入口，全部100个KITTI登记位置的原图字节哈希与历史教师输入一致，再只续跑剩余300回答。失败进程wall1287.891秒保留，不重复已完成调用，不改候选、帧、视角或样本。')
    add('第一次采集在异常退出前没有终末整模型哈希；600个回答均经过每次前向的冻结版本检查，保留各自模型初始哈希。续跑结束后整模型冻结封签通过，全部900个回答模型哈希相同。收据明确区分已完成的终末封签与第一次异常退出的证据边界，不补造第一次最终哈希或其丢失的细相位时间。')
    add('新显式协议与原v8P回归测试14项通过。源代码编译、结果一致性与bootstrap复算见validation_receipt.json；最终分支体积由tools/check_publication_size.py检查最终提交，机器收据留本地a100_runs。原MINUS466条预测逐文件字节一致；profile四类数值逐位一致；教师冻结参数收据与生成GT打开0均保留。')
    add('本轮用户已明确无独立数据、先完成机制诊断。因此没有未执行的独立盲测结果，也不把原视频分组留出称为新的跨域泛化证明。没有宣称新时序模型、全候选探索策略或完整C4加速已完成；这些需要另有明确验证设计，不能从有限诊断自动推出。')
    add('四组回放和有限点诊断共用开发数据，多个比较均为解释性观察。所有指标、负结果和不确定性一并保留，不按HOTA挑一个最好的方法重写故事。')
    add('')
    add('十三、下一步应当具体改什么')
    add('优先围绕未查到正确轨迹的查询设计覆盖感知选题：将当前高先验/高信息增益的集中查证与分层探索、每句覆盖、负证据可靠性保护分开；用同候选、同初值、同新回答预算以及固定输出数量/同墙钟分别比较。要新增未查轨迹的证据，才有资格声称解决本次发现的覆盖缺口。')
    add('若视角或DS在某方向无稳定收益，保留简单固定视角/固定可靠性作为默认候选，避免为了叙事强留模块。可靠性估计应有足够跨轨迹证据并检查同源视角相关，不把三个视角当作独立专家。')
    add('同样优先审计来源先验与证据力度：保留同预算单视角原始概率、受控先验温度、二值/软证据和固定数量排序对照，区分“教师说对但先验不让改”“阈值/数量改变”与真正语义排序改善。应预先登记规则，不能在当前开发GT上不断寻找最好阈值。')
    add('对物理ID混杂严重的Dance，应先区分前端身份修复、语义变化分段与错误信念整轨迹传播；只有证明具体错误能被修复，再引入时序机制。工程侧先测大切片重复读取与缓存命中，再决定预取/解码复用；近似抽帧与精确复用必须分账。')
    add('论文中心仍可围绕跨域语言指代的证据与信念管理，但方法是否提升需要新的实证。现在最有价值的是把“视觉本能”变成可证伪机制，以及拿到明确的失败定位；没有必要也没有依据宣称现有每个模块已不可替代。')
    add('')
    add('十四、文件与复核入口')
    add('执行计划：docs/A100_机制诊断执行计划_20261121.txt')
    add('精选收据：docs/a100_mechanism_20261121/；每个JSON保留实际口径与来源。')
    add('精确HOTA充分统计：analysis/official_sufficient_statistics.json；主置信区间按13原视频聚类。')
    add('点结果和离线标签：analysis/bank_labeled_rows.json；GT附着晚于生成封签。')
    add('仅用小型JSON复算：python tools/verify_a100_mechanism_evidence.py；无需图像、checkpoint、GT目录或GPU，能复核新旧原视频聚类区间及点指标。')
    add('代码入口：scripts/a100_mechanism_{prepare,collect,offline,replay,bank,profile,analyze,report}.py；监督器只启动本轮登记单元。')
    add('完整原始结果保存在本地a100_runs/mechanism_20261121；NAS大型资产和历史结果保持只读。')
    REPORT.write_text('\n'.join(lines)+'\n')
    evidence_files=[]
    for p in sorted(PUBLIC.rglob('*')):
        if p.is_file() and p.name!='evidence_index.json':
            evidence_files.append(dict(path=str(p.relative_to(PUBLIC)),bytes=p.stat().st_size,sha256=file_sha(p)))
    atomic_json(PUBLIC/'evidence_index.json',dict(status='SELECTED_LIGHTWEIGHT_EVIDENCE',files=evidence_files,
        selected_bytes=sum(r['bytes'] for r in evidence_files),raw_outputs=str(OUT),report_sha256=file_sha(REPORT)))
    print(json.dumps(dict(event='MECHANISM_REPORT_WRITTEN',report=str(REPORT),characters=len(REPORT.read_text()),evidence_bytes=sum(r['bytes'] for r in evidence_files))),flush=True)


if __name__=='__main__':main()
