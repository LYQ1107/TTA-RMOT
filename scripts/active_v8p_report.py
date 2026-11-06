"""Independent sealed-prediction report. Never imported by a GT-free worker."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.active.execution import authorize_active
from flex_uda.presence.predictions import verify_predictions
from scripts.active_run import PAIRS
from scripts.active_v8_stage0_gt import write_csv
from scripts.active_v8_stage0_report import table


def verdict(rows):
    wins=[r['pair'] for r in rows if r['v8P']>max(r['YT_PRIOR'],r['UNIFORM_K'])]
    h={r['pair']:r['v8P'] for r in rows}
    checks=dict(control_wins_at_least_five=len(wins)>=5,
                C1_floor=h['C1']>=35.724,C6_floor=h['C6']>=34.469,
                C3_floor=h['C3']>=63.35)
    return dict(result='PASS' if all(checks.values()) else 'FAIL',wins=wins,
        directions_above_both_controls=len(wins),checks=checks,
        stage_B_authorized=all(checks.values()),thresholds=dict(C1=35.724,C6=34.469,C3=63.35))


def plots(public,rounds,histograms,glances,streams):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for protocol in ('CHECKLIST','PRESUPPOSED'):
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        rs=[r for r in rounds if r['protocol']==protocol]
        kinds=('GLANCE','PRESENCE','PRESENCE_COVER','IDENTITY','POSITIVE_RECHECK')
        bottom=np.zeros(4)
        for kind in kinds:
            fractions=[]
            for rnd in range(1,5):
                subset=[r for r in rs if r['round']==rnd]
                total=sum(r['actual_questions'] for r in subset)
                fractions.append(sum(r[kind] for r in subset)/total if total else 0)
            axes[0,0].bar(range(1,5),fractions,bottom=bottom,label=kind);bottom+=fractions
        axes[0,0].set(title='Question type / round',xlabel='Round',ylabel='Call fraction')
        axes[0,0].legend(fontsize=7)
        hs=[h for h in histograms if h['protocol']==protocol]
        for field,label in [('before','At dispatch'),('after','After teacher')]:
            counts=np.sum([h[field] for h in hs],axis=0) if hs else np.zeros(20)
            axes[0,1].step(np.arange(20)/20+.025,counts,where='mid',label=label)
        axes[0,1].set(title='Queried pair posterior (not GT accuracy)',xlabel='Posterior',ylabel='Answers')
        axes[0,1].legend()
        gs=[g for g in glances if g['protocol']==protocol]
        if gs:
            groups=[False,True]
            vals=[sum(g['yes'] for g in gs if g['GT_present']==p)/max(1,sum(g['GT_present']==p for g in gs)) for p in groups]
            axes[1,0].bar(['GT empty','GT present'],vals);axes[1,0].set_ylim(0,1)
            axes[1,0].set(title='GLANCE Yes rate (offline GT groups)')
        else:
            axes[1,0].text(.5,.5,'N/A: no GLANCE by protocol',ha='center')
        ss=[s for s in streams if s['protocol']==protocol]
        for pair in sorted({s['pair'] for s in ss}):
            group=[s for s in ss if s['pair']==pair]
            axes[1,1].plot([s['video_index'] for s in group],[s['rho_hat'] for s in group],label=pair)
        axes[1,1].set(title='rho estimate' if protocol=='CHECKLIST' else 'Presence fixed at 1; EM disabled',xlabel='Video index')
        axes[1,1].legend()
        fig.suptitle('v8P '+protocol+' (post-D3 protocol rule)');fig.tight_layout()
        fig.savefig(public/('behavior_'+protocol.lower()+'.png'),dpi=150);plt.close(fig)


def report(execution):
    from scripts.active_eval import bootstrap,offline_labels
    from scripts.active_attribution import attribute
    from scripts.report_active import saved_presence_decisions,final_confidence_summary,behavior_and_reliability
    public=execution.public
    inputs=json.loads((public/'inputs.json').read_text())
    reuses={r['pair']:r for r in inputs['reused_v8']}
    prior8=Path(execution.manifest['prior_v8_active_root'])
    prior7=Path(execution.manifest['prior_v7_active_root'])
    with (Path(execution.manifest['prior_v7_public_root'])/'direction_summary.csv').open() as handle:
        old={r['pair']:r for r in csv.DictReader(handle)}
    protocols=json.loads(Path(execution.manifest['protocol_path']).read_text())['targets']
    main=[];presence=[];decisions=[];costs=[];recall=[];artifacts=[];rounds=[];histograms=[]
    streams=[];glances=[];reliability=[];intervals={};truths={};budget_rows=[]
    paths={pair:Path(reuses[pair]['folder']) if pair in reuses else execution.active/pair/'MINUS_SLOW/rho_0.3'
           for pair in PAIRS}
    seals={pair:verify_predictions(path) for pair,path in paths.items()}
    for pair,(model,target) in PAIRS.items():
        path=paths[pair]
        seal=seals[pair];digest=file_sha(path/'predictions_seal.json')
        if pair in reuses:assert digest==reuses[pair]['seal_sha256']
        summary=json.loads((path/'offline_summary.json').read_text())
        assert summary['prediction_seal_sha256']==digest and summary['GT_open_attempts']==0
        v8path=prior8/pair/'MINUS_SLOW/rho_0.3'
        v8=json.loads((v8path/'offline_summary.json').read_text())
        r=old[pair];protocol=protocols[target]['protocol']
        main.append(dict(pair=pair,source=model,target=target,protocol=protocol,
            NATIVE=float(r['NATIVE']),PRES_TOPK=float(r['PRES_TOPK']),YT_PRIOR=float(r['YT_PRIOR']),
            UNIFORM_K=float(r['UNIFORM_K']),v7=float(r['MINUS_SLOW_v7']),v8=v8['HOTA_unrounded'],
            v8P=summary['HOTA_unrounded'],delta_v8P_v8=summary['HOTA_unrounded']-v8['HOTA_unrounded'],
            DetA=summary['DetA'],AssA=summary['AssA'],IDF1=summary['IDF1'],IDSW=summary['IDSW'],
            prediction_rows=summary['prediction_rows'],Present_HOTA=summary['Present_HOTA'],
            query_empty_prediction_rows=summary['query_empty_prediction_rows'],reused_v8=pair in reuses))
        # All six current/reused prediction seals have been verified before GT.
        if target not in truths:truths[target]=offline_labels(target)
        meta,truth,_=truths[target]
        rows=saved_presence_decisions(path,meta,truth,pair,'MINUS_SLOW_v8P',.3)
        per_query={}
        for vi,video in enumerate(sorted(meta['videos'])):
            directory=path/'videos'/video
            p=json.loads((directory/'presence.json').read_text())
            for q in p['queries']:per_query[video,q['query']]=q
            receipt=json.loads((directory/'receipt.json').read_text())
            if protocol=='PRESUPPOSED':
                assert receipt['protocol']==protocol and not receipt['presence_inference']
                assert receipt['EB']['after']['iterations']==0
            streams.append(dict(pair=pair,protocol=protocol,video=video,video_index=vi,
                rho_hat=receipt['EB']['after']['rho_hat'],
                pi_strata=json.dumps(receipt['EB']['after']['pi_strata']),
                presence_inference=protocol=='CHECKLIST'))
            answers=json.loads((directory/'questions.json').read_text())
            if protocol=='PRESUPPOSED':
                assert all(a['kind'] in ('IDENTITY','POSITIVE_RECHECK') for a in answers)
            for rr in receipt['rounds']:
                rounds.append(dict(pair=pair,protocol=protocol,video=video,round=rr['round'],
                    actual_questions=rr['actual_questions'],**rr['question_counts']))
                subset=[a for a in answers if a['round']==rr['round']]
                histograms.append(dict(pair=pair,protocol=protocol,video=video,round=rr['round'],
                    before=np.histogram([a['posterior_probability_before'] for a in subset],bins=np.linspace(0,1,21))[0].tolist(),
                    after=np.histogram([a['posterior_probability_after'] for a in subset],bins=np.linspace(0,1,21))[0].tolist()))
            for a in answers:
                if a['kind']=='GLANCE':
                    glances.append(dict(pair=pair,protocol=protocol,video=video,query=a['query_id'],
                        GT_present=bool(truth[video,a['query_id']]['counts'].sum()),yes=a['yes_probability']>=.5))
        for row in rows:
            q=per_query[row['video'],row['expression']]
            row.update(protocol=protocol,decision_source=q['decision_source'],
                output_absent=q['decision_source'] in ('EVIDENCE','PRIOR_BUDGET_EXHAUSTED'))
        decisions+=rows
        presence.append(dict(pair=pair,protocol=protocol,presence_inference=protocol=='CHECKLIST',
            false_absent=sum(r['output_absent'] and r['query_present_GT'] for r in rows),
            correct_empty=sum(r['output_absent'] and not r['query_present_GT'] for r in rows),
            present_queries_zero_prediction_rows=sum(r['query_present_GT'] and r['selected_prediction_rows']==0 for r in rows),
            empty_queries_zero_prediction_rows=sum(not r['query_present_GT'] and r['selected_prediction_rows']==0 for r in rows),
            present_episodes=sum(r['query_present_GT'] for r in rows),
            empty_episodes=sum(not r['query_present_GT'] for r in rows),
            query_empty_prediction_rows=summary['query_empty_prediction_rows']))
        _,_,rel=behavior_and_reliability(path,target,pair,'MINUS_SLOW_v8P',.3,truth);reliability+=rel
        arpath=path/'offline_attribution.json'
        ar=json.loads(arpath.read_text()) if arpath.exists() else attribute(path,target)
        assert ar['prediction_seal_sha256']==digest
        oldar=json.loads((v8path/'offline_attribution.json').read_text())
        recall.append(dict(pair=pair,GT_recall=ar['adapted_matched_gt']/ar['gt_observations'],
            v8_GT_recall=oldar['adapted_matched_gt']/oldar['gt_observations'],
            recovered_GT=ar['recovered_gt'],lost_GT=ar['lost_gt'],net_GT_vs_Native=ar['net_gt_recovery'],
            new_FP_rows=ar['new_false_positive_rows'],removed_FP_rows=ar['removed_false_positive_rows'],
            unchanged_rows_new_FP=ar['unchanged_rows_new_FP'],unchanged_rows_removed_FP=ar['unchanged_rows_removed_FP'],
            output_rows=ar['adapted_selected_rows'],v8_output_rows=oldar['adapted_selected_rows']))
        teacher=seal['teacher']
        # rho=.3 is explicit historical reuse, not an unverified assertion
        # that nominal rho guarantees identical realized teacher budgets.
        for control in ('YESTRACK_BUDGET_PRIOR','UNIFORM_K'):
            controlpath=Path(execution.manifest['prior_v5_active_root'])/pair/control/'rho_0.3'
            cs=json.loads((controlpath/'predictions_seal.json').read_text())
            nc=cs['teacher']['logical_calls'];nm=teacher['logical_calls']
            budget_rows.append(dict(pair=pair,control=control,MINUS_v8P_calls=nm,
                historical_control_calls=nc,relative_difference=(nm-nc)/nc if nc else None,
                reuse_as_explicitly_registered=True,control_path=str(controlpath),
                control_seal_sha256=file_sha(controlpath/'predictions_seal.json')))
        attempts=[json.loads(p.read_text()) for p in path.glob('worker_attempts/*.json')]
        sessions=[json.loads(p.read_text()) for p in (path/'teacher_sessions').glob('*.json')]
        costs.append(dict(pair=pair,protocol=protocol,reused_v8=pair in reuses,
            logical_calls=teacher['logical_calls'],original_new_calls=teacher.get('new_calls',0),
            original_cache_hits=teacher.get('cache_hits',0),
            new_calls_this_round=0 if pair in reuses else teacher.get('new_calls',0),
            cache_hits_this_round=0 if pair in reuses else teacher.get('cache_hits',0),
            teacher_GPU_hours=teacher.get('forward_seconds',0)/3600,
            new_teacher_GPU_hours_this_round=0 if pair in reuses else teacher.get('forward_seconds',0)/3600,
            teacher_equivalent_GPU_hours=teacher.get('equivalent_forward_seconds',0)/3600,
            wall_seconds=seal['wall_seconds'],assigned_worker_wall_seconds=sum(a.get('wall_seconds',0) for a in attempts),
            peak_teacher_allocated_bytes=max((s.get('peak_allocated_bytes',0) for s in sessions),default=0),
            optimizer_updates=0,trainable_parameters=0,GT_open_attempts=0))
        for label,previous in [('v7',prior7/pair/'MINUS_SLOW/rho_0.3'),('v8',v8path)]:
            intervals[pair+'__v8P_minus_'+label]=bootstrap(path,previous,target)
        artifacts.append(dict(pair=pair,protocol=protocol,path=str(path),prediction_seal_sha256=digest,
            reuse=pair in reuses,episodes=seal['episodes']))
        print(json.dumps(dict(event='V8P_REPORT_DIRECTION_CLOSED',pair=pair,HOTA=summary['HOTA_unrounded'])),flush=True)
    gate=verdict(main)
    atomic_json(public/'stage_A_gate.json',gate)
    for name,rows in [('direction_summary',main),('presence_summary',presence),('presence_decisions',decisions),
                      ('final_confidence',final_confidence_summary(decisions)),('costs',costs),
                      ('recall_compare',recall),('question_behavior',rounds),('eb_stream',streams),
                      ('glance_hits',glances),('reliability_vs_GT',reliability)]:
        write_csv(public/(name+'.csv'),rows)
    write_csv(public/'baseline_budget_comparison.csv',budget_rows)
    atomic_json(public/'posterior_histograms.json',histograms)
    atomic_json(public/'bootstrap.json',intervals)
    atomic_json(public/'prediction_paths.json',dict(files=artifacts))
    plots(public,rounds,histograms,glances,streams)
    text='# v8P：查询协议条件存在性推断\n\n'
    text+='阶段 A：**'+gate['result']+'**。超过两项对照：'+str(gate['directions_above_both_controls'])+'/6；逐项门槛见 stage_A_gate.json。\n\n'
    text+='## 1. 主表（HOTA 未四舍五入）\n\n'+table(main,['pair','protocol','NATIVE','PRES_TOPK','YT_PRIOR','UNIFORM_K','v7','v8','v8P','DetA','AssA','IDF1','delta_v8P_v8'])+'\n\n'
    text+='### 成对视频聚类不确定性\n\n'+table([
        dict(comparison=k,delta_HOTA=v['delta_HOTA'],CI95=v['CI95'],clusters=v['clusters'])
        for k,v in intervals.items()],['comparison','delta_HOTA','CI95','clusters'])+'\n\nKITTI只有4个视频，CI=None表示按预注册只报告点估计，不表示差值为零。\n\n'
    text+='## 2. 存在性（最终输出拒绝口径）\n\n'+table(presence,['pair','protocol','false_absent','correct_empty','present_queries_zero_prediction_rows','empty_queries_zero_prediction_rows','present_episodes','empty_episodes','query_empty_prediction_rows'])+'\n\n'
    text+='PRESUPPOSED 的判空为 0 是协议的设计结果，不是学习出了完美存在性。C1/C6 用原 v8 seal SHA 直接引用，新增调用为 0。\n\n'
    text+='false_absent/correct_empty 沿用 EVIDENCE+PRIOR_BUDGET_EXHAUSTED 的拒绝原因口径；zero_prediction_rows 另计最终零行输出，包含未拒绝整句但轨迹后验没有选中任何目标的情形。二者不能混用。\n\n'
    text+='## 3. 消融与预算曲线\n\n'+('阶段 A 通过，按预注册 B1→B2→空闲卡 B3 顺序接续；未完成项目不填数。' if gate['result']=='PASS' else '阶段 A 未通过，按用户停止条件不运行任何阶段 B 项目。')+'\n\n'
    text+='## 4. 行为分析与归因\n\n![CHECKLIST](behavior_checklist.png)\n\n![PRESUPPOSED](behavior_presupposed.png)\n\n'
    text+='完整错拒、高置信错负、教师支持覆盖在 presence_decisions.csv、final_confidence.csv；DS 与 GT 事后特异度在 reliability_vs_GT.csv，分别保留锚点与轨迹多数标签。\n\n'
    text+='PRESUPPOSED 中未问对的 evidence=0，组合先验截断[-2,2]且不判整句空，因此未问对 posterior≥sigmoid(-2)≈.119；其 p<.05 高置信错负为0是构造保证，不是学会了语义或恢复了所有GT的证据。普通漏检、已问错误负例和新增误检仍需看下表。\n\n'
    text+=table(recall,['pair','GT_recall','v8_GT_recall','output_rows','v8_output_rows','recovered_GT','lost_GT','new_FP_rows','removed_FP_rows'])+'\n\n'
    text+='GT 恢复采用离线 IoU≥.5 最大基数匹配，不冒充官方 HOTA；新增行 FP 与不变行匹配重分配分开记录。\n\n'
    text+='## 5. 成本与审计\n\n'+table(costs,['pair','reused_v8','logical_calls','new_calls_this_round','cache_hits_this_round','new_teacher_GPU_hours_this_round','teacher_GPU_hours','teacher_equivalent_GPU_hours','wall_seconds','GT_open_attempts'])+'\n\n'
    text+='teacher_GPU_hours 是该方法原有的新调用相位时长，new_teacher_GPU_hours_this_round 单列本轮新增；引用 C1/C6 不等于原方法无成本。equivalent 时长包含缓存条目的原始 forward 成本，不等于本轮实际 GPU 耗时。所有相位时长均非独占利用率积分。\n\n'
    text+=table(budget_rows,['pair','control','MINUS_v8P_calls','historical_control_calls','relative_difference'])+'\n\nρ=.3按本轮明确指令保留历史对照，不因名义预算相同就声称实际调用严格相等；B2的.1/1.0逐视频匹配。\n\n'
    text+='bootstrap.json：Dance/SportsMOT 按视频聚类，2000 次、seed 0；KITTI 只有 4 视频，仅点估计。生成 GT 打开尝试全部为 0，逐预测 SHA 在 prediction_paths.json；先封存后独立 GT 评测。MINUS 不挂 LoRA、无学生参数更新。原 v8 阶段 3 未中断，旧 80 项未启动。\n\n'
    text+='## 6. 限制与停止条件\n\nC1 协议规则明确在看到 D3 后制定，在这三个数据集上等价于按数据集设定存在性先验，不是普遍可识别性定理。KITTI 仅四视频；这些官方 val 已用于开发，不宣称盲测。所有方向与失败均上报。\n'
    (public/'REPORT.md').write_text(text)
    return gate


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--partial',action='store_true');p.parse_args()
    report(authorize_active()[0])
