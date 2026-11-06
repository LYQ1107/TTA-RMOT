"""v8 closed-prediction metrics and attribution; GT never feeds runtime."""
import argparse
from collections import Counter
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


def verdict(directions):
    completed=all(r['MINUS_v8'] is not None for r in directions)
    wins=[r['pair'] for r in directions if r['MINUS_v8'] is not None and r['MINUS_v8']>max(r['YT_PRIOR'],r['UNIFORM_K'])]
    lookup={r['pair']:r['MINUS_v8'] for r in directions}
    retained=all(lookup.get(p) is not None and lookup[p]>=floor for p,floor in [('C1',35.724),('C6',34.469)])
    passed=completed and len(wins)>=5 and retained
    return dict(result='PASS' if passed else 'FAIL' if completed else 'NOT_EVALUATED',
        complete_six_directions=completed,directions_above_both_controls=len(wins),wins=wins,
        Dance_retention=retained,stage4_authorized=passed,minimum_control_wins=5)


def report(execution):
    from scripts.active_eval import summarize,bootstrap,offline_labels
    from scripts.active_attribution import attribute
    from scripts.report_active import saved_presence_decisions,final_confidence_summary,behavior_and_reliability
    public=execution.public;plan=json.loads((public/'registered_jobs.json').read_text())
    oldroot=Path(execution.manifest['prior_v7_public_root'])
    with (oldroot/'direction_summary.csv').open() as handle:old={r['pair']:r for r in csv.DictReader(handle)}
    metrics=[];presence=[];rounds=[];pools=[];streams=[];answers=[];glances=[];histograms=[]
    costs=[];recall=[];reliability=[];artifacts=[];intervals={};truths={}
    def folder(pair,method):return execution.active/pair/method/'rho_0.3'
    for job in plan['jobs']:
        pair,method=job['pair'],job['method'];target=PAIRS[pair][1];path=folder(pair,method)
        if not (path/'predictions_seal.json').exists():continue
        seal=verify_predictions(path)
        # Independent official evaluator must already have closed this job.
        summary=path/'offline_summary.json'
        if not summary.exists():continue
        result=json.loads(summary.read_text())
        assert result['prediction_seal_sha256']==file_sha(path/'predictions_seal.json')
        metrics.append(dict(result,pair=pair,method=method))
        if target not in truths:truths[target]=offline_labels(target)
        meta,truth,_=truths[target]
        rows=saved_presence_decisions(path,meta,truth,pair,method,.3)
        for row in rows:
            state=json.loads((path/'videos'/row['video']/'presence.json').read_text())
            query=next(q for q in state['queries'] if q['query']==row['expression'])
            row.update(decision_source=query['decision_source'],low_tentative=query['low_tentative'],
                       output_absent=query['decision_source'] in ('EVIDENCE','PRIOR_BUDGET_EXHAUSTED'))
        presence+=rows
        _,_,ds=behavior_and_reliability(path,target,pair,method,.3,truth);reliability+=ds
        video_receipts=[]
        for vi,directory in enumerate(sorted((path/'videos').iterdir())):
            receipt=json.loads((directory/'receipt.json').read_text());video_receipts.append(receipt)
            qdata=json.loads((directory/'questions.json').read_text())
            eb=receipt['EB'];after=eb['after']
            streams.append(dict(pair=pair,method=method,video=directory.name,video_index=vi,
                before_rho=eb['before']['rho_hat'],rho_hat=after['rho_hat'],
                pi_0=after['pi_strata'][0],pi_1=after['pi_strata'][1],pi_2=after['pi_strata'][2],
                answered_tracks=json.dumps(after.get('tracks_per_stratum')),
                identification=json.dumps(after.get('identification_status')),
                iterations=after.get('iterations'),converged=after.get('converged'),
                objective_trace_monotone=after.get('objective_trace_monotone'),
                contraction_rate=after.get('contraction_rate'),
                v7_shadow_rho=eb['v7_shadow']['after']['rho_hat'],v7_shadow_pi=eb['v7_shadow']['after']['pi_hat']))
            for rr in receipt['rounds']:
                counts=rr['question_counts'];total=rr['actual_questions'];slow=rr['slow']
                rounds.append(dict(pair=pair,method=method,video=directory.name,video_index=vi,round=rr['round'],
                    actual_questions=total,new_confirmed=rr['new_distinct_confirmed_pairs'],
                    **{k+'_count':v for k,v in counts.items()},
                    **{k+'_fraction':v/total if total else None for k,v in counts.items()}))
                pools.append(dict(pair=pair,method=method,video=directory.name,round=rr['round'],
                    status=slow['status'],optimizer_updates=slow['optimizer_updates'],
                    distinct_positive_pairs=slow.get('distinct_positive_pairs',0),
                    distinct_negative_pairs=slow.get('distinct_negative_pairs',0),
                    **slow.get('pair_pools',{}),
                    frame_pools=json.dumps(slow.get('frame_pools',{})),
                    actual_positive_samples=slow['actual_optimized_positive_samples'],
                    actual_negative_samples=slow['actual_optimized_negative_samples']))
                sampled=[a['posterior_probability_before'] for a in qdata if a['round']==rr['round']]
                asked_hist=np.histogram(sampled,bins=np.linspace(0,1,21))[0]
                all_hist=rr['all_pair_posterior_before_round_histogram']
                for b,(a,c) in enumerate(zip(asked_hist,all_hist)):
                    histograms.append(dict(pair=pair,method=method,video=directory.name,round=rr['round'],
                        bin_left=b/20,bin_right=(b+1)/20,asked_count=int(a),all_pair_count=int(c),
                        all_pair_reference='eligible pairs at round start; asked values at actual dispatch'))
            for a in qdata:
                present=bool(truth[directory.name,a['query_id']]['counts'].sum())
                answers.append(dict(pair=pair,method=method,video=directory.name,query=a['query_id'],
                    track=a['track_id'],round=a['round'],kind=a['kind'],
                    posterior_before=a['posterior_probability_before']))
                if a['kind']=='GLANCE':glances.append(dict(pair=pair,method=method,video=directory.name,
                    query=a['query_id'],GT_present=present,yes=a['yes_probability']>=.5))
        teacher=seal['teacher']
        costs.append(dict(pair=pair,method=method,logical_questions=teacher['logical_calls'],
            new_calls=teacher.get('new_calls',0),cache_hits=teacher.get('cache_hits',0),
            replayed_answers=teacher.get('replayed_answers',0),teacher_GPU_hours=teacher.get('forward_seconds',0)/3600,
            teacher_equivalent_GPU_hours=teacher.get('equivalent_forward_seconds',0)/3600,
            LoRA_GPU_hours=seal['LoRA_GPU_seconds']/3600,rescore_GPU_hours=seal['rescore_GPU_seconds']/3600,
            stream_wall_seconds=seal['wall_seconds'],
            optimizer_updates=sum(r['slow']['optimizer_updates'] for v in video_receipts for r in v['rounds']),
            GLANCE_BUDGET_SHORTFALL_videos=sum(v['GLANCE_BUDGET_SHORTFALL'] for v in video_receipts),
            unasked_queries=sum(v['glance_unanswered_queries'] for v in video_receipts),
            replay_teacher_dependency_in_MINUS=method=='SLOW_REPLAY'))
        ar_path=path/'offline_attribution.json'
        ar=json.loads(ar_path.read_text()) if ar_path.exists() else attribute(path,target)
        assert ar['prediction_seal_sha256']==file_sha(path/'predictions_seal.json')
        recall.append(dict(pair=pair,method=method,GT_recall=ar['adapted_matched_gt']/ar['gt_observations'],
            output_rows=ar['adapted_selected_rows'],recovered_GT=ar['recovered_gt'],lost_GT=ar['lost_gt'],
            net_GT_vs_Native=ar['net_gt_recovery'],net_FP_vs_Native=ar['net_FP_change'],
            new_FP_rows=ar['new_false_positive_rows'],removed_FP_rows=ar['removed_false_positive_rows'],
            unchanged_rows_new_FP=ar['unchanged_rows_new_FP'],unchanged_rows_removed_FP=ar['unchanged_rows_removed_FP']))
        artifacts.append(dict(job,path=str(path),seal_sha256=file_sha(path/'predictions_seal.json'),
                              episodes=seal['episodes']))
        previous=Path(execution.manifest['prior_v7_active_root'])/pair/method/'rho_0.3'
        if (previous/'predictions_seal.json').exists():intervals[pair+'__'+method+'_v8_minus_v7']=bootstrap(path,previous,target)
    lookup={(r['pair'],r['method']):r for r in metrics}
    directions=[]
    for pair,(model,target) in PAIRS.items():
        r=old[pair]
        directions.append(dict(pair=pair,source=model,target=target,NATIVE=float(r['NATIVE']),PRES_TOPK=float(r['PRES_TOPK']),
            FULL_v5=float(r['FULL_v5']),MINUS_v7=float(r['MINUS_SLOW_v7']),
            MINUS_v8=lookup.get((pair,'MINUS_SLOW'),{}).get('HOTA_unrounded'),
            FULL_v8=lookup.get((pair,'FULL'),{}).get('HOTA_unrounded'),
            REPLAY_v8=lookup.get((pair,'SLOW_REPLAY'),{}).get('HOTA_unrounded'),
            YT_PRIOR=float(r['YT_PRIOR']),UNIFORM_K=float(r['UNIFORM_K'])))
    effects=[]
    for pair in PAIRS:
        for method in ('FULL','SLOW_REPLAY'):
            if (pair,method) not in lookup or (pair,'MINUS_SLOW') not in lookup:continue
            interval=bootstrap(folder(pair,method),folder(pair,'MINUS_SLOW'),PAIRS[pair][1])
            key=pair+'__'+method+'_minus_MINUS';intervals[key]=interval
            effects.append(dict(pair=pair,method=method,**interval))
    gate=verdict(directions)
    replays=[x for x in effects if x['method']=='SLOW_REPLAY' and x['pair'] in ('C1','C4')]
    slow_complete=len(replays)==2
    slow_pass=slow_complete and all(x['delta_HOTA']>=0 for x in replays) and any(x['CI95'][0]>0 for x in replays)
    slow_gate=dict(result='MAIN_METHOD' if slow_pass else 'APPENDIX' if slow_complete else 'NOT_EVALUATED',
                   compared_pairs=[r['pair'] for r in replays],both_nonnegative=all(r['delta_HOTA']>=0 for r in replays))
    write_csv(public/'main.csv',metrics);write_csv(public/'direction_summary.csv',directions)
    write_csv(public/'presence_decisions.csv',presence);write_csv(public/'final_confidence.csv',final_confidence_summary(presence))
    write_csv(public/'slow_rounds.csv',pools);write_csv(public/'eb_stream.csv',streams)
    write_csv(public/'question_behavior.csv',rounds);write_csv(public/'glance_hits.csv',glances)
    write_csv(public/'posterior_histograms.csv',histograms);write_csv(public/'reliability_vs_GT.csv',reliability)
    write_csv(public/'costs.csv',costs);write_csv(public/'recall_compare.csv',recall);write_csv(public/'slow_effects.csv',effects)
    atomic_json(public/'bootstrap.json',intervals);atomic_json(public/'verdict.json',dict(main=gate,slow=slow_gate))
    atomic_json(public/'artifacts.json',dict(new=artifacts))
    text='# v8：阶段2/3完整指标与独立归因\n\n'
    text+=f"已完成官方评测{len(metrics)}/{len(plan['jobs'])}项。主判据：{gate['result']}；慢分支：{slow_gate['result']}。未完成格为None，不冒充0或完成。\n\n"
    text+='## 1. 六方向主表（HOTA原精度）\n\n'+table(directions,list(directions[0]))+'\n\n'
    text+='## 2. 全部官方指标\n\n'+table(metrics,['pair','method','HOTA_unrounded','DetA','AssA','IDF1','IDSW','prediction_rows','query_empty_prediction_rows','Present_HOTA'])+'\n\n'
    text+='## 3. 存在性、估计器与行为\n\n'
    text+='presence_decisions.csv分列EVIDENCE、PRIOR_BUDGET_EXHAUSTED及low_tentative；eb_stream.csv保留rho/pi、>=200档位标记、EM诊断和v7影子计数。影子值是当前v8判定上的旧更新公式，不冒充历史v7闭环重跑。question_behavior.csv、glance_hits.csv、posterior_histograms.csv记录问题组成、新确认和GT分组第一眼命中；全池分布取轮初，提问分布取实际调度前，不混称同一时刻的随机样本。\n\n'
    text+='## 4. 慢分支与固定存在性回放\n\n'+table(effects,['pair','method','delta_HOTA','CI95','clusters'])+'\n\n'
    text+='FULL−MINUS还包含选题反馈变化。REPLAY固定MINUS问答、rho/pi及每轮六个存在性数组，允许LoRA通过打分/f/快校准影响身份后验；最终存在性mask仍固定。slow_rounds.csv分列各池、跳过原因和实际16/16优化暴露，不由轮末标签倒推。列表CE/NONE损失已删除，统一BCE。\n\n'
    text+='## 5. 高置信错负与GT恢复\n\n'+table(recall,['pair','method','GT_recall','output_rows','recovered_GT','lost_GT','net_FP_vs_Native'])+'\n\n'
    text+='final_confidence.csv保留v6的.05高置信错负口径。GT恢复/FP来自独立IoU>=.5最大基数匹配，不冒充官方HOTA匹配；新增行与不变行的重分配已分列。DS/GT特异度是所问集合的事后诊断，不反馈适应。\n\n'
    text+='## 6. 成本\n\n'+table(costs,['pair','method','logical_questions','new_calls','cache_hits','replayed_answers','teacher_GPU_hours','LoRA_GPU_hours','rescore_GPU_hours','optimizer_updates'])+'\n\n'
    text+='共享设备相位用时不是GPU利用率积分；冻结bank、StageA与固定MOT准备成本未重新测量，不等于0。REPLAY依赖MINUS教师问答成本，不将它描述成独立无教师系统。恢复前的pilot会话调用计入最终流。\n\n'
    text+='## 7. 阶段0与协议记录\n\n完整0b–0e表、合成验收、负面反事实与A2/A3诊断见[阶段0报告](amendments_A2_A3/REPORT.md)。本报告只读取封存预测；GT从未进入生成进程。v7与旧200迭代失败不覆盖。主判据要求六方向中至少5胜并满足C1>=35.724、C6>=34.469，不沿用v7的4胜规则。\n'
    (public/'REPORT.md').write_text(text)
    return dict(completed=len(metrics),expected=len(plan['jobs']),main=gate,slow=slow_gate)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--partial',action='store_true');p.parse_args()
    execution,_=authorize_active();print(json.dumps(report(execution)),flush=True)
