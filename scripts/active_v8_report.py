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


def stratified_truth(folder,meta,truth,public):
    digest=file_sha(folder/'predictions_seal.json')
    path=public/'offline_diagnostics'/(digest+'_stratified_GT.json')
    if path.exists():return json.loads(path.read_text())
    num=np.zeros(3,int);den=np.zeros(3,int);asked_num=np.zeros(3,int);asked_den=np.zeros(3,int)
    present=queries=0
    for video,bank in sorted(meta['videos'].items()):
        with np.load(folder/'videos'/video/'belief.npz') as b:
            tracks=b['tracks'];obs=np.asarray(bank['observations']);ti=np.searchsorted(tracks,obs[:,1])
            assert np.array_equal(tracks[ti],obs[:,1])
            counts=np.bincount(ti,minlength=len(tracks))
            for qi,query in enumerate(b['queries'].tolist()):
                q=truth[video,query];queries+=1
                if not q['counts'].sum():continue
                present+=1;y=np.bincount(ti,weights=q['labels'],minlength=len(tracks))>=.5*counts
                for k in range(3):
                    mask=b['prior_strata'][qi]==k;asked=mask & b['asked_mask'][qi]
                    num[k]+=int((y & mask).sum());den[k]+=int(mask.sum())
                    asked_num[k]+=int((y & asked).sum());asked_den[k]+=int(asked.sum())
    result=dict(prediction_seal_sha256=digest,GT_rho=present/queries,
        GT_pi_all=[float(n/d) if d else None for n,d in zip(num,den)],
        GT_pi_asked=[float(n/d) if d else None for n,d in zip(asked_num,asked_den)],
        GT_all_den=den.tolist(),GT_asked_den=asked_den.tolist(),offline_only=True)
    atomic_json(path,result);return result


def report(execution):
    from scripts.active_eval import bootstrap,offline_labels,official_stats
    from scripts.active_attribution import attribute
    from scripts.report_active import saved_presence_decisions,final_confidence_summary,behavior_and_reliability
    public=execution.public;plan=json.loads((public/'registered_jobs.json').read_text())
    oldroot=Path(execution.manifest['prior_v7_public_root'])
    with (oldroot/'direction_summary.csv').open() as handle:old={r['pair']:r for r in csv.DictReader(handle)}
    metrics=[];presence=[];rounds=[];pools=[];streams=[];answers=[];glances=[];histograms=[]
    costs=[];recall=[];reliability=[];artifacts=[];intervals={};truths={};estimator_rows=[];slow_summary=[];sports=[]
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
        if target=='SportsMOT':
            subset=path/'sensitivity_418';subset_completion=subset/'official_completion.json'
            # The coordinator evaluates this dependency before closing its receipt.
            # Never rerun sports418() here: that would rewrite an existing seal.
            if subset_completion.exists():
                sub_seal=verify_predictions(subset);sub=json.loads(subset_completion.read_text())
                assert sub['prediction_seal_sha256']==file_sha(subset/'predictions_seal.json')
                assert sub['detailed_sha256']==file_sha(subset/'results/pedestrian_detailed.csv')
                assert sub_seal['original_full466_seal_sha256']==file_sha(path/'predictions_seal.json')
                sports.append(dict(pair=pair,method=method,HOTA_full466=result['HOTA_unrounded'],
                    HOTA_418=official_stats(subset)['HOTA_unrounded'],
                    **{k:sub['metrics'][k] for k in ('DetA','AssA','IDF1','IDSW')},
                    episodes=sub_seal['episode_count'],prediction_rows=sub_seal['prediction_rows'],
                    GT_protocol='same v4 GT content; original418 IDs only',
                    subset_path=str(subset),subset_seal_sha256=file_sha(subset/'predictions_seal.json')))
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
        attempts=[json.loads(p.read_text()) for p in path.glob('worker_attempts/*.json')]
        sessions=[json.loads(p.read_text()) for p in (path/'teacher_sessions').glob('*.json')]
        costs.append(dict(pair=pair,method=method,logical_questions=teacher['logical_calls'],
            new_calls=teacher.get('new_calls',0),cache_hits=teacher.get('cache_hits',0),
            replayed_answers=teacher.get('replayed_answers',0),teacher_GPU_hours=teacher.get('forward_seconds',0)/3600,
            teacher_equivalent_GPU_hours=teacher.get('equivalent_forward_seconds',0)/3600,
            LoRA_GPU_hours=seal['LoRA_GPU_seconds']/3600,rescore_GPU_hours=seal['rescore_GPU_seconds']/3600,
            stream_wall_seconds=seal['wall_seconds'],
            assigned_worker_wall_seconds=sum(a.get('wall_seconds',0) for a in attempts),
            failed_worker_attempts=sum(a['status']=='FAILED_RETAINED_FOR_RECOVERY' for a in attempts),
            peak_parent_allocated_bytes=max((a['parent_peak_allocated_bytes'] for a in attempts if a.get('parent_peak_allocated_bytes') is not None),default=None),
            peak_teacher_allocated_bytes=max((a['peak_allocated_bytes'] for a in sessions),default=None),
            configured_LoRA_trainable_parameters=0 if method=='MINUS_SLOW' else 24576,
            parameter_count_observed_in_updates=max((r['slow']['trainable_parameters'] for v in video_receipts for r in v['rounds'] if 'trainable_parameters' in r['slow']),default=None),
            optimizer_updates=sum(r['slow']['optimizer_updates'] for v in video_receipts for r in v['rounds']),
            GLANCE_BUDGET_SHORTFALL_videos=sum(v['GLANCE_BUDGET_SHORTFALL'] for v in video_receipts),
            unasked_queries=sum(v['glance_unanswered_queries'] for v in video_receipts),
            replay_teacher_dependency_in_MINUS=method=='SLOW_REPLAY'))
        gt_params=stratified_truth(path,meta,truth,public)
        end=video_receipts[-1]['EB']['after']
        shadow=video_receipts[-1]['EB']['v7_shadow']['after']
        for k in range(3):
            estimator_rows.append(dict(pair=pair,method=method,stratum=k,rho_hat=end['rho_hat'],
                pi_hat=end['pi_strata'][k],GT_rho=gt_params['GT_rho'],
                v7_shadow_rho=shadow['rho_hat'],v7_shadow_pi=shadow['pi_hat'],
                GT_pi_all=gt_params['GT_pi_all'][k],GT_pi_asked=gt_params['GT_pi_asked'][k],
                answered_tracks=end['tracks_per_stratum'][k],identification=end['identification_status'][k],
                presence_likelihood_evaluations=end['presence_likelihood_evaluations_per_stratum'][k],
                parameter_scope='MINUS video-start snapshot' if method=='SLOW_REPLAY' else 'video-end EM fit'))
        slow_rounds=[r['slow'] for v in video_receipts for r in v['rounds']]
        slow_summary.append(dict(pair=pair,method=method,
            updated_videos=sum(any(r['slow']['optimizer_updates'] for r in v['rounds']) for v in video_receipts),
            updated_rounds=sum(s['optimizer_updates']>0 for s in slow_rounds),
            skipped_rounds=sum(s['optimizer_updates']==0 for s in slow_rounds),
            status_counts=json.dumps(dict(Counter(s['status'] for s in slow_rounds)),sort_keys=True),
            optimizer_updates=sum(s['optimizer_updates'] for s in slow_rounds),
            actual_positive_samples=sum(s['actual_optimized_positive_samples'] for s in slow_rounds),
            actual_negative_samples=sum(s['actual_optimized_negative_samples'] for s in slow_rounds),
            balance_16_16_verified=all(
                r['actual_positive_samples']==16 and r['actual_negative_samples']==16
                for s in slow_rounds for r in s.get('records',[]) if r.get('optimizer_updated')),
            balance_verification_scope='actual optimizer steps only; zero-step jobs not a GPU update smoke'))
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
    v9_manifest=ROOT/'results/presence_active/redesign_v9_offline/manifest.json'
    if v9_manifest.exists() and json.loads(v9_manifest.read_text()).get('stage4_paused'):
        gate.update(stage4_authorized=False,stage4_policy='PAUSED_BY_USER_V9_INSTRUCTION',
                    formal_main_rule_unchanged=True)
    replays=[x for x in effects if x['method']=='SLOW_REPLAY' and x['pair'] in ('C1','C4')]
    slow_complete=len(replays)==2
    slow_pass=slow_complete and all(x['delta_HOTA']>=0 for x in replays) and any(x['CI95'][0]>0 for x in replays)
    slow_gate=dict(result='MAIN_METHOD' if slow_pass else 'APPENDIX' if slow_complete else 'NOT_EVALUATED',
                   compared_pairs=[r['pair'] for r in replays],both_nonnegative=all(r['delta_HOTA']>=0 for r in replays) if slow_complete else None)
    presence_summary=[];glance_summary=[]
    for pair,method in sorted({(r['pair'],r['method']) for r in presence}):
        group=[r for r in presence if r['pair']==pair and r['method']==method]
        presence_summary.append(dict(pair=pair,method=method,present=sum(r['query_present_GT'] for r in group),
            empty=sum(not r['query_present_GT'] for r in group),
            correct_empty=sum(r['output_absent'] and not r['query_present_GT'] for r in group),
            false_absent=sum(r['output_absent'] and r['query_present_GT'] for r in group),
            evidence_false_absent=sum(r['teacher_absent'] and r['query_present_GT'] for r in group),
            low_tentative=sum(r['low_tentative'] for r in group),
            PRIOR_BUDGET_EXHAUSTED=sum(r['decision_source']=='PRIOR_BUDGET_EXHAUSTED' for r in group),
            query_empty_rows=sum(r['selected_prediction_rows'] for r in group if not r['query_present_GT'])))
        for present in (False,True):
            samples=[r for r in glances if r['pair']==pair and r['method']==method and r['GT_present']==present]
            glance_summary.append(dict(pair=pair,method=method,GT_present=present,samples=len(samples),
                Yes=sum(r['yes'] for r in samples),yes_rate=sum(r['yes'] for r in samples)/len(samples) if samples else None))
    write_csv(public/'main.csv',metrics);write_csv(public/'direction_summary.csv',directions)
    write_csv(public/'presence_decisions.csv',presence);write_csv(public/'final_confidence.csv',final_confidence_summary(presence))
    write_csv(public/'slow_rounds.csv',pools);write_csv(public/'eb_stream.csv',streams)
    write_csv(public/'question_behavior.csv',rounds);write_csv(public/'glance_hits.csv',glances)
    write_csv(public/'posterior_histograms.csv',histograms);write_csv(public/'reliability_vs_GT.csv',reliability)
    write_csv(public/'costs.csv',costs);write_csv(public/'recall_compare.csv',recall);write_csv(public/'slow_effects.csv',effects)
    write_csv(public/'presence_summary.csv',presence_summary);write_csv(public/'glance_summary.csv',glance_summary)
    write_csv(public/'estimator_vs_GT.csv',estimator_rows)
    write_csv(public/'slow_summary.csv',slow_summary)
    write_csv(public/'sports_sensitivity.csv',sports)
    atomic_json(public/'bootstrap.json',intervals);atomic_json(public/'verdict.json',dict(main=gate,slow=slow_gate))
    atomic_json(public/'artifacts.json',dict(new=artifacts))
    plot_diagnostics(public,streams,rounds,histograms,estimator_rows,glance_summary)
    text='# v8：阶段2/3完整指标与独立归因\n\n'
    text+=f"已完成官方评测{len(metrics)}/{len(plan['jobs'])}项。主判据：{gate['result']}；慢分支：{slow_gate['result']}。未完成格为None，不冒充0或完成。\n\n"
    text+='## 1. 六方向主表（HOTA原精度）\n\n'+table(directions,list(directions[0]))+'\n\n'
    text+='### v8−v7成对不确定性\n\n'
    text+=table([dict(comparison=k,**v) for k,v in intervals.items() if k.endswith('_v8_minus_v7')],
        ['comparison','delta_HOTA','CI95','clusters'])+'\n\n'
    text+='Dance/SportsMOT按视频聚类2000次、seed 0；KITTI只有4视频，仅点估计。区间条件于本次已运行的数据流，不包含重新适应后的顺序或seed不确定性。\n\n'
    text+='## 2. 全部官方指标\n\n'+table(metrics,['pair','method','HOTA_unrounded','DetA','AssA','IDF1','IDSW','prediction_rows','query_empty_prediction_rows','Present_HOTA'])+'\n\n'
    text+='## 3. 存在性、估计器与行为\n\n'
    text+=table(presence_summary,['pair','method','present','empty','correct_empty','false_absent','evidence_false_absent','low_tentative','PRIOR_BUDGET_EXHAUSTED','query_empty_rows'])+'\n\n'
    text+='presence_decisions.csv分列EVIDENCE、PRIOR_BUDGET_EXHAUSTED及low_tentative；eb_stream.csv和estimator_vs_GT.csv保留rho/pi、>=200档位标记、EM诊断和v7影子计数。影子值是当前v8判定上的旧更新公式，不冒充历史v7闭环重跑。question_behavior.csv、glance_summary.csv、posterior_histograms.csv及相应图记录问题组成、新确认和GT分组第一眼命中；全池分布取轮初，提问分布取实际调度前，不混称同一时刻的随机样本。GT曲线和分组只在独立封存后报告进程读取。\n\n'
    text+='### 估计器终值与独立GT对照\n\n'
    text+=table(estimator_rows,['pair','method','stratum','rho_hat','GT_rho','pi_hat','GT_pi_all','GT_pi_asked','answered_tracks','identification','presence_likelihood_evaluations'])+'\n\n'
    text+='每档>=200仅表示通过A3的数据量门槛，不保证真实数据上的估计准确。完整CSV另列同一v8流上的v7影子值；REPLAY参数取MINUS视频开始时的固定值，不将其误称为回放重新估计的流终值。\n\n'
    text+='### SportsMOT敏感性\n\n'
    text+=table(sports,['pair','method','HOTA_full466','HOTA_418','episodes','prediction_rows'])+'\n\n'
    text+='仅限制为原418个序列ID，GT内容仍为统一重建的v4，不使用旧冲突模板；主判定始终用完整466条。\n\n'
    text+='## 4. 慢分支与固定存在性回放\n\n'+table(effects,['pair','method','delta_HOTA','CI95','clusters'])+'\n\n'
    text+='FULL−MINUS还包含选题反馈变化。REPLAY固定MINUS问答、rho/pi及每轮六个存在性数组，允许LoRA通过打分/f/快校准影响身份后验；最终存在性mask仍固定。slow_rounds.csv分列各池、跳过原因和实际16/16优化暴露，不由轮末标签倒推。列表CE/NONE损失已删除，统一BCE。\n\n'
    text+=table(slow_summary,['pair','method','updated_videos','updated_rounds','skipped_rounds','optimizer_updates','actual_positive_samples','actual_negative_samples','balance_16_16_verified'])+'\n\n'
    text+='零步数的16/16检查只表示没有违反断言，不构成真实慢分支更新通过。slow_summary.csv保留逐原因计数；回放逐字节完整性以完整队列结束后的validation.json为准。\n\n'
    text+='## 5. 高置信错负与GT恢复\n\n'+table(recall,['pair','method','GT_recall','output_rows','recovered_GT','lost_GT','net_FP_vs_Native'])+'\n\n'
    text+='final_confidence.csv保留v6的.05高置信错负口径。GT恢复/FP来自独立IoU>=.5最大基数匹配，不冒充官方HOTA匹配；新增行与不变行的重分配已分列。DS/GT特异度是所问集合的事后诊断，不反馈适应。\n\n'
    text+='## 6. 成本\n\n'+table(costs,['pair','method','logical_questions','new_calls','cache_hits','replayed_answers','teacher_GPU_hours','LoRA_GPU_hours','rescore_GPU_hours','assigned_worker_wall_seconds','optimizer_updates'])+'\n\n'
    text+='共享设备相位用时不是GPU利用率积分；assigned_worker_wall_seconds包含同一流的pilot与正式恢复尝试，stream_wall_seconds是最后一次流调用时长，不把后者当独立部署总墙钟。冻结bank、StageA与固定MOT准备成本未重新测量，不等于0。REPLAY依赖MINUS教师问答成本，不将它描述成独立无教师系统。MINUS没有神经LoRA更新，但仍做CPU快速校准；不是零优化。\n\n'
    text+='## 7. 阶段0与协议记录\n\n完整0b–0e表、合成验收、负面反事实与A2/A3诊断见[阶段0报告](amendments_A2_A3/REPORT.md)。本报告只读取封存预测；GT从未进入生成进程。视频内使用全片评分与主动关键帧，是视频级转导式适应，不冒充严格逐帧因果或盲测集。v7与旧200迭代失败不覆盖。主判据要求六方向中至少5胜并满足C1>=35.724、C6>=34.469，不沿用v7的4胜规则。\n\n'
    text+=f"已关闭的{len(metrics)}项预测，其生成进程GT打开尝试合计为{sum(r['GT_open_attempts'] for r in metrics)}；预测路径、episode数和seal SHA见artifacts.json。尚未关闭的作业不纳入该合计。\n"
    (public/'REPORT.md').write_text(text)
    return dict(completed=len(metrics),expected=len(plan['jobs']),main=gate,slow=slow_gate)


def plot_diagnostics(public,streams,rounds,histograms,truth,glances):
    if not streams:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for field in ('rho_hat','pi_0','pi_1','pi_2'):
        fig,axes=plt.subplots(2,3,figsize=(15,7))
        for ax,pair in zip(axes.flat,PAIRS):
            for method in sorted({r['method'] for r in streams if r['pair']==pair}):
                values=[r for r in streams if r['pair']==pair and r['method']==method]
                line,=ax.plot([r['video_index']+1 for r in values],[r[field] for r in values],label=method)
                target=next(r for r in truth if r['pair']==pair and r['method']==method and (field=='rho_hat' or r['stratum']==int(field[-1])))
                gt=target['GT_rho'] if field=='rho_hat' else target['GT_pi_all']
                if gt is not None:ax.axhline(gt,color=line.get_color(),ls='--',alpha=.6)
            ax.set(title=pair,xlabel='Video ID order',ylabel=field,ylim=(0,1));ax.grid(alpha=.2)
            if ax.lines:ax.legend(fontsize=6)
        fig.suptitle('GT dashed lines are offline diagnostics only');fig.tight_layout()
        fig.savefig(public/(field+'_stream.png'),dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,7))
    for ax,pair in zip(axes.flat,PAIRS):
        for method in sorted({r['method'] for r in rounds if r['pair']==pair}):
            values=[r for r in rounds if r['pair']==pair and r['method']==method]
            videos=sorted({r['video_index'] for r in values})
            ax.plot(videos,[sum(r['actual_questions'] for r in values if r['video_index']==v) for v in videos],label=method+' calls')
            ax.plot(videos,[sum(r['new_confirmed'] for r in values if r['video_index']==v) for v in videos],ls='--',label=method+' new')
        ax.set(title=pair,xlabel='Video index');ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=5)
    fig.tight_layout();fig.savefig(public/'questions_and_confirmations.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,7))
    for ax,pair in zip(axes.flat,PAIRS):
        for scope,key in [('asked','asked_count'),('round-start pool','all_pair_count')]:
            values=[r for r in histograms if r['pair']==pair and r['method']=='MINUS_SLOW']
            counts=np.array([sum(r[key] for r in values if r['bin_left']==i/20) for i in range(20)],dtype=float)
            if counts.sum():ax.step(np.arange(20)/20,counts/counts.sum(),where='post',label=scope)
        ax.set(title=pair,xlabel='Posterior probability',ylabel='Descriptive mass');ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(public/'acquisition_posterior_histogram.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,7))
    kinds=['GLANCE','PRESENCE','PRESENCE_COVER','IDENTITY','POSITIVE_RECHECK']
    for ax,pair in zip(axes.flat,PAIRS):
        bottom=np.zeros(4)
        values=[r for r in rounds if r['pair']==pair and r['method']=='MINUS_SLOW']
        for kind in kinds:
            fractions=[]
            for r in range(1,5):
                rows=[x for x in values if x['round']==r];total=sum(x['actual_questions'] for x in rows)
                fractions.append(sum(x[kind+'_count'] for x in rows)/total if total else 0.)
            ax.bar(range(1,5),fractions,bottom=bottom,label=kind);bottom+=fractions
        ax.set(title=pair,xlabel='Acquisition round',ylabel='Question fraction',ylim=(0,1))
    axes.flat[0].legend(fontsize=6);fig.tight_layout();fig.savefig(public/'question_type_fractions.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,7))
    for ax,pair in zip(axes.flat,PAIRS):
        methods=sorted({r['method'] for r in glances if r['pair']==pair})
        for offset,present in [(-.18,False),(.18,True)]:
            values=[next(r for r in glances if r['pair']==pair and r['method']==m and r['GT_present']==present) for m in methods]
            ax.bar(np.arange(len(methods))+offset,[v['yes_rate'] if v['yes_rate'] is not None else np.nan for v in values],
                   width=.36,label='Query-Present' if present else 'Query-Empty')
        ax.set(title=pair,ylabel='First GLANCE Yes rate',ylim=(0,1))
        ax.set_xticks(range(len(methods)),methods,rotation=20,fontsize=6)
        if methods:ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(public/'glance_yes_rate.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--partial',action='store_true');p.parse_args()
    execution,_=authorize_active();print(json.dumps(report(execution)),flush=True)
