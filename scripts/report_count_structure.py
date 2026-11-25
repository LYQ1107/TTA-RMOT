"""All registered outcomes, offline video uncertainty, Q1–Q4, then STOP."""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.count_structure_common import *
from scripts.pf_diag_common import official_metrics,official_video_values
from scripts.analyze_pf_video_uncertainty import HOTA,unpack,paired_clusters,pooled_hota
from scripts.analyze_psr_matrix import official_query_groups
from flex_uda.data.manifests import read_jsonl

def table(headers,rows):
    def fmt(x):
        if x is None:return 'N/A'
        if isinstance(x,(float,np.floating)):return f'{x:.6f}'
        return str(x)
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
        ['| '+' | '.join(map(fmt,r))+' |' for r in rows])+'\n'

def report():
    started=time.perf_counter();setup();require_plan()
    complete=json.loads((RUN/'official_completion.json').read_text())
    if complete['status']!='ALL36_OFFICIAL_PREDICTION_SETS_COMPLETE':raise ValueError('finish every registered official job')
    structure=json.loads((PUBLIC/'structure_stats.json').read_text());probe=json.loads((PUBLIC/'teacher_probe.json').read_text())
    selector_plan=json.loads((RUN/'selectors_seal.json').read_text());probe_seal=json.loads((RUN/'probe_seal.json').read_text())
    meta,_,_=load_inputs(verify=False);methods=list(complete['methods']);metrics={};values={};sealed={};detail={};videos=None
    present={(e['video'],e['expression']):e['GT_observations'] for e in structure['episodes']}
    train={' '.join(r['query_text'].lower().split()) for r in read_jsonl('/data2/user/TTA-RMOT-assets/train2val/20261031/runtime_train.jsonl')}
    seen={(q['video_id'],q['expression_id']):' '.join(q['query_text'].lower().split()) in train for q in meta['episodes']}
    hota=HOTA({'PRINT_CONFIG':False});rows={};per_query=[]
    for m in methods:
        sealed[m]=v2.verify_seal(m);metrics[m]=official_metrics(v2.EVAL/m/'results')
        v,stat=official_video_values(v2.EVAL/m/'results/pedestrian_detailed.csv')
        if videos is not None and videos!=v:raise ValueError('video cluster mismatch')
        videos=v;values[m]=stat
        with (v2.EVAL/m/'results/pedestrian_detailed.csv').open() as h:csv_rows=list(csv.DictReader(h))
        combined=next(r for r in csv_rows if r['seq']=='COMBINED');detail[m]={r['seq']:r for r in csv_rows if r['seq']!='COMBINED'}
        full=float(pooled_hota(stat,np.ones(25)))
        if not np.isclose(full,100*float(combined['HOTA___AUC']),atol=1e-8,rtol=0):raise ValueError('official TP-weighted pooling mismatch')
        if sealed[m]['prediction_rows']!=metrics[m]['Dets']:raise ValueError('prediction versus official count mismatch')
        grouped=official_query_groups(v2.EVAL/m,dict(episodes=[dict(video=v,expression=e,gt_observations=n) for (v,e),n in present.items()]))
        scoped={}
        for label,flag in [('seen',True),('unseen',False)]:
            chosen={v+'+'+e:unpack(r,hota) for (v,e),value in seen.items() if value==flag and (r:=detail[m].get(v+'+'+e)) is not None}
            r=hota.combine_sequences(chosen);scoped[label]=100*float(np.mean(r['HOTA']))
        distribution=np.asarray([e['selected_rows'] for e in sealed[m]['episodes']])
        rows[m]=dict(method=m,evaluation_role='ORACLE / GT, only offline evaluation',**metrics[m],HOTA_unrounded=full,
            prediction_rows=sealed[m]['prediction_rows'],query_empty_prediction_rows=sum(e['selected_rows'] for e in sealed[m]['episodes'] if not present[e['video'],e['expression']]),
            Present_HOTA=grouped['query_present']['HOTA'],Seen_HOTA=scoped['seen'],Unseen_HOTA=scoped['unseen'],
            min_query_rows=int(distribution.min()),p10_query_rows=float(np.percentile(distribution,10)),
            p50_query_rows=float(np.percentile(distribution,50)),p90_query_rows=float(np.percentile(distribution,90)),
            max_query_rows=int(distribution.max()),mean_query_rows=float(distribution.mean()),zero_output_queries=int((distribution==0).sum()))
        if '__C' in m:
            plan=next(r for r in selector_plan['methods'] if r['method']==m)
            rows[m].update(margin=plan['margin'],selector=plan['selector'],uses_future_frames_selector=plan['uses_future_frames'],
                generation_role='GT_FREE_SELECTOR_NO_NEW_MODEL_TRAINING',target_GT_open_attempts=plan['target_GT_open_attempts'],
                source_lineage='HISTORICAL_SAME_VAL_TEACHER_TRAINING' if plan['margin']=='SAME_SET_D1' else 'FROZEN_SOURCE' if plan['margin']=='FROZEN' else 'TRAIN40_TO_VAL25_WITH8_SHARED_PHRASES')
            per_query.extend(dict(method=m,video=e['video'],expression=e['expression'],output_rows=e['selected_rows']) for e in sealed[m]['episodes'])
        elif m.startswith('ORACLE__'):rows[m].update(generation_role='ORACLE / GT, DIAGNOSTIC ONLY')
    oracle_rows=[rows['ORACLE__'+name] for name in ORACLES]
    selector_rows=[rows[m+'__'+c] for m in MARGINS for c in SELECTORS]
    v2.write_csv(PUBLIC/'granularity.csv',oracle_rows);v2.write_csv(PUBLIC/'selectors.csv',selector_rows)
    v2.write_csv(PUBLIC/'selector_per_query.csv',per_query)
    draws=np.random.default_rng(0).multinomial(25,np.ones(25)/25,size=2000);comparisons={}
    def compare(a,b):
        va,vb=values[a],values[b];samples=paired_clusters(va,vb,draws);per=paired_clusters(va,vb,np.eye(25))
        comparisons[a+'_minus_'+b]=dict(first=a,second=b,delta_HOTA=float(paired_clusters(va,vb,np.ones(25))),
            CI95=np.percentile(samples,[2.5,97.5]).tolist(),positive_videos=int((per>1e-9).sum()),
            negative_videos=int((per < -1e-9).sum()),tied_videos=int((np.abs(per)<=1e-9).sum()),
            per_video=[dict(video=v,delta_HOTA=float(d)) for v,d in zip(videos,per)])
    for a,b in [('ORACLE__V_MEAN','FROZEN'),('ORACLE__F_MEAN','ORACLE__V_MEAN'),
        ('ORACLE__TRACK_ORACLE','ORACLE__V_MEAN'),('ORACLE__F_FRAME','ORACLE__F_MEAN')]:compare(a,b)
    for m in MARGINS:
        for c in SELECTORS:
            for baseline in ('FROZEN','B1_SRC_RATE_GLOBAL'):compare(m+'__'+c,baseline)
    frozen_passes=[c for c in SELECTORS if comparisons['FROZEN__'+c+'_minus_B1_SRC_RATE_GLOBAL']['CI95'][0]>0]
    passes=[m+'__'+c for m in MARGINS for c in SELECTORS if comparisons[m+'__'+c+'_minus_B1_SRC_RATE_GLOBAL']['CI95'][0]>0]
    verdict=dict(granularity=granularity_verdict(rows['ORACLE__V_MEAN']['HOTA_unrounded'],
        rows['ORACLE__TRACK_ORACLE']['HOTA_unrounded'],rows['ORACLE__F_MEAN']['HOTA_unrounded']),
        whole_outfit='整组服装结构成立' if structure['whole_outfit_structure_holds'] else '整组服装结构的预注册条件未满足',
        mutual_exclusion='句子互斥结构成立' if structure['mutual_exclusion_structure_holds'] else '句子互斥结构的预注册条件未满足',
        structure_beyond_global_calibration='结构信息带来超出全局校准的收益' if passes else '预注册条件未满足',
        FROZEN_selectors_satisfying_CI_lower_above0=frozen_passes,
        all28_selectors_satisfying_CI_lower_above0=passes,
        literal_any_selector_scope='all28 preregistered selectors; Frozen-only subset separately disclosed; learned margins are not untrained Frozen',
        FROZEN_structure_only_condition_satisfied=bool(frozen_passes),no_best_selector_or_teacher_recommended=True,
        granularity_threshold_definition='user fixed31.771=FROZEN ranking with OFFICIAL_COMPLETE k, not native official32.139')
    atomic_json(PUBLIC/'bootstrap.json',dict(role='ORACLE / GT, only offline evaluation',clusters=25,samples=2000,seed=0,
        pooling='unchanged v2 official TP-weighted sufficient statistics',comparisons=comparisons,verdict=verdict,
        uncertainty='paired-video descriptive intervals; correlated queries/frames not independent; Val is prior development data;60 comparisons all disclosed, no multiplicity correction'))
    with tarfile.open(PUBLIC/'official_metrics.tar.gz','w:gz') as archive:
        official_index={}
        for m in methods:
            official_index[m]={}
            for name in ('pedestrian_summary.txt','pedestrian_detailed.csv'):
                p=v2.EVAL/m/'results'/name;archive.add(p,arcname=m+'/'+name)
                official_index[m][name]=dict(path=str(p),sha256=file_sha(p))
    atomic_json(PUBLIC/'official_metrics_index.json',dict(files=official_index,archive_sha256=file_sha(PUBLIC/'official_metrics.tar.gz')))
    with gzip.open(PUBLIC/'prediction_files.csv.gz','wt') as h:
        writer=csv.DictWriter(h,fieldnames=['method','video','expression','prediction_path','prediction_sha256','selected_rows']);writer.writeheader()
        for m,s in sealed.items():
            for e in s['episodes']:writer.writerow(dict(method=m,**{k:e[k] for k in writer.fieldnames if k!='method'}))
    budgets=[]
    for model,r in probe_seal['models'].items():
        smoke=json.loads((RUN/'probe'/model/'smoke_receipt.json').read_text())
        for method,c in r['budgets'].items():
            s=smoke['budgets'][method];budgets.append(dict(model=model,method=method,logical_calls=c['planned_logical_calls'],
                actual_new_calls_including_smoke=c['new_logical_calls']+s['new_logical_calls'],
                model_forward_calls=c['model_forward_calls']+s['model_forward_calls'],
                forward_path_seconds=c['forward_path_seconds']+s['forward_path_seconds'],
                peak_allocated_GiB=max(c['peak_allocated_bytes'],s['peak_allocated_bytes'])/2**30,
                parse_failures=c['parse_failures']))
    atomic_json(PUBLIC/'costs.json',dict(new_GPU_process_hours=probe_seal['GPU_process_hours'],
        teacher_parent_wall_seconds=probe_seal['actual_wall_seconds'],teacher_methods=budgets,
        CPU_source_Platt_seconds=json.loads((RUN/'source_calibration.json').read_text())['CPU_wall_seconds'],
        CPU_selectors_seconds=selector_plan['CPU_wall_seconds'],CPU_oracle_seconds=json.loads((RUN/'oracle_seal.json').read_text())['CPU_wall_seconds'],
        official_new_jobs=complete['new_jobs'],official_exact_reference_reuses=complete['reused_jobs'],
        official_sum_job_wall_seconds=sum(r['worker_wall_seconds'] for r in complete['methods'].values()),
        official_actual_batch_wall_seconds_not_recorded=True,CPU_failed_source_attempt_seconds_not_recorded=True,
        student_updates=0,teacher_updates=0,maximum_physical_GPUs=1,
        interpretation='GPU-process elapsed includes CPU/startup, not utilization integral; forward path includes original processor/synchronization timing',
        source_two_scalar_Platt_fit_is_the_only_authorized_parameter_fit=True,
        selectors_inherit_original_margin_model_inference_and_any_historical_training_cost=True))
    large=[]
    for p in sorted((RUN/'masks').glob('*.npy')):
        large.append(dict(path=str(p),bytes=p.stat().st_size,sha256=file_sha(p)))
    for model,r in probe_seal['models'].items():
        for name in ('outputs.jsonl','progress.sqlite'):
            p=RUN/'probe'/model/name;large.append(dict(path=str(p),bytes=p.stat().st_size,sha256=file_sha(p)))
        for name in ('receipt.json','smoke_receipt.json','GT_free_agreements.json','access.json','smoke_access.json'):
            shutil.copy2(RUN/'probe'/model/name,PUBLIC/(model+'_'+name))
    for p in (RUN/'probe_offline_frames.json',RUN/'oracle_seal.json'):
        if p.name=='oracle_seal.json':shutil.copy2(p,PUBLIC/p.name)
        else:large.append(dict(path=str(p),bytes=p.stat().st_size,sha256=file_sha(p)))
    for p in sorted((RUN/'archives/report_before_scope_clarification').glob('*')):
        if p.is_file():large.append(dict(path=str(p),bytes=p.stat().st_size,sha256=file_sha(p)))
    atomic_json(PUBLIC/'server_artifacts.json',dict(large_files_not_in_git=True,root=str(RUN),files=large,
        sealed_input_margins='input_audit.json',raw_MOT_source='official source calibration inputs; target runtime manifest',
        no_new_checkpoint_or_dense_teacher_labels=True))
    with tarfile.open(PUBLIC/'stage_logs.tar.gz','w:gz') as archive:
        for p in sorted(RUN.rglob('*')):
            if p.is_file() and ((any(part in ('official_jobs','probe_launchers','logs') for part in p.parts) and p.suffix in ('.log','.txt','.json')) or p.name=='trackeval.log'):
                archive.add(p,arcname=str(p.relative_to(RUN)))
    gt_fields=('HOTA','DetA','AssA','IDF1','IDSW','prediction_rows','query_empty_prediction_rows','Present_HOTA','Seen_HOTA','Unseen_HOTA')
    lines=['# 数量/存在性来源、结构选择器与教师问法诊断','',
        '状态：COMPLETE_STOP；起点61a388e8，分支research/v2dance-count-structure-20261031。学生/教师更新0，不追加实验。','',
        '所有HOTA、结构标签、probe准确率与bootstrap均为【ORACLE / GT，仅离线诊断/评测】；oracle预测用GT，C系列预测生成不读目标GT，两者不可混称。',
        '全部选择器/问法先固定并发布PREREG.md，28套GT-free预测与probe均封存后才打开Dance GT；不选最佳选择器、不选教师问法。','',
        '## Q1–Q4（各不超过五句）','',
        f'Q1：按用户31.771固定门槛，判为“{verdict["granularity"]}”。V_MEAN/轨迹oracle/F_MEAN的HOTA分别为'+
        '/'.join(f'{rows["ORACLE__"+n]["HOTA"]:.4f}' for n in ('V_MEAN','TRACK_ORACLE','F_MEAN'))+'；这些是GT提供数量/存在性的诊断，不是可部署方法。','',
        f'Q2：整组比例为{structure["whole_outfit_qualifying_episodes"]}/{structure["present_episodes"]}={structure["whole_outfit_fraction"]:.6f}，{verdict["whole_outfit"]}；'+
        f'轨迹≤1正例句子的比例为{1-structure["mutual_exclusion_fractions"]["two_or_more"]:.6f}，{verdict["mutual_exclusion"]}。C2/C3/C4以及T2单选都受该互斥假设限制，不能当通用RMOT事实。','',
        'Q3：FROZEN上七个选择器的HOTA依次为'+', '.join(f'{c}={rows["FROZEN__"+c]["HOTA"]:.4f}' for c in SELECTORS)+'。'+
        f'Frozen子集没有CI下界>0的选择器；按“任一选择器”的全部28套字面范围，{verdict["structure_beyond_global_calibration"]}，但满足项依赖已训练margin，不能据此宣称Frozen结构单独超过校准。全部28套和60个配对比较原样报告，不推荐赢家。','']
    all_probe={r['method']:r for r in probe['rows'] if r['model']=='2B' and r['text_scope']=='ALL'};t1=all_probe['T1_CROP_YESNO']
    q4='Q4：固定2B探针上，'
    for method in ('T2_CROP_CHOICE','T3_FRAME_SOM'):
        r=all_probe[method];q4+=f'{method}相对T1的存在性准确率Δ={r["presence_accuracy_all"]-t1["presence_accuracy_all"]:+.6f}、有效判断数量MAEΔ={r["quantity_MAE_valid"]-t1["quantity_MAE_valid"]:+.6f}；'
    t3_failures=all_probe['T3_FRAME_SOM']['unknown_frame_queries']
    q4+=f'三种问法总GPU-process成本{probe_seal["GPU_process_hours"]:.6f}小时，T3失败{t3_failures}/850保留UNKNOWN，'+('8B无本地可核验权重而跳过。' if probe_seal['skipped_8B'] else '已有8B的T2/T3也完整上报。')
    q4+='这是5视频50帧描述性探针，不是完整HOTA；不据此选定教师，下一步采用哪种由用户决定。'
    q4+=f'T2候选正例召回{all_probe["T2_CROP_CHOICE"]["positive_recall_valid"]:.6f}低于T1的{t1["positive_recall_valid"]:.6f}，且全空存在性基线已为{1-t1["GT_present_frame_queries"]/t1["frame_queries"]:.6f}，不能仅凭总体准确率宣布教师更可靠。'
    lines+=[q4,'','## 1A. 粒度分解【ORACLE / GT，仅诊断】','',
        table(['格子',*gt_fields],[[r['method'],*[r[k] for k in gt_fields]] for r in oracle_rows]),
        'F_FRAME：425个原始预测文件逐SHA相同，201009行/HOTA45.794复现通过；对应旧FROZEN×ORACLE_GT。31.771是用户指定Frozen排序@官方k水平，不是原生同域官方32.139。','',
        '## 1B. 结构统计【ORACLE / GT，仅描述】','',
        table(['句子','出现视频','正轨迹比mean','p10','p50','p90','GT帧比例mean(出现视频)'],
            [[r['expression'],r['present_videos'],*[r['positive_track_ratio_given_present'][k] for k in ('mean','p10','p50','p90')],
                r['GT_frame_fraction_given_present']['mean']] for r in structure['query_statistics']]),
        '轨迹正例句子数0/1/≥2比例：'+json.dumps(structure['mutual_exclusion_fractions'],ensure_ascii=False)+'。',
        '每视频出现过的不同句子数分布：'+json.dumps(dict(Counter(r['present_queries'] for r in structure['distinct_present_queries_per_video'])))+'；逐帧同时出现句子数分布：'+json.dumps(structure['frame_simultaneous_present_query_count_histogram'])+'。','',
        '### 距离同q最近的实际Naive教师判断帧','',
        table(['margin','距离帧桶','帧数','GT','预测','TP','FP','FN','precision','recall'],
            [[r[k] for k in ('margin','distance_to_nearest_actual_Naive_teacher_frame','frames','GT','predictions','TP','FP','FN','precision','recall')] for r in structure['temporal_distance_buckets']]),
        '全部原始Naive8500判断帧用于距离，按全query完整时间轴统计；旧SAME_SET_D1是同Val训练，这个距离分析不能证明跨视频泛化。','',
        '## 1C. 全部结构选择器（生成GT-free，指标【ORACLE / GT，仅评测】）','',
        table(['margin','选择器',*gt_fields,'future(selector)','GT打开'],
            [[r['margin'],r['selector'],*[r[k] for k in gt_fields],r['uses_future_frames_selector'],r['target_GT_open_attempts']] for r in selector_rows]),
        '每q输出分布min/p10/p50/p90/max/mean/零输出数见selectors.csv，全425明细见selector_per_query.csv。',
        'C1c/C2的无未来仅指选择规则；SAME_SET_D1输入来自旧同Val教师训练，不是完整因果/盲测系统。C2/C3/C4强加每人单句假设；C4源均值和C5源Platt均来自Frozen，不能假设适应后logit分布满足label shift。',
        'C4三个学生库的全空输出仍完整上报；未重新校准或强制补正例。','',
        '## 2. 教师问法（预测GT-free，评价【ORACLE / GT，仅描述】）','',
        table(['模型','问法','文本组','状态','AUROC(valid)','acc(valid)','recall(valid)','FPR(valid)','存在准确率(失败计错)','数量MAE(valid)','数量完全正确率(失败计错)','未知帧query'],
            [[r[k] for k in ('model','method','text_scope','status','AUROC_valid','accuracy_at_0p5_valid','positive_recall_valid',
                'negative_FPR_valid','presence_accuracy_all','quantity_MAE_valid','quantity_exact_accuracy_all','unknown_frame_queries')] for r in probe['rows']]),
        f'6426候选–句子对、850帧–句子；T2的18类概率用p>.5二元化，R比例另按argmax。T3为0/1硬分数，其AUROC不能称概率校准。所有{t3_failures}个解析失败原始文本保留，条件指标附有效覆盖；全请求准确率将失败计错，UNKNOWN不补NEG。',
        'GT-free两两/相邻探针帧同ID一致率与T2 none比例见probe_GT_free_agreements.json；高一致率可来自大量共同否定，不是准确率。','',
        '## 3. 25视频bootstrap【ORACLE / GT，仅诊断】','',
        table(['配对比较','ΔHOTA','95% CI','上升/下降/平'],[[name,r['delta_HOTA'],str(r['CI95']),f"{r['positive_videos']}/{r['negative_videos']}/{r['tied_videos']}"] for name,r in comparisons.items()]),
        '2000次、seed0，完整视频聚类、官方TP加权HOTA重组；不是帧独立推断。全部60比较原样披露，按用户规则描述，不做多重比较校正或泛化显著性承诺。','',
        '## 4. GT隔离、成本、失败与恢复','',
        '源Platt、C28、教师的真实Dance GT打开尝试均0；源GT是明确允许的2参数校准输入，pi_s为源GT匹配比例.150564而非输出正例率.318754。新学生/教师更新0，原YesNo后端/裁剪/vendor/历史结果不改。',
        table(['模型','问法','逻辑调用','实际新调用含smoke','模型forward次数','forward路径秒数','峰值GiB','解析失败'],
            [[r[k] for k in ('model','method','logical_calls','actual_new_calls_including_smoke','model_forward_calls','forward_path_seconds','peak_allocated_GiB','parse_failures')] for r in budgets]),
        f'单张GPU（记录为{sorted({r["physical_gpu"] for r in probe_seal["models"].values()})}）；完整smoke+probe进程elapsed合计{probe_seal["GPU_process_hours"]:.6f}GPU-process小时，不是利用率积分。源校准/C选择/Oracle/官方CPU时间见costs.json；C选择器仍继承原margin模型推理及历史训练成本，不称免费独立系统。',
        '首个源校准因预测ID/GT ID混淆拟合前失败，已按真实预测MOT的frame−1和ped ID offset修复、全2355528行不跳过；失败和修复见EXECUTION_NOTES.md，原始源码及日志保留。',
        '首次CPU失败耗时及整批官方CPU实际wall未单独保存，标为未记录，不填0或用job wall之和冒充并行wall；没有未计入的GPU重试。',
        '所有原始官方summary/detailed在official_metrics.tar.gz，阶段stdout在stage_logs.tar.gz；预测路径/SHA在prediction_files.csv.gz，大数组/原始teacher缓存路径SHA在server_artifacts.json。',
        'Train/Val整句有8类重合，Val已被旧研究反复开发；见过/没见过分组只做报告。','',
        '**完成并停止：不推荐最佳选择器或教师，不训练、不补标、不调阈值、不启动新方法。**','']
    agreement=json.loads((PUBLIC/'probe_GT_free_agreements.json').read_text())
    lines+=['## 5. 教师进程内计算的GT-free一致性','',
        table(['问法对','计划候选query对','有效对','二元一致率'],
            [[name,r['planned_pairs'],r['valid_pairs'],r['agreement']] for name,r in agreement['pairwise'].items()]),
        table(['问法','相邻探针帧同ID计划对','有效对','一致率'],
            [[name,r['planned_pairs'],r['valid_pairs'],r['agreement']] for name,r in agreement['adjacent_probe_frame_same_track'].items()]),
        f'T2 argmax为R的比例={agreement["T2_none_argmax_fraction"]:.6f}（1/378）；“相邻”指相邻采样帧，不是相邻视频帧。','']
    interpretation=PUBLIC/'INTERPRETATION.md'
    if interpretation.exists():lines+=['## 6. 科研解释（不新增判定规则）','',interpretation.read_text(),'']
    (PUBLIC/'REPORT.md').write_text('\n'.join(lines))
    atomic_json(PUBLIC/'completion.json',dict(status='COMPLETE_STOP_NO_TRAINING_NO_SELECTION',verdict=verdict,
        official_prediction_sets=len(methods),registered_selectors=28,registered_oracles=6,teacher_models=list(probe_seal['models']),
        target_GT_open_attempts_in_GT_free_generators=0,student_updates=0,teacher_updates=0,
        report_sha256=file_sha(PUBLIC/'REPORT.md'),CPU_report_seconds=time.perf_counter()-started))
    print(json.dumps(dict(event='COUNT_FINAL_REPORT_COMPLETE',verdict=verdict),ensure_ascii=False),flush=True)

if __name__=='__main__':report()
