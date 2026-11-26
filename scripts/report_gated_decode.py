"""Offline-only scientific accounting after every prediction/gate is sealed."""
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import sys
import tarfile
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.gated_common import *
from scripts.evaluate_gated_decode import before_GT
from scripts.pf_diag_common import official_metrics,official_video_values
from scripts.analyze_pf_video_uncertainty import HOTA,unpack,paired_clusters,pooled_hota
from scripts.report_count_structure import table
from flex_uda.data.manifests import read_jsonl

def teacher_costs():
    full=json.loads((RUN/'teacher/receipt.json').read_text());smoke=json.loads((RUN/'teacher/smoke_receipt.json').read_text())
    launch=json.loads((RUN/'teacher_launch_completion.json').read_text());plan=json.loads((RUN/'teacher_plan.json').read_text())
    aggregate={};by_video=[]
    for dataset in ('Dance','KITTI'):
        for mode in ('normal','early'):
            selected=[r for r in full['counts'] if r['dataset']==dataset and r['gate_mode']==mode]
            if not selected:continue
            for video in sorted({r['video'] for r in selected}):
                rr=[r for r in selected if r['video']==video]
                by_video.append(dict(dataset=dataset,video=video,gate_mode=mode,frames=len(rr),
                    requested_calls=sum(r['requested_calls'] for r in rr),new_calls=sum(r['new_calls'] for r in rr),cache_hits=sum(r['cache_hits'] for r in rr),
                    forward_seconds=sum(r['forward_seconds'] for r in rr),peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rr)))
            aggregate[dataset+'_'+mode]=dict(requested_calls=sum(r['requested_calls'] for r in selected),new_calls=sum(r['new_calls'] for r in selected),
                cache_hits=sum(r['cache_hits'] for r in selected),forward_seconds=sum(r['forward_seconds'] for r in selected),peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in selected))
    # Batch seconds counted ONCE per batch, NOT summed for each output row.
    all_counts=smoke['counts']+full['counts'];calls=sum(r['new_calls'] for r in all_counts);seconds=sum(r['forward_seconds'] for r in all_counts)
    dance=[r for r in all_counts if r['dataset']=='Dance'];dance_calls=sum(r['new_calls'] for r in dance);dance_seconds=sum(r['forward_seconds'] for r in dance)
    if calls!=full['cache_unique_calls']:raise ValueError('teacher attempted-cost accounting needs failed-attempt recovery ledger')
    result=dict(status='COMPLETE_COST_ACCOUNTING',actual_unique_forwarded_calls=calls,requested_gate_judgements=plan['planned_logical_calls'],
        forward_batches=sum(r['forward_batches'] for r in all_counts),forward_seconds=seconds,mean_amortized_forward_seconds=seconds/calls,
        Dance_forward_calls=dance_calls,Dance_forward_seconds=dance_seconds,Dance_amortized_forward_seconds=dance_seconds/dance_calls,
        exhaustive_Dance_candidate_query_pairs=3588445,exhaustive_Dance_forward_GPU_hours=3588445*(dance_seconds/dance_calls)/3600,
        exhaustive_all_domain_mean_GPU_hours=3588445*(seconds/calls)/3600,
        GPU_process_hours=launch['GPU_process_hours'],worker_attempts=launch['jobs'],smoke_receipt=smoke,
        full_worker_seconds=full['worker_seconds'],full_frozen_seal=full['frozen'],by_dataset_mode=aggregate,by_video=by_video,
        peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in all_counts),maximum_physical_GPUs=1,
        source_candidate_truncation=plan['source_budget'],GT_open_attempts=full['GT_open_attempts']+smoke['GT_open_attempts'],
        semantics='Forward timings include original synchronized upstream preprocessing/forward path, batches amortized; process hours also include model loading/hash/cropping/cache I/O. Teacher generated once, reused across score sources. No training cost incurred; prior student training inherited, not priced as free.',
        inherited_costs='Fixed MOT frontend and all six FlexHook margin banks were already computed. Their prior GPU inference/training is excluded from this round incremental costs, not zero standalone-system cost. GPU_calls=0 inside file-only prediction seals means CPU decoding stage only; teacher calls are reported separately.',
        extrapolation_caveat='Throughput estimate, not a measured exhaustive run; excludes repeated model loads and full-data cache/image I/O. No full-video teacher RMOT system was run.')
    atomic_json(PUBLIC/'teacher_costs.json',result);v2.write_csv(PUBLIC/'teacher_per_video.csv',by_video);return result

def report():
    setup();require_plan();started=time.perf_counter();prediction,source_seal=before_GT();complete=json.loads((RUN/'official_completion.json').read_text())
    if complete['status']!='ALL63_TARGET_AND4_SOURCE_OFFICIAL_COMPLETE':raise ValueError('all formal evaluations required')
    costs=teacher_costs();meta,_,_=inputs();plan={r['method']:r for r in prediction['methods']}
    structure=json.loads((cs.PUBLIC/'structure_stats.json').read_text());presence={(r['video'],r['expression']):bool(r['GT_observations']) for r in structure['episodes']}
    train={' '.join(r['query_text'].lower().split()) for r in read_jsonl('/data2/user/TTA-RMOT-assets/train2val/20261031/runtime_train.jsonl')}
    seen={q['video_id']+'+'+q['expression_id']:' '.join(q['query_text'].lower().split()) in train for q in meta['episodes']}
    h=HOTA({'PRINT_CONFIG':False});rows={};values={};videos=None;seals={}
    for m in complete['methods']:
        seal=v2.verify_seal(m);seals[m]=seal;metrics=official_metrics(v2.EVAL/m/'results')
        vs,stat=official_video_values(v2.EVAL/m/'results/pedestrian_detailed.csv')
        if videos is not None and vs!=videos:raise ValueError('video clusters differ')
        videos=vs;values[m]=stat
        with (v2.EVAL/m/'results/pedestrian_detailed.csv').open() as f:data=list(csv.DictReader(f))
        detail={r['seq']:r for r in data if r['seq']!='COMBINED'};combined=next(r for r in data if r['seq']=='COMBINED')
        if any(bool(float(detail[v+'+'+q]['GT_Dets']))!=p for (v,q),p in presence.items()):raise ValueError('historical query-presence grouping differs from current official GT')
        full=float(pooled_hota(stat,np.ones(25)))
        if not np.isclose(full,100*float(combined['HOTA___AUC']),atol=1e-8,rtol=0):raise ValueError('official pooling mismatch')
        if metrics['Dets']!=seal['prediction_rows']:raise ValueError('prediction/TrackEval rows differ')
        def subgroup(keys):return 100*float(h.combine_sequences({k:unpack(detail[k],h) for k in keys})['HOTA'].mean())
        r=dict(method=m,**metrics,HOTA_unrounded=full,prediction_rows=seal['prediction_rows'],
            query_empty_prediction_rows=sum(e['selected_rows'] for e in seal['episodes'] if not presence[e['video'],e['expression']]),
            Present_HOTA=subgroup([v+'+'+q for (v,q),p in presence.items() if p]),
            Seen_HOTA=subgroup([k for k,p in seen.items() if p]),Unseen_HOTA=subgroup([k for k,p in seen.items() if not p]))
        spec=plan.get(m,dict(margin=m,decoder='NATIVE_REFERENCE',uses_future_frames=None,uses_all_test_video_statistics=False,gate_mode=None,GT_open_attempts=0))
        mode=spec['gate_mode'];calls=costs['by_dataset_mode']['Dance_'+mode]['requested_calls'] if mode else 0
        r.update(margin=spec['margin'],decoder=spec['decoder'],uses_future_frames=spec['uses_future_frames'],
            uses_all_test_video_statistics=spec['uses_all_test_video_statistics'],gate_mode=mode,
            startup_delay_fraction=.20 if spec['decoder']=='TG_EARLY_X1' else 0.,teacher_logical_calls=calls,
            GT_open_attempts=spec['GT_open_attempts'],lineage='同集测试时蒸馏参照' if spec['margin']=='SAME_SET_D1' else 'FROZEN_SOURCE' if spec['margin']=='FROZEN' else 'TRAIN40_TO_VAL25_8_SHARED_PHRASES' if spec['margin'].startswith('T2V') else 'UNCHANGED_HISTORICAL_REFERENCE')
        rows[m]=r
    v2.write_csv(PUBLIC/'decoders.csv',[rows[m] for m in complete['methods']])
    # GT labels are accessed only here, AFTER the gate/outputs SHA checks.
    with (PUBLIC/'gates.csv').open() as f:gates=list(csv.DictReader(f))
    with (PUBLIC/'exclusivity.csv').open() as f:exclusivity=list(csv.DictReader(f))
    qualities=[];quality_episodes=[]
    for mode in ('normal','early'):
        gr=[r for r in gates if r['dataset']=='Dance' and r['gate_mode']==mode]
        if len(gr)!=425:raise ValueError('gate quality requires all425')
        for scope,flag in [('ALL',None),('SEEN_EXACT_PHRASE',True),('UNSEEN_EXACT_PHRASE',False)]:
            rr=[r for r in gr if flag is None or seen[r['video']+'+'+r['expression']]==flag]
            y=np.array([presence[r['video'],r['expression']] for r in rr]);p=np.array([r['predicted_present']=='True' for r in rr]);tp=int((p&y).sum());fp=int((p&~y).sum())
            qualities.append(dict(gate_mode=mode,text_scope=scope,episodes=len(rr),GT_present=int(y.sum()),predicted_present=int(p.sum()),TP=tp,FP=fp,FN=int((~p&y).sum()),TN=int((~p&~y).sum()),
                accuracy=float((p==y).mean()),recall=tp/int(y.sum()) if y.sum() else None,FPR=fp/int((~y).sum()) if (~y).sum() else None))
        quality_episodes.extend(dict(video=r['video'],expression=r['expression'],gate_mode=mode,seen=seen[r['video']+'+'+r['expression']],GT_present=presence[r['video'],r['expression']],predicted_present=r['predicted_present']) for r in gr)
    v2.write_csv(PUBLIC/'gate_quality.csv',qualities);v2.write_csv(PUBLIC/'gate_offline_labels.csv',quality_episodes)
    draws=np.random.default_rng(0).multinomial(25,np.ones(25)/25,size=2000);comparisons={}
    def compare(a,b):
        name=a+'_minus_'+b
        if name in comparisons:return comparisons[name]
        samples=paired_clusters(values[a],values[b],draws);per=paired_clusters(values[a],values[b],np.eye(25))
        r=dict(first=a,second=b,delta_HOTA=float(paired_clusters(values[a],values[b],np.ones(25))),CI95=np.percentile(samples,[2.5,97.5]).tolist(),
            positive_videos=int((per>1e-9).sum()),negative_videos=int((per< -1e-9).sum()),tied_videos=int((abs(per)<=1e-9).sum()),
            per_video=[dict(video=v,delta_HOTA=float(x)) for v,x in zip(videos,per)])
        comparisons[name]=r;return r
    for spec in prediction['methods']:
        if spec['decoder'] in ('X0_C3','X1_C3_CAUSAL'):continue
        a=spec['method'];compare(a,spec['margin']+'__X0_C3')
        for b in ('FROZEN','B1_SRC_RATE_GLOBAL','OFFICIAL_DANCE_COMPLETE'):compare(a,b)
    for seed in (17,23,42):
        m=f'T2V_D1_s{seed}';compare(m+'__X0_C3','FROZEN__X0_C3')
        for b in ('FROZEN__TG_ONLY_FROZEN_X0','FROZEN__TG_ONLY_NATIVE'):compare(m+'__TG_X0',b)
    if len(comparisons)!=201:raise ValueError('all201 fixed decoder/reference comparisons required')
    meanrows=[]
    d1_decoders=('X0_C3','X1_C3_CAUSAL',*GENERAL,'SA_DECODE')
    for d in d1_decoders:
        rr=[rows[f'T2V_D1_s{s}__{d}'] for s in (17,23,42)];r=dict(decoder=d,seeds='17,23,42',n=3)
        for k in ('HOTA_unrounded','DetA','AssA','IDF1','IDSW','prediction_rows','query_empty_prediction_rows','Present_HOTA','Seen_HOTA','Unseen_HOTA'):
            r[k+'_mean']=float(np.mean([x[k] for x in rr]));r[k+'_std_ddof1']=float(np.std([x[k] for x in rr],ddof=1))
        meanrows.append(r)
    v2.write_csv(PUBLIC/'D1_seed_summary.csv',meanrows)
    source_rows=[];source_full={}
    for m in complete['source_methods']:
        with (RUN/'source_boundary/evaluation'/m/'results/pedestrian_detailed.csv').open() as f:
            combined=next(r for r in csv.DictReader(f) if r['seq']=='COMBINED')
        source_full[m]=100*float(combined['HOTA___AUC'])
    for m,r in complete['source_methods'].items():
        native=source_full[r['margin']+'__NATIVE']
        source_rows.append(dict(method=m,margin=r['margin'],decoder=r['decoder'],**r['metrics'],HOTA_unrounded=source_full[m],prediction_rows=r['prediction_rows'],
            delta_HOTA_vs_native=source_full[m]-native,teacher_calls=costs['by_dataset_mode']['KITTI_normal']['requested_calls'] if r['decoder']=='SA_DECODE' else 0))
    v2.write_csv(PUBLIC/'source_boundary.csv',source_rows)
    def stable(decoder):return all(comparisons[f'T2V_D1_s{s}__{decoder}_minus_T2V_D1_s{s}__X0_C3']['CI95'][0]>0 for s in (17,23,42))
    unseen_tg=float(np.mean([rows[f'T2V_D1_s{s}__TG_X0']['Unseen_HOTA'] for s in (17,23,42)]))
    unseen_gz=float(np.mean([rows[f'T2V_D1_s{s}__G0_GZ_X0']['Unseen_HOTA'] for s in (17,23,42)]))
    sa_mean=float(np.mean([rows[f'T2V_D1_s{s}__SA_DECODE']['HOTA_unrounded'] for s in (17,23,42)]))
    source_pass=all(r['delta_HOTA_vs_native']>=-1. for r in source_rows if r['decoder']=='SA_DECODE')
    verdict=dict(gate='视频级教师门控带来稳定增益' if stable('TG_X0') else '门控增益不稳定',
        unseen_teacher_gate='门控恢复了没见过的句子' if unseen_tg>14.229 else '未达到预注册的未见句子恢复门槛',
        unseen_global_z='全局句子标准化恢复了没见过的句子' if unseen_gz>14.229 else '未达到预注册的未见句子恢复门槛',
        TG_unseen_HOTA_mean=unseen_tg,G0_unseen_HOTA_mean=unseen_gz,fixed_unseen_threshold=14.229,
        GT_free_rejection='GT-free 拒绝有效' if stable('G1_GMM_REJECT_X0') else '未达到预注册的GT-free拒绝有效条件',
        structural='结构自适应解码在两个数据集上同时成立' if source_pass and sa_mean>=26. else '未达到预注册的双数据集结构自适应条件',
        source_boundary_within1=source_pass,Dance_D1_SA_HOTA_mean=sa_mean,
        student_contribution=[comparisons[f'T2V_D1_s{s}__X0_C3_minus_FROZEN__X0_C3'] for s in (17,23,42)])
    atomic_json(PUBLIC/'bootstrap.json',dict(clusters=25,samples=2000,seed=0,pooling='unchanged official TP-weighted sufficient statistics',
        comparisons=comparisons,verdict=verdict,multiplicity='All preregistered comparisons descriptive, unadjusted; no best-variant selection. Dance Val repeatedly developed, not a blind test.'))
    # Archive all official per-sequence sufficient statistics, not just summaries.
    metric_index={};log_index=[]
    with tarfile.open(PUBLIC/'official_metrics.tar.gz','w:gz') as archive:
        for domain,base,names in [('Dance',v2.EVAL,list(complete['methods'])),('KITTI',RUN/'source_boundary/evaluation',list(complete['source_methods']))]:
            for m in names:
                for filename in ('pedestrian_summary.txt','pedestrian_detailed.csv'):
                    p=base/m/'results'/filename;key=domain+'/'+m+'/'+filename;archive.add(p,arcname=key);metric_index[key]=dict(path=str(p),sha256=file_sha(p))
    with tarfile.open(PUBLIC/'stage_logs.tar.gz','w:gz') as archive:
        for p in sorted(RUN.rglob('*')):
            if p.is_file() and not p.is_symlink() and (p.name=='trackeval.log' or (p.suffix in ('.log','.txt') and any(k in p.parts for k in ('official_jobs','teacher_launchers'))) or (p.suffix=='.json' and 'teacher_launchers' in p.parts) or ('CPU_launchers' in p.parts and p.name in ('generate.txt','official.txt','generate.json','official.json'))):
                archive.add(p,arcname=str(p.relative_to(RUN)));log_index.append(dict(path=str(p),sha256=file_sha(p)))
    atomic_json(PUBLIC/'official_metrics_index.json',dict(files=metric_index,archive_sha256=file_sha(PUBLIC/'official_metrics.tar.gz')))
    with gzip.open(PUBLIC/'prediction_files.csv.gz','wt') as f:
        w=csv.DictWriter(f,fieldnames=['domain','method','video','expression','prediction_path','prediction_sha256','selected_rows']);w.writeheader()
        for domain,mm in [('Dance',[dict(method=m,**{k:v for k,v in s.items() if k!='method'}) for m,s in seals.items()]),('KITTI',source_seal['methods'])]:
            for s in mm:
                for e in s['episodes']:w.writerow(dict(domain=domain,method=s['method'],**{k:e[k] for k in w.fieldnames if k in e}))
    artifacts=[]
    for name in ('teacher/manifest.jsonl','teacher/outputs.jsonl','teacher/progress.sqlite','teacher/receipt.json','teacher/smoke_receipt.json','predictions_seal.json','gates_seal.json','source_boundary/predictions_seal.json'):
        p=RUN/name;artifacts.append(dict(path=str(p),sha256=file_sha(p),bytes=p.stat().st_size))
    for r in prediction['methods']:
        if r['mask_path']:artifacts.append(dict(path=r['mask_path'],sha256=r['mask_sha256'],bytes=Path(r['mask_path']).stat().st_size))
    atomic_json(PUBLIC/'server_artifacts.json',dict(run_root=str(RUN),large_files_not_in_git=True,files=artifacts,logs=log_index))
    sections=['# 跨句竞争解码：全局去偏、视频教师门控与结构自适应',
        '状态：COMPLETE_STOP。TTA-RMOT，起点ecf57122；分支research/v2dance-gated-decode-20261101。不训练学生或教师；全部规则在新评测前固定并推送。',
        '## 五项预注册结果',
        '1. 门控：'+verdict['gate']+'。下方列出三个seed的TG_X0−X0全部区间。',
        f'2. 没见过的句子：{verdict["unseen_teacher_gate"]}（TG_X0均值{unseen_tg:.6f}）；{verdict["unseen_global_z"]}（G0_GZ_X0均值{unseen_gz:.6f}）；固定门槛14.229。',
        '3. 不用教师的拒绝：'+verdict['GT_free_rejection']+'。',
        f'4. 结构自适应：{verdict["structural"]}。源域两模型均在原生−1以内={source_pass}；Dance三个D1的SA均值={sa_mean:.6f}，固定门槛26。',
        '5. 学生贡献：只报告同一X0解码下三个D1−Frozen的差值和CI，不做新判定。',
        table(['student','reference','delta HOTA','CI low','CI high'],[[r['first'],r['second'],r['delta_HOTA'],*r['CI95']] for r in verdict['student_contribution']]),
        '## 全部目标域组合：25视频、425表达、211085条固定MOT观测',
        '60个解码组合及3个原生参照全部报告。TG_ONLY_FROZEN_X0是Frozen TG_X0的精确别名；教师门控只生成一次，在各分数源之间复用，不能将调用次数乘以学生数。',
        table(['margin','decoder','HOTA','DetA','AssA','IDF1','IDSW','rows','Empty rows','Present HOTA','Seen HOTA','Unseen HOTA','future','all-video stats','T1 logical calls','GT opens'],
            [[r[k] for k in ('margin','decoder','HOTA','DetA','AssA','IDF1','IDSW','prediction_rows','query_empty_prediction_rows','Present_HOTA','Seen_HOTA','Unseen_HOTA','uses_future_frames','uses_all_test_video_statistics','teacher_logical_calls','GT_open_attempts')] for r in rows.values()]),
        'X1原始解码只用过去；G0_GZ_X1虽只累计过去轨迹分数，但统计量使用全部25测试视频，属于离线转导。TG_EARLY_X1使用前20%完整门控并回用于早期帧，有20%启动延迟，不是严格因果。其余全轨迹/视频门控使用未来信息。SAME_SET_D1是同集测试时蒸馏参照，不能描述为Train→Val或完整系统因果。',
        '### D1三个seed：均值与样本标准差(ddof=1)',
        table(['decoder','HOTA mean','HOTA std','Seen mean','Unseen mean','Empty rows mean'],[[r[k] for k in ('decoder','HOTA_unrounded_mean','HOTA_unrounded_std_ddof1','Seen_HOTA_mean','Unseen_HOTA_mean','query_empty_prediction_rows_mean')] for r in meanrows]),
        '## 全部配对视频聚类bootstrap',
        '25视频、2000次、seed0；复用官方TP加权充分统计，不把帧或表达当独立样本，不算视频HOTA简单平均。所有区间是未校正的描述性区间；Val已反复开发，不能称盲测。',
        table(['first','second','delta HOTA','CI low','CI high','videos +','videos −'],[[r['first'],r['second'],r['delta_HOTA'],*r['CI95'],r['positive_videos'],r['negative_videos']] for r in comparisons.values()]),
        '## 视频级门控质量（封存后的独立GT离线分组）',
        '存在性GT：完整425个(video,query)中40个Present、385个Empty。seen/unseen沿用规范化完整句子精确相等（8/17见过），不是所有属性词是否见过；标签不进入生成进程。',
        table(['gate mode','scope','episodes','GT present','pred present','TP','FP','FN','TN','accuracy','recall','FPR'],[[r[k] for k in ('gate_mode','text_scope','episodes','GT_present','predicted_present','TP','FP','FN','TN','accuracy','recall','FPR')] for r in qualities]),
        '## E(v)与结构分支',
        'E按候选计算多个表达p_yes≥0.5的比例，不是按单一argmax票数计算。分母为零按本轮用户明确答复输出空集，E保持未定义，单列而不伪造E=0。',
        table(['dataset','video','positive candidates','multiple candidates','E(v)','undefined','selected decoder'],[[r[k] for k in ('dataset','video','positive_candidates','multiple_candidates','E','zero_denominator','selected_decoder')] for r in exclusivity]),
        '### KITTI-V2边界：4验证视频、862表达',
        '完全复用上一轮2355528行源域分数与原MOT候选覆盖；图像帧=MOT帧−1，预测保持原MOT帧。源域GT只在本节官方评价用，不计算任何阈值。',
        table(['method','HOTA','DetA','AssA','IDF1','rows','HOTA−native','teacher calls'],[[r[k] for k in ('method','HOTA','DetA','AssA','IDF1','prediction_rows','delta_HOTA_vs_native','teacher_calls')] for r in source_rows]),
        '## 教师成本与全量调用外推',
        f'实际唯一forward判断{costs["actual_unique_forwarded_calls"]}；本轮清单请求{costs["requested_gate_judgements"]}（重叠门控帧缓存复用）。真实forward路径总秒数={costs["forward_seconds"]:.6f}，GPU进程小时（含smoke和加载/哈希/I/O）={costs["GPU_process_hours"]:.6f}，峰值分配显存={costs["peak_allocated_bytes"]}字节。',
        f'Dance实际批次摊销每判断{costs["Dance_amortized_forward_seconds"]:.8f}秒；穷举3588445候选–句子对预计forward路径{costs["exhaustive_Dance_forward_GPU_hours"]:.6f} GPU小时。这是实测吞吐外推，不是已运行的全量教师RMOT结果，不含全量图像I/O/多次加载。源域与目标域提示长度不同，混合域外推另列teacher_costs.json。',
        table(['dataset','video','mode','frames','requested','new','cache hit','forward seconds','peak bytes'],[[r[k] for k in ('dataset','video','gate_mode','frames','requested_calls','new_calls','cache_hits','forward_seconds','peak_allocated_bytes')] for r in costs['by_video']]),
        'Teacher门控属于新的test-video teacher-assisted设定，并非不使用外部教师的source-free online TTA。本轮增量成本不含历史学生训练、固定MOT前端和已封存margin的推理；这些不是独立RMOT系统的零成本。文件预测seal中的GPU_calls=0仅指CPU解码阶段，教师调用单独完整计入。',
        '## GT隔离、复现和来源',
        '六套X0/X1共5100预测文件逐字节复现。输入2550个原始margin文件与六数组核SHA；源域两数组和MOT哈希核验。所有教师原始输出、gates.csv、exclusivity.csv及目标60套/源域4套预测在新GT评测前封存。Python打开审计均0；教师另加内核Landlock运行时白名单。Qwen revision、冻结权重前后相等、原始logits/soft概率、输入/框/提示哈希均保留。backend和320/margin.2裁剪代码未改。',
        f'官方评价：目标新作业{complete["new_target_jobs"]}，精确逐行等价复用{complete["reused_target_jobs"]}；源域新作业{sum(r.get("new_TrackEval_job",False) for r in complete["source_methods"].values())}。等价复用保留全部指标，不省略完整数据。',
        '## 失败、修复与恢复',
        '初次CPU测试7过2失败，原因是共享teacher环境未安装sklearn；将scikit-learn1.7.2、joblib1.5.2、threadpoolctl3.6.0装入本轮RUN/deps，不改共享环境，原9项随后全过。其他真实执行记录见EXECUTION.md与stage_logs.tar.gz。无参数修改、无新seed选择、无GT阈值、无删除失败样本。',
        '运行根目录：'+str(RUN)+'。大文件路径/SHA见server_artifacts.json；全部预测路径/SHA见prediction_files.csv.gz；完整官方CSV和日志见official_metrics.tar.gz与stage_logs.tar.gz。教师progress.sqlite按完整输入哈希断点续跑。',
        '本轮结果解释与限制见INTERPRETATION.md。完成后停止，不训练、不新增方法、不自行扩大教师标注。']
    v2.write_new(PUBLIC/'REPORT.md','\n\n'.join(sections)+'\n')
    atomic_json(PUBLIC/'completion.json',dict(status='COMPLETE_STOP',verdict=verdict,target_decoder_sets=60,target_reference_sets=3,source_sets=4,
        comparisons=len(comparisons),GT_open_attempts_in_generation=0,student_updates=0,teacher_updates=0,
        report_sha256=file_sha(PUBLIC/'REPORT.md'),CPU_report_seconds=time.perf_counter()-started))
    print(json.dumps(dict(event='GATED_REPORT_COMPLETE',verdict=verdict,comparisons=len(comparisons))),flush=True)

if __name__=='__main__':report()
