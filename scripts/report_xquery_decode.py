"""All preregistered decoders and paired-video uncertainty; report then STOP."""
import csv
import gzip
import json
from pathlib import Path
import sys
import tarfile
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.xquery_common import *
from scripts.pf_diag_common import official_metrics,official_video_values
from scripts.analyze_pf_video_uncertainty import HOTA,unpack,paired_clusters,pooled_hota
from scripts.report_count_structure import table
from flex_uda.data.manifests import read_jsonl

def report():
    setup();require_plan();started=time.perf_counter()
    complete=json.loads((RUN/'official_completion.json').read_text());source=json.loads((PUBLIC/'source_boundary.json').read_text())
    if complete['status']!='ALL39_OFFICIAL_TARGET_SETS_COMPLETE' or source['status']!='ALL8_SOURCE_BOUNDARY_OFFICIAL_COMPLETE':raise ValueError('finish all registered evaluations')
    probe=json.loads((PUBLIC/'teacher_combo.json').read_text());thresholds=json.loads((PUBLIC/'thresholds.json').read_text())
    ps=json.loads((RUN/'predictions_seal.json').read_text());plan={r['method']:r for r in ps['methods']}
    structure=json.loads((cs.PUBLIC/'structure_stats.json').read_text());presence={(r['video'],r['expression']):bool(r['GT_observations']) for r in structure['episodes']}
    meta,_,_=inputs(verify=False);train={' '.join(r['query_text'].lower().split()) for r in read_jsonl('/data2/user/TTA-RMOT-assets/train2val/20261031/runtime_train.jsonl')}
    seen={q['video_id']+'+'+q['expression_id']:' '.join(q['query_text'].lower().split()) in train for q in meta['episodes']}
    h=HOTA({'PRINT_CONFIG':False});rows={};values={};videos=None;seals={};perquery=[]
    for m in complete['methods']:
        seals[m]=v2.verify_seal(m);metrics=official_metrics(v2.EVAL/m/'results')
        vs,stat=official_video_values(v2.EVAL/m/'results/pedestrian_detailed.csv')
        if videos is not None and videos!=vs:raise ValueError('different video clusters')
        videos=vs;values[m]=stat
        with (v2.EVAL/m/'results/pedestrian_detailed.csv').open() as f:data=list(csv.DictReader(f))
        combined=next(r for r in data if r['seq']=='COMBINED');detail={r['seq']:r for r in data if r['seq']!='COMBINED'}
        full=float(pooled_hota(stat,np.ones(25)))
        if not np.isclose(full,100*float(combined['HOTA___AUC']),atol=1e-8,rtol=0):raise ValueError('official pooling mismatch')
        if metrics['Dets']!=seals[m]['prediction_rows']:raise ValueError('official versus sealed prediction row mismatch')
        def subgroup(keys):
            stats=h.combine_sequences({k:unpack(detail[k],h) for k in keys})
            return 100*float(np.mean(stats['HOTA']))
        row=dict(method=m,**metrics,HOTA_unrounded=full,prediction_rows=seals[m]['prediction_rows'],
            query_empty_prediction_rows=sum(e['selected_rows'] for e in seals[m]['episodes'] if not presence[e['video'],e['expression']]),
            Present_HOTA=subgroup([v+'+'+q for (v,q),p in presence.items() if p]),
            Seen_HOTA=subgroup([k for k,p in seen.items() if p]),Unseen_HOTA=subgroup([k for k,p in seen.items() if not p]))
        if m in plan:
            r=plan[m];row.update(margin=r['margin'],decoder=r['decoder'],uses_future_frames_selector=r['uses_future_frames'],target_GT_open_attempts=r['target_GT_open_attempts'],
                lineage='同集测试时蒸馏参照' if r['margin']=='SAME_SET_D1' else 'FROZEN_SOURCE' if r['margin']=='FROZEN' else 'TRAIN40_TO_VAL25_WITH8_SHARED_PHRASES')
        else:row.update(margin=m,decoder='NATIVE_REFERENCE',uses_future_frames_selector=None,target_GT_open_attempts=0,lineage='EXACT_REUSED_HISTORICAL_REFERENCE')
        rows[m]=row
        perquery.extend(dict(method=m,video=e['video'],expression=e['expression'],prediction_rows=e['selected_rows']) for e in seals[m]['episodes'])
    decoders=[rows[m+'__'+d] for m in MARGINS for d in DECODERS];v2.write_csv(PUBLIC/'decoders.csv',decoders);v2.write_csv(PUBLIC/'decoder_per_query.csv',perquery)
    draws=np.random.default_rng(0).multinomial(25,np.ones(25)/25,size=2000);comparisons={}
    def compare(a,b):
        sample=paired_clusters(values[a],values[b],draws);per=paired_clusters(values[a],values[b],np.eye(25))
        r=dict(first=a,second=b,delta_HOTA=float(paired_clusters(values[a],values[b],np.ones(25))),CI95=np.percentile(sample,[2.5,97.5]).tolist(),
            positive_videos=int((per>1e-9).sum()),negative_videos=int((per< -1e-9).sum()),tied_videos=int((np.abs(per)<=1e-9).sum()),
            per_video=[dict(video=v,delta_HOTA=float(d)) for v,d in zip(videos,per)])
        comparisons[a+'_minus_'+b]=r;return r
    for m in MARGINS:
        for d in DECODERS:
            for b in ('FROZEN','B1_SRC_RATE_GLOBAL','OFFICIAL_DANCE_COMPLETE'):compare(m+'__'+d,b)
        for a,b in [(1,0),(3,2),(5,4),(4,2)]:compare(m+'__'+DECODERS[a],m+'__'+DECODERS[b])
    if len(comparisons)!=132:raise ValueError('all132 preregistered comparisons required')
    confirm=all(comparisons[m+'__X0_C3_minus_B1_SRC_RATE_GLOBAL']['CI95'][0]>0 for m in ('T2V_D1_s23','T2V_D1_s42'))
    unseen=float(np.mean([rows[f'T2V_D1_s{s}__X2_C3_Z']['Unseen_HOTA'] for s in (17,23,42)]));debiased=unseen>14.229
    verdict=dict(confirmation='跨句竞争解码在 Train→Val 下可复现' if confirm else '未能复现',
        debias='句子标准化恢复了没见过的句子' if debiased else '未恢复',X2_D1_unseen_HOTA_mean=unseen,fixed_unseen_threshold=14.229,
        causal='只报告各模型X1−X0、X3−X2、X5−X4及CI；不做判定',rejection='只报告X4−X2及CI、Query-Empty行数变化；不做判定',
        default_offline='X4_C3_Z_REJECT' if debiased else 'X0_C3',default_causal='X5_C3_Z_REJECT_CAUSAL' if debiased else 'X1_C3_CAUSAL',
        default_is_rule_result_not_authorization_to_run_next_experiment=True)
    atomic_json(PUBLIC/'bootstrap.json',dict(clusters=25,samples=2000,seed=0,comparisons=comparisons,verdict=verdict,
        pooling='unchanged v2 TP-weighted official sufficient statistics; never arithmetic mean of episode HOTA',
        limitations='132 descriptive comparisons all reported, no selection or multiplicity correction; Val repeatedly developed, confirmation seeds are not a new blind dataset'))
    meanrows=[]
    fields=['HOTA_unrounded','DetA','AssA','IDF1','IDSW','prediction_rows','query_empty_prediction_rows','Present_HOTA','Seen_HOTA','Unseen_HOTA']
    for d in DECODERS:
        rr=[rows[f'T2V_D1_s{s}__{d}'] for s in (17,23,42)];r=dict(decoder=d,seeds='17,23,42',n=3)
        for k in fields:
            x=[r[k] for r in rr];r[k+'_mean']=float(np.mean(x));r[k+'_std_ddof1']=float(np.std(x,ddof=1))
        meanrows.append(r)
    v2.write_csv(PUBLIC/'D1_seed_summary.csv',meanrows)
    source_gpu=json.loads((PUBLIC/'source_GPU_cost.json').read_text());source_scores=json.loads((RUN/'source/scores_seal.json').read_text())
    logs=[];metric_index={}
    with tarfile.open(PUBLIC/'official_metrics.tar.gz','w:gz') as archive:
        for domain,methods,base in [('Dance',list(complete['methods']),v2.EVAL),('KITTI',[r['method'] for r in source['methods']],RUN/'source_boundary/evaluation')]:
            for m in methods:
                key=domain+'/'+m;metric_index[key]={}
                for filename in ('pedestrian_summary.txt','pedestrian_detailed.csv'):
                    p=base/m/'results'/filename;archive.add(p,arcname=key+'/'+filename);metric_index[key][filename]=dict(path=str(p),sha256=file_sha(p))
    atomic_json(PUBLIC/'official_metrics_index.json',dict(files=metric_index,archive_sha256=file_sha(PUBLIC/'official_metrics.tar.gz')))
    with tarfile.open(PUBLIC/'stage_logs.tar.gz','w:gz') as archive:
        for p in sorted(RUN.rglob('*')):
            if p.is_file() and (p.name=='trackeval.log' or (any(x in ('launchers','CPU_launchers') for x in p.parts) and p.suffix in ('.txt','.json')) or ('official_jobs' in p.parts and p.suffix in ('.txt','.json'))):
                archive.add(p,arcname=str(p.relative_to(RUN)));logs.append(dict(path=str(p),sha256=file_sha(p)))
    with gzip.open(PUBLIC/'prediction_files.csv.gz','wt') as f:
        writer=csv.DictWriter(f,fieldnames=['domain','method','video','expression','prediction_path','prediction_sha256','selected_rows']);writer.writeheader()
        for m,s in seals.items():
            for e in s['episodes']:writer.writerow(dict(domain='Dance',method=m,**{k:e[k] for k in writer.fieldnames if k in e}))
        boundary=json.loads((RUN/'source_boundary/predictions_seal.json').read_text())
        for s in boundary['methods']:
            for e in s['episodes']:writer.writerow(dict(domain='KITTI',method=s['method'],**e))
    artifacts=[]
    for p in sorted((RUN/'source').glob('*.npy')):artifacts.append(dict(path=str(p),sha256=file_sha(p),bytes=p.stat().st_size))
    for name in ('source/metadata.json','source/GT_seal.json','source/scores_seal.json','teacher_combo.jsonl','teacher_combo_offline_frames.json'):
        p=RUN/name;artifacts.append(dict(path=str(p),sha256=file_sha(p),bytes=p.stat().st_size))
    for r in ps['methods']:artifacts.append(dict(path=r['mask_path'],sha256=r['mask_sha256'],bytes=Path(r['mask_path']).stat().st_size))
    atomic_json(PUBLIC/'server_artifacts.json',dict(run_root=str(RUN),files=artifacts,logs=logs,large_files_not_in_git=True))
    costs=dict(source_GPU=source_gpu,target_new_GPU_calls=0,new_Dance_teacher_calls=0,student_updates=0,teacher_updates=0,
        target_decoding_CPU_seconds=ps['CPU_wall_seconds'],source_threshold_CPU_seconds=thresholds['CPU_seconds'],
        official_target_actual_wall_seconds=complete['actual_batch_wall_seconds'],official_source_actual_wall_seconds=source['actual_batch_wall_seconds'],
        new_target_TrackEval_jobs=complete['new_jobs'],exact_reused_target_results=complete['reused_jobs'],new_source_TrackEval_jobs=8,
        source_smoke_and_full_cost_both_included=True,cost_semantics='GPU-process hours include model loading/CPU segments, not utilization integral; prior checkpoint training excluded and explicitly inherited',
        feature_cache='512MiB bounded RAM per current video, no persistent new visual feature cache')
    atomic_json(PUBLIC/'costs.json',costs)
    sections=['# 跨句竞争解码：确认、去偏、因果化与适用边界',
        '状态：COMPLETE_STOP；项目TTA-RMOT，起点000e196d，分支research/v2dance-xquery-decode-20261101。全部规则固定后执行，无学生/教师训练、无Dance教师调用。',
        '## 五项预注册结果',
        '1. 确认：'+verdict['confirmation']+'。s23/s42的X0−B1配对区间分别为'+str(comparisons['T2V_D1_s23__X0_C3_minus_B1_SRC_RATE_GLOBAL']['CI95'])+' / '+str(comparisons['T2V_D1_s42__X0_C3_minus_B1_SRC_RATE_GLOBAL']['CI95'])+'。',
        f'2. 去偏：{verdict["debias"]}。三个D1 seed的X2未见句子HOTA均值={unseen:.6f}，固定门槛14.229；不据目标GT改变标准化或拒绝阈值。',
        '3. 因果：'+verdict['causal']+'。见下方全部18项差值。',
        '4. 拒绝：'+verdict['rejection']+'。见下方全部6项差值；不强制正例。',
        '5. 后续默认：离线'+verdict['default_offline']+'，因果'+verdict['default_causal']+'；仅记录用户固定规则的结果，本轮到此停止，不启动下一实验。',
        '## 全部六模型×六解码器：25视频、425表达、211085个MOT观测',
        'GT仅用于封存后的独立评价；Causal列只描述解码器，SAME_SET_D1继承同Val训练，不能把整个系统称为因果/source-free。源域阈值只由KITTI GT确定；X2/X4使用全视频矩，属于离线。',
        table(['margin','decoder','HOTA','DetA','AssA','IDF1','IDSW','rows','Empty rows','Present HOTA','Seen HOTA','Unseen HOTA','future','GT打开'],
              [[r[k] for k in ('margin','decoder','HOTA','DetA','AssA','IDF1','IDSW','prediction_rows','query_empty_prediction_rows','Present_HOTA','Seen_HOTA','Unseen_HOTA','uses_future_frames_selector','target_GT_open_attempts')] for r in decoders]),
        '### 原生固定参照（精确复用，不重新推理）',table(['method','HOTA','DetA','AssA','IDF1','rows'],[[rows[m][k] for k in ('method','HOTA','DetA','AssA','IDF1','prediction_rows')] for m in ('FROZEN','B1_SRC_RATE_GLOBAL','OFFICIAL_DANCE_COMPLETE')]),
        '### D1三个seed均值与样本标准差(ddof=1)',table(['decoder','HOTA mean','HOTA std','Seen mean','Seen std','Unseen mean','Unseen std'],
            [[r[k] for k in ('decoder','HOTA_unrounded_mean','HOTA_unrounded_std_ddof1','Seen_HOTA_mean','Seen_HOTA_std_ddof1','Unseen_HOTA_mean','Unseen_HOTA_std_ddof1')] for r in meanrows]),
        '## 视频级配对不确定性：全部132比较',
        '2000次，seed0，25个视频作为聚类；使用官方TP加权充分统计，不把帧或表达当独立样本。全部区间是描述性；先前已开发Val，s23/s42仅作为未评测该解码器的seed确认，不能称全新盲测。',
        table(['first','second','delta HOTA','CI low','CI high','videos +','videos −','ties'],[[r['first'],r['second'],r['delta_HOTA'],*r['CI95'],r['positive_videos'],r['negative_videos'],r['tied_videos']] for r in comparisons.values()]),
        '### 因果差值与拒绝的Empty行数变化',table(['margin','comparison','delta','CI low','CI high','Empty delta'],
            [[m,DECODERS[a]+'−'+DECODERS[b],(r:=comparisons[m+'__'+DECODERS[a]+'_minus_'+m+'__'+DECODERS[b]])['delta_HOTA'],*r['CI95'],
              rows[m+'__'+DECODERS[a]]['query_empty_prediction_rows']-rows[m+'__'+DECODERS[b]]['query_empty_prediction_rows']] for m in MARGINS for a,b in [(1,0),(3,2),(5,4),(4,2)]]),
        '## 源域拒绝阈值：模型独立、只读KITTI GT',
        '每模型在完全相同2355528行、862表达、4验证视频上计算；候选轨迹匹配>=50%为正，穷举>=阈值的全部不同均值与空集边界，F1并列取最小阈值。不是数值优化、没有更新权重；没有把Frozen阈值套给其他模型。',
        table(['margin','theta','source track F1','TP','FP','FN','pairs','candidate thresholds'],[[m,*[r[k] for k in ('threshold','F1','TP','FP','FN','query_track_pairs','candidate_thresholds')]] for m,r in thresholds['models'].items()]),
        '## KITTI-V2适用边界（不参与Dance判定）',
        '仅Frozen与T2V_D1_s17，原生/X0/X2/X4全部评测。历史官方源域dataloader覆盖2355528分数，保持该覆盖，不冒充补末尾观测后的源域全库。图像帧=原MOT帧−1，评价输出恢复原MOT帧；ped ID偏移、候选框和ID不改。Frozen原生862序列逐行数值复现原官方输出；源域阈值和本节评测共用验证集，因此源域拒绝收益不是独立泛化证明。',
        table(['method','HOTA','DetA','AssA','IDF1','rows'],[[r['method'],*[r['metrics'][k] for k in ('HOTA','DetA','AssA','IDF1')],r['prediction_rows']] for r in source['methods']]),
        '源域每条物理预测轨迹的正例句子数分布：'+json.dumps(source['source_track_query_histogram'],ensure_ascii=False)+'；0/1/>=2比例：'+json.dumps(source['source_track_fractions'],ensure_ascii=False)+f'，共{source["physical_tracks"]}轨迹。多句可以同时描述同一源域目标，跨句唯一分配的结构适用性必须受此限制。',
        '## 教师组合：原封存5视频50帧，不调用教师',
        '三种组合全部报告，不选定教师。T1_ARGMAX/T1_ARGMAX_REJECT是旧T1软分数的候选间硬解码，T2_ARGMAX为18选项硬解码含none；不能将硬分数AUROC解释为新概率校准。固定5视频50帧，6426候选–句子对、850帧–句子，非完整RMOT HOTA。',
        table(['method','text scope','candidate accuracy','positive recall','negative FPR','presence accuracy','quantity MAE','exact count','selected'],
            [[r[k] for k in ('method','text_scope','candidate_accuracy','positive_recall','negative_FPR','presence_accuracy','quantity_MAE','quantity_exact_accuracy','selected_rows')] for r in probe['rows']]),
        '## GT隔离、复现与成本',
        '四套历史X0共1700个原始预测文件逐SHA一致；六套输入2550原margin文件核验。36套目标预测与3种教师组合均在GT评测前完整封存；各生成进程目标GT打开尝试0。源域匹配重现354658正观测；GT只在源域阈值/源域评测及目标独立评测进程读取。',
        '因果z：纳入当前帧全部候选后统计ddof=0总体矩，观测数<50用raw margin，常量sigma取0；同一历史观测的z不会用未来矩重算。所有拒绝均为raw均值<theta才拒绝，等于保留；空输出不修救。',
        '成本：'+json.dumps(costs,ensure_ascii=False),
        '## 失败与修复记录、恢复路径',
        '本轮真实失败及修复见EXECUTION.md和stage_logs.tar.gz；不跳样本、不改种子、不改固定判定。23项CPU回归已通过(6新解码/teacher tests、8原count_structure、9原v2)；测试通过不代替科研结论。',
        '服务器运行根目录：'+str(RUN)+'。源域分数/模型/掩码等大文件仅记录路径与SHA于server_artifacts.json、source_inputs.json、source_scores_seal.json；所有预测SHA在prediction_files.csv.gz；完整官方CSV在official_metrics.tar.gz、完整TrackEval和源域GPU stdout在stage_logs.tar.gz。',
        '完成后停止：不训练，不调用Dance教师，不自行启动新方法或默认解码器的新实验。']
    v2.write_new(PUBLIC/'REPORT.md','\n\n'.join(sections)+'\n')
    result=dict(status='COMPLETE_STOP_NO_TRAINING_NO_NEW_DANCE_TEACHER',verdict=verdict,decoder_sets=36,target_reference_sets=3,source_boundary_sets=8,
        target_GT_open_attempts_in_generators=0,student_updates=0,new_teacher_calls=0,report_sha256=file_sha(PUBLIC/'REPORT.md'),CPU_report_seconds=time.perf_counter()-started)
    atomic_json(PUBLIC/'completion.json',result);print(json.dumps(dict(event='XQUERY_REPORT_COMPLETE',**result)),flush=True)

if __name__=='__main__':report()
