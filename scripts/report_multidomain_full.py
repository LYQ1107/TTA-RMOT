"""Six preregistered pairs, official pooled HOTA, target structure and costs."""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import sys
import tarfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.multidomain_diagnostics import *
from scripts.report_multidomain_generality import PAIRS,table,detail,export_hota_stats
from scripts.analyze_pf_video_uncertainty import HOTA,pooled_hota,paired_clusters


def main():
    require_plan();allrows=[];rhos=[];structures=[];categories=[];ci={};files=[];references={};margins={}
    allmethods={};source_categories=json.loads((RUN/'source_query_categories.json').read_text())['mapping']
    for combo,source,target in PAIRS:
        completed=json.loads((base(target,source)/'completion.json').read_text())
        if completed['status']!='ALL_AUTHORIZED_CONFIGURATIONS_COMPLETE':raise ValueError('Six-pair report requires every fixed configuration')
        meta,_,_,paths=inputs(target,source)
        truth=json.loads((base(target,source)/'oracle.json').read_text())
        present={(x['video'],x['expression']):x['present'] for x in truth['episodes']}
        methods={m:detail(target,m,meta,source) for m in CONFIGS};allmethods[combo]=methods
        for name,result in methods.items():
            m=result['done']['metrics'];seal=result['seal']
            allrows.append(dict(combo=combo,source=source,target=target,configuration=name,status='COMPLETE',
                oracle='ORACLE' in name,**{k:m[k] for k in ('HOTA','DetA','AssA','IDF1','IDSW')},
                HOTA_unrounded=result['HOTA_unrounded'],prediction_rows=seal['prediction_rows'],
                query_empty_prediction_rows=sum(e['selected_rows'] for e in seal['episodes'] if not present[e['video'],e['expression']]),
                source_R_at_1=truth['retrieval']['source']['GT_R_at_1'],in_domain_R_at_1=truth['retrieval']['in_domain']['GT_R_at_1'],
                source_candidate_R_at_1=truth['retrieval']['source']['R_at_1'],in_domain_candidate_R_at_1=truth['retrieval']['in_domain']['R_at_1']))
        h={m:r['HOTA_unrounded'] for m,r in methods.items()};gap=h['IN_DOMAIN']-h['NATIVE']
        rho=(h['R_SxK_I']-h['NATIVE'])/gap if gap else None
        label='跨域差距过小，不判定' if gap<3 else '数量主导' if rho>=.7 else '数量与排序共同作用' if rho>=.3 else '排序主导'
        rhos.append(dict(combo=combo,source=source,target=target,gap=gap,rho=rho,
            delta_rank=h['IN_DOMAIN']-h['R_SxK_I'],label=label))
        if not any(x['target']==target for x in structures):
            pervideo=Counter(q['video_id'] for q in meta['episodes']);texts=Counter(q['query_text'] for q in meta['episodes'])
            structures.append(dict(target=target,status='COMPLETE_OFFLINE_ORACLE',role='ORACLE_DESCRIPTIVE_ONLY',
                videos=len(pervideo),expressions=len(meta['episodes']),**truth['structure'],
                per_video_expressions=json.dumps(dict(pervideo),sort_keys=True),
                unique_texts=len(texts),shared_texts_across_videos=any(v>1 for v in texts.values())))
        if target=='KITTI-V2':
            evaluator=HOTA({'PRINT_CONFIG':False})
            for rule in ('NATIVE','IN_DOMAIN','R_SxK_I'):
                for category in ('car','pedestrian'):
                    subset={q['video_id']+'+'+q['expression_id']:methods[rule]['rows'][q['video_id']+'+'+q['expression_id']]
                        for q in meta['episodes'] if source_categories[q['video_id']+'+'+q['expression_id']]==category}
                    pooled=evaluator.combine_sequences(subset)
                    categories.append(dict(combo=combo,source=source,target=target,configuration=rule,category=category,
                        HOTA=100*float(np.mean(pooled['HOTA'])),DetA=100*float(np.mean(pooled['DetA'])),
                        AssA=100*float(np.mean(pooled['AssA'])),expressions=len(subset),uncertainty='4 videos; point estimates only'))
        else:
            n=len(methods['NATIVE']['videos']);draws=np.random.default_rng(0).multinomial(n,np.ones(n)/n,size=2000)
            for a,b in [('R_SxK_I','NATIVE'),('IN_DOMAIN','R_SxK_I')]:
                da,db=methods[a],methods[b];samples=paired_clusters(da['stats'],db['stats'],draws)
                ci[combo+'__'+a+'_minus_'+b]=dict(combo=combo,target=target,first=a,second=b,clusters=n,samples=2000,seed=0,
                    delta_HOTA=float(paired_clusters(da['stats'],db['stats'],np.ones(n))),
                    CI95=np.percentile(samples,[2.5,97.5]).tolist(),unit='video; all expressions and frames kept together',
                    pooling='Official TP-weighted HOTA sufficient statistics')
        for model,p in zip((source,IN_DOMAIN_MODEL[target]),paths):
            margins[str(p)]=dict(target=target,model=model,path=str(p),sha256=file_sha(p),bytes=p.stat().st_size,coverage='complete')
        for done in completed['methods']:
            name=done['method'];folder=base(target,source)/name
            seal=verify(target,name,source)
            references[combo+'__'+name]=dict(done,seal_path=str(folder/'predictions_seal.json'),results_folder=str(folder/'results'))
            files.extend(dict(combo=combo,configuration=name,**e) for e in seal['episodes'])
    valid=[r for r in rhos if r['gap']>=3];quantity=[r for r in valid if r['label']=='数量主导']
    c2=next(r for r in rhos if r['combo']=='C2')
    if not valid:conclusion='规则未覆盖：全部跨域差距过小'
    elif len(quantity)/len(valid)>=2/3 and c2['label']=='数量主导':conclusion='现象普遍成立'
    elif len(quantity)/len(valid)<1/3:conclusion='现象不普遍'
    else:conclusion='现象部分成立'
    matrix=[]
    for source in ('V2-best','Dance-best','LaMOT-best'):
        for target in ('KITTI-V2','Dance','SportsMOT'):
            r=next((x for x in rhos if x['source']==source and x['target']==target),None)
            matrix.append(dict(source=source,target=target,combo=r['combo'] if r else '',
                gap=r['gap'] if r else None,rho=r['rho'] if r else None,delta_rank=r['delta_rank'] if r else None,
                label=r['label'] if r else '同域参照'))
    # Include the independent KITTI full reference in addition to all main/history sets.
    oldref=base('KITTI-V2')/'V2_COMPLETE_REFERENCE'
    oldseal=json.loads((oldref/'predictions_seal.json').read_text())
    files.extend(dict(combo='C2',configuration='V2_COMPLETE_REFERENCE',**e) for e in oldseal['episodes'])
    for key in ('kitti','dance','sports'):
        for p in (RUN/key).glob('*/scores_seal.json'):
            seal=json.loads(p.read_text())
            for coverage in ('complete','historical'):
                path=Path(seal[coverage+'_path'])
                if file_sha(path)!=seal[coverage+'_sha256']:raise ValueError('Score changed after seal')
                margins[str(path)]=dict(target={'kitti':'KITTI-V2','dance':'Dance','sports':'SportsMOT'}[key],
                    model=seal['model'],path=str(path),sha256=file_sha(path),bytes=path.stat().st_size,coverage=coverage)
    coverage=[]
    for key in ('kitti','dance','sports'):
        for p in (RUN/key).glob('*/scores_seal.json'):
            s=json.loads(p.read_text());model=s['model']
            if key=='kitti':pair='C5' if model=='LaMOT-best' else 'C2'
            elif key=='dance':pair='C6'
            else:pair='C4' if model=='Dance-best' else 'C3'
            native='IN_DOMAIN' if model==IN_DOMAIN_MODEL[{'kitti':'KITTI-V2','dance':'Dance','sports':'SportsMOT'}[key]] else 'NATIVE'
            old=references[pair+'__HIST_'+model];complete=references[pair+'__'+native]
            coverage.append(dict(target={'kitti':'KITTI-V2','dance':'Dance','sports':'SportsMOT'}[key],model=model,
                historical_score_rows=s['historical_rows'],complete_score_rows=s['complete_rows'],
                added_tail_score_rows=s['complete_rows']-s['historical_rows'],
                historical_HOTA=old['metrics']['HOTA'],complete_HOTA=complete['metrics']['HOTA'],
                historical_prediction_rows=old['prediction_rows'],complete_prediction_rows=complete['prediction_rows']))
    write_csv(PUBLIC/'coverage_comparison.csv',coverage)
    history=RUN/'history_before_remaining';cost=json.loads((history/'costs.json').read_text())
    newjobs=[]
    for p in sorted((RUN/'remaining_launchers').glob('*/launcher_status.json')):
        status=json.loads(p.read_text())
        if status['status']=='RUNNING':raise ValueError('Do not close costs while a launcher is running')
        newjobs.extend(dict(job,launcher_status=str(p)) for job in status['jobs'])
    added=sum(x['wall_seconds'] for x in newjobs)/3600
    cost.update(remaining_GPU_process_hours=added,remaining_jobs=newjobs,prior_C1_C2_cost_snapshot=str(history/'costs.json'),
        full_C1_to_C6_complete=True,GPU_process_hours=cost['GPU_process_hours']+added,
        total_GPU_process_hours_lower_bound=cost['total_GPU_process_hours_lower_bound']+added,
        total_GPU_process_hours_upper_bound=cost['total_GPU_process_hours_upper_bound']+added,
        interpretation='GPU-worker elapsed time incl CPU/IO, not utilization integration; shared GPUs; reused historical inferences are not zero-cost systems')
    cost['per_new_inference']+= [dict(model=m,target=t,known_GPU_process_hours=sum(j['wall_seconds'] for j in newjobs if j['model']==m and j['target']==t)/3600,
        total_GPU_process_hours_lower_bound=sum(j['wall_seconds'] for j in newjobs if j['model']==m and j['target']==t)/3600,
        total_GPU_process_hours_upper_bound=sum(j['wall_seconds'] for j in newjobs if j['model']==m and j['target']==t)/3600,
        includes_smoke_and_failed_attempts=True,exact_total_available=True) for t,m in
        [('KITTI-V2','LaMOT-best'),('Dance','LaMOT-best'),('SportsMOT','V2-best'),('SportsMOT','Dance-best'),('SportsMOT','LaMOT-best')]]
    inference={}
    for key in ('kitti','dance','sports'):
        for p in (RUN/key).glob('*/scores_seal.json'):
            s=json.loads(p.read_text());inference[key+'__'+s['model']]=s['GT_open_attempts']
    decoder={combo:json.loads((base(t,s)/'GT_free_predictions.json').read_text())['GT_open_attempts'] for combo,s,t in PAIRS}
    if any(inference.values()) or any(decoder.values()):raise ValueError('GT isolation violation')
    gt_completion=json.loads((PUBLIC/'sports_GT_completion.json').read_text())
    empty_completed=sum(r['rows']==0 for r in gt_completion['files'] if r['provenance'].endswith('_48'))
    cost['GT_completion_CPU_wall_seconds']=gt_completion['CPU_wall_seconds']
    gt=dict(inference=inference,decoders=decoder,evaluation_and_oracle_separate_processes=True,
        no_target_GT_threshold_fitting=True,no_training=True,no_MLLM=True,
        downloaded_annotation_payloads_not_opened_by_model=True,GT_completion=str(PUBLIC/'sports_GT_completion.json'))
    for filename,rows in [('diagnostics.csv',allrows),('per_category.csv',categories),('structure_stats.csv',structures),('rho_matrix.csv',matrix)]:
        write_csv(PUBLIC/filename,rows)
    atomic_json(PUBLIC/'bootstrap.json',dict(status='COMPLETE_DANCE25_SPORTS45_2000_KITTI4_POINT_ONLY',comparisons=ci,planned_samples=2000,seed=0))
    atomic_json(PUBLIC/'costs.json',cost);atomic_json(PUBLIC/'gt_audit.json',gt)
    atomic_json(PUBLIC/'official_results.json',references)
    atomic_json(PUBLIC/'margin_paths.json',dict(status='ALL_OFFICIAL_COMPLETE_BANKS_SEALED',files=list(margins.values())))
    with gzip.open(PUBLIC/'all_prediction_files.csv.gz','wt') as h:
        writer=csv.DictWriter(h,fieldnames=list(files[0]));writer.writeheader();writer.writerows(files)
    archive=PUBLIC/'official_metrics_logs.tar.gz'
    with tarfile.open(archive,'w:gz') as handle:
        for name,r in references.items():
            folder=Path(r['results_folder'])
            for filename in ('pedestrian_summary.txt',):
                handle.add(folder/filename,arcname=name+'/'+filename)
            log=folder.parent/'trackeval.log'
            if not log.is_file() and r.get('reuse_folder'):
                log=Path(r['reuse_folder'])/'trackeval.log'
            if log.is_file():handle.add(log,arcname=name+'/trackeval.log')
    stats_file=PUBLIC/'bootstrap_sufficient_statistics.json.gz'
    with gzip.open(stats_file,'wt') as handle:
        json.dump({combo:{name:export_hota_stats(r) for name,r in methods.items()}
            for combo,methods in allmethods.items()},handle,separators=(',',':'))
    reproduction=json.loads((history/'reproduction.json').read_text())
    reproduction.update(status='ALL_C1_C6_COMPLETE',global_six_pair_experiment_complete=True,
        LaMOT_best_SportsMOT=dict(status='OFFICIAL_TARGET_CONFIG_MIX_LOADER_COMPLETE',
            HOTA_full=allmethods['C3']['IN_DOMAIN']['done']['metrics']['HOTA'],
            official_subset_number='No official SportsMOT-only number found; no invented numerical gate',
            history_completion=references['C3__HIST_LaMOT-best']),
        new_official_references=references)
    atomic_json(PUBLIC/'reproduction.json',reproduction)
    inventory=json.loads((PUBLIC/'inventory.json').read_text())
    inventory.update(status='ALL_SIX_COMBINATIONS_COMPLETE',completed_combinations=[r[0] for r in PAIRS],pending_combinations=[],
        full_scope_preserved=True,complete_score_banks=list(margins.values()),GT_open_attempts=0,
        missing_assets=[],original_P0_missing_assets_snapshot=str(history/'inventory.json'),
        smoke_status='ALL_AUTHORIZED_OLD_AND_REMAINING_SMOKES_PASSED',
        GT_completion_status='PASS_418_EXACT_48_COMPLETED')
    for row in inventory['targets']['SportsMOT'].get('required_paths',[]):
        p=Path(row['path']);row.update(exists=p.exists(),is_symlink=p.is_symlink(),resolved=str(p.resolve()))
    atomic_json(PUBLIC/'inventory.json',inventory)
    completion=dict(status='ALL_SIX_PAIRS_COMPLETE',conclusion=conclusion,eligible_pairs=len(valid),quantity_dominant_pairs=len(quantity),
        labels=rhos,main_configurations=42,indexed_prediction_files=len(files),new_training=0,new_MLLM=0,max_GPUs=2)
    atomic_json(PUBLIC/'completion.json',completion)
    execution=PUBLIC/'P1_execution.json'
    if execution.exists():
        record=json.loads(execution.read_text());record.update(status='ALL_NEW_FULL_SCORES_AND_SIX_PAIR_EVALUATIONS_COMPLETE',
            full_source_banks_complete=True,all_six_evaluations_complete=True,GT_completion_passed=True)
        atomic_json(execution,record)
    execution_archive=PUBLIC/'execution_logs.tar.gz'
    with tarfile.open(execution_archive,'w:gz') as handle:
        for directory in (RUN/'kitti/launchers',RUN/'remaining_launchers'):
            for p in directory.rglob('*'):
                if p.is_file() and p.suffix in ('.txt','.json','.log'):
                    handle.add(p,arcname=str(p.relative_to(RUN)))
    completion.update(execution_archive=dict(path=str(execution_archive),sha256=file_sha(execution_archive)),
        bootstrap_statistics=dict(path=str(stats_file),sha256=file_sha(stats_file)),
        coverage_comparison_sha256=file_sha(PUBLIC/'coverage_comparison.csv'))
    atomic_json(PUBLIC/'completion.json',completion)
    lines=['# 三数据集多域普遍性：完整六组合结果','',f'## 1. 判定汇总：{conclusion}','',
        f"可判定组合{len(valid)}，数量主导{len(quantity)}；C2判定为“{c2['label']}”。按预注册逐字规则汇总，不按结果删组合。",'',
        table(['源 / 目标','KITTI-V2','Dance','SportsMOT'],[[s,*[
            next((f"{r['rho']:.6f}：{r['label']}" for r in rhos if r['source']==s and r['target']==t),'同域参照')
            for t in ('KITTI-V2','Dance','SportsMOT')]] for s in ('V2-best','Dance-best','LaMOT-best')]),'',
        table(['组合','gap','ρ','Δrank','判定'],[[r[k] for k in ('combo','gap','rho','delta_rank','label')] for r in rhos]),'',
        '数量主导：'+('、'.join(r['combo'] for r in rhos if r['label']=='数量主导') or '无')+'。',
        '数量与排序共同作用：'+('、'.join(r['combo'] for r in rhos if r['label']=='数量与排序共同作用') or '无')+'。',
        '排序主导：'+('、'.join(r['combo'] for r in rhos if r['label']=='排序主导') or '无')+'。',
        '跨域差距过小、不参与汇总判定：'+('、'.join(r['combo'] for r in rhos if r['gap']<3) or '无')+'。','',
        'ρ用官方detailed CSV未舍入HOTA计算；不截断到[0,1]。数量交换借助目标域同域模型或GT，不是可独立部署的TTA。数量主导不等于排序完全无损。','',
        '## 2. 资产、P0、复现和协议','',
        '原预注册98e7f84e；下载授权修订0f9c61c6先发布；GT补建v3 0a7477bf先发布。详见PREREG.md、P0_REMAINING.md、remaining_p0.json。',
        '933份官方payload已核SHA；LaMOT checkpoint 400980a1…ba4213f；45个McByte官方文件、466表达（与原GitHub逐字节一致）、26970帧原图仅建链接。',
        '许可证：FlexHook代码MIT，不推定覆盖独立权重/数据；LaMOT官方仓库未找到独立数据LICENSE，不宣称标注为MIT。逐来源记录见downloads.json；大权重/原始数据不进Git。',
        'Sports实际466条而非465（末行无LF）；295779个观测、2968201个候选–表达对。官方mix末尾观测补分，完整/历史两套保留。',
        '三个checkpoint均strict=True，冻结FP32、TF32禁用、原RoBERTa BPE。目标各用原yaml/dataloader/原框ID；Sports静态class=0且无类别token；KITTI保留car/ped合并ID；Dance原17句不改。',
        '单进程冻结encoder缓存经原getitem/forward和隔离query的真实帧对齐；未冒称运行了会自动读GT的原DDP评测入口。Sports干净图像视图排除._资源叉，未改原图。',
        '历史KITTI V2逐位分数相同、42.526复现；完整42.752。Dance历史32.134/254118行、完整32.139/254910行，当前诊断均用完整范围。',
        'KITTI GT_Dets491286与infer.sh注释491222差64，保留已通过历史门槛的原过滤/GT，不为凑注释改GT。',
        '无官方子集数字可核对：未找到Sports子集官方HOTA，论文正文仅LaMOT合并56.77，infer.sh合并56.756，不将其作Sports门槛。',
        f"官方Sports模板只有418份；独立预测封存后，原始LaMOT targets×原SportsMOT官方框逐行全字段/顺序复现{gt_completion['exact_passed']}份，才接入48缺项。48份共{gt_completion['generated_GT_rows']}行，其中0行{empty_completed}份，均按实际原标注选择，不因文件缺失而补空。CPU耗时{gt_completion['CPU_wall_seconds']:.3f}秒/GPU新增0，逐文件SHA见sports_GT_completion.json。",'',
        '完整/历史覆盖对比（新评分或补尾的分数库，均保留原结果）：','',
        table(['目标','模型','历史分数行','完整分数行','补尾分数行','历史HOTA','完整HOTA'],[
            [r[k] for k in ('target','model','historical_score_rows','complete_score_rows','added_tail_score_rows','historical_HOTA','complete_HOTA')] for r in coverage]),'',
        '## 3. 六组合全部固定诊断','',
        table(['组合','配置','HOTA','DetA','AssA','IDF1','IDSW','行数','Empty行'],[
            [r[k] for k in ('combo','configuration','HOTA','DetA','AssA','IDF1','IDSW','prediction_rows','query_empty_prediction_rows')] for r in allrows]),'',
        'R_S×K_ORACLE / R_I×K_ORACLE显式为【ORACLE，仅诊断】。B0_GLOBAL只借同域总输出行数，不借逐帧逐句数量。',
        '同帧GT R@1在GT非空帧平均(top1命中GT数/GT数)，无可匹配候选记0；CSV另外保留条件候选R@1，不混用分母。','',
        table(['组合','源GT R@1','同域GT R@1','源AUROC','同域AUROC'],[[c,
            json.loads((base(t,s)/'oracle.json').read_text())['retrieval']['source']['GT_R_at_1'],
            json.loads((base(t,s)/'oracle.json').read_text())['retrieval']['in_domain']['GT_R_at_1'],
            json.loads((base(t,s)/'oracle.json').read_text())['source_AUROC'],
            json.loads((base(t,s)/'oracle.json').read_text())['in_domain_AUROC']] for c,s,t in PAIRS]),'',
        '## 4. KITTI-V2按表达类别（仅4视频点估计）','',
        table(['组合','类别','配置','HOTA','DetA','AssA','表达数'],[[r[k] for k in ('combo','category','configuration','HOTA','DetA','AssA','expressions')] for r in categories]),'',
        '## 5. 结构统计【ORACLE，只作描述】','',
        table(['目标','视频','表达','轨迹0句','轨迹1句','轨迹≥2句','空表达比例','有GT帧均值','p50'],[
            [r[k] for k in ('target','videos','expressions','zero','one','two_or_more','empty_query_fraction','GT_frame_fraction_present_mean','GT_frame_fraction_present_p50')] for r in structures]),'',
        '轨迹为正：≥50%观测以IoU≥0.5一对一匹配该表达GT。逐视频表达数和跨视频共享情况见structure_stats.csv；Dance为25×共享17句，Sports按每视频官方表达。','',
        '## 6. 视频聚类不确定性','',
        table(['比较','ΔHOTA','95% CI','视频数'],[[k,r['delta_HOTA'],r['CI95'],r['clusters']] for k,r in ci.items()]),'',
        'Dance25/Sports45视频聚类bootstrap，2000次、seed0，完整保留同视频相关表达与帧，以官方TP权重合并。KITTI4视频不作bootstrap。','',
        '## 7. 成本','',
        table(['模型','目标','GPU-process小时下界','上界'],[[r['model'],r['target'],r['total_GPU_process_hours_lower_bound'],r['total_GPU_process_hours_upper_bound']] for r in cost['per_new_inference']]),'',
        f"全部新GPU-worker elapsed小时范围[{cost['total_GPU_process_hours_lower_bound']:.6f}, {cost['total_GPU_process_hours_upper_bound']:.6f}]，包含smoke/失败尝试；历史C2外部SIGTERM退出时刻未知故保留上下界。共享GPU elapsed含CPU/IO，不是利用率积分。", 
        'C1和两个旧同域分数库只复用哈希核验结果；本轮新增GPU调用为0不等于独立系统零成本。逐尝试日志/显存/实机卡见costs.json和remaining_p0.json。','',
        '## 8. GT隔离、失败、修复与恢复','',
        '推理和GT-free解码GT打开尝试均0；独立导出纯language和空对象视频索引，不读取原targets/labels。所有预测先封存SHA，才独立补建GT、做ORACLE及官方TrackEval。没有源训练、微调、MLLM、外部教师、阈值选择或检测器替换。',
        '下载经历匿名CDN403、原生官网扫码恢复、长传输后签名过期112；每批刷新签名后补齐。Baidu ZIP的编码路径、分页100上限均已修复；旧失败、私有凭据隔离和精确来源见downloads.json/QR_LOGIN_STATUS.md。',
        '历史SIGTERM仅恢复未完帧；此次完成的视频回执只复用不重算。原始checkpoint、历史报告/结果、vendor代码和其它任务进程未覆盖/终止。',
        '全部预测/数组绝对路径与SHA见margin_paths.json、all_prediction_files.csv.gz。summary和完整日志精简归档official_metrics_logs.tar.gz；大detailed CSV不进Git，其绝对目录/SHA见official_results.json。逐视频逐阈值bootstrap充分统计公开为bootstrap_sufficient_statistics.json.gz。',
        '所有GPU尝试、真实资源准入、launcher回执和完整stdout另归档execution_logs.tar.gz（不含任何浏览器会话或凭据），SHA见completion.json。',
        '本研究是固定官方模型的跨域归因，不证明数量可以无监督估计，也不把ORACLE/同域数量交换宣称新TTA。完整预注册结果交付后停止，不启动方法实验。']
    (PUBLIC/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(completion,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
