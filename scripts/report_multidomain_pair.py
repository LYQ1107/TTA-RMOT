"""Publish each fully evaluated pair while other frozen workers keep running."""
from argparse import ArgumentParser
import csv
import gzip
import json
from pathlib import Path
import sys
import tarfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.multidomain_diagnostics import *
from scripts.report_multidomain_generality import PAIRS,detail,table,export_hota_stats
from scripts.analyze_pf_video_uncertainty import HOTA,paired_clusters


def main():
    parser=ArgumentParser();parser.add_argument('--combo',choices=[r[0] for r in PAIRS],required=True)
    args=parser.parse_args();combo,source,target=next(r for r in PAIRS if r[0]==args.combo)
    done=json.loads((base(target,source)/'completion.json').read_text())
    meta,_,_,_=inputs(target,source)
    truth=json.loads((base(target,source)/'oracle.json').read_text())
    present={(q['video'],q['expression']):q['present'] for q in truth['episodes']}
    methods={name:detail(target,name,meta,source) for name in CONFIGS}
    rows=[];references={};predictions=[]
    for name,r in methods.items():
        m=r['done']['metrics']
        rows.append(dict(combo=combo,source=source,target=target,configuration=name,oracle='ORACLE' in name,status='COMPLETE',
            GT_protocol=SPORTS_GT_PROTOCOL if target=='SportsMOT' else 'OFFICIAL_UNCHANGED',
            **{k:m[k] for k in ('HOTA','DetA','AssA','IDF1','IDSW')},prediction_rows=r['seal']['prediction_rows'],
            query_empty_prediction_rows=sum(e['selected_rows'] for e in r['seal']['episodes'] if not present[e['video'],e['expression']]),
            source_R_at_1=truth['retrieval']['source']['GT_R_at_1'],in_domain_R_at_1=truth['retrieval']['in_domain']['GT_R_at_1'],
            source_candidate_R_at_1=truth['retrieval']['source']['R_at_1'],in_domain_candidate_R_at_1=truth['retrieval']['in_domain']['R_at_1'],
            HOTA_unrounded=r['HOTA_unrounded']))
    h={k:v['HOTA_unrounded'] for k,v in methods.items()};gap=h['IN_DOMAIN']-h['NATIVE']
    ratio=(h['R_SxK_I']-h['NATIVE'])/gap if gap else None
    label='跨域差距过小，不判定' if gap<3 else '数量主导' if ratio>=.7 else '数量与排序共同作用' if ratio>=.3 else '排序主导'
    categories=[]
    if target=='KITTI-V2':
        mapping=json.loads((RUN/'source_query_categories.json').read_text())['mapping'];evaluator=HOTA({'PRINT_CONFIG':False})
        for name in ('NATIVE','IN_DOMAIN','R_SxK_I'):
            for category in ('car','pedestrian'):
                selected={q['video_id']+'+'+q['expression_id']:methods[name]['rows'][q['video_id']+'+'+q['expression_id']]
                    for q in meta['episodes'] if mapping[q['video_id']+'+'+q['expression_id']]==category}
                pooled=evaluator.combine_sequences(selected)
                categories.append(dict(combo=combo,source=source,target=target,category=category,configuration=name,status='COMPLETE',
                    HOTA=100*float(np.mean(pooled['HOTA'])),DetA=100*float(np.mean(pooled['DetA'])),AssA=100*float(np.mean(pooled['AssA'])),
                    expressions=len(selected),uncertainty='4 videos; point estimates only'))
    ci={}
    if target!='KITTI-V2':
        n=len(methods['NATIVE']['videos']);draws=np.random.default_rng(0).multinomial(n,np.ones(n)/n,size=2000)
        for a,b in [('R_SxK_I','NATIVE'),('IN_DOMAIN','R_SxK_I')]:
            samples=paired_clusters(methods[a]['stats'],methods[b]['stats'],draws)
            ci[a+'_minus_'+b]=dict(delta_HOTA=h[a]-h[b],CI95=np.percentile(samples,[2.5,97.5]).tolist(),clusters=n,samples=2000,seed=0)
    for result in done['methods']:
        name=result['method'];seal=verify(target,name,source);folder=base(target,source)/name
        references[name]=dict(result,results_folder=str(folder/'results'))
        predictions.extend(dict(configuration=name,**e) for e in seal['episodes'])
    index=PUBLIC/(combo+'_all_prediction_files.csv.gz')
    with gzip.open(index,'wt') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    archive=PUBLIC/(combo+'_official_logs.tar.gz')
    with tarfile.open(archive,'w:gz') as handle:
        for name,r in references.items():
            for filename in ('pedestrian_summary.txt',):
                handle.add(Path(r['results_folder'])/filename,arcname=name+'/'+filename)
            log=Path(r['results_folder']).parent/'trackeval.log'
            if not log.is_file() and r.get('reuse_folder'):log=Path(r['reuse_folder'])/'trackeval.log'
            if log.exists():handle.add(log,arcname=name+'/trackeval.log')
    stats_file=PUBLIC/(combo+'_bootstrap_stats.json.gz')
    with gzip.open(stats_file,'wt') as handle:
        json.dump({name:export_hota_stats(r) for name,r in methods.items()},handle,separators=(',',':'))
    jobs=[json.loads(p.read_text()) for p in (RUN/'remaining_launchers').glob('*/'+target+'_'+source+'.json')]
    shared_jobs=[json.loads(p.read_text()) for p in (RUN/'remaining_launchers').glob('*/SportsMOT_LaMOT-best.json')] if target=='SportsMOT' else []
    shared_hours=sum(r['wall_seconds'] for r in shared_jobs)/3600
    result=dict(status='COMPLETE_SINGLE_PAIR_NOT_GLOBAL_CONCLUSION',combo=combo,source=source,target=target,
        GT_protocol=SPORTS_GT_PROTOCOL if target=='SportsMOT' else 'OFFICIAL_UNCHANGED',
        gap=gap,rho=ratio,delta_rank=h['IN_DOMAIN']-h['R_SxK_I'],label=label,metrics=rows,categories=categories,
        uncertainty=ci or 'KITTI4 videos: point estimates only',structure=truth['structure'],retrieval=truth['retrieval'],
        source_AUROC=truth['source_AUROC'],in_domain_AUROC=truth['in_domain_AUROC'],references=references,
        new_GPU_worker_elapsed_hours=sum(r['wall_seconds'] for r in jobs)/3600,cost_jobs=jobs,
        shared_reference_new_GPU_worker_elapsed_hours=shared_hours,shared_reference_cost_jobs=shared_jobs,
        cost_scope=('Source inference only in the source cost field; newly computed shared LaMOT Sports reference cost is separately disclosed, charge once in the global total' if target=='SportsMOT' else
            'Only new source inference including smoke; inherited reference inference not remeasured, not a zero-cost system'),
        prediction_index=dict(path=str(index),sha256=file_sha(index),files=len(predictions)),
        official_archive=dict(path=str(archive),sha256=file_sha(archive)),
        sufficient_statistics=dict(path=str(stats_file),sha256=file_sha(stats_file)),
        inference_GT_open_attempts=0,GT_free_decoder_open_attempts=0,GT_only_independent_offline=True,
        optimizer_updates=0,teacher_calls=0,full_six_pair_experiment_complete=False)
    atomic_json(PUBLIC/(combo+'_RESULTS.json'),result);write_csv(PUBLIC/(combo+'_diagnostics.csv'),rows)
    # Update only the completed cells; preserve every historical metric value.
    prior=list(csv.DictReader((PUBLIC/'diagnostics.csv').open()));fresh={r['configuration']:r for r in rows}
    shared=[]
    for r in prior:
        if r['combo']==combo:r={k:fresh[r['configuration']].get(k,'') for k in r}
        elif r['status']=='BLOCKED_OFFICIAL_ASSETS':r['status']='P1_RUNNING_AWAIT_EVALUATION'
        shared.append(r)
    write_csv(PUBLIC/'diagnostics.csv',shared)
    matrix=list(csv.DictReader((PUBLIC/'rho_matrix.csv').open()))
    for r in matrix:
        if r['combo']==combo:r.update(gap=gap,rho=ratio,delta_rank=result['delta_rank'],label=label)
        elif r['label']=='待官方资产':r['label']='P1运行中，待评测'
    write_csv(PUBLIC/'rho_matrix.csv',matrix)
    if categories:
        old=list(csv.DictReader((PUBLIC/'per_category.csv').open()));new={(r['category'],r['configuration']):r for r in categories}
        write_csv(PUBLIC/'per_category.csv',[{k:new[r['category'],r['configuration']].get(k,'') for k in r} if r['combo']==combo else r for r in old])
    lines=[f'# {combo} {source} → {target} 完整阶段结果','',
        f"预注册判定：{label}；ρ={ratio}；gap={gap:.6f}；Δrank={result['delta_rank']:.6f}。",'',
        f"完整{len(meta['videos'])}视频/{len(meta['episodes'])}表达。预测先封存后GT，C1/C2不重跑，其余组合仍继续；此表不是六组合最终汇总结论。",'',
        'SportsMOT GT采用用户预注册v4：466条统一按原始LaMOT targets×SportsMOT框重建；旧418模板410一致/8不一致，原文件和v3失败均保留。不冒称原模板协议精确复现。' if target=='SportsMOT' else 'GT协议保持原官方模板/过滤规则不变。','',
        table(['配置','HOTA','DetA','AssA','IDF1','IDSW','预测行','Empty行'],[[r[k] for k in
            ('configuration','HOTA','DetA','AssA','IDF1','IDSW','prediction_rows','query_empty_prediction_rows')] for r in rows]),'',
        'R_S×K_ORACLE、R_I×K_ORACLE是【ORACLE，仅诊断】，不当新TTA结果。数量交换借助目标同域模型的k，不是GT-free可部署方法。','',
        table(['类别','配置','HOTA','DetA','AssA','表达数'],[[r[k] for k in ('category','configuration','HOTA','DetA','AssA','expressions')] for r in categories]) if categories else
        table(['比较','ΔHOTA','95% CI','视频数'],[[k,r['delta_HOTA'],r['CI95'],r['clusters']] for k,r in ci.items()]),'',
        f"GT R@1：源{truth['retrieval']['source']['GT_R_at_1']:.6f}，同域{truth['retrieval']['in_domain']['GT_R_at_1']:.6f}；AUROC：源{truth['source_AUROC']:.6f}，同域{truth['in_domain_AUROC']:.6f}。",'',
        'KITTI只有4视频，报告点估计，不构造帧级显著性。' if target=='KITTI-V2' else '视频聚类2000次/seed0，不独立重采样表达或帧。','',
        f"本组新增source GPU-worker elapsed（含smoke）{result['new_GPU_worker_elapsed_hours']:.6f}小时；共享GPU含CPU/IO，不是利用率积分。",
        f'本轮新算的共享LaMOT→Sports同域参照另需{shared_hours:.6f} GPU-worker小时，在总账只计一次；上行只是source成本，不是独立系统总成本。' if target=='SportsMOT' else
        '旧同域分数虽复用，不能将其解释为独立系统零成本。','',
        f'机器表、summary/完整日志精简归档、逐预测SHA分别见{combo}_RESULTS.json、{combo}_official_logs.tar.gz、{combo}_all_prediction_files.csv.gz。大detailed CSV保留原绝对路径/SHA，不进Git。',
        f'逐视频、逐HOTA阈值的完整bootstrap充分统计见{combo}_bootstrap_stats.json.gz，可独立重算官方合并HOTA和视频聚类区间。',
        '保持原模型/参数/候选/GT过滤。其它固定组合未齐之前不宣布跨三域普遍性。']
    (PUBLIC/(combo+'_REPORT.md')).write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('status','combo','gap','rho','delta_rank','label','new_GPU_worker_elapsed_hours')}),flush=True)


if __name__=='__main__':main()
