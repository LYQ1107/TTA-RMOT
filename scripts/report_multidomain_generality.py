"""Report every fixed cell, including blocked cells; no method/threshold fitting."""
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
from scripts.analyze_pf_video_uncertainty import HOTA,unpack,sufficient,pooled_hota,paired_clusters

PAIRS=[('C1','V2-best','Dance'),('C2','Dance-best','KITTI-V2'),('C3','V2-best','SportsMOT'),
       ('C4','Dance-best','SportsMOT'),('C5','LaMOT-best','KITTI-V2'),('C6','LaMOT-best','Dance')]


def table(headers,rows):
    def fmt(x):return '—' if x is None or x=='' else f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x)
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
        ['| '+' | '.join(fmt(x) for x in r)+' |' for r in rows])


def detail(target,name,meta):
    folder=base(target)/name;seal=verify(target,name)
    done=json.loads((folder/'official_completion.json').read_text())
    for f,k in [('pedestrian_summary.txt','summary_sha256'),('pedestrian_detailed.csv','detailed_sha256')]:
        if file_sha(folder/'results'/f)!=done[k]:raise ValueError('Official result changed')
    with (folder/'results/pedestrian_detailed.csv').open() as h:raw=list(csv.DictReader(h))
    hota=HOTA({'PRINT_CONFIG':False});by={r['seq']:unpack(r,hota) for r in raw if r['seq']!='COMBINED'}
    if len(by)!=len(meta['episodes']):raise ValueError('Official episode coverage differs')
    videos=sorted({q['video_id'] for q in meta['episodes']})
    stat=sufficient([hota.combine_sequences({k:r for k,r in by.items() if k.startswith(v+'+')}) for v in videos])
    exact=float(pooled_hota(stat,np.ones(len(videos))))
    combined=next(r for r in raw if r['seq']=='COMBINED')
    if not np.isclose(exact,100*float(combined['HOTA___AUC']),atol=1e-8,rtol=0):raise ValueError('Official pooling mismatch')
    return dict(seal=seal,done=done,rows=by,stats=stat,videos=videos,HOTA_unrounded=exact)


def main():
    require_plan();diagnostics=[];rho=[];structures=[];categories=[];bootstrap={};targets={};allpred=[];references={}
    completed=[]
    for combo,source,target in PAIRS:
        available=combo in ('C1','C2') and (base(target)/'completion.json').exists()
        if not available:
            for configuration in CONFIGS:diagnostics.append(dict(combo=combo,source=source,target=target,configuration=configuration,
                oracle='ORACLE' in configuration,status='BLOCKED_OFFICIAL_ASSETS' if combo not in ('C1','C2') else 'INFERENCE_RUNNING',
                **dict.fromkeys(('HOTA','DetA','AssA','IDF1','IDSW','prediction_rows','query_empty_prediction_rows','source_R_at_1','in_domain_R_at_1','HOTA_unrounded'),None)))
            rho.append(dict(combo=combo,source=source,target=target,gap=None,rho=None,delta_rank=None,label='待官方资产' if combo not in ('C1','C2') else '推理进行中'))
            continue
        completed.append(combo);meta,_,_,_=inputs(target)
        truth=json.loads((base(target)/'oracle.json').read_text());presence={(e['video'],e['expression']):e['present'] for e in truth['episodes']}
        methods={c:detail(target,c,meta) for c in CONFIGS};targets[target]=dict(meta=meta,truth=truth,methods=methods)
        for c in CONFIGS:
            r=methods[c];m=r['done']['metrics']
            diagnostics.append(dict(combo=combo,source=source,target=target,configuration=c,oracle='ORACLE' in c,status='COMPLETE',
                **{k:m[k] for k in ('HOTA','DetA','AssA','IDF1','IDSW')},prediction_rows=r['seal']['prediction_rows'],
                query_empty_prediction_rows=sum(e['selected_rows'] for e in r['seal']['episodes'] if not presence[e['video'],e['expression']]),
                source_R_at_1=truth['retrieval']['source']['R_at_1'],in_domain_R_at_1=truth['retrieval']['in_domain']['R_at_1'],
                HOTA_unrounded=r['HOTA_unrounded']))
            allpred.extend(dict(combo=combo,configuration=c,**e) for e in r['seal']['episodes'])
        h={c:r['HOTA_unrounded'] for c,r in methods.items()};gap=h['IN_DOMAIN']-h['NATIVE']
        ratio=(h['R_SxK_I']-h['NATIVE'])/gap if gap else None
        label='跨域差距过小，不判定' if gap<3 else '数量主导' if ratio>=.7 else '数量与排序共同作用' if ratio>=.3 else '排序主导'
        rho.append(dict(combo=combo,source=source,target=target,gap=gap,rho=ratio,delta_rank=h['IN_DOMAIN']-h['R_SxK_I'],label=label))
        texts=Counter(q['query_text'] for q in meta['episodes']);pervideo=Counter(q['video_id'] for q in meta['episodes'])
        structures.append(dict(target=target,status='COMPLETE_OFFLINE_ORACLE',role='ORACLE_DESCRIPTIVE_ONLY',
            videos=len(pervideo),expressions=len(meta['episodes']),**truth['structure'],per_video_expressions=json.dumps(dict(pervideo),sort_keys=True),
            shared_texts_across_videos=any(n>1 for n in texts.values()),unique_texts=len(texts)))
        for c,r in methods.items():references[target+'__'+c]=dict(r['done'],seal_path=str(base(target)/c/'predictions_seal.json'))
        if target=='KITTI-V2':
            hota=HOTA({'PRINT_CONFIG':False})
            for c in ('NATIVE','IN_DOMAIN','R_SxK_I'):
                for category in ('car','pedestrian'):
                    chosen={q['video_id']+'+'+q['expression_id']:methods[c]['rows'][q['video_id']+'+'+q['expression_id']] for q in meta['episodes']
                        if ('car' if 'car' in q['query_text'].lower() else 'pedestrian')==category}
                    result=hota.combine_sequences(chosen)
                    categories.append(dict(combo=combo,source=source,target=target,category=category,configuration=c,status='COMPLETE',
                        HOTA=100*float(np.mean(result['HOTA'])),DetA=100*float(np.mean(result['DetA'])),AssA=100*float(np.mean(result['AssA'])),
                        expressions=len(chosen),uncertainty='4 videos; point estimates only'))
            for name in ('HIST_V2-best','HIST_Dance-best'):
                r=detail(target,name,meta);references[target+'__'+name]=dict(r['done'],seal_path=str(base(target)/name/'predictions_seal.json'))
        else:
            nv=len(methods['NATIVE']['videos']);draws=np.random.default_rng(0).multinomial(nv,np.ones(nv)/nv,size=2000)
            for a,b in [('R_SxK_I','NATIVE'),('IN_DOMAIN','R_SxK_I')]:
                ma,mb=methods[a],methods[b];samples=paired_clusters(ma['stats'],mb['stats'],draws)
                bootstrap[combo+'__'+a+'_minus_'+b]=dict(combo=combo,target=target,first=a,second=b,clusters=nv,samples=2000,seed=0,
                    delta_HOTA=float(paired_clusters(ma['stats'],mb['stats'],np.ones(nv))),CI95=np.percentile(samples,[2.5,97.5]).tolist(),
                    unit='video, all queries/frames kept together',pooling='official TP-weighted sufficient statistics')
    known={s['target'] for s in structures}
    for t in ('KITTI-V2','Dance','SportsMOT'):
        if t not in known:structures.append(dict.fromkeys(list(structures[0]),None)|dict(target=t,status='BLOCKED_OFFICIAL_ASSETS' if t=='SportsMOT' else 'NOT_YET_EVALUATED',role='ORACLE_NOT_RUN'))
    # Add blocked category rows explicitly rather than silently drop C5.
    for c,s,t in PAIRS:
        if t=='KITTI-V2' and c not in completed:
            for cat in ('car','pedestrian'):
                for rule in ('NATIVE','IN_DOMAIN','R_SxK_I'):categories.append(dict(combo=c,source=s,target=t,category=cat,configuration=rule,status='BLOCKED_OFFICIAL_ASSETS' if c=='C5' else 'INFERENCE_RUNNING',
                    HOTA=None,DetA=None,AssA=None,expressions=None,uncertainty='4 videos; point estimates only'))
    for f,rows in [('diagnostics.csv',diagnostics),('per_category.csv',categories),('structure_stats.csv',structures)]:write_csv(PUBLIC/f,rows)
    matrix=[]
    for s in ('V2-best','Dance-best','LaMOT-best'):
        for t,inside in [('KITTI-V2','V2-best'),('Dance','Dance-best'),('SportsMOT','LaMOT-best')]:
            r=next((r for r in rho if r['source']==s and r['target']==t),None)
            matrix.append(dict(source=s,target=t,combo=r['combo'] if r else '',gap=r['gap'] if r else None,rho=r['rho'] if r else None,
                delta_rank=r['delta_rank'] if r else None,label=r['label'] if r else '同域参照'))
    write_csv(PUBLIC/'rho_matrix.csv',matrix)
    atomic_json(PUBLIC/'bootstrap.json',dict(status='COMPLETE_FOR_C1_OTHER_PAIRS_BLOCKED_OR_RUNNING',comparisons=bootstrap,
        planned_samples=2000,seed=0,KITTI='4 videos: no bootstrap',SportsMOT='not run; missing official assets'))
    atomic_json(PUBLIC/'official_results.json',references)
    with gzip.open(PUBLIC/'all_prediction_files.csv.gz','wt') as h:
        w=csv.DictWriter(h,fieldnames=list(allpred[0]));w.writeheader();w.writerows(allpred)
    margins=[]
    for target in targets:
        _,_,_,paths=inputs(target)
        margins.extend(dict(target=target,model=m,path=str(p),sha256=file_sha(p),bytes=p.stat().st_size)
            for m,p in zip(('V2-best','Dance-best') if target=='Dance' else ('Dance-best','V2-best'),paths))
    atomic_json(PUBLIC/'margin_paths.json',dict(status='SEALED_COMPLETED_TARGET_BANKS',files=margins))
    costs=[]
    for status in (RUN/'kitti/launchers').glob('*/launcher_status.json'):
        for job in json.loads(status.read_text())['jobs']:costs.append(dict(job,launcher_status=str(status)))
    cost=dict(GPU_process_hours=sum(r['wall_seconds'] for r in costs)/3600,per_job=costs,max_GPUs=2,
        includes_failed_environment_attempts_and_smokes=True,teacher_calls=0,optimizer_updates=0,
        actual_full_C2_complete='C2' in completed,partial_completed_job_costs_only_if_launcher_still_running=True)
    atomic_json(PUBLIC/'costs.json',cost)
    gt=dict(inference={m:json.loads((RUN/'kitti'/m/'scores_seal.json').read_text())['GT_open_attempts'] for m in CHECKPOINTS if (RUN/'kitti'/m/'scores_seal.json').exists()},
        decoders={t:json.loads((base(t)/'GT_free_predictions.json').read_text())['GT_open_attempts'] for t in targets},
        evaluation_and_oracle_separate_processes=True,no_target_GT_threshold_fitting=True,no_training=True,no_MLLM=True,
        downloaded_annotation_payloads_not_opened_by_model=True)
    atomic_json(PUBLIC/'gt_audit.json',gt)
    publication=json.loads((RUN/'prereg_amendment_publication.json').read_text())
    lines=['# 三数据集多域普遍性：授权恢复后的真实结果','',
        '## 1. 判定汇总','',
        '状态：C1、C2独立授权部分'+('均已完成' if 'C2' in completed else '中C1已完成、C2进行中')+'；C3–C6仍等待官方LaMOT权重和McByte/FlexHook数据包。**完整六组合未完成，不计算全局2/3判定，不宣称普遍成立/部分成立/不普遍。**','',
        table(['源 / 目标','KITTI-V2','Dance','SportsMOT'],[[s]+[next((f"{r['rho']:.6f}：{r['label']}" if r['rho'] is not None else r['label']) for r in matrix if r['source']==s and r['target']==t) for t in ('KITTI-V2','Dance','SportsMOT')] for s in ('V2-best','Dance-best','LaMOT-best')]),'',
        table(['组合','gap','ρ','Δrank','固定规则判定'],[[r[k] for k in ('combo','gap','rho','delta_rank','label')] for r in rho]),'',
        'ρ使用官方detailed CSV未舍入HOTA计算；表内HOTA展示summary数值。缺失单元不是ρ=0，不删组合来做普遍性结论。','',
        '## 2. P0、下载、冒烟和自检','',
        '补充预注册先推送'+publication['remote_commit']+'；恢复P0先推送e7bf73cf81a1ed6afc775815581e0564a77cbea3，随后才继续P1。原93行PREREG正文和首次P0封存保留。详见[P0_RESUMED.md](P0_RESUMED.md)、[downloads.json](downloads.json)。','',
        'Sports seqmap实际466条、45视频；原465为文件末行无换行符造成wc -l少计。原图45视频26970有效帧已链接，没有重新下载。官方LaMOT GitHub原始Sports标注508份、45视频已下载，数字ID名不同于FlexHook文本名，未冒充通过466条一一映射。','',
        '百度提取码验证成功且可列官方LaMOT-best（995768578字节）和45个McByte预测；匿名sharedownload errno2且无dlink，缺签名/参数。未把errno2谎称登录错误；公开BaiduPCS-Go客户端规定账号登录为必做，无账号凭据故停止该路径，不索取密码、不借他人登录、不自跑tracker。','',
        '需要手动下载的精确网盘路径（分享根实际为/FlexHook）：','',
        '- /FlexHook/SOTA_ckpts/LaMOT-best.pth','- /FlexHook/tracker_outputs/McByte-SportsMOT/（45文件逐个列在downloads.json）',
        '- /FlexHook/datasets/LaMOT-main/annotations_v1/val/SportsMOT/','- /FlexHook/datasets/LaMOT-main/sequences/val/SportsMOT_labels.json',
        '- /FlexHook/datasets/LaMOT-main/sequences/val/SportsMOT_RESOLUTION.json','- /FlexHook/datasets/LaMOT-main/sequences/val/TAO_frame2file.json（仅mix构造器辅助元数据，不运行TAO）','',
        '目的目录：/data2/user/TTA-RMOT-assets/flexhook/lamot_20261101/flexhook_baidu/；不覆盖已有文件。LaMOT公开仓库无独立数据LICENSE，不能假定MIT；FlexHook代码MIT但单独checkpoint/数据授权未写。','',
        '两套KITTI实机smoke strict全部key匹配，原生getitem/forward与冻结编码缓存数值误差≤2.1e-5、无符号翻转；box/grid完全一致。使用官方kitti2.yaml、infer.sh显式rope-swin-tiny/roberta覆盖、FP32、规范RoBERTa BPE、官方car/ped合并/帧偏移；没有额外类别token，所有候选同等打分。1视频估算V2 0.572601、Dance 0.630457 GPU小时，均未触发6小时停止门槛。','',
        'V2 KITTI历史2355528行原值保留，只新增69854尾观测分数，完整2425382行已封存，历史npy SHA保持cb0a935c…1051687。完整覆盖原生HOTA如实见下表，不假定等于42.526。','']
    if 'KITTI-V2' in targets:
        rows=[]
        for m in ('IN_DOMAIN','HIST_V2-best','NATIVE','HIST_Dance-best'):
            r=references['KITTI-V2__'+m];rows.append([m,r['metrics']['HOTA'],r['prediction_rows'],r['reused_after_all_episode_row_equality']])
        lines+=[table(['KITTI覆盖/模型','HOTA','行数','逐行相等后复用旧评测'],rows),'',
            '历史V2分数逐位相同，862份原生预测逐行数值相同，HOTA42.526自检通过。新完整覆盖与历史覆盖均保留，不混用参照。','']
    lines+=['Dance历史32.134/254118行与完整32.139/254910行已核对SHA，新诊断统一完整范围。论文最新版正文只有LaMOT合并56.77，infer.sh合并56.756；未找到Sports子集官方数字，仓库无补充PDF、CVF补充入口403。只有官方资产核验全过后才允许Sports smoke和后续正式推理。','',
        '## 3. 所有组合和固定配置','',
        table(['组合','配置','HOTA','DetA','AssA','IDF1','IDSW','预测行','Empty行','状态'],[[r[k] for k in ('combo','configuration','HOTA','DetA','AssA','IDF1','IDSW','prediction_rows','query_empty_prediction_rows','status')] for r in diagnostics]),'',
        'R_S×K_I和逆向数量交换是事后诊断；I(T)是目标域官方模型，不是可免费获得的Source-Free TTA门控。ORACLE两臂显式使用目标GT数量，仅诊断，不是可部署结果。B0_GLOBAL保持同域总行数，但不含逐帧逐表达的数量信息。','']
    retrieval_rows=[]
    for t,r in targets.items():
        for m,x in r['truth']['retrieval'].items():retrieval_rows.append([t,m,x['R_at_1'],x['Hit_at_1'],x['valid_frames'],r['truth'][m+'_AUROC']])
    lines+=[table(['目标','排序来源','R@1(原候选召回口径)','Hit@1','有效帧','全候选AUROC'],retrieval_rows),'',
        'R@1复用旧离线脚本的条件候选召回：在存在IoU≥0.5候选的帧，top1命中的正候选数/该帧正候选数，再对帧求均值；正候选按任一GT IoU≥0.5，tie=(-margin,numeric ID)。不是全局AP，也不是top1命中率（Hit@1另列）。GT轨迹结构统计使用一对一最大基数匹配，二者分母不混用。','',
        '## 4. KITTI按表达类别','',table(['组合','类别','配置','HOTA','DetA','AssA','表达数','状态'],[[r[k] for k in ('combo','category','configuration','HOTA','DetA','AssA','expressions','status')] for r in categories]),'',
        '按已导出的句子car/people目标类型分组，保留该类表达的全部候选（不事后按预测类别删框）；官方HOTA.combine_sequences聚合。只有4个验证视频，不做bootstrap。','',
        '## 5. 结构统计（ORACLE，仅描述）','',table(['目标','轨迹0表达','轨迹1表达','轨迹≥2表达','空表达率','Present GT帧率均值','p50','状态'],
            [[s.get(k) for k in ('target','zero','one','two_or_more','empty_query_fraction','GT_frame_fraction_present_mean','GT_frame_fraction_present_p50','status')] for s in structures]),'',
        '轨迹正例：一对一IoU≥0.5匹配观测达到该物理预测轨迹≥50%；不是预测ID=GT ID。逐视频表达数、共享文本见structure_stats.csv；Dance17句跨25视频共享。Sports未知项留空，不能以现成图像替代官方McByte轨迹统计。','',
        '## 6. 不确定性','',table(['比较','ΔHOTA','95%视频聚类CI'],[[k,v['delta_HOTA'],str(v['CI95'])] for k,v in bootstrap.items()]),'',
        '每次2000抽样、seed0，整视频为聚类单位，复用官方TP加权HOTA充分统计且对COMBINED自检；不是对相关表达/帧独立重采样。KITTI仅4视频，点估计。Sports45视频未运行，不伪造CI。','',
        '## 7. 成本','',f"已记账GPU进程时间共{cost['GPU_process_hours']:.6f}小时（含本轮失败环境尝试和smoke；若C2仍运行，其未完成时长暂不计）。每次job详见costs.json。最多2物理GPU，入场按进程少/显存足够共享，不终止别人。训练、微调、MLLM调用均0。",'',
        'C1七套官方CSV全部在425份预测逐行一致和原SHA核验后复用，未重跑健康推理/评测。新KITTI评测记录在official_results.json；原始日志、margin、预测保留在服务器，Git只存小型表/清单/压缩指标日志。','',
        '## 8. GT审计、失败和恢复','',
        '模型/GT-free解码的GT打开尝试为0；官方评价和oracle在独立进程、预测SHA封存后执行，未向模型写回标签、阈值或参数。纯文本runtime从旧白名单元数据导出，不打开原表达标签。下载的JSON只复制/解压/哈希，未作为模型GT输入。','',
        '首次选错vllmedit环境缺yacs，两子进程6秒退出；切回原dfot（torch2.7.0+cu128/transformers5.4.0），配置回归通过，未修改共享环境。smoke后P1发布锁按要求阻止提前全量运行，远端盘点发布后恢复，0005完成单元未重跑。V2 smoke对旧分数最大重算差0.0010823保留记录，不覆盖旧分数；实际官方forward缓存误差≤2.1e-5且无符号翻转。','',
        '所有新代码和run均独立；vendor已有mydataloader旧test补丁原样保留，本轮未改vendor代码；仅新增授权数据软链接。历史结果、源checkpoint和其他项目进程未覆盖。完整范围继续保留，当前仅因缺官方二进制资产停止C3–C6。','',
        '恢复路径：results/runs/multidomain_generality_20261101；大文件SHA见margin_paths.json、all_prediction_files.csv.gz、official_results.json和downloads.json。手动资产到位后先strict/45视频/466序列/smoke门槛，再按同一PREREG继续；不生成替代tracker，不开展方法或训练。','',
        '官方来源：[FlexHook](https://github.com/buptLwz/FlexHook)、[DATASET.md](https://github.com/buptLwz/FlexHook/blob/main/datasets/DATASET.md)、[LaMOT](https://github.com/Nathan-Li123/LaMOT)、[论文v5](https://arxiv.org/html/2503.07516v5)。']
    (PUBLIC/'REPORT.md').write_text('\n'.join(lines)+'\n')
    atomic_json(PUBLIC/'completion.json',dict(status='PARTIAL_C1_C2_COMPLETE_OFFICIAL_ASSETS_BLOCK_C3_TO_C6' if 'C2' in completed else 'PARTIAL_C1_COMPLETE_C2_RUNNING',
        completed_combinations=completed,full_scope=['C1','C2','C3','C4','C5','C6'],global_verdict='NOT_COMPUTED_INCOMPLETE_PREREGISTERED_SCOPE',
        prereg_amendment=publication,gt_audit=gt,costs=cost['GPU_process_hours']))
    print(json.dumps(dict(event='MULTIDOMAIN_REPORT_WRITTEN',completed=completed,rho=rho),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
