#!/usr/bin/env python3
"""Offline D-FIT table B, only after every available final-step prediction seal.

Reuse the prior official evaluator, GT grouping, dense ranking and video
bootstrap. Frozen/D0/old PSR results are copies, never retrained or rerun.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from contextlib import redirect_stdout
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.prepare_psr_controls import prepare
from scripts.prepare_mllm_evaluation_controls import stage_reference
from scripts.pf_diag_common import FROZEN,DATA,TRACKS,official_metrics,official_video_values,load_source_video
from scripts.analyze_pf_full_offline import analyze as attribution
from scripts.analyze_psr_matrix import dense,official_query_groups
from scripts.analyze_pf_video_uncertainty import paired_clusters
from scripts.analyze_uda_text_signal_offline import scores_for
from scripts.report_diag_fit import RUN,PUBLIC,TAGS,package

OLD=ROOT/'results/runs/psr_val_20261029_v1/evaluation/merged'


def execution_history(report):
    """Retain the real compatibility failures when replacing placeholder tables."""
    return report.split('### 兼容性执行记录',1)[1].split('### 已完成的共同教师标签基线',1)[0].strip()


def classify(fit,results):
    qualified={tag:r.get('fit_qualified',False) for tag,r in fit['methods'].items() if tag in TAGS.values()}
    frozen=results['FROZEN']
    wins={tag:tag in results and results[tag]['Present_HOTA']>frozen['Present_HOTA'] and results[tag]['R1_outside_explicit_window']>frozen['R1_outside_explicit_window'] for tag in qualified}
    if any(qualified[t] and wins[t] for t in qualified):
        return '知识迁移通路成立，之前是训练太弱','下一步直接放大到Dance Train密集标注，做正式Train→Val；本轮不自行执行。'
    shallow=[t for t in ('D1_OPT','D2_OPT_POS') if qualified.get(t)]
    if any(results[t]['Present_HOTA']<=frozen['Present_HOTA'] and results[t]['R1_outside_explicit_window']<=frozen['R1_outside_explicit_window'] for t in shallow):
        return '学得进去，但稀疏标签本身不够','下一步做Dance Train GT轨迹上的密集标注与帧内列表式蒸馏；本轮不自行执行。'
    if not shallow and qualified.get('D3_CAP'):
        return '浅层adapter容量不足','正式方法放开全部PCD层和head；本轮不自行开展正式训练。'
    if not any(qualified.values()):
        return '教师和学生的输入不一致，或者教师标签自相矛盾','按用户预注册分支原样归类；同时报告端点一致率。若D3数值不可训练，须明确它没有最终拟合指标，不能把分支标签当已证实的因果结论。'
    return '预注册规则未覆盖的混合结果','至少一个拟合合格，但Present/R@1只改善其中一项；不添加阈值或事后新判定，停止并交由用户决定。'


def analyze():
    sealed=json.loads((RUN/'predictions_seal.json').read_text())
    if sealed['status']!='ALL_TRAINABLE_ARMS_25VIDEO_425QUERY_FINAL_STEP_SEALED':raise ValueError('seal all trained arms before offline GT')
    root=RUN/'evaluation/merged';root.mkdir(parents=True,exist_ok=True)
    methods=list(sealed['methods'])
    for method,r in sealed['methods'].items():
        if file_sha(Path(r['run_metadata']))!=r['run_metadata_sha256']:raise ValueError('prediction seal changed')
        for e in r['episodes']:
            if file_sha(Path(e['prediction_path']))!=e['prediction_sha256'] or file_sha(Path(e['margins_path']))!=e['margins_sha256']:raise ValueError('sealed prediction changed')
    # This separate child exits before any GT reader is called. Its existing
    # target-annotation audit hook remains enabled for the entire preparation.
    subprocess.run([sys.executable,str(ROOT/'scripts/prepare_psr_controls.py'),'--evaluation-root',str(root),
        '--training-root',str(RUN/'students'),'--methods',*methods],cwd=ROOT,check=True)
    for method,original in [('FROZEN',FROZEN/'ZERO_UPDATE'),('SOURCE_REHEARSAL',ROOT/'results/runs/uda_val_evaluation_20261026_v1/SOURCE_REHEARSAL'),
                            ('NAIVE_PAIRED',OLD/'NAIVE_PAIRED'),('PSR',OLD/'PSR')]:
        stage_reference(original,root,method)
        for name in ('pedestrian_summary.txt','pedestrian_detailed.csv'):
            shutil.copy2(original/'results'/name,root/method/'results'/name)
    derived=[name for m in methods for name in ('FROZEN_AT_'+m+'_COUNT',m+'_AT_FROZEN_COUNT')]
    official_new=[*methods,*derived]
    def evaluate(method):
        dest=RUN/'official_jobs'/method;dest.mkdir(parents=True,exist_ok=True)
        link=dest/method
        if not link.exists():link.symlink_to(root/method,target_is_directory=True)
        done=dest/'completion.json'
        if done.exists() and json.loads(done.read_text())['exit_code']==0:return
        start=time.perf_counter()
        with (dest/'stdout.txt').open('a') as log:
            code=subprocess.call([sys.executable,str(ROOT/'scripts/evaluate_flexhook_dance_tta.py'),
                '--output-root',str(dest),'--dataset-root',str(DATA),'--vendor-root',str(ROOT/'vendor/FlexHook'),
                '--methods',method],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        atomic_json(done,dict(exit_code=code,seconds=time.perf_counter()-start,method=method))
        print(json.dumps(dict(event='D_FIT_OFFICIAL_DONE',method=method,exit_code=code)),flush=True)
        if code:raise RuntimeError('official evaluation failed: '+method)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(evaluate,m) for m in official_new]):future.result()
    names=['FROZEN','NAIVE_PAIRED',*methods];metrics={};reports={};groups={}
    for method in names:
        dest=PUBLIC/'Val'/method;dest.mkdir(parents=True,exist_ok=True)
        for name in ('pedestrian_summary.txt','pedestrian_detailed.csv'):shutil.copy2(root/method/'results'/name,dest/name)
        metrics[method]=official_metrics(root/method/'results')
        with (dest/'offline_stdout.txt').open('w') as log,redirect_stdout(log):
            reports[method]=attribution(root,method,FROZEN,DATA,TRACKS,dest/'offline_attribution.json')
        groups[method]=official_query_groups(root/method,reports[method])
    if abs(metrics['FROZEN']['HOTA']-8.218)>1e-4 or reports['FROZEN']['query_present_FP']!=1996 or reports['FROZEN']['query_empty_FP']!=87371:
        raise ValueError('historical full-candidate baseline/GT protocol mismatch; do not retune it')
    (PUBLIC/'dense').mkdir(exist_ok=True)
    ranking,teaching=dense(root,['NAIVE_PAIRED','PSR',*methods],PUBLIC/'dense')
    # Fixed historical PSR correct-POS374 is held constant across ALL arms;
    # it is not their Naive363 training-positive subset and is never a label.
    with (ROOT/'results/psr/full/PSR_direct_teacher_offline.csv').open() as stream:
        correct=[r for r in csv.DictReader(stream) if r['state']=='POS' and r['official_match']=='True']
    if len(correct)!=374:raise ValueError('historical fixed correct-POS diagnostic set changed')
    selected={m:0 for m in names}
    for video in sorted({r['video'] for r in correct}):
        bank=load_source_video(video);positions={(int(t),int(i)):k for k,(t,i) in enumerate(zip(bank['frame'],bank['ids']))}
        qcol={q:j for j,(q,_) in enumerate(bank['expressions'])}
        for method in names:
            scores=bank['scores'] if method=='FROZEN' else scores_for(root,method,bank)
            selected[method]+=sum(scores[positions[int(r['frame']),int(r['track'])],qcol[r['query']]]>0 for r in correct if r['video']==video)
    results={};draws=np.random.default_rng(0).multinomial(25,np.ones(25)/25,size=2000)
    values={m:official_video_values(root/m/'results/pedestrian_detailed.csv') for m in names}
    for method in names:
        tag=TAGS.get(method,method);r=reports[method]
        row=dict(**metrics[method],Present_HOTA=groups[method]['query_present']['HOTA'],
            Present_FP=r['query_present_FP'],Empty_FP=r['query_empty_FP'],AUROC=ranking[method]['all425_official']['AUROC'],
            R1_all=ranking[method]['GT_recall_at_K']['all']['1'],
            R1_outside_explicit_window=ranking[method]['GT_recall_at_K']['outside_explicit_window']['1'],
            R1_excluding_only_actual_teacher_endpoints=ranking[method]['GT_recall_at_K']['non_teacher_endpoint']['1'],
            fixed_PSR_correct_POS374_selected=int(selected[method]),video_CI95={})
        for baseline in ('FROZEN','NAIVE_PAIRED'):
            if values[method][0]!=values[baseline][0]:raise ValueError('video cluster alignment differs')
            boot=paired_clusters(values[method][1],values[baseline][1],draws)
            row['video_CI95'][baseline]=np.percentile(boot,[2.5,97.5]).tolist()
        if method in methods:
            row.update(Frozen_at_method_count=official_metrics(root/('FROZEN_AT_'+method+'_COUNT')/'results')['HOTA'],
                method_at_Frozen_count=official_metrics(root/(method+'_AT_FROZEN_COUNT')/'results')['HOTA'])
        elif method=='NAIVE_PAIRED':
            row.update(Frozen_at_method_count=official_metrics(OLD/'FROZEN_AT_NAIVE_PAIRED_COUNT/results')['HOTA'],
                method_at_Frozen_count=official_metrics(OLD/'NAIVE_PAIRED_AT_FROZEN_COUNT/results')['HOTA'])
        else:row.update(Frozen_at_method_count=metrics[method]['HOTA'],method_at_Frozen_count=metrics[method]['HOTA'])
        results[tag]=row
    fit=json.loads((PUBLIC/'fit_summary.json').read_text());verdict,next_step=classify(fit,results)
    columns=['FROZEN','NAIVE_PAIRED',*TAGS.values()]
    fields=[('HOTA','HOTA'),('DetA','DetA'),('AssA','AssA'),('IDF1','IDF1'),('IDSW','IDSW'),('预测行数','Dets'),
        ('Present HOTA','Present_HOTA'),('Present FP（离线行匹配）','Present_FP'),('Empty FP','Empty_FP'),
        ('同帧 GT R@1 全部GT帧','R1_all'),('R@1 无教师标签帧（旧outside-window口径）','R1_outside_explicit_window'),
        ('R@1 只排除实际教师端点','R1_excluding_only_actual_teacher_endpoints'),('全候选 AUROC','AUROC'),
        ('Frozen@方法计数 HOTA','Frozen_at_method_count'),('方法@Frozen计数 HOTA','method_at_Frozen_count'),
        ('固定 PSR correct-POS374 选中数','fixed_PSR_correct_POS374_selected')]
    table=['# 表 B — 固定final-step完整25Val/425表达','',
        'Frozen与D0旧Naive直接复用完整预测和官方CSV，未重跑；不可训练臂记N/A，不用smoke或source-best冒充final8973。','',
        '| 指标 | Frozen | D0 Naive(s17) | D1_OPT | D2_OPT_POS | D3_CAP |','|---|---:|---:|---:|---:|---:|']
    for label,key in fields:
        entries=[]
        for name in columns:
            v=results.get(name,{}).get(key);entries.append('N/A' if v is None else str(v) if isinstance(v,int) else f'{v:.6f}')
        table.append('| '+label+' | '+' | '.join(entries)+' |')
    for label,baseline in [('视频95% CI: 方法−Frozen','FROZEN'),('视频95% CI: 方法−D0','NAIVE_PAIRED')]:
        entries=['N/A' if name not in results else '['+', '.join(f'{v:+.4f}' for v in results[name]['video_CI95'][baseline])+']' for name in columns]
        table.append('| '+label+' | '+' | '.join(entries)+' |')
    table+=['','outside-window沿旧口径排除anchor±1三帧；本轮仅端点有直接标签，因此另报只排除实际端点的R@1，不混淆分母。',
        'Present FP沿旧IoU≥0.5离线匹配（Frozen1996），不同于官方CLEAR的2005；官方CLEAR随完整CSV保留。',
        '374是历史PSR教师正确POS固定集合，并非三新臂实际使用的Naive363正确POS；全部GT只由此封存后独立进程读取。',
        'Count-matched是继承父模型成本的后验诊断，不是独立部署系统；25视频聚类2000次/seed0，不以帧作为独立样本。']
    (PUBLIC/'VAL_TABLE.md').write_text('\n'.join(table)+'\n')
    atomic_json(PUBLIC/'val_summary.json',dict(status='COMPLETE_SEALED_OFFLINE_DIAGNOSIS',methods=results,
        untrainable=sealed['untrainable_methods'],rule_verdict=verdict,next_step_recommendation_only=next_step,
        complete_predictions_manifest=str(RUN/'predictions_seal.json'),GT_evaluation_independent_and_after_seal=True,
        not_Train_to_Val_or_blind_test=True))
    report=(PUBLIC/'REPORT.md').read_text();history=execution_history(report);prefix=report.split('## 2. 表A：')[0]
    body=prefix+'## 2. 表A：拟合教师\n\n'+(PUBLIC/'FIT_TABLE.md').read_text()+'\n## 3. 表B：封存后Val结果\n\n'+(PUBLIC/'VAL_TABLE.md').read_text()
    body+='\n## 4. 按预注册规则判定\n\n'+verdict+'。\n\n'+next_step+'\n\n教师相邻端点一致率：\n\n```json\n'+json.dumps(fit['teacher_endpoint_consistency'],indent=2,ensure_ascii=False)+'\n```\n'
    body+='\n## 5. 成本、checkpoint与GT隔离\n\n实际运行、所有尝试/失败耗时与checkpoint路径SHA见server_artifacts.json、costs.json和各provenance。\n三新臂训练与拟合进程目标GT打开必须全部为0，任何不可训练结果仍如实呈现；不拿工程测试、loss下降或分支标签当因果证明。\n完整预测/大checkpoint不进入git，已封存路径和SHA在prediction_manifest.json。\n'
    body+='\n以下保留历史阶段记录，其中的运行状态指当时，不代表最终状态；最终结果以表A/B为准。\n\n### 兼容性执行记录\n\n'+history+'\n'
    body=body.replace('状态：IMPLEMENTATION；','状态：COMPLETE_STOP；').replace('状态：SMOKE_COMPLETE_BASELINE_AND_FULL_TRAINING_NEXT；','状态：COMPLETE_STOP；')
    body=body.replace('状态：FULL_TRAINING_RUNNING；','状态：COMPLETE_STOP；')
    body=body.replace('状态：FIT_COMPLETE_VAL_PENDING；','状态：COMPLETE_STOP；')
    (PUBLIC/'REPORT.md').write_text(body)
    shutil.copy2(RUN/'predictions_seal.json',PUBLIC/'prediction_manifest.json');package()
    print(json.dumps(dict(event='D_FIT_OFFLINE_REPORT_COMPLETE',verdict=verdict),ensure_ascii=False),flush=True)


if __name__=='__main__':analyze()
