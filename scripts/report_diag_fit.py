#!/usr/bin/env python3
"""Publishable teacher-only table A. Does not read target GT or run inference."""
import gzip
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.diag_fit import METHODS,POS_THRESHOLD

RUN=ROOT/'results/runs/diag_fit_20261030_v1';PUBLIC=ROOT/'results/diag_fit'
PREVIOUS=ROOT/'results/runs/psr_val_20261029_v1/students'
TAGS=dict(zip(METHODS,('D1_OPT','D2_OPT_POS','D3_CAP')))


def gradient_summary(folder):
    config=json.loads((folder/'config.json').read_text())
    rows=[json.loads(s) for s in (folder/'metrics_train.jsonl').read_text().splitlines()]
    values=[r.get('gradient_norm_preclip',r['gradient_norm']) for r in rows]
    return dict(preclip_median=float(np.median(values)),preclip_p90=float(np.percentile(values,90)),
        clipped_step_fraction=sum(v>config['optimizer']['grad_clip'] for v in values)/len(values),
        actual_trace_steps=len(values))


def endpoint_consistency():
    from flex_uda.teacher.psr_dataset import PSRPseudoDataset
    from collections import defaultdict
    data=PSRPseudoDataset('/data2/user/TTA-RMOT-assets/psr/val_v1/NAIVE',allow_val_diagnostic=True,expanded_kind='none')
    pairs=defaultdict(list)
    for r in data.rows:pairs[r['video_id'],r['query_id'],r['track_id'],r['anchor_frame']].append(r)
    rows=[]
    for group in pairs.values():
        group=sorted(group,key=lambda r:r['frame_id'])
        if len(group)!=2 or [r['frame_id']-r['anchor_frame'] for r in group]!=[-1,1]:raise ValueError('teacher endpoint pairing differs')
        if all(r['valid'] for r in group):
            a,b=group;rows.append(dict(binary_POS_agreement=(a['teacher_prob']>=POS_THRESHOLD)==(b['teacher_prob']>=POS_THRESHOLD),
                YesNo_half_agreement=(a['teacher_prob']>=.5)==(b['teacher_prob']>=.5),
                original_three_state_agreement=a['state']==b['state'],abs_probability_difference=abs(a['teacher_prob']-b['teacher_prob'])))
    return dict(pairs=len(rows),endpoint_offset=1,
        **{k:float(np.mean([r[k] for r in rows])) for k in rows[0]},
        primary_POS_threshold=POS_THRESHOLD,target_GT_used=False,used_for_training_or_tuning=False)


def package():
    import tarfile
    PUBLIC.mkdir(parents=True,exist_ok=True);files=[];weights=[]
    for pattern in ('validation/*_stdout.txt','validation/*process*.json','validation/*/receipt.json',
                    'validation/*/training_receipt.json','validation/*/provenance.json','validation/*/metrics_train.jsonl',
                    'logs/*/*.txt','launchers/*/*/*.json',
                    'validation/completion.json','smoke_completion.json','full_completion.json'):
        files.extend(p for p in RUN.glob(pattern) if p.is_file())
    for phase in ('smoke','students'):
        for method in METHODS:
            folder=RUN/phase/method
            if not folder.exists():continue
            target=PUBLIC/phase/method;target.mkdir(parents=True,exist_ok=True)
            for name in ('training_receipt.json','provenance.json','config.json','pos_weight.json','fit_subset.json',
                'teacher_fit_curve.jsonl','teacher_fit_summary.json','source_validation.jsonl','final_step_selection.json'):
                if (folder/name).exists():shutil.copy2(folder/name,target/name)
            for name in ('metrics_train.jsonl',):
                if (folder/name).exists():
                    with (folder/name).open('rb') as src,gzip.open(target/(name+'.gz'),'wb') as dst:shutil.copyfileobj(src,dst)
            for p in folder.glob('checkpoint_*.pt'):weights.append(dict(phase=phase,method=method,path=str(p.resolve()),sha256=file_sha(p)))
    for p in RUN.glob('attempts/**/provenance.json'):
        target=PUBLIC/'attempts'/p.relative_to(RUN/'attempts');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for p in RUN.glob('attempts/**/attempt_runtimes/*.json'):
        target=PUBLIC/'attempts'/p.relative_to(RUN/'attempts');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for base in ('attempts','retained_code_failures'):
        for p in (RUN/base).glob('**/*'):
            if not p.is_file() or p.name not in ('config.json','provenance.json','pos_weight.json',
                'teacher_fit_curve.jsonl','source_validation.jsonl','numerical_retries.jsonl','metrics_train.jsonl'):continue
            if base=='attempts' and (p.parent/'training_receipt.json').exists() and p.name!='numerical_retries.jsonl':continue
            target=PUBLIC/base/p.relative_to(RUN/base);target.parent.mkdir(parents=True,exist_ok=True)
            if p.name=='metrics_train.jsonl':
                with p.open('rb') as src,gzip.open(str(target)+'.gz','wb') as dst:shutil.copyfileobj(src,dst)
            else:shutil.copy2(p,target)
    for p in RUN.glob('baselines/**/*.json'):
        target=PUBLIC/p.relative_to(RUN);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for method in METHODS:
        for scope in ('smoke','students'):
            p=RUN/scope/(method+'_status.json')
            if p.exists():
                target=PUBLIC/scope/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    with tarfile.open(PUBLIC/'runtime_logs.tar.gz','w:gz') as stream:
        for p in sorted(set(files)):stream.add(p,arcname=str(p.relative_to(RUN)),recursive=False)
    atomic_json(PUBLIC/'server_artifacts.json',dict(checkpoints=weights,run_root=str(RUN),
        complete_predictions_stay_on_server=True,raw_teacher_cache_stays_on_server=True,
        target_GT_never_read_by_packager=True))
    runtimes=[]
    for base in ('attempts','retained_code_failures'):
        for p in (RUN/base).glob('**/attempt_runtimes/*.json'):
            r=json.loads(p.read_text());runtimes.append(dict(r,path=str(p),revoked_implementation_attempt=base=='retained_code_failures'))
    inference=[]
    for p in (RUN/'evaluation/merged').glob('DIAG_FIT_*/run_metadata.json'):
        r=json.loads(p.read_text());inference.append(dict(method=r['method'],GPU_hours=r['worker_seconds']/3600,
            actual_wall_seconds=r['actual_scheduler_wall_seconds'],peak_allocated_bytes=r['peak_allocated_bytes']))
    atomic_json(PUBLIC/'costs.json',dict(training_attempts=runtimes,
        training_GPU_process_hours_including_retained_attempts=sum(r['elapsed_seconds'] for r in runtimes)/3600,
        inference=inference,teacher_new_calls=0,required_teacher_labels_per_arm=8500,
        reused_Frozen_and_D0_are_not_free=True,
        validation_process_times={p.stem:json.loads(p.read_text()) for p in (RUN/'validation').glob('*process*.json')},
        limits='training entries are actual worker-body elapsed, not exclusive GPU utilization; baseline probe and extra standalone parity calls without an elapsed receipt are not silently counted as zero',
        baseline_fit_cost=json.loads((RUN/'baselines/completion.json').read_text()).get('GPU_process_seconds') if (RUN/'baselines/completion.json').exists() else None,
        unmeasured_validation_costs=['standalone full-capacity zero-output parity with no separate elapsed receipt']))


def report():
    complete=json.loads((RUN/'full_completion.json').read_text())
    baselines=json.loads((RUN/'baselines/completion.json').read_text())
    answer=baselines['methods']
    for name,method in [('Naive(s17)','NAIVE_PAIRED'),('PSR(s17)','PSR')]:
        answer[name]['gradients']=gradient_summary(PREVIOUS/method)
    for method,tag in TAGS.items():
        status=complete['arms'][method]
        if status['status']!='COMPLETE':answer[tag]=dict(status=status['status']);continue
        folder=RUN/'students'/method
        curve=[json.loads(s) for s in (folder/'teacher_fit_curve.jsonl').read_text().splitlines()]
        summary=json.loads((folder/'teacher_fit_summary.json').read_text())
        receipt=json.loads((folder/'training_receipt.json').read_text())
        if receipt['steps']!=8973 or receipt['unique_teacher_samples_seen']!=8500 or receipt['target_GT_open_attempts']:raise ValueError('incomplete or invalid D-FIT training')
        if receipt['fit_subset_sha256']!=baselines['fit_subset_sha256']:raise ValueError('baseline/arm subset changed')
        weight=json.loads((folder/'pos_weight.json').read_text())
        answer[tag]=dict(status='COMPLETE',all=summary,subset=curve[-1],trainable_parameters=receipt['trainable_parameters'],
            pos_weight=weight['pos_weight'],source_val_CE=curve[-1]['source_val_CE'],source_val_CE_step0=curve[0]['source_val_CE'],
            gradients=gradient_summary(folder),fallback_lr=status['fallback_lr'],
            fit_qualified=summary['teacher_POS_selected_rate']>=.80 and summary['teacher_NEG_selected_rate']<=.10,
            selection=json.loads((folder/'final_step_selection.json').read_text()))
    fields=[('可训练参数量','trainable_parameters'),('实际 pos_weight（未加权=1）','pos_weight'),
        ('teacher_POS_selected_rate','teacher_POS_selected_rate'),('teacher_NEG_selected_rate','teacher_NEG_selected_rate'),
        ('mean_student_prob_on_POS','mean_student_prob_on_POS'),('mean_student_prob_on_NEG','mean_student_prob_on_NEG'),
        ('pair_agreement','pair_agreement'),('结束 source_val_CE','source_val_CE'),('step0 source_val_CE','source_val_CE_step0'),
        ('裁剪前总梯度范数中位数','preclip_median'),('裁剪前总梯度范数 p90','preclip_p90'),('被裁剪步数比例','clipped_step_fraction')]
    names=['Frozen','Naive(s17)','PSR(s17)',*TAGS.values()]
    lines=['# 表 A — D-FIT 教师拟合（不使用目标 GT）','',
        '全部主拟合列使用相同Naive8500判断、POS≥0.95，NEG为有效p<0.95；300窗口子集的SHA相同。',
        '旧PSR在自身8500标签的历史拟合另列summary，不混入这个共同标签主表。',
        '这是训练锚点拟合，不是GT精度或完整RMOT；新臂固定最后8973步，不选择source-best作目标评价。','',
        '| 指标 | '+' | '.join(names)+' |','|---|'+'---:|'*len(names)]
    for label,key in fields:
        values=[]
        for name in names:
            r=answer[name]
            value=r.get(key,r.get('all',{}).get(key,r.get('gradients',{}).get(key)))
            if key=='pos_weight' and 'all' in r:value=1. if value is None else value
            values.append('N/A' if value is None else str(value) if isinstance(value,int) else f'{value:.6f}')
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    lines+=['','新臂状态与source-selected记录（只记录，不对其source-best做Val评测）：']
    for tag in TAGS.values():lines.append(f"- {tag}: {answer[tag].get('status')}; fallback={answer[tag].get('fallback_lr')}; selection={answer[tag].get('selection')}")
    (PUBLIC/'FIT_TABLE.md').write_text('\n'.join(lines)+'\n')
    consistency=endpoint_consistency()
    atomic_json(PUBLIC/'fit_summary.json',dict(status='COMPLETE_ALL_ARMS_REPORTED_TEACHER_ONLY',methods=answer,
        fit_subset_sha256=baselines['fit_subset_sha256'],teacher_endpoint_consistency=consistency,
        target_GT_open_attempts=0,full_RMOT_inference_started=False,all_arms_reported=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharex=True)
    for method,tag in TAGS.items():
        path=RUN/'students'/method/'teacher_fit_curve.jsonl'
        if not path.exists():continue
        curve=[json.loads(s) for s in path.read_text().splitlines()]
        for ax,key in zip(axes,('teacher_POS_selected_rate','teacher_NEG_selected_rate')):
            ax.plot([r['step'] for r in curve],[r[key] for r in curve],marker='.',label=tag)
    for ax,title,threshold in zip(axes,('Teacher POS selected rate','Teacher NEG selected rate'),(.8,.1)):
        ax.axhline(threshold,color='grey',linestyle='--',label='preregistered fit criterion');ax.set(title=title,xlabel='Optimizer step',ylim=(0,1));ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Fixed 300-window teacher-only fit subset; not independent validation')
    fig.tight_layout();fig.savefig(PUBLIC/'teacher_fit_curves.png',dpi=160);plt.close(fig)
    package()
    print(json.dumps(dict(event='D_FIT_TABLE_A_READY_NOT_YET_PUBLISHED',fit_qualified={k:answer[k].get('fit_qualified') for k in TAGS.values()},table_SHA256=file_sha(PUBLIC/'FIT_TABLE.md'))),flush=True)


if __name__=='__main__':report()
