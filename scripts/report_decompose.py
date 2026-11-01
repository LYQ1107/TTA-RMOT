"""Final all-results diagnostic report; no experimental decisions are fitted."""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import sys
import tarfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decompose_common import *
from scripts.report_multidomain_generality import table
from flex_uda.data.manifests import read_jsonl


def read_csv(name):
    with (PUBLIC/name).open() as h:return list(csv.DictReader(h))


def teacher_verdicts(rows,field):
    verdicts=[]
    for target in TARGETS:
        metrics={r['scorer']:float(r[field]) for r in rows if r['target']==target and r['expression_type']=='ALL'}
        inside=metrics[previous.IN_DOMAIN_MODEL[target]]
        sources=[s for _,s,t in PAIRS if t==target]
        for method in ('CROP','CLIP','FRAME_BOX'):
            value=metrics[method]
            if value>=inside-.02 and all(value>=metrics[s]+.05 for s in sources):label='可以修排序';level=2
            elif all(value>metrics[s] for s in sources):label='部分修排序';level=1
            else:label='不能修排序';level=0
            verdicts.append(dict(target=target,input=method,metric=field,score=value,in_domain=inside,
                                 cross_domain_sources=json.dumps({s:metrics[s] for s in sources}),verdict=label,level=level))
    return verdicts


def collect_costs():
    sealed=json.loads((RUN/'probe/seal.json').read_text());plan=json.loads((RUN/'probe/plan.json').read_text())
    records=[];attempts=[];admissions=[]
    for p in sorted((RUN/'probe/launchers').glob('*/worker_*.json')):
        records.extend(json.loads(p.read_text()))
    for p in sorted((RUN/'probe/launchers').glob('*/admission.json')):
        admissions.append(dict(path=str(p),sha256=file_sha(p),record=json.loads(p.read_text())))
    registry_path=RUN/'probe/resume_registry.json'
    registry=json.loads(registry_path.read_text()) if registry_path.exists() else {}
    overrides=registry.get('receipt_overrides',{})
    totals={m:Counter() for m in ('CROP','CLIP','FRAME_BOX')}
    peak=Counter();original=[]
    unassigned_forward_upper_bound=0.;discarded_call_upper_bound=0
    for worker in sealed['workers']:
        if worker['GT_open_attempts'] or not worker['frozen']['frozen_parameters_unchanged'] or file_sha(worker['outputs_path'])!=worker['outputs_sha256']:
            raise ValueError('Invalid final teacher seal')
        original.append(dict(path=worker['outputs_path'],sha256=worker['outputs_sha256'],pairs=worker['output_pairs'],
                             receipt_path=str(RUN/'probe/workers'/str(worker['shard'])/'receipt.json'),
                             receipt_sha256=file_sha(RUN/'probe/workers'/str(worker['shard'])/'receipt.json')))
    for record in records:
        suffix='smoke_receipt.json' if record['tag'].endswith('smoke') else 'receipt.json'
        shard=record['tag'][5:record['tag'].index('_')]
        p=Path(overrides[record['log']]) if record['log'] in overrides else RUN/'probe/workers'/shard/suffix
        r=json.loads(p.read_text());attempts.append(dict(**record,receipt_path=str(p),receipt_sha256=file_sha(p)))
        unassigned_forward_upper_bound+=r.get('unknown_inflight_forward_seconds_upper_bound',0.)
        discarded_call_upper_bound+=r.get('possible_discarded_inflight_judgments_upper_bound',0)
        for method,b in r['budgets'].items():
            for field in ('new_logical_calls','forward_batches','forward_seconds'):totals[method][field]+=b.get(field,0)
            peak[method]=max(peak[method],b.get('peak_allocated_bytes',0))
    archived=json.loads((COUNT/'probe/2B/receipt.json').read_text());oldrows=read_jsonl(archived['outputs_path'])
    old_batches={r['batch_id']:r['batch_forward_seconds'] for r in oldrows if r['method']=='T1_CROP_YESNO'}
    methods=[]
    for method in totals:
        b=dict(totals[method]);b.update(input=method,planned_paired_judgments=29146,
            peak_allocated_bytes=max(peak[method],archived['budgets']['T1_CROP_YESNO']['peak_allocated_bytes'] if method=='CROP' else 0),
            original_Dance_CROP_reused_calls=6426 if method=='CROP' else 0,
            inherited_original_CROP_forward_seconds=sum(old_batches.values()) if method=='CROP' else 0.,
            CLIP_fallbacks=sum(w['budgets'][method].get('fallback_to_CROP',0) for w in sealed['workers']) if method=='CLIP' else 0,
            effective_forward_GPU_hours_new_lower_bound=b.get('forward_seconds',0)/3600,
            forward_seconds_exact=not bool(overrides),
            forward_seconds_lower_bound=b.get('forward_seconds',0),
            forward_seconds_upper_bound=b.get('forward_seconds',0)+unassigned_forward_upper_bound,
            upper_bound_is_shared_not_additive_across_views=True,
            peak_memory_scope='observed committed/successful batches; allocation budget bounds any discarded batch')
        methods.append(b)
    for b in methods:
        expected=29146-b['original_Dance_CROP_reused_calls']-b['CLIP_fallbacks']
        if b.get('new_logical_calls',0)!=expected:
            raise ValueError('Successful teacher-call conservation failed: '+b['input'])
    cpu=[]
    for p in (RUN/'offline_launchers').glob('*/completion.json'):cpu.append(dict(path=str(p),sha256=file_sha(p),**json.loads(p.read_text())))
    result=dict(status='ALL_ACTUAL_TEACHER_COSTS_ACCOUNTED',methods=methods,attempts=attempts,admissions=admissions,
        max_physical_GPUs=2,actual_teacher_scheduler_wall_seconds=sealed['wall_seconds'],
        new_GPU_process_hours=sum(r['worker_seconds'] for r in records)/3600,
        new_teacher_forward_GPU_hours_lower_bound=sum(r['forward_seconds'] for r in totals.values())/3600,
        new_teacher_forward_GPU_hours_upper_bound=(sum(r['forward_seconds'] for r in totals.values())+unassigned_forward_upper_bound)/3600,
        exact_forward_seconds_available=not bool(overrides),
        successful_teacher_call_conservation_pass=True,
        shared_unassigned_inflight_forward_seconds_upper_bound=unassigned_forward_upper_bound,
        possible_discarded_inflight_judgments_upper_bound=discarded_call_upper_bound,
        observed_new_judgments_lower_bound=sum(r['new_logical_calls'] for r in totals.values()),
        observed_new_judgments_upper_bound=sum(r['new_logical_calls'] for r in totals.values())+discarded_call_upper_bound,
        resource_resume_registry=registry if registry else None,
        original_Dance_CROP_inherited_forward_seconds=sum(old_batches.values()),original_Dance_CROP_inherited_calls=6426,
        source_and_in_domain_margin_inference_costs_inherited_not_zero=str(OLD_PUBLIC/'costs.json'),
        estimates_registered_before_calls=plan['final_budget'],
        cost_definitions='GPU-process hours include load/hash/CPU/IO/shared-GPU waiting; forward seconds synchronize CUDA; neither is device utilization integral',
        raw_outputs=original,cpu_offline_launchers=cpu,student_updates=0,teacher_updates=0)
    atomic_json(PUBLIC/'teacher_costs.json',result)
    return result


def main():
    require_plan();plan=json.loads((RUN/'probe/plan.json').read_text())
    b_status=json.loads((RUN/'b_report.json').read_text())
    if set(b_status['combos'])!={c for c,_,_ in PAIRS}:raise ValueError('Report every fixed pair')
    if json.loads((RUN/'probe/evaluation.json').read_text())['status']!='ALL_FIXED_PAIRED_PROBES_OFFLINE_EVALUATED':
        raise ValueError('Teacher metrics not complete')
    rows=read_csv('b_all.csv');teachers=read_csv('teacher_probe.csv');g=read_csv('granularity_verdicts.csv')
    cal=read_csv('calibrators.csv');rho=read_csv('rho_free.csv');splits=read_csv('b_splits.csv')
    v_gt=teacher_verdicts(teachers,'GT_R_at_1');v_hit=teacher_verdicts(teachers,'Hit_at_1')
    description=[]
    frames=read_jsonl(RUN/'probe/offline_frames.jsonl')
    for target in TARGETS:
        for method in ('CROP','CLIP','FRAME_BOX',*MODELS):
            fs=[r for r in frames if r['target']==target and r['method']==method]
            positive=[r for r in fs if r['GT_count']>0];empty=[r for r in fs if r['GT_count']==0]
            description.append(dict(target=target,scorer=method,frame_queries=len(fs),GT_present=len(positive),GT_empty=len(empty),
                presence_recall=sum(r['predicted_count']>0 for r in positive)/len(positive),
                presence_false_positive_rate=sum(r['predicted_count']>0 for r in empty)/len(empty),
                constant_absence_accuracy=len(empty)/len(fs),constant_presence_accuracy=len(positive)/len(fs),
                constant_zero_count_MAE=sum(r['GT_count'] for r in fs)/len(fs),
                role='POST_HOC_CLASS_BALANCE_DESCRIPTION_ONLY_NO_NEW_PREDICTIONS_OR_VERDICT'))
    write_csv(PUBLIC/'presence_description.csv',description)
    write_csv(PUBLIC/'teacher_verdicts.csv',v_gt+v_hit)
    b_boot=json.loads((PUBLIC/'b_bootstrap.json').read_text());a_boot=json.loads((PUBLIC/'teacher_bootstrap.json').read_text())
    atomic_json(PUBLIC/'bootstrap.json',dict(part_A=a_boot,part_B=b_boot,clusters='whole videos; never independent frames/queries',samples=2000,seed=0))
    cost=collect_costs();index={(r['combo'],r['configuration']):r for r in rows}
    final_seals=[]
    for combo,model,target in PAIRS:
        for name in (*ORACLES,*CALIBRATORS):
            s=verify(combo,name)
            if s['episode_count']!={'KITTI-V2':862,'SportsMOT':466,'Dance':425}[target]:raise ValueError('Incomplete target episodes')
            final_seals.append(dict(combo=combo,method=name,path=str(folder(combo,name)/'predictions_seal.json'),
                                    sha256=file_sha(folder(combo,name)/'predictions_seal.json'),rows=s['prediction_rows']))
    summaries=[]
    for field,verdicts in [('GT_R_at_1',v_gt),('Hit_at_1',v_hit)]:
        for target in TARGETS:
            choices=[r for r in verdicts if r['target']==target];best=max(r['level'] for r in choices)
            summaries.append(dict(target=target,metric=field,best_input_modes=[r['input'] for r in choices if r['level']==best],
                                  best_verdict=next(r['verdict'] for r in choices if r['level']==best)))
    validation=dict(status='ALL_PREREGISTERED_A_AND_B_DIAGNOSTICS_COMPLETE',prereg_sha256=file_sha(PUBLIC/'PREREG.md'),
        expression_taxonomy_sha256=file_sha(PUBLIC/'expression_types.csv'),margins_verified=plan['margin_banks'],
        paired_teacher_entries=87438,GT_free_decoder_attempts=0,teacher_GT_open_attempts=0,
        new_prediction_configurations=72,all_prediction_seals=final_seals,
        mandatory_G4_checks=[json.loads((RUN/'oracle'/f'{c}.json').read_text())['self_checks'] for c,_,_ in PAIRS],
        teacher_best_verdicts=summaries,all_variants_reported=True,student_updates=0,teacher_updates=0,
        no_new_methods_queued=True,GT_feedback=False,strictly_diagnostic_not_TTA_training=True)
    audit_records=[]
    covered=set()
    for p in (RUN/'calibrator_audit').glob('*.json'):
        audit=json.loads(p.read_text())
        if audit['GT_open_attempts'] or not audit['isolation']['kernel_read_whitelist']:
            raise ValueError('GT-free calibrator isolation failed')
        covered.update(r['combo'] for r in audit['combos'])
        audit_records.append(dict(path=str(p),sha256=file_sha(p),GT_open_attempts=0,kernel_whitelist=True))
    if covered!={c for c,_,_ in PAIRS}:raise ValueError('Missing GT-free production access audit')
    validation['calibrator_audits']=audit_records
    validation['expression_types_sealed_before_any_evaluation']=plan['expression_types']
    if file_sha(PUBLIC/'expression_types.csv')!=plan['expression_types']['sha256']:
        raise ValueError('Expression types changed after pre-evaluation seal')
    atomic_json(PUBLIC/'completion.json',validation)
    archive=PUBLIC/'execution_logs.tar.gz'
    with tarfile.open(archive,'w:gz') as h:
        for pattern in ('probe/launchers/*/*.log','offline_launchers/*/*.log','predictions/*/*/trackeval.log'):
            for p in RUN.glob(pattern):h.add(p,arcname=str(p.relative_to(RUN)))
        for p in RUN.glob('probe/workers/*/*receipt.json'):h.add(p,arcname=str(p.relative_to(RUN)))
        for m in json.loads((PUBLIC/'prediction_paths.json').read_text())['configurations']:
            p=Path(m['path'])/'results/pedestrian_summary.txt'
            h.add(p,arcname='official_summaries/'+m['combo']+'/'+m['configuration']+'/pedestrian_summary.txt')
    paths=json.loads((PUBLIC/'prediction_paths.json').read_text())
    atomic_json(PUBLIC/'artifact_paths.json',dict(teacher_outputs=cost['raw_outputs'],predictions=paths,
        logs=dict(path=str(archive),sha256=file_sha(archive)),full_runtime_root=str(RUN),
        all_prediction_files=dict(path=str(PUBLIC/'all_prediction_files.csv.gz'),sha256=file_sha(PUBLIC/'all_prediction_files.csv.gz'))))
    lines=['# 跨域退化分解第二步：冻结教师排序与数量粒度诊断','',
        '状态：A/B计算与本地验收完成；GitHub最终发布状态须另据实际推送回读确认。不训练、不微调、不启动方法实验。源模型及同域参照均为上一轮封存的官方完整候选分数。','',
        '## 1. 预注册判定','',
        '主表沿用 v2 的 GT-R@1：在所有有 GT 的探针帧–表达上，top1 的 IoU≥0.5 命中数除以该帧 GT 数后平均；无候选计0。',
        '本轮文字也可读作二值 Hit@1。为避免口径偷换，两种指标、两套原阈值判定和区间全部保存；下方另列 Hit@1 复核矩阵，不挑更有利口径。','',
        '本轮两种R@1口径给出的全部9格判定一致。','',
        table(['输入方式',*TARGETS],[[m,*[next(r['verdict'] for r in v_gt if r['target']==t and r['input']==m) for t in TARGETS]] for m in ('CROP','CLIP','FRAME_BOX')]),'',
        table(['目标','同等最好方式（全部）','判定'],[[r['target'],', '.join(r['best_input_modes']),r['best_verdict']] for r in summaries if r['metric']=='GT_R_at_1']),'',
        table(['组合','所需数量粒度','源排序×同域数量 HOTA'],[[r['combo'],r['required_granularity'],r['R_SxK_I_HOTA']] for r in g]),'',
        '“需要逐帧数量”是按 G0–G3 均未达参照的固定规则命名，不保证 G4 就一定达标；TRACK_ORACLE 是另一种 GT 轨迹语义上限，不能作为无标签方法。','',
        '二值 Hit@1 复核（不与前轮归一化数字混用）：','',
        table(['输入方式',*TARGETS],[[m,*[next(r['verdict'] for r in v_hit if r['target']==t and r['input']==m) for t in TARGETS]] for m in ('CROP','CLIP','FRAME_BOX')]),'',
        '## 2. Part A：固定配对探针','',
        table(['目标','视频','帧','候选–表达对'],[[t,*[plan['dataset_counts'][t][k] for k in ('videos','frames','candidate_query_pairs')]] for t in TARGETS]),'',
        'KITTI 每视频按原类别分层、RandomState(0)抽60句；Sports排序每隔3视频取1个，全部原句；Dance原5视频50帧完整复用。8锚点总预算估算1.239264 GPU小时，不触发4锚点回退。清单/预算先封存后调用；相同条目比较三教师、两跨域源模型、一个同域模型，共六种打分。','',
        'CROP原320裁剪/0.2 margin/YesNo路径未改；CLIP使用给定历史时刻的前向最近观测，少于2个可用槽退回CROP，不用未来；重复替代帧保留并记录unique_frames。FRAME_BOX长边1024、当前框红色4px、其余框不画。冻结BF16、无量化/adapter、revision `'+REVISION+'`。','',
        'KITTI Δ=3（10fps）、Dance Δ=5（20fps）、Sports各视频25fps、Δ=6。本地KITTI/Dance转换包无seqinfo，fps来自已封存独立配置及官方来源：[KITTI](https://www.cvlibs.net/datasets/kitti/raw_data.php)、[Dance官方视频配置](https://github.com/DanceTrack/DanceTrack/blob/main/tools/txt2video_dance.py#L53)；Sports直接读取原seqinfo。','']
    lines+=['归一化上限【事后描述，不增加判定规则】：多目标时GT-R@1的逐帧上限只有1/g，即使top1总命中也不等于1。下表同时给出完全理想算术上限和现有MOT可匹配候选的上限；它们不进入任何阈值选择、预测或方法。','',
        table(['目标','有目标帧–表达数','平均GT数','GT-R@1算术上限','固定MOT的GT-R@1上限','固定MOT的Hit@1上限'],
            [[r[k] for k in ('target','GT_present_frame_queries','GT_count_present_mean','arithmetic_GT_R_at_1_upper_bound',
              'fixed_MOT_oracle_GT_R_at_1','fixed_MOT_oracle_Hit_at_1')] for r in read_csv('probe_ceiling.csv') if r['expression_type']=='ALL']),'']
    for target in TARGETS:
        lines += ['### '+target,'',table(['打分','GT-R@1','Hit@1','AUROC','存在性准确率','数量MAE'],
            [[r[k] for k in ('scorer','GT_R_at_1','Hit_at_1','AUROC','presence_accuracy','count_MAE')] for r in teachers if r['target']==target and r['expression_type']=='ALL']),'',
            '按文本类型（多标签，不互斥；无样本为未定义而非0）：','',
            table(['打分','文本类型','GT-R@1','Hit@1','AUROC','存在性准确率','数量MAE'],
            [[r[k] for k in ('scorer','expression_type','GT_R_at_1','Hit_at_1','AUROC','presence_accuracy','count_MAE')] for r in teachers if r['target']==target and r['expression_type']!='ALL']),'']
    lines+=['### 存在性类别构成【事后描述，不参与判定】','',
        table(['目标','打分','有GT帧–表达','无GT帧–表达','存在性召回','空帧误报率'],
            [[r[k] for k in ('target','scorer','GT_present','GT_empty','presence_recall','presence_false_positive_rate')] for r in description]),'',
        'Dance探针97/850个帧–表达有GT。恒判断不存在的准确率为753/850=0.885882，恒输出数量0的MAE为331/850=0.389412。它们只是样本类别构成的算术描述，没有生成新增MOT预测，也不参与方法/输入判定；说明整体存在性准确率与MAE必须结合正负构成理解，不能用多数空帧掩盖漏检。','']
    lines += ['### Part A 视频聚类区间','',
        'Sports15视频、Dance5视频，2000次seed0；KITTI只有4视频只报告点估计。全部表达和帧随视频一并重采样。','',
        table(['目标','教师−参照','Δ GT-R@1','95% CI'],[[r['target'],r['first']+' − '+r['second'],r['delta'],r['CI95']]
            for k,r in a_boot.items() if r['metric']=='GT_R_at_1' and not k.startswith('SportsMOT418')]),'',
        '二值Hit@1与Sports418的全部对应区间见 `bootstrap.json`，不是按数万个候选独立抽样。','',
        '### 教师成本','',table(['输入','逻辑配对','已提交新增判断','已记录forward秒（下界）','已观测峰值GiB','CLIP回退'],
            [[r['input'],r['planned_paired_judgments'],r.get('new_logical_calls',0),r.get('forward_seconds',0),r['peak_allocated_bytes']/2**30,r['CLIP_fallbacks']] for r in cost['methods']]),'',
        f"本轮新增GPU-process小时 {cost['new_GPU_process_hours']:.6f}；调度wall {cost['actual_teacher_scheduler_wall_seconds']:.3f}秒；包含smoke、模型加载/哈希、CPU/IO和共享卡等待。不是设备利用率积分。",'',
        '资源迁移：GPU2分片完成后，GPU7从原准入0进程变为3进程，GPU2仍空闲。只对经/proc精确核验的本任务shard1发送一次SIGINT，保留全部SQLite已提交调用，重新审计后在GPU2续跑；未信号其他进程、未重跑shard0、未改输入/权重/批大小/阈值。缓存逐请求键与模型revision核对通过。',
        f"完整GPU-process耗时包含中断尝试并精确计入。原GPU7最后至多1个未提交批次（≤{cost['possible_discarded_inflight_judgments_upper_bound']}判断）可能被重算；其forward秒不能伪造，表中为已提交/成功批次计时下界，完整保守区间见teacher_costs.json。共享未分配时间上界不可以重复加给三个输入。",'',
        f"Dance CROP继承6426判断，原始可追溯forward合计 {cost['original_Dance_CROP_inherited_forward_seconds']:.6f} 秒，单列不冒充免费独立系统；复用的官方分数推理成本同样是继承成本。",'',
        '三种输入在联合探针执行中共享模型加载，逐输入表是forward计时而非独立系统端到端成本；实测还包含不同时段的共享GPU负载，不是独占设备微基准。','',
        '## 3. Part B：全部六组合、全部配置','',
        'G0–G4及TRACK_ORACLE都使用目标GT，仅离线诊断。G4六组合逐字节等于上一轮 R_S×K_ORACLE；C1四个既有Oracle和TRACK_MEAN逐字节复现后才复用。G0/G1预算超候选数按v2封顶；排序/tie与原v2一致。',
        'G1_QUERY的q是（视频,表达），并非所有视频共用的一句话：GT总预算同时给出了该视频是否存在目标及全视频总数量，然后只用源分数在时间和候选间排序。它不是可部署的无标签存在性估计。所有GT-free校准都是封存分数上的离线诊断，部分使用整视频/整目标域统计，不冒充因果在线TTA。','']
    for combo,model,target in PAIRS:
        chosen=[r for r in rows if r['combo']==combo]
        lines += ['### '+combo+' '+model+' → '+target,'',
            table(['配置','HOTA','DetA','AssA','IDF1','IDSW','预测行','Empty行'],
                [[r[k] for k in ('configuration','HOTA','DetA','AssA','IDF1','IDSW','prediction_rows','query_empty_prediction_rows')] for r in chosen]),'',
            'GT-free逐个可恢复比例（不挑最优）：','',
            table(['校准','ρ_free'],[[r['calibrator'],r['rho_free']] for r in rho if r['combo']==combo]),'',
            '按表达类型的HOTA：','',
            table(['配置',*WORDS,'OTHER'],[[n,*[next(r['HOTA'] for r in splits if r['combo']==combo and r['configuration']==n and r['split_kind']=='expression_type' and r['split']==t) for t in (*WORDS,'OTHER')]] for n in (*ORACLES,*CALIBRATORS,'NATIVE','IN_DOMAIN','R_SxK_I')]),'']
        if target=='KITTI-V2':
            lines += ['KITTI类别拆分（4视频，仅点估计）：','',
                table(['配置','类别','HOTA','DetA','AssA','IDF1','IDSW','预测行'],[[r[k] for k in ('configuration','split','HOTA','DetA','AssA','IDF1','IDSW','prediction_rows')]
                    for r in splits if r['combo']==combo and r['split_kind']=='target_category']),'']
    lines += ['### 源统计与历史差异','',
        '每个源模型只用自身同域完整评测分数统计 r、μ、z0；来自同域评测集而非源训练集，未用目标GT。详见 `source_statistics.json`。C1旧v2采用2355528历史行，而本轮明确要求2425382完整行，不满足直接复用条件；故独立重算，不改旧结果。旧B1_GLOBAL 13.848，本轮13.846；两种源统计不能混为同一协议。','',
        '### Part B 视频聚类区间','',
        table(['组合','配置−Native','ΔHOTA','95% CI'],[[r['combo'],r['first'],r['delta_HOTA'],r['CI95']] for k,r in b_boot.items() if '__418__' not in k]),'',
        'KITTI只有4视频，无帧级伪显著性。Sports45、Dance25视频，官方TP加权HOTA充分统计聚合；重采样2000/seed0。全部充分统计另存 `b_bootstrap_stats.json.gz`。','',
        '## 4. SportsMOT 官方覆盖418条的敏感性','',
        '主结果仍为授权v4全部466条。本节将同一v4 GT限制到旧官方模板实际覆盖的418个序列ID，仅改变覆盖范围，不混入8个已知缺行的旧GT。预测/教师调用完全复用，不删主结果、不增加教师预算。原418模板和v3失败记录均未改。','',
        table(['组合','配置','HOTA','DetA','AssA','IDF1','IDSW','预测行','Empty行'],[[r[k] for k in ('combo','configuration','HOTA','DetA','AssA','IDF1','IDSW','prediction_rows','query_empty_prediction_rows')] for r in read_csv('sports418_b.csv')]),'',
        '418覆盖下的全部文本类型拆分（所有配置/指标）另见 `sports418_b_splits.csv`；全部对应bootstrap见 `bootstrap.json`。','',
        '同一探针限定官方覆盖表达后的教师指标：','',
        table(['打分','类型','GT-R@1','Hit@1','AUROC','存在性准确率','数量MAE'],[[r[k] for k in ('scorer','expression_type','GT_R_at_1','Hit_at_1','AUROC','presence_accuracy','count_MAE')] for r in read_csv('sports418_teacher.csv')]),'',
        '## 5. 科学解释与适用边界','',
        'KITTI：三种输入均只达到“部分修排序”。CLIP的GT-R@1为0.396039，高于CROP的0.362350，但仍低于同域V2的0.470664。按既定多标签分组，CLIP对MOTION为0.473200、对SPATIAL为0.291032，CROP分别0.440808/0.224584；只报告点值，不增加类型判定。',
        'Sports：FRAME_BOX达到固定“可以修排序”门槛，0.220316接近同域0.229128；相对V2的增益0.070853，15视频CI为[0.038971,0.102829]；相对Dance源为0.109394，CI[0.060212,0.155279]。空间与角色表达的差异尤其明显：FRAME_BOX为0.315263/0.298354，CROP为0.145788/0.158815。按原词表，本轮Sports探针没有MOTION条目，不能据此判断体育动作句的时序效果。',
        'Dance：三种输入均通过排序判据，CROP数值最高0.453681；对较强LaMOT源的增益0.143299，5视频CI[0.016667,0.208065]。但CROP存在性准确率0.787059低于同域0.874118，数量MAE0.716471也劣于同域0.634118；CLIP和FRAME_BOX没有带来排序上的额外优势。97个有GT帧–表达的MOTION与APPEARANCE分组完全重合，不能把其差异当成独立动作理解证据。',
        '数量粒度并不普遍相同：Dance两源组合在G1就超过源排序×同域数量参照；KITTI两源和Sports两源直到G3仍未达到。C3/C4的逐帧GT数量G4也仍略低于同域模型数量参照，说明“真实目标数”与有检测漏检/误检的固定候选库上最有效的输出预算并不完全相同。',
        '这组固定解码器并不是严格单调嵌套的信息阶梯：G1在全视频内分配一个总预算，G2则把GT有目标帧的平均数量应用到所有帧，可能扩大总输出量；因此G1优于G2不是“更多信息反而有害”的因果证明。所需粒度只按预注册的首次达标规则命名，不强行把曲线变单调。',
        '整条GT轨迹选择的TRACK_ORACLE不依赖源分数排序，因而可以显著高于源排序的数量阶梯。它说明固定MOT中仍有可利用的轨迹信息，但不能证明冻结源语义分数、任一教师或GT-free方法已经能识别这些轨迹。',
        'A只回答同一稀疏固定探针上的排序、存在性和数量可靠性，不是完整视频RMOT HOTA；不能把任何教师探针增益直接当作全量TTA成功。文本类型是多标签，KITTI4、Dance5、Sports15探针视频的不确定性和全量B的样本量不同；不进行额外类型判定或事后选prompt。',
        '三臂同时按原文更改了视觉输入和对应固定提示词，属于输入/问法组合的比较，不能把差异单独因果归给时序或整帧上下文。',
        '冻结教师即使符合排序判据，仍须单独看存在性准确率和数量MAE；高空表达比例下较高的存在性准确率本身不足以说明能恢复正确目标。原始概率、两种R@1、AUROC、分类型结果和全部置信区间均保留，不从loss或工程测试推导科研结论。','',
        '固定“超过每个跨域源模型0.05”的门槛还受R@1上限约束：若源模型已经接近固定MOT的top1上限，教师可能没有0.05的可提升空间。这只是对“不能修排序”标签的边界解释，不改变原文判定，也不据此降低门槛。','',
        '## 6. GT隔离、失败与修复、交付','',
        '教师与GT-free校准均有Python打开审计+Landlock内核白名单，生产GT/指代标签打开尝试为0；GT只在各自预测/教师输出SHA封存后的独立Oracle/评测进程出现。A/B独立并行，无GT反馈。冻结教师初终参数SHA一致，原T1与vendor未改；没有训练、优化或新增方法队列。','',
        '7项新增科学测试和13项多域回归共20项通过；compileall通过。首次测试把阻止读取的文件放在有意允许的/tmp，纠正测试位置后通过；阶段报告在只有KITTI时触发空Sports表导出错误，修复阶段导出，不改任何预测/指标。资源迁移的中断/缓存恢复与计时不确定性完整保留。GitHub阶段上传发生传输失败，本地实验不受影响；最终推送状态另据实际回读确认，不能由本地完成推断已推送。完整说明见 `execution_notes.json`。','',
        '大文件不进Git，原始教师输出与每个预测绝对路径/SHA见 `artifact_paths.json`、`prediction_paths.json`、`all_prediction_files.csv.gz`；完整精简日志/回执归档为 `execution_logs.tar.gz`。固定探针与taxonomy见 `probe_sets.json`、`expression_types.csv`；72配置已全部封存并核验。','',
        '恢复根目录：`'+str(RUN)+'`。完成后停止；不训练、不追加探针、不根据本轮GT调阈值或新开方法。']
    (PUBLIC/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(event='DECOMPOSE_FINAL_REPORT_COMPLETE',teacher_verdicts=summaries,granularity=g,
                         GPU_process_hours=cost['new_GPU_process_hours'],prediction_configurations=72)),flush=True)


if __name__=='__main__':main()
