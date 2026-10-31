#!/usr/bin/env python3
"""Final sealed-output evaluation/attribution only; GT never feeds generators."""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diag_attrib_v2_common import *
from scripts.analyze_pf_video_uncertainty import paired_clusters, pooled_hota
from scripts.pf_diag_common import official_metrics, official_video_values
from scripts.analyze_pf_full_offline import load_eval_gt


def display(value):
    if value is None:
        return 'N/A'
    if isinstance(value, (int, np.integer)):
        return f'{int(value):,}'
    if isinstance(value, (float, np.floating)):
        return f'{float(value):.4f}'
    return str(value)


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '|'+ '|'.join(['---']*len(headers))+'|'] +
                     ['| '+' | '.join(display(v) for v in row)+' |' for row in rows])


def classify(comparisons, rows):
    explainers = [m for m in BASELINES if comparisons['D1_minus_'+m]['CI95'][0] <= 0 <= comparisons['D1_minus_'+m]['CI95'][1]
                  or comparisons['D1_minus_'+m]['CI95'][1] < 0]
    a = 'D1 的增益可被无学习校准解释' if explainers else '学生提供了超出简单校准的存在性信息'
    wins = {k: comparisons['D3_sort_minus_FROZEN_K_'+k]['CI95'][0] > 0
            for k in ('OFFICIAL_DANCE_COMPLETE', 'D1', 'ORACLE_GT')}
    if wins['OFFICIAL_DANCE_COMPLETE'] and wins['D1']:
        b = 'D3 的排序增益在合理数量下成立'
    elif not wins['OFFICIAL_DANCE_COMPLETE'] and not wins['D1'] and wins['ORACLE_GT']:
        b = '排序信号只在 oracle 数量下可见'
    elif not any(wins.values()):
        b = '未观察到可转化的排序增益'
    else:
        b = '预注册规则未覆盖的混合结果：'+json.dumps(wins, ensure_ascii=False)
    c = {m:dict(ratio=n/254910, label='与同域官方同量级' if .75 <= n/254910 <= 1.5 else '过量或不足')
         for m,n in rows.items()}
    return dict(A=a, A_explainer_baselines=explainers, B=b, B_positive_CI_columns=wins, C=c)


def report():
    gate = require_prereg()
    start, cpu = time.perf_counter(), time.process_time()
    gt_free = json.loads((RUN / 'GT_free_predictions.json').read_text())
    oracle = json.loads((RUN / 'ORACLE_predictions.json').read_text())
    references = json.loads((RUN / 'reference_validation.json').read_text())
    if gt_free['GT_free_target_open_attempts'] or references['status'] != 'PASS':
        raise ValueError('GT isolation/reference gate failed')
    methods = ['OFFICIAL_DANCE_COMPLETE', historical_name('OFFICIAL_DANCE_COMPLETE'), *gt_free['methods'], *oracle['methods']]
    if len(methods) != 58 or len(set(methods)) != 58:
        raise ValueError('complete preregistered matrix/sensitivity/baseline coverage is required')
    PUBLIC.mkdir(parents=True, exist_ok=True)
    manifests, metrics, values, completion = {}, {}, {}, {}
    for method in methods:
        manifests[method] = verify_seal(method)
        folder = EVAL / method
        done = json.loads((folder / 'official_completion.json').read_text())
        if done['exit_code'] or file_sha(folder / 'results/pedestrian_summary.txt') != done['summary_sha256'] or file_sha(folder / 'results/pedestrian_detailed.csv') != done['detailed_sha256']:
            raise ValueError('official outputs incomplete/changed: '+method)
        completion[method] = done
        metrics[method] = official_metrics(folder / 'results')
        videos, stats = official_video_values(folder / 'results/pedestrian_detailed.csv')
        exact = float(pooled_hota(stats, np.ones(25)))
        with (folder / 'results/pedestrian_detailed.csv').open() as stream:
            combined = next(r for r in csv.DictReader(stream) if r['seq'] == 'COMBINED')
        if not np.isclose(exact,100*float(combined['HOTA___AUC']),atol=1e-8,rtol=0):
            raise ValueError('official video pooling does not reproduce COMBINED exactly')
        if manifests[method]['prediction_rows'] != metrics[method]['Dets']:
            raise ValueError('official Dets differs from sealed row counts')
        values[method] = dict(videos=videos, stats=stats, exact_HOTA=exact)
    if any(item['videos'] != videos for item in values.values()):
        raise ValueError('video cluster alignment differs')
    recovery=json.loads((RUN/'B0_recovery.json').read_text())
    for row in recovery['affected']:
        if metrics[row['method']]['Dets']!=row['correct_registered_rows']:
            raise ValueError('corrected B0 official count still violates preregistration')
    recovery.update(status='FOUR_B0_POINTS_CORRECTED_AND_OFFICIALLY_REEVALUATED_PASS',
        corrected_official_metrics={r['method']:metrics[r['method']] for r in recovery['affected']})
    atomic_json(RUN/'B0_recovery.json',recovery)
    native = {m:matrix_name(m,m) for m in NAMES}
    native['D0_NAIVE'] = 'D0_NAIVE'
    counts = ['FROZEN','D1','D2','D3','OFFICIAL_DANCE_COMPLETE','ORACLE_GT']
    matrix = []
    for rank in NAMES:
        for count in counts:
            method = matrix_name(rank,count)
            row = dict(ranking=rank,count=count,role='ORACLE_TARGET_GT_DIAGNOSTIC_ONLY' if count=='ORACLE_GT' else 'POST_HOC_RANK_COUNT_DIAGNOSTIC',
                **{k:metrics[method][k] for k in ('HOTA','DetA','AssA','IDF1','IDSW','Dets')},
                exact_HOTA=values[method]['exact_HOTA'],
                minus_FROZEN_same_K=values[method]['exact_HOTA']-values[matrix_name('FROZEN',count)]['exact_HOTA'])
            matrix.append(row)
    for count in counts:
        if len({r['Dets'] for r in matrix if r['count']==count}) != 1:
            raise ValueError('rank/count matrix column does not keep exact output counts')
    write_csv(PUBLIC / 'attrib_matrix.csv', matrix)
    metadata, scores, groups, historical = load_bank()
    g = metadata['query_frame_groups']
    gt_counts = np.zeros(g, dtype=np.int32)
    query_truth = {}
    for q in metadata['episodes']:
        truth = load_eval_gt(DATA,q['video_id'],q['expression_id'])
        query_truth[q['video_id'],q['expression_id']] = sum(map(len,truth.values()))
        for t,a,b,group in q['groups']:
            gt_counts[group] = len(truth.get(t,[]))
    if int(gt_counts.sum()) != 212541 or sum(n>0 for n in query_truth.values()) != 40:
        raise ValueError('official GT count/presence protocol differs')
    group_lookup = {(q['video_id'],q['expression_id']):dict((t,group) for t,a,b,group in q['groups']) for q in metadata['episodes']}
    candidates_per_group = np.bincount(groups,minlength=g)
    alignment = {}
    alignment_order = ['OFFICIAL_DANCE_COMPLETE','FROZEN','D0_NAIVE','D1','D2','D3']
    baseline_roles = {
        'B1_SRC_RATE_GLOBAL':'源官方选中率+目标全局无标签分位；离线未来帧',
        'B1_SRC_RATE_QUERY':'源官方选中率+目标逐表达无标签分位；离线未来帧',
        'B2_Z_OFFLINE':'源z0+目标逐表达均值/总体方差；离线未来帧',
        'B2_Z_CAUSAL':'源因果z0+目标仅过去100数值帧；不含当前/未来',
        'B3_MEAN_GLOBAL':'源全局均值+目标全局均值；离线未来帧',
        'B3_MEAN_QUERY':'源全局均值+目标逐表达均值；离线未来帧',
        'B4_TEACHER_RATE':'Naive8500教师已判断样本+Frozen全视频；离线未来信息',
        'B4b_TEACHER_GATE':'Naive8500教师表达门控+Frozen；离线未来信息'}
    for name in alignment_order+BASELINES:
        method = native[name] if name in native else name
        k = np.zeros(g,dtype=np.int32)
        query_empty_rows = present_rows = 0
        for episode in manifests[method]['episodes']:
            key = episode['video'], episode['expression']
            counter = Counter(int(float(line.split(',',1)[0])) for line in Path(episode['prediction_path']).read_text().splitlines())
            for frame, count in counter.items():
                k[group_lookup[key][frame]] = count
            if query_truth[key]:
                present_rows += episode['selected_rows']
            else:
                query_empty_rows += episode['selected_rows']
        nonempty = k[k>0]
        row = dict(method=name, **metrics[method], prediction_rows=int(k.sum()),
            relative_GT=float(k.sum()/212541),relative_OFFICIAL_COMPLETE=float(k.sum()/254910),
            selection_rate=float(k.sum()/3588445), nonempty_query_frames=int(len(nonempty)),
            nonempty_query_frame_fraction=float(len(nonempty)/g), mean_k_nonempty=float(nonempty.mean()) if len(nonempty) else 0.,
            p50_k_nonempty=float(np.quantile(nonempty,.5)) if len(nonempty) else None,
            p90_k_nonempty=float(np.quantile(nonempty,.9)) if len(nonempty) else None,
            mean_k_GT_present_frame=float(k[gt_counts>0].mean()), mean_k_GT_empty_frame=float(k[gt_counts==0].mean()),
            query_present_selected_rows=present_rows, query_empty_selected_rows=query_empty_rows,
            target_GT_open_attempts_in_generator=0,
            GT_frame_labels_used_only_in_this_independent_evaluator=True,
            information=baseline_roles.get(name,'已冻结/已训练封存模型；非本轮新模型，非盲测'))
        if row['prediction_rows'] != metrics[method]['Dets']:
            raise ValueError('alignment quantity differs from official Dets')
        alignment[name] = row
    gt_info = dict(official_GT_detections=212541,official_GT_IDs=252,episodes=425,query_present_episodes=40,
        query_empty_episodes=385,full_query_frames=g,active_candidate_query_frames=int((candidates_per_group>0).sum()),
        zero_candidate_query_frames=int((candidates_per_group==0).sum()),GT_present_frames=int((gt_counts>0).sum()),
        GT_empty_frames=int((gt_counts==0).sum()),mean_GT_count_when_present=float(gt_counts[gt_counts>0].mean()),
        p50_GT_count_when_present=float(np.quantile(gt_counts[gt_counts>0],.5)),
        p90_GT_count_when_present=float(np.quantile(gt_counts[gt_counts>0],.9)),
        GT_present_without_candidates=int(((gt_counts>0)&(candidates_per_group==0)).sum()),
        filter='official mark!=0 and class==1; original timelines and425 expressions unchanged')
    atomic_json(PUBLIC / 'alignment.json',dict(status='COMPLETE',GT=gt_info,methods=alignment))
    fields = ['method','HOTA','DetA','AssA','DetRe','DetPr','AssRe','AssPr','IDF1','IDSW','CLR_FP','CLR_FN','Dets','IDs',
              'relative_GT','relative_OFFICIAL_COMPLETE','selection_rate','nonempty_query_frames','nonempty_query_frame_fraction',
              'mean_k_nonempty','p50_k_nonempty','p90_k_nonempty','mean_k_GT_present_frame','mean_k_GT_empty_frame',
              'query_present_selected_rows','query_empty_selected_rows','target_GT_open_attempts_in_generator','information']
    write_csv(PUBLIC / 'baselines.csv',[{k:alignment[name][k] for k in fields} for name in BASELINES])
    per_video=[]
    official_by_video=Counter()
    for e in manifests['OFFICIAL_DANCE_COMPLETE']['episodes']:
        official_by_video[e['video']]+=e['selected_rows']
    for method in methods:
        rows_by_video=Counter()
        for e in manifests[method]['episodes']:
            rows_by_video[e['video']]+=e['selected_rows']
        individual=pooled_hota(values[method]['stats'],np.eye(25))
        for v,hota in zip(videos,individual):
            per_video.append(dict(method=method,video=v,HOTA=float(hota),prediction_rows=rows_by_video[v],
                relative_OFFICIAL_COMPLETE_rows=rows_by_video[v]/official_by_video[v] if official_by_video[v] else None))
    write_csv(PUBLIC / 'per_video_all.csv',per_video)
    draws=np.random.default_rng(0).multinomial(25,np.ones(25)/25,size=2000)
    comparisons={}
    def compare(label,a,b,role='MAIN'):
        va,vb=values[a]['stats'],values[b]['stats']
        delta=float(paired_clusters(va,vb,np.ones(25)))
        ci=np.percentile(paired_clusters(va,vb,draws),[2.5,97.5]).tolist()
        per=paired_clusters(va,vb,np.eye(25))
        loo=paired_clusters(va,vb,np.ones((25,25))-np.eye(25))
        comparisons[label]=dict(first=a,second=b,role=role,delta_HOTA=delta,CI95=ci,
            leave_one_video_out_range=[float(loo.min()),float(loo.max())],negative_videos=int((per < -1e-9).sum()),
            positive_videos=int((per > 1e-9).sum()),tied_videos=int((np.abs(per)<=1e-9).sum()),
            row_count_difference=int(metrics[a]['Dets']-metrics[b]['Dets']),
            per_video=[dict(video=v,delta_HOTA=float(d),delta_without_video=float(l)) for v,d,l in zip(videos,per,loo)])
    for m in ('D1','D2','D3'):
        compare(m+'_minus_FROZEN_at_own_K',native[m],matrix_name('FROZEN',m))
    for baseline in BASELINES:
        compare('D1_minus_'+baseline,native['D1'],baseline)
    best=sorted(BASELINES,key=lambda name:(-values[name]['exact_HOTA'],name))[0]
    compare('D1_minus_best_GT_free_baseline',native['D1'],best)
    for count in ('OFFICIAL_DANCE_COMPLETE','D1','ORACLE_GT'):
        for rank in ('D1','D3'):
            compare(rank+'_sort_minus_FROZEN_K_'+count,matrix_name(rank,count),matrix_name('FROZEN',count),
                    role='ORACLE_DIAGNOSTIC' if count=='ORACLE_GT' else 'MAIN')
    compare('D1_minus_OFFICIAL_DANCE_COMPLETE',native['D1'],native['OFFICIAL_DANCE_COMPLETE'])
    compare('HIST_D1_minus_OFFICIAL',historical_name('D1'),historical_name('OFFICIAL_DANCE_COMPLETE'),role='SENSITIVITY_NOT_IN_VERDICT')
    verdict=classify(comparisons,{name:alignment[name]['prediction_rows'] for name in alignment_order})
    bootstrap=dict(status='COMPLETE',clusters=25,samples=2000,seed=0,pooling='official TP-weighted HOTA; verified unrounded COMBINED',
        interval='paired percentile 95%',best_GT_free_baseline_post_hoc=best,comparisons=comparisons,verdict=verdict,
        caveat='same-Val development analysis, not blind test; contains-zero rule is not an equivalence test; best-baseline selection not corrected')
    atomic_json(PUBLIC / 'bootstrap.json',bootstrap)
    sensitivity=[]
    for rank in NAMES:
        h=historical_name(rank)
        sensitivity.append(dict(ranking=rank,count=rank,role='SENSITIVITY_ONLY',**{k:metrics[h][k] for k in ('HOTA','DetA','AssA','IDF1','IDSW','Dets')},
            full_native_HOTA=metrics[native[rank]]['HOTA'],full_native_rows=metrics[native[rank]]['Dets']))
    for rank in ('D1','D3'):
        h=historical_name(rank,'OFFICIAL_DANCE_COMPLETE')
        sensitivity.append(dict(ranking=rank,count='OFFICIAL_DANCE_COMPLETE',role='SENSITIVITY_ONLY',**{k:metrics[h][k] for k in ('HOTA','DetA','AssA','IDF1','IDSW','Dets')},
            full_native_HOTA=None,full_native_rows=None))
    write_csv(PUBLIC / 'sensitivity.csv',sensitivity)
    b0=[dict(method=r['method'],rows=int(metrics[r['method']]['Dets']),HOTA=metrics[r['method']]['HOTA'],
             margin_boundary=r['threshold'],boundary_ties=r['boundary_ties'],selected_boundary_ties=r['selected_boundary_ties'],
             role='GT_INFORMED_SCALE_DIAGNOSTIC_NOT_METHOD') for r in oracle['B0_boundaries']]
    write_csv(PUBLIC / 'B0_curve.csv',b0)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.6,5.3))
    curve=sorted(b0,key=lambda r:r['rows'])
    ax.plot([r['rows'] for r in curve],[r['HOTA'] for r in curve],'o-',label='B0 Frozen global-threshold curve (diagnostic)',lw=1.5,ms=4)
    for name,color in [('FROZEN','black'),('D1','tab:orange'),('D2','tab:red'),('D3','tab:purple'),('OFFICIAL_DANCE_COMPLETE','tab:green')]:
        row=alignment[name]
        ax.scatter([row['prediction_rows']],[row['HOTA']],label=name,color=color,s=65,zorder=3)
    ax.set(xlabel='Prediction rows (complete fixed candidate bank)',ylabel='Official HOTA',title='25 videos / 425 queries: output-count calibration diagnostic')
    ax.ticklabel_format(axis='x',style='plain');ax.grid(alpha=.25);ax.legend(fontsize=8,loc='best')
    fig.text(.5,.01,'B0 uses GT-informed scale points; no best threshold is selected as a method.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1));fig.savefig(PUBLIC / 'B0_HOTA_rows.png',dpi=160);plt.close(fig)
    with gzip.open(PUBLIC / 'all_prediction_files.csv.gz','wt') as stream:
        writer=csv.DictWriter(stream,fieldnames=['method','video','expression','prediction_path','prediction_sha256','rows','margins_path','margins_sha256'],lineterminator='\n')
        writer.writeheader()
        for method,manifest in manifests.items():
            for e in manifest['episodes']:
                writer.writerow(dict(method=method,video=e['video'],expression=e['expression'],prediction_path=e['prediction_path'],
                    prediction_sha256=e['prediction_sha256'],rows=e['selected_rows'],margins_path=e.get('margins_path',''),margins_sha256=e.get('margins_sha256','')))
    atomic_json(PUBLIC / 'prediction_manifest.json',dict(complete_prediction_sets=58,episodes_each=425,total_prediction_files=24650,
        all_files_list='all_prediction_files.csv.gz',all_files_list_sha256=file_sha(PUBLIC / 'all_prediction_files.csv.gz'),
        sealed_predictions_on_server=True,models={m:dict(seal_path=str(EVAL / m / 'predictions_seal.json'),
        seal_sha256=file_sha(EVAL / m / 'predictions_seal.json'),rows=manifests[m]['prediction_rows']) for m in methods}))
    for source in [RUN/'GT_free_predictions.json',RUN/'ORACLE_predictions.json',RUN/'reference_validation.json',RUN/'matrix_self_checks.json',RUN/'prereg_publication.json',RUN/'B0_recovery.json']:
        shutil.copy2(source,PUBLIC / source.name)
    invalid_jobs=[]
    for path in (RUN/'retained_failures/b0_scale_clamp_1/evaluation').glob('*/official_completion.json'):
        invalid_jobs.append(dict(json.loads(path.read_text()),retained_receipt=str(path)))
    valid_new_jobs=sum(r['new_TrackEval_job'] for r in completion.values())
    invalid_wall=sum(r['worker_wall_seconds'] for r in invalid_jobs)
    costs=dict(new_GPU_calls=0,new_GPU_hours=0,optimizer_steps=0,new_teacher_calls=0,
        CPU_preparation_wall_seconds=sum(r['CPU_wall_seconds'] for r in [gt_free,oracle,json.loads((RUN/'reference_reproduction.json').read_text())]),
        CPU_preparation_process_seconds=sum(r['CPU_process_seconds'] for r in [gt_free,oracle,json.loads((RUN/'reference_reproduction.json').read_text())]),
        new_TrackEval_jobs=valid_new_jobs+len(invalid_jobs),valid_unique_new_TrackEval_jobs=valid_new_jobs,
        invalid_replaced_TrackEval_jobs=len(invalid_jobs),invalid_jobs=invalid_jobs,
        reused_exactly_equivalent_official_results=sum(not r['new_TrackEval_job'] for r in completion.values()),
        official_sum_worker_wall_seconds=sum(r['worker_wall_seconds'] for r in completion.values())+invalid_wall,
        invalid_official_worker_wall_seconds=invalid_wall,
        official_CPU_core_seconds=None,official_CPU_core_seconds_note='not instrumented; worker wall not mislabeled as CPU core time',
        per_job=completion,
        inherited_official_missing_score_GPU_worker_hours=json.loads((ROOT/'results/diag_attrib/official_reproduction.json').read_text())['GPU_worker_hours'],
        inherited_DFIT_costs=dict(path=str(ROOT/'results/diag_fit/costs.json'),sha256=file_sha(ROOT/'results/diag_fit/costs.json')),
        additional_Frozen_overlap_audit=json.loads((PUBLIC/'frozen_historical_overlap.json').read_text()),
        retained_initial_report_costs=json.loads((RUN/'retained_failures/b0_scale_clamp_1/completion_initial_revoked.json').read_text()),
        derived_controls_not_free_independent_models=True)
    atomic_json(PUBLIC/'costs.json',costs)
    atomic_json(PUBLIC/'gt_audit.json',dict(GT_free_generation_open_attempts=gt_free['GT_free_target_open_attempts'],
        reference_generation_open_attempts=json.loads((RUN/'reference_reproduction.json').read_text())['target_GT_open_attempts'],
        baseline_open_attempts={m:0 for m in BASELINES},ORACLE_GT_open_count=oracle['target_GT_open_count'],
        final_report_offline_GT_episodes_read=425,GT_group_labels_never_saved_in_method_score_bank=True,
        input_whitelist='runtime text-only manifest, fixed MOT, sealed margins, source scores, sealed Naive8500',
        oracle_and_evaluation_are_independent_processes=True,GT_feedback_to_online_or_calibration=False))
    archive=PUBLIC/'official_metrics_and_logs.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for method in methods:
            for name in ('pedestrian_summary.txt','pedestrian_detailed.csv'):
                tar.add(EVAL/method/'results'/name,arcname='official/'+method+'/'+name)
            tar.add(EVAL/method/'official_completion.json',arcname='official/'+method+'/completion.json')
            for path in (RUN/'official_jobs'/method).glob('stdout_*.txt'):
                tar.add(path,arcname='official/'+method+'/'+path.name)
        for path in (RUN/'logs').glob('*.txt'):
            if not path.name.startswith('report_'):
                tar.add(path,arcname='stage_logs/'+path.name)
        for name in ('reference_reproduction.json','source_statistics.json','GT_free_predictions.json','ORACLE_predictions.json','matrix_self_checks.json'):
            tar.add(RUN/name,arcname='provenance/'+name)
        retained=RUN/'retained_failures/b0_scale_clamp_1'
        if retained.exists():
            for path in (retained/'evaluation').glob('*/results/pedestrian*'):
                tar.add(path,arcname='retained_invalid_B0/'+path.relative_to(retained).as_posix())
            for path in (retained/'official_jobs').glob('*/stdout_*.txt'):
                tar.add(path,arcname='retained_invalid_B0/'+path.relative_to(retained).as_posix())
            for name in ('REPORT.md','B0_curve.csv','completion.json'):
                tar.add(retained/'initial_public_report'/name,arcname='retained_invalid_B0/initial_report/'+name)
            tar.add(RUN/'B0_recovery.json',arcname='provenance/B0_recovery.json')
    atomic_json(PUBLIC/'artifacts.json',dict(server_run_root=str(RUN),score_bank=str(CACHE),
        score_bank_metadata_sha256=file_sha(CACHE/'bank.json'),
        metrics_archive_sha256=file_sha(archive),metrics_archive_bytes=archive.stat().st_size,
        original_DFIT_results_unchanged=True,vendor_code_unchanged=True,
        full_reference_prediction_seal=file_sha(EVAL/'OFFICIAL_DANCE_COMPLETE/predictions_seal.json')))
    self_checks=json.loads((RUN/'matrix_self_checks.json').read_text())
    q1='；'.join(name+'='+f"{alignment[name]['relative_OFFICIAL_COMPLETE']:.4f}"+'倍' for name in ('FROZEN','D0_NAIVE','D1','D2','D3'))
    f1=comparisons['D1_minus_FROZEN_at_own_K'];f3=comparisons['D3_minus_FROZEN_at_own_K']
    d3o=comparisons['D3_sort_minus_FROZEN_K_OFFICIAL_DANCE_COMPLETE'];d3d=comparisons['D3_sort_minus_FROZEN_K_D1']
    summary=[
        f"Q1/判定C：相对完整同域官方254,910行，{q1}；D1与官方同量级，D2/D3过量、Frozen/D0不足（仅描述，不等于方法优劣）。",
        f"Q2：D1−Frozen@D1数量={f1['delta_HOTA']:+.4f} HOTA（CI{f1['CI95']}），D3−Frozen@D3数量={f3['delta_HOTA']:+.4f}；数量/存在性效应与帧内排序不能混为一谈。",
        f"Q3/判定A：{verdict['A']}；事后最好固定GT-free基线为{best}，HOTA={metrics[best]['HOTA']:.4f}，D1差值区间={comparisons['D1_minus_best_GT_free_baseline']['CI95']}（只报告、不改定义）。",
        f"Q4/判定B：{verdict['B']}；D3排序−Frozen在K=完整官方/D1下分别{d3o['delta_HOTA']:+.4f}/{d3d['delta_HOTA']:+.4f}，区间{d3o['CI95']} / {d3d['CI95']}。",
        "主参照完整HOTA32.139，历史覆盖重新评测32.134；全部25视频425表达，开发Val事后诊断、不是盲测/Train→Val，不训练新模型。"]
    body=['# D-FIT事后归因 v2：完整候选参照、数量校准与排序分离','', '状态：COMPLETE；完成后停止。v1停止记录与全部历史结果/vendor不改。','',
        '## 1. 结论摘要（Q1–Q4）','',*['- '+s for s in summary],'',
        '## 2. 表C：完整覆盖对齐','',
        'GT统计：212,541检测、252个GT IDs；40 Query-Present/385 Query-Empty；完整视频时间轴'+str(g)+'个query-frame，其中GT有目标'+str(gt_info['GT_present_frames'])+'帧、空'+str(gt_info['GT_empty_frames'])+'帧。GT不作为一个可部署模型，其HOTA/IDF1无意义，记N/A。','']
    c_fields=[('预测/GT行数','prediction_rows'),('/GT检测','relative_GT'),('/完整官方行数','relative_OFFICIAL_COMPLETE'),
        ('选中率','selection_rate'),('有输出query-frames','nonempty_query_frames'),('有输出帧比例','nonempty_query_frame_fraction'),
        ('非空帧平均k','mean_k_nonempty'),('非空帧p50 k','p50_k_nonempty'),('非空帧p90 k','p90_k_nonempty'),
        ('GT有目标帧平均k','mean_k_GT_present_frame'),('GT空帧平均k','mean_k_GT_empty_frame'),
        *[(k,k) for k in ('HOTA','DetA','AssA','DetRe','DetPr','AssRe','AssPr','IDF1','IDSW','CLR_FP','CLR_FN','Dets','IDs')],
        ('Query-Empty全部预测行','query_empty_selected_rows')]
    gt_column={'prediction_rows':212541,'relative_GT':1.,'relative_OFFICIAL_COMPLETE':212541/254910,
               'nonempty_query_frames':gt_info['GT_present_frames'],'nonempty_query_frame_fraction':gt_info['GT_present_frames']/g,
               'mean_k_nonempty':gt_info['mean_GT_count_when_present'],'p50_k_nonempty':gt_info['p50_GT_count_when_present'],
               'p90_k_nonempty':gt_info['p90_GT_count_when_present'],'mean_k_GT_present_frame':gt_info['mean_GT_count_when_present'],
               'mean_k_GT_empty_frame':0,'Dets':212541,'IDs':252}
    body+=[table(['指标','GT统计']+alignment_order,[[label,gt_column.get(key)]+[alignment[n][key] for n in alignment_order] for label,key in c_fields]),'',
        'GT列IDs指GT_IDs，其余模型IDs为预测IDs；FP/FN是官方CLEAR，不混用旧IoU离线行匹配。k统计包含零候选帧，p50/p90仅在有输出帧上，分位为numpy默认线性插值。每视频HOTA/行数/同视频官方倍数见per_video_all.csv。','',
        '## 3. 排序×数量矩阵（30格）','',
        'K=ORACLE_GT【ORACLE，使用GT，仅诊断】不属于方法；非oracle所有K都是对应模型实际margin>0的逐帧数量。下表HOTA采用官方summary，差值采用未舍入官方TP加权值；最后位的差异不是更换指标。','']
    for label,field in [('HOTA矩阵','HOTA'),('预测行数矩阵','Dets'),('同列减Frozen排序的HOTA差值','minus_FROZEN_same_K')]:
        body+=['### '+label,'',table(['排序R / 数量K']+[c if c!='ORACLE_GT' else 'ORACLE_GT【GT】' for c in counts],
            [[r]+[next(x[field] for x in matrix if x['ranking']==r and x['count']==c) for c in counts] for r in NAMES]),'']
    body+=['混合格继承排序模型和数量模型的已付训练/推理成本，是后验诊断，不是免费Frozen系统、独立新算法或GT-free oracle。固定列严格保持同样输出行数，变化只能来自选择哪些同帧候选及其关联效果。','',
        '### 贡献解释：逐帧/逐表达k并非一个全局数量','',
        'D1、Frozen@D1逐帧k、B0 Frozen全局top345,302，三者总输出同为345,302行，但HOTA分别29.702、29.400、12.930；关键不是单纯把阈值降低到相同总数，而是把数量分配到哪些表达和时刻。Frozen@D1继承了训练后D1提供的查询条件数量信息，不能把它描述成无需学习的Frozen校准。','',
        'D1相对Frozen的整体改善约21.484点，而固定D1数量后排序差只有约0.3025且区间跨0；HOTA非线性、数量与排序有交互，不能据此计算一个因果贡献百分比。表C也显示D1非空帧81,061多于官方41,979，虽然非空帧平均k反而更低（4.26 vs6.07），剩余差距仍包含存在性误发射，不只是每帧目标过多。','',
        'D3在两列实际可用数量下区间均跨0，仅GT数量列为正；这支持受理想数量条件约束的内部排序信号，不支持宣称已转化为可部署HOTA收益。D1点值仍低于同域参考32.139；D1−官方区间包含0不等于证明追平或等效。此诊断定位输出机制，不单独证明教师监督是唯一训练因果因素，也不是盲测泛化。','',
        '## 4. 固定校准基线（B1–B4）','',
        table(['版本','HOTA','DetA','AssA','IDF1','行数','/GT','/完整官方','GT空帧平均k','Query-Empty预测行'],
            [[m]+[alignment[m][k] for k in ('HOTA','DetA','AssA','IDF1','prediction_rows','relative_GT','relative_OFFICIAL_COMPLETE','mean_k_GT_empty_frame','query_empty_selected_rows')] for m in BASELINES]),'',
        *['- '+m+'：'+baseline_roles[m]+'；生成进程目标GT打开0。' for m in BASELINES],'',
        '完整表C同字段（包括DetRe/DetPr/AssRe/AssPr/FP/FN/IDs、帧数量统计）见baselines.csv/alignment.json。源r_src=0.3187540118393838、源均值−8.394860413379645，z0_offline=1.4466019189252521、z0_causal=1.8597054552922299；均在源官方4视频862表达计算，不能称17源Train统计。','',
        'B4_RATE按预注册P_q/N_q最终只输出656行，而非将稀疏教师正率外推到全视频；这是固定基线的定义，不据GT事后改公式。回退明细和边界tie见GT_free_predictions.json，未采样/无效候选不补硬负例。','',
        '## 5. 25视频聚类bootstrap与判定','',
        '固定2000次/seed0，未舍入官方HOTA精确合并，95%配对percentile区间；leave-one-video-out删除整段视频及全部表达，不能把帧当独立样本。','',
        table(['比较','ΔHOTA','95% CI','LOO范围','负/正/平视频','行数差'],
              [[name,r['delta_HOTA'],'['+', '.join(f'{x:+.4f}' for x in r['CI95'])+']',
                '['+', '.join(f'{x:+.4f}' for x in r['leave_one_video_out_range'])+']',
                f"{r['negative_videos']}/{r['positive_videos']}/{r['tied_videos']}",r['row_count_difference']] for name,r in comparisons.items()]),'',
        '判定A：'+verdict['A']+'；触发基线：'+(', '.join(verdict['A_explainer_baselines']) or '无')+'。',
        '判定B：'+verdict['B']+'。',
        '判定C：'+json.dumps(verdict['C'],ensure_ascii=False)+'。','',
        '判定逐字依预注册，但“区间包含0”不是等效检验，更不是完整因果证明；只针对这八个固定基线，事后选最好基线的区间未校正选择，不能夸大成所有校准都足够/所有存在性都已解决。敏感性和B0不参与A/B/C。','',
        '## 6. B0全局阈值曲线（不是方法）','',
        '![B0 HOTA–预测行数](B0_HOTA_rows.png)','',
        table(['曲线点','行数','HOTA','margin边界','边界tie数'],[[r['method'],r['rows'],r['HOTA'],r['margin_boundary'],r['boundary_ties']] for r in b0]),'',
        '预算含用户指定GT检测规模，仅作阈值/数量可恢复性曲线；没有挑最优阈值部署，没有用目标GT优化B1–B4。原始模型散点不是曲线上相同行数预测的等价物。','',
        '## 7. 自检与两种覆盖范围','',
        '完整参照254,910行HOTA32.139，历史参照254,118行HOTA32.134；原覆盖425表达逐行复现，新792行均来自692末观测。其余四主对角和六旧count_controls逐行等价通过，原HOTA8.218/29.702/22.844/27.768与29.400/22.854/27.103及7.8978/7.8380/8.1589不变，详见matrix_self_checks.json。','',
        'FlexHook data/mydataloader.py:354/359的排他range漏每轨迹末观测；完整211,085 vs历史210,393，差692 MOT/11,764候选–表达。vendor不改，历史预测不覆写。敏感性只是预测候选统一减末观测，GT、425表达、时间轴及官方过滤均保持原样。','',
        table(['历史覆盖排序R','历史覆盖K','HOTA','行数','完整原生HOTA','完整原生行数'],
            [[r['ranking'],r['count'],r['HOTA'],r['Dets'],r['full_native_HOTA'],r['full_native_rows']] for r in sensitivity]),'',
        '历史覆盖D1−OFFICIAL的CI单列在bootstrap最后一行，不参与主判定；七套敏感性预测均本轮重新官方评价，不拿旧结果冒充重评。完整/历史HOTA可对照，但不根据哪个较好挑选覆盖。','',
        '这里历史覆盖Frozen是**当前已封存ZERO_UPDATE margin的同一历史候选子集**，因此为8.2072/99,320，而非替换为早期批量推理的8.2060/99,316；两份文件均未改。全3,576,681重叠分数审计查到6个接近零的符号差（5新增、1删除，净+4），所有边界margin绝对值不超过0.000128；全局最大margin差0.00118494，符合不同既有数值路径的小差异。具体逐候选分数在frozen_historical_overlap.json；不声称当前子集逐行复现早期batch文件，不按GT挑选或替换这些margin。官方Dance历史对角仍按硬门槛严格复现254,118/32.134。','',
        '## 8. GT隔离审计','',
        '参考构建与全部40套GT-free预测生成独立进程目标GT打开尝试均0；只用剥离标签runtime文本、固定MOT、封存margin、源分数、Naive8500。最终封存后独立ORACLE进程读425 GT文件，仅生成明确ORACLE/B0；报告进程读425 GT做帧分组。没有GT进入源统计、分位阈值、学生、checkpoint选择或因果窗口。','',
        '所有58预测集各425表达，24,650个预测文件均封存并在官方评测后复核SHA；逐文件清单all_prediction_files.csv.gz，服务器完整文件路径与seal见prediction_manifest.json。旧D-FIT/v1权重、margin、预测不动。','',
        '## 9. 成本与可恢复路径','',
        f"本轮新GPU调用0、训练/优化0、新教师0；CPU准备wall合计{costs['CPU_preparation_wall_seconds']:.3f}s、进程CPU {costs['CPU_preparation_process_seconds']:.3f}s；有效新官方jobs={valid_new_jobs}，保留无效B0尝试={len(invalid_jobs)}，新评价实际总次数={costs['new_TrackEval_jobs']}，逐行等价复用={costs['reused_exactly_equivalent_official_results']}，官方worker-wall累计{costs['official_sum_worker_wall_seconds']:.3f}s（含无效尝试、并行求和，不是调度墙钟或CPU-core小时）。报告/打包耗时另见completion。",'',
        '先前官方补分数0.051710 GPU-process小时及D-FIT训练/推理成本仍存在，见costs.json继承路径SHA；不能将本轮低CPU成本说成免费独立RMOT。每job日志/完成收据与完整官方summary+detailed在official_metrics_and_logs.tar.gz。',
        '服务器恢复根：`'+str(RUN)+'`；仅恢复缺失/失败单元，已通过单元不重跑。所有计算完成后停止，不启动新训练。','',
        '## 10. 失败与修复记录','',
        'v1明确停止记录保留：完整官方与旧254,118条件不相容；用户批准并先推送v2后才改用完整参照，不称为科研收益。v2初始8项针对性CPU测试通过；补充判定和B0绝对预算回归后共12项通过（不累加重叠运行次数）。','',
        '汇总发现真实B0实现错误：round_budget(scale,GT_count)误把1.25/1.5/2/3倍GT预算均截为212,541；撤销未推送的初版完整报告，保留全部四套错误预测、官方CSV/日志与初版报告。最小修复改为绝对round-half-up(scale×212,541)，只按实际3,588,445候选上限截断；正确预算265,676/318,812/425,082/637,623，仅重评这四点。其余健康评测不重跑，预注册定义、seed、权重、阈值算法均不改；A/B/C不依赖B0且未因修复改变。详情见EXECUTION_NOTES.md、B0_recovery.json及归档retained_invalid_B0。','',
        '## 建议（只建议，不执行）','',
        '当前证据支持保留D1作为学到逐表达/逐帧数量分配的基线，而不是把它的收益归为一个全局阈值；不把D3的Oracle排序收益包装成实际方法成功。若后续获授权，优先研究无GT的查询存在性/时间分配与空目标误发射控制，保留简单源统计校准作强制对照，不继续通过更大pos_weight扩大输出。两模型拼接后验格子不是可部署新算法；本轮不开始新训练。']
    (PUBLIC/'REPORT.md').write_text('\n'.join(body)+'\n')
    protected=json.loads((ROOT/'results/runs/diag_attrib_20261031_v1/protected_inputs.json').read_text())
    for row in protected:
        for kind in ('prediction','margins'):
            if file_sha(row[kind+'_path'])!=row[kind+'_sha256']:
                raise ValueError('original D-FIT/Frozen sealed input was changed')
    old_weights=[r for r in json.loads((ROOT/'results/diag_fit/server_artifacts.json').read_text())['checkpoints']
                 if r['phase']=='students' and r['path'].endswith('/checkpoint_final_step.pt')]
    for row in old_weights:
        if file_sha(row['path'])!=row['sha256']:
            raise ValueError('old D-FIT final checkpoint changed')
    if file_sha(ROOT/'results/diag_attrib/REPORT.md')!='248270bf103e8ff72ca7c6705586d0d1fb85d079dac7dde80ab251e6c77c454c':
        raise ValueError('v1 historical stop report changed')
    atomic_json(PUBLIC/'post_eval_integrity.json',dict(status='PASS',old_prediction_margin_pairs=len(protected),
        old_final_checkpoints_verified=len(old_weights),v1_report_and_prereg_unchanged=True,
        all_58_new_prediction_sets_reverified_after_official_evaluation=True,GT_free_generator_target_GT_open_attempts=0))
    result=dict(status='COMPLETE_STOP',methods=58,episodes_each=425,official_jobs=costs['new_TrackEval_jobs'],
        best_GT_free_baseline=best,verdict=verdict,new_GPU_calls=0,optimizer_steps=0,
        report_path=str(PUBLIC/'REPORT.md'),report_sha256=file_sha(PUBLIC/'REPORT.md'),
        CPU_report_wall_seconds=time.perf_counter()-start,CPU_report_process_seconds=time.process_time()-cpu,
        all_experiment_subprocesses_finished=True)
    atomic_json(RUN/'completion.json',result)
    atomic_json(PUBLIC/'completion.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    report()
