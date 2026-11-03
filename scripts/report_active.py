"""Post-seal scientific accounting. Never imported by a GT-free runtime worker."""
from argparse import ArgumentParser
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.presence.runtime import RUN, PUBLIC, target_metadata
from flex_uda.presence.predictions import verify_predictions
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.active_run import PAIRS
from scripts.active_plan import jobs


def write_table(path, rows):
    if not rows: return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields,lineterminator='\n'); writer.writeheader(); writer.writerows(rows)


def committed_teacher_equivalent_hours(receipts,teacher,method):
    """One cost per committed question, separate from failed-session work."""
    if method=='GATE_REASSIGN_REF':return teacher.get('equivalent_forward_seconds',0)/3600
    if receipts and all('teacher_delta' in row for row in receipts):
        return sum(row['teacher_delta']['equivalent_forward_seconds'] for row in receipts)/3600
    # The budget baselines retain their exact ordered question-key lists.
    # A repeated logical call stays repeated; only the physical cache is shared.
    import sqlite3
    keys=[key for row in receipts for key in row['question_keys']]
    if len(keys)!=teacher['logical_calls']:raise ValueError('Incomplete committed teacher question list')
    seconds=0.
    with sqlite3.connect((RUN/'teacher/raw.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as database:
        for key in keys:
            raw=database.execute('SELECT payload FROM outputs WHERE key=?',(key,)).fetchone()
            if raw is None or raw[0] is None:raise ValueError('Missing sealed teacher answer '+key)
            answer=json.loads(raw[0]);seconds+=answer['batch_forward_seconds']/answer['batch_size']
    return seconds/3600


def plot_adaptation_cost(rows, costs, public):
    """Use independent-equivalent teacher cost, not cache-dependent new calls."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=('FULL','MINUS_SLOW','UNIFORM_K','YESTRACK_BUDGET_RANDOM','YESTRACK_BUDGET_PRIOR')
    pairs=[pair for pair in PAIRS if any(r['pair']==pair and r['method'] in methods for r in rows)]
    if not pairs:return
    columns=min(3,len(pairs));lines=(len(pairs)+columns-1)//columns
    fig,axes=plt.subplots(lines,columns,figsize=(5.5*columns,4.5*lines),squeeze=False)
    lookup={(r['pair'],r['method'],str(r['rho'])):r for r in costs}
    labels=dict(FULL='FULL',MINUS_SLOW='MINUS_SLOW',UNIFORM_K='UNIFORM',
        YESTRACK_BUDGET_RANDOM='YT_RANDOM',YESTRACK_BUDGET_PRIOR='YT_PRIOR')
    palette=plt.get_cmap('tab10')
    for ax,pair in zip(axes.flat,pairs):
        for mi,method in enumerate(methods):
            for row in (r for r in rows if r['pair']==pair and r['method']==method):
                cost=lookup[pair,method,str(row['rho'])]
                hours=float(cost.get('committed_teacher_equivalent_forward_GPU_hours',cost['teacher_equivalent_forward_GPU_hours']))
                hours+=sum(float(cost[k]) for k in ('LoRA_GPU_hours','rescore_GPU_hours'))
                if hours<=0:continue
                ax.scatter(hours,float(row['HOTA']),color=palette(mi),s=40)
                ax.annotate(labels[method]+' (rho='+str(row['rho'])+')',
                    (hours,float(row['HOTA'])),xytext=(5,4),textcoords='offset points',fontsize=7)
        ax.set(xscale='log',title=pair+': '+PAIRS[pair][0]+' -> '+PAIRS[pair][1],
            xlabel='Committed teacher + valid-lineage LoRA/rescore hours (log)',ylabel='Official HOTA')
        ax.margins(x=.7,y=.18);ax.grid(alpha=.2)
    for ax in axes.flat[len(pairs):]:ax.set_visible(False)
    fig.suptitle('Completed jobs only; no cache discounts in teacher cost')
    fig.text(.5,.01,'Excludes shared source preparation, frozen scoring, front-end and loading overhead.\n'
        'Shared-device elapsed phase time is not a GPU-utilization integral; only sealed budget points are shown.\n'
        'Discarded interrupted work is recorded separately in the wall-time ledger.',
        ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.10,1,.94))
    fig.savefig(Path(public) / 'adaptation_cost_vs_HOTA.png',dpi=160);plt.close(fig)


def folder_for(job):
    from flex_uda.active.execution import active_paths
    suffix = 'original_delta10' if job['method'] == 'YESTRACK_FULL_D10' else 'rho_'+str(job['rho'])
    return active_paths().active / job['pair'] / job['method'] / suffix


def video_differences(rows,execution):
    from scripts.active_eval import official_stats
    from scripts.presence_finish_a import references
    from scripts.analyze_pf_video_uncertainty import pooled_hota
    records=[]
    for full in rows:
        if full['method']!='FULL':continue
        pair,rho,target=full['pair'],full['rho'],full['target']
        native=dict(references(target,PAIRS[pair][0]))['NATIVE']
        controls=[('NATIVE',native),('PRES_TOPK',execution.stage_a / target / PAIRS[pair][0] / 'PRES_TOPK')]
        controls += [(r['method'],Path(r['path'])) for r in rows
            if r['pair']==pair and r['method']!='FULL' and r['rho'] in (rho,None)]
        a=official_stats(full['path']);weights=np.eye(len(a['videos']))
        for method,folder in controls:
            b=official_stats(folder)
            if a['videos']!=b['videos']:raise ValueError('Different video coverage in attribution')
            for i,video in enumerate(a['videos']):
                first,second=float(pooled_hota(a['stats'],weights[i])),float(pooled_hota(b['stats'],weights[i]))
                records.append(dict(pair=pair,rho=rho,target=target,comparison='FULL_minus_'+method,
                    video=video,FULL_HOTA=first,reference_HOTA=second,delta_HOTA=first-second))
    return records


def calibration_calls(receipts,pair,method,rho):
    rows=[]
    for video in receipts:
        for round_info in video.get('rounds',[]):
            before=round_info['fast']
            positions=[('BEFORE_SLOW',before)]
            if 'after_slow' in before:positions.append(('AFTER_SLOW',before['after_slow']))
            for position,fast in positions:
                identity,presence=fast.get('identity',{}),fast.get('presence',{})
                rows.append(dict(pair=pair,method=method,rho=rho,video=video['video'],round=round_info['round'],
                    position=position,status=fast['status'],slope=fast.get('slope'),
                    temperature=fast.get('temperature'),bias=fast.get('bias'),
                    identity_status=identity.get('status'),supported_positive=identity.get('positive'),
                    supported_negative=identity.get('negative'),
                    unqueried_excluded=identity.get('unqueried_confident_excluded'),
                    presence_status=presence.get('status'),confirmed_positive=presence.get('positive'),
                    confirmed_negative=presence.get('negative'),
                    identity_objective_before=identity.get('objective_before'),
                    identity_objective_after=identity.get('objective_after'),
                    presence_objective_before=presence.get('objective_before'),
                    presence_objective_after=presence.get('objective_after')))
    return rows


def saved_presence_decisions(folder,meta,truth,pair,method,rho):
    """Offline only: distinguish query absence from missing MOT candidates."""
    rows=[]
    for path in sorted(Path(folder).glob('videos/*/belief.npz')):
        video=path.parent.name;bank=meta['videos'][video]
        by_track=defaultdict(list)
        for j,(_,track) in enumerate(bank['observations']):by_track[track].append(j)
        mask=np.load(path.parent / 'mask.npy')
        asked={(q['query_id'],q['track_id']) for q in json.loads((path.parent / 'questions.json').read_text())}
        with np.load(path) as belief:
            tracks=belief['tracks'].tolist();eligible=np.asarray([len(by_track[t])>=3 for t in tracks])
            for qi,query in enumerate(belief['queries'].tolist()):
                gt=truth[video,query];labels=gt['labels']
                present=bool(gt['counts'].sum());retrievable=bool(labels.any())
                yes,no=bool(belief['confirmed_present'][qi]),bool(belief['absent'][qi])
                selected_tracks=sum(bool(mask[qi,by_track[t]].any()) for t in tracks)
                matched=np.asarray([labels[by_track[t]].mean()>=.5 for t in tracks],bool)
                majority=int((matched&eligible).sum())
                high_positive=(belief['posterior'][qi]>.95)&eligible
                high_negative=(belief['posterior'][qi]<.05)&eligible
                queried=np.asarray([(query,t) in asked for t in tracks],bool)
                unqueried_positive=int((matched&eligible&~queried).sum())
                wrong_negative=int((high_negative&matched).sum())
                wrong_negative_unqueried=int((high_negative&matched&~queried).sum())
                rows.append(dict(pair=pair,method=method,rho=rho,video=video,expression=query,
                    query_present_GT=present,query_retrievable_GT=retrievable,
                    GT_observations=int(gt['counts'].sum()),candidate_positive_observations=int(labels.sum()),
                    eligible_tracks=int(eligible.sum()),matched_candidate_positive_tracks_majority=int(majority),
                    belief_expected_count=float(belief['expected_counts'][qi]),
                    low_presence_flag=bool(belief['low_presence'][qi]),
                    teacher_confirmed_present=yes,teacher_absent=no,teacher_contradiction=yes and no,
                    false_absence_with_retrievable_GT=no and retrievable,
                    teacher_false_presence_on_empty=yes and not present,
                    posterior_positive_tracks_p50=int((belief['posterior'][qi]>.5).sum()),
                    confident_positive_tracks_p95=int(((belief['posterior'][qi]>.95)&eligible).sum()),
                    confident_negative_tracks_p05=int(high_negative.sum()),
                    confident_positive_GT_TP=int((high_positive&matched).sum()),
                    confident_positive_GT_FP=int((high_positive&~matched).sum()),
                    confident_negative_GT_TN=int((high_negative&~matched).sum()),
                    confident_negative_GT_FN=wrong_negative,
                    confident_negative_GT_FN_unqueried=wrong_negative_unqueried,
                    queried_eligible_pairs=int((queried&eligible).sum()),
                    unqueried_GT_positive_tracks=unqueried_positive,
                    final_confidence_coverage=float((high_positive|high_negative).sum()/eligible.sum()) if eligible.any() else None,
                    final_GTpositive_confident_negative_rate=wrong_negative/majority if majority else None,
                    unqueried_GTpositive_confident_negative_rate=wrong_negative_unqueried/unqueried_positive if unqueried_positive else None,
                    final_belief_not_optimizer_sample_labels=True,
                    selected_prediction_tracks=selected_tracks,selected_prediction_rows=int(mask[qi].sum()),
                    ORACLE_GT_DIAGNOSTIC_ONLY=True,GT_feedback_to_runtime=False))
    return rows


def final_confidence_summary(rows):
    """Descriptive track-query counts, not labels actually seen by SGD."""
    groups=defaultdict(list)
    for row in rows:groups[row['pair'],row['method'],row['rho']].append(row)
    fields=('eligible_tracks','matched_candidate_positive_tracks_majority',
        'confident_positive_tracks_p95','confident_negative_tracks_p05',
        'confident_positive_GT_TP','confident_positive_GT_FP','confident_negative_GT_TN',
        'confident_negative_GT_FN','confident_negative_GT_FN_unqueried',
        'queried_eligible_pairs','unqueried_GT_positive_tracks')
    result=[]
    for (pair,method,rho),values in sorted(groups.items()):
        counts={key:sum(row[key] for row in values) for key in fields}
        ratio=lambda a,b:counts[a]/counts[b] if counts[b] else None
        result.append(dict(pair=pair,method=method,rho=rho,episodes=len(values),**counts,
            high_positive_GT_precision=ratio('confident_positive_GT_TP','confident_positive_tracks_p95'),
            high_negative_GT_error_rate=ratio('confident_negative_GT_FN','confident_negative_tracks_p05'),
            GTpositive_high_negative_rate=ratio('confident_negative_GT_FN','matched_candidate_positive_tracks_majority'),
            unqueried_GTpositive_high_negative_rate=ratio('confident_negative_GT_FN_unqueried','unqueried_GT_positive_tracks'),
            unit='eligible (query, predicted track) pair; n>=3 and GT-majority>=0.5',
            final_belief_not_optimizer_sample_labels=True,GT_feedback_to_runtime=False))
    return result


def behavior_and_reliability(folder, target, pair, method, rho, truth):
    """Keep anchor-observation and majority-track GT diagnostic labels separate."""
    meta = target_metadata(target); rounds = []; videos = []; reliability = []
    cumulative = defaultdict(list); identity_seen = set()
    for vi, video in enumerate(sorted(meta['videos'])):
        root = folder / 'videos' / video
        if not (root / 'receipt.json').exists(): continue
        receipt = json.loads((root / 'receipt.json').read_text())
        bank = meta['videos'][video]
        positions = {tuple(o): j for j, o in enumerate(bank['observations'])}
        track_indices = defaultdict(list)
        for j, (t, track) in enumerate(bank['observations']): track_indices[track].append(j)
        question_path = root / 'questions.json'
        answers = json.loads(question_path.read_text()) if question_path.exists() else []
        for answer in answers:
            query, track, frame = answer['query_id'], answer['track_id'], answer['frame_id']
            y = truth[video, query]['labels']; at = positions[frame, track]
            track_y = bool(y[track_indices[track]].mean() >= .5)
            cumulative[answer['effective_view']].append((answer['yes_probability'] >= .5, bool(y[at]), track_y))
            key = video, query, track
            if key not in identity_seen:
                identity_seen.add(key)
                cumulative['IDENTITY'].append((bool(answer['identity_vote']), bool(y[at]), track_y))
        used = 0; thresholds = {}; previous = 0
        for row in receipt.get('rounds', []):
            used += row['actual_questions']; total = row['actual_questions']
            rounds.append(dict(pair=pair, method=method, rho=rho, video=video, video_index=vi, round=row['round'],
                actual_questions=total, presence_question_fraction=row['presence_questions']/total if total else None,
                identity_question_fraction=row['identity_questions']/total if total else None,
                new_confirmed_teacher=row['new_confirmed_pairs_teacher'], new_confirmed_final=row['new_confirmed_pairs_final'],
                new_distinct_confirmed=row['new_distinct_confirmed_pairs'], confirmed_pairs=row['confirmed_pairs'],
                additional_distinct_confirmed=row['additional_distinct_confirmed_pairs'],
                confident_pairs_never_asked=row['confident_pairs_never_asked'],
                optimizer_updates=row['slow']['optimizer_updates'],
                LoRA_GPU_seconds=row['slow']['GPU_seconds']))
            added = row['additional_distinct_confirmed_pairs']
            for k in (1, 5, 10, 25, 50):
                if k not in thresholds and added >= k: thresholds[k] = used
            previous = added
        videos.append(dict(pair=pair, method=method, rho=rho, video=video, video_index=vi,
            actual_questions=receipt.get('questions', receipt.get('actual_calls', 0)),
            initial_confident_pairs=receipt.get('initial_confident_pairs'), additional_distinct_confirmed=previous,
            calls_per_additional_confirmation=used/previous if previous else None,
            confirmation_quota_calls_at_round_boundary=json.dumps(thresholds, sort_keys=True),
            quota_caveat='round boundary upper bound, confidence is not correctness',
            coverage_fraction=receipt.get('coverage_fraction'), delta=receipt.get('delta')))
        final_ds = receipt.get('rounds', [{}])[-1].get('ds', {}).get('reliability', {})
        for worker, values in cumulative.items():
            for scope, column in [('ANCHOR_OBSERVATION', 1), ('TRACK_MAJORITY_50_PERCENT', 2)]:
                # IDENTITY is a track prior, so its anchor frame is only the
                # first queried observation; majority is the primary label.
                p = np.asarray([v[0] for v in values], bool); y = np.asarray([v[column] for v in values], bool)
                pos, neg = int(y.sum()), int((~y).sum())
                estimated = final_ds.get(worker, {})
                reliability.append(dict(pair=pair, method=method, rho=rho, through_video=video, worker=worker,
                    GT_scope=scope, observations=len(y), positives=pos, negatives=neg,
                    sensitivity_GT=float((p & y).sum()/pos) if pos else None,
                    specificity_GT=float((~p & ~y).sum()/neg) if neg else None,
                    sensitivity_DS=estimated.get('sensitivity'), specificity_DS=estimated.get('specificity'),
                    ORACLE_GT_DIAGNOSTIC_ONLY=True,
                    correlation_caveat='repeated views/track frames are correlated; no independent-frame significance claim'))
    return rounds, videos, reliability


def main():
    global PUBLIC
    from flex_uda.active.execution import active_paths
    execution=active_paths();PUBLIC=execution.public
    p = ArgumentParser(description=__doc__); p.add_argument('--partial', action='store_true'); a = p.parse_args()
    from scripts.active_eval import summarize, bootstrap, sports418, offline_labels
    from scripts.presence_finish_a import references
    rows = []; status = []; artifacts = []; round_rows = []; video_rows = []; reliability_rows = []; costs = []
    cached_truth = {};cached_meta={};attribution_rows=[];calibration_rows=[];presence_decisions=[];sports_sensitivity=[]
    specifications = jobs()+[dict(pair=pair, method='GATE_REASSIGN_REF', rho=None) for pair in ('C1', 'C6')]
    for job in specifications:
        pair, method, rho = job['pair'], job['method'], job['rho']; target = PAIRS[pair][1]
        folder = RUN / 'active' / pair / method if method == 'GATE_REASSIGN_REF' else folder_for(job)
        if not (folder / 'predictions_seal.json').exists():
            status.append(dict(job, status='INCOMPLETE_NO_FULL_SEAL' if folder.exists() else 'NO_OUTPUT_YET'))
            continue
        seal = verify_predictions(folder)
        result = summarize(folder, target)
        result.update(pair=pair, rho=rho); rows.append(result)
        status.append(dict(job, status='SEALED_AND_OFFICIALLY_EVALUATED'))
        if target == 'SportsMOT':
            sensitivity=sports418(folder);subset=folder / 'sensitivity_418'
            subset_seal=verify_predictions(subset)
            sports_sensitivity.append(dict(pair=pair,method=method,rho=rho,model=seal['model'],
                episode_count=418,prediction_rows=subset_seal['prediction_rows'],
                **{k:sensitivity['metrics'][k] for k in ('HOTA','DetA','AssA','IDF1','IDSW')},
                main466_HOTA=result['HOTA'],GT_protocol='same v4 GT content, original418 sequence IDs only',
                teacher_calls_original_full466=result['teacher_calls'],
                source_full466_seal_sha256=file_sha(folder / 'predictions_seal.json'),
                subset_seal_sha256=file_sha(subset / 'predictions_seal.json'),path=str(subset)))
        if method!='GATE_REASSIGN_REF':
            from scripts.active_attribution import attribute
            attribution=attribute(folder,target)
            attribution_rows.append(dict(pair=pair,method=method,rho=rho,
                **{k:v for k,v in attribution.items() if k not in ('episodes','query_presence','method','model')},
                query_present_GT_recall=attribution['query_presence']['query_present']['adapted_GT_recall'],
                query_present_FP=attribution['query_presence']['query_present']['adapted_false_positive_rows'],
                query_empty_FP=attribution['query_presence']['query_empty']['adapted_false_positive_rows'],
                query_empty_nonempty_prediction_rate=attribution['query_presence']['query_empty']['adapted_nonempty_prediction_rate']))
            path=folder / 'offline_attribution.json'
            artifacts.append(dict(pair=pair,method=method,rho=rho,path=str(path),
                bytes=path.stat().st_size,sha256=file_sha(path)))
        attempts = [json.loads(path.read_text()) for path in (folder / 'worker_attempts').glob('*.json')]
        receipts = [json.loads(path.read_text()) for path in (folder / 'videos').glob('*/receipt.json')]
        calibration_rows+=calibration_calls(receipts,pair,method,rho)
        updates = sum(r['slow']['optimizer_updates'] for v in receipts for r in v.get('rounds', []))
        costs.append(dict(pair=pair, method=method, rho=rho, teacher_calls=result['teacher_calls'],
            teacher_new_forward_calls=seal.get('teacher', {}).get('new_calls', 0),
            teacher_attempted_requests=seal.get('teacher', {}).get('attempted_logical_calls'),
            teacher_cache_hits=seal.get('teacher', {}).get('cache_hits', 0),
            teacher_forward_GPU_hours=result['teacher_forward_GPU_hours'],
            teacher_equivalent_forward_GPU_hours=result['teacher_equivalent_forward_GPU_hours'],
            committed_teacher_equivalent_forward_GPU_hours=committed_teacher_equivalent_hours(receipts,seal.get('teacher',{}),method),
            LoRA_GPU_hours=result['LoRA_GPU_hours'], rescore_GPU_hours=result['rescore_GPU_hours'],
            assigned_worker_wall_hours=sum(r.get('wall_seconds', 0) for r in attempts)/3600,
            failed_worker_attempts=sum(r['status'] != 'RETURNED_SUCCESSFULLY' for r in attempts),
            worker_wall_is_upper_bound=any(r.get('wall_seconds_is_upper_bound',False) for r in attempts),
            student_peak_is_lower_bound=any(r.get('parent_peak_allocated_bytes_unavailable_due_to_signal',False) for r in attempts),
            interrupted_teacher_sessions=len(seal.get('teacher',{}).get('interrupted_sessions_accounting',[])),
            teacher_peak_is_lower_bound=any(s.get('teacher_peak_memory_is_lower_bound',False)
                for s in seal.get('teacher',{}).get('interrupted_sessions_accounting',[])),
            interrupted_teacher_accounting=json.dumps(seal.get('teacher',{}).get('interrupted_sessions_accounting',[]),sort_keys=True),
            wall_is_not_utilization_integral=True, optimizer_updates=updates,
            LoRA_trainable_parameters=24576 if method in ('FULL', 'RANDOM_ACQ', 'MINUS_PRES', 'MINUS_DS', 'MINUS_ROUTE', 'MINUS_FAST', 'RESET_LORA') else 0,
            peak_teacher_allocated_bytes=seal.get('teacher', {}).get('peak_allocated_bytes'),
            peak_student_allocated_bytes=max((r.get('parent_peak_allocated_bytes', 0) for r in attempts), default=0),
            covered_pairs=sum(r.get('covered_pairs', 0) for r in receipts) if method.startswith('YESTRACK') else None,
            all_pairs=sum(r.get('all_pairs', 0) for r in receipts) if method.startswith('YESTRACK') else None,
            delta_by_video=json.dumps({r['video']:r['delta'] for r in receipts if 'delta' in r}, sort_keys=True)))
        if target not in cached_truth:
            cached_meta[target],cached_truth[target],_=offline_labels(target)
        presence_decisions+=saved_presence_decisions(folder,cached_meta[target],cached_truth[target],pair,method,rho)
        rr, vv, dd = behavior_and_reliability(folder, target, pair, method, rho, cached_truth[target])
        round_rows += rr; video_rows += vv; reliability_rows += dd
        for path in [folder / 'predictions_seal.json', folder / 'results/pedestrian_detailed.csv',
                     folder / 'results/pedestrian_summary.txt', *sorted((folder / 'videos').glob('*/questions.json'))]:
            artifacts.append(dict(pair=pair, method=method, rho=rho, path=str(path), bytes=path.stat().st_size, sha256=file_sha(path)))
    incomplete = [r for r in status if r['status'] != 'SEALED_AND_OFFICIALLY_EVALUATED' and r['method'] != 'GATE_REASSIGN_REF']
    if incomplete and not a.partial: raise RuntimeError('Requested matrix incomplete; use --partial to report honestly without claiming completion')
    comparisons = {}
    for row in rows:
        if row['method'] != 'FULL': continue
        pair, rho, target = row['pair'], row['rho'], row['target']
        native = dict(references(target, PAIRS[pair][0]))['NATIVE']
        targets = [('NATIVE', native), ('PRES_TOPK', execution.stage_a / target / PAIRS[pair][0] / 'PRES_TOPK')]
        targets += [(r['method'], Path(r['path'])) for r in rows if r['pair'] == pair and r['method'] != 'FULL' and r['rho'] in (rho, None)]
        for method, folder in targets:
            comparisons[f'{pair}/rho{rho}/FULL_minus_{method}'] = bootstrap(row['path'], folder, target)
    write_table(PUBLIC / 'main.csv', rows); write_table(PUBLIC / 'matrix_status.csv', status)
    write_table(PUBLIC / 'costs.csv', costs); write_table(PUBLIC / 'behavior_round.csv', round_rows)
    write_table(PUBLIC / 'recovery_FP_attribution.csv',attribution_rows)
    write_table(PUBLIC / 'calibration_calls.csv',calibration_rows)
    write_table(PUBLIC / 'presence_decisions.csv',presence_decisions)
    confidence=final_confidence_summary(presence_decisions)
    write_table(PUBLIC / 'final_confidence.csv',confidence)
    write_table(PUBLIC / 'video_differences.csv',video_differences(rows,execution))
    write_table(PUBLIC / 'sports418.csv',sports_sensitivity)
    write_table(PUBLIC / 'behavior_video.csv', video_rows); write_table(PUBLIC / 'reliability_vs_GT.csv', reliability_rows)
    atomic_json(PUBLIC / 'bootstrap.json', dict(comparisons=comparisons, samples=2000, seed=0,
        cluster_unit='video; correlated expressions/frames kept together', KITTI='four videos, points only',
        uncertainty_scope='conditional on fixed predictions from the original ordered stream; no adaptation rerun under sampled video order, no seed/order uncertainty, and carried DS/LoRA state induces cross-video dependence',
        multiplicity='descriptive comparisons, no target-GT selection of variants'))
    atomic_json(PUBLIC / 'active_artifacts.json', dict(large_files_not_in_git=True, files=artifacts))
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        curves = ('FULL', 'YESTRACK_BUDGET_RANDOM', 'YESTRACK_BUDGET_PRIOR', 'UNIFORM_K')
        for ax, pair in zip(axes.flat, PAIRS):
            for method in curves:
                rr = sorted([r for r in rows if r['pair'] == pair and r['method'] == method], key=lambda r:r['rho'])
                if rr:
                    ax.plot([r['teacher_calls'] for r in rr], [r['HOTA'] for r in rr], 'o-', label=method)
                    for r in rr: ax.annotate(str(r['rho']), (r['teacher_calls'], r['HOTA']), fontsize=7)
            for method, marker in [('YESTRACK_FULL_D10', 'x'), ('GATE_REASSIGN_REF', 'D')]:
                for r in rows:
                    if r['pair'] == pair and r['method'] == method: ax.scatter(r['teacher_calls'], r['HOTA'], marker=marker, label=method)
            ax.set(title=pair+': '+PAIRS[pair][0]+' -> '+PAIRS[pair][1], xlabel='Teacher logical judgements', ylabel='Official HOTA')
            ax.grid(alpha=.2)
        handles, labels = [], []
        for ax in axes.flat:
            hh, ll = ax.get_legend_handles_labels()
            for h, label in zip(hh, ll):
                if label not in labels: handles.append(h); labels.append(label)
        fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8)
        fig.suptitle('PARTIAL: missing points lack a completed evaluation' if incomplete else 'All fixed budgets; no best-method selection')
        fig.tight_layout(rect=(0, .09, 1, .96)); fig.savefig(PUBLIC / 'teacher_calls_vs_HOTA.png', dpi=160); plt.close(fig)
        plot_adaptation_cost(rows,costs,PUBLIC)
    text = (PUBLIC / 'REPORT.md').read_text(); marker = '\n## 阶段B/C已封存结果'
    if marker in text: text = text[:text.index(marker)]
    table = ['|组合|方法|ρ|HOTA|DetA|AssA|IDF1|IDSW|行数|Empty行|教师次数|', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        table.append('|'+ '|'.join(str(r[k]) for k in ('pair','method','rho','HOTA','DetA','AssA','IDF1','IDSW','prediction_rows','query_empty_prediction_rows','teacher_calls'))+'|')
    text += marker+'\n\n状态：'+('完整108项运行矩阵完成。' if not incomplete else f'部分完成：{len(incomplete)}项尚未完成，不宣称完整方法通过。')+'\n\n'
    text += '\n'.join(table)+'\n\n'
    text += '### 最终信念置信诊断（GT仅离线，不是优化器训练样本标签）\n\n'
    text += '|组合|方法|ρ|GT正长轨迹对|高置信正TP/FP|高置信负FN|其中未问教师|GT正例被高置信错负比例|\n'
    text += '|---|---|---:|---:|---:|---:|---:|---:|\n'
    for r in confidence:
        rate=r['GTpositive_high_negative_rate'];rate='NA' if rate is None else f'{100*rate:.2f}%'
        text += '|'+ '|'.join(map(str,(r['pair'],r['method'],r['rho'],r['matched_candidate_positive_tracks_majority'],
            str(r['confident_positive_GT_TP'])+'/'+str(r['confident_positive_GT_FP']),r['confident_negative_GT_FN'],
            r['confident_negative_GT_FN_unqueried'],rate)))+'|\n'
    text += ('\n长轨迹限n>=3，GT标签沿用每帧IoU0.5匹配后>=50%观测为正；阈值仍是原p>.95/p<.05。'
        '这是各视频结束时保存的belief，不是优化器每一步实际用过的标签，不能据此单独断言slow造成错误。'
        '空目标多时总体错负率会掩盖GT正类受损，同时报告两个分母；不把高置信数量当正确证据或独立样本显著性。'
        '完整计数见final_confidence.csv与presence_decisions.csv，因果组件比较仍依照预注册消融和视频bootstrap。\n\n')
    text += ('固定候选/原ID/原框不变；GT仅在封存后的独立评测中使用。Dance和Sports按视频聚类2000次seed0；KITTI仅点估计。'
        'C1 Gate-Reassign逐行复现旧参照，C6是同门控的新CPU派生参照；二者沿用同一批29410次历史判断，不将其成本解释成零成本独立系统。\n\n'
        '成本见costs.csv：真实新增forward、缓存复用、等价独立forward及LoRA/重打分分别报告；worker wall包含恢复失败尝试，'
        '不是GPU利用率积分，不能与同驻教师进程wall重复相加。源Head训练、前端与特征预计算另见阶段A。\n\n'
        '外部中断单独记录：worker wall如只有观测区间则标upper bound；未收尾教师会话的原始缓存/已完成视频可恢复开销与未知缓存请求分别注明。'
        'LoRA/重打分相位求和以最终有效checkpoint路径为准，未完成视频的浪费另见中断记录，不能默认为零或冒充精确计时。\n\n'
        'teacher_equivalent_forward_GPU_hours保留含失败重试的会话口径；新增committed_teacher_equivalent_forward_GPU_hours只计最终有效问题一次，'
        '用于计算量图，并与有效checkpoint路径的LoRA/重打分配对。全部真实新增forward及重试仍在原计时/worker账目中，不从实际支出抹去。\n\n'
        'adaptation_cost_vs_HOTA.png另画独立等价教师forward加LoRA/重打分的相位耗时，避免把同一教师预算等同总计算预算；'
        '它不包含共享源端/冻结打分/前端准备和载入开销，不能解释成完整系统总成本或GPU利用率积分。\n\n'
        '行为见behavior_round/video.csv。置信不等于正确；分别标记先验即高置信、从未问过但高置信，以及教师/最终新增确认。'
        '达到固定确认数的调用是轮边界上界，不据此单独宣称越问越省。DS估计与GT后验诊断见reliability_vs_GT.csv，'
        '同时给出提问帧与50%轨迹多数标签；身份先验以轨迹标签为主要参照，重复视图/帧不是独立样本。\n\n'
        'recovery_FP_attribution.csv用独立IoU>=0.5逐帧匹配比较Native，区分新增行FP与原有行匹配重分配的FP变化，'
        '只作离线归因、不代替官方TrackEval，也不反馈在线阈值。\n\n'
        'calibration_calls.csv逐调用保留MAP更新/SKIP、正负教师支持、排除的未问伪标签、b/T/beta和目标函数变化；'
        '区分“有证据才更新”与“没有学到存在性”的情况，不用数值稳定性代替精度。\n\n'
        'presence_decisions.csv仅在完整封存后，将保存的存在/缺席/矛盾标志与官方GT及可检索候选对照，'
        '同时区分p>0.5的轨迹、p>0.95的高置信伪正例和实际输出轨迹；帧级期望数量不等于视频内不同轨迹数。'
        '这是事后诊断，不据此调整当前方法。\n\n'
        'video_differences.csv按视频独立聚合官方HOTA，列出FULL相对每个对照的收益分布；'
        '其简单平均不是全数据集TP加权HOTA差值，不混为一谈。\n\n'
        'sports418.csv明确列出同一v4 GT内容限制原418条ID的敏感性数值与封存SHA；'
        '不是旧冲突模板，不用其结果回选方法，成本仍来自原完整466预测。\n\n'
        '完整机器表、bootstrap及大文件路径/SHA均在本目录；Sports使用v4全466和相同v4内容的旧418 ID敏感性。'
        '没有根据目标GT调整阈值、种子或选择最优变体。\n')
    (PUBLIC / 'REPORT.md').write_text(text)
    from scripts.active_archive import package
    packaged=package(execution,rows)
    print(json.dumps(dict(status='FULL_MATRIX_COMPLETE' if not incomplete else 'PARTIAL_REPORTED_NOT_COMPLETE',
        evaluated=len(rows), pending=len(incomplete), comparisons=len(comparisons),
        logs_manifest=str(PUBLIC / 'logs.json'),raw_teacher_keys=packaged['raw_teacher_response_keys'])), flush=True)


if __name__ == '__main__': main()
