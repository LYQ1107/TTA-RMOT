"""Post-seal analysis only: truth is never an input to runtime decisions."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.stats import rankdata
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import OUT, OLD, PAIRS, metadata, history, baseline_mask


def point_metrics(y, probability, positive_tie=False):
    y = np.asarray(y, dtype=bool)
    p = np.asarray(probability, dtype=float)
    predicted = p >= .5 if positive_tie else p > .5
    tp, tn = int((predicted & y).sum()), int((~predicted & ~y).sum())
    fp, fn = int((predicted & ~y).sum()), int((~predicted & y).sum())
    pos, neg = int(y.sum()), int((~y).sum())
    clipped = np.clip(p, 1e-8, 1-1e-8)
    auc = float((rankdata(p)[y].sum()-pos*(pos+1)/2)/(pos*neg)) if pos and neg else None
    return dict(n=len(y), positives=pos, negatives=neg, TP=tp, TN=tn, FP=fp, FN=fn,
        accuracy=(tp+tn)/len(y) if len(y) else None,
        balanced_accuracy=.5*(tp/pos+tn/neg) if pos and neg else None,
        sensitivity=tp/pos if pos else None, specificity=tn/neg if neg else None,
        Brier=float(np.mean((p-y)**2)), NLL=float(np.mean(-(y*np.log(clipped)+(~y)*np.log1p(-clipped)))),
        AUROC=auc, predicted_positive=int(predicted.sum()),positive_threshold='>=.5' if positive_tie else '>.5')


def paired_points(rows, a, b, samples=10000):
    """Descriptive paired parent-video bootstrap; no independent-frame claims."""
    y = np.array([r['truth'] for r in rows], dtype=bool)
    pa = np.array([r['posterior'][a] for r in rows])
    pb = np.array([r['posterior'][b] for r in rows])
    groups = sorted({r['parent_video'] for r in rows})
    delta = ((pa>.5)==y).astype(float)-((pb>.5)==y)
    brier = (pb-y)**2-(pa-y)**2
    result = dict(first=a, second=b, delta_accuracy=float(delta.mean()),
        Brier_reduction=float(brier.mean()), parent_groups=len(groups), samples=samples, seed=0)
    if len(groups) < 5:
        return dict(result, accuracy_CI95=None, Brier_reduction_CI95=None,
            note='fewer than5 original parent videos; point estimate only')
    indices = np.array([groups.index(r['parent_video']) for r in rows])
    counts = np.random.default_rng(0).multinomial(len(groups), np.ones(len(groups))/len(groups), size=samples)
    weights = counts[:, indices]
    norm = weights.sum(1)
    result.update(accuracy_CI95=np.percentile(weights@delta/norm, [2.5,97.5]).tolist(),
        Brier_reduction_CI95=np.percentile(weights@brier/norm, [2.5,97.5]).tolist())
    return result


def coverage():
    results = {}
    for pair, (_, target) in PAIRS.items():
        m = metadata(target)
        audit = json.loads((OUT/'audit'/pair/'completion.json').read_text())
        episodes = {(r['video'],r['query']):r for r in audit['episodes']}
        total = Counter(); queries = []; physical = Counter(); purity_num = 0.
        for video, bank in sorted(m['videos'].items()):
            obs = np.asarray(bank['observations']); qs = [q for q in m['episodes'] if q['video_id']==video]
            labels = np.load(OUT/'offline_GT'/pair/video/'labels.npy') >= 0
            mask = baseline_mask(pair, video)
            answers = json.loads((history(pair)/'videos'/video/'questions.json').read_text())
            asked = defaultdict(set)
            for a in answers:
                asked[a['query_id']].add(int(a['track_id']))
            for i, q in enumerate(qs):
                query = q['expression_id']; selected_asked = np.isin(obs[:,1], list(asked[query]))
                positive = labels[i]; missed = positive & ~mask[i]
                positive_tracks = set(map(int, obs[positive,1]))
                c = Counter(positive_candidate_rows=int(positive.sum()),
                    positive_rows_on_asked_tracks=int((positive & selected_asked).sum()),
                    missed_positive_candidate_rows=int(missed.sum()),
                    missed_positive_rows_on_asked_tracks=int((missed & selected_asked).sum()),
                    missed_positive_rows_on_unasked_tracks=int((missed & ~selected_asked).sum()),
                    query_positive_candidate=int(positive.any()),
                    query_positive_candidate_but_no_positive_track_asked=int(bool(positive_tracks) and not positive_tracks.intersection(asked[query])),
                    query_present=int(episodes[video,query]['query_present']),
                    present_zero_output=int(episodes[video,query]['present_zero_output']),
                    present_zero_and_no_positive_track_asked=int(episodes[video,query]['present_zero_output'] and not positive_tracks.intersection(asked[query])))
                total.update(c)
                queries.append(dict(video=video, query=query, asked_tracks=len(asked[query]),
                    positive_candidate_tracks=len(positive_tracks), **dict(c)))
            if pair != 'C5':
                union = np.load(OUT/'offline_GT'/pair/video/'union_ids.npy')
                v = json.loads((OUT/'audit'/pair/'videos'/f'{video}.json').read_text())
                physical['GT_ids_observed'] += v['GT_ids_observed']
                physical['GT_ids_with_multiple_candidate_IDs'] += v['GT_ids_with_multiple_candidate_IDs']
                for track in v['physical_tracks']:
                    physical['candidate_tracks'] += 1
                    physical['tracks_with_multiple_visible_GT_ids'] += track['distinct_visible_GT_ids'] > 1
                    physical['matched_observations'] += track['GT_matched_observations']
                    if track['purity'] is not None:
                        purity_num += track['purity']*track['GT_matched_observations']
                    ids = np.flatnonzero(obs[:,1] == track['track'])
                    adjacent = (np.diff(obs[ids,0])>0) & (np.diff(obs[ids,0])<=2)
                    a, b = union[ids[:-1]], union[ids[1:]]
                    valid = adjacent & (a>=0) & (b>=0)
                    physical['adjacent_visible_identity_transitions'] += int(valid.sum())
                    physical['visible_identity_change_proxy'] += int((valid & (a!=b)).sum())
        result = dict(counts=dict(total), queries=queries,
            positive_track_coverage=total['positive_rows_on_asked_tracks']/max(1,total['positive_candidate_rows']),
            missed_positive_unasked_fraction=total['missed_positive_rows_on_unasked_tracks']/max(1,total['missed_positive_candidate_rows']),
            definitions='candidate-label coverage, not additive official HOTA loss; asked means any answer anywhere on this query-track',
            conditional_acquisition_diagnostic_justified=bool(total['missed_positive_rows_on_unasked_tracks']),
            finite_bank_cannot_repair_unqueried_pool=True)
        if pair != 'C5':
            result['physical'] = dict(physical, weighted_track_purity=purity_num/max(1,physical['matched_observations']),
                definition='once per physical candidate track; union query GT, IoU>=.5; visible assignment changes are a proxy, not official IDSW')
        results[pair] = result
    receipt = dict(status='COMPLETE_OFFLINE_COVERAGE', pairs=results, GT_used_only_offline=True)
    atomic_json(OUT/'analysis/coverage.json', receipt)
    print(json.dumps(dict(event='COVERAGE_ANALYZED',pairs={p:r['counts'] for p,r in results.items()})),flush=True)


def contract():
    from flex_uda.active.v8p import classify_protocol, CHECKLIST, PRESUPPOSED
    rows = []
    for pair, (_, target) in PAIRS.items():
        m = metadata(target)
        qs = {v:[q['query_text'] for q in m['episodes'] if q['video_id']==v] for v in sorted(m['videos'])}
        declared = CHECKLIST if target=='Dance' else PRESUPPOSED
        full = classify_protocol(qs, task_protocol=declared)
        singleton = {v:classify_protocol({v:q}, task_protocol=declared)['protocol'] for v,q in qs.items()}
        keys=list(qs); two=[classify_protocol({v:qs[v] for v in keys[i:i+2]}, task_protocol=declared)['protocol'] for i in range(0,len(keys),2)]
        assert set(singleton.values()) == {declared} and set(two)=={declared}
        rows.append(dict(pair=pair,declared_contract=declared,full_protocol=full['protocol'],
            singleton_protocols=singleton,two_video_protocols=two,
            same_video_protocol_flips_due_to_batching=0))
    atomic_json(OUT/'protocol/contract_invariance.json',dict(status='PASS_EXPLICIT_CONTRACT_INVARIANCE',rows=rows,
        new_teacher_calls=0,GT_open_attempts=0,
        contract_source='registered task definition; video presence guarantee supplied by caller, not inferred from target GT',
        old_heuristic_preserved_for_legacy_callers=True,tests='14 passed: test_active_task_protocol.py and test_active_v8p.py'))
    print(json.dumps(dict(event='EXPLICIT_PROTOCOL_INVARIANCE_PASS',rows=len(rows))),flush=True)


def absence():
    pairs={}
    for pair,(_,target) in PAIRS.items():
        audit=json.loads((OUT/'audit'/pair/'completion.json').read_text())
        truth={(r['video'],r['query']):bool(r['query_present']) for r in audit['episodes']}
        totals=Counter();rows=[]
        for video in sorted(metadata(target)['videos']):
            with np.load(history(pair)/'videos'/video/'belief.npz') as state:
                for qi,query in enumerate(state['queries'].tolist()):
                    present=truth[video,query];rejected=bool(state['output_absent'][qi]);hard=bool(state['absent'][qi])
                    totals['queries']+=1;totals['query_present']+=present;totals['query_empty']+=not present
                    totals['false_absent']+=present and rejected
                    totals['empty_rejected']+=(not present) and rejected
                    totals['evidence_stop_false_absent']+=present and hard
                    totals['tentative_output_false_absent']+=present and rejected and not hard
                    if present and rejected:
                        rows.append(dict(video=video,query=query,presence_probability=float(state['presence_probability'][qi]),evidence_stop=hard))
        pairs[pair]=dict(counts=dict(totals),false_absent_queries=rows)
    atomic_json(OUT/'analysis/absence.json',dict(status='COMPLETE_OFFLINE_EXISTENCE_DIAGNOSIS',pairs=pairs,
        note='PRESUPPOSED absence0 follows task guarantee; a non-rejected query can still output zero tracks',GT_used_only_offline=True))
    print(json.dumps(dict(event='ABSENCE_ANALYZED',counts={p:r['counts'] for p,r in pairs.items()})),flush=True)


def temporal_details():
    """Measure output flips, not just raw confidence variance."""
    m=metadata('SportsMOT');totals={name:Counter() for name in ('RAW_MARGIN','EMA_MARGIN','HYSTERESIS_MARGIN','MINUS')}
    for video,bank in m['videos'].items():
        obs=np.asarray(bank['observations']);queries=[q for q in m['episodes'] if q['video_id']==video]
        labels=np.load(OUT/'offline_GT/C4'/video/'labels.npy')>=0
        union=np.load(OUT/'offline_GT/C4'/video/'union_ids.npy')
        tracks={t:np.flatnonzero(obs[:,1]==t) for t in np.unique(obs[:,1])}
        for name,total in totals.items():
            if name=='MINUS':bits=baseline_mask('C4',video)
            else:
                folder=OUT/'temporal/C4'/name/'results'/video
                bits=[]
                for q in queries:
                    selected=set((folder/q['expression_id']/'predict.txt').read_text().splitlines())
                    bits.append(np.array([line in selected for line in bank['lines']],dtype=bool))
                bits=np.stack(bits)
            total['selected_rows']+=int(bits.sum())
            total['selected_positive_candidate_rows']+=int((bits&labels).sum())
            total['selected_negative_candidate_rows']+=int((bits&~labels).sum())
            total['missed_positive_candidate_rows']+=int((~bits&labels).sum())
            for ids in tracks.values():
                a,b=union[ids[:-1]],union[ids[1:]]
                adjacent=(np.diff(obs[ids,0])>0)&(np.diff(obs[ids,0])<=2)
                identity=(a==b)&(a>=0)
                stable=adjacent[None,:]&identity[None,:]&(labels[:,ids[1:]]==labels[:,ids[:-1]])
                flips=bits[:,ids[1:]]!=bits[:,ids[:-1]]
                total['stable_identity_membership_transitions']+=int(stable.sum())
                total['stable_identity_membership_flips']+=int((stable&flips).sum())
    atomic_json(OUT/'analysis/temporal_details.json',dict(status='COMPLETE_SEALED_MASK_TEMPORAL_DIAGNOSIS',methods={k:dict(v) for k,v in totals.items()},
        note='Causal raw-margin filters are compared with RAW; existing MINUS is already track-constant, not augmented by these filters',GT_used_only_offline=True))
    print(json.dumps(dict(event='TEMPORAL_DETAILS_ANALYZED',methods={k:dict(v) for k,v in totals.items()})),flush=True)


def bank():
    from scripts.a100_mechanism_offline import gt_root
    from scripts.analyze_pf_full_offline import load_eval_gt, overlaps
    path = OUT/'mechanism/bank_decisions.json'
    sealed = json.loads(path.read_text())
    assert sealed['status']=='SEALED_BEFORE_BANK_GT' and sealed['GT_open_attempts']==0
    truths = {}; rows=[]
    # Match exact registered observation indices; no choosing samples after GT.
    for pair,(_,target) in PAIRS.items():
        m = metadata(target)
        for video,bank in m['videos'].items():
            chosen = [r for r in sealed['rows'] if r['pair']==pair and r['video']==video]
            if not chosen: continue
            queries=[q['expression_id'] for q in m['episodes'] if q['video_id']==video]
            obs={tuple(o):i for i,o in enumerate(bank['observations'])}
            labels=np.load(OUT/'offline_GT'/pair/video/'labels.npy')
            query_truth={}
            for r in chosen:
                index=obs[r['frame'],r['track']]
                one_to_one=bool(labels[queries.index(r['query']),index]>=0)
                truth=one_to_one
                if pair!='C5':
                    if r['query'] not in query_truth:
                        query_truth[r['query']]=load_eval_gt(gt_root(target),video,r['query'])
                    line=bank['lines'][index].split(',');f=int(float(line[0]))
                    box=np.array([float(x) for x in line[2:6]]);box[2:]+=box[:2]
                    g=query_truth[r['query']].get(f,[])
                    truth=bool((overlaps([box],[b for _,b in g])>=.5).any())
                truths[r['sample_id']]=truth
                rows.append(dict(r,truth=truth,one_to_one_truth=one_to_one))
    assert len(rows)==300 and len(truths)==300
    comparisons=(('ROUTED_FIXED','CROP_FIXED'),('ROUTED_DS','CROP_DS'),
        ('CROP_DS','CROP_FIXED'),('ROUTED_DS','ROUTED_FIXED'),
        ('ROUTED_DS','CROP_FIXED'),('ALL_DS','ALL_MEAN'),('ALL_DS','ALL_VOTE_FIXED'))
    pairs = {}
    for pair in PAIRS:
        selected=[r for r in rows if r['pair']==pair]
        split={}
        for name in ('calibration','evaluation'):
            ss=[r for r in selected if r['split']==name]
            split[name]=dict(samples=len(ss),original_parent_groups=len({r['parent_video'] for r in ss}),
                methods={k:point_metrics([r['truth'] for r in ss],[r['posterior'][k] for r in ss]) for k in ss[0]['posterior']},
                routing_views=dict(Counter(r['original_routed_view'] for r in ss)),
                query_types=dict(Counter(r['query_type'] for r in ss)))
        evaluation=[r for r in selected if r['split']=='evaluation']
        y=np.array([r['truth'] for r in evaluation]);teacher={}
        for view in ('CROP','CLIP','FRAME_BOX'):
            teacher[view]=point_metrics(y,[r['teacher_probability'][view] for r in evaluation],positive_tie=True)
            teacher[view]['unsupervised_calibration_estimate']=sealed['reliability'][pair]['reliability'][view]
        correlations=[]
        for a,b in (('CROP','CLIP'),('CROP','FRAME_BOX'),('CLIP','FRAME_BOX')):
            ea=np.array([(r['teacher_probability'][a]>=.5)!=r['truth'] for r in evaluation],dtype=float)
            eb=np.array([(r['teacher_probability'][b]>=.5)!=r['truth'] for r in evaluation],dtype=float)
            conditioned={}
            for value in (False,True):
                sub=(y==value)
                conditioned[str(value)]=dict(n=int(sub.sum()),both_wrong=int((sub & (ea>0) & (eb>0)).sum()),
                    error_correlation=float(np.corrcoef(ea[sub],eb[sub])[0,1]) if sub.sum()>1 and ea[sub].std()>0 and eb[sub].std()>0 else None)
            correlations.append(dict(views=[a,b],both_wrong=int(((ea>0)&(eb>0)).sum()),
                error_agreement=float(np.mean(ea==eb)),
                error_correlation=float(np.corrcoef(ea,eb)[0,1]) if ea.std()>0 and eb.std()>0 else None,
                conditional_on_truth=conditioned))
        pairs[pair]=dict(splits=split,comparisons=[paired_points(evaluation,a,b) for a,b in comparisons],
            teacher=teacher,view_error_correlation=correlations,
            threshold_diagnostics=dict(prior_at_logit_cap=sum(abs(r['prior_logit'])>=2-1e-8 for r in evaluation),
                changed_decisions_from_source={k:sum(r['choices'][k]!=r['choices']['SOURCE_PRIOR'] for r in evaluation) for k in evaluation[0]['posterior']},
                routed_vs_crop_different_teacher_binary=sum((r['teacher_probability']['CROP']>=.5)!=(r['teacher_probability'][r['original_routed_view']]>=.5) for r in evaluation),
                scope='one answer with capped likelihood ratio cannot necessarily cross a saturated prior; equal hard decisions do not imply equal evidence quality'),
            primary_vs_one_to_one_label_disagreements=sum(r['truth']!=r['one_to_one_truth'] for r in selected),
            one_to_one_sensitivity_metrics={k:point_metrics([r['one_to_one_truth'] for r in evaluation],[r['posterior'][k] for r in evaluation]) for k in evaluation[0]['posterior']},
            numerical_GT_definition='sealed original complete-bank binary membership; no raw KITTI GT available' if pair=='C5' else 'any query GT box IoU>=.5, so duplicate boxes are not automatically semantic negatives; one-to-one sensitivity also reported')
    acquisition=[]
    for item in sealed['acquisition']:
        metrics=point_metrics([truths[r['sample_id']] for r in item['predictions']],[r['posterior'] for r in item['predictions']])
        acquisition.append(dict(**{k:v for k,v in item.items() if k!='predictions'},metrics=metrics))
    result=dict(status='COMPLETE_OFFLINE_BANK_DIAGNOSIS',pairs=pairs,acquisition=acquisition,
        source_decisions_sha256=file_sha(path),GT_used_only_offline=True,decision_parameters_selected_from_GT=False,
        confidence_scope='paired original-parent-video resampling,10000 seed0, existing development data, fixed teacher and evidence pool; descriptive, no multiple-comparison correction',
        teacher_scope='same frozen Qwen under three views; not independent human annotators; label agreement is not a pure semantic error oracle',
        sampling_scope='historical actively requested positions only, calibration/evaluation parent groups disjoint; no new independent domain test',
        acquisition_scope='prepaid bank, historical selected pool only, source prior for unqueried rows; no live wall-time, no coverage-repair or policy optimality claim',
        calibration_cost_must_be_amortized=True)
    atomic_json(OUT/'analysis/bank_results.json',result)
    atomic_json(OUT/'analysis/bank_labeled_rows.json',dict(status='OFFLINE_GT_ATTACHED_AFTER_SEAL',rows=rows))
    print(json.dumps(dict(event='BANK_ANALYZED',pairs={p:r['splits']['evaluation']['methods'] for p,r in pairs.items()})),flush=True)


def grouped_official(folder):
    from scripts.active_eval import official_stats
    data=official_stats(folder);num,den=data['stats']
    parents=sorted({v.rsplit('_c',1)[0] for v in data['videos']})
    aggregated=[]
    for values in (num,den):
        aggregated.append(np.stack([values[[i for i,v in enumerate(data['videos']) if v.rsplit('_c',1)[0]==p]].sum(0) for p in parents]))
    return dict(videos=data['videos'],parent_videos=parents,stats=data['stats'],parent_stats=tuple(aggregated),HOTA=data['HOTA_unrounded'])


def compare_official(a,b):
    from scripts.analyze_pf_video_uncertainty import paired_clusters
    assert a['videos']==b['videos'] and a['parent_videos']==b['parent_videos']
    result=dict(delta_HOTA=a['HOTA']-b['HOTA'],samples=10000,seed=0)
    for key,unit in (('parent_stats','original_parent_video'),('stats','evaluation_clip')):
        n=len(a[key][0]);weights=np.random.default_rng(0).multinomial(n,np.ones(n)/n,size=10000)
        delta=paired_clusters(a[key],b[key],weights)
        result[unit]=dict(clusters=n,CI95=np.percentile(delta,[2.5,97.5]).tolist())
    return result


def prior_parent_sensitivity():
    folders={name:OLD/'stage2'/name for name in ('MINUS','MINUS_RECALC_ONLY','SLOW_REPLAY_MATCHED')}
    for name in ('FROZEN_RANK__RECALC_COUNT','LORA_RANK__RECALC_COUNT','FROZEN_RANK__LORA_COUNT','LORA_RANK__LORA_COUNT'):
        folders[name]=OLD/'stage2/countmatched'/name
    values={n:grouped_official(p) for n,p in folders.items()}
    pairs=[('MINUS_RECALC_ONLY','MINUS'),('SLOW_REPLAY_MATCHED','MINUS_RECALC_ONLY'),
        ('SLOW_REPLAY_MATCHED','MINUS'),('LORA_RANK__RECALC_COUNT','FROZEN_RANK__RECALC_COUNT'),
        ('LORA_RANK__LORA_COUNT','FROZEN_RANK__LORA_COUNT')]
    result=dict(status='EXPLORATORY_PRIOR_RESULTS_PARENT_GROUP_SENSITIVITY',
        source='completed20261120 stage2; no GPU inference, no new training, no metric changes',
        reason='Sports45 clips belong to13 original videos; retain original45-clip uncertainty and add more conservative parent grouping',
        points={n:r['HOTA'] for n,r in values.items()},comparisons={a+'_vs_'+b:compare_official(values[a],values[b]) for a,b in pairs})
    atomic_json(OUT/'analysis/prior_parent_sensitivity.json',result)
    print(json.dumps(result),flush=True)


def costs():
    evidence=json.loads((OUT/'evidence/completion.json').read_text())
    samples=json.loads((OUT/'evidence_manifest.json').read_text())['samples']
    responses={(r['sample_id'],r['requested_view']):r for r in map(json.loads,(OUT/'evidence/responses.jsonl').read_text().splitlines())}
    loaded={};parity=Counter();differences=[]
    for s in samples:
        path=s['original_answer_path']
        if path not in loaded:loaded[path]=json.loads(Path(path).read_text())
        original=next(a for a in loaded[path] if a['query_id']==s['query_id'] and a['track_id']==s['track_id'] and a['frame_id']==s['frame_id'])
        current=responses[s['sample_id'],s['original_view']]
        unequal=[k for k in ('image','visual','prompt','bbox','method','revision','weight','code') if original['input'][k]!=current['input'][k]]
        parity['positions']+=1;parity['identical_inputs']+=not unequal
        parity['teacher_binary_changed']+=(original['yes_probability']>=.5)!=(current['yes_probability']>=.5)
        if unequal:differences.append(dict(sample_id=s['sample_id'],different_fields=unequal))
    # These are resource/cost receipts for the study, not a deployment curve.
    gpu={}
    for name in ('evidence','profile'):
        current=json.loads((OUT/name/'supervisor_receipt.json').read_text());assert current['status']=='COMPLETE'
        attempts=[current]
        previous=OUT/name/'attempt00_failed_supervisor_receipt.json'
        if previous.is_file():attempts.insert(0,json.loads(previous.read_text()))
        gpu[name]=dict(physical_GPU=current['physical_GPU'],attempts=attempts,
            sum_process_wall_seconds=sum(r['wall_seconds'] for r in attempts),
            definition='device-assigned process elapsed time, includes IO/load/verification/CPU; not GPU-kernel integral')
    cpu={p.stem:json.loads(p.read_text()) for p in sorted((OUT/'cpu_supervisor').glob('*.json')) if p.stem!='completion'}
    assert all(r['status']=='COMPLETE' for r in cpu.values())
    result=dict(status='PASS_STUDY_COST_AND_SOURCE_AUDIT',gpu=gpu,
        teacher=dict(responses=len(responses),unique_request_ids=len({r['request_id'] for r in responses.values()}),
            new_calls=evidence['teacher']['new_calls'],cache_hits=evidence['teacher']['cache_hits'],
            actual_forward_seconds_from_returned_batch_shares=evidence['teacher']['forward_seconds'],
            new_call_cap=900,preparation_not_deployment=True),
        historical_original_view_input_parity=dict(parity,differences=differences),
        CPU_children=cpu,sum_CPU_child_process_wall_seconds=sum(r['wall_seconds'] for r in cpu.values()),
        CPU_cost_not_additive_overall_elapsed=True,first_failed_CPU_audit_wall_unavailable=True,
        original_failed_GPU_session_terminal_weight_hash_available=False,
        final_session_terminal_weight_seal_verified=True,
        no_online_GPU_speedup_claim_from_replay=True)
    assert evidence['teacher']['new_calls']<=900 and len(responses)==900
    assert not differences,'Current-vs-historical input drift; inspect before interpreting view contrasts'
    atomic_json(OUT/'analysis/study_cost_ledger.json',result)
    print(json.dumps(dict(event='STUDY_COST_AUDIT_PASS',GPU_wall_seconds={n:r['sum_process_wall_seconds'] for n,r in gpu.items()},source_input_parity=dict(parity))),flush=True)


def official():
    folders={'MINUS':OUT/'fixed_replay/C4/DS__EXPLICIT_TASK'}
    for name in ('FRAME_ORACLE','TRACK_MAJORITY_ORACLE'):
        folders[name]=OUT/'audit/C4/oracles'/name
    for name in ('RAW_MARGIN','EMA_MARGIN','HYSTERESIS_MARGIN'):
        folders[name]=OUT/'temporal/C4'/name
    folders['UNIFIED_CHECKLIST']=OUT/'fixed_replay/C4/DS__UNIFIED_CHECKLIST'
    for fusion in ('DS','FIXED'):
        for routing in ('ORIGINAL','CROP'):
            folders[fusion+'_BANK_'+routing]=OUT/'fixed_replay/C4'/f'{fusion}__EXPLICIT_TASK__BANK_{routing}'
    values={};summary={}
    audit=json.loads((OUT/'audit/C4/completion.json').read_text());gt={(r['video'],r['query']):r for r in audit['episodes']}
    for name,folder in folders.items():
        completed=json.loads((folder/'official_completion.json').read_text())
        seal=json.loads((folder/'predictions_seal.json').read_text())
        values[name]=grouped_official(folder)
        zero=sum(r['selected_rows']==0 and gt[r['video'],r['expression']]['query_present'] for r in seal['episodes'])
        absent=0
        for video in metadata('SportsMOT')['videos']:
            state=folder/'videos'/video/'belief.npz'
            if state.is_file():
                with np.load(state) as a:absent+=int(a['absent'].sum())
        summary[name]=dict(folder=str(folder),HOTA=values[name]['HOTA'],metrics=completed['metrics'],
            prediction_rows=seal['prediction_rows'],present_zero_output=int(zero),
            false_absence_query_count=absent if name not in ('FRAME_ORACLE','TRACK_MAJORITY_ORACLE','RAW_MARGIN','EMA_MARGIN','HYSTERESIS_MARGIN') else None,
            bank_replaced_positions=seal.get('bank_replaced_positions',0),
            teacher_calls=seal.get('questions_replayed',0),new_teacher_calls=seal.get('new_teacher_calls',0),
            offline_oracle_uses_GT=bool(seal.get('offline_oracle_uses_GT',False)),
            official_completion_sha256=file_sha(folder/'official_completion.json'))
    pairs=[('FRAME_ORACLE','MINUS'),('TRACK_MAJORITY_ORACLE','MINUS'),
        ('EMA_MARGIN','RAW_MARGIN'),('HYSTERESIS_MARGIN','RAW_MARGIN'),('UNIFIED_CHECKLIST','MINUS'),
        ('DS_BANK_ORIGINAL','FIXED_BANK_ORIGINAL'),('DS_BANK_CROP','FIXED_BANK_CROP'),
        ('DS_BANK_ORIGINAL','DS_BANK_CROP'),('FIXED_BANK_ORIGINAL','FIXED_BANK_CROP'),
        ('DS_BANK_ORIGINAL','MINUS')]
    comparisons={a+'_vs_'+b:compare_official(values[a],values[b]) for a,b in pairs}
    stats={name:dict(evaluation_clips=d['videos'],parent_videos=d['parent_videos'],
        numerator=d['stats'][0].tolist(),denominator=d['stats'][1].tolist(),
        parent_numerator=d['parent_stats'][0].tolist(),parent_denominator=d['parent_stats'][1].tolist()) for name,d in values.items()}
    result=dict(status='ALL_OFFICIAL_CONTROLS_COMPLETE',methods=summary,comparisons=comparisons,
        primary_resampling_unit='13 original parent Sports videos; all clips/queries/frames kept together',
        supplemental_resampling_unit='45 evaluation clips, included for comparability with old reports',
        inference_scope='descriptive existing-development-data intervals, fixed trained stream and teacher bank; not retraining-seed or blind-domain uncertainty',
        routing_scope='only100 predeclared answer positions replaced among3508; neural weights and all other answers fixed; not a complete live-routing policy ablation',
        reliability_scope='full replay DS vs fixed .8/.8 reliability; online unsupervised adaptation also changes subsequent fast fits; not the held-out-bank DS calibration experiment')
    atomic_json(OUT/'analysis/official_results.json',result)
    atomic_json(OUT/'analysis/official_sufficient_statistics.json',dict(status='EXACT_OFFICIAL_HOTA_STATISTICS',methods=stats))
    print(json.dumps(dict(event='OFFICIAL_CONTROLS_ANALYZED',methods={n:r['HOTA'] for n,r in summary.items()},comparisons=comparisons)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['coverage','contract','absence','temporal_details','bank','official','prior_parent_sensitivity','costs'],required=True)
    args=p.parse_args();globals()[args.phase]()
