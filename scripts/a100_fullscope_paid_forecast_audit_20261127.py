"""Read-only prequential teacher-risk audit; no GT, models or RMOT outputs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path('/data3/liuyeqiang/FlexHook-TTA')
SUP = ROOT / 'a100_runs/visual_classical_20261121/supervisor'
BASE = ROOT / 'a100_runs/visual_classical_20261121/six_direction_20261122/fullscope_visual_six_20261126'
OWN = SUP / 'paid_forecast_audit_registration_20261127.json'
OUT = SUP / 'fullscope_paid_forecast_audit_20261127.json'
PLAN_SHA = 'd159ed56af4d614edd68c991ab752f6341074752c648650cd61d336d5a68ba12'
FORECASTS = ('student_probability_BEFORE_answer', 'observer_mean_BEFORE_answer',
             'observer_residual_proxy_BEFORE_answer')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, doc):
    assert not path.exists(), path
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n')


def identities(y, s, r, v):
    actual = (y-s)**2
    residual = (y-r)**2
    disagreement = (r-s)**2
    cross = 2*(y-r)*(r-s)
    proxy = v+disagreement
    assert np.max(np.abs(actual-residual-disagreement-cross)) < 2e-14
    assert np.max(np.abs(actual-proxy-(residual-v+cross))) < 2e-14
    return actual, residual, disagreement, cross, proxy


def check_math():
    y = np.array([0., .4, 1.]); s = np.array([.9, .4, .1])
    r = np.array([.2, .7, .9]); v = np.array([.01, .3, .02])
    a, e, d, c, p = identities(y, s, r, v)
    assert np.allclose(a, [.81, 0., .81])
    assert np.allclose(a-p, e-v+c)
    return dict(status='PASS_EXACT_OBSERVED_SQUARE_ERROR_IDENTITIES',
                random_samples_or_training_seeds=0)


def prepare():
    assert sha(BASE/'registration.json') == PLAN_SHA
    plan = read(BASE/'registration.json')
    barrier = read(BASE/'all4588_sealed_before_GT.json')
    assert barrier['registration_sha256'] == PLAN_SHA
    assert sha(BASE/'all4588_sealed_before_GT.json') == '2a33eaf9cc1178797bd3c577df74534e6925d4870ca2e15be56a97232120440d'
    assert len(plan['cases']) == 148 and len(barrier['tasks']) == 4588
    assert barrier['GT_open_attempts'] == 0 and barrier['teacher_final_weight_seals'] == 4
    assert len(barrier['all_attempt_teacher_sessions']) == 10
    ledger_path = SUP/'observed_fullscope_generation_receipts_20261126.json'
    ledger = read(ledger_path)
    paths = {Path(__file__), BASE/'registration.json', BASE/'all4588_sealed_before_GT.json', ledger_path}
    units = []
    for case in plan['cases']:
        pair, video = case['pair'], case['evaluated_video']
        receipt = ROOT/ledger['cases'][pair+'/'+video]['receipt']
        assert sha(receipt) == ledger['cases'][pair+'/'+video]['sha256']
        unit = receipt.parent
        entry = dict(pair=pair, video=video, observations=case['observations'],
                     queries=case['temporal_queries'], draws=str(unit/'primary/draws.json'),
                     selections=[str(unit/'primary'/f'stage_{i}_selected_before_answers.json') for i in range(4)])
        paths.update([receipt, Path(entry['observations']), Path(entry['queries']), Path(entry['draws'])])
        paths.update(map(Path, entry['selections']))
        assert sha(entry['observations']) == case['observations_sha256']
        assert sha(entry['queries']) == case['temporal_queries_sha256']
        units.append(entry)
    for name, expected in plan['source_sha256'].items():
        p = Path(name) if Path(name).is_absolute() else ROOT/name
        assert sha(p) == expected
        paths.add(p)
    write(OWN, dict(status='REGISTERED_BEFORE_PAID_FORECAST_VALUE_ANALYSIS',
        source_sha256=sha(__file__), original_registration_sha256=PLAN_SHA,
        inputs={str(p): sha(p) for p in sorted(paths)}, units=units,
        full_seal_sha256=sha(BASE/'all4588_sealed_before_GT.json'),
        math_check=check_math(), observation_scope='All55019 actually requested primary events, no unrequested answers.',
        diagnostic_thresholds=dict(hard_probability=.5, confident_low=.2, confident_high=.8, risk_rank_bins=5),
        no_new_sampling_training_or_algorithm=True, GT_model_and_RMOT_output_access_forbidden=True,
        interpretation=['Paired forecasts are verified against the stage file selected before its answers.',
            'Mean risk calibration and cross terms describe the adaptively selected footprint, not the complete pool.',
            'A teacher probability is not a target semantic label; repeated-time disagreement may be noise, identity mixing or real state change.',
            'Changes across stages use different chosen events and do not identify a causal effect of learning.',
            'Only the risk-proxy component is diagnosed; the acquisition priority also uses a Jacobian and coverage factors.',
            'This audit does not establish a HOTA benefit or calibrated epistemic uncertainty.'],
        time_unix=time.time()))


def summarize(rows):
    y, s, r, v = np.asarray([[row[k] for row in rows] for k in ('y', 's', 'r', 'v')])
    a, e, d, c, p = identities(y, s, r, v)
    bins = []
    for indices in np.array_split(np.argsort(p, kind='stable'), min(5, len(rows))):
        bins.append(dict(events=len(indices), predicted_risk=float(p[indices].mean()),
                         observed_teacher_risk=float(a[indices].mean()),
                         predicted_R_residual=float(v[indices].mean()), observed_R_residual=float(e[indices].mean())))
    repeats = [row for row in rows if row['revisit']]
    corr = float(np.corrcoef(p, a)[0, 1]) if np.std(p)>0 and np.std(a)>0 else None
    return dict(events=len(rows), student_preanswer_teacher_MSE=float(a.mean()),
        observer_preanswer_teacher_MSE=float(e.mean()), neutral_half_teacher_MSE=float(((y-.5)**2).mean()),
        predicted_R_residual_mean=float(v.mean()), actual_R_squared_error_mean=float(e.mean()),
        R_student_disagreement_squared_mean=float(d.mean()), observed_cross_term_mean=float(c.mean()),
        predicted_student_risk_mean=float(p.mean()), actual_student_risk_mean=float(a.mean()),
        actual_minus_proxy_mean=float((a-p).mean()), risk_proxy_observed_error_Pearson=corr,
        risk_rank_bins=bins, object_revisit_events=len(repeats),
        revisit_teacher_hard_flip_events=sum(row['flip'] for row in repeats),
        revisit_confident_opposite_events=sum(row['confident_opposite'] for row in repeats),
        revisit_teacher_probability_abs_change_mean=float(np.mean([row['change'] for row in repeats])) if repeats else None,
        revisit_anchor_frame_distance_median=float(np.median([row['distance'] for row in repeats])) if repeats else None)


def main():
    contract = read(OWN)
    assert contract['source_sha256'] == sha(__file__)
    assert not OUT.exists()
    allowed = set()
    for p, expected in contract['inputs'].items():
        assert sha(p) == expected, p
        allowed.update((os.path.abspath(p), os.path.realpath(p)))
    allowed.update((str(OWN), str(OWN.resolve())))
    output_names = {str(OUT), str(OUT.resolve())}
    rejected = []
    def guard(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = os.path.abspath(os.fsdecode(args[0]))
        mode = args[1] if len(args)>1 else 'r'
        flags = args[2] if len(args)>2 and isinstance(args[2], int) else 0
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT))
        library_prefixes = (str(ROOT/'.venv/lib')+'/', str(Path(sys.base_prefix)/'lib')+'/')
        permitted = p in output_names if writing else p in allowed or p.startswith(library_prefixes)
        if any(x in p for x in ('/offline_GT/', '/gt.txt', '/official_completion.json', '/predictions_seal.json')):
            permitted = False
        if not permitted:
            rejected.append(p)
            raise PermissionError('Unregistered diagnostic IO: '+p)
    sys.addaudithook(guard)
    grouped = defaultdict(list); staged = defaultdict(list); query_rows = defaultdict(list)
    per_view = defaultdict(list)
    for case in contract['units']:
        obs = np.load(case['observations'], mmap_mode='r', allow_pickle=False)
        queries = read(case['queries'])['queries']; n = len(obs)
        selected = {}
        for i, path in enumerate(case['selections']):
            for row in read(path)['selected']:
                assert row['stage'] == i and row['uid'] not in selected
                selected[row['uid']] = row
        draws = read(case['draws'])['draws']
        assert len(draws) == len(selected) == sum(int(q['logical_budget']) for q in queries)
        seen = set(); counts = Counter(); previous = {}
        for draw in draws:
            uid = draw['uid']; q, oi = divmod(int(uid), n)
            assert uid not in seen and int(draw['query']) == q
            seen.add(uid); counts[q] += 1
            prior = selected[uid]
            assert all(draw[k] == prior[k] for k in (*FORECASTS, 'stage', 'index', 'proposal_probability'))
            y, s, r, v = [float(draw[k]) for k in ('probability', *FORECASTS)]
            assert all(np.isfinite(x) and 0<=x<=1 for x in (y,s,r,v))
            assert np.isclose(draw['observer_squared_error'], (r-y)**2, atol=1e-14, rtol=0)
            key = (q, int(obs[oi,1])); frame = int(obs[oi,0]); old = previous.get(key)
            change = abs(y-old[0]) if old else 0.
            row = dict(y=y,s=s,r=r,v=v,revisit=old is not None,
                flip=old is not None and (y>=.5)!=(old[0]>=.5),
                confident_opposite=old is not None and ((y>=.8 and old[0]<=.2) or (y<=.2 and old[0]>=.8)),
                change=change,distance=abs(frame-old[1]) if old else 0)
            previous[key] = (y,frame)
            grouped[case['pair']].append(row); staged[(case['pair'],draw['stage'])].append(row)
            query_rows[(case['pair'],case['video'],q)].append(row)
            per_view[(case['pair'],queries[q]['view'])].append(row)
        assert all(counts[q] == int(query['logical_budget']) for q,query in enumerate(queries))
    assert sum(map(len,grouped.values())) == 55019 and len(query_rows) == 3506
    assert sum(row['revisit'] for rows in grouped.values() for row in rows) == 9726
    for p, expected in contract['inputs'].items():
        assert sha(p) == expected, p
    result = dict(status='COMPLETE_ALL148_PAID_PREANSWER_FORECAST_AND_REVISIT_DIAGNOSTIC',
        source_sha256=sha(__file__), registration_sha256=sha(OWN), all_units=148,
        paid_primary_events=55019, query_instances=3506, by_setting={p:summarize(rows) for p,rows in sorted(grouped.items())},
        by_setting_stage={p+'/'+str(i):summarize(rows) for (p,i),rows in sorted(staged.items())},
        by_setting_requested_view={p+'/'+v:summarize(rows) for (p,v),rows in sorted(per_view.items())},
        query_macro_student_MSE=float(np.mean([np.mean([(z['y']-z['s'])**2 for z in rows]) for rows in query_rows.values()])),
        query_macro_observer_MSE=float(np.mean([np.mean([(z['y']-z['r'])**2 for z in rows]) for rows in query_rows.values()])),
        preanswer_forecasts_equal_preanswer_selection_records=True, GT_or_HOTA_reads=0,
        model_or_RMOT_output_file_reads=0, unrequested_teacher_answer_reads=0,
        new_GPU_teacher_training_or_sampling_operations=0, rejected_IO=rejected,
        current_numerical_study_unchanged=True, interpretation=contract['interpretation'], goal_achieved=False)
    write(OUT,result)
    print(json.dumps(dict(output=str(OUT),status=result['status'],by_setting=result['by_setting'],
        query_macro_student_MSE=result['query_macro_student_MSE'],query_macro_observer_MSE=result['query_macro_observer_MSE'],
        GT_or_HOTA_reads=0),ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else main()
