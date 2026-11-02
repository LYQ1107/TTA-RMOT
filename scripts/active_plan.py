"""Fixed requested matrix and GT-free compute estimate from C1/C3 real pilots."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.presence.runtime import RUN, PUBLIC, target_metadata
from scripts.active_run import PAIRS
from flex_uda.active.baselines import keyframe_indices
from flex_tta.runtime.pf_receipts import atomic_json, file_sha


def jobs():
    result = []
    for pair in PAIRS:
        for rho in (.1, .3, 1.):
            for method in ('FULL', 'YESTRACK_BUDGET_RANDOM', 'YESTRACK_BUDGET_PRIOR', 'UNIFORM_K'):
                result.append(dict(pair=pair, method=method, rho=rho))
        result += [dict(pair=pair, method='YESTRACK_FULL_D10', rho=None),
                   dict(pair=pair, method='RANDOM_ACQ', rho=.3)]
    for pair in ('C1', 'C3', 'C4', 'C6'):
        for method in ('MINUS_PRES', 'MINUS_DS', 'MINUS_ROUTE', 'MINUS_FAST', 'MINUS_SLOW', 'RESET_LORA'):
            result.append(dict(pair=pair, method=method, rho=.3))
    return result


def dimensions(target):
    from collections import Counter, defaultdict
    meta = target_metadata(target); rows = []
    for video, bank in meta['videos'].items():
        lengths = Counter(i for t, i in bank['observations'])
        by_track = defaultdict(list)
        for t, i in bank['observations']: by_track[i].append(dict(frame_id=t))
        q = sum(e['video_id'] == video for e in meta['episodes'])
        rows.append(dict(video=video, queries=q, all_tracks=len(lengths), eligible_tracks=sum(n >= 3 for n in lengths.values()),
            rescore_rows=q*sum(min(8, n) for n in lengths.values()),
            yestrack_full_d10_calls=q*sum(len(keyframe_indices(obs, 10)) for obs in by_track.values())))
    return rows


def estimate():
    records = []; sources = []
    for pair in ('C1', 'C3'):
        path = RUN / 'active' / pair / 'FULL/rho_0.3/pilot_receipt.json'
        record = json.loads(path.read_text())
        if record['GT_open_attempts']: raise ValueError('Pilot GT access')
        updates = sum(r['slow']['optimizer_updates'] for v in record['video_receipts'] for r in v['rounds'])
        calls = sum(v['questions'] for v in record['video_receipts'])
        if not updates or not calls:
            raise ValueError('Pilot lacks actual teacher calls or nonzero LoRA updates: cost estimate is not established')
        dims = dimensions(PAIRS[pair][1]); chosen = [d for d in dims if d['video'] in record['videos']]
        # First video has no warm stream rescore; at most one rescore after each
        # round. Extrapolate four per-video rounds plus one warm stream rescore.
        rescore_passes = sum(r['slow']['optimizer_updates'] > 0 for v in record['video_receipts'] for r in v['rounds'])
        record['seconds_per_call'] = record['teacher']['equivalent_forward_seconds']/calls
        record['seconds_per_update'] = record['LoRA_GPU_seconds']/updates
        record['seconds_per_rescore_row'] = record['rescore_GPU_seconds']/(rescore_passes*sum(d['rescore_rows'] for d in chosen))
        crop = []
        for video in record['videos']:
            questions = json.loads((path.parent / 'videos' / video / 'questions.json').read_text())
            crop.extend(q['batch_forward_seconds']/q['batch_size'] for q in questions if q['effective_view'] == 'CROP')
        record['seconds_per_crop_call'] = float(np.mean(crop)) if crop else record['seconds_per_call']
        records.append(record); sources.append(dict(path=str(path), sha256=file_sha(path)))
    # Two targets measured; KITTI not measured. Use the larger measured unit
    # cost, and state this uncertainty rather than treating sparse zero cost as
    # free computation. The 40h gate uses a conservative no-cache estimate.
    call = max(r['seconds_per_call'] for r in records)
    crop = max(r['seconds_per_crop_call'] for r in records)
    update = max(r['seconds_per_update'] for r in records)
    rescore = max(r['seconds_per_rescore_row'] for r in records)
    matrix = []
    for job in jobs():
        target = PAIRS[job['pair']][1]; ds = dimensions(target); method = job['method']; rho = job['rho']
        calls = (sum(d['yestrack_full_d10_calls'] for d in ds) if method == 'YESTRACK_FULL_D10' else
                 sum(int(np.floor(rho*d['eligible_tracks']*d['queries'])) for d in ds))
        no_slow = method.startswith('YESTRACK') or method in ('UNIFORM_K', 'MINUS_SLOW')
        teacher_seconds = calls*(crop if method.startswith('YESTRACK') else call)
        slow_seconds = 0. if no_slow else 4*50*len(ds)*update
        rescore_seconds = 0. if no_slow else 5*sum(d['rescore_rows'] for d in ds)*rescore
        matrix.append(dict(job, target=target, budget_upper_bound_calls=calls,
            teacher_equivalent_GPU_hours=teacher_seconds/3600, LoRA_GPU_hours=slow_seconds/3600,
            rescore_GPU_hours=rescore_seconds/3600,
            total_GPU_hours_estimate=(teacher_seconds+slow_seconds+rescore_seconds)/3600))
    total = sum(j['total_GPU_hours_estimate'] for j in matrix)
    result = dict(status='MATRIX_OVER_40H_REQUIRE_MAIN_AND_CRITICAL_THEN_USER_DECISION' if total > 40 else 'WITHIN_40H',
        total_GPU_hours_estimate=total, limit_GPU_hours=40, jobs=matrix, pilot_sources=sources,
        limitations=['shared-device elapsed timing, not utilization integral',
            'KITTI unit timing not observed; larger of C1/C3 unit costs used',
            'FULL questions are budget upper bounds; confirmed absence may leave budget unspent',
            'no-cache equivalent teacher cost; actual reused-cache GPU time separately reported',
            'all four rounds assumed to execute50 updates; fewer confident tracks can reduce cost'],
        unit_costs=dict(seconds_per_call=call, seconds_per_CROP=crop, seconds_per_LoRA_update=update,
                        seconds_per_rescore_candidate_query=rescore),
        teacher_or_GPU_called_by_estimator=False, GT_opened_by_estimator=False)
    atomic_json(PUBLIC / 'BC_cost_estimate.json', result)
    return result


def main():
    p = ArgumentParser(description=__doc__); p.add_argument('--estimate', action='store_true'); a = p.parse_args()
    if a.estimate:
        result = estimate(); print(json.dumps({k:v for k,v in result.items() if k != 'jobs'}, indent=2), flush=True)
    else:
        plan = dict(jobs=jobs(), first_real_pilots=[dict(pair=p, method='FULL', rho=.3, videos=1) for p in ('C1', 'C3')],
            uniform_extra_budgets='rho0.1 and1.0 retained to satisfy explicitly requested three-point call-HOTA curves',
            native_and_presence='already sealed Stage A / historical official results, no teacher',
            teacher_calls_before_published_A_gate=0, GPU_limit=2, matrix_40h_gate_not_bypassed=True)
        atomic_json(PUBLIC / 'BC_plan.json', plan); print(json.dumps(dict(jobs=len(plan['jobs']), GPU_calls=0, GT_opens=0)), flush=True)


if __name__ == '__main__': main()
