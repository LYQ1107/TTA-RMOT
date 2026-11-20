"""CPU-only registered synthetic and exact sealed-v7 estimator replay."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.active.acquisition import round_budget, VIEWS
from flex_uda.active.v7 import BeliefStateV7, OnlinePresenceEB
from flex_uda.active.v8 import BeliefStateV8, PresenceEM, questions_v8

PUBLIC = ROOT/'results/presence_active/redesign_v8'
AMENDED_PUBLIC = PUBLIC/'amendments_A2_A3'


def synthetic_recovery(rho, *, config=None):
    from flex_uda.presence.runtime import runtime_guard
    attempts = runtime_guard()
    if config is None:
        config = json.loads((PUBLIC/'manifest.json').read_text())['phase0']['synthetic']
    rng = np.random.RandomState(config['seed'])
    nq, nt = config['queries_per_video'], config['tracks_per_video']
    lengths = [config['observations_per_track']]*nt
    fractions = np.tile(np.linspace(.95, .05, nt), (nq, 1))
    scores = fractions.copy(); maximum = np.full(nq, .5)
    texts = ['red person']*nq
    rel = {v: (config['sensitivity'], config['specificity']) for v in VIEWS}
    em = PresenceEM(); old = OnlinePresenceEB(); pi0_alpha = pi0_beta = 1.
    videos = []; counts = defaultdict(int)
    for vi in range(config['videos']):
        state = BeliefStateV8(scores, fractions, maximum, track_lengths=lengths,
                              rho_hat=em.rho, pi_strata=em.pi.copy())
        legacy = BeliefStateV7(scores, fractions, maximum, track_lengths=lengths,
                               rho_hat=old.rho, pi_hat=old.pi)
        present = rng.random_sample(nq) < rho
        truth = (rng.random_sample((nq, nt)) < np.asarray(config['true_pi'])[state.strata]) & present[:, None]
        budget = int(np.floor(config['budget_rho']*nq*nt)); carry = 0
        for r in range(config['rounds']):
            possible = questions_v8(state, texts, lengths, rel, r)
            allowance = round_budget(budget, r, config['rounds'])+carry
            used = set(); actual = 0
            for question in possible:
                q, t = question.query, question.track
                if actual >= allowance: break
                if state.absent[q] or (q, t) in used: continue
                if state.low_tentative[q] and question.kind=='IDENTITY': continue
                used.add((q, t)); actual += 1
                p_yes = rel[question.view][0] if truth[q, t] else 1-rel[question.view][1]
                yes = bool(rng.random_sample() < p_yes)
                kind = question.kind
                is_presence = kind in ('GLANCE', 'PRESENCE', 'PRESENCE_COVER')
                state.observe(q, t, yes, question.view, rel[question.view], presence_question=is_presence)
                legacy.observe(q, t, yes, question.view, rel[question.view], presence_question=is_presence)
                counts[kind] += 1
            carry = allowance-actual
        before = em.receipt()
        em.add_video(f'v{vi:03d}', state)
        fit = em.fit(rel)
        old_record = old.finish_video(legacy, rel)
        # The actual v7 estimator has a single global pi. Report it separately
        # from the requested pi_0 regression: the latter groups its original
        # once-per-pair soft counts by the frozen initial stratum, without
        # changing either the old decisions or its global-prior feedback.
        for item in old_record['pi_pair_soft_counts']:
            if state.strata[item['query'], item['track']] == 0:
                pi0_alpha += item['probability']; pi0_beta += 1-item['probability']
        videos.append(dict(video_index=vi, before=before, after=fit,
                           old=old_record['after']))
        if not fit['converged']:
            return dict(status='STOP_EM_NONCONVERGENCE', true_rho=rho,
                        true_pi=config['true_pi'], completed_videos=vi+1,
                        EM=fit, stream=videos, question_counts=dict(counts),
                        GT_open_attempts=len(attempts), GPU_calls=0, teacher_calls=0)
    pi0_old = pi0_alpha/(pi0_alpha+pi0_beta)
    stratum_counts = np.asarray(em.receipt()['tracks_per_stratum'])
    identifiable = stratum_counts >= config['pi_precision_minimum_answered_tracks']
    errors = np.abs(em.pi-np.asarray(config['true_pi']))
    checks = dict(rho=(em.rho >= .95-1e-9 if rho == 1. else abs(em.rho-rho) < .05),
                  pi=bool(np.all(errors[identifiable] < .07)),
                  old_v7_pi0_bias=bool(abs(pi0_old-config['true_pi'][0]) > .1))
    if rho == 1.:
        checks['rho1_expected_identification'] = bool(np.array_equal(identifiable, [True, True, False]))
    return dict(status='PASS' if all(checks.values()) else 'FAIL_REGISTERED_RECOVERY',
                true_rho=rho, true_pi=config['true_pi'], completed_videos=config['videos'],
                EM=em.receipt(), old_v7_global_pi=old.pi, old_v7_pi0_grouped_soft_counts=pi0_old,
                pi0_old_definition='v7 original pair soft counts grouped by initial stratum 0; global prior feedback unchanged',
                pi_accuracy=[dict(stratum=int(k), true_pi=float(config['true_pi'][k]),
                    estimated_pi=float(em.pi[k]), answered_tracks=int(stratum_counts[k]),
                    absolute_error=float(errors[k]),
                    status='PASS' if errors[k] < .07 else 'FAIL') if identifiable[k] else
                    dict(stratum=int(k), true_pi=float(config['true_pi'][k]), estimated_pi=float(em.pi[k]),
                         answered_tracks=int(stratum_counts[k]), absolute_error=float(errors[k]),
                         status='UNIDENTIFIED_INSUFFICIENT_DATA') for k in range(3)],
                checks=checks, stream=videos, question_counts=dict(counts),
                GT_open_attempts=len(attempts), GPU_calls=0, teacher_calls=0)


def exact_v7_replay():
    from flex_uda.presence.runtime import runtime_guard
    from flex_uda.presence.predictions import verify_predictions
    from scripts.active_run import PAIRS
    attempts = runtime_guard()
    previous = ROOT/'results/presence_active/redesign_v7/execution.json'
    m = json.loads(previous.read_text()); base = Path(m['active_root'])
    all_rows = []; worst = 0.
    for pair in PAIRS:
        folder = base/pair/'MINUS_SLOW/rho_0.3'
        seal = verify_predictions(folder)
        eb = OnlinePresenceEB()
        for vi, directory in enumerate(sorted((folder/'videos').iterdir())):
            receipt = json.loads((directory/'receipt.json').read_text())
            registered = json.loads((directory/'presence.json').read_text())
            answers = json.loads((directory/'questions.json').read_text())
            with np.load(directory/'belief.npz') as b:
                qid = {q: i for i, q in enumerate(b['queries'].tolist())}
                tid = {t: i for i, t in enumerate(b['tracks'].tolist())}
                groups = defaultdict(list)
                for a in answers:
                    if a['kind'] == 'PRESENCE':
                        groups[qid[a['query_id']], tid[a['track_id']]].append((a['effective_view'], a['yes_probability'] >= .5))
                state = SimpleNamespace(presence_probability=b['presence_probability'],
                    absent=b['absent'], pi_start=eb.pi, presence_answers=groups)
                ds = receipt['rounds'][-1]['ds']['reliability']
                rel = {k: (x['sensitivity'], x['specificity']) for k, x in ds.items()}
                result = eb.finish_video(state, rel)
            errors = [abs(result[side][k]-registered[side][k])
                      for side in ('before', 'after') for k in result[side]]
            registered_counts = {(x['query'], x['track']): x['probability'] for x in registered['pi_pair_soft_counts']}
            actual_counts = {(x['query'], x['track']): x['probability'] for x in result['pi_pair_soft_counts']}
            assert registered_counts.keys() == actual_counts.keys()
            errors += [abs(v-registered_counts[k]) for k, v in actual_counts.items()]
            error = max(errors, default=0.); worst = max(worst, error)
            all_rows.append(dict(pair=pair, video=directory.name, video_index=vi,
                max_absolute_error=error, before=result['before'], after=result['after'],
                prediction_seal_sha256=file_sha(folder/'predictions_seal.json'),
                input_questions_sha256=file_sha(directory/'questions.json'),
                input_belief_sha256=file_sha(directory/'belief.npz'),
                input_receipt_sha256=file_sha(directory/'receipt.json')))
            if not error < 1e-9:
                return dict(status='STOP_V7_REPRODUCTION_FAILED', max_absolute_error=worst,
                            records=all_rows, GT_open_attempts=len(attempts))
    assert not attempts
    return dict(status='PASS', pairs=6, videos=len(all_rows), max_absolute_error=worst,
                records=all_rows, GT_open_attempts=0, GPU_calls=0, teacher_calls=0,
                v7_execution_sha256=file_sha(previous))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--exact-v7', action='store_true')
    args = parser.parse_args(); start = time.perf_counter()
    if args.exact_v7:
        result = exact_v7_replay()
        atomic_json(AMENDED_PUBLIC/'stage0_v7_reproduction.json', result)
        print(json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)
        if result['status'] != 'PASS': raise SystemExit(1)
    if args.synthetic:
        runs = []
        for rho in (.1, 1.):
            result = synthetic_recovery(rho); runs.append(result)
            atomic_json(AMENDED_PUBLIC/'stage0_synthetic.json', dict(runs=runs, wall_seconds=time.perf_counter()-start))
            print(json.dumps({k: v for k, v in result.items() if k != 'stream'}), flush=True)
            if result['status'] != 'PASS': raise SystemExit(1)


if __name__ == '__main__': main()
