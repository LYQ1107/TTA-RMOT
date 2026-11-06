"""Registered v8 CPU gates, including the unchanged recovery thresholds."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.special import expit
from flex_uda.active.v8 import PresenceEM, BeliefStateV8, questions_v8
from flex_uda.active.v8_slow import training_pools, balanced_batch, SlowAdapterV8
from flex_uda.active.acquisition import VIEWS

REL = {view: (.8, .8) for view in VIEWS}


def state(nq=1, nt=9, **kwargs):
    fractions = np.tile(np.linspace(.95, .05, nt), (nq, 1))
    return BeliefStateV8(fractions, fractions, np.full(nq, .5), track_lengths=[8]*nt, **kwargs)


def test_no_answers_leave_em_statistics_unchanged():
    em = PresenceEM(); before = em.receipt()
    assert em.add_video('v', state()) == 0
    assert em.fit(REL) == before


def test_scalar_terms_and_vectorized_fit_follow_registered_capped_map():
    em = PresenceEM(); s = state()
    s.observe(0, 0, True, 'CROP', (.8, .8))
    s.observe(0, 4, False, 'CLIP', (.8, .8))
    em.add_video('v', s)
    terms = [(k, *em.track_terms(k, a, REL)) for k, a in em.records[0]['tracks']]
    g = expit(sum(x[1] for x in terms))
    expected_rho = (g+1)/3
    num, den = np.zeros(3), np.zeros(3)
    for k, _, r in terms: num[k] += g*r; den[k] += g
    expected_pi = (num+1)/(den+2)
    em.iters = 1; em.fit(REL)
    assert np.isclose(em.rho, expected_rho)
    assert np.allclose(em.pi, expected_pi)


def test_low_prior_requires_stratum_coverage_and_stop_is_terminal():
    s = state(rho_hat=.05)
    assert s.low_tentative[0] and not s.absent[0]
    for t in (0, 3):
        s.observe(0, t, False, 'CROP', REL['CROP'], presence_question=True)
        assert not s.absent[0]
    s.observe(0, 6, False, 'CROP', REL['CROP'], presence_question=True)
    assert s.absent[0] and not s.low_tentative[0]
    stopped = s.presence_probability[0]
    for _ in range(5): s.observe(0, 6, True, 'CROP', REL['CROP'])
    assert s.absent[0] and s.presence_probability[0] == stopped


def test_glance_covers_every_sentence_in_first_round_when_budget_suffices():
    s = state(17, 30)
    possible = questions_v8(s, ['red person']*17, [8]*30, REL, 0)
    first = possible[:17]
    assert [x.query for x in first] == list(range(17))
    assert all(x.kind == 'GLANCE' and s.strata[x.query, x.track] == 0 for x in first)
    for x in first: s.observe(x.query, x.track, False, x.view, REL[x.view], presence_question=True)
    assert s.asked_mask.any(axis=1).all()
    assert not any(x.kind == 'GLANCE' for x in questions_v8(s, ['red person']*17, [8]*30, REL, 1))


def test_low_tentative_requests_cover_and_never_identity():
    s = state(rho_hat=.05)
    s.observe(0, 0, False, 'CROP', REL['CROP'], presence_question=True)
    possible = questions_v8(s, ['red person'], [8]*9, REL, 1)
    assert {x.kind for x in possible} == {'PRESENCE_COVER'}
    assert {s.strata[x.query, x.track] for x in possible} == {1, 2}


def episode(nt=9, frames=8):
    return SimpleNamespace(observations_by_track={t: [SimpleNamespace(frame_id=f) for f in range(frames)] for t in range(nt)})


def test_slow_gate_rejects_single_class_and_uses_no_model_on_skip():
    s = state(2)
    s.evidence[0, :8] = -8; s.asked_mask[0, :8] = True
    pools, counts = training_pools(s, list(range(9)), episode())
    assert not counts['balanced_gate_pass'] and counts['distinct_positive_pairs'] == 0
    adapter = SlowAdapterV8.__new__(SlowAdapterV8)
    adapter.cache = SimpleNamespace(episode=episode())
    result = adapter.slow_step(s, list(range(9)), ['red person', 'blue person'])
    assert result['status'] == 'SKIP_INSUFFICIENT_BALANCE'
    assert result['optimizer_updates'] == 0


def test_slow_balanced_batches_exclusive_absent_and_tentative_pools():
    s = state(4)
    s.evidence[0, :8] = 8; s.asked_mask[0, :8] = True
    s.evidence[1, :8] = -8; s.asked_mask[1, :8] = True
    # Evidence-supported absence is distinct from budget-only output masking.
    s.absent[2] = s.presence_stopped[2] = True
    s.low_tentative[3] = True
    pools, counts = training_pools(s, list(range(9)), episode(), exclusive=True)
    assert counts['balanced_gate_pass']
    assert counts['pair_pools']['EXCLUSIVE_NEG'] > 0
    assert counts['pair_pools']['ABSENT_NEG'] > 0
    assert not any(q == 3 for q, _, _, _ in pools['ABSENT_NEG'])
    rng = np.random.RandomState(0)
    for _ in range(50):
        batch = balanced_batch(pools, rng)
        assert sum(x[3] == 'DIRECT_POS' for x in batch) == 16
        assert sum(x[3] == 'DIRECT_NEG' for x in batch) == 8
        assert sum(x[3] in ('EXCLUSIVE_NEG', 'ABSENT_NEG') for x in batch) == 8


def test_replay_two_videos_freezes_all_presence_bytes_and_stream_parameters():
    for vi in range(2):
        rho = .2+.1*vi; pi = np.array([.8, .3, .05])
        minus = state(3, rho_hat=rho, pi_strata=pi)
        replay = state(3, rho_hat=rho, pi_strata=pi)
        for r in range(5):
            if r:
                minus.observe(r % 3, r, r % 2 == 0, 'CROP', REL['CROP'], presence_question=True)
                replay.observe(r % 3, r, r % 2 == 0, 'CROP', REL['CROP'], presence_question=True)
            replay.freeze_presence_snapshot(minus.presence_snapshot())
            replay.replace_fractions(np.full((3, 9), .9))
            replay.refresh_presence([.01, .5, .99])
            assert replay.rho_start == minus.rho_start
            assert replay.pi_strata_start.tobytes() == minus.pi_strata_start.tobytes()
            assert all(replay.presence_snapshot()[k].tobytes() == minus.presence_snapshot()[k].tobytes()
                       for k in minus.snapshot_fields)


def test_runtime_guard_blocks_gt_and_normal_v8_generation_opens_none():
    root = Path(__file__).resolve().parents[1]
    code = '''from flex_uda.presence.runtime import runtime_guard
from flex_uda.active.v8 import PresenceEM
attempts=runtime_guard()
PresenceEM().fit({'CROP':(.8,.8)})
assert len(attempts)==0
try: open('/data2/user/FlexHook-TTA/results/presence_active/redesign_v8/gt.txt')
except PermissionError: pass
else: raise AssertionError('GT open was not blocked')
assert len(attempts)==1
print('normal_generation_GT_open_attempts=0; deliberate_denied=1')
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=root,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'normal_generation_GT_open_attempts=0' in result.stdout


@pytest.mark.parametrize('rho', [.1, 1.])
def test_registered_synthetic_200_video_recovery(rho):
    from scripts.active_v8_stage0 import synthetic_recovery, PUBLIC
    from flex_tta.runtime.pf_receipts import atomic_json, file_sha
    result = synthetic_recovery(rho)
    atomic_json(PUBLIC/f'stage0_synthetic_rho_{rho}.json', dict(result,
        prereg_sha256=file_sha(PUBLIC/'PREREG.md'), manifest_sha256=file_sha(PUBLIC/'manifest.json')))
    assert result['status'] == 'PASS', json.dumps({k: v for k, v in result.items() if k != 'stream'})
