"""Checks needed for numerical fidelity, count augmentation, and evidence use."""
import torch
from torch import nn
from flex_uda.presence.features import margins_and_features, sentence_vector
from flex_uda.presence.head import PresenceHead
from flex_uda.presence.data import collate_units, head_loss


class FakeFlexHook(nn.Module):
    def __init__(self):
        super().__init__(); self.sample_expression_num = 1
        self.output_ca_qkv = nn.ModuleList([nn.Identity()]); self.head = nn.Linear(768, 2)

    def decode(self, outputs, text, mask, positions, boxes):
        pcd = self.output_ca_qkv[-1](text.mean(-2))
        return self.head(pcd)[:, None], None


def test_final_hook_and_sample_count_restore():
    torch.manual_seed(0)
    model = FakeFlexHook()
    texts = [(torch.randn(1, 1, 25, 768), torch.ones(1, 1, 25)) for _ in range(3)]
    prep = dict(track_ids=[1, 2], outputs=None, positions=None, boxes=None)
    margin, pcd = margins_and_features(model, prep, texts)
    logits = model.head(pcd)
    assert torch.equal(margin, logits[..., 1]-logits[..., 0])
    assert pcd.shape == (2, 3, 768)
    assert model.sample_expression_num == 1
    assert len(model.output_ca_qkv[-1]._forward_hooks) == 0


def test_masked_sentence_pooling():
    h = torch.ones(1, 1, 25, 768); h[:, :, 3:] = 1000
    mask = torch.zeros(1, 1, 25); mask[:, :, :3] = 1
    assert torch.equal(sentence_vector((h, mask)), torch.ones(768))


def test_head_padding_does_not_change_result():
    torch.manual_seed(0); head = PresenceHead().eval()
    pcd, margin, box, sent = torch.randn(2, 768), torch.randn(2), torch.rand(2, 4), torch.randn(768)
    with torch.no_grad():
        single = head(pcd, margin, box, sent)
        padded = head(torch.cat([pcd, torch.randn(3, 768)]), torch.cat([margin, torch.randn(3)]),
                      torch.cat([box, torch.rand(3, 4)]), sent, torch.tensor([0, 0, 1, 1, 1], dtype=torch.bool))
    assert torch.allclose(single[0], padded[0], atol=1e-6)
    assert torch.allclose(single[1], padded[1], atol=1e-6)


def test_empty_candidates_and_presence_loss_gradient():
    head = PresenceHead(); presence, logits = head(torch.empty(0, 768), torch.empty(0), torch.empty(0, 4), torch.ones(768))
    loss, _, ce = head_loss((presence[None], logits[None]), torch.tensor([0]), torch.tensor(1.))
    loss.backward()
    assert torch.isfinite(loss) and ce == 0
    assert head.presence_token.grad.abs().sum() > 0


def test_candidate_dropout_recomputes_count():
    torch.manual_seed(0)
    row = dict(pcd=torch.ones(64, 768), margin=torch.ones(64), box=torch.ones(64, 4), sent=torch.ones(768),
               y=torch.ones(64, dtype=torch.bool), count=64, index=0)
    batch = collate_units([row], augment=True)
    assert batch['count'][0] == (~batch['padding_mask'][0]).sum()
    assert 0 < batch['count'][0] < 64


def test_prior_replacement_does_not_double_teacher_evidence():
    import numpy as np
    from flex_uda.active.belief import BeliefState
    b = BeliefState([[1., 0., -1.]], [1.], [.6])
    b.observe(0, 0, True, 'CROP', (.8, .8))
    evidence = b.evidence.copy()
    b.replace_prior(intercept=[.2], slope=2.)
    first = b.logits.copy()
    b.replace_prior(intercept=[.2], slope=2.)
    assert np.array_equal(b.evidence, evidence) and np.array_equal(first, b.logits)


def test_presence_rejection_requires_two_different_tracks():
    from flex_uda.active.belief import BeliefState
    b = BeliefState([[1., 0., -1.]], [.1], [.1])
    for _ in range(2): b.observe(0, 0, False, 'CROP', (.8, .8), presence_question=True)
    assert not b.absent[0]
    b.observe(0, 1, False, 'CROP', (.8, .8), presence_question=True)
    assert b.absent[0] and b.posterior.sum() == 0


def test_acquisition_route_and_information_gain():
    from flex_uda.active.acquisition import route, expression_types, information_gain
    assert 'MOTION' not in expression_types('the dancing person in a white shirt')
    assert route('the walking person') == 'CLIP'
    assert route('the person on the left') == 'FRAME_BOX'
    assert information_gain(.5, .8, .8) > information_gain(.01, .8, .8)
    assert abs(information_gain(.5, .5, .5)) < 1e-10


def test_ds_recovers_reliable_synthetic_workers():
    import numpy as np
    from flex_uda.active.reliability import Reliability, WORKERS
    rng = np.random.RandomState(0); ds = Reliability()
    y = rng.rand(1000) < .4
    for i, label in enumerate(y):
        ds.answers[i] = {w: [int(label if rng.rand() < .85 else not label)] for w in WORKERS}
    ds.fit()
    assert all(.75 < s < .95 and .75 < c < .95 for s, c in ds.values.values())
    assert np.mean([(ds.posteriors[i] > .5) == label for i, label in enumerate(y)]) > .9


def test_ds_checkpoint_roundtrip():
    import pickle
    from flex_uda.active.reliability import Reliability
    ds = Reliability(); ds.add(('video', 'query', 7), 'CROP', True, True); ds.fit()
    restored = pickle.loads(pickle.dumps(ds))
    assert restored.values == ds.values
    restored.add(('video', 'query', 8), 'CLIP', False, False)
    assert len(restored.answers) == 2


def test_fast_step_keeps_evidence_fixed():
    import numpy as np
    from flex_uda.active.belief import BeliefState
    from flex_uda.active.adapt import fast_step
    state = BeliefState([[4., 1., -1., -3.]], [.2], [.2])
    for _ in range(5): state.observe(0, 0, True, 'CROP', (.8, .8), presence_question=True)
    original = state.evidence.copy()
    result = fast_step(state, np.array([[-2., -1.]]))
    assert result['loss_after'] <= result['loss_before'] + 1e-7
    assert np.array_equal(state.evidence, original)
    assert np.isfinite(state.logits).all()


def test_gt_runtime_guard_denies_raw_and_numerical_labels(tmp_path):
    import os, subprocess, sys
    code = """from flex_uda.presence.runtime import runtime_guard
access = runtime_guard()
for name in ['labels.json','gt.txt','labels.npy','GT_counts.npy']:
    try: open(name,'rb')
    except PermissionError: pass
    else: raise RuntimeError('GT not blocked')
assert len(access) == 4
"""
    env = dict(os.environ, PYTHONPATH=str(__import__('pathlib').Path(__file__).resolve().parents[1]))
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_yestrack_native_fallback_budget_and_prior_order():
    import numpy as np
    from flex_uda.active.baselines import yestrack_video, budget_coverage
    queries = [dict(expression_id='q'+str(q), query_text='person') for q in range(2)]
    bank = dict(observations=[[t, i] for t in [1, 5, 11] for i in [1, 2]],
                lines=[f'{t},{i},0,0,10,20,1,-1,-1,-1' for t in [1, 5, 11] for i in [1, 2]])
    margin = np.array([[1., -1., 1., -1., 1., -1.], [-1., 1., -1., 1., -1., 1.]])
    class Teacher:
        def ask_many(self, rows):
            return [dict(yes_probability=.9, request_id=str(i)) for i, _ in enumerate(rows)]
    mask, receipt = yestrack_video('Dance', 'v', queries, bank, margin, Teacher(), 'YESTRACK_BUDGET_RANDOM', full_calls=1)
    assert receipt['actual_calls'] == receipt['covered_pairs'] == 1
    assert receipt['uncovered_native_exact'] and receipt['coverage_fraction'] == .25
    assert len(budget_coverage('v', queries, [1, 2], np.array([[1, 0], [0, 1]]), 2, prior=True)) == 2
    mask, receipt = yestrack_video('Dance', 'v', queries, bank, margin, Teacher(), 'YESTRACK_FULL_D10')
    assert receipt['actual_calls'] == 8 and mask.all()


def test_short_tracks_keep_initial_prior_and_do_not_supply_pseudo_labels():
    import numpy as np
    from flex_uda.active.belief import BeliefState
    state = BeliefState([[2., 1., -3.]], [1.], [.5], track_lengths=[1, 2, 4])
    before = state.prior.copy()
    state.replace_prior(intercept=[5.], slope=4., scores=[[9., 8., 1.]])
    assert np.array_equal(state.prior[:, :2], before[:, :2])
    assert (state.pseudo_labels()[:, :2] == -1).all()
