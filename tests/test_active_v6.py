import numpy as np
from flex_uda.active.v6 import BeliefStateV6, ReliabilityV6, questions_v6


def state(fractions=None, presence=.1, lengths=None):
    f = [[1., .8, .6, .4, .2, 0.]] if fractions is None else fractions
    return BeliefStateV6([np.arange(len(f[0]))], f, [presence],
                         track_lengths=lengths or [4]*len(f[0]))


def test_fraction_prior_and_identity_scores_are_not_prior():
    s = state()
    assert np.array_equal(s.prior, np.clip(np.log(np.clip(s.fractions,.05,.95)/
                                                   (1-np.clip(s.fractions,.05,.95))),-2,2))
    before = s.prior.copy()
    s.replace_prior(scores=[np.arange(6)[::-1]])
    assert np.array_equal(s.prior, before)


def test_unasked_pairs_never_supply_fast_or_slow_pseudo_labels():
    s = state()
    s.replace_prior(intercept=[20.])
    assert (s.pseudo_labels() == -1).all()
    s.observe(0, 0, True, 'CROP', (.999, .999))
    assert s.pseudo_labels()[0, 0] == 1
    assert (s.pseudo_labels()[0, 1:] == -1).all()
    assert s.events[-1]['delta'] == 2.


def test_absence_needs_three_thirds_low_presence_and_no_yes():
    s = state()
    for t in (0, 1, 2):
        s.observe(0, t, False, 'CROP', (.8, .8), presence_question=True)
    assert not s.absent[0]
    s.observe(0, 4, False, 'CROP', (.8, .8))
    assert s.absent[0]
    assert (s.pseudo_labels()[~s.asked_mask] == -1).all()
    s.refresh_presence([.2])
    assert not s.absent[0]
    s.refresh_presence([.19])
    assert s.absent[0]
    s.observe(0, 3, True, 'CROP', (.8, .8))
    assert not s.absent[0] and s.confirmed_present[0]


def test_two_tracks_cannot_certify_absence_or_short_tracks_train():
    s = state([[0., 0., 0.]], lengths=[4, 4, 2])
    for t in (0, 1):
        s.observe(0, t, False, 'CROP', (.999, .999))
    assert not s.absent[0]
    s.replace_prior(intercept=[-20.])
    assert s.pseudo_labels()[0, 2] == -1


def test_ds_has_symmetric_prior_without_identity_annotator():
    s = ReliabilityV6()
    for i in range(8):
        for v in s.workers:
            s.add(i, v, i%2 == 0, identity_answer=not i%2)
    s.fit()
    assert set(s.values) == {'CROP', 'CLIP', 'FRAME_BOX'}
    assert all('IDENTITY' not in row for row in s.answers.values())
    assert s.receipt()['Beta_prior'] == [2., 2.]
    assert all(np.isfinite([a,b]).all() for a,b in s.values.values())


def test_acquisition_covers_missing_prior_thirds():
    s = state()
    qs = questions_v6(s, ['red person'], [4]*6, ReliabilityV6().values, 0)
    presence = [x for x in qs if x.kind == 'PRESENCE']
    assert {s.strata[x.query, x.track] for x in presence} == {0, 1, 2}
    s.observe(0, presence[0].track, False, 'CROP', (.8,.8))
    next_q = questions_v6(s, ['red person'], [4]*6, ReliabilityV6().values, 1)
    assert len([x for x in next_q if x.kind == 'PRESENCE']) == 2


def test_v6_fast_is_bounded_regularized_and_evidence_preserving():
    from flex_uda.active.v6 import fast_step_v6
    s = state()
    for _ in range(3):
        s.observe(0, 0, True, 'CROP', (.9,.9))
        s.observe(0, 5, False, 'CROP', (.9,.9))
    evidence = s.evidence.copy()
    r = fast_step_v6(s, [[0.,1.]])
    assert r['identity']['status'] == 'MAP_UPDATED'
    assert 0 <= r['w'] <= 1
    assert r['identity']['objective_after'] <= r['identity']['objective_before']+1e-8
    assert np.array_equal(evidence, s.evidence)
    assert (s.pseudo_labels()[~s.asked_mask] == -1).all()


def test_unknown_exclusive_query_does_not_receive_negative_gradient():
    import torch
    from flex_uda.active.v6 import exclusive_training_logits
    margin = torch.tensor([[2., 50., -1.]], requires_grad=True)
    logits = exclusive_training_logits(margin, np.array([[1],[-1],[0]]), [0])
    loss = torch.nn.functional.cross_entropy(torch.cat([logits, torch.zeros_like(logits[:,:1])],1),
                                            torch.tensor([0]))
    loss.backward()
    assert margin.grad[0,1] == 0 and margin.grad[0,2] != 0
    assert torch.isfinite(loss) and torch.isfinite(margin.grad).all()


def test_fraction_is_selected_observations_not_frame_count_median():
    from flex_uda.active.v6 import selection_fractions
    obs = [[1,10],[1,20],[2,10],[2,20],[3,10],[4,10]]
    margin = np.array([[3.,2.,3.,2.,3.,3.]])
    utilities = np.array([[[.1,.8,.7],[.1,.8,.7],[.1,.8,-1],[.1,.8,-1]]])
    f, _, mask = selection_fractions(obs,[10,20],{1:0,2:1,3:2,4:3},margin,utilities,
                                     [[2.,2.,2.,2.]],raw_k=[[1,1,1,1]])
    assert np.array_equal(f, [[1.,0.]])
    assert np.array_equal(mask, [[1,0,1,0,1,1]])
    low, _, _ = selection_fractions(obs,[10,20],{1:0,2:1,3:2,4:3},margin,utilities,
                                    [[2.,2.,2.,2.]],bias=-20.)
    assert (low == 0).all()


def test_current_frame_cache_track_ids_can_cross_teacher_json_boundary():
    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory
    import torch
    from flex_uda.presence.cardinality import BernoulliCardinalityHead,ARCHITECTURE
    from flex_uda.active.v6_frames import FrameCacheV6
    class Cache:
        def _text(self,text):return text
    with TemporaryDirectory() as directory:
        path=Path(directory);head=path/'head.pt'
        # Match the real source-head checkpoint schema, not a mock parser.
        torch.save(dict(architecture=ARCHITECTURE,head=BernoulliCardinalityHead().state_dict()),head)
        bank=dict(observations=[[1,7],[2,7],[3,7]])
        try:
            frame_cache=FrameCacheV6(None,Cache(),bank,[dict(query_text='person')],[1,2,3],head,path/'cache')
        except KeyError as error:
            # The real head loader may use state_dict rather than head.
            raise AssertionError('Use the actual checkpoint schema in this regression') from error
        assert json.loads(json.dumps(dict(track_id=frame_cache.tracks[0])))=={'track_id':7}
