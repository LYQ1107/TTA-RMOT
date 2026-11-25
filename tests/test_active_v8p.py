"""Only the protocol-switch correctness gates; no real GT or GPU."""
from collections import defaultdict
import pickle
from types import SimpleNamespace
import numpy as np
import pytest
from flex_uda.active.acquisition import VIEWS
from flex_uda.active.v8 import BeliefStateV8, PresenceEM, questions_v8, fast_step_v8
from flex_uda.active.v8p import (CHECKLIST, PRESUPPOSED, classify_protocol,
    belief_state, presence_estimator, questions_v8p)
from flex_uda.active.v8_slow import training_pools

REL = {v: (.8, .8) for v in VIEWS}


@pytest.mark.parametrize('queries,expected,nsets', [
    ({'v1': ['red', 'blue'], 'v2': ['blue', 'red', 'red']}, CHECKLIST, 1),
    ({'v1': ['red'], 'v2': ['blue']}, PRESUPPOSED, 2),
    ({'v1': ['red'], 'v2': ['red ']}, PRESUPPOSED, 2),
])
def test_exact_raw_query_sets(queries, expected, nsets):
    r = classify_protocol(queries)
    assert r['protocol'] == expected and r['unique_query_sets'] == nsets


def inputs():
    f = np.tile(np.linspace(.95, .05, 9), (3, 1))
    return (f.copy(), f, [.01, .5, .99]), dict(track_lengths=[8]*8+[2])


def test_presupposed_never_infers_presence_or_generates_presence_questions(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('PRESUPPOSED called PresenceEM')
    monkeypatch.setattr(PresenceEM, '__init__', forbidden)
    args, kwargs = inputs()
    s = belief_state(PRESUPPOSED, *args, **kwargs)
    em = presence_estimator(PRESUPPOSED)
    for r in range(4):
        possible = questions_v8p(PRESUPPOSED, s, ['red person']*3, kwargs['track_lengths'], REL, r)
        assert possible and {q.kind for q in possible} <= {'IDENTITY', 'POSITIVE_RECHECK'}
        for q in range(3):
            for t in range(8):
                s.observe(q, t, False, 'CROP', REL['CROP'])
        s.recompute_presence(REL)
        assert np.all(s.presence_probability == 1)
        assert not s.absent.any() and not s.low_tentative.any()
        assert not s.output_absence().any()
    em.add_video('fake', s)
    assert em.fit(REL)['presence_inference'] is False
    assert not s.presence_answers
    assert (s.pseudo_labels()[~s.asked_mask] == -1).all()
    episode = SimpleNamespace(observations_by_track={t: [SimpleNamespace(frame_id=f)
        for f in range(8)] for t in range(9)})
    pools, _ = training_pools(s, list(range(9)), episode)
    assert not pools['ABSENT_NEG']


def test_presupposed_fast_fit_requires_actual_teacher_support():
    args, kwargs = inputs()
    s = belief_state(PRESUPPOSED, *args, **kwargs)
    assert not s.presence_supported.any()
    r = fast_step_v8(s, np.zeros((3, 8)))
    assert r['presence']['positive'] == 0 and r['presence']['negative'] == 0
    s.observe(0, 0, True, 'CROP', REL['CROP'])
    assert s.presence_supported.tolist() == [True, False, False]
    r = fast_step_v8(s, np.zeros((3, 8)))
    assert r['presence']['positive'] == 1 and r['presence']['negative'] == 0
    assert r['temperature'] == 1 and r['bias'] == 0


def test_checklist_two_video_stream_is_v8_byte_exact():
    # Exact class, questions, state and online estimator; not merely allclose.
    em, switched_em = PresenceEM(), presence_estimator(CHECKLIST)
    for vi in range(2):
        args, kwargs = inputs()
        a = BeliefStateV8(*args, **kwargs, rho_hat=em.rho, pi_strata=em.pi)
        b = belief_state(CHECKLIST, *args, **kwargs,
                        rho_hat=switched_em.rho, pi_strata=switched_em.pi)
        assert type(a) is type(b)
        for r in range(4):
            qa = questions_v8(a, ['red', 'blue', 'walking'], kwargs['track_lengths'], REL, r)
            qb = questions_v8p(CHECKLIST, b, ['red', 'blue', 'walking'], kwargs['track_lengths'], REL, r)
            assert pickle.dumps(qa) == pickle.dumps(qb)
            for q in qa[:5]:
                yes = (q.query + q.track + r + vi) % 3 == 0
                for s in (a, b):
                    s.observe(q.query, q.track, yes, q.view, REL[q.view],
                        presence_question=q.kind in ('GLANCE', 'PRESENCE', 'PRESENCE_COVER'))
            ra = fast_step_v8(a, np.zeros((3, 8)))
            rb = fast_step_v8(b, np.zeros((3, 8)))
            assert pickle.dumps(ra) == pickle.dumps(rb)
            assert pickle.dumps(a) == pickle.dumps(b)
            assert ((a.posterior > .5) & ~a.output_absence()[:, None]).tobytes() == \
                   ((b.posterior > .5) & ~b.output_absence()[:, None]).tobytes()
        em.add_video(str(vi), a); switched_em.add_video(str(vi), b)
        assert pickle.dumps(em.fit(REL)) == pickle.dumps(switched_em.fit(REL))


def test_live_dispatch_presupposed_keeps_all_no_queries_eligible():
    from scripts.active_v8_run import live_round
    args, kwargs = inputs(); s = belief_state(PRESUPPOSED, *args, **kwargs)
    tracks = {t: [dict(frame_id=f, bbox_xywh=[10, 10, 20, 30]) for f in range(8)] for t in range(9)}
    class Teacher:
        def ask_many(self, requests):
            return [dict(r, effective_view=r['view'], yes_probability=.01) for r in requests]
    qs = [dict(expression_id=str(q), query_text='red person') for q in range(3)]
    possible = questions_v8p(PRESUPPOSED, s, ['red person']*3, kwargs['track_lengths'], REL, 0)
    answered = live_round(s, possible, 12, defaultdict(set), tracks,
        SimpleNamespace(tracks=list(range(9))), qs, 'SportsMOT', 'fake',
        dict(width=100, height=100), Teacher())
    assert len(answered) == 12
    assert {a['kind'] for a in answered} == {'IDENTITY'}
    assert not s.output_absence().any()


@pytest.mark.parametrize('c3,expected', [(63.35, 'PASS'), (63.349999, 'FAIL')])
def test_stage_A_c3_floor_is_not_replaced_by_five_wins(c3, expected):
    from scripts.active_v8p_report import verdict
    rows=[dict(pair='C'+str(i),v8P=40 if i in (1,6) else c3 if i==3 else 30,
               YT_PRIOR=20,UNIFORM_K=21) for i in range(1,7)]
    result=verdict(rows)
    assert result['directions_above_both_controls']==6
    assert result['result']==expected


def test_conditional_B_scope_is_exact_and_does_not_resume_old80():
    from scripts.active_v8p_stage_b import tier_jobs
    b1,b2,b3=(tier_jobs(t) for t in ('B1','B2','B3'))
    assert len(b1)==24 and len(b2)==36 and len(b3)==2
    assert {j['method'] for j in b1}=={'RANDOM_ACQ','MINUS_DS','MINUS_ROUTE','MINUS_GLANCE','MINUS_EB','MINUS_PRES'}
    assert all(j['pair'] in ('C1','C6') for j in b1 if j['method'] in ('MINUS_GLANCE','MINUS_EB','MINUS_PRES'))
    assert {j['rho'] for j in b2}=={.1,1.}
    assert {j['method'] for j in b2}=={'MINUS_SLOW','YESTRACK_BUDGET_PRIOR','UNIFORM_K'}
    assert b3==[dict(pair='C4',method=m,rho=.3) for m in ('FULL','SLOW_REPLAY')]


def test_registered_ablations_preserve_the_unchanged_fields():
    from flex_uda.active.v8p import setup,variant_questions
    from flex_uda.active.baselines import video_rng
    effective,em,ds=setup(CHECKLIST,'MINUS_EB')
    assert effective==CHECKLIST and em.receipt()['rho_hat']==.5
    assert em.fit(REL)['pi_strata']==[.5,.5,.5]
    effective,_,_=setup(CHECKLIST,'MINUS_PRES')
    assert effective==PRESUPPOSED
    args,kwargs=inputs();s=belief_state(PRESUPPOSED,*args,**kwargs)
    original=questions_v8p(PRESUPPOSED,s,['red person']*3,kwargs['track_lengths'],REL,0)
    random1=variant_questions(PRESUPPOSED,s,['red person']*3,kwargs['track_lengths'],REL,0,
        ablation='RANDOM_ACQ',rng=video_rng('fake'))
    random2=variant_questions(PRESUPPOSED,s,['red person']*3,kwargs['track_lengths'],REL,0,
        ablation='RANDOM_ACQ',rng=video_rng('fake'))
    assert random1==random2 and sorted(random1,key=str)==sorted(original,key=str)
    assert [q.kind for q in original]==[q.kind for q in random1]
    _,_,fixed=setup(PRESUPPOSED,'MINUS_DS')
    fixed.add(('v','q',0),'CROP',True);fixed.fit()
    assert fixed.values==REL
