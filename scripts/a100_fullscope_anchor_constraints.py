"""Postseal-only canonical paid/unasked errors and a teacher-fit lower bound.

No training, neural forward, new prediction, teacher call, seed, or HOTA. The
bound concerns the original query-balanced paid-anchor KL gate; it is not a
semantic oracle, a feasible solution, or an additive explanation of HOTA.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.a100_fullscope_visual_six import OUT, PLAN, PUBLIC, STEM, PRIMARY, COMPARATORS, registration

OWN = OUT / 'anchor_constraints_registration.json'
READER_SOURCES = ('scripts/a100_fullscope_visual_postseal.py',
    'flex_uda/a100/mechanism.py', 'scripts/a100_observation_evidence_transfer_diagnose.py')
TEACHER_SHA = '9a13380b4279bc98f4680bd083ff4995fdcfec120cfded459513be90d041a956'
LIMIT = .005


def boundary_cost(probability, truth):
    p = np.asarray(probability, dtype='float64')
    truth = np.asarray(truth, dtype=bool)
    assert p.shape == truth.shape and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    h = -p*np.log(p.clip(1e-15, 1))-(1-p)*np.log((1-p).clip(1e-15, 1))
    wrong = (p > .5) != truth
    return wrong, np.where(wrong, np.maximum(0., math.log(2)-h), 0.)


def paid_weights(query_index, quotas):
    q = np.asarray(query_index, dtype=int)
    assert quotas and min(quotas.values()) > 0
    assert Counter(q.tolist()) == Counter(quotas)
    w = np.asarray([1/(1.25*len(quotas)*quotas[int(i)]) for i in q])
    assert np.isclose(w.sum(), .8)
    return w


def confusion(truth, mask):
    return dict(TP=int((truth & mask).sum()), FP=int((~truth & mask).sum()),
                FN=int((truth & ~mask).sum()), TN=int((~truth & ~mask).sum()))


def math_check():
    wrong, cost = boundary_cost([.9, .9, .5, 0., 1.], [False, True, True, True, False])
    expected = .9*math.log(1.8)+.1*math.log(.2)
    assert np.array_equal(wrong, [True, False, True, True, True])
    assert np.allclose(cost, [expected, 0., 0., math.log(2), math.log(2)], atol=1e-14)
    assert np.allclose(paid_weights([0, 1, 1, 1], {0: 1, 1: 3}), [.4, 2/15, 2/15, 2/15])
    assert .016*expected > LIMIT
    return dict(status='PASS_DATA_FREE_BOUNDARY_AND_QUERY_WEIGHT_CHECKS',
        high_confidence_wrong_anchor_probability=.9, example_weight=.016,
        example_correction_infimum=.016*expected, unchanged_gate=LIMIT,
        gate_anchor_mean_absolute_probability_deviation_bound=math.sqrt(LIMIT/2),
        GT_GPU_teacher_fits_predictions_or_training_seeds=0)


def prepare():
    plan = registration()
    assert not OWN.exists() and not (OUT/'all4588_sealed_before_GT.json').exists()
    assert not (OUT/'quality_results.json').exists()
    proof = math_check()
    doc = dict(status='REGISTERED_POSTSEAL_CANONICAL_ANCHOR_CONSTRAINT_DIAGNOSIS',
        source_sha256=file_sha(__file__), parent_registration_sha256=file_sha(PLAN),
        reader_source_sha256={p: file_sha(ROOT/p) for p in READER_SOURCES},
        postseal_registration_sha256=file_sha(OUT/'postseal_diagnosis_registration.json'),
        original_sources_verified=len(plan['source_sha256']), video_units=len(plan['cases']),
        trigger='Only after all4588/allteacherattempt seals,186HOTA/180effects reproduction and originalpostseal diagnosis; only if the fixedprimary strictscreen is false. Otherwise skip before readingGT.',
        hypothesis='A strong rawteacher-fit gate may limit correction at some paidanchors. Determine canonical paid/unasked error locations and a necessary lowerbound, without claiming this caused the current result.',
        math='For each wrongly classified teacherprobabilityp, the infimum KL to any corrected thresholdside is KL(Ber(p)||Ber(.5)). Multiply by the originalplain paidweight1/(1.25*nqueries*nquestions_for_query). Otheranchors and auxiliary losses are optimistically zero.',
        gate=LIMIT, math_check=proof,
        scopes=['Originalcanonical matching labels and masks; not pureteacher semantic accuracy.',
            'Primarypolicy floor constrainsPRIMARY; uniformpolicy floor constrainsUNIFORM_SOFT_KD288, notPRIMARY.',
            'Floor abovegate rules out correcting allthose anchors underthat rawfit gate. Floor belowgate doesnot prove neural orjointfeasibility.',
            'Auxiliary lossrelaxedtozero, featurecoupling ignored, plain gate only; LUREgate can impose furtherconstraints.',
            'No pervideoHOTA averaging, newHOTA, candidatepromotion, GTparameterselection or prepositive moduleablation.'],
        no_neural_checkpoint_or_prediction_score_values=True,
        primary_and_all10ordinary_binary_masks_only_after_barrier=True,
        no_new_training_teacher_GPU_seeds_or_predictions=True,
        original42numerical_sources_unchanged=True, current_GT_reads=0,
        references=['https://www.jmlr.org/papers/v11/raykar10a.html', 'https://arxiv.org/abs/2012.13052'],
        all_data_DEV=True, goal_status='ACTIVE', scientific_goal_met=False)
    atomic_json(OWN, doc)
    atomic_json(PUBLIC/(STEM+'_anchor_constraints_registration.json'), doc)
    print(json.dumps(dict(status=doc['status'], video_units=len(plan['cases']),
        math_check=proof['status'], GT_GPU_teacher_or_fits=0)), flush=True)


def main():
    plan = registration();own = json.loads(OWN.read_text())
    assert own['source_sha256'] == file_sha(__file__)
    assert own['parent_registration_sha256'] == file_sha(PLAN)
    for name, digest in own['reader_source_sha256'].items():
        assert file_sha(ROOT/name) == digest
    assert file_sha(OUT/'postseal_diagnosis_registration.json') == own['postseal_registration_sha256']
    barrier = json.loads((OUT/'all4588_sealed_before_GT.json').read_text())
    assert barrier['GT_open_attempts'] == 0 and len(barrier['tasks']) == 4588
    assert barrier['teacher_final_weight_seals'] == 4
    sessions = barrier['all_attempt_teacher_sessions'];assert len(sessions) >= 4
    for row in sessions:
        path = Path(row['path']);assert file_sha(path) == row['sha256']
        teacher = json.loads(path.read_text())['teacher']
        assert teacher['backend_unmodified'] and teacher['teacher_weight_sha256'] == TEACHER_SHA
    for gpu in range(4):
        teacher = json.loads((OUT/'supervisor'/f'GPU{gpu}_teacher_final.json').read_text())['teacher']
        assert teacher['backend_unmodified'] and teacher['teacher_weight_sha256'] == TEACHER_SHA
    reproduced = json.loads((PUBLIC/(STEM+'_reproduction.json')).read_text())
    assert reproduced['status'] == 'PASS_ALL186_HOTA_AND180_EFFECTS'
    quality = json.loads((OUT/'quality_results.json').read_text())
    if quality['candidate_eligible_for_larger_screen']:
        result = dict(status='SKIPPED_PRIMARY_SCREEN_POSITIVE_NO_GT_READ', source_sha256=file_sha(__file__))
        atomic_json(OUT/'anchor_constraints.json', result);print(json.dumps(result), flush=True);return
    postseal = json.loads((OUT/'postseal_diagnosis.json').read_text())
    assert postseal['status'] == 'COMPLETE_FULL_SCOPE_SIX_SETTING_POSTSEAL_DIAGNOSIS'
    assert postseal['source_sha256'] == own['reader_source_sha256'][READER_SOURCES[0]]
    # No manual-label or prediction-mask access occurs above this barrier.
    from flex_uda.a100.mechanism import OUT as DIAGNOSTIC
    from scripts.a100_observation_evidence_transfer_diagnose import CANONICAL
    names = (PRIMARY, *COMPARATORS);reports = []
    for case in plan['cases']:
        pair, video = case['pair'], case['evaluated_video'];folder = OUT/'units'/pair/video
        assert file_sha(case['observations']) == case['observations_sha256']
        assert file_sha(case['temporal_queries']) == case['temporal_queries_sha256']
        observations = np.load(case['observations'])
        queries = json.loads(Path(case['temporal_queries']).read_text())['queries']
        truth = np.load(DIAGNOSTIC/'offline_GT'/CANONICAL[pair]/video/'labels.npy') >= 0
        assert truth.shape == (len(queries), len(observations))
        prior = json.loads((folder/'postseal_query_diagnosis.json').read_text())
        masks = {name: np.load(folder/'predictions'/name/'mask.npy') for name in names}
        for name, mask in masks.items():
            assert mask.shape == truth.shape and mask.dtype == np.bool_
            assert confusion(truth, mask) == prior['methods'][name]['canonical']
        quotas = {q: int(row['logical_budget']) for q, row in enumerate(queries)}
        policies = {}
        for policy, path in [('primary', folder/'primary/draws.json'), ('uniform', folder/'uniform_draws.json')]:
            draws = json.loads(path.read_text())['draws'];indices = np.asarray([d['index'] for d in draws], dtype=int)
            assert len(indices) == len(set(indices.tolist())) == case['logical_budget']
            assert (indices >= 0).all() and (indices < truth.size).all()
            qi = indices//len(observations);w = paid_weights(qi, quotas)
            labels = truth.reshape(-1)[indices];p = np.asarray([d['probability'] for d in draws])
            wrong, c = boundary_cost(p, labels);cost = w*c
            model_errors = {}
            for name, mask in masks.items():
                paid = confusion(labels, mask.reshape(-1)[indices]);total = prior['methods'][name]['canonical']
                unasked = {key: total[key]-paid[key] for key in paid};assert min(unasked.values()) >= 0
                model_errors[name] = dict(paid=paid, unasked=unasked)
            policies[policy] = dict(questions=len(indices), teacher_wrong_canonical_anchors=int(wrong.sum()),
                optimistic_plain_gate_correction_infimum=float(cost.sum()),
                all_wrong_anchor_corrections_ruled_out_by_plain_gate=bool(cost.sum() > LIMIT+1e-12),
                individually_ruled_out_canonical_anchor_count=int((cost > LIMIT+1e-12).sum()),
                constrained_model=PRIMARY if policy == 'primary' else 'UNIFORM_SOFT_KD288',
                models=model_errors)
        row = dict(pair=pair, video=video, policies=policies)
        reports.append(row);atomic_json(folder/'anchor_constraints.json', row)
        print(json.dumps(dict(status='CANONICAL_ANCHOR_CONSTRAINT_CASE_COMPLETE', pair=pair, video=video)), flush=True)
    assert len(reports) == 148
    aggregate = {}
    for pair in sorted({c['pair'] for c in plan['cases']}):
        group = [row for row in reports if row['pair'] == pair];aggregate[pair] = {}
        for policy in ('primary', 'uniform'):
            values = [row['policies'][policy] for row in group]
            aggregate[pair][policy] = dict(videos=len(group),
                teacher_wrong_canonical_anchors=sum(v['teacher_wrong_canonical_anchors'] for v in values),
                videos_all_wrong_anchor_corrections_ruled_out=sum(v['all_wrong_anchor_corrections_ruled_out_by_plain_gate'] for v in values),
                individually_ruled_out_canonical_anchor_count=sum(v['individually_ruled_out_canonical_anchor_count'] for v in values),
                minimum_video_correction_infimum=min(v['optimistic_plain_gate_correction_infimum'] for v in values),
                maximum_video_correction_infimum=max(v['optimistic_plain_gate_correction_infimum'] for v in values),
                models={name: {scope: {key: sum(v['models'][name][scope][key] for v in values)
                    for key in ('TP', 'FP', 'FN', 'TN')} for scope in ('paid', 'unasked')} for name in names})
    result = dict(status='COMPLETE_POSTSEAL_CANONICAL_ANCHOR_CONSTRAINTS', aggregate=aggregate,
        source_sha256=file_sha(__file__), registration_sha256=file_sha(OWN), video_units=148,
        original_gate=LIMIT, only_necessary_relaxed_bound_not_feasibility=True,
        canonical_matching_not_pure_teacher_semantic_accuracy=True,
        no_new_HOTA_fits_teacher_GPU_predictions_or_training_seeds=True,
        not_an_additive_HOTA_decomposition=True, all_data_DEV=True, goal_status='ACTIVE')
    atomic_json(OUT/'anchor_constraints.json', dict(videos=reports, **result))
    atomic_json(PUBLIC/(STEM+'_anchor_constraints.json'), result)
    print(json.dumps(dict(status=result['status'], settings=6, video_units=148)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--check-math', action='store_true');args = parser.parse_args()
    if args.check_math:print(json.dumps(math_check()), flush=True)
    elif args.prepare:prepare()
    else:main()
