"""Real frozen-teacher pilot for joint, rather than recycled scalar, evidence.

All cases come from unlabelled metadata. Four-way and same-context binary
controls use identical image groups. No HOTA, GT, fitting or task predictions.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, metadata, generation_guard
from scripts.a100_six_direction import OUT as SIX, PUBLIC
from scripts.a100_anchor_observability import samples_for
from scripts.a100_observation_design_collect import TEACHER_SHA, PREP

OUT = SIX/'joint_observation_teacher_pilot_20261123'
STATE = OUT/'supervisor/state.json'
PLAN = PUBLIC/'joint_observation_teacher_pilot_plan.json'
OPTIONS = ('A', 'B', 'C', 'D')  # A-only, B-only, both, neither
MODES = ('JOINT_AB', 'JOINT_BA', 'INDEPENDENT_A', 'INDEPENDENT_B', 'SHARED_A', 'SHARED_B')
SUCCESS = 'PASS_JOINT_OBSERVATION_PHYSICAL_TEACHER_PILOT_GOAL_ACTIVE'


def joint_from_latent(a, r, s):
    return np.asarray([a*r*(1-s), a*(1-r)*s, a*r*s, 1-a+a*(1-r)*(1-s)])


def latent_moments(p):
    p = np.asarray(p, dtype=float)
    x, y, both = p[0]+p[2], p[1]+p[2], p[2]
    covariance = both-x*y
    valid = covariance >= -1e-8 and min(x, y, both) > 1e-12
    result = dict(marginal_A=x, marginal_B=y, covariance=covariance,
        common_support_family_admissible=bool(valid))
    if valid:
        result.update(common_support=x*y/both, conditional_A=both/y, conditional_B=both/x)
    return result


def mathematics():
    cases = [(0.7, .8, .6), (.4, .3, .9), (1., .5, .5), (.5, 1., 1.)]
    error = 0.
    for a, r, s in cases:
        p = joint_from_latent(a, r, s)
        m = latent_moments(p)
        error = max(error, float(np.max(np.abs(np.asarray([m['common_support'], m['conditional_A'], m['conditional_B']])-[a, r, s]))))
    independent = np.asarray([.25, .25, .25, .25])
    shared = np.asarray([0., 0., .5, .5])
    assert latent_moments(independent)['marginal_A'] == latent_moments(shared)['marginal_A'] == .5
    assert error < 1e-12
    report = dict(status='PASS_SYNTHETIC_SHARED_SUPPORT_MOMENT_IDENTIFICATION',
        synthetic_cases=len(cases), maximum_parameter_error=error,
        identical_marginals_different_joint_examples=[latent_moments(independent), latent_moments(shared)],
        assumptions=['Same conditioning information for all factors',
            'One structural-zero common Bernoulli support and conditionally independent local Bernoulli states',
            'Nonzero marginals and joint-positive mass; negative covariance is outside this family'],
        not_a_new_theorem_or_ground_truth_semantic_identification=True,
        teacher_probability_calibration_not_established=True,
        no_accuracy_HOTA_or_full_algorithm_claim=True, no_GPU_teacher_or_training_calls=True)
    atomic_json(PUBLIC/'joint_observation_moment_audit.json', report)
    print(json.dumps(report), flush=True)


def prepare():
    guard = generation_guard()
    cases = []
    for pair in ('C1', 'C2', 'C4'):
        target = PAIRS[pair][1]
        md = metadata(target)
        video = sorted(md['videos'])[0]
        bank = md['videos'][video]
        obs = np.asarray(bank['observations'], dtype=int)
        tracks, samples = samples_for(obs)
        chosen = [(int(t), rows) for t, rows in zip(tracks, samples) if len(rows) == 3][:2]
        requests = json.loads((PREP/pair/video/'DESIGN_requests.json').read_text())['requests']
        available = [q for q in md['episodes'] if q['video_id'] == video]
        route = {r['query_id']:r['view'] for r in requests}
        queries = []
        for view in ('CROP', 'FRAME_BOX', 'CLIP'):
            query = next((q for q in available if route[q['expression_id']] == view), None)
            if query is not None:
                queries.append(query)
        queries += [q for q in available if q not in queries][:4-len(queries)]
        assert len(chosen) == 2 and len(queries) == 4
        for query in queries:
            view = next(r['view'] for r in requests if r['query_id'] == query['expression_id'])
            for track, rows in chosen:
                cases.append(dict(pair=pair, target=target, video=video, query_id=query['expression_id'],
                    query_text=query['query_text'], track_id=track, view=view,
                    frame_A=int(obs[rows[-2], 0]), frame_B=int(obs[rows[-1], 0])))
    assert len(cases) == 24 and not guard
    atomic_json(OUT/'cases.json', dict(cases=cases, GT_open_attempts=0,
        selection='First video; first task query per available CROP/FRAME_BOX/CLIP route, then metadata-order fill to four; first two TrackIDs with three anchors; middle and last anchor',
        no_teacher_answers_method_outputs_or_HOTA_used=True))
    print(json.dumps(dict(status='PASS_UNLABELLED_JOINT_PILOT_CASES', cases=24,
        domains=dict(Counter(r['target'] for r in cases)), views=dict(Counter(r['view'] for r in cases)),
        no_six_setting_quality_claim=True, GT_open_attempts=0)), flush=True)


def context(case, na, nb, fa, fb):
    return (f'Two observation groups of a candidate target are supplied. Group A is the first {na} images, '
        f'in chronological order, ending at frame {fa}. Group B is the next {nb} images, '
        f'in chronological order, ending at frame {fb}. Crops show the target; in a full-frame image '
        'the target is inside the red box. Judge each group at its OWN LAST moment, using its temporal '
        f'context if needed. The complete description is: "{case["query_text"]}". ')


def worker():
    plan = json.loads(PLAN.read_text())
    for name, sha in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == sha, name
    assert file_sha(OUT/'cases.json') == plan['cases_sha256']
    guard = generation_guard()
    import torch
    from scripts.active_run import video_spec
    from flex_uda.a100.runtime import teacher
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    atomic_json(STATE, dict(status='LOADING_FROZEN_TEACHER_FOR_JOINT_PILOT', pid=os.getpid(), completed=0, goal_status='ACTIVE'))
    cases = json.loads((OUT/'cases.json').read_text())['cases']
    groups = sorted({(c['target'], c['video']) for c in cases})
    specs = [video_spec(t, v, metadata(t)['videos'][v]) for t, v in groups]
    client = teacher(specs, OUT/'unused_client_cache.sqlite')
    assert client.backend.weight_sha == TEACHER_SHA
    teacher_load_and_setup_seconds = time.perf_counter()-started
    atomic_json(OUT/'teacher_load_receipt.json', dict(teacher_load_and_setup_seconds=teacher_load_and_setup_seconds,
        teacher_parameter_sha256=client.backend.weight_sha, physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES')))
    tok = client.backend.processor.tokenizer
    tokens = [tok.encode(s, add_special_tokens=False) for s in OPTIONS]
    assert all(len(row) == 1 for row in tokens) and len({row[0] for row in tokens}) == 4, tokens
    token_ids = [row[0] for row in tokens]
    prepared = []
    for i, case in enumerate(cases):
        views = []
        for role in ('A', 'B'):
            request = dict(target=case['target'], video_id=case['video'], query_id=case['query_id'],
                query_text=case['query_text'], track_id=case['track_id'], frame_id=case['frame_'+role], view=case['view'])
            _, images, prompt, info = client.prepare(request)
            views.append((images, prompt, info))
        a, b = views
        for mode in MODES:
            if mode.startswith('INDEPENDENT'):
                images, prompt, _ = a if mode.endswith('A') else b
            else:
                first, second = (b, a) if mode == 'JOINT_BA' else (a, b)
                images = first[0]+second[0]
                fa, fb = (case['frame_B'], case['frame_A']) if mode == 'JOINT_BA' else (case['frame_A'], case['frame_B'])
                prompt = context(case, len(first[0]), len(second[0]), fa, fb)
                if mode.startswith('JOINT'):
                    prompt += ('Which statement is correct? A: only group A matches. B: only group B matches. '
                        'C: both groups match. D: neither group matches. Answer with only one letter: A, B, C, or D.')
                else:
                    prompt += f'Does group {mode[-1]} match the description at its own last moment? Answer Yes or No.'
            prepared.append(dict(case=i, mode=mode, images=images, prompt=prompt,
                metadata=dict(A=a[2], B=b[2])))
    rows, captured = [], []
    def capture(module, args, kwargs, output):
        z = output.logits[:, -1, :].detach().float()
        subset = z[:, token_ids]
        probs = torch.softmax(subset, -1).cpu().tolist()
        mass = torch.exp(torch.logsumexp(subset, -1)-torch.logsumexp(z, -1)).cpu().tolist()
        top = z.argmax(-1).cpu().tolist()
        captured.extend(dict(choice_probabilities=p, allowed_vocabulary_mass=m,
            top1_is_choice=t in token_ids, top_token_id=t) for p, m, t in zip(probs, mass, top))
    hook = client.backend.model.register_forward_hook(capture, with_kwargs=True)
    forward_seconds = 0.
    try:
        for n in sorted({len(r['images']) for r in prepared}):
            same = [r for r in prepared if len(r['images']) == n]
            for offset in range(0, len(same), 2):
                batch = same[offset:offset+2]
                captured.clear()
                result = client.backend.forward([r['images'] for r in batch], [r['prompt'] for r in batch])
                forward_seconds += result[0]['batch_forward_seconds']
                assert len(captured) == len(result) == len(batch)
                for request, binary, categorical in zip(batch, result, captured):
                    rows.append({**binary, **categorical, 'case':request['case'], 'mode':request['mode'],
                        'images':len(request['images']), 'prompt':request['prompt'],
                        'input_metadata':request['metadata']})
                atomic_json(OUT/'responses_in_progress.json', dict(cases=cases, responses=rows, GT_open_attempts=0))
                atomic_json(STATE, dict(status='REAL_JOINT_TEACHER_FORWARD_NO_GT', pid=os.getpid(),
                    completed=len(rows), total=144, goal_status='ACTIVE'))
    finally:
        hook.remove()
    frozen = client.backend.seal_frozen()
    assert len(rows) == 144 and not guard
    atomic_json(OUT/'responses.json', dict(cases=cases, responses=rows, GT_open_attempts=0))
    by = {(r['case'], r['mode']):r for r in rows}
    statistics = []
    for i in range(len(cases)):
        p = np.asarray(by[i, 'JOINT_AB']['choice_probabilities'])
        swapped = np.asarray(by[i, 'JOINT_BA']['choice_probabilities'])[[1, 0, 2, 3]]
        sym = (p+swapped)/2
        joint = latent_moments(sym)
        binary = [by[i, 'SHARED_'+s]['yes_probability_fp32'] for s in ('A', 'B')]
        statistics.append(dict(case=i, order_swap_L1=float(np.abs(p-swapped).sum()), **joint,
            shared_binary_vs_joint_marginal_L1=float(np.abs(np.asarray(binary)-[joint['marginal_A'], joint['marginal_B']]).sum())))
    joint_rows = [r for r in rows if r['mode'].startswith('JOINT')]
    report = dict(status=SUCCESS, cases=24, targets=dict(Counter(c['target'] for c in cases)),
        request_modes=list(MODES), actual_new_model_examples=144, cache_hits=0,
        client_SQLite_cache_not_used=True, source_teacher_weight_sha256=TEACHER_SHA,
        choices=list(OPTIONS), choice_token_ids=token_ids,
        joint_examples=len(joint_rows), joint_top1_choice_count=sum(r['top1_is_choice'] for r in joint_rows),
        allowed_choice_vocabulary_mass_quantiles=np.quantile([r['allowed_vocabulary_mass'] for r in joint_rows], [0, .25, .5, .75, 1]).tolist(),
        views=dict(Counter(c['view'] for c in cases)),
        teacher_load_and_setup_seconds=teacher_load_and_setup_seconds,
        model_forward_wall_seconds=forward_seconds,
        symmetrized_positive_association_family_cases=sum(r['common_support_family_admissible'] for r in statistics),
        mean_order_swap_L1=float(np.mean([r['order_swap_L1'] for r in statistics])),
        mean_shared_binary_vs_joint_marginal_L1=float(np.mean([r['shared_binary_vs_joint_marginal_L1'] for r in statistics])),
        total_worker_wall_seconds=time.perf_counter()-started, frozen=frozen,
        physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'), GT_open_attempts=0,
        no_new_training_or_task_predictions=True, no_teacher_accuracy_or_HOTA_claim=True,
        pilot_not_six_setting_quality_or_single_method_cold_cost=True,
        admissibility_not_ground_truth_semantic_identification=True,
        ordinary_future_controls_require_same_information_and_adequate_joint_KD=True, goal_status='ACTIVE')
    atomic_json(OUT/'analysis.json', dict(report, per_case_joint_diagnostics=statistics))
    (PUBLIC/'joint_observation_teacher_pilot_analysis.json').write_text(json.dumps(report, separators=(',', ':'))+'\n')
    atomic_json(STATE, dict(status=SUCCESS, pid=os.getpid(), completed=144, goal_status='ACTIVE'))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('math', 'prepare', 'worker'), required=True)
    args = parser.parse_args()
    try:
        {'math':mathematics, 'prepare':prepare, 'worker':worker}[args.mode]()
    except Exception as error:
        atomic_json(STATE, dict(status='FAILED_JOINT_OBSERVATION_PILOT_REQUIRES_ROOT_CAUSE',
            pid=os.getpid(), error=repr(error), goal_status='ACTIVE'))
        raise
