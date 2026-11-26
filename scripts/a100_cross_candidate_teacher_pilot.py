"""Prospective, label-free teacher audit: distinct objects at the same time.

This measures the available supervision, not RMOT accuracy. Both/neither
remain valid answers; cross-object judgments are never fabricated from the
earlier same-object, two-time answer collection.
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
from scripts.a100_observation_design_collect import PREP, TEACHER_SHA

OUT = SIX/'cross_candidate_teacher_pilot_20261123'
PLAN = PUBLIC/'cross_candidate_teacher_pilot_plan.json'
STATE = OUT/'supervisor/state.json'
MODES = ('JOINT_AB', 'JOINT_BA', 'SHARED_A', 'SHARED_B', 'INDEPENDENT_A', 'INDEPENDENT_B')
SUCCESS = 'PASS_REAL_CROSS_CANDIDATE_TEACHER_PILOT_NOT_TASK_QUALITY'
SOURCES = ('scripts/a100_cross_candidate_teacher_pilot.py',
           'scripts/a100_joint_observation_collect.py', 'scripts/active_run.py',
           'flex_uda/a100/mechanism.py', 'flex_uda/a100/runtime.py',
           'flex_uda/active/teacher_client.py', 'flex_uda/teacher/decompose_inputs.py',
           'flex_uda/teacher/yestrack_backend.py')


def prepare():
    guard = generation_guard()
    cases = []
    for pair in ('C1', 'C2', 'C4'):
        target = PAIRS[pair][1]
        md = metadata(target); video = sorted(md['videos'])[0]
        bank = md['videos'][video]; obs = np.asarray(bank['observations'], dtype=int)
        requests = json.loads((PREP/pair/video/'DESIGN_requests.json').read_text())['requests']
        queries = [q for q in md['episodes'] if q['video_id'] == video]
        routes = {r['query_id']:r['view'] for r in requests}
        selected = []
        for view in ('CROP', 'FRAME_BOX', 'CLIP'):
            q = next((q for q in queries if routes[q['expression_id']] == view), None)
            if q is not None and q not in selected:
                selected.append(q)
        selected += [q for q in queries if q not in selected][:4-len(selected)]
        assert len(selected) == 4
        frames, counts = np.unique(obs[:, 0], return_counts=True)
        eligible = frames[counts >= 3]
        assert len(eligible)
        frame = int(eligible[len(eligible)//2])
        tracks = sorted(set(obs[obs[:, 0] == frame, 1].tolist()))
        pairs = ((tracks[0], tracks[1]), (tracks[0], tracks[-1]))
        for q in selected:
            for ta, tb in pairs:
                for view in ('CROP', 'FRAME_BOX', 'CLIP'):
                    cases.append(dict(pair=pair, target=target, video=video,
                        query_id=q['expression_id'], query_text=q['query_text'],
                        track_A=int(ta), track_B=int(tb), frame=frame, view=view,
                        original_route=routes[q['expression_id']]))
    assert len(cases) == 72 and all(c['track_A'] != c['track_B'] for c in cases) and not guard
    atomic_json(OUT/'cases.json', dict(cases=cases, GT_open_attempts=0,
        selection='First target video; first query per available route then metadata-order fill to four; median candidate frame with >=3 objects; first-vs-second and first-vs-last IDs; all three views',
        no_teacher_answers_native_scores_predictions_or_GT_in_selection=True))
    plan = dict(status='PROSPECTIVE_CROSS_CANDIDATE_TEACHER_PILOT', cases=72,
        physical_teacher_examples=72*len(MODES), modes=list(MODES),
        source_sha256={s:file_sha(ROOT/s) for s in SOURCES},
        cases_sha256=file_sha(OUT/'cases.json'), teacher_sha256=TEACHER_SHA,
        fixed_teacher_external_evidence=True, no_teacher_or_task_training=True,
        no_manual_GT_no_HOTA_no_task_predictions=True,
        independent_binary_same_object_images=True, shared_binary_identical_joint_context=True,
        compare_CROP_FRAME_BOX_causal_CLIP=True,
        condition_on_exactly_one_only_for_diagnostic_not_as_task_constraint=True,
        count_all_AB_BA_and_binary_calls_and_model_load=True,
        task_contract=dict(reference='https://arxiv.org/pdf/2303.03366',
            arbitrary_referent_count_and_temporally_varying_membership=True,
            same_requested_frame_both_candidates=True,
            CLIP_uses_past_and_current_only=True),
        learning_hypothesis='Retain candidate-specific stable identity h_i; conditionalize out query-shared b_q, never identity itself.',
        no_full_six_setting_or_RSI_quality_claim=True)
    atomic_json(PLAN, plan)
    print(json.dumps({k:v for k,v in plan.items() if k != 'source_sha256'}), flush=True)


def context(case, a, b):
    text = 'Two DIFFERENT candidate objects from the SAME scene and requested moment are shown. '
    text += 'Crops show the candidate; in a full-frame image the candidate is inside the red box. '
    for role, item in (('A', a), ('B', b)):
        text += f'Group {role} contains the next {len(item[0])} images in chronological order. '
        text += 'Its frame and normalized candidate position/size records are '+json.dumps(item[1]['moments'], separators=(',', ':'))+'. '
    return text+f'Judge EACH candidate at requested frame {case["frame"]}, using its own earlier images if needed. The complete description is: '+json.dumps(case['query_text'])+'. '


def worker():
    guard = generation_guard(); plan = json.loads(PLAN.read_text())
    for name, sha in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == sha, name
    assert file_sha(OUT/'cases.json') == plan['cases_sha256']
    import torch
    from scripts.active_run import video_spec
    from scripts.a100_joint_observation_collect import visual
    from flex_uda.a100.runtime import teacher
    started = time.perf_counter(); torch.set_num_threads(4)
    torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.65)
    cases = json.loads((OUT/'cases.json').read_text())['cases']
    groups = sorted({(c['target'], c['video']) for c in cases})
    specs = [video_spec(t, v, metadata(t)['videos'][v]) for t, v in groups]
    atomic_json(STATE, dict(status='LOADING_FROZEN_TEACHER', pid=os.getpid(), completed=0))
    client = teacher(specs, OUT/'unused_cache.sqlite')
    assert client.backend.weight_sha == TEACHER_SHA
    load_seconds = time.perf_counter()-started
    atomic_json(OUT/'teacher_load_receipt.json', dict(seconds=load_seconds,
        frozen_weight_sha256=TEACHER_SHA, physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES')))
    tok = client.backend.processor.tokenizer
    choices = [tok.encode(c, add_special_tokens=False) for c in 'ABCD']
    assert all(len(x) == 1 for x in choices) and len({x[0] for x in choices}) == 4
    ids = [x[0] for x in choices]; captured = []
    def capture(module, args, kwargs, result):
        logits = result.logits[:, -1, :].detach().float(); z = logits[:, ids]
        for p, mass, top in zip(z.softmax(-1).cpu().tolist(),
                (z.logsumexp(-1)-logits.logsumexp(-1)).exp().cpu().tolist(), logits.argmax(-1).cpu().tolist()):
            captured.append(dict(choice_probabilities=p, allowed_choice_mass=mass, top1_is_choice=top in ids))
    handle = client.backend.model.register_forward_hook(capture, with_kwargs=True)
    responses = []; forward_seconds = preparation_seconds = 0.
    checkpoint = OUT/'responses_checkpoint.json'
    if checkpoint.is_file():
        responses = json.loads(checkpoint.read_text())['responses']
    seen = {(r['case'], r['mode']) for r in responses}
    assert len(seen) == len(responses)
    new_examples = 0; batch_receipts = []
    try:
        for ci, case in enumerate(cases):
            if all((ci, mode) in seen for mode in MODES):
                continue
            begin = time.perf_counter()
            views = []
            for role in ('A', 'B'):
                candidate = dict(case, track_id=case['track_'+role])
                images, info = visual(client, candidate, case['frame'])
                assert info['moments'][-1]['frame'] == case['frame']
                assert all(x['frame'] <= case['frame'] for x in info['moments'])
                request = dict(target=case['target'], video_id=case['video'], query_id=case['query_id'],
                    query_text=case['query_text'], track_id=case['track_'+role], frame_id=case['frame'], view=case['view'])
                _, _, prompt, _ = client.prepare(request)
                views.append((images, info, prompt))
            a, b = views; prepared = []
            for mode in MODES:
                if (ci, mode) in seen:
                    continue
                if mode.startswith('INDEPENDENT'):
                    group = a if mode.endswith('A') else b
                    images, prompt = group[0], group[2]
                else:
                    first, second = (b, a) if mode == 'JOINT_BA' else (a, b)
                    images = first[0]+second[0]; prompt = context(case, first, second)
                    if mode.startswith('JOINT'):
                        prompt += 'Which statement is correct? A: only group A matches. B: only group B matches. C: both groups match. D: neither group matches. Answer with only one letter: A, B, C, or D.'
                    else:
                        prompt += f'Does group {mode[-1]} match the description at the requested moment? Answer Yes or No.'
                prepared.append(dict(case=ci, mode=mode, images=images, prompt=prompt))
            preparation_seconds += time.perf_counter()-begin
            for image_count in sorted({len(r['images']) for r in prepared}):
                same = [r for r in prepared if len(r['images']) == image_count]
                for begin in range(0, len(same), 2):
                    batch = same[begin:begin+2]; captured.clear()
                    raw = client.backend.forward([r['images'] for r in batch], [r['prompt'] for r in batch])
                    assert len(raw) == len(captured) == len(batch)
                    seconds = raw[0]['batch_forward_seconds']; forward_seconds += seconds
                    new_examples += len(batch)
                    for item, binary, cat in zip(batch, raw, captured):
                        responses.append(dict(binary, **cat, case=ci, mode=item['mode'],
                            prompt=item['prompt'], image_count=len(item['images']),
                            A=a[1], B=b[1]))
                        seen.add((ci, item['mode']))
                    batch_receipts.append(dict(case=ci, examples=len(batch), seconds=seconds))
                    atomic_json(checkpoint, dict(responses=responses, GT_open_attempts=0))
            atomic_json(STATE, dict(status='ACTUAL_FROZEN_TEACHER_FORWARD', pid=os.getpid(),
                completed=len(responses), total=plan['physical_teacher_examples'],
                session_new_examples=new_examples, cache_hits=0))
            if (ci+1) % 6 == 0:
                print(json.dumps(dict(case=ci+1, total_cases=len(cases), completed=len(responses))), flush=True)
    finally:
        handle.remove()
    frozen = client.backend.seal_frozen()
    assert len(responses) == plan['physical_teacher_examples'] and not guard
    atomic_json(OUT/'responses.json', dict(cases=cases, responses=responses, GT_open_attempts=0))
    atomic_json(OUT/'worker_receipt.json', dict(status=SUCCESS, session_new_model_examples=new_examples,
        retained_checkpoint_examples=len(responses)-new_examples, cache_hits=0, SQLite_not_used=True,
        teacher_load_seconds=load_seconds, preparation_seconds=preparation_seconds,
        actual_forward_wall_seconds=forward_seconds, total_worker_wall_seconds=time.perf_counter()-started,
        frozen=frozen, physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'), batch_receipts=batch_receipts,
        no_GT_no_HOTA_no_training_no_task_predictions=True))
    analyze()


def analyze():
    data = json.loads((OUT/'responses.json').read_text()); cases = data['cases']
    by = {(r['case'], r['mode']):r for r in data['responses']}; per_case = []
    for i, case in enumerate(cases):
        a = np.asarray(by[i, 'JOINT_AB']['choice_probabilities'])
        b = np.asarray(by[i, 'JOINT_BA']['choice_probabilities'])[[1, 0, 2, 3]]
        q = (a+b)/2; marginal = np.asarray([q[0]+q[2], q[1]+q[2]])
        shared = np.asarray([by[i, 'SHARED_'+s]['yes_probability_fp32'] for s in ('A', 'B')])
        single = np.asarray([by[i, 'INDEPENDENT_'+s]['yes_probability_fp32'] for s in ('A', 'B')])
        per_case.append(dict(case=i, target=case['target'], view=case['view'],
            symmetrized_joint=q.tolist(), conditional_exactly_one=float(q[0]+q[1]),
            conditional_A=float(q[0]/max(q[0]+q[1], 1e-300)),
            candidate_marginal_gap=float(abs(marginal[0]-marginal[1])),
            order_swap_L1=float(abs(a-b).sum()), shared_binary_joint_L1=float(abs(shared-marginal).sum()),
            independent_binary_joint_L1=float(abs(single-marginal).sum())))
    def aggregate(rows):
        return dict(cases=len(rows), joint_class_mean=np.mean([r['symmetrized_joint'] for r in rows], axis=0).tolist(),
            mean_exactly_one_mass=float(np.mean([r['conditional_exactly_one'] for r in rows])),
            mean_candidate_marginal_gap=float(np.mean([r['candidate_marginal_gap'] for r in rows])),
            mean_order_swap_L1=float(np.mean([r['order_swap_L1'] for r in rows])),
            mean_shared_binary_joint_L1=float(np.mean([r['shared_binary_joint_L1'] for r in rows])),
            mean_independent_binary_joint_L1=float(np.mean([r['independent_binary_joint_L1'] for r in rows])))
    report = dict(status=SUCCESS, overall=aggregate(per_case),
        by_view={v:aggregate([r for r in per_case if r['view'] == v]) for v in ('CROP', 'FRAME_BOX', 'CLIP')},
        by_target={t:aggregate([r for r in per_case if r['target'] == t]) for t in sorted({r['target'] for r in per_case})},
        physical_examples=len(data['responses']), cache_hits=0,
        distinct_candidate_same_requested_frame=True, no_GT_or_task_accuracy_claim=True,
        no_trained_RSI_or_six_setting_quality_claim=True,
        teacher_distribution_contrast_is_not_semantic_truth=True,
        receipt_path=str(OUT/'worker_receipt.json'))
    atomic_json(OUT/'analysis.json', dict(report=report, per_case=per_case))
    atomic_json(PUBLIC/'cross_candidate_teacher_pilot_analysis.json', report)
    atomic_json(STATE, dict(status=SUCCESS, completed=len(data['responses'])))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('prepare', 'worker', 'analyze'), required=True)
    args = parser.parse_args()
    try:
        globals()[args.mode]()
    except Exception as error:
        atomic_json(STATE, dict(status='FAILED_CROSS_CANDIDATE_TEACHER_PILOT', pid=os.getpid(), error=repr(error)))
        raise
