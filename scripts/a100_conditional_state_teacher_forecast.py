"""Prospective, read-only teacher-decision forecast diagnostic.

Inventory is model/GT-free. Forecasts activate only after BOTH complete studies
seal, reproduce, and fail at least one fixed primary condition. They never
select an adapter, checkpoint, threshold, query or published primary method.
Teacher decisions are observations, not semantic truth or independent RMOT GT.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from flex_tta.runtime.pf_receipts import atomic_json, file_sha

SIX = ROOT/'a100_runs/visual_classical_20261121/six_direction_20261122'
MAIN = SIX/'conditional_state_20261127/primary'
PARENT = MAIN.parent/'parent_control_supplement'
DEST = ROOT/'a100_runs/visual_classical_20261121/supervisor/conditional_state_teacher_forecast_20261127'
PLAN = DEST/'registration.json'
AUX = ROOT/'a100_runs/mechanism_20261121/evidence/responses.jsonl'
PUBLIC = ROOT/'docs/a100_six_direction_20261122'
VIEWS = ('CROP', 'CLIP', 'FRAME_BOX')
T0 = '9a13380b4279bc98f4680bd083ff4995fdcfec120cfded459513be90d041a956'
ROLES = (
    'PRIMARY_CHANNEL_SCOPE', 'SOFT_KD_ANCHOR', 'SOFT_KD_SCOPE',
    'SOFT_SHARED_CHANNEL_SCOPE', 'PARENT_SOFT_KD_ANCHOR',
    'PARENT_SOFT_KD_SCOPE', 'PARENT_SOFT_SHARED_CHANNEL_SCOPE')


def read(path):
    return json.loads(Path(path).read_text())


def restrict_io(paths):
    """Forbid undeclared data reads, writes outside this diagnostic, and network."""
    allowed = {os.path.abspath(p) for p in paths}
    allowed |= {os.path.realpath(p) for p in allowed}
    outputs = (str(DEST)+'/', str(DEST.resolve())+'/')
    opened = set()
    def audit(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo'):
            raise PermissionError('Offline diagnostic forbids network')
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = os.path.abspath(os.fsdecode(args[0])); mode = args[1] if len(args)>1 else 'r'
        flags = args[2] if len(args)>2 and isinstance(args[2],int) else 0
        writing = (isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT))
        permitted = path.startswith(outputs) if writing else (path in allowed or path.startswith(outputs) or path.endswith(('.py','.pyc')))
        if Path(path).suffix.lower() in ('.jpg','.jpeg','.png','.pt','.pth','.safetensors'):
            permitted = False
        if not permitted:
            raise PermissionError('Undeclared teacher-forecast IO: '+path)
        if not writing:
            opened.add(path)
    sys.addaudithook(audit)
    return opened


def text_key(value):
    return ' '.join(value.split()).casefold()


def scope_nodes(obs, node, view, delta, tracks):
    """Reconstruct original PairedViews, preserving actual crop multiplicity."""
    if view not in VIEWS:
        raise ValueError(view)
    if view != 'CLIP':
        return [int(node)], [int(node)]
    frame, track = map(int, obs[node])
    candidates = tracks[track]
    times = obs[candidates, 0]
    inputs = []
    for offset in (3*delta, 2*delta, delta, 0):
        i = int(np.searchsorted(times, frame-offset, side='right'))-1
        if i >= 0:
            inputs.append(int(candidates[i]))
    if len(inputs) < 2 or inputs[-1] != node:
        raise ValueError('Effective CLIP must have the actual nonfallback input')
    return inputs, sorted(set(inputs), key=lambda i: int(obs[i, 0]))


def keys_for(query, nodes, obs):
    return {(text_key(query), int(obs[i, 0]), int(obs[i, 1])) for i in nodes}


def rejected_reason(event, obs, train_keys, train_requests, aux_physical, full_frames):
    if event['request_id'] in train_requests:
        return 'same_teacher_request'
    if keys_for(event['text'], event['scope'], obs) & train_keys:
        return 'same_literal_query_track_any_scope_overlap'
    physical = {(int(obs[i, 0]), int(obs[i, 1])) for i in event['scope']}
    if physical & aux_physical:
        return 'AUX_actual_physical_scope_any_query'
    if any(int(obs[i, 0]) in full_frames for i in event['scope']):
        return 'training_or_AUX_full_frame_image_overlap'
    return None


def sigmoid(logits):
    e = np.exp(-np.abs(np.asarray(logits, dtype=float)))
    return np.where(np.asarray(logits) >= 0, 1/(1+e), e/(1+e))


def forecast(scores, event, channel, mode):
    if mode not in ('anchor', 'scope', 'channel_scope'):
        raise ValueError(mode)
    nodes = [event['node']] if mode == 'anchor' else event['scope']
    state = float(np.mean(sigmoid(scores[event['q'], nodes])))
    if mode == 'anchor':
        return state
    if mode == 'scope':
        return state
    sensitivity, specificity = channel[VIEWS.index(event['view'])]
    return float((1-specificity)+(sensitivity+specificity-1)*state)


def failure_gate(primary_eligible, parent_eligible):
    if type(primary_eligible) is not bool or type(parent_eligible) is not bool:
        raise ValueError('Expected complete reproduced Boolean primary conditions')
    return not (primary_eligible and parent_eligible)


def contract():
    obs = np.array([[1, 7], [5, 7], [8, 7], [8, 9]])
    tracks = {7: np.array([0, 1, 2]), 9: np.array([3])}
    actual, distinct = scope_nodes(obs, 2, 'CLIP', 2, tracks)
    assert actual == [0, 0, 1, 2] and distinct == [0, 1, 2]
    event = dict(q=0, node=2, scope=distinct, text=' WHITE  shirt ', request_id='new', view='CLIP')
    k = keys_for('white shirt', [1], obs)
    assert rejected_reason(event, obs, k, set(), set(), set()) == 'same_literal_query_track_any_scope_overlap'
    assert rejected_reason(event, obs, set(), {'new'}, set(), set()) == 'same_teacher_request'
    assert rejected_reason(event, obs, set(), set(), {(1, 7)}, set()) == 'AUX_actual_physical_scope_any_query'
    assert rejected_reason(event, obs, set(), set(), {(1, 9)}, set()) is None
    assert rejected_reason(event, obs, set(), set(), set(), {5}) == 'training_or_AUX_full_frame_image_overlap'
    p = np.array([[.1, .4, .7, .9]])
    scores = np.log(p/(1-p))
    c = np.tile([.8, .9], (3, 1))
    assert abs(forecast(scores, event, c, 'anchor')-.7) < 1e-14
    assert abs(forecast(scores, event, c, 'scope')-.4) < 1e-14
    assert abs(forecast(scores, event, c, 'channel_scope')-.38) < 1e-14
    assert not failure_gate(True, True) and failure_gate(False, True) and failure_gate(True, False)
    assert np.isfinite(sigmoid(np.array([-1000., 0., 1000.]))).all()
    return dict(status='PASS_SCOPE_ALIAS_AUX_CONTEXT_CHANNEL_AND_ACTIVATION_CONTRACT', checks=11,
                no_target_inputs_models_GT_GPU_or_teacher=True)


def prepare():
    from flex_uda.a100.mechanism import generation_guard
    guard = generation_guard()
    if PLAN.exists():
        raise FileExistsError(PLAN)
    assert not (MAIN/'quality_results.json').exists(), 'Fix diagnostic before current quality access'
    current = read(MAIN/'registration.json')
    paths = {Path(__file__), MAIN/'registration.json', PARENT/'registration.json', AUX}
    cases = []
    for case in current['cases']:
        case = dict(case)
        case['active_answers'] = str(Path(case['parent'])/'active_answers.jsonl')
        case['active_draws'] = str(Path(case['parent'])/'primary/draws.json')
        case['auxiliary_keys'] = str(Path(case['auxiliary_path']).parent/'row_keys.json')
        for key in ('observations', 'temporal_queries', 'uniform_answers', 'uniform_draws',
                    'active_answers', 'active_draws', 'auxiliary_keys'):
            paths.add(Path(case[key]))
        folder = MAIN/'units'/case['pair']/case['evaluated_video']
        if (folder/'generation_complete.json').exists():
            provenance = folder/'primary/teacher_event_provenance.json'
            assert provenance.exists()
            case['provenance_check'] = str(provenance); paths.add(provenance)
        cases.append(case)
    for name, expected in current['source_sha256'].items():
        assert file_sha(ROOT/name) == expected, name
    plan = dict(status='REGISTERED_PROSPECTIVE_FAILURE_ONLY_TEACHER_FORECAST', cases=cases,
        inputs={str(p): file_sha(p) for p in sorted(paths)}, original48_source_sha256=current['source_sha256'],
        contract=contract(), roles=ROLES, seed_grids=0, new_fits_teacher_GPU_or_GT=False,
        inventory='All148 units and every query, including zero supported queries; no model predictions or teacher probability metrics are read for inventory.',
        exclusions='Exclude same request, normalized literal-query/same-track ANY actual scope overlap across views; AUX actual crop/CLIP physical scope regardless of query; and any image frame used by training or AUX FRAME_BOX regardless of track/query. These conservative exclusions are fixed, not chosen for favorable performance.',
        activation='Evaluate only after main3256 and supplement1036 barriers, both complete official analyses and independent reproductions, and at least one fixed primary condition fails. A successful primary proceeds to contribution validation instead.',
        prediction='Binary teacher decision y=(p_yes>=.5). Current latent-state outputs forecast Yes via learned per-view noise channel and UNIFORM PRIOR over DISTINCT held-out input times, never answer-conditioned responsibilities. Raw soft KD forecasts at nominal anchor and fixed mean over the same scope are separately named.',
        pooling='Report event means and equal-weight supported-query means, plus all-query support and empty units. No frame-uniform or independence assumption, bootstrap or new training seed.',
        limitations=['Same-video transductive DEV, not a new independent semantic or image split.',
            'All images/features and other teacher questions were available to adaptation; different queries can be semantically dependent.',
            'Teacher forecasting is not semantic accuracy, HOTA, teacher calibration or proof of causal RSI.',
            'Uniform-time CLIP and noisy decision channel are explicit unverified observation assumptions.',
            'Parent soft/shared controls and continued soft/shared controls distinguish extra fitting from memory readout; no new methods or checkpoints are chosen.'],
        no_adaptation_or_primary_changes=True, goal_status='ACTIVE')
    assert not guard
    atomic_json(PLAN, plan)
    print(json.dumps(dict(status=plan['status'],units=len(cases),inputs=len(paths))),flush=True)


def load_plan():
    plan = read(PLAN)
    for path, expected in plan['inputs'].items():
        assert file_sha(path) == expected, path
    for name, expected in plan['original48_source_sha256'].items():
        assert file_sha(ROOT/name) == expected, name
    return plan


def events(case, role, obs, queries):
    draws = read(case[role+'_draws'])['draws']
    rows = [json.loads(line) for line in Path(case[role+'_answers']).read_text().splitlines() if line]
    journal = {r['uid']: r for r in rows}
    assert len(journal) == len(rows) == len(draws) and set(journal) == {r['uid'] for r in draws}
    tracks = {int(t): np.flatnonzero(obs[:, 1] == t) for t in np.unique(obs[:, 1])}
    delta = {'KITTI-V2': 1, 'Dance': 2, 'SportsMOT': 3}[case['target']]
    result = []
    for draw in draws:
        row = journal[draw['uid']]
        assert row['frozen_weight_sha256'] == T0 and row['probability'] == draw['probability']
        q, node = divmod(int(row['uid']), len(obs))
        assert q == int(row['query']) == int(draw['query']) and 0 <= q < len(queries)
        actual, scope = scope_nodes(obs, node, row['effective_view'], delta, tracks)
        result.append(dict(uid=row['uid'], q=q, node=node, text=queries[q]['query_text'],
            view=row['effective_view'], scope=scope, actual=actual, request_id=row['teacher_request_id']))
    return result


def footprint(case, aux_rows):
    obs = np.load(case['observations'], allow_pickle=False)
    queries = read(case['temporal_queries'])['queries']
    train, heldout = events(case, 'uniform', obs, queries), events(case, 'active', obs, queries)
    if 'provenance_check' in case:
        original = read(case['provenance_check'])['events']
        assert len(original) == len(train)
        for old, event in zip(original, train):
            assert old['uid'] == event['uid']
            assert old['actual_input_observation_nodes'] == event['actual']
            assert old['distinct_scope_nodes'] == event['scope']
    train_keys, requests, aux_physical, full_frames = set(), set(), set(), set()
    for event in train:
        train_keys.update(keys_for(event['text'], event['scope'], obs)); requests.add(event['request_id'])
        if event['view'] == 'FRAME_BOX':
            full_frames.update(int(obs[i, 0]) for i in event['scope'])
    aux_keys = {tuple(k) for k in read(case['auxiliary_keys'])['keys'] if k[0] == case['evaluated_video']}
    seen_aux = set()
    for row in aux_rows:
        key = row['video_id'], row['query_id'], int(row['track_id']), int(row['frame_id'])
        if row['target'] != case['target'] or key not in aux_keys:
            continue
        seen_aux.add(key)
        frames = row['clip']['frames'] if row['effective_view'] == 'CLIP' else [int(row['frame_id'])]
        assert frames
        aux_physical.update((int(t), int(row['track_id'])) for t in frames)
        if row['effective_view'] == 'FRAME_BOX':
            full_frames.update(map(int, frames))
    assert seen_aux == aux_keys, (case['pair'], case['evaluated_video'], aux_keys-seen_aux)
    keep, rejected = [], Counter()
    for event in heldout:
        reason = rejected_reason(event, obs, train_keys, requests, aux_physical, full_frames)
        if reason:
            rejected[reason] += 1
        else:
            keep.append(event)
    support = Counter(e['q'] for e in keep)
    return obs, dict(pair=case['pair'], video=case['evaluated_video'], queries=len(queries),
        active_events=len(heldout), uniform_training_events=len(train), retained_events=len(keep),
        supported_queries=len(support), empty_queries=len(queries)-len(support),
        rejections=dict(rejected), retained=keep)


def inventory():
    from flex_uda.a100.mechanism import generation_guard
    guard = generation_guard(); plan = load_plan()
    output = DEST/'inventory.json'
    if output.exists():
        raise FileExistsError(output)
    opened = restrict_io([PLAN, *plan['inputs'], *(ROOT/n for n in plan['original48_source_sha256'])])
    aux = [json.loads(s) for s in AUX.read_text().splitlines() if s]
    rows = [footprint(c, aux)[1] for c in plan['cases']]
    summary = {}
    for pair in sorted({r['pair'] for r in rows}):
        own = [r for r in rows if r['pair'] == pair]
        summary[pair] = {k: sum(r[k] for r in own) for k in
            ('queries', 'active_events', 'uniform_training_events', 'retained_events', 'supported_queries', 'empty_queries')}
        summary[pair]['videos'] = len(own)
        summary[pair]['rejections'] = dict(sum((Counter(r['rejections']) for r in own), Counter()))
    assert not guard
    load_plan()
    result = dict(status='COMPLETE148_MODEL_FREE_TEACHER_SCOPE_FOOTPRINT', units=len(rows),
        settings=summary, rows=rows, registration_sha256=file_sha(PLAN),
        no_RMOT_model_prediction_GT_or_quality_read=True, new_fits_teacher_GPU_or_seeds=0,
        completed_provenance_units_verified=sum('provenance_check' in c for c in plan['cases']),
        strict_declared_data_IO=True, opened_files=len(opened))
    atomic_json(output, result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False),flush=True)


def evaluate():
    from flex_uda.a100.mechanism import generation_guard
    guard = generation_guard(); plan = load_plan()
    if (DEST/'result.json').exists():
        raise FileExistsError(DEST/'result.json')
    barriers = [(MAIN/'all3256_sealed_before_GT.json', 3256),
                (PARENT/'all1036_sealed_before_GT.json', 1036)]
    for path, count in barriers:
        barrier = read(path); assert len(barrier['tasks']) == count and barrier['GT_open_attempts'] == 0
    a = PUBLIC/'conditional_state_primary_20261127_analysis.json'
    b = PUBLIC/'conditional_state_parent_controls_20261127_analysis.json'
    reproductions = []
    for stem, status in [('conditional_state_primary_20261127','PASS_INDEPENDENT_ALL132_HOTA_AND126_EFFECTS'),
                         ('conditional_state_parent_controls_20261127','PASS_INDEPENDENT_ALL60_PARENT_CONTROL_HOTA_AND54_EFFECTS')]:
        path = PUBLIC/(stem+'_reproduction.json')
        assert read(path)['status'] == status
        reproductions.append(path)
    if not failure_gate(read(a)['candidate_eligible_for_contribution_validation'], read(b)['additional_parent_control_condition_met']):
        print(json.dumps(dict(status='SKIP_POSITIVE_PRIMARY_PROCEED_TO_CONTRIBUTIONS')),flush=True)
        return
    own = read(DEST/'inventory.json'); assert own['registration_sha256'] == file_sha(PLAN)
    paths = {PLAN, DEST/'inventory.json', a, b, *reproductions, *(p for p,_ in barriers)}
    specs = [('PRIMARY_CHANNEL_SCOPE', MAIN, 'CONDITIONAL_STATE_MEMORY', 'channel_scope'),
        ('SOFT_KD_ANCHOR', MAIN, 'SOFT_KD', 'anchor'), ('SOFT_KD_SCOPE', MAIN, 'SOFT_KD', 'scope'),
        ('SOFT_SHARED_CHANNEL_SCOPE', MAIN, 'SOFT_SHARED_MEMORY', 'channel_scope'),
        ('PARENT_SOFT_KD_ANCHOR', None, None, 'anchor'), ('PARENT_SOFT_KD_SCOPE', None, None, 'scope'),
        ('PARENT_SOFT_SHARED_CHANNEL_SCOPE', PARENT, 'PARENT_SOFT_SHARED_MEMORY', 'channel_scope')]
    for case in plan['cases']:
        paths.add(MAIN/'units'/case['pair']/case['evaluated_video']/'generation_complete.json')
        for _, root, method, _ in specs:
            paths.add(Path(case['original_scores']) if root is None else
                root/'units'/case['pair']/case['evaluated_video']/'predictions'/method/'scores.npy')
    snapshots = {str(p): file_sha(p) for p in sorted(paths)}
    snapshot_path = DEST/'evaluation_inputs_before_forecasts.json'
    if snapshot_path.exists():
        assert read(snapshot_path)['inputs'] == snapshots, 'Preserve previous failed diagnostic input snapshot'
    else:
        atomic_json(snapshot_path, dict(inputs=snapshots,
            timing='After complete study barriers, before diagnostic forecasts; not proof of original preGT score-byte sealing.'))
    opened = restrict_io([PLAN, *plan['inputs'], *paths, *(ROOT/n for n in plan['original48_source_sha256'])])
    by_unit = {(r['pair'],r['video']):r for r in own['rows']}
    sums = defaultdict(lambda: np.zeros((len(ROLES), 5)))
    support = defaultdict(Counter)
    for case in plan['cases']:
        row = by_unit[case['pair'],case['evaluated_video']]
        obs = np.load(case['observations'], allow_pickle=False)
        labels = {r['uid']:float(r['probability']) >= .5 for r in
            (json.loads(s) for s in Path(case['active_answers']).read_text().splitlines() if s)}
        channel = np.asarray(read(MAIN/'units'/case['pair']/case['evaluated_video']/'generation_complete.json')['primary_fit']['rounds'][-1]['channel'])
        assert channel.shape == (3,2) and np.all((channel > .5)&(channel < .99))
        y = np.asarray([labels[e['uid']] for e in row['retained']], dtype=float)
        support[case['pair']].update(queries=row['queries'],supported_queries=row['supported_queries'],events=len(y),videos=1)
        cache = {}
        for j, (_, root, method, mode) in enumerate(specs):
            p = Path(case['original_scores']) if root is None else root/'units'/case['pair']/case['evaluated_video']/'predictions'/method/'scores.npy'
            if p not in cache:
                scores=np.load(p,mmap_mode='r',allow_pickle=False)
                assert scores.shape == (row['queries'],len(obs))
                cache[p]=scores
            scores = cache[p]
            predictions=[forecast(scores,event,channel,mode) for event in row['retained']]
            p=np.clip(np.asarray(predictions),1e-12,1-1e-12)
            brier=(p-y)**2; bce=-y*np.log(p)-(1-y)*np.log1p(-p)
            q_brier=[];q_bce=[]
            for q in sorted({e['q'] for e in row['retained']}):
                ids=[i for i,e in enumerate(row['retained']) if e['q']==q]
                q_brier.append(brier[ids].mean());q_bce.append(bce[ids].mean())
            sums[case['pair']][j]+=np.array([len(y),brier.sum(),bce.sum(),sum(q_brier),sum(q_bce)])
    settings={}
    for pair, values in sums.items():
        settings[pair]=dict(support=support[pair],roles={})
        for name, v in zip(ROLES, values):
            n,q=v[0],support[pair]['supported_queries']
            settings[pair]['roles'][name]=dict(events=int(n),hard_teacher_Brier=float(v[1]/n) if n else None,
                hard_teacher_BCE=float(v[2]/n) if n else None,
                supported_query_macro_Brier=float(v[3]/q) if q else None,
                supported_query_macro_BCE=float(v[4]/q) if q else None)
    assert not guard
    for p, expected in snapshots.items():assert file_sha(p)==expected,p
    load_plan()
    result=dict(status='COMPLETE_POSTSEAL_FAILURE_ONLY_TEACHER_DECISION_FORECAST',settings=settings,
        new_fits_teacher_GPU_or_seeds=0,GT_open_attempts=0,primary_or_threshold_changes=False,
        not_semantic_accuracy_HOTA_independent_generalization_or_RSI_causal_evidence=True,
        strict_declared_data_IO=True,opened_files=len(opened),goal_status='ACTIVE',time_unix=time.time())
    atomic_json(DEST/'result.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); group=parser.add_mutually_exclusive_group(required=True)
    for flag in ('prepare','contract','inventory','evaluate'):group.add_argument('--'+flag,action='store_true')
    args=parser.parse_args()
    if args.contract:print(json.dumps(contract()),flush=True)
    elif args.prepare:prepare()
    elif args.inventory:inventory()
    else:evaluate()
