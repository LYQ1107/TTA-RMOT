"""Metadata support for controlled candidate comparisons, not a new method.

The quotient graph has provided Track IDs as nodes. Its rank statement assumes
one static potential per track; it does not establish that RMOT membership is
static, that teacher biases cancel, or that a shared neural model cannot
generalize to graph-isolated tracks. No answers or image pixels are opened.
"""
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, metadata, generation_guard
from flex_uda.active.acquisition import route
from flex_uda.active.native_binary_inputs import inventory
from flex_uda.teacher.decompose_inputs import PairedViews
from scripts.a100_six_direction import OUT as SIX, PUBLIC

PLAN = PUBLIC/'same_context_comparison_support_registration_20261124.json'
OUT = SIX/'same_context_comparison_support_20261124'
SOURCES = ('scripts/a100_same_context_comparison_support.py',
    'flex_uda/teacher/decompose_inputs.py', 'scripts/active_run.py',
    'flex_uda/active/native_binary_inputs.py', 'flex_uda/active/acquisition.py',
    'flex_uda/a100/mechanism.py')


def support(tracks, groups):
    tracks = sorted(tracks); parent = {t:t for t in tracks}; edges = set()
    touched = set(); requests = set(); forest = []; observation_nodes = 0
    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]; t = parent[t]
        return t
    for (frame, view), values in sorted(groups.items()):
        ids = sorted(set(values)); observation_nodes += len(ids)
        if len(ids) < 2:
            continue
        touched.update(ids); edges.update(combinations(ids, 2))
        for other in ids[1:]:
            a, b = find(ids[0]), find(other)
            if a != b:
                parent[b] = a
                forest.append([frame, view, ids[0], other])
                requests.update(((frame, view, ids[0]), (frame, view, other)))
    sizes = Counter(find(t) for t in tracks); components = len(sizes)
    assert len(forest) == len(tracks)-components
    return dict(candidate_tracks=len(tracks), context_groups=len(groups),
        observation_nodes=observation_nodes,
        possible_pair_actions=sum(len(set(v))*(len(set(v))-1)//2 for v in groups.values()),
        unique_quotient_edges=len(edges), pairable_tracks=len(touched),
        no_same_context_peer_tracks=len(tracks)-len(touched),
        quotient_components=components, quotient_rank=len(forest),
        largest_component_tracks=max(sizes.values()),
        observation_graph_rank=observation_nodes-len(groups),
        diagnostic_forest_unique_binary_requests=len(requests), forest=forest)


def view_groups(observations, requested_view, first, delta):
    groups = defaultdict(set)
    for frame, track in observations:
        effective = 'CROP' if requested_view == 'CLIP' and frame-first[track] < delta else requested_view
        groups[int(frame), effective].add(int(track))
    return groups


def clip_source_check(observations, first, delta):
    # Invoke the actual existing implementation, replacing only pixel loading.
    builder = object.__new__(PairedViews); key = ('metadata', 'only')
    builder.video_specs = {key:dict(delta=delta)}; builder.tracks = {}
    builder.patch = lambda _, ob: ob.frame_id
    for track in first:
        times = sorted(int(f) for f,t in observations if t == track)
        builder.tracks[key, track] = [SimpleNamespace(frame_id=f) for f in times]
    checks = 0
    for track in sorted(first):
        obs = builder.tracks[key, track]
        for i in sorted({0, len(obs)//2, len(obs)-1}):
            frame = obs[i].frame_id; crops, info = builder.clip(key, frame, track)
            assert info['fallback'] == (frame-first[track] < delta)
            assert all(t <= frame for t in crops)
            checks += 1
    return checks


def main():
    plan = json.loads(PLAN.read_text())
    assert all(file_sha(ROOT/p) == sha for p,sha in plan['source_sha256'].items())
    guard = generation_guard(); full = {}; details = []; settings = {}; clip_checks = 0
    from scripts.active_run import video_spec
    for target in dict.fromkeys(target for _,target in PAIRS.values()):
        md = metadata(target)
        for video, bank in sorted(md['videos'].items()):
            observations = [tuple(map(int,row)) for row in bank['observations']]
            assert len(observations) == len(set(observations))
            tracks = sorted({t for _,t in observations})
            first = {t:min(f for f,other in observations if other == t) for t in tracks}
            delta = video_spec(target, video, bank)['delta']
            clip_checks += clip_source_check(observations, first, delta)
            views = {}
            for requested in ('CROP','FRAME_BOX','CLIP'):
                groups = view_groups(observations, requested, first, delta)
                row = support(tracks, groups)
                # Actual full-video global ID bijection: topology is unchanged.
                rename = dict(zip(tracks, reversed(tracks)))
                remapped = support(rename.values(), {k:[rename[t] for t in v] for k,v in groups.items()})
                invariant = set(row)-{'forest', 'diagnostic_forest_unique_binary_requests'}
                assert all(row[k] == remapped[k] for k in invariant)
                views[requested] = row
            full[target,video] = dict(first=first, delta=delta, tracks=tracks, views=views)
    for pair, (_, target) in PAIRS.items():
        totals = Counter(); full_components = []; library_components = []
        for video in sorted(metadata(target)['videos']):
            actions, quotas = inventory(pair,video); raw = full[target,video]
            queries = [q for q in metadata(target)['episodes'] if q['video_id'] == video]
            grouped = defaultdict(lambda:defaultdict(set)); views = defaultdict(set)
            for action in actions.values():
                q = action['local_query']; case = action['case']; t = int(case['track_id']); f = int(case['frame_id'])
                requested = case['view']; views[q].add(requested)
                effective = 'CROP' if requested == 'CLIP' and f-raw['first'][t] < raw['delta'] else requested
                grouped[q][f,effective].add(t)
            for q,budget in sorted(quotas.items()):
                assert views[q] == {route(queries[q]['query_text'])}
                requested = next(iter(views[q])); entire = raw['views'][requested]
                library = support(raw['tracks'],grouped[q])
                totals.update(query_episodes=1, binary_request_budget=budget,
                    candidate_query_tracks=len(raw['tracks']),
                    full_pairable_query_tracks=entire['pairable_tracks'],
                    library_pairable_query_tracks=library['pairable_tracks'],
                    full_no_peer_query_tracks=entire['no_same_context_peer_tracks'],
                    library_no_peer_query_tracks=library['no_same_context_peer_tracks'],
                    full_quotient_rank=entire['quotient_rank'], library_quotient_rank=library['quotient_rank'],
                    queries_with_zero_library_comparisons=library['quotient_rank'] == 0,
                    queries_with_connected_full_quotient=entire['quotient_components'] == 1,
                    budget_below_all_track_touch_lower_bound=budget < len(raw['tracks']),
                    budget_can_pay_full_diagnostic_forest=budget >= entire['diagnostic_forest_unique_binary_requests'])
                full_components.append(entire['quotient_components']); library_components.append(library['quotient_components'])
                details.append(dict(pair=pair,video=video,local_query=q,requested_view=requested,
                    binary_request_budget=budget,full_quotient_components=entire['quotient_components'],
                    full_pairable_tracks=entire['pairable_tracks'], library=library))
        settings[pair] = dict(**totals, full_quotient_components_median=float(np.median(full_components)),
            library_quotient_components_median=float(np.median(library_components)))
        print(json.dumps(dict(setting=pair,summary=settings[pair])),flush=True)
    assert len(full) == 74 and len(details) == 3506 and not guard
    local = OUT/'details.json'
    atomic_json(local,dict(physical_videos=[dict(target=t,video=v,**r) for (t,v),r in full.items()], queries=details))
    result = dict(status='COMPLETED_ALL_SIX_SAME_CONTEXT_COMPARISON_METADATA_SUPPORT_NO_GT',
        settings=settings, physical_videos=len(full), setting_video_units=148, query_episodes=len(details),
        actual_PairedViews_clip_metadata_checks=clip_checks,
        rank_statement='For arbitrary static per-Track-ID potentials and incidence B, rank(B)=n-components. This is not the identifiability rank of the shared nonlinear visual model.',
        dynamic_membership_and_teacher_context_bias_cancellation_not_established=True,
        graph_isolation_does_not_imply_unlearnable_with_shared_visual_features=True,
        same_frame_view_is_metadata_comparability_not_identical_pixels_or_semantic_correctness=True,
        diagnostic_forest_is_not_a_proposed_policy_or_optimal_question_budget=True,
        binary_teacher_requests_per_new_pair=2, no_free_paired_answer=True,
        no_new_pool_injection_into_live_studies=True, no_teacher_answers_images_features_fits_predictions_or_HOTA=True,
        manual_GT_open_attempts=0, new_GPU_calls=0, local_details=str(local), local_details_sha256=file_sha(local),
        source_sha256=plan['source_sha256'], goal_status='ACTIVE')
    atomic_json(PUBLIC/'same_context_comparison_support_20261124.json',result)
    print(json.dumps(dict(status=result['status'],physical_videos=len(full),query_episodes=len(details))),flush=True)


if __name__ == '__main__':
    main()
