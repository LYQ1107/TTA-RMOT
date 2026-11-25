"""Audit available question actions without reading teacher answers or GT."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, generation_guard, metadata
from scripts.a100_joint_observation_collect import OUT as COLLECTION
from scripts.a100_six_direction import PUBLIC, OUT as SIX


def main():
    guard = generation_guard()
    aggregate = {p: Counter() for p in PAIRS}
    leaves = {}
    for target in sorted({t for _, t in PAIRS.values()}):
        md = metadata(target)
        for video, bank in sorted(md['videos'].items()):
            path = COLLECTION/'inputs'/target/video/'manifest.json'
            doc = json.loads(path.read_text())
            assert doc.get('GT_open_attempts', 0) == 0
            tracks = Counter(int(t) for _, t in bank['observations'])
            # Arithmetic action-space bound only: three views of each real
            # unordered pair in the <=3-observation reference pool. Singleton
            # tracks contribute one real-image request per view, not fake pairs.
            per_query = sum(3*max(math.comb(min(3, count), 2), 1) for count in tracks.values())
            owners = defaultdict(dict)
            for owner, rows in doc['own'].items():
                pair, policy = owner.split('/')
                owners[pair][policy] = rows
            for pair, policies in owners.items():
                by = {}
                for policy in ('DESIGN', 'UNIFORM'):
                    by[policy] = defaultdict(set)
                    for row in policies[policy]:
                        assert row['uid'] in doc['cases']
                        by[policy][int(row['local_query'])].add(row['uid'])
                    assert sum(map(len, by[policy].values())) == len(policies[policy])
                assert set(by['DESIGN']) == set(by['UNIFORM'])
                counts = Counter(videos=1, candidate_tracks=len(tracks),
                                 query_blocks=len(by['DESIGN']),
                                 three_view_reference_pool_arithmetic_bound=per_query*len(by['DESIGN']))
                details = []
                for qi in sorted(by['DESIGN']):
                    a, b = by['DESIGN'][qi], by['UNIFORM'][qi]
                    assert len(a) == len(b) and len(a | b) <= per_query
                    counts.update(design_bundles=len(a), uniform_bundles=len(b),
                                  union_bundles=len(a | b), intersection_bundles=len(a & b),
                                  identical_policy_query_sets=int(a == b),
                                  additional_known_choices_over_one_budget=len(a | b)-len(a),
                                  not_in_existing_union_arithmetic_bound=per_query-len(a | b))
                    details.append(dict(local_query=qi, budget=len(a), union=len(a | b),
                                        same_set=a == b, potential_action_bound=per_query))
                aggregate[pair].update(counts)
                leaves[pair+'/'+video] = dict(counts=dict(counts), query_blocks=details,
                                             input_manifest_sha256=file_sha(path))
    summary = {}
    for pair, counter in aggregate.items():
        row = dict(counter)
        row['existing_union_over_one_policy_budget'] = row['union_bundles']/row['design_bundles']
        row['identical_policy_query_fraction'] = row['identical_policy_query_sets']/row['query_blocks']
        row['possible_RSI_replay_selection_change'] = row['additional_known_choices_over_one_budget'] > 0
        summary[pair] = row
    assert not guard
    report = dict(status='PASS_RSI_UNLABELLED_ACTION_SPACE_AUDIT', settings=summary,
                  teacher_answers_read=0, new_teacher_or_student_calls=0,
                  GT_open_attempts=0, RMOT_quality_or_RSI_improvement_claim=False,
                  action_bound_warning='Arithmetic only; assumes each view request is feasible. Not a teacher accuracy or useful-information bound.',
                  interpretation='Existing union permits some policy replay, not an unlimited environment. Only requested QA may reach a future learner; missing actions need actual teacher inference.',
                  source_sha256=file_sha(Path(__file__)), goal_status='ACTIVE')
    out = SIX/'rsi_action_space_audit_20261123'
    atomic_json(out/'per_video_unlabelled_inputs.json', leaves)
    atomic_json(out/'analysis.json', report)
    atomic_json(PUBLIC/'rsi_action_space_audit.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
