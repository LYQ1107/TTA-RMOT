"""Post-seal old-joint stability attribution, never imported by adaptation.

Reuse exactly the GT files linked by completed official evaluations. The
IoU-.5 membership/visible-ID diagnostic is not official HOTA or an additive
causal error decomposition. No filters, new predictions, or fits are made.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, metadata
from flex_uda.active.acquisition import expression_types
from scripts.a100_joint_observation_six_setting import OUT, PRIMARY, PUBLIC

BASE = 'COARSE_UNIT_SOFT_DESIGN_FINAL'
METHODS = (PRIMARY, BASE, 'BINARY_CONDITIONAL_REFERENCE_DESIGN_FINAL',
    'FULL_LIKELIHOOD_REFERENCE_DESIGN_FINAL', 'FULL_CONDITIONAL_LOCAL_DESIGN_FINAL',
    PRIMARY+'__AT_'+BASE+'_COUNT', BASE+'__AT_'+PRIMARY+'_COUNT')
DEST = OUT/'stability_diagnosis_20261124'


def truth_arrays(pair, video, bank, queries):
    """One target-level cache; official candidates/queries are source-independent."""
    from scripts.analyze_pf_full_offline import overlaps, matched_pairs
    target = PAIRS[pair][1]; folder = DEST/'offline_GT'/target/video
    observations = np.asarray(bank['observations'], dtype=int)
    source_key = hashlib.sha256(json.dumps(dict(observations=bank['observations'],
        lines=bank['lines'], queries=[q['expression_id'] for q in queries]), separators=(',', ':')).encode()).hexdigest()
    if (folder/'receipt.json').is_file():
        receipt = json.loads((folder/'receipt.json').read_text())
        assert receipt['candidate_query_signature'] == source_key
        return np.load(folder/'labels.npy'), np.load(folder/'union_ids.npy')
    truths = []; union = defaultdict(dict); digests = []
    for query in queries:
        path = OUT/pair/'seed0'/PRIMARY/'results'/video/query['expression_id']/'gt.txt'
        data = path.read_bytes(); digests.append(hashlib.sha256(data).hexdigest())
        truth = defaultdict(list)
        for line in data.decode().splitlines():
            fields = line.strip().split(',')
            assert len(fields) >= 8
            if int(float(fields[6])) == 0 or int(float(fields[7])) != 1:
                continue
            frame, gid = int(float(fields[0])), int(float(fields[1]))
            x, y, w, h = map(float, fields[2:6]); box = [x, y, x+w, y+h]
            if gid in union[frame]:
                assert union[frame][gid] == box, (pair, video, frame, gid, 'Inconsistent physical GT ID')
            union[frame][gid] = box; truth[frame].append((gid, box))
        truths.append(truth)
    raw = [line.split(',') for line in bank['lines']]
    frames = np.asarray([int(float(v[0])) for v in raw])
    boxes = np.asarray([[float(x) for x in v[2:6]] for v in raw]); boxes[:, 2:] += boxes[:, :2]
    labels = np.full((len(queries), len(observations)), -1, dtype='int32')
    union_ids = np.full(len(observations), -1, dtype='int32')
    for frame in np.unique(frames):
        ids = np.flatnonzero(frames == frame); gt = list(union.get(int(frame), {}).items())
        columns = {gid: i for i, (gid, _) in enumerate(gt)}
        iou = overlaps(boxes[ids], [box for _, box in gt])
        for i, j in matched_pairs(iou, list(range(len(ids)))).items():
            union_ids[ids[i]] = gt[j][0]
        for qi, truth in enumerate(truths):
            own = truth.get(int(frame), [])
            if not own:
                continue
            selected = [columns[gid] for gid, _ in own]
            for i, j in matched_pairs(iou[:, selected], list(range(len(ids)))).items():
                labels[qi, ids[i]] = own[j][0]
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder/'labels.npy', labels); np.save(folder/'union_ids.npy', union_ids)
    atomic_json(folder/'receipt.json', dict(candidate_query_signature=source_key,
        source_completed_pair=pair, actual_official_GT_files=len(digests),
        ordered_GT_hash_list_sha256=hashlib.sha256('\n'.join(digests).encode()).hexdigest(),
        diagnostic_matching='IoU>=.5 maximum valid cardinality then IoU; class1/mark filtering',
        no_labels_or_settings_returned_to_adaptation=True))
    return labels, union_ids


def mask(folder, video, bank, queries):
    saved = folder/'videos'/video/'mask.npy'
    if saved.is_file():
        result = np.load(saved)
        assert result.shape == (len(queries), len(bank['observations'])) and result.dtype == bool
        return result
    lookup = {}
    for i, line in enumerate(bank['lines']):
        fields = line.split(','); key = int(float(fields[0])), int(float(fields[1]))
        assert key not in lookup; lookup[key] = i
    result = np.zeros((len(queries), len(lookup)), dtype=bool)
    for qi, query in enumerate(queries):
        for line in (folder/'results'/video/query['expression_id']/'predict.txt').read_text().splitlines():
            fields = line.split(','); i = lookup[int(float(fields[0])), int(float(fields[1]))]
            assert line == bank['lines'][i] and not result[qi, i]
            result[qi, i] = True
    return result


def count_transitions(bits, labels, left, right, selected_queries):
    truth_a, truth_b = labels[:, left], labels[:, right]
    before, after = bits[:, left], bits[:, right]
    include = selected_queries[:, None]
    stable = (truth_a == truth_b) & include; change = (truth_a != truth_b) & include
    flipped = before != after
    positive = stable & truth_b; negative = stable & ~truth_b
    return Counter(stable_transitions=int(stable.sum()), stable_output_flips=int((stable & flipped).sum()),
        stable_positive_transitions=int(positive.sum()), stable_positive_flips=int((positive & flipped).sum()),
        stable_negative_transitions=int(negative.sum()), stable_negative_flips=int((negative & flipped).sum()),
        stable_endpoint_errors=int((stable & (before != truth_a)).sum()+(stable & (after != truth_b)).sum()),
        true_membership_change_transitions=int(change.sum()),
        membership_change_flips=int((change & flipped).sum()),
        membership_change_both_endpoints_correct=int((change & (before == truth_a) & (after == truth_b)).sum()),
        selected_rows=int(bits[selected_queries].sum()),
        selected_positive_rows=int((bits[selected_queries] & labels[selected_queries]).sum()),
        selected_negative_rows=int((bits[selected_queries] & ~labels[selected_queries]).sum()))


def main():
    barrier = json.loads((OUT/'all_predictions_sealed_before_GT.json').read_text())
    official = json.loads((OUT/'official.json').read_text())
    reproduction = json.loads((PUBLIC/'joint_observation_quality_seed0_reproduction.json').read_text())
    assert barrier['methods'] == official['methods'] == reproduction['statistical_records'] == 582
    assert barrier['GT_open_attempts'] == 0 and reproduction['maximum_absolute_error'] < 1e-8
    settings = {}; detail = []
    for pair, (_, target) in PAIRS.items():
        totals = {scope: {name: Counter() for name in METHODS} for scope in ('all_queries', 'appearance_only')}
        query_counts = Counter(); eligible_pairs = 0
        for video, bank in sorted(metadata(target)['videos'].items()):
            queries = [q for q in metadata(target)['episodes'] if q['video_id'] == video]
            labels, union_ids = truth_arrays(pair, video, bank, queries)
            labels = labels >= 0; obs = np.asarray(bank['observations'])
            assert labels.shape == (len(queries), len(obs)) and union_ids.shape == (len(obs),)
            left, right = [], []
            for track in np.unique(obs[:, 1]):
                ids = np.flatnonzero(obs[:, 1] == track); ids = ids[np.argsort(obs[ids, 0], kind='stable')]
                a, b = ids[:-1], ids[1:]; gap = obs[b, 0]-obs[a, 0]
                valid = (gap > 0) & (gap <= 2) & (union_ids[a] >= 0) & (union_ids[a] == union_ids[b])
                left.extend(a[valid]); right.extend(b[valid])
            left, right = np.asarray(left, dtype=int), np.asarray(right, dtype=int)
            eligible_pairs += len(left)
            appearance = np.asarray(['APPEARANCE' in expression_types(q['query_text']) and
                not set(expression_types(q['query_text'])) & {'MOTION', 'SPATIAL', 'ROLE'} for q in queries])
            subsets = dict(all_queries=np.ones(len(queries), dtype=bool), appearance_only=appearance)
            query_counts.update({k: int(v.sum()) for k, v in subsets.items()})
            masks = {}
            for name in METHODS:
                root = OUT/pair/'seed0'; folder = root/'countmatched'/name if '__AT_' in name else root/name
                masks[name] = mask(folder, video, bank, queries)
            order = np.argsort(obs[:, 0], kind='stable')
            boundaries = np.flatnonzero(np.diff(obs[order, 0]))+1
            for ids in np.split(order, boundaries):
                assert np.array_equal(masks[METHODS[-2]][:, ids].sum(1), masks[BASE][:, ids].sum(1))
                assert np.array_equal(masks[METHODS[-1]][:, ids].sum(1), masks[PRIMARY][:, ids].sum(1))
            for name, bits in masks.items():
                for scope, chosen in subsets.items():
                    row = count_transitions(bits, labels, left, right, chosen)
                    totals[scope][name].update(row)
                    detail.append(dict(pair=pair, video=video, scope=scope, method=name, **row))
            print(json.dumps(dict(pair=pair, completed_video=video, shared_physical_identity_edges=len(left))), flush=True)
        summary = {}
        for scope, methods in totals.items():
            summary[scope] = {name: dict(row,
                stable_flip_rate=row['stable_output_flips']/row['stable_transitions'] if row['stable_transitions'] else None,
                stable_endpoint_error_rate=row['stable_endpoint_errors']/(2*row['stable_transitions']) if row['stable_transitions'] else None,
                membership_change_two_endpoint_accuracy=row['membership_change_both_endpoints_correct']/row['true_membership_change_transitions'] if row['true_membership_change_transitions'] else None)
                for name, row in methods.items()}
        settings[pair] = dict(query_counts=dict(query_counts), physical_identity_edges=eligible_pairs, summaries=summary)
    local = DEST/'per_video.json'; atomic_json(local, dict(rows=detail))
    result = dict(status='COMPLETED_OLD_JOINT_POSTSEAL_STABILITY_DIAGNOSIS_ALL_SIX', settings=settings,
        methods=list(METHODS), adjacency='Same provided TrackID, same nonnegative union-GT identity, positive frame gap<=2',
        stable='Query membership unchanged at the two eligible endpoints, not a full interval guarantee',
        matching='Diagnostic IoU.5 matching, not official HOTA or a count/error additive decomposition',
        appearance_subset='Existing lexical APPEARANCE with no MOTION/SPATIAL/ROLE tag; no GT selection',
        official_annotation_sources='Actual gt.txt links used by the completed old joint primary official evaluation, including restored original KITTI files',
        old582_study_barrier_sha256=file_sha(OUT/'all_predictions_sealed_before_GT.json'),
        new_fits_teacher_calls_predictions_or_official_HOTA_evaluations=0,
        GT_used_only_postseal_offline=True, no_label_parameter_or_filter_written_to_adaptation=True,
        all_current_native_RSI_sources_unchanged=True, all_data_DEVELOPMENT=True,
        no_independent_frame_sample_or_universal_temporal_module_claim=True,
        local_detail=str(local), local_detail_sha256=file_sha(local), goal_status='ACTIVE')
    atomic_json(PUBLIC/'joint_observation_stability_20261124.json', result)
    print(json.dumps(dict(status=result['status'], scopes={p: r['query_counts'] for p, r in settings.items()})), flush=True)


if __name__ == '__main__':
    main()
