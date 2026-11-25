"""User-defined budget-matched YesTrack controls; never label unasked pairs."""
import hashlib
import numpy as np


def video_rng(video):
    # The preregistered seed0 derives a stable 32-bit video sub-seed. The same
    # derivation is reused by random acquisition and random coverage.
    root = np.random.RandomState(0)
    salt = int(root.randint(0, 2**32, dtype='uint32'))
    digest = int(hashlib.sha256(video.encode()).hexdigest()[:8], 16)
    return np.random.RandomState(salt ^ digest)


def budget_coverage(video, queries, track_ids, z, actual_full_calls, *, prior=False):
    pairs = [(q, t) for q in range(len(queries)) for t in range(len(track_ids))]
    if not 0 <= actual_full_calls <= len(pairs):
        raise ValueError('Registered rho<=1 FULL budget must not exceed the number of all candidate/query pairs')
    if prior:
        query_order = sorted(range(len(queries)), key=lambda q: queries[q]['expression_id'])
        rankings = {q: sorted(range(len(track_ids)), key=lambda t: (-z[q, t], track_ids[t])) for q in query_order}
        order = [(q, rankings[q][rank]) for rank in range(len(track_ids)) for q in query_order]
    else:
        order = [pairs[i] for i in video_rng(video).permutation(len(pairs))]
    return order[:actual_full_calls]


def keyframe_indices(observations, delta):
    chosen = [0]
    last = observations[0]['frame_id']
    for i, obs in enumerate(observations[1:], 1):
        if obs['frame_id']-last >= delta:
            chosen.append(i); last = obs['frame_id']
    return chosen


def inherit(probabilities, keyframes, length, threshold=.4):
    result = np.empty(length, dtype=bool)
    for j, start in enumerate(keyframes):
        stop = keyframes[j+1] if j+1 < len(keyframes) else length
        result[start:stop] = probabilities[j] >= threshold
    return result


def yestrack_video(target, video, queries, bank, native_margin, teacher, method, *, full_calls=None):
    from collections import defaultdict
    from flex_uda.active.belief import normalize_track_scores
    tracks = defaultdict(list); indices = defaultdict(list)
    for j, ((frame, track), line) in enumerate(zip(bank['observations'], bank['lines'])):
        indices[track].append(j)
        tracks[track].append(dict(frame_id=frame, track_id=track, bbox_xywh=list(map(float, line.split(',')))[2:6]))
    track_ids = sorted(tracks)
    if method == 'YESTRACK_FULL_D10':
        covered = [(q, t) for q in range(len(queries)) for t in range(len(track_ids))]
        delta = 10
    else:
        mean = np.stack([[native_margin[q, indices[t]].mean() for t in track_ids] for q in range(len(queries))])
        covered = budget_coverage(video, queries, track_ids, normalize_track_scores(mean), int(full_calls),
                                  prior=method == 'YESTRACK_BUDGET_PRIOR')
        delta = max(obs[-1]['frame_id']-obs[0]['frame_id'] for obs in tracks.values())+1
    # The Native fallback is left untouched for every unasked pair, including
    # in Dance: no additional presence gate/exclusivity may change those rows.
    selected = native_margin > 0
    requests = []; assignments = []
    for q, ti in covered:
        track = track_ids[ti]; keyframes = keyframe_indices(tracks[track], delta)
        for frame_index in keyframes:
            obs = tracks[track][frame_index]
            requests.append(dict(target=target, video_id=video, frame_id=obs['frame_id'], track_id=track,
                query_id=queries[q]['expression_id'], query_text=queries[q]['query_text'], view='CROP'))
        assignments.append((q, track, keyframes))
    # Bound service messages while preserving exactly the original questions.
    answers = []
    for a in range(0, len(requests), 128): answers.extend(teacher.ask_many(requests[a:a+128]))
    offset = 0
    for q, track, keyframes in assignments:
        probabilities = [a['yes_probability'] for a in answers[offset:offset+len(keyframes)]]
        selected[q, indices[track]] = inherit(probabilities, keyframes, len(indices[track]))
        offset += len(keyframes)
    if method != 'YESTRACK_FULL_D10' and len(answers) != full_calls:
        raise ValueError('Budget-matched control does not match actual FULL calls')
    covered_set = {(q, track_ids[t]) for q, t in covered}
    untouched = sum(np.array_equal(selected[q, rows], native_margin[q, rows] > 0)
                    for q in range(len(queries)) for track, rows in indices.items() if (q, track) not in covered_set)
    if untouched != len(queries)*len(track_ids)-len(covered): raise ValueError('Uncovered pair deviates from Native')
    return selected, dict(method=method, video=video, covered_pairs=len(covered),
        all_pairs=len(queries)*len(track_ids), coverage_fraction=len(covered)/(len(queries)*len(track_ids)),
        actual_calls=len(answers), target_full_calls=full_calls, delta=delta, threshold=.4,
        cadence='first observation, then first actual observation >=Delta frames after last call; user supplement',
        uncovered_native_exact=True, question_keys=[a['request_id'] for a in answers])


def uniform_video(target, video, queries, bank, teacher, budget, *, frame_presence=None, frame_index=None):
    from collections import defaultdict
    from flex_uda.active.acquisition import route, exclusive_group, expression_types
    tracks = defaultdict(list); positions = defaultdict(list)
    for j, ((frame, track), line) in enumerate(zip(bank['observations'], bank['lines'])):
        positions[track].append(j)
        tracks[track].append(dict(frame_id=frame, track_id=track, bbox_xywh=list(map(float, line.split(',')))[2:6]))
    tids = sorted(tracks)
    pairs = [(q, track) for q in range(len(queries)) for track in tids if len(tracks[track]) >= 3]
    if budget > len(pairs): raise ValueError('Registered rho<=1 uniform budget should have at most one question per eligible pair')
    selected_pairs = [pairs[i] for i in video_rng(video).choice(len(pairs), budget, replace=False)] if budget else []
    requests = []
    for q, track in selected_pairs:
        obs = tracks[track][(len(tracks[track])-1)//2]
        requests.append(dict(target=target, video_id=video, frame_id=obs['frame_id'], track_id=track,
            query_id=queries[q]['expression_id'], query_text=queries[q]['query_text'], view=route(queries[q]['query_text'])))
    answers = []
    for a in range(0, len(requests), 128): answers.extend(teacher.ask_many(requests[a:a+128]))
    probability = np.full((len(queries), len(tids)), np.nan)
    ti = {track: i for i, track in enumerate(tids)}
    for (q, track), answer in zip(selected_pairs, answers): probability[q, ti[track]] = answer['yes_probability'] >= .5
    positive = probability > .5
    if exclusive_group([q['query_text'] for q in queries], target):
        # Majority-vote ties are broken lexically; UNKNOWN is not a No label.
        for t in range(len(tids)):
            valid = np.flatnonzero(positive[:, t])
            if len(valid):
                winner = min(valid, key=lambda q: queries[q]['expression_id'])
                positive[:, t] = False; positive[winner, t] = True
    mask = np.zeros((len(queries), len(bank['observations'])), dtype=bool)
    for q, query in enumerate(queries):
        for track in tids:
            ids = positions[track]
            mask[q, ids] = positive[q, ti[track]]
            if frame_presence is not None and 'MOTION' in expression_types(query['query_text']):
                mask[q, ids] &= [frame_presence[q, frame_index[bank['observations'][i][0]]] >= .5 for i in ids]
    return mask, dict(method='UNIFORM_K', video=video, actual_calls=len(answers),
        covered_pairs=len(selected_pairs), eligible_pairs=len(pairs), all_pairs=len(queries)*len(tids),
        coverage_fraction=len(selected_pairs)/(len(queries)*len(tids)),
        k_per_eligible_pair='0 or1 under registered rho<=1; uniform sampling without replacement',
        unasked_state='UNKNOWN, no prior and no training; omitted from predictions',
        motion_presence_gate=frame_presence is not None,
        question_keys=[a['request_id'] for a in answers])
