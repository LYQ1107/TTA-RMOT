"""Shared fresh-state inputs for a teacher-only current-video prefix pilot."""
from collections import defaultdict
import json
import numpy as np
from flex_uda.a100.mechanism import metadata, PAIRS
from flex_uda.active.relational_features import geometry
from scripts.a100_anchor_pcd_collect import OUT as FEATURES
from scripts.a100_observation_design_collect import OUT as ANSWERS, TEACHER_SHA


def first_prefix(pair='C2', policy='DESIGN'):
    target = PAIRS[pair][1]; data = metadata(target); video = sorted(data['videos'])[0]
    bank = data['videos'][video]; queries = [q for q in data['episodes'] if q['video_id'] == video]
    folder = FEATURES/pair/'videos'/video
    assert (folder/'receipt.json').is_file(), 'Wait for the fresh source video receipt.'
    with np.load(folder/'anchor_pcd.npz') as values:
        pcd, tracks, present, sample_rows, sample_frames = [values[k].copy() for k in
            ('features','tracks','present','sample_rows','sample_frames')]
        assert list(values['query_ids']) == [q['expression_id'] for q in queries]
        assert values['seen'][:, present].all()
    obs = np.asarray(bank['observations'], dtype=int); coords, _, _ = geometry(bank, target)
    position = {tuple(row):i for i,row in enumerate(obs.tolist())}
    assert len(position) == len(obs)
    extra = np.zeros((len(tracks), 3, 6), dtype='float32')
    first, last = int(obs[:,0].min()), int(obs[:,0].max())
    for ti, track in enumerate(tracks):
        ids = np.flatnonzero(obs[:,1] == track); lo, hi = obs[ids,0].min(), obs[ids,0].max()
        for si in np.flatnonzero(present[ti]):
            row = int(sample_rows[ti,si]); frame = int(sample_frames[ti,si])
            assert tuple(obs[row]) == (frame,int(track))
            extra[ti,si] = np.r_[coords[row],(frame-first)/max(last-first,1),(frame-lo)/max(hi-lo,1)]
    raw = np.concatenate((pcd.astype('float32'), np.broadcast_to(extra, pcd.shape[:-1]+(6,))), -1)
    observed = raw[:,present].reshape(-1,raw.shape[-1]).astype(float)
    mean = observed.mean(0); std = np.maximum(observed.std(0), 1e-3)
    normalized = ((raw-mean)/std).astype('float32')
    normalized[:,~present] = 0
    document = json.loads((ANSWERS/'policy_answers'/pair/video/(policy+'_answers.json')).read_text())
    assert document['only_this_policy_requested_responses'] and document['GT_open_attempts'] == 0
    seen = set(); grouped = defaultdict(list); logical = document['logical_requested']
    for row in document['answers']:
        if row['request_id'] in seen:
            continue
        seen.add(row['request_id']); qi,ti = int(row['local_query']),int(row['local_track'])
        assert row['query_id'] == queries[qi]['expression_id'] and row['track_id'] == tracks[ti]
        assert row['frozen_weight_sha256'] == TEACHER_SHA
        frame = int(row['frame_id']); slot = np.flatnonzero(present[ti] & (sample_frames[ti] == frame))
        assert len(slot) == 1, (pair,video,frame,int(tracks[ti]))
        assert (frame,int(tracks[ti])) in position
        grouped[qi,ti,int(slot[0]),row['effective_view']].append(float(row['yes_probability']))
    keys = sorted(grouped); group_ids = {}; bags=[]; masks=[]; current=[]; target=[]; groups=[]
    for qi,ti,si,view in keys:
        key = qi,ti,view
        if key not in group_ids:
            group_ids[key] = len(group_ids)
        bags.append(normalized[qi,ti]); masks.append(present[ti]); current.append(normalized[qi,ti,si])
        target.append(float(np.mean(grouped[qi,ti,si,view]))); groups.append(group_ids[key])
    bags = np.asarray(bags,dtype='float32'); masks=np.asarray(masks,bool); current=np.asarray(current,dtype='float32')
    raw_target = np.asarray(target,float); group=np.asarray(groups,dtype='int64')
    flat = np.concatenate((bags.reshape(len(bags),-1),masks.astype('float32'),current),1)
    counts = np.bincount(group); receipt=json.loads((folder/'receipt.json').read_text())
    return dict(bags=bags,present=masks,current=current,flat=flat,raw_target=raw_target,
        target=np.clip(raw_target,.02,.98).astype('float32'),
        weight=(1.+3.*np.abs(2.*raw_target-1.)).astype('float32'),group=group,
        mean=mean,std=std,report=dict(pair=pair,video=video,policy=policy,logical_requests=logical,
            unique_request_ids=len(seen),observation_view_targets=len(keys),matched_query_bag_view_groups=len(counts),
            groups_with_multiple_anchor_observations=int((counts>=2).sum()),max_observations_per_group=int(counts.max()),
            bag_shape=list(bags.shape),flat_shape=list(flat.shape),feature_dimension=bags.shape[-1],
            normalized_from_current_video_unlabeled_anchors_only=True,
            previous_A100_mean_max_error=receipt['previous_A100_mean_max_error'],
            fresh_states_identical_for_all_models=True,language_scope_cache_not_consumed_in_this_pilot=True,
            CLIP_assigned_to_requested_anchor_as_common_prototype_approximation=True,
            no_manual_GT_or_quality_predictions=True,no_full_video_decoder_or_HOTA_output=True))
