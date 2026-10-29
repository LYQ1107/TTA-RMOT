"""Direct soft evidence and finite positive-only local temporal weak labels."""
from collections import Counter,defaultdict
import math
import numpy as np
from .mllm_pseudo_dataset import digest


def box_iou(a,b):
    inter=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    aa=(a[2]-a[0])*(a[3]-a[1]);bb=(b[2]-b[0])*(b[3]-b[1])
    return inter/(aa+bb-inter) if aa+bb-inter>0 else 0.


def direct_evidence(sample,response,revision,manifest_sha):
    valid=response is not None and response.get('top1_is_yes_no',False)
    valid=valid and all(math.isfinite(response.get(k,float('nan'))) for k in ('yes_logit','no_logit','yes_probability'))
    p=float(response['yes_probability']) if valid else .5
    state='POS' if valid and p>=.95 else 'NEG' if valid and p<=.05 else 'UNKNOWN'
    # Low-confidence but valid raw soft labels remain trainable; UNKNOWN from
    # missing/nonbinary response is not silently mapped to a negative.
    record=dict(sample,teacher_prob=p,valid=bool(valid),state=state,supervision_source='DIRECT_TEACHER',
        teacher_yes_logit=float(response['yes_logit']) if valid else None,
        teacher_no_logit=float(response['no_logit']) if valid else None,
        teacher_revision=revision,manifest_sha256=manifest_sha,applicable_frames=[sample['frame_id'],sample['frame_id']],
        observed_teacher_frames=[sample['frame_id']],invalid_reason=None if valid else 'INVALID_OR_NONBINARY_TEACHER_RESPONSE')
    return record


def expand_local(direct,observations,appearance=None,*,full=True,minimum_iou=.5,maximum_area_ratio=2.,appearance_cosine=.95):
    """One middle-frame weak label per positive bracketing pair, never a track label.

    observations are scoped to ONE video. appearance[(frame,ID)] is an RGB
    histogram from that same input crop, not an external learned teacher.
    """
    groups=defaultdict(list);expanded=[];rejections=[];counts=Counter()
    for row in direct:groups[row['video_id'],row['query_id'],row['track_id'],row['anchor_frame']].append(row)
    existing={(r['video_id'],r['query_id'],r['frame_id'],r['track_id']) for r in direct}
    for (video,query,track,anchor),rows in sorted(groups.items()):
        rows=sorted(rows,key=lambda r:r['frame_id']);reason=None
        if len(rows)!=2 or [r['frame_id'] for r in rows]!=[anchor-1,anchor+1]:reason='MISSING_BRACKETING_DIRECT_EVIDENCE'
        elif not all(r['valid'] and r['state']=='POS' for r in rows):reason='NO_TWO_VALID_FULL_EXPRESSION_POS_ENDPOINTS'
        times=(anchor-1,anchor,anchor+1)
        if reason is None and not all(track in observations.get(t,{}) for t in times):reason='MISSING_FRAME_OR_PREDICTED_ID_BREAK'
        if reason is None:
            boxes=[observations[t][track]['bbox_xyxy'] for t in times]
            areas=[(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
            if min(areas)<=0 or max(areas)/min(areas)>maximum_area_ratio:reason='ABRUPT_SCALE_OR_INVALID_BOX'
            elif min(box_iou(a,b) for a,b in zip(boxes,boxes[1:]))<minimum_iou:reason='GEOMETRIC_IDENTITY_DISCONTINUITY'
        if reason is None and full:
            if any(box_iou(observations[t][track]['bbox_xyxy'],other['bbox_xyxy'])>=.7 for t in times for i,other in observations[t].items() if i!=track):
                reason='AMBIGUOUS_OVERLAP_OR_OCCLUSION'
            elif appearance is None or any((t,track) not in appearance for t in times):reason='MISSING_INPUT_APPEARANCE_EVIDENCE'
            else:
                vectors=[np.asarray(appearance[t,track],dtype=float) for t in times]
                similarity=[float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a)*np.linalg.norm(b)>0 else 0. for a,b in zip(vectors,vectors[1:])]
                if min(similarity)<appearance_cosine:reason='APPEARANCE_CHANGE_OR_POSSIBLE_ID_DRIFT'
        key=(video,query,anchor,track)
        if reason is None and key in existing:reason='ALREADY_DIRECTLY_SUPERVISED_NOT_NEW_COVERAGE'
        if reason:
            counts[reason]+=1;rejections.append(dict(video_id=video,query_id=query,frame_id=anchor,track_id=track,state='UNKNOWN',reason=reason));continue
        left,right=rows;row=dict(left,frame_id=anchor,bbox_xyxy=list(observations[anchor][track]['bbox_xyxy']),
            raw_mot_line=observations[anchor][track]['raw_mot_line'],teacher_prob=min(left['teacher_prob'],right['teacher_prob']),
            valid=True,state='POS',supervision_source='EXPANDED_WEAK',applicable_frames=[anchor-1,anchor+1],
            observed_teacher_frames=[anchor-1,anchor+1],parent_sample_ids=[left['sample_id'],right['sample_id']],
            teacher_yes_logit=None,teacher_no_logit=None,teacher_revision=left['teacher_revision'],
            expansion_warning='offline one-frame interpolation; predicted ID and action continuity are not ground truth')
        row['sample_id']=digest(('EXPANDED_WEAK',*key))[:24];expanded.append(row);existing.add(key)
    return expanded,rejections,dict(direct_count=len(direct),expanded_count=len(expanded),expanded_POS=len(expanded),expanded_NEG=0,
        rejected_UNKNOWN=len(rejections),rejection_reasons=dict(counts),extra_teacher_calls=0,maximum_interval_frames=2,
        expanded_are_weak_not_direct=True,unasked_are_NOT_NEG=True)
