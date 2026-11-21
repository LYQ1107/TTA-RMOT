"""Explicit task contracts, not target-label estimates of video existence."""
import numpy as np
from flex_uda.active.color_cohort import part_mask


def nonempty_static(score,mask,observations,queries,protocol):
    """Constrained MAP fallback for static queries known to have a referent.

    Only video-level emptiness is changed. No claim of per-frame presence is
    made. CHECKLIST can stay empty. If every Bernoulli track probability is
    below .5, the highest-probability singleton is the nonempty MAP set under
    that working model. These scores are not claimed calibrated probabilities.
    """
    mask=np.array(mask,dtype=bool,copy=True);changes=[]
    if protocol!='PRESUPPOSED':return mask,changes
    tracks=sorted(set(map(int,observations[:,1])))
    by_track={t:np.flatnonzero(observations[:,1]==t) for t in tracks}
    for qi,q in enumerate(queries):
        if mask[qi].any() or not part_mask(q['query_text']).any() or not tracks:continue
        values=[float(np.mean(score[qi,by_track[t]])) for t in tracks]
        ti=int(np.argmax(values));track=tracks[ti];mask[qi,by_track[track]]=True
        changes.append(dict(query=q['expression_id'],track=track,score=values[ti],selected_rows=len(by_track[track])))
    return mask,changes


def nonempty_from_reference(mask,reference,observations,tracks,prior,protocol):
    """Shared video-presence fallback, with a common strong teacher reference.

    This leaves provided boxes/IDs intact and does not promise a target in
    every frame. All compared heads get the same reference, even hard heads.
    The caller declares the task contract; it is not estimated from GT.
    """
    result=np.array(mask,dtype=bool,copy=True);changed=[]
    if protocol!='PRESUPPOSED':return result,changed
    observations=np.asarray(observations);tracks=np.asarray(tracks)
    first=np.asarray([np.flatnonzero(observations[:,1]==track)[0] for track in tracks])
    for q in np.flatnonzero(~result.any(1)):
        selected=int(np.lexsort((tracks,-np.asarray(prior[q]),-np.asarray(reference[q,first])))[0])
        track=int(tracks[selected]);result[q,observations[:,1]==track]=True
        changed.append(dict(query_index=int(q),track=track))
    return result,changed


def retain_nonempty_distillation(graph_mask,baseline_mask,protocol):
    """Reject a graph-induced empty set only under declared video presence.

    A sparse set of No answers does not prove that all matching people are
    absent. Retaining the baseline's entire set differs from the old singleton
    fallback and from disabling all propagation whenever no Yes was observed.
    """
    result=np.array(graph_mask,dtype=bool,copy=True)
    rejected=np.zeros(len(result),dtype=bool)
    if protocol=='PRESUPPOSED':
        rejected=~result.any(1)&np.asarray(baseline_mask).any(1)
        result[rejected]=baseline_mask[rejected]
    return result,rejected
