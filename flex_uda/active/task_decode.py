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
