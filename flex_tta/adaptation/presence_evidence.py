"""Unlabelled interaction audit, NOT a calibrated presence classifier or loss.

The four scores come from the same frozen FlexHook. Their interaction removes
additive visual/query offsets, but can still be semantically wrong. Therefore
even stable positive interaction remains UNKNOWN and cannot authorize an update.
"""
from collections import deque

import torch


def interaction_contrast(real_query, real_reference, null_query, null_reference):
    if not (real_query.shape == real_reference.shape == null_query.shape == null_reference.shape):
        raise ValueError("all four branches must describe the same candidate set")
    return real_query-real_reference-null_query+null_reference


class CausalInteractionAudit:
    """Past-only diagnostic history; never turns agreement into a hard label."""

    def __init__(self, min_history=3, max_gap=2):
        self.min_history, self.max_gap = min_history, max_gap
        self.reset()

    def reset(self):
        self.rows = {}
        self.last_frame = -1

    def step(self, frame, ids, real_query, real_reference, null_query, null_reference):
        if frame <= self.last_frame:
            raise ValueError("interaction history must advance causally")
        interaction = interaction_contrast(real_query,real_reference,null_query,null_reference).detach().cpu().tolist()
        if len(ids) != len(interaction):
            raise ValueError("candidate IDs and interaction scores differ")
        self.rows = {i:hs for i,hs in self.rows.items() if frame-hs[-1][0] <= self.max_gap}
        support, latest = [], max((hs[-1][0] for hs in self.rows.values()),default=None)
        for i,value in zip(ids,interaction):
            past = self.rows.get(i,())
            if len(past) >= self.min_history and value > 0 and all(v > 0 for _,v in past):
                support.append(i)
        result = dict(frame=frame, ids=list(ids), interaction=interaction,
                      past_history_latest_frame=latest, provisional_interaction_support_ids=support,
                      source_selected_ids=[i for i,v in zip(ids,real_query.detach().cpu().tolist()) if v > 0],
                      presence_status="UNKNOWN",
                      reason="UNCALIBRATED_INTERACTION" if ids else "NO_OBSERVED_MOT_CANDIDATES",
                      support_is_semantic_positive=False, prediction_override=None,
                      optimizer_update_allowed=False)
        for i,value in zip(ids,interaction):
            self.rows.setdefault(i,deque(maxlen=self.min_history)).append((frame,float(value)))
        self.last_frame=frame
        return result


@torch.no_grad()
def frozen_interaction_scores(cache, candidates, sentence, current_frame, reference_sentence="person"):
    """Four original decode calls; V0 is zero FEATURE reference, not an empty image.

    Keep all original positions, motion, boxes and IDs. Only causal requested
    encoder features are read. No adapters, new teacher or source training.
    """
    features = {}
    for _,history in candidates:
        for observation in history:
            if observation.frame_id not in features:
                features[observation.frame_id]=cache._visual(observation.frame_id,current_frame)
    histories=[features[o.frame_id] for _,hs in candidates for o in hs]
    outputs=[torch.cat([h[k] for h in histories],dim=0) for k in range(len(histories[0]))]
    null_outputs=[torch.zeros_like(x) for x in outputs]
    positions,boxes,ids=cache.episode.geometry_inputs(candidates,cache.device)

    def decode(visual,query):
        text,mask=cache._text(query)
        text=text.expand(len(ids),-1,-1,-1).contiguous()
        mask=mask.expand(len(ids),-1,-1).contiguous()
        logits,_=cache.model.decode(visual,text,mask,positions,boxes)
        if logits.ndim==4:
            logits=logits[:,-1]
        if logits.shape[1:] != (1,2):
            raise RuntimeError("unexpected original FlexHook decode output")
        return (logits[:,0,1]-logits[:,0,0]).detach()

    return ids,dict(real_query=decode(outputs,sentence),real_reference=decode(outputs,reference_sentence),
                    null_query=decode(null_outputs,sentence),null_reference=decode(null_outputs,reference_sentence))
