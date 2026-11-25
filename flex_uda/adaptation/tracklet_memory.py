"""Temporal consistency is restricted by semantic type and actual time gap."""
import torch
import torch.nn.functional as F
from flex_uda.losses.objectives import semantic_kind


class TrackletMemory:
    def __init__(self):self.rows={}

    def loss(self,video,query,frame,ids,margin):
        losses=[]
        if semantic_kind(query)=='STATIC_APPEARANCE':
            for i,score in zip(ids,margin):
                previous=self.rows.get((video,query,i))
                if previous and 0<frame-previous[0]<=2:
                    losses.append(F.smooth_l1_loss(score.float(),score.new_tensor(previous[1]).float()))
        for i,score in zip(ids,margin):self.rows[video,query,i]=(frame,float(score.detach()))
        if len(self.rows)>4096:self.rows=dict(list(self.rows.items())[-2048:])
        return (torch.stack(losses).mean() if losses else margin.sum()*0),len(losses)
