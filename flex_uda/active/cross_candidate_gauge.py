"""Fix the unidentifiable COMMON utility shift, retaining identity contrasts.

Conditional evidence identifies u_A-u_B, not their common offset. Centering
the pair utilities removes that free coordinate from count fitting; the
independent count network learns the resulting common intercept. This is
an internal likelihood parameterization, not output probability calibration.
"""
import torch
from torch.nn import functional as F
from flex_uda.active.cross_candidate_learning import CrossCandidateHead


class GaugeFixedCrossCandidateHead(CrossCandidateHead):
    def forward(self, anchors_A, present_A, anchors_B, present_B, first, second):
        ua=self.utility(anchors_A,present_A,first)
        ub=self.utility(anchors_B,present_B,second)
        common_offset=(ua+ub)/2
        # The same amount is removed from BOTH objects. Identity difference
        # ua-ub is unchanged; candidate-specific offsets are never removed.
        ua,ub=ua-common_offset,ub-common_offset
        ca=self.context(self.count_encoder(anchors_A),present_A)
        cb=self.context(self.count_encoder(anchors_B),present_B)
        symmetric=torch.cat(((ca+cb)/2,torch.abs(ca-cb)),-1)
        a,b=self.count_encoder(first),self.count_encoder(second)
        common=self.base(symmetric).squeeze(-1)
        interaction=self.interaction(torch.cat((symmetric,torch.abs(a-b),a*b),-1)).squeeze(-1)
        va,vb=(ua.detach(),ub.detach()) if self.conditional_estimation else (ua,ub)
        count=F.log_softmax(torch.stack((torch.zeros_like(common),
            common+torch.logaddexp(va,vb),2*common+va+vb+interaction),-1),-1)
        relative=F.log_softmax(torch.stack((ua,ub),-1),-1)
        return torch.stack((count[...,1]+relative[...,0],count[...,1]+relative[...,1],
                            count[...,2],count[...,0]),-1)
