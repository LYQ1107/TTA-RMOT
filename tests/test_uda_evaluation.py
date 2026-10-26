"""Necessary execution/decision parity, not a substitute for real GPU parity."""
import pytest
import torch
from flex_uda.evaluation import query_margins
from scripts.diag_pf_countmatched import topk_ids


class IndependentQueries:
    sample_expression_num=1
    def decode(self,outputs,text,mask,positions,boxes):
        n=text.shape[1]
        if n!=self.sample_expression_num:raise ValueError('query shape mismatch')
        score=text.mean(dim=(-1,-2))
        logits=torch.stack([-score,score],dim=-1).unsqueeze(1).expand(-1,4,-1,-1)
        return logits,torch.tensor(0.)


def test_batched_distinct_queries_match_isolated_and_restore_state():
    model=IndependentQueries();prepared=dict(track_ids=[1,2],outputs=None,positions=None,boxes=None)
    texts=[(torch.full((1,1,25,3),v),torch.ones(1,1,25)) for v in (-1.,0.,2.)]
    multi=query_margins(model,prepared,texts,batched=True)
    single=query_margins(model,prepared,texts,batched=False)
    assert torch.equal(multi,single) and multi.tolist()==[[-2.,0.,4.],[-2.,0.,4.]]
    assert model.sample_expression_num==1


def test_countmatched_can_select_zero_or_multiple_without_gt():
    ids=[2,8,3];scores={2:-3.,8:-1.,3:-1.}
    assert topk_ids(ids,scores,0)==set()
    assert topk_ids(ids,scores,2)=={3,8}
    assert topk_ids(ids,scores,1)=={3}
