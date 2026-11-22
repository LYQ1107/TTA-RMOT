"""Query-fair visual search with an observed-answer search-deficit proxy.

Keep the existing response-blind within-query COVER order. After every query
has received one examination, redistribute only the remaining query budget.
The product of observed No probabilities is a heuristic search deficit, not
a calibrated existence posterior: the teacher answers may be correlated.
"""
import numpy as np


class CoverageSearch:
    def __init__(self,orders,protocol,*,feedback=True):
        assert protocol in ('PRESUPPOSED','CHECKLIST')
        self.orders=[list(map(int,order)) for order in orders]
        self.count=np.zeros(len(orders),dtype=int)
        self.log_deficit=np.zeros(len(orders))
        self.pending=set()
        self.feedback=feedback and protocol=='PRESUPPOSED'

    def select_batch(self,size):
        assert not self.pending,'Consume the preceding requested answers first'
        result=[]
        for _ in range(size):
            eligible=self.count<np.asarray([len(order) for order in self.orders])
            assert eligible.any()
            unseen=eligible&(self.count==0)
            allowed=unseen if unseen.any() else eligible
            # Diminishing allocation by already spent query budget prevents
            # a difficult query from monopolizing the finite examination set.
            score=self.log_deficit-np.log1p(self.count)
            score[~allowed]=-np.inf;q=int(np.argmax(score));t=self.orders[q][self.count[q]]
            result.append(dict(local_query=q,local_track=t,search_log_priority=float(score[q]),
                observed_log_deficit=float(self.log_deficit[q]),previous_query_calls=int(self.count[q])))
            self.count[q]+=1;self.pending.add((q,t))
        return result

    def consume(self,query,track,probability):
        key=int(query),int(track);assert key in self.pending
        self.pending.remove(key)
        if self.feedback:self.log_deficit[key[0]]+=float(np.log1p(-np.clip(probability,.02,.98)))
