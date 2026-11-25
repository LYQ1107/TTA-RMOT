"""The seven preregistered, frozen-model stage4 ablations, not new methods.

Only stage4 imports this module; the running eleven-job v8 tier is unchanged.
"""
from dataclasses import replace
import numpy as np
from flex_uda.active.v8 import PresenceEM, BeliefStateV8, questions_v8
from flex_uda.active.v7 import OnlinePresenceEB
from flex_uda.active.v6 import ReliabilityV6
from flex_uda.active.acquisition import Question, VIEWS, information_gain, route

ABLATIONS=('MINUS_EM','MINUS_GLANCE','V7_ESTIMATOR','RANDOM_ACQ',
           'MINUS_DS','MINUS_ROUTE','MINUS_PRES')


class FixedPresenceEM(PresenceEM):
    def fit(self, rel):
        self.rho=.5;self.pi[:]=.5
        self.iterations=0;self.converged=True;self.last_delta=0.
        self.delta_trace=[];self.objective_trace=[]
        return self.receipt()

    def receipt(self, iterations=None):
        return dict(super().receipt(iterations),estimator_mode='FIXED_HALF_NO_EM',
                    convergence_not_applicable=True)


class V7PresenceEstimator(PresenceEM):
    """Replace EM numbers only; retain all other v8 coverage/decision rules."""
    def __init__(self):
        super().__init__();self.old=OnlinePresenceEB();self.pending=None

    def add_video(self, video, state):
        result=super().add_video(video,state);self.pending=state;return result

    def fit(self, rel):
        self.pending.pi_start=self.old.pi
        self.old.finish_video(self.pending,rel)
        self.rho=self.old.rho;self.pi[:]=self.old.pi;self.pending=None
        self.iterations=0;self.converged=True;self.last_delta=0.
        self.delta_trace=[];self.objective_trace=[]
        return self.receipt()

    def receipt(self, iterations=None):
        return dict(super().receipt(iterations),estimator_mode='V7_OUTCOME_FILTERED_NUMBERS',
                    convergence_not_applicable=True,old_estimator=self.old.receipt(),
                    v8_coverage_GLANCE_and_decoding_retained=True)


class FixedReliability(ReliabilityV6):
    def fit(self):
        self.values={v:(.8,.8) for v in self.workers}
        return self.values

    def receipt(self):
        return dict(super().receipt(),estimator_mode='FIXED_SENSITIVITY_SPECIFICITY_0.8')


class NoPresenceState(BeliefStateV8):
    def recompute_presence(self, reliability=None, queries=None):
        super().recompute_presence(reliability,queries)
        # Neutral bookkeeping only: no existence decisions or presence targets.
        # The original per-frame StageA inputs and identity fast fit remain.
        self.presence_probability[:]=.5;self.presence_probability_recomputed[:]=.5
        self.presence_stopped[:]=False;self.stop_probability[:]=np.nan
        self.absent[:]=False;self.confirmed_present[:]=False
        self.low_tentative[:]=False;self.low[:]=False

    def output_absence(self):
        return np.zeros(len(self.a),dtype=bool)


def setup(ablation):
    assert ablation is None or ablation in ABLATIONS
    estimator=(FixedPresenceEM() if ablation=='MINUS_EM' else
               V7PresenceEstimator() if ablation=='V7_ESTIMATOR' else PresenceEM())
    reliability=FixedReliability() if ablation=='MINUS_DS' else ReliabilityV6()
    state_type=NoPresenceState if ablation=='MINUS_PRES' else BeliefStateV8
    return reliability,estimator,state_type


def variant_questions(state,texts,lengths,reliability,round_index,*,ablation,rng):
    if ablation=='MINUS_PRES':
        result=[]
        for q,text in enumerate(texts):
            for t,n in enumerate(lengths):
                if n<3:continue
                gains={v:information_gain(state.posterior[q,t],*reliability[v]) for v in VIEWS}
                view=route(text) if round_index<2 else max(VIEWS,key=lambda v:gains[v])
                result.append(Question(q,t,view,'IDENTITY',max(gains.values())))
        return sorted(result,key=lambda x:(-x.information_gain,-state.z[x.query,x.track],x.query,x.track,x.view))
    result=questions_v8(state,texts,lengths,reliability,round_index)
    if ablation=='MINUS_GLANCE':
        result=[x for x in result if x.kind!='GLANCE']
    if ablation=='MINUS_ROUTE' and round_index<2:
        # Change the routed view only, retaining the acquisition value/order.
        result=[replace(x,view='CROP') for x in result]
    if ablation=='RANDOM_ACQ':
        # Existing seed0 video RNG; permute within kind, keep phase priorities.
        groups={}
        for x in result:groups.setdefault(x.kind,[]).append(x)
        shuffled={k:iter([v[i] for i in rng.permutation(len(v))]) for k,v in groups.items()}
        result=[next(shuffled[x.kind]) for x in result]
    return result
