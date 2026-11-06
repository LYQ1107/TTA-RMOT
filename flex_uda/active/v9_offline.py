"""CPU-only random-effects likelihood. No GT, model, teacher or live dispatch.

B1: exact smooth likelihood for fitting; node-wise track clipping for decisions.
This is not a selected/deployed v9 method until the offline gates pass.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp
from flex_uda.active.v8 import PresenceEM, BeliefStateV8

_nodes,_weights=np.polynomial.hermite.hermgauss(20)
GH_X=np.sqrt(2.)*_nodes
GH_LOGW=np.log(_weights)-.5*np.log(np.pi)


def answer_terms(answers,rel):
    lp=sum(np.log(rel[v][0] if y else 1-rel[v][0]) for v,y in answers)
    ln=sum(np.log(1-rel[v][1] if y else rel[v][1]) for v,y in answers)
    return lp,ln


def decision_llr(tracks,mu,tau,rel,*,capped=True):
    if not tracks:return 0.,0
    total=np.zeros(20);clipped_tracks=0
    for k,answers in tracks:
        lp,ln=answer_terms(answers,rel)
        eta=logit(mu[k])+tau*GH_X
        logp=-np.logaddexp(0.,-eta);logn=-np.logaddexp(0.,eta)
        raw=np.logaddexp(logp+lp,logn+ln)-ln
        clipped_tracks+=int(np.any(np.abs(raw)>2.))
        total+=np.clip(raw,-2.,2.) if capped else raw
    return float(logsumexp(GH_LOGW+total)),clipped_tracks


class RandomEffectsPresence(PresenceEM):
    def __init__(self,*,rho_prior=(2.,2.)):
        super().__init__();self.rho_prior=tuple(rho_prior);self.tau=0.
        a,b=self.rho_prior
        self.rho=float(np.clip((a-1)/(a+b-2),.05,.95))
        self.fit_receipt={};self.rho_laplace_std=None;self.laplace={}

    def arrays(self,rel):
        qs=[];ks=[];lp=[];ln=[]
        for q,rec in enumerate(self.records):
            for k,answers in rec['tracks']:
                x,y=answer_terms(answers,rel)
                qs.append(q);ks.append(k);lp.append(x);ln.append(y)
        return tuple(np.asarray(x,dtype=int if i<2 else float) for i,x in enumerate((qs,ks,lp,ln)))

    def objective(self,theta,data):
        """Negative exact marginal log posterior and analytic gradient."""
        rho=theta[0];mu=np.asarray(theta[1:4]);tau=theta[4]
        qs,ks,lp,ln=data;n=len(self.records)
        eta=logit(mu)[ks,None]+tau*GH_X[None,:]
        logp=-np.logaddexp(0.,-eta);logn=-np.logaddexp(0.,eta)
        mix=np.logaddexp(logp+lp[:,None],logn+ln[:,None])
        sums=np.stack([np.bincount(qs,weights=mix[:,j],minlength=n) for j in range(20)],axis=1)
        log1=logsumexp(sums+GH_LOGW,axis=1)
        log0=np.bincount(qs,weights=ln,minlength=n)
        observed=np.logaddexp(np.log(rho)+log1,np.log1p(-rho)+log0)
        gamma=np.exp(np.log(rho)+log1-observed)
        a,b=self.rho_prior
        prior=(a-1)*np.log(rho)+(b-1)*np.log1p(-rho)
        prior+=np.log(mu).sum()+np.log1p(-mu).sum()-tau*tau/8.
        grad=np.zeros(5)
        grad[0]=(gamma/rho-(1-gamma)/(1-rho)).sum()+(a-1)/rho-(b-1)/(1-rho)
        node_post=np.exp(sums+GH_LOGW-log1[:,None])
        delta=np.exp(logp+lp[:,None]-mix)-expit(eta)
        for k in range(3):
            chosen=ks==k
            deriv=np.stack([np.bincount(qs[chosen],weights=delta[chosen,j],minlength=n) for j in range(20)],axis=1)
            grad[1+k]=(gamma[:,None]*node_post*deriv).sum()/(mu[k]*(1-mu[k]))+1/mu[k]-1/(1-mu[k])
        deriv_tau=np.stack([np.bincount(qs,weights=delta[:,j]*GH_X[j],minlength=n) for j in range(20)],axis=1)
        grad[4]=(gamma[:,None]*node_post*deriv_tau).sum()-tau/4.
        return -float(observed.sum()+prior),-grad

    def fit(self,rel):
        if not self.records:return self.receipt()
        data=self.arrays(rel)
        start=np.r_[self.rho,self.pi,self.tau if self.tau>0 else 1.]
        result=minimize(self.objective,start,args=(data,),jac=True,method='L-BFGS-B',
            bounds=[(.05,.95)]*4+[(0.,None)],
            options=dict(maxiter=5000,ftol=1e-12,gtol=1e-7))
        self.rho=float(result.x[0]);self.pi=result.x[1:4].copy();self.tau=float(result.x[4])
        self.iterations=int(result.nit);self.converged=bool(result.success)
        self.last_delta=float(np.max(np.abs(result.x-start)))
        self.fit_receipt=dict(success=bool(result.success),status=int(result.status),message=str(result.message),
            iterations=int(result.nit),function_evaluations=int(result.nfev),
            negative_log_posterior=float(result.fun),gradient=result.jac.tolist(),
            initialization=start.tolist(),smooth_unclipped_objective=True)
        hessian=np.zeros((5,5))
        for j in range(5):
            h=1e-4*max(1.,abs(result.x[j]));left=result.x.copy();right=result.x.copy()
            left[j]-=h;right[j]+=h
            hessian[:,j]=(self.objective(right,data)[1]-self.objective(left,data)[1])/(2*h)
        hessian=(hessian+hessian.T)/2
        eig=np.linalg.eigvalsh(hessian)
        self.rho_laplace_std=None
        if eig.min()>0:self.rho_laplace_std=float(np.sqrt(np.linalg.inv(hessian)[0,0]))
        self.laplace=dict(coordinates='rho,mu0,mu1,mu2,tau',hessian=hessian.tolist(),
            eigenvalues=eig.tolist(),positive_definite=bool(eig.min()>0),
            status='LOCAL_GAUSSIAN' if eig.min()>0 else 'UNIDENTIFIED_NON_POSITIVE_HESSIAN',
            at_boundary=bool(any(np.isclose(x,.05) or np.isclose(x,.95) for x in result.x[:4]) or np.isclose(self.tau,0)),
            boundary_caveat='local unconstrained curvature; truncation can make the Gaussian approximation inaccurate',
            finite_difference='central analytic-gradient differences, h=1e-4*max(1,abs(parameter))')
        return self.receipt()

    def receipt(self,iterations=None):
        base=super().receipt(iterations)
        marginal=np.exp(GH_LOGW)@expit(logit(self.pi)[None,:]+self.tau*GH_X[:,None])
        return dict(base,estimator='RANDOM_EFFECTS_EXACT_MARGINAL_MAP',rho_Beta=list(self.rho_prior),
            mu_strata=self.pi.tolist(),tau=self.tau,marginal_pi_strata=marginal.tolist(),
            rho_laplace_std=self.rho_laplace_std,laplace=self.laplace,
            optimizer=self.fit_receipt,GH_nodes=20,fit_track_logLR_clipping=False,
            objective_trace=[],delta_trace=[],objective_trace_monotone=None,
            convergence_not_EM=True)


class RandomEffectsBelief(BeliefStateV8):
    def __init__(self,*args,tau=0.,uncapped_decision=False,**kwargs):
        self.tau_start=float(tau);self.uncapped_decision=bool(uncapped_decision)
        super().__init__(*args,**kwargs)

    def recompute_presence(self,reliability=None,queries=None):
        if reliability is not None:self.presence_reliability=dict(reliability)
        if not hasattr(self,'presence_probability'):
            self.presence_probability=np.full(len(self.a),.5)
            self.presence_probability_recomputed=np.full(len(self.a),.5)
            self.presence_logLR_unclipped_sum=np.zeros(len(self.a))
            self.presence_logLR_clipped_sum=np.zeros(len(self.a))
            self.presence_logLR_clip_count=np.zeros(len(self.a),int)
        for q in range(len(self.a)) if queries is None else queries:
            needed={int(k) for k in self.strata[q] if k>=0};covered=set();tracks=[]
            # Same increasing track-index order as filtering the global sorted
            # answer dictionary, without scanning other sentences per answer.
            for t in np.flatnonzero(self.asked_mask[q]):
                answers=self.answer_history[q,int(t)]
                k=int(self.strata[q,t])
                if k<0 or not answers:continue
                tracks.append((k,answers));covered.add(k);self.presence_likelihood_evaluations[k]+=1
            raw,_=decision_llr(tracks,self.pi_strata_start,self.tau_start,self.presence_reliability,capped=False)
            clipped,count=decision_llr(tracks,self.pi_strata_start,self.tau_start,self.presence_reliability,capped=True)
            total=logit(np.clip(self.rho_start,.05,.95))+self.weak_presence_logits[q]
            value=float(expit(total+(raw if self.uncapped_decision else clipped)))
            self.needed_strata[q]=[k in needed for k in range(3)]
            self.covered_strata[q]=[k in covered for k in range(3)]
            self.presence_probability_recomputed[q]=value
            self.presence_logLR_unclipped_sum[q]=raw;self.presence_logLR_clipped_sum[q]=clipped
            self.presence_logLR_clip_count[q]=count
            if not self.presence_stopped[q] and value<.1 and covered and covered>=needed:
                self.presence_stopped[q]=True;self.stop_probability[q]=value
            self.presence_probability[q]=self.stop_probability[q] if self.presence_stopped[q] else value
            self.low_tentative[q]=bool(not self.presence_stopped[q] and value<.1 and not covered>=needed)
        self.absent=self.presence_stopped.copy()
        self.confirmed_present=(self.presence_probability>.9)&~self.absent
        self.low=self.presence_probability<.1
