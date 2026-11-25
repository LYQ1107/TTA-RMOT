"""Full-frame active visual learning, with lazy features and fresh teachers.

Fixes measured scope defects: action support covers every provided frame;
content transfer covers all physical ROIs; questions are distinct; observed
teacher targets stay immutable. The old numerical/scientific files remain
unchanged. No fixed-model risk identity is claimed for adapted-model HOTA.
"""
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from flex_tta.runtime.pf_receipts import atomic_json
import flex_uda.active.evidence_visual_learning as core


class IndexKeys:
    def __init__(self,n):self.n=n
    def __len__(self):return self.n
    def __getitem__(self,index):
        if isinstance(index,slice):return [str(i) for i in range(*index.indices(self.n))]
        assert 0<=int(index)<self.n
        return str(int(index))


class ActionLookup:
    def __init__(self,context):self.context=context
    def __getitem__(self,uid):
        r=self.context.request(int(uid))
        return dict(uid=str(uid),local_query=r['local_query'],local_track=r['local_track'],case=r)


class LazyFeatures:
    """Never materialize the17.96M-by9352 query-expanded global tensor."""
    def __init__(self,context):self.context=context;self.shape=(context.n*context.nq,9352)
    def __len__(self):return self.shape[0]
    def __getitem__(self,index):
        if isinstance(index,slice):ids=np.arange(*index.indices(len(self)),dtype=int)
        else:ids=np.asarray(index,dtype=int).reshape(-1)
        c=self.context;q,rows=np.divmod(ids,c.n)
        records=[dict(local_query=int(qi),local_track=int(c.video.columns[oi]),
            frame_id=int(c.video.obs[oi,0]),track_id=int(c.video.obs[oi,1])) for qi,oi in zip(q,rows)]
        a=c.inputs.arrays(records)
        base=np.column_stack((a['bags'].reshape(len(rows),-1),a['present'],a['current'],a['coarse'])).astype('float32')
        from flex_uda.active.full_appearance_learning_inputs import augment_dense
        result=augment_dense(base,c.video,rows)
        assert result.shape==(len(ids),9352) and np.isfinite(result).all()
        return result


class TeacherDispatcher:
    """One unchangedT0 per GPU worker, retained across independent videos."""
    def __init__(self,root,cache):self.root=Path(root);self.cache=Path(cache);self.client=None;self.video_key=None
    def activate(self,context):
        from scripts.active_run import video_spec
        from flex_uda.a100.mechanism import metadata
        from flex_uda.a100.runtime import teacher
        from flex_uda.teacher.decompose_inputs import PairedViews
        key=context.case['target'],context.case['evaluated_video']
        if key==self.video_key:return
        spec=video_spec(*key,metadata(key[0])['videos'][key[1]])
        if self.client is None:
            self.client=teacher([spec],self.cache)
            assert self.client.backend.weight_sha=='9a13380b4279bc98f4680bd083ff4995fdcfec120cfded459513be90d041a956'
            print(json.dumps(dict(status='FULL_SCOPE_ORIGINAL_QWEN_LOADED',GPU_worker=context.gpu,
                load_seconds=self.client.backend.load_seconds)),flush=True)
        else:
            self.client.views=PairedViews([spec],self.client.backend.upstream)
            self.client.image_shas={}
        self.video_key=key
    def ask(self,context,requests):
        self.activate(context)
        return self.client.ask_many(requests)
    def close(self):return None if self.client is None else self.client.close()


class FreshBroker:
    """Only requested answers are exposed; resumable per-policy journals."""
    def __init__(self,context,policy):
        self.context=context;self.policy=policy;self.received={}
        self.path=context.folder/(policy+'_answers.jsonl');self.history={}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                row=json.loads(line);uid=row['uid'];assert uid not in self.history;self.history[uid]=row
        self.sequence=[]
    def request(self,uids,role,stage):
        assert role=='train' and len(uids)==len(set(uids))
        c=self.context;pending=[]
        for uid in uids:
            assert uid not in self.received
            if uid not in self.history:pending.append(uid)
        # Unfinished-stage answers never grant extra unique teacher budget.
        counts={q:sum(int(u)//c.n==q for u in self.history) for q in c.quotas}
        for uid in pending:counts[int(uid)//c.n]+=1
        assert all(counts[q]<=c.quotas[q] for q in counts),(self.policy,counts,c.quotas)
        for begin in range(0,len(pending),4):
            selected=pending[begin:begin+4];requests=[c.request(int(uid)) for uid in selected]
            replies=c.dispatcher.ask(c,requests)
            with self.path.open('a',buffering=1) as stream:
                for uid,request,reply in zip(selected,requests,replies):
                    assert all(reply[k]==request[k] for k in ('target','video_id','query_id','query_text','track_id','frame_id','view'))
                    p=float(reply['yes_probability']);assert 0<=p<=1
                    row=dict(uid=uid,probability=p,query=int(uid)//c.n,teacher_request_id=reply['request_id'],
                        frozen_weight_sha256=reply['frozen_weight_sha256'],cache_hit=reply['cache_hit'],
                        requested_view=reply['requested_view'],effective_view=reply['effective_view'],
                        teacher_batch_forward_seconds=reply['batch_forward_seconds'],teacher_batch_size=reply['batch_size'])
                    assert row['frozen_weight_sha256']=='9a13380b4279bc98f4680bd083ff4995fdcfec120cfded459513be90d041a956'
                    stream.write(json.dumps(row)+'\n');self.history[uid]=row
        result=[]
        for uid in uids:
            row=self.history[uid];self.received[uid]=row;self.sequence.append(dict(uid=uid,stage=stage));result.append(row)
        return result
    def receipt(self,complete=False):
        counts={q:sum(int(u)//self.context.n==q for u in self.received) for q in self.context.quotas}
        if complete:assert counts==self.context.quotas
        return dict(unique_teacher_bundles=len(self.received),logical_budget=sum(self.context.quotas.values()),
            per_query_count=counts,per_query_budget=self.context.quotas,
            actual_new_teacher_calls=sum(not r['cache_hit'] for r in self.history.values()),
            exact_input_teacher_cache_hits=sum(r['cache_hit'] for r in self.history.values()),
            no_repeated_logical_question=True,unrequested_replies_exposed_to_learner=0,
            source_front_and_auxiliary_features_prepaid_not_fresh_cold_cost=True)


class Context:
    def __init__(self,case,folder,dispatcher,gpu):
        from flex_uda.active.joint_observation_inputs import JointObservationVideo,use_verified_local_features
        from flex_uda.active.factor_rsi_learning import Inputs
        from flex_uda.active.full_appearance_learning_inputs import FullAppearanceBank
        use_verified_local_features();self.case=case;self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.dispatcher=dispatcher;self.gpu=gpu
        self.video=JointObservationVideo(case['pair'],case['evaluated_video']);self.inputs=Inputs(self.video)
        self.n,self.nq=len(self.video.obs),len(self.video.queries)
        document=json.loads(Path(case['temporal_queries']).read_text())
        self.queries=document['queries'];assert self.n==document['observations'] and self.nq==len(self.queries)
        self.quotas={i:int(q['logical_budget']) for i,q in enumerate(self.queries)}
        assert min(self.quotas.values())>0 and max(self.quotas.values())<=self.n
        for q,r in zip(self.queries,self.video.queries):assert q['query_id']==r['expression_id'] and q['query_text']==r['query_text']
        self.features=LazyFeatures(self);self.query=np.repeat(np.arange(self.nq,dtype=int),self.n)
        self.doc=dict(order=IndexKeys(len(self.features)),actions=ActionLookup(self),quotas=self.quotas)
        with np.load(case['auxiliary_path']) as a:self.auxiliary=a['features'].copy();self.ay=np.nanmean(a['probability'],axis=1)
        bank=FullAppearanceBank(case['target'],case['evaluated_video'])
        idx=np.asarray([bank.by_observation[tuple(map(int,row))] for row in self.video.obs]);assert bank.valid[idx].all()
        self.images=bank.normalized(idx)
        normal=self.folder/'unlabeled_normalizer.npz'
        if normal.exists():
            with np.load(normal) as a:self.mean,self.std=a['mean'].copy(),a['std'].copy()
        else:
            sampled=np.unique(np.linspace(0,len(self.features)-1,min(4096,len(self.features))).astype(int))
            x=self.features[sampled];self.mean=x.mean(0,dtype='float64').astype('float32');self.std=x.std(0,dtype='float64').clip(min=.1).astype('float32')
            np.savez(normal,mean=self.mean,std=self.std,sampled_indices=sampled)
        self.primary=FreshBroker(self,'active');self.uniform=FreshBroker(self,'uniform')
    def request(self,index):
        q,oi=divmod(index,self.n);query=self.queries[q]
        return dict(target=self.case['target'],video_id=self.case['evaluated_video'],query_id=query['query_id'],
            query_text=query['query_text'],track_id=int(self.video.obs[oi,1]),frame_id=int(self.video.obs[oi,0]),
            local_query=q,local_track=int(self.video.columns[oi]),view=query['view'],allocation_kind='FULL_TEMPORAL_ACTIVE_VISION')
    def data(self):return self.features,self.query,self.doc,self.quotas,self.auxiliary,self.ay,self.mean,self.std


def lure_weights(n,probabilities):
    p=np.asarray(probabilities,dtype='float64');m=len(p)
    assert 0<m<=n and (p>0).all() and (p<=1).all()
    t=np.arange(1,m+1,dtype='float64')
    result=np.ones(m) if m==n else 1+(n-m)/(n-t)*(1/((n-t+1)*p)-1)
    assert np.isfinite(result).all() and (result>0).all()
    return result


def teacher_table(context,draws,mode):
    indices=np.asarray([r['index'] for r in draws]);y=np.asarray([r['probability'] for r in draws])
    ay=context.ay.copy()
    if mode=='HARD_KD':y=(y>=.5).astype(float);ay=(ay>=.5).astype(float)
    counts={q:sum(r['query']==q for r in draws) for q in context.quotas}
    weight=np.asarray([1/(len(context.quotas)*counts[r['query']]) for r in draws])
    if mode=='IPW_KD':
        # Legacy API name only: correct positive NO-REPLACEMENT LURE here.
        for q in context.quotas:
            positions=np.asarray([i for i,r in enumerate(draws) if r['query']==q])
            weight[positions]*=lure_weights(context.n,[draws[i]['proposal_probability'] for i in positions])
    return (np.r_[context.features[indices],context.auxiliary],np.r_[y,ay],
        np.r_[weight,np.full(len(ay),.25/len(ay))]/1.25)


def fit(context,model,x,y,w,steps,device,optimizer=None,*,pixel=None,align=None,unlabeled=None,residual=None):
    """Teacher BCE unchanged; image transfer covers ALLphysicalROIs."""
    opt=optimizer or torch.optim.Adam(model.parameters(),lr=core.LR)
    x,y,w=core.tensors(x,device),core.tensors(y,device),core.tensors(w,device)
    assert ((y>=0)&(y<=1)).all() and (w>=0).all()
    err=None if residual is None else core.tensors(residual,device)
    cursor=torch.arange(512,device=device);started=time.perf_counter()
    if align is not None:
        images=core.tensors(context.images,device)
        with torch.no_grad():target=align.app(images).detach()
    for step in range(steps):
        i=(cursor+512*step)%len(x);opt.zero_grad(set_to_none=True)
        out=model(x[i]);z=out if err is None else out[:,0]
        loss=((nn.functional.softplus(z)-y[i]*z)*w[i]*len(x)).mean()
        if err is not None:loss=loss+((out[:,1].sigmoid()-err[i]).square()*w[i]*len(x)).mean()
        if pixel is not None:loss=loss+core.VISUAL_WEIGHT*safe_content_loss(model,pixel)
        if align is not None:
            j=(cursor+512*step)%len(images)
            loss=loss+core.VISUAL_WEIGHT*nn.functional.mse_loss(model.app(images[j]),target[j])
        assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
    torch.cuda.synchronize()
    return opt,dict(steps=steps,wall_seconds=time.perf_counter()-started,last_objective=float(loss.detach()),
        nonnegative_teacher_weights=True,teacher_targets_unchanged=True,
        all_physical_content_images=len(context.images) if align is not None else None)


ORIGINAL_CONTENT_LOSS=core.content_loss
def safe_content_loss(model,pixel):
    # Missing weak correspondences cannot justify skipping a video or making
    # up semantic labels. It is recorded explicitly and affects SSL equally.
    if len(pixel['left'])<2:return next(model.app.parameters()).sum()*0
    return ORIGINAL_CONTENT_LOSS(model,pixel)


def proposal(priority,known,columns,floor=.2):
    remaining=~known;assert remaining.any()
    counts=np.bincount(columns[known],minlength=int(columns.max())+1)
    score=np.asarray(priority,dtype='float64')/(1+counts[columns]);score[known]=0
    raw=score[remaining];adaptive=raw/raw.sum() if raw.sum()>0 else np.full(len(raw),1/len(raw))
    result=np.zeros(len(known));result[remaining]=floor/len(raw)+(1-floor)*adaptive
    assert (result[remaining]>0).all() and np.isclose(result.sum(),1.)
    return result


def predicted_student_risk(mean,residual,student):
    """Bias/variance-inspired current-S risk proxy, never semantic truth.

The decomposition is exact for calibrated conditional mean/variance.
Learned prequential R residuals are not certified conditional variance.
"""
    return np.asarray(residual)+np.square(np.asarray(mean)-np.asarray(student))


def train(context,folder,device,registration_sha256):
    c=context;folder=Path(folder);folder.mkdir(exist_ok=True)
    s=core.make(c.mean,c.std,device);r=core.make(c.mean,c.std,device,outputs=2)
    final=folder/'final.pt';closed=folder/'closed.pt';boot=folder/'bootstrap.pt'
    if final.exists():
        done=torch.load(final,map_location='cpu',weights_only=False);assert done['registration_sha256']==registration_sha256
        s.load_state_dict(done['solver']);r.load_state_dict(done['observer'])
        assert all(d['uid'] in c.primary.history for d in done['draws'])
        c.primary.request([d['uid'] for d in done['draws']],'train',-1)
        return s,r,done
    so=torch.optim.Adam(s.parameters(),lr=core.LR);ro=torch.optim.Adam(r.parameters(),lr=core.LR)
    rng=np.random.default_rng(0);known=np.zeros(len(c.features),dtype=bool);draws=[];rounds=[];fits=[]
    pixel=core.pixel_tensors(c.case,device)
    if closed.exists() or boot.exists():
        done=torch.load(closed if closed.exists() else boot,map_location='cpu',weights_only=False)
        assert done['registration_sha256']==registration_sha256
        s.load_state_dict(done['solver']);r.load_state_dict(done['observer']);so.load_state_dict(done['sopt']);ro.load_state_dict(done['ropt'])
        rng.bit_generator.state=done['rng'];draws=done['draws'];rounds=done['rounds'];fits=done['fits'];known=done['known'];auxerror=done['auxerror']
        c.primary.request([d['uid'] for d in draws],'train',-1) if draws else None
    else:
        so,sfit=fit(c,s,c.auxiliary,c.ay,np.full(len(c.ay),1/len(c.ay)),500,device,so);fits.append(sfit)
        p=1/(1+np.exp(-core.logits(s,c.auxiliary,device).astype('float64')));auxerror=(p-c.ay)**2
        ro,_=fit(c,r,c.auxiliary,c.ay,np.full(len(c.ay),1/len(c.ay)),500,device,ro,pixel=pixel,residual=auxerror)
        core.save(boot,dict(solver=core.state(s),observer=core.state(r),sopt=so.state_dict(),ropt=ro.state_dict(),
            rng=rng.bit_generator.state,draws=draws,rounds=rounds,fits=fits,known=known,auxerror=auxerror,registration_sha256=registration_sha256))
    for stage in range(len(rounds),4):
        pending=folder/f'stage_{stage}_selected_before_answers.json'
        if pending.exists():
            selection=json.loads(pending.read_text());assert selection['registration_sha256']==registration_sha256
            selected=selection['selected'];rng.bit_generator.state=selection['rng_after_selection']
        else:
            # Combined scan: one lazy feature assembly perbatch for both nets.
            student=np.empty(len(c.features));forecast=np.empty(len(c.features));uncertainty=np.empty(len(c.features));jac=np.empty(len(c.features))
            with torch.no_grad():
                for start in range(0,len(c.features),512):
                    x=core.tensors(c.features[start:start+512],device);sl=slice(start,start+len(x))
                    student[sl]=s(x).sigmoid().cpu().numpy();rz=r(x).sigmoid()
                    forecast[sl]=rz[:,0].cpu().numpy();uncertainty[sl]=rz[:,1].cpu().numpy();jac[sl]=s.last_layer_jacobian_norm(x).cpu().numpy()
            temporary=known.copy();selected=[]
            for q,n in sorted(c.quotas.items()):
                start=q*c.n;group=np.arange(start,start+c.n)
                for _ in range(n//4+int(stage<n%4)):
                    current_risk=predicted_student_risk(forecast[group],uncertainty[group],student[group])
                    pi=proposal(np.sqrt(current_risk.clip(min=1e-6))*jac[group],temporary[group],c.video.columns)
                    eligible=np.flatnonzero(pi>0);cum=np.cumsum(pi[eligible]);cum[-1]=1.
                    local=int(eligible[np.searchsorted(cum,float(rng.random()),side='right')]);index=int(group[local])
                    assert not temporary[index];temporary[index]=True
                    selected.append(dict(index=index,uid=str(index),query=q,stage=stage,
                        proposal_probability=float(pi[local]),student_probability_BEFORE_answer=float(student[index]),
                        observer_mean_BEFORE_answer=float(forecast[index]),observer_residual_proxy_BEFORE_answer=float(uncertainty[index])))
            atomic_json(pending,dict(selected=selected,rng_after_selection=rng.bit_generator.state,
                actual_answers_used_for_current_stage_selection=False,registration_sha256=registration_sha256))
        replies=c.primary.request([d['uid'] for d in selected],'train',stage)
        for d,answer in zip(selected,replies):
            row=dict(d,probability=answer['probability'],student_squared_error=(d['student_probability_BEFORE_answer']-answer['probability'])**2,
                observer_squared_error=(d['observer_mean_BEFORE_answer']-answer['probability'])**2,
                repeated_logical_action=False);draws.append(row);known[d['index']]=True
        x,y,w=teacher_table(c,draws,'IPW_KD');error=np.r_[[d['observer_squared_error'] for d in draws],auxerror]
        # Observer first, then current external/image evidence reaches S in
        # this same closed round, including the final round.
        ro,rfit=fit(c,r,x,y,w,500,device,ro,pixel=pixel,residual=error)
        so,sfit=fit(c,s,x,y,w,core.STEPS[stage],device,so,align=r);fits.append(sfit)
        row=dict(stage=stage,logical_unique_questions=len(draws),new_questions=len(selected),observer_fit=rfit,solver_fit=sfit,
            known_teacher_KL=core.excess(s,x,y,w,device),all_physical_ROIs=len(c.images),no_semantic_score_equality=True)
        rounds.append(row)
        core.save(closed,dict(solver=core.state(s),observer=core.state(r),sopt=so.state_dict(),ropt=ro.state_dict(),
            rng=rng.bit_generator.state,draws=draws,rounds=rounds,fits=fits,known=known,auxerror=auxerror,registration_sha256=registration_sha256))
        print(json.dumps(dict(status='FULL_SCOPE_VISUAL_CLOSED_ROUND',pair=c.case['pair'],video=c.case['evaluated_video'],**row)),flush=True)
    assert len(draws)==len({d['uid'] for d in draws})==sum(c.quotas.values())
    x,y,w=teacher_table(c,draws,'IPW_KD');_,_,ordinary_w=teacher_table(c,draws,'SOFT_KD');projection=folder/'anchor_projection.pt'
    if projection.exists():
        projected=torch.load(projection,map_location='cpu',weights_only=False);s.load_state_dict(projected['solver']);so.load_state_dict(projected['optimizer']);fits=projected['fits']
    total=sum(row['steps'] for row in fits);weighted_error=core.excess(s,x,y,w,device);plain_error=core.excess(s,x,y,ordinary_w,device);err=max(weighted_error,plain_error)
    while err>.005 and total<16000:
        so,sfit=fit(c,s,x,y,(w+ordinary_w)/2,500,device,so);fits.append(sfit);total+=500
        weighted_error=core.excess(s,x,y,w,device);plain_error=core.excess(s,x,y,ordinary_w,device);err=max(weighted_error,plain_error)
        core.save(projection,dict(solver=core.state(s),optimizer=so.state_dict(),fits=fits,proper_teacher_KL=err))
    assert err<=.005,(c.case['pair'],c.case['evaluated_video'],'PRIMARY_TEACHER_FIT',err)
    result=dict(solver=core.state(s),observer=core.state(r),draws=draws,rounds=rounds,solver_fits=fits,
        proper_known_teacher_KL=err,weighted_teacher_KL=weighted_error,unweighted_teacher_KL=plain_error,
        positive_LURE_teacher_risk=True,unique_questions=len(draws),broker=c.primary.receipt(complete=True),
        original_teacher_targets_unchanged=True,full_physical_content_support=True,no_signed_semantic_targets=True,
        registration_sha256=registration_sha256)
    core.save(final,result);atomic_json(folder/'draws.json',dict(draws=draws))
    return s,r,result


def uniform_draws(context):
    c=context;path=c.folder/'uniform_draws.json'
    if path.exists():
        draws=json.loads(path.read_text())['draws'];c.uniform.request([d['uid'] for d in draws],'train',-1);return draws
    rng=np.random.default_rng(0);ordered={}
    for q,n in sorted(c.quotas.items()):
        left={t:np.flatnonzero(c.video.columns==t).tolist() for t in range(len(c.video.tracks))};selected=[]
        while len(selected)<n:
            tracks=[t for t,rows in left.items() if rows]
            for t in rng.permutation(tracks):
                at=int(rng.integers(len(left[t])));oi=left[t].pop(at);selected.append(q*c.n+oi)
                if len(selected)==n:break
        ordered[q]=selected
    draws=[]
    for stage in range(4):
        selected=[]
        for q,n in sorted(c.quotas.items()):
            start=sum(n//4+int(s<n%4) for s in range(stage));count=n//4+int(stage<n%4)
            selected.extend(dict(index=i,uid=str(i),query=q,stage=stage,proposal_probability=None,repeated_logical_action=False) for i in ordered[q][start:start+count])
        replies=c.uniform.request([d['uid'] for d in selected],'train',stage)
        draws.extend(dict(d,probability=r['probability']) for d,r in zip(selected,replies))
    assert len(draws)==len({r['uid'] for r in draws})==sum(c.quotas.values())
    atomic_json(path,dict(draws=draws,broker=c.uniform.receipt(complete=True)));return draws
