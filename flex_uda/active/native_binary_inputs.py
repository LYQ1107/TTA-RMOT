"""Unlabelled observation actions and budgeted native binary teacher access.

Only the private broker indexes prepaid replies. Learners receive requested
answers, never the remaining library. Pair products are modelling assumptions,
not measured cross-object joint answers. No manual target annotation enters.
"""
from collections import Counter,defaultdict
import hashlib
from itertools import zip_longest
import json
import numpy as np

from scripts.a100_observation_design_collect import OUT as ANSWERS,PREP,TEACHER_SHA
from scripts.a100_observation_design_prepare import POLICIES
from scripts.a100_observation_teacher_pilot import key as request_key


def inventory(pair,video):
    actions={};quotas=None
    for policy in POLICIES:
        requests=json.loads((PREP/pair/video/(policy+'_requests.json')).read_text())['requests']
        counts=Counter(int(r['local_query']) for r in requests)
        if quotas is None:quotas=dict(counts)
        else:assert dict(counts)==quotas
        for r in requests:
            uid=hashlib.sha256(request_key(r).encode()).hexdigest()
            row=dict(uid=uid,local_query=int(r['local_query']),local_track=int(r['local_track']),case=r)
            if uid in actions:
                assert request_key(actions[uid]['case'])==request_key(r)
            actions[uid]=row
    assert actions and min(quotas.values())>=1
    return actions,quotas


class BinaryBroker:
    def __init__(self,pair,video,actions,quotas,reservation):
        self.pair,self.video,self.actions,self.quotas=pair,video,actions,quotas
        self.reservation=reservation;self.received={};self.sequence=[];self._answers=None

    def _load_private(self):
        bank={}
        for policy in POLICIES:
            doc=json.loads((ANSWERS/'policy_answers'/self.pair/self.video/(policy+'_answers.json')).read_text())
            assert doc['GT_open_attempts']==0 and doc['only_this_policy_requested_responses']
            for row in doc['answers']:
                uid=hashlib.sha256(request_key(row).encode()).hexdigest()
                assert uid in self.actions and row['frozen_weight_sha256']==TEACHER_SHA
                if uid in bank:assert bank[uid]['yes_probability']==row['yes_probability']
                bank[uid]=row
        assert set(bank)==set(self.actions)
        self._answers=bank

    def request(self,uids,role,stage):
        assert role in ('train','probe') and len(uids)==len(set(uids))
        if self._answers is None:self._load_private()
        counts=Counter(r['local_query'] for r in self.received.values());result=[]
        for uid in uids:
            assert uid in self.actions and uid not in self.received
            action=self.actions[uid];q,t=action['local_query'],action['local_track']
            if role=='train':assert (q,t) not in self.reservation['groups']
            else:assert uid in self.reservation['probes'][stage]
            counts[q]+=1;assert counts[q]<=self.quotas[q]
            answer=self._answers[uid];p=float(answer['yes_probability'])
            assert np.isfinite(p) and 0<=p<=1
            row=dict(action,probability=p,teacher_request_id=answer['request_id'],
                     frozen_weight_sha256=TEACHER_SHA,effective_view=answer['effective_view'])
            self.received[uid]=row;result.append(row)
            self.sequence.append(dict(uid=uid,role=role,stage=stage))
        return result

    def receipt(self,complete=False):
        counts=dict(Counter(r['local_query'] for r in self.received.values()))
        if complete:assert counts==self.quotas
        return dict(unique_teacher_bundles=len(self.received),budget=sum(self.quotas.values()),
            per_query_count=counts,per_query_budget=self.quotas,
            paid_probe_bundles=sum(r['role']=='probe' for r in self.sequence),
            normal_binary_teacher=True,frozen_weight_sha256=TEACHER_SHA,
            actual_new_teacher_GPU_calls=0,prepaid_library_replay_not_deployment_cost=True,
            requested_UID_SHA256=hashlib.sha256('\n'.join(sorted(self.received)).encode()).hexdigest(),
            exposed_teacher_value_SHA256=hashlib.sha256(json.dumps(
                [(u,self.received[u]['probability'],self.received[u]['teacher_request_id']) for u in sorted(self.received)],
                separators=(',',':')).encode()).hexdigest(),
            unrequested_replies_exposed_to_learner=0)


def coarse_context(anchors,present):
    weight=present[...,None]
    result=(anchors*weight).sum(-2)/weight.sum(-2)
    histogram=np.square(anchors[...,768:1080]).reshape(*anchors.shape[:-1],3,104)
    flags=anchors[...,1080:1083]*weight
    hist=(histogram*weight[...,None]).sum(-3)
    result[...,768:1080]=np.sqrt(hist/np.maximum(flags.sum(-2)[...,None],1)).reshape(*result.shape[:-1],312)
    return result.astype('float32')


class NativeVisual:
    def __init__(self,video,actions):
        self.video,self.actions=video,actions
        self.order=sorted(actions);self.partners={}
        groups=defaultdict(list)
        for uid in self.order:groups[actions[uid]['local_query']].append(uid)
        for group in groups.values():
            for i,uid in enumerate(group):
                t=actions[uid]['local_track']
                self.partners[uid]=next((u for u in group[i+1:]+group[:i] if actions[u]['local_track']!=t),group[(i+1)%len(group)])
        values=video.raw_bags[:,video.present].astype('float64')
        self.norm_sum=values.sum((0,1));self.norm_square=np.square(values).sum((0,1));self.norm_rows=values.shape[0]*values.shape[1]
        self.coarse=coarse_context(video.raw_bags,np.broadcast_to(video.present,video.raw_bags.shape[:-1]))

    def point_arrays(self,uids):
        actions=[self.actions[u] for u in uids]
        q=np.asarray([r['local_query'] for r in actions]);t=np.asarray([r['local_track'] for r in actions])
        rows=np.asarray([self.video.positions[int(r['case']['frame_id']),int(r['case']['track_id'])] for r in actions])
        assert all(self.video.queries[qi]['expression_id']==r['case']['query_id'] for qi,r in zip(q,actions))
        assert all(int(self.video.tracks[ti])==r['case']['track_id'] for ti,r in zip(t,actions))
        return dict(point_anchors=self.video.raw_bags[q,t],point_present=self.video.present[t],
            point_current=self.video.dense_rows(q,rows),coarse_inputs=self.coarse[q,t])

    def arrays(self,uids):
        a=self.point_arrays(uids);b=self.point_arrays([self.partners[u] for u in uids])
        return dict(anchors_A=a['point_anchors'],present_A=a['point_present'],
            anchors_B=b['point_anchors'],present_B=b['point_present'],
            first=a['point_current'],second=b['point_current'])

    def record(self,received):
        grouped=defaultdict(list)
        for row in received:
            r=row['case'];key=row['local_query'],row['local_track'],int(r['frame_id'])
            grouped[key].append(row)
        keys=sorted(grouped);assert keys
        representatives=[grouped[k][0]['uid'] for k in keys]
        point=self.point_arrays(representatives)
        probability=np.asarray([np.mean([r['probability'] for r in grouped[k]]) for k in keys])
        query=defaultdict(lambda:defaultdict(list))
        for i,(q,t,_) in enumerate(keys):query[q][t].append(i)
        ia=[];ib=[]
        for q,tracks in sorted(query.items()):
            order=[x for layer in zip_longest(*(tracks[t] for t in sorted(tracks))) for x in layer if x is not None]
            ia.extend(order);ib.extend(order[1:]+order[:1])
        ia,ib=np.asarray(ia),np.asarray(ib)
        a,b=probability[ia],probability[ib]
        target=np.c_[a*(1-b),(1-a)*b,a*b,(1-a)*(1-b)]
        self_pair=ia==ib;target[self_pair]=np.c_[np.zeros(self_pair.sum()),np.zeros(self_pair.sum()),a[self_pair],1-a[self_pair]]
        assert np.array_equal(np.sort(ia),np.arange(len(keys))) and np.array_equal(np.sort(ib),np.arange(len(keys)))
        record=dict(anchors_A=point['point_anchors'][ia],present_A=point['point_present'][ia],
            anchors_B=point['point_anchors'][ib],present_B=point['point_present'][ib],
            first=point['point_current'][ia],second=point['point_current'][ib],pair_target=target,
            point_target=probability,**point)
        record['report']=dict(paid_binary_requests=len(received),unique_visual_nodes=len(keys),pair_rows=len(ia),
            candidate_identity_pairs=int(sum(keys[i][1]!=keys[j][1] for i,j in zip(ia,ib))),
            temporal_same_track_pairs=int(sum(keys[i][1]==keys[j][1] and i!=j for i,j in zip(ia,ib))),
            singleton_same_event_pairs=int(self_pair.sum()),every_unique_node_occurs_once_as_each_endpoint=True,
            causal_CLIP_binary_teacher_assigned_to_requested_anchor_approximation=True,
            product_targets_are_independence_model_not_observed_joint_answers=True)
        return record


def prefix_arrays(records,mean,std):
    keys=('anchors_A','present_A','anchors_B','present_B','first','second','pair_target',
          'point_anchors','point_present','point_current','point_target','coarse_inputs')
    result={key:np.concatenate([r[key] for r in records]) for key in keys}
    for key in ('anchors_A','anchors_B','first','second','point_anchors','point_current','coarse_inputs'):
        result[key]=((result[key]-mean)/std).astype('float32')
    for a,p in (('anchors_A','present_A'),('anchors_B','present_B'),('point_anchors','point_present')):
        result[a][~result[p]]=0
    return result
