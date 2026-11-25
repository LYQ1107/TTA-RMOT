"""Shared frozen calls for A/B/C/D and matching-visual B-plus controls."""
from collections import deque
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from .budget_meter import BudgetMeter
from .evidence_verifier import confidence_decision,verify
from .teacher_cache import TeacherCache,request_key
from .visual_input_builder import VisualInputBuilder

VARIANTS=('original','no_image','candidate_occluded','background_occluded','previous_observation')


class TeacherProbe:
    def __init__(self,backend,stage,cache_path,prompts,rules,batch_size=4):
        self.backend=backend;self.views=VisualInputBuilder(stage,backend.upstream,
            backend.config['image_size'],backend.config['margin_ratio'])
        self.cache=TeacherCache(cache_path);self.prompts=prompts;self.rules=rules;self.batch_size=batch_size
        self.budget=BudgetMeter();self.last_refs=[]
        self.version=dict(weight_sha256=backend.weight_sha,condition=backend.config['condition'],
            yestrack_commit=backend.config['yestrack_commit'],precision=backend.config['precision'],
            backend_sha256=hashlib.sha256(Path(__file__).with_name('yestrack_backend.py').read_bytes()).hexdigest())

    def calls(self,sample,requests):
        return self.calls_many([(sample,request) for request in requests])

    def calls_many(self,pairs):
        outputs=[None]*len(pairs);pending=[];self.last_refs=[]
        def flush():
            if not pending:return
            raw=self.backend.forward([p[2] for p in pending],[p[3] for p in pending]);self.budget.add(raw)
            for (index,key,images,prompt,meta),record in zip(pending,raw):
                record=dict(record,request_id=key,prompt=prompt,input_view=meta,teacher_version=self.version)
                self.cache.put(key,record);outputs[index]=record
            pending.clear()
        for index,(sample,request) in enumerate(pairs):
            view,variant,template,fields,frame=request
            images,frames,coord,meta=self.views.build(sample,view,variant,frame)
            if meta.get('missing'):
                self.last_refs.append(None);continue
            prompt=template.format(coord=coord,**fields)
            spec=dict(teacher=self.version,prompt=prompt,image_sha256=meta.get('image_sha256',[]))
            key=request_key(spec);self.last_refs.append(key)
            if any(p[1]==key for p in pending):flush()
            record=self.cache.get_or_claim(key,wait=False)
            if record is False:
                flush();record=self.cache.get_or_claim(key)
            if record is not None:
                self.budget.cache_hits+=1;outputs[index]=record
            else:pending.append((index,key,images,prompt,meta))
            if len(pending)>=self.batch_size:flush()
        flush();return outputs

    def prefetch(self,samples,method):
        """Batch only independent frozen calls across candidates; never labels."""
        pairs=[]
        for sample in samples:
            attrs=sample['attributes'];fields=dict(sentence=sample['query_text'])
            if method=='YES_FULL' or not attrs and method in ('STRUCTURED_EVIDENCE','EVIDENCE_VERIFIED'):
                jobs=[('current','original',self.prompts['official_single'],fields,None)]
            elif method in ('YES_ENSEMBLE_BUDGET_MATCHED','YES_ENSEMBLE_B_PLUS'):
                jobs=self._full_requests(sample,attrs or [dict(view='current')],VARIANTS if method=='YES_ENSEMBLE_B_PLUS' else ('original',))
            elif method in ('STRUCTURED_EVIDENCE','EVIDENCE_VERIFIED'):
                variants=VARIANTS if method=='EVIDENCE_VERIFIED' else ('original',)
                jobs=[(a['view'],variant,self.prompts['attribute'],dict(attribute=a['text']),None) for a in attrs for variant in variants]
            elif method=='SOURCE_RELIABILITY':
                jobs=[('history',variant,self.prompts['official_video'],fields,None) for variant in VARIANTS]
            else:return
            pairs.extend((sample,request) for request in jobs)
        self.calls_many(pairs)

    def _full_requests(self,sample,attributes,variants=('original',)):
        result=[]
        for j,a in enumerate(attributes):
            template=self.prompts['equivalent_full'][j%len(self.prompts['equivalent_full'])]
            for variant in variants:result.append((a['view'],variant,template,dict(sentence=sample['query_text']),None))
        return result

    def evaluate(self,sample,method):
        attributes=sample['attributes'];positive=self.rules['positive_threshold'];negative=self.rules['negative_threshold']
        if method=='YES_FULL':
            raw=self.calls(sample,[('current','original',self.prompts['official_single'],dict(sentence=sample['query_text']),None)])
            r=raw[0];decision=confidence_decision(r['yes_probability'],r['top1_is_yes_no'],positive,negative)
            return dict(decision=asdict(decision),request_ids=self.last_refs.copy(),raw_response=r)
        if method in ('YES_ENSEMBLE_BUDGET_MATCHED','YES_ENSEMBLE_B_PLUS'):
            attrs=attributes or [dict(view='current')]
            variants=VARIANTS if method=='YES_ENSEMBLE_B_PLUS' else ('original',)
            raw=self.calls(sample,self._full_requests(sample,attrs,variants));refs=self.last_refs.copy()
            usable=[r for i,r in enumerate(raw) if r is not None and i%len(variants)==0]
            probability=sum(r['yes_probability'] for r in usable)/len(usable) if usable else 0.5
            valid=bool(usable) and all(r['top1_is_yes_no'] for r in usable)
            decision=confidence_decision(probability,valid,positive,negative)
            if method=='YES_ENSEMBLE_B_PLUS':
                observations=[dict(zip(VARIANTS,raw[j*len(VARIANTS):(j+1)*len(VARIANTS)])) for j in range(len(attrs))]
                full_conditions=[dict(text=sample['query_text'],kind='FULL_EXPRESSION',view=a['view']) for a in attrs]
                decision=verify(full_conditions,observations,positive,negative,self.rules['visual_dependence_min'],self.rules['probability_spread_max'])
            return dict(decision=asdict(decision),request_ids=refs,all_full_probabilities=[r['yes_probability'] if r else None for r in raw],
                unverified_ensemble_probability=probability,
                control_note='B-plus receives exactly D views AND the same verifier, but asks equivalent complete queries rather than decomposed attributes; stronger decomposition control')
        if method=='SOURCE_RELIABILITY':
            raw=self.calls(sample,[('history',variant,self.prompts['official_video'],dict(sentence=sample['query_text']),None) for variant in VARIANTS])
            return dict(request_ids=self.last_refs.copy(),source_full_expression_views=dict(zip(VARIANTS,raw)),
                note='legal source-domain-only calibration; no target label is consumed')
        if method in ('STRUCTURED_EVIDENCE','EVIDENCE_VERIFIED'):
            if not attributes:
                result=self.evaluate(sample,'YES_FULL');result['decision'].update(state='UNKNOWN',reasons=['UNSUPPORTED_EXPRESSION'])
                return result
            variants=('original',) if method=='STRUCTURED_EVIDENCE' else VARIANTS
            jobs=[(a['view'],variant,self.prompts['attribute'],dict(attribute=a['text']),None) for a in attributes for variant in variants]
            raw=self.calls(sample,jobs);refs=self.last_refs.copy()
            if method=='STRUCTURED_EVIDENCE':
                usable=[r for r in raw if r is not None];p=min((r['yes_probability'] for r in usable),default=0.5)
                state=confidence_decision(p,len(usable)==len(attributes) and all(r['top1_is_yes_no'] for r in usable),positive,negative)
                return dict(decision=asdict(state),request_ids=refs,attribute_probabilities=[r['yes_probability'] if r else None for r in raw])
            observations=[dict(zip(VARIANTS,raw[j*len(VARIANTS):(j+1)*len(VARIANTS)])) for j in range(len(attributes))]
            decision=verify(attributes,observations,positive,negative,self.rules['visual_dependence_min'],self.rules['probability_spread_max'])
            return dict(decision=asdict(decision),request_ids=refs,attribute_views=observations)
        if method=='YESTRACK_TCP_TRP':return self.tcp_trp(sample,positive,negative)
        raise ValueError('unknown teacher method '+method)

    def tcp_trp(self,sample,positive,negative):
        ep=self.views.episode(sample['video_id']);anchor=sample['frame_id'];track=sample['track_id']
        index=ep.frames.index(anchor);clip=ep.frames[max(0,index-12):index+1]
        history=deque(maxlen=6);last=None;refs=[];inference_frames=[];raw=None
        for counter,frame in enumerate(clip):
            if track not in ep.observations_by_frame[frame]:continue
            fresh=counter%4==0 or last is None
            if fresh:
                raw=self.calls(sample,[('current','original',self.prompts['official_single'],dict(sentence=sample['query_text']),frame)])[0]
                refs.extend(self.last_refs);last=raw['yes_probability'];inference_frames.append(frame)
                if 0.2<=last<0.8:
                    raw=self.calls(sample,[('history','original',self.prompts['official_video'],dict(sentence=sample['query_text']),frame)])[0]
                    refs.extend(self.last_refs);last=raw['yes_probability']
            if len(history)>=6 and all(p>=0.4 for p in history):last=min(1.,last+0.3)
            history.append(last)
        state=confidence_decision(last,raw is not None and raw['top1_is_yes_no'],positive,negative)
        return dict(decision=asdict(state),request_ids=refs,policy='official_TCP_TRP_with_user_approved_causal13frame_clip_reset',
            clip_frames=clip,inference_frames=inference_frames,full_video_HOTA=False)
