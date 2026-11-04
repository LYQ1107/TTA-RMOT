"""v6 current-model full-video cache and fixed32-frame refresh, no GT IO."""
from collections import defaultdict
import time
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.presence.features import margins_and_features
from flex_uda.model.flexhook_uda import prepared_inputs
from flex_uda.presence.cardinality import expected_iou_utilities
from flex_uda.presence.residual import head_from_checkpoint
from flex_uda.active.v6 import selection_fractions


class FrameCacheV6:
    def __init__(self,model,cache,bank,queries,frames,head_path,folder):
        from scripts.active_v6_prepare import sample32
        self.model,self.cache,self.bank,self.queries=model,cache,bank,queries
        self.frames=list(map(int,frames));self.frame_index={t:i for i,t in enumerate(self.frames)}
        self.sample_frames=sample32(self.frames)
        self.observations=np.asarray(bank['observations']);self.tracks=sorted(set(self.observations[:,1]))
        self.positions={tuple(row):i for i,row in enumerate(self.observations.tolist())}
        self.by_track={t:np.flatnonzero(self.observations[:,1]==t) for t in self.tracks}
        self.by_frame={t:np.flatnonzero(self.observations[:,0]==t) for t in self.frames}
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.texts=[cache._text(q['query_text']) for q in queries]
        q,n=len(queries),len(self.observations)
        self.margin=np.lib.format.open_memmap(self.folder/'margin.npy',mode='w+',dtype='float32',shape=(q,n))
        self.pcd=np.lib.format.open_memmap(self.folder/'pcd.npy',mode='w+',dtype='float32',shape=(q,n,768))
        self.row_model_steps=np.zeros(n,dtype=np.int64)
        checkpoint=torch.load(head_path,map_location='cpu',weights_only=False)
        self.head=head_from_checkpoint(checkpoint).eval()
        self.head_path=str(head_path);self.head_sha=file_sha(head_path)
        self.raw_presence=None;self.utilities=None;self.raw_k=None
        self.score_records=[];self.decode_seconds=0.
        atomic_json(self.folder/'fixed32.json',dict(frames=self.sample_frames,
                    rule='uniform frame-number anchors; nearest existing frame, ties lower; fixed at video start'))

    @torch.no_grad()
    def score(self,*,sampled=False,model_step=0):
        started=time.perf_counter();frames=self.sample_frames if sampled else self.frames
        rows=0
        for fi,frame in enumerate(frames):
            if not len(self.by_frame[frame]):continue
            candidates=list(self.cache.episode.current_candidates(frame))
            for ca in range(0,len(candidates),8):
                prepared=prepared_inputs(self.cache,candidates[ca:ca+8],self.queries[0]['query_text'],frame)
                ids=np.asarray([self.positions[frame,t] for t in prepared['track_ids']])
                for qa in range(0,len(self.texts),8):
                    value,feature=margins_and_features(self.model,prepared,self.texts[qa:qa+8])
                    if not torch.isfinite(value).all() or not torch.isfinite(feature).all():
                        raise FloatingPointError('Nonfinite current-model v6 frame cache')
                    self.margin[qa:qa+value.shape[1],ids]=value.float().cpu().numpy().T
                    self.pcd[qa:qa+value.shape[1],ids]=feature.float().cpu().numpy().transpose(1,0,2)
                    rows+=value.numel()
                self.row_model_steps[ids]=model_step
            if fi%32==0 or fi==len(frames)-1:
                import json
                print(json.dumps(dict(event='V6_FRAME_CACHE_PROGRESS',scope='FIXED32' if sampled else 'ALL',
                    completed_frames=fi+1,total_frames=len(frames),candidate_query_rows=rows)),flush=True)
        torch.cuda.synchronize(self.cache.device)
        self.margin.flush();self.pcd.flush()
        record=dict(kind='FIXED32' if sampled else 'VIDEO_START_ALL',frames=list(frames),
            candidate_query_rows=rows,model_step=model_step,GPU_seconds=time.perf_counter()-started)
        self.score_records.append(record)
        self._head_update(frames)
        return record

    @torch.no_grad()
    def _head_update(self,frames):
        begin=time.perf_counter();nq=len(self.queries);nf=len(self.frames)
        width=max(map(len,self.by_frame.values()))+1
        if self.raw_presence is None:
            self.raw_presence=np.zeros((nq,nf),dtype=float)
            self.utilities=np.full((nq,nf,width),-1.,dtype=float)
            self.raw_k=np.zeros((nq,nf),dtype=np.int64)
        units=[(q,t) for q in range(nq) for t in frames]
        for start in range(0,len(units),512):
            part=units[start:start+512];size=max(len(self.by_frame[t]) for _,t in part)
            matrix=np.zeros((len(part),size),dtype='float64');padding=np.ones_like(matrix,dtype=bool)
            for j,(q,t) in enumerate(part):
                ids=self.by_frame[t];matrix[j,:len(ids)]=self.margin[q,ids];padding[j,:len(ids)]=False
            result=self.head(margin=torch.from_numpy(matrix),padding_mask=torch.from_numpy(padding))
            utility=expected_iou_utilities(result['candidate_probabilities'],torch.from_numpy(padding)).numpy()
            for j,(q,t) in enumerate(part):
                column=self.frame_index[t]
                self.raw_presence[q,column]=float(result['presence'][j])
                self.utilities[q,column]=-1.
                self.utilities[q,column,:size+1]=utility[j]
                self.raw_k[q,column]=int(utility[j].argmax())
        self.decode_seconds+=time.perf_counter()-begin

    def decode(self,temperature,bias,*,sampled=False):
        begin=time.perf_counter()
        if not sampled:
            result=selection_fractions(self.observations,self.tracks,self.frame_index,self.margin,
                self.utilities,self.raw_presence,temperature=temperature,bias=bias,raw_k=self.raw_k)
            self.decode_seconds+=time.perf_counter()-begin
            return result
        chosen=np.isin(self.observations[:,0],self.sample_frames)
        obs=self.observations[chosen]
        if not len(obs):return None
        tracks=sorted(set(obs[:,1]));columns=[self.frame_index[t] for t in self.sample_frames]
        index={t:i for i,t in enumerate(self.sample_frames)}
        fractions,prob,mask=selection_fractions(obs,tracks,index,np.asarray(self.margin[:,chosen]),
            self.utilities[:,columns],self.raw_presence[:,columns],temperature=temperature,bias=bias,
            raw_k=self.raw_k[:,columns])
        self.decode_seconds+=time.perf_counter()-begin
        return tracks,fractions

    def refresh(self,state,temperature,bias,*,retain_sampled=False):
        fractions,prob,_=self.decode(temperature,bias)
        state.replace_fractions(fractions)
        if retain_sampled:
            self.refresh_sampled(state,temperature,bias)
        state.refresh_presence(prob.max(1))
        state.replace_prior(scores=self.track_means())

    def refresh_sampled(self,state,temperature,bias):
        result=self.decode(temperature,bias,sampled=True)
        if result is None:return
        tracks,fractions=result;updated=state.fractions.copy()
        lookup={t:i for i,t in enumerate(self.tracks)}
        for j,t in enumerate(tracks):updated[:,lookup[t]]=fractions[:,j]
        state.replace_fractions(updated)
        state.refresh_presence(expit(self.raw_presence/temperature+bias).max(1))
        state.replace_prior(scores=self.track_means())

    def track_means(self):
        return np.stack([self.margin[:,self.by_track[t]].mean(axis=1) for t in self.tracks],axis=1)

    def close(self):
        self.margin.flush();self.pcd.flush()
        np.save(self.folder/'row_model_steps.npy',self.row_model_steps)
        np.save(self.folder/'presence_logits.npy',self.raw_presence)
        record=dict(status='CURRENT_MODEL_CACHE_COMPLETE',source_head=self.head_path,source_head_sha256=self.head_sha,
            pcd_precision='FP32',margin_precision='FP32',source_head_uses='margin only, unchanged two-slope Bernoulli head',
            records=self.score_records,decode_CPU_seconds=self.decode_seconds,
            files={p.name:dict(path=str(p),bytes=p.stat().st_size,sha256=file_sha(p))
                   for p in self.folder.iterdir() if p.suffix in ('.npy','.json') and p.name!='cache_receipt.json'})
        atomic_json(self.folder/'cache_receipt.json',record)
        return record
