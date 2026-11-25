"""Three real frames: strict model, real answers, and balanced LoRA updates."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.active.execution import authorize_active
from flex_uda.presence.runtime import target_metadata,load_model,episode_from_bank,cache_for,RUN,runtime_guard
from flex_uda.active.v6 import ReliabilityV6
from flex_uda.active.v6_frames import FrameCacheV6
from flex_uda.active.v7 import BeliefStateV7,OnlinePresenceEB,fast_step_v7
from flex_uda.active.v7_slow import SlowAdapterV7
from flex_uda.active.teacher_client import TeacherClient
from scripts.active_run import video_spec


def main():
    execution,_=authorize_active();opened=runtime_guard();torch.manual_seed(0);np.random.seed(0)
    torch.cuda.set_per_process_memory_fraction(min(.9,(16<<30)/torch.cuda.get_device_properties(0).total_memory))
    out=execution.active/'smoke_C1';out.mkdir(parents=True,exist_ok=True)
    meta=target_metadata('Dance');video=sorted(meta['videos'])[0];original=meta['videos'][video]
    frames=sorted({t for t,_ in original['observations']})[:3]
    selected=[i for i,(t,_) in enumerate(original['observations']) if t in frames]
    bank=dict(original,observations=[original['observations'][i] for i in selected],lines=[original['lines'][i] for i in selected])
    queries=[q for q in meta['episodes'] if q['video_id']==video]
    model,tokenizer,transform,loaded=load_model('Dance','V2-best',out/'model_log',torch.device('cuda:0'))
    episode=episode_from_bank(video,bank,'Dance',transform);cache=cache_for(model,tokenizer,episode,torch.device('cuda:0'))
    slow=SlowAdapterV7(model,cache)
    fc=FrameCacheV6(model,cache,bank,queries,frames,execution.manifest['source_heads']['V2-best']['checkpoint'],out/'frames')
    fc.score();fraction,prob,_=fc.decode(1.,0.)
    state=BeliefStateV7(fc.track_means(),fraction,prob.max(1),track_lengths=[len(fc.by_track[t]) for t in fc.tracks])
    initial=state.pseudo_labels();assert (initial==-1).all()
    runtime=out/'teacher_runtime.json';atomic_json(runtime,[video_spec('Dance',video,bank)])
    teacher=TeacherClient(runtime,RUN/'teacher/raw.sqlite',out/'teacher.log',ROOT)
    ds=ReliabilityV6();track=next(t for t in fc.tracks if len(fc.by_track[t])==3);ti=fc.tracks.index(track)
    answers=[]
    try:
        for frame in frames:
            requests=[dict(target='Dance',video_id=video,frame_id=frame,track_id=track,
                query_id=q['expression_id'],query_text=q['query_text'],view='CROP') for q in queries]
            batch=teacher.ask_many(requests);answers+=batch
            for qi,a in enumerate(batch):
                yes=a['yes_probability']>=.5
                state.observe(qi,ti,yes,a['effective_view'],ds.values[a['effective_view']],presence_question=False)
                ds.add((video,queries[qi]['expression_id'],track),a['effective_view'],yes)
        ds.fit();state.recompute_presence(ds.values)
        result=fast_step_v7(state,fc.raw_presence)
        fc.refresh(state,result['temperature'],result['bias'])
        trained=slow.slow_step(state,fc.tracks,[q['query_text'] for q in queries],exclusive=True,steps=2)
        before=fc.margin.copy()
        if trained['optimizer_updates']:
            fc.score(sampled=True,model_step=slow.updates);fc.refresh_sampled(state,result['temperature'],result['bias'])
            assert all(np.isfinite([x['loss'],x['gradient_norm']]).all() for x in trained['records'] if x.get('optimizer_updated'))
        record=dict(status='V7_THREE_REAL_FRAME_SMOKE_PASSED',video=video,frames=frames,load=loaded,
            candidate_query_rows=fc.margin.size,PCD_shape=list(fc.pcd.shape),PCD_dtype=str(fc.pcd.dtype),
            trainable_parameters=sum(p.numel() for p in slow.parameters),teacher=teacher.close(),
            GT_open_attempts=len(opened),teacher_question_count=len(answers),initial_unasked_pseudo_labels=0,
            fast=result,slow=trained,ds=ds.receipt(),cache=fc.close(),
            mean_margin_change=float(np.abs(fc.margin-before).mean()),physical_gpu=os.environ.get('PRESENCE_PHYSICAL_GPU'),
            functional_smoke_not_registered_rho_episode=True,formal_RMOT_HOTA_not_computed=True)
        teacher=None
        assert not opened;atomic_json(execution.public/'gpu_smoke.json',record)
        print(json.dumps({k:v for k,v in record.items() if k not in ('teacher','cache','load')},default=str),flush=True)
    finally:
        if teacher is not None and teacher.child.poll() is None:teacher.close()


if __name__=='__main__':main()
