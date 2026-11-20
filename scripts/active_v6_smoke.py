"""Three real frames through strict FlexHook, cached presence, and real teacher."""
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.active.execution import authorize_active
from flex_uda.presence.runtime import target_metadata,load_model,episode_from_bank,cache_for,RUN,runtime_guard
from flex_uda.active.v6 import BeliefStateV6,ReliabilityV6,fast_step_v6
from flex_uda.active.v6_frames import FrameCacheV6
from flex_uda.active.teacher_client import TeacherClient
from flex_uda.active.adapt import SlowAdapter
from flex_tta.runtime.pf_receipts import atomic_json
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
    slow=SlowAdapter(model,cache)
    fc=FrameCacheV6(model,cache,bank,queries,frames,execution.manifest['source_heads']['V2-best']['checkpoint'],out/'frames')
    fc.score();fractions,prob,_=fc.decode(1.,0.)
    state=BeliefStateV6(fc.track_means(),fractions,prob.max(1),
        track_lengths=[len(fc.by_track[t]) for t in fc.tracks])
    assert (state.pseudo_labels()==-1).all()
    runtime=out/'teacher_runtime.json';atomic_json(runtime,[video_spec('Dance',video,bank)])
    teacher=TeacherClient(runtime,RUN/'teacher/raw.sqlite',out/'teacher.log',ROOT)
    ds=ReliabilityV6();requests=[]
    for t in fc.tracks[:3]:
        frame=int(fc.observations[fc.by_track[t][0],0])
        requests.append(dict(target='Dance',video_id=video,frame_id=frame,track_id=int(t),
            query_id=queries[0]['expression_id'],query_text=queries[0]['query_text'],view='CROP'))
    answers=teacher.ask_many(requests)
    for t,answer in zip(fc.tracks,answers):
        ti=fc.tracks.index(t);yes=answer['yes_probability']>=.5
        state.observe(0,ti,yes,answer['effective_view'],ds.values[answer['effective_view']],presence_question=True)
        ds.add((video,queries[0]['expression_id'],int(t)),answer['effective_view'],yes)
    ds.fit();result=fast_step_v6(state,fc.raw_presence)
    fc.refresh(state,result['temperature'],result['bias'])
    before=fc.margin.copy();fc.score(sampled=True);fc.refresh_sampled(state,result['temperature'],result['bias'])
    assert np.array_equal(before,fc.margin)
    assert not ((state.pseudo_labels()>=0)&~state.asked_mask).any()
    assert max(abs(e['delta']) for e in state.events)<=2
    record=dict(status='V6_REAL_THREE_FRAME_SMOKE_PASSED',video=video,frames=frames,load=loaded,
        candidate_query_rows=fc.margin.size,PCD_shape=list(fc.pcd.shape),PCD_dtype=str(fc.pcd.dtype),
        actual_trainable_parameters=sum(p.numel() for p in slow.parameters),
        unchanged_model_rescore_exact=True,teacher=teacher.close(),GT_open_attempts=len(opened),
        teacher_question_count=len(answers),initial_unasked_pseudo_labels=0,fast=result,ds=ds.receipt(),
        cache=fc.close(),physical_gpu=os.environ.get('PRESENCE_PHYSICAL_GPU'),full_RMOT_not_evaluated=True)
    assert not opened;atomic_json(execution.public/'gpu_smoke.json',record)
    print(json.dumps({k:v for k,v in record.items() if k not in ('cache','teacher','load')},default=str),flush=True)


if __name__=='__main__':main()
