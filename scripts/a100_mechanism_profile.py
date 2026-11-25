"""Frozen, numerically checked feature-flow profiling on a fixed small slice."""
import time
PROCESS_BEGIN=time.perf_counter()
from collections import defaultdict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import OUT, metadata, generation_guard
from flex_uda.a100.runtime import load_student, source_head
from flex_uda.presence.runtime import episode_from_bank
from flex_tta.runtime.frozen_features import FrozenFeatureCache
from flex_uda.active.v6_frames import FrameCacheV6


def main():
    attempts=generation_guard();torch.set_num_threads(4);torch.manual_seed(0)
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    folder=OUT/'profile';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completion.json').is_file():return
    m=metadata('SportsMOT');video=sorted(m['videos'])[0];bank=m['videos'][video]
    queries=[q for q in m['episodes'] if q['video_id']==video]
    allframes=np.unique(np.asarray(bank['observations'])[:,0]);frames=allframes[np.linspace(0,len(allframes)-1,16,dtype=int)].tolist()
    t=time.perf_counter();model,tok,transform,loaded=load_student(folder/'model_log',device);load_seconds=time.perf_counter()-t
    records=[];reference=None
    for capacity in (512,8192):
        timers=defaultdict(float);calls=defaultdict(int)
        def timed(obj,name,key):
            previous=getattr(obj,name)
            def wrapped(*args,**kwargs):
                torch.cuda.synchronize(device);begin=time.perf_counter()
                result=previous(*args,**kwargs)
                torch.cuda.synchronize(device);timers[key]+=time.perf_counter()-begin;calls[key]+=1
                return result
            setattr(obj,name,wrapped)
            return previous
        ep=episode_from_bank(video,bank,'SportsMOT',transform)
        cache=FrozenFeatureCache(model,ep,tok,device,max_bytes=capacity<<20)
        restored=[]
        restored.append((ep,'_image',timed(ep,'_image','image_IO_and_transform')))
        backbone_method='forward_features' if hasattr(model.backbone,'forward_features') else 'forward'
        restored.append((model.backbone,backbone_method,timed(model.backbone,backbone_method,'backbone')))
        restored.append((cache,'_text',timed(cache,'_text','text_cache_and_encoder')))
        restored.append((model,'decode',timed(model,'decode','PCD_decode')))
        begin=time.perf_counter()
        fc=FrameCacheV6(model,cache,bank,queries,frames,source_head(),folder/str(capacity),store_pcd=False)
        restored.append((fc,'_head_update',timed(fc,'_head_update','CPU_cardinality')))
        fc.score();elapsed=time.perf_counter()-begin
        selected=np.isin(fc.observations[:,0],frames)
        values=np.asarray(fc.margin[:,selected]).copy()
        current=dict(margin=values,presence=fc.raw_presence.copy(),utilities=fc.utilities.copy(),raw_k=fc.raw_k.copy())
        equivalence={k:True for k in current} if reference is None else {k:np.array_equal(current[k],reference[k]) for k in current}
        if reference is None:reference=current
        row=dict(cache_mib=capacity,frames=frames,candidate_query_rows=int(values.size),feature_flow_wall_seconds=elapsed,
            phase_wall_seconds=dict(timers),phase_calls=dict(calls),other_seconds=elapsed-sum(timers.values()),
            frozen_cache_stats=cache.stats,bitwise_equal_to_first=equivalence,
            phases_include_synchronization_overhead=True,model_already_resident=True)
        records.append(row);atomic_json(folder/f'profile_{capacity}.json',row)
        fc.close()
        for obj,name,previous in reversed(restored):setattr(obj,name,previous)
        del fc,cache,ep;torch.cuda.empty_cache()
        print(json.dumps(dict(event='PROFILE_CAPACITY_COMPLETE',**row)),flush=True)
    assert all(all(r['bitwise_equal_to_first'].values()) for r in records)
    assert not attempts
    result=dict(status='PASS_NUMERICALLY_EQUIVALENT_PROFILE',physical_GPU=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
        model_load=loaded,student_load_wall_seconds=load_seconds,records=records,
        process_wall_seconds=time.perf_counter()-PROCESS_BEGIN,GT_open_attempts=len(attempts),
        limitations=['one fixed16-anchor slice; cannot extrapolate whole-C4 speedup',
            'capacity512 measured first,8192 second; shared OS cache retained, no randomized crossover',
            'phase timings include CPU/IO/synchronization and are not pureGPU kernel integrals',
            'exact feature caching already exists; no low-dimensional approximation used'])
    atomic_json(folder/'completion.json',result)


if __name__=='__main__':main()
