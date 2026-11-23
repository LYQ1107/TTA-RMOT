"""Rebuild missing frozen A100 source arrays, without any old TTA experiment."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import OLD,metadata,generation_guard


def main(list_only=False):
    guard=generation_guard();m=metadata('SportsMOT');root=OLD/'stage2/canonical'
    missing=[v for v in sorted(m['videos']) if not (root/v/'ready.json').is_file()]
    if list_only or not missing:
        print(json.dumps(dict(status='CANONICAL_INVENTORY',videos=len(m['videos']),missing=missing,
            no_existing_cache_overwritten=True,no_TTA_or_teacher_started=True)));return
    import torch
    from flex_uda.a100.runtime import load_student,source_head
    from flex_uda.presence.runtime import episode_from_bank
    from flex_tta.runtime.frozen_features import FrozenFeatureCache
    from flex_uda.active.v6_frames import FrameCacheV6
    from scripts.a100_visual_classical_prepare import OUT
    torch.set_num_threads(4);torch.manual_seed(0);device=torch.device('cuda:0')
    work=OUT/'canonical_rebuild';work.mkdir(parents=True,exist_ok=True)
    model,tok,transform,loaded=load_student(work/'model_log',device)
    atomic_json(work/'model_load.json',loaded)
    for video in missing:
        begin=time.perf_counter();bank=m['videos'][video];queries=[q for q in m['episodes'] if q['video_id']==video]
        ep=episode_from_bank(video,bank,'SportsMOT',transform);cache=FrozenFeatureCache(model,ep,tok,device,max_bytes=512<<20)
        fc=FrameCacheV6(model,cache,bank,queries,sorted(set(np.asarray(bank['observations'])[:,0])),source_head(),work/video,store_pcd=False)
        fc.score(model_step=0);destination=root/video;destination.mkdir(parents=True,exist_ok=True)
        assert not (destination/'ready.json').exists(),'Existing canonical inputs are immutable'
        path=destination/'initial.npz';temporary=path.with_suffix('.partial')
        with temporary.open('wb') as handle:np.savez(handle,margin=np.asarray(fc.margin),raw_presence=fc.raw_presence,utilities=fc.utilities,raw_k=fc.raw_k)
        temporary.replace(path)
        atomic_json(destination/'ready.json',dict(status='A100_ZERO_WEIGHT_CANONICAL',video=video,optimizer_updates=0,
            checkpoint='official Dance-best',source_head=str(source_head()),initial_arrays_sha256=file_sha(path),
            source_only_rebuild=True,GT_open_attempts=len(guard)))
        fc.close();del fc,cache,ep;torch.cuda.empty_cache()
        print(json.dumps(dict(event='FROZEN_SOURCE_CANONICAL_REBUILT',video=video,wall_seconds=time.perf_counter()-begin)),flush=True)
    assert not guard


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--list',action='store_true');a=p.parse_args();main(a.list)
