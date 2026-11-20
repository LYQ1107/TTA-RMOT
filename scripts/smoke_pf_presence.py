#!/usr/bin/env python3
"""One-query <=12-frame frozen interaction prototype; NEVER formal method seven."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def run(args):
    import compat.sitecustomize  # noqa: F401
    import torch
    from transformers import RobertaTokenizerFast
    from scripts.run_flexhook_dance_pcd_tta import _load_model, _model_margin
    from scripts.run_flexhook_positivefree import SOURCE_SHA256
    from scripts.summarize_pf_full import METHOD_ORDER
    from flex_tta.runtime.pf_receipts import atomic_json, file_sha
    from flex_tta.runtime.dance_episode import DanceEpisode
    from flex_tta.runtime.frozen_features import FrozenFeatureCache
    from flex_tta.adaptation.presence_evidence import CausalInteractionAudit, frozen_interaction_scores, interaction_contrast
    from flex_tta.dance_io import dance_videos,load_expressions

    visible=os.environ.get("CUDA_VISIBLE_DEVICES","")
    if not visible or len(visible.split(","))!=1:
        raise ValueError("select exactly one physical GPU explicitly")
    for method in METHOD_ORDER:
        m=json.loads((ROOT/"results/positive_free/full"/method.lower()/"metrics.json").read_text())
        if m["full_audit"]["valid_episodes"]!=425:
            raise ValueError("finish the registered six full methods first")
    output=args.output_root.resolve()
    output.mkdir(parents=True,exist_ok=False)
    dataset=args.dataset_root.resolve()
    args.tracker_root=args.tracker_root.resolve()
    access=dict(target_gt_open_attempts=[],target_file_opens=0)

    def audit_open(event,values):
        if event!="open" or not values or not isinstance(values[0],(str,bytes,os.PathLike)):
            return
        path=Path(os.fsdecode(values[0])).absolute()
        if path.is_relative_to(dataset):
            if "gt_template" in path.parts or path.name in {"gt.txt","labels.json"}:
                access["target_gt_open_attempts"].append(str(path))
                raise PermissionError("target GT is not an interaction-prototype input")
            access["target_file_opens"]+=1
    sys.addaudithook(audit_open)
    checkpoint=Path("/data3/user/TTA-RMOT-assets/flexhook/SOTA_ckpts/refer-kitti-v2-best.pth")
    if file_sha(checkpoint)!=SOURCE_SHA256:
        raise ValueError("pinned source checkpoint mismatch")
    device=torch.device("cuda:0")
    budget=int(os.environ.get("PF_GPU_MEMORY_BUDGET_MIB","12288"))<<20
    torch.cuda.set_per_process_memory_fraction(budget/torch.cuda.get_device_properties(device).total_memory,device)
    torch.manual_seed(0)
    started=time.perf_counter()
    vendor=ROOT/"vendor/FlexHook"
    _,model,_=_load_model(vendor,checkpoint,dataset,args.tracker_root,output/"model_log",device,24,strict_checkpoint=True)
    tokenizer=RobertaTokenizerFast.from_pretrained(str(vendor/"pretrained/roberta-base"))
    person_ids=tokenizer.encode("person",add_special_tokens=False)
    if person_ids!=[5970]:
        raise ValueError("RoBERTa tokenizer regression")
    versions={name:p._version for name,p in model.named_parameters()}
    video=dance_videos(args.tracker_root)[0]
    expression,sentence=load_expressions(dataset,video)[0]
    episode=DanceEpisode(dataset,args.tracker_root,video)
    cache=FrozenFeatureCache(model,episode,tokenizer,device,max_bytes=256<<20)
    history=CausalInteractionAudit()
    traces=[]
    reference_check=None
    with torch.no_grad():
        for frame in episode.frames[:args.max_frames]:
            candidates=list(episode.current_candidates(frame))
            latest=max(o.frame_id for _,hs in candidates for o in hs)
            if latest>frame:
                raise RuntimeError("future feature requested")
            ids,branches=frozen_interaction_scores(cache,candidates,sentence,frame)
            if any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in branches.values()):
                raise RuntimeError("interaction branches must be finite FP32")
            if reference_check is None:
                original,original_ids=_model_margin(model,episode,candidates,sentence,tokenizer,device)
                mismatch=int(((original>0)!=(branches["real_query"]>0)).sum())
                error=float((original-branches["real_query"]).abs().max())
                if original_ids!=ids or mismatch or error>=1e-3:
                    raise RuntimeError("original frozen branch changed")
                reference_check=dict(frame=frame,source_sign_mismatches=mismatch,max_abs_error=error,ids_equal=original_ids==ids)
            trace=history.step(frame,ids,**branches)
            trace.update(max_image_history_frame=latest,branch_margins={k:v.cpu().tolist() for k,v in branches.items()})
            # Algebraic synthetic offset probe; not a labelled semantic test.
            offset=torch.full_like(branches["real_query"],3.)
            altered=interaction_contrast(branches["real_query"]+offset+5,branches["real_reference"]+offset-2,
                                         branches["null_query"]-offset+5,branches["null_reference"]-offset-2)
            actual=interaction_contrast(**branches)
            trace["additive_offset_cancellation_max_error"]=float((altered-actual).abs().max())
            traces.append(trace)
    torch.cuda.synchronize()
    unchanged=all(p._version==versions[name] for name,p in model.named_parameters())
    if not unchanged or any(p.grad is not None or p.requires_grad for p in model.parameters()):
        raise RuntimeError("prototype must remain frozen and update-free")
    history.reset()
    result=dict(status="PASS_FROZEN_DIAGNOSTIC_ONLY",formal_method_seven=False,
                experiment_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                physical_gpu=os.environ.get("PF_PHYSICAL_GPU"),gpu_uuid=os.environ.get("PF_GPU_UUID",visible),
                resource_admission=dict(free_mib=os.environ.get("PF_GPU_ADMISSION_FREE_MIB"),
                                        process_count=os.environ.get("PF_GPU_ADMISSION_PROCESS_COUNT"),
                                        allocator_budget_mib=budget>>20,headroom_mib=4096,
                                        shared_gpu_authorized=True,not_a_memory_reservation=True),
                video=video,expression=expression,sentence=sentence,reference_sentence="person",
                frames=len(traces),expressions=1,source_checkpoint=str(checkpoint),checkpoint_sha256=SOURCE_SHA256,
                checkpoint_strict_load=True,person_token_ids=person_ids,precision="FP32",
                parameter_versions_unchanged=unchanged,trainable_parameters=0,optimizer_steps=0,
                no_target_gt_opened=not access["target_gt_open_attempts"],file_access_audit=access,
                source_branch_check=reference_check,cache_stats=cache.stats,
                wall_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                episode_reset_pass=not history.rows and history.last_frame==-1,
                presence_predictions="UNKNOWN, including stable interactions; no calibrated semantic absence/presence signal",
                prediction_override=None,no_new_loss=True,traces=traces)
    atomic_json(output/"smoke_receipt.json",result)
    atomic_json(ROOT/"results/positive_free/full/manifests/query_presence_prototype_smoke.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="traces"},indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--max-frames",type=int,default=12,choices=range(1,13))
    p.add_argument("--dataset-root",type=Path,default=Path("/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance"))
    p.add_argument("--tracker-root",type=Path,default=Path("/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs"))
    args=p.parse_args()
    run(args)


if __name__=="__main__": main()
