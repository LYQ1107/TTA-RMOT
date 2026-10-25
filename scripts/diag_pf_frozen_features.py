#!/usr/bin/env python3
"""Frozen75-frame diagnosis, not TTA. Extraction and GT evaluation are separate.

plan/smoke/extract never open target annotations. evaluate uses offline labels
for cross-video retrieval relevance. Optional probe is GT-supervised diagnosis
with video-grouped CV; no weights are saved or exported to TTA.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.pf_diag_common import DATA,TRACKS,FROZEN,OUT,candidate_labels,protocol_pass,write_csv
from flex_tta.dance_io import dance_videos,load_tracker_lines
from flex_tta.runtime.pf_receipts import atomic_json,file_sha

QUANTILES=(.1,.5,.9)
SOURCE_SHA="71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613"


def sampled_frames(frames):
    return [frames[int(round(q*(len(frames)-1)))] for q in QUANTILES]


def plan(output=OUT):
    output=Path(output)
    metadata=json.loads((FROZEN/"ZERO_UPDATE/run_metadata.json").read_text())
    episodes={(e["video"],e["expression_id"]):e for e in metadata["episodes"]}
    videos=[]
    for video in dance_videos(TRACKS):
        selected=sampled_frames(sorted(load_tracker_lines(TRACKS,video)))
        expressions=sorted((e,record["sentence"]) for (v,e),record in episodes.items() if v==video)
        records={t:dict(frame=t,reference_margins=[]) for t in selected}
        for expression,_ in expressions:
            path=FROZEN/"ZERO_UPDATE/results"/video/expression/"candidate_margins.jsonl.gz"
            with gzip.open(path,"rt") as stream:
                for line in stream:
                    row=json.loads(line);t=row["frame"]
                    if t in records:
                        records[t]["ids"]=row["ids"]
                        records[t]["reference_margins"].append(row["adapted"])
                    if t>=max(selected):break
        videos.append(dict(video=video,expressions=expressions,samples=[records[t] for t in selected]))
    result=dict(protocol="GT_INDEPENDENT_FIXED_SAMPLING_FROZEN_DIAGNOSIS",quantiles=list(QUANTILES),videos=videos,
        total_videos=25,total_frames=75,real_expressions_per_frame=17,extra_generic_control="person",
        source_checkpoint_sha256=SOURCE_SHA,uses_target_annotations=False,uses_gt_to_select_frames=False,
        selection="round(q*(N_active_MOT_frames-1)), all candidates; source score references do NOT select frames",
        interpretation="Internal feature availability does not imply source-free learnability; no cross-modal cosine between unaligned visual/RoBERTa spaces")
    path=output/"features/sampling_manifest.json"
    if path.exists() and json.loads(path.read_text())!=result:raise ValueError("fixed sampling manifest may not change")
    atomic_json(path,result)
    print(json.dumps({k:v for k,v in result.items() if k!="videos"},indent=2))
    return result


def extract(output=OUT,smoke=False):
    import compat.sitecustomize  # noqa: F401
    import torch
    import torch.nn.functional as F
    from transformers import RobertaTokenizerFast
    from scripts.run_flexhook_dance_pcd_tta import _load_model
    from flex_tta.runtime.dance_episode import DanceEpisode
    from flex_tta.runtime.frozen_features import FrozenFeatureCache
    from flex_tta.adaptation.identity_contrast import PCDFeatureTap
    output=Path(output).resolve();folder=output/"features"
    manifest=json.loads((folder/"sampling_manifest.json").read_text());fingerprint=file_sha(folder/"sampling_manifest.json")
    visible=os.environ.get("CUDA_VISIBLE_DEVICES","")
    if not visible or len(visible.split(","))!=1:raise ValueError("one audited GPU required for this small extraction")
    opens=dict(target_annotation_open_attempts=[],target_image_opens=0)
    def deny(event,values):
        if event!="open" or not isinstance(values[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(values[0])).absolute()
        if path.is_relative_to(DATA):
            if "gt_template" in path.parts or "expression" in path.parts or path.name=="labels.json":
                opens["target_annotation_open_attempts"].append(str(path));raise PermissionError("target annotations are not feature-extraction inputs")
            opens["target_image_opens"]+=1
    sys.addaudithook(deny)
    checkpoint=Path("/data3/user/TTA-RMOT-assets/flexhook/SOTA_ckpts/refer-kitti-v2-best.pth")
    if file_sha(checkpoint)!=SOURCE_SHA:raise ValueError("wrong source checkpoint")
    device=torch.device("cuda:0");budget=int(os.environ.get("PF_GPU_MEMORY_BUDGET_MIB","12288"))<<20
    torch.cuda.set_per_process_memory_fraction(budget/torch.cuda.get_device_properties(device).total_memory,device)
    torch.manual_seed(0);started=time.perf_counter()
    _,model,_=_load_model(ROOT/"vendor/FlexHook",checkpoint,DATA,TRACKS,folder/"model_logs"/("smoke" if smoke else "full"),device,24,strict_checkpoint=True)
    tokenizer=RobertaTokenizerFast.from_pretrained(str(ROOT/"vendor/FlexHook/pretrained/roberta-base"))
    if tokenizer.encode("person",add_special_tokens=False)!=[5970]:raise ValueError("tokenizer regression")
    tap=PCDFeatureTap(model);versions={n:p._version for n,p in model.named_parameters()}
    packets=[];computed=0;reused=0;maximum_error=0.;signs=0
    (folder/"cache").mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        for entry in manifest["videos"][:1] if smoke else manifest["videos"]:
            video=entry["video"];episode=DanceEpisode(DATA,TRACKS,video)
            cache=FrozenFeatureCache(model,episode,tokenizer,device,max_bytes=256<<20)
            for sample in entry["samples"][:1] if smoke else entry["samples"]:
                t=sample["frame"];path=folder/"cache"/(video+"_"+str(t)+".npz");rp=path.with_suffix(".json")
                if rp.exists():
                    receipt=json.loads(rp.read_text())
                    if receipt["plan_sha256"]!=fingerprint or file_sha(path)!=receipt["data_sha256"]:raise ValueError("feature resume mismatch")
                    packets.append(receipt);reused+=1;continue
                candidates=list(episode.current_candidates(t));ids=[i for i,_ in candidates]
                if ids!=sample["ids"]:raise ValueError("sample candidate IDs differ from sealed source")
                latest=max(o.frame_id for _,hs in candidates for o in hs)
                if latest>t:raise ValueError("future feature requested")
                pcd=[];margins=[];language=[]
                for expression,sentence in entry["expressions"]:
                    margin,actual=cache.margin(candidates,sentence,t);features=tap.take()
                    if actual!=ids or features.shape!=(len(ids),768):raise ValueError("PCD dimensions/IDs changed")
                    pcd.append(features.float().cpu().numpy().copy());margins.append(margin.cpu().numpy())
                    text,mask=cache._text(sentence)
                    language.append(((text[0,0]*mask[0,0,:,None]).sum(0)/mask.sum()).cpu().numpy())
                cache.margin(candidates,"person",t);person=tap.take().float().cpu().numpy().copy()
                layers=cache._visual(t,t);positions,boxes,_=episode.geometry_inputs(candidates,device)
                grid=positions[:,-1].permute(0,2,3,1)
                def roi(layer):return F.grid_sample(layer.expand(len(ids),-1,-1,-1),grid,padding_mode="zeros",align_corners=False)
                current=roi(layers[-1]);shallow=roi(layers[0])
                upper_lower=torch.cat([current[:,:,:8].mean((2,3)),current[:,:,8:].mean((2,3))],dim=1)
                source=np.stack(margins,axis=1);reference=np.asarray(sample["reference_margins"],dtype=np.float32).T
                error=float(np.max(np.abs(source-reference)));mismatch=int(np.sum((source>0)!=(reference>0)))
                if error>=1e-3 or mismatch:raise ValueError("frozen feature source logits do not reproduce sampled sealed source")
                arrays=dict(frame=np.full(len(ids),t),ids=np.asarray(ids),pcd=np.stack(pcd,axis=1),pcd_person=person,
                    visual_roi=current.mean((2,3)).cpu().numpy(),visual_upper_lower=upper_lower.cpu().numpy(),
                    shallow_roi=shallow.mean((2,3)).cpu().numpy(),background=np.repeat(layers[-1].mean((2,3)).cpu().numpy(),len(ids),axis=0),
                    geometry=boxes[:,-1].cpu().numpy(),language=np.stack(language),margin=source)
                if any(not np.isfinite(a).all() for a in arrays.values()):raise ValueError("nonfinite frozen feature")
                partial=path.with_suffix(".partial.npz");np.savez_compressed(partial,**arrays);os.replace(partial,path)
                receipt=dict(video=video,frame=t,ids=ids,candidates=len(ids),real_expressions=17,max_image_history_frame=latest,
                    source_max_abs_error=error,source_sign_mismatches=mismatch,plan_sha256=fingerprint,data_sha256=file_sha(path),
                    path=str(path),dimensions={k:list(v.shape) for k,v in arrays.items()},precision="FP32",checkpoint_strict_load=True)
                atomic_json(rp,receipt);packets.append(receipt);computed+=1;maximum_error=max(maximum_error,error);signs+=mismatch
            print(json.dumps(dict(event="feature_video_complete",video=video,samples=len(packets))),flush=True)
    torch.cuda.synchronize();tap.close()
    unchanged=all(p._version==versions[n] for n,p in model.named_parameters())
    if not unchanged or any(p.requires_grad or p.grad is not None for p in model.parameters()):raise ValueError("source model was not fully frozen")
    if len(packets)!=(1 if smoke else 75):raise ValueError("incomplete fixed feature plan")
    result=dict(status="PASS_FROZEN_SMOKE" if smoke else "COMPLETE_FIXED_25_VIDEO_75_FRAME_FEATURES",samples=packets,
        frames=len(packets),videos=len({p["video"] for p in packets}),newly_computed_frames=computed,reused_frames=reused,
        observations=sum(p["candidates"] for p in packets),all17_real_expressions=True,extra_control="generic person PCD to avoid clothing-text injection",
        optimizer_steps=0,trainable_parameters=0,parameter_versions_unchanged=unchanged,file_access_audit=opens,
        physical_gpu=os.environ.get("PF_PHYSICAL_GPU"),gpu_uuid=os.environ.get("PF_GPU_UUID",visible),
        wall_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        checkpoint_sha256=SOURCE_SHA,person_token_id=5970,plan_sha256=fingerprint,
        experiment_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        source_max_abs_error=max(p["source_max_abs_error"] for p in packets),source_sign_mismatches=sum(p["source_sign_mismatches"] for p in packets))
    atomic_json(folder/("smoke_receipt.json" if smoke else "extraction_receipt.json"),result)
    print(json.dumps({k:v for k,v in result.items() if k!="samples"},indent=2));return result


def load_features(output):
    folder=Path(output)/"features";manifest=json.loads((folder/"sampling_manifest.json").read_text())
    extraction=json.loads((folder/"extraction_receipt.json").read_text())
    if extraction["frames"]!=75 or extraction["videos"]!=25:raise ValueError("evaluate only the complete fixed sample")
    arrays={};labels=[];groups=[]
    for video in manifest["videos"]:
        packets=[];slices=[];start=0
        for sample in video["samples"]:
            path=folder/"cache"/(video["video"]+"_"+str(sample["frame"])+".npz")
            with np.load(path) as data:packet={k:data[k] for k in data.files}
            n=len(packet["ids"]);slices.append((sample["frame"],start,start+n));start+=n;packets.append(packet)
        combined={k:np.concatenate([p[k] for p in packets]) for k in packets[0] if k!="language"}
        combined["language"]=np.broadcast_to(packets[0]["language"],(start,17,768)).copy()
        bank=dict(video=video["video"],expressions=video["expressions"],scores=combined["margin"],slices=slices,ids=combined["ids"])
        y,_=candidate_labels(bank);labels.append(y);groups.extend([video["video"]]*start)
        for k,v in combined.items():arrays.setdefault(k,[]).append(v)
    return {k:np.concatenate(v) for k,v in arrays.items()},np.concatenate(labels),np.asarray(groups)


def cross_video_nn(features,labels,groups):
    """No fitted weights. GT defines relevance/evaluation set, never similarities."""
    f=np.asarray(features,dtype=np.float64);f=f/np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-12)
    similarities=f@f.T;allowed=groups[:,None]!=groups[None,:]
    shared=(labels.astype(np.int8)@labels.astype(np.int8).T)>0
    similarities[~allowed]=-np.inf
    nearest=allowed&np.isclose(similarities,similarities.max(axis=1,keepdims=True),atol=1e-7,rtol=0)
    hit=(nearest&shared).sum(axis=1)/nearest.sum(axis=1)
    random=(allowed&shared).sum(axis=1)/allowed.sum(axis=1)
    retrievable=(allowed&shared).any(axis=1)
    return hit,random,retrievable


def evaluate(output=OUT):
    output=Path(output);protocol_pass(output);x,y,groups=load_features(output)
    annotated=y.any(axis=1);g=groups[annotated];ys=y[annotated]
    features=("visual_roi","visual_upper_lower","shallow_roi","pcd_person","background","geometry")
    results={};rows=[];videos=sorted(set(groups));draws=np.random.default_rng(0).integers(0,25,size=(2000,25))
    for name in features:
        hit,random,retrievable=cross_video_nn(x[name][annotated],ys,g)
        per=[np.flatnonzero(g==v) for v in videos]
        boot=[]
        for draw in draws:
            ix=np.concatenate([per[j] for j in draw])
            if len(ix):boot.append(float(np.mean(hit[ix]-random[ix])))
        results[name]=dict(annotated_candidate_observations=len(g),top1_same_clothing_label=float(hit.mean()),
            random_other_video_gallery=float(random.mean()),delta=float((hit-random).mean()),
            paired_video_CI95=np.percentile(boot,[2.5,97.5]).tolist(),
            gallery_retrievable_fraction=float(retrievable.mean()))
        for video,ix in zip(videos,per):rows.append(dict(video=video,feature=name,observations=len(ix),
            top1=float(hit[ix].mean()) if len(ix) else None,random=float(random[ix].mean()) if len(ix) else None))
    label_support=[dict(expression=e,observations=int(y[:,j].sum()),videos=len(set(groups[y[:,j]]))) for j,(e,_) in enumerate(json.loads((output/"features/sampling_manifest.json").read_text())["videos"][0]["expressions"])]
    result=dict(protocol="OFFLINE_FROZEN_FEATURE_RETRIEVAL_NOT_SOURCE_FREE_TTA",total_frames=75,total_videos=25,
        all_candidate_observations=len(groups),annotated_observations=int(annotated.sum()),features=results,expression_support=label_support,
        frozen_source_sample_metrics=None,trained_weights=0,
        gallery_rule="GT-matched clothing-annotated candidates from other videos; GT defines relevance only, no model training; ties averaged",
        clothing_label="at least one shared actual official referring-clothing expression; not an independently annotated exhaustive clothing attribute set",
        limits="diagnostic GT relevance/gallery, correlated people/frames and possible cross-video scene/dancer similarity; compare background/geometry and random gallery; does not establish unsupervised identifiability")
    from scripts.diag_pf_query_information import binary_summary
    result["frozen_source_sample_metrics"]=binary_summary(x["margin"].ravel(),y.ravel())
    atomic_json(output/"features/retrieval_summary.json",result);write_csv(output/"features/per_video_retrieval.csv",rows)
    print(json.dumps(result,indent=2));return result


def probe(output=OUT):
    """Optional fixed-C linear readout; only after no-training retrieval review."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from scripts.diag_pf_query_information import binary_summary
    output=Path(output);protocol_pass(output);x,y,groups=load_features(output)
    labels=y.ravel();g=np.repeat(groups,17);q=x["language"].reshape(-1,768)
    representations=dict(pcd_query=x["pcd"].reshape(-1,768),text_only=q,
        background_and_text=np.concatenate([np.repeat(x["background"],17,axis=0),q],axis=1),
        geometry_and_text=np.concatenate([np.repeat(x["geometry"],17,axis=0),q],axis=1))
    outputs={};folds=[]
    for name,features in representations.items():
        scores=np.zeros(len(labels),dtype=np.float64)
        for fold,(train,test) in enumerate(GroupKFold(n_splits=5).split(features,labels,g)):
            if set(g[train])&set(g[test]):raise ValueError("video leakage in diagnostic probe")
            scale=StandardScaler().fit(features[train]);model=LogisticRegression(C=1.,class_weight="balanced",solver="liblinear",random_state=0,max_iter=1000)
            model.fit(scale.transform(features[train]),labels[train]);scores[test]=model.decision_function(scale.transform(features[test]))
            folds.append(dict(feature=name,fold=fold,train_videos=sorted(set(g[train])),test_videos=sorted(set(g[test])),iterations=model.n_iter_.tolist()))
        outputs[name]=binary_summary(scores,labels)
    result=dict(protocol="GT_SUPERVISED_DIAGNOSTIC_LINEAR_PROBE_NOT_TTA",group_unit="video",folds=5,C=1.,seed=0,
        selected_hyperparameters_from_target_metrics=False,probe_weights_saved=False,probe_weights_used_in_TTA=False,metrics=outputs,fold_details=folds)
    atomic_json(output/"features/linear_probe_summary.json",result);print(json.dumps(result,indent=2));return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=("plan","smoke","extract","evaluate","probe"),required=True)
    p.add_argument("--output-root",type=Path,default=OUT);a=p.parse_args();root=a.output_root.resolve()
    if a.stage=="plan":plan(root)
    elif a.stage in ("smoke","extract"):extract(root,smoke=a.stage=="smoke")
    elif a.stage=="evaluate":evaluate(root)
    else:probe(root)
