#!/usr/bin/env python3
"""Package already-completed diagnostic outputs; no inference or fitting."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.pf_diag_common import OUT


def sha(data):
    return hashlib.sha256(data).hexdigest()


def archive(path,files):
    entries=[]
    with tarfile.open(path,"w:gz") as target:
        for source,name in sorted(files,key=lambda item:item[1]):
            data=source.read_bytes()
            item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o644;item.mtime=0
            target.addfile(item,io.BytesIO(data))
            entries.append(dict(name=name,bytes=len(data),sha256=sha(data)))
    with tarfile.open(path,"r:gz") as packed:
        members=packed.getmembers()
        if [m.name for m in members]!=[e["name"] for e in entries]:raise ValueError("archive listing differs")
        for member,entry in zip(members,entries):
            if sha(packed.extractfile(member).read())!=entry["sha256"]:raise ValueError("archive payload differs")
    return dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path.read_bytes()),
        verified_entries=len(entries),entries=entries)


def plot(output,count,query,retrieval,probe,qualification):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    ax=axes[0,0];names=["ZERO_UPDATE","COUNT_MATCHED_FROZEN","PF_IDENTITY"]
    vals=[count["metrics"][n]["HOTA"] for n in names]
    ax.bar(range(3),vals,color=["#64748b","#d97706","#2563eb"])
    ax.set(xticks=range(3),xticklabels=["Frozen","Count-matched\nPOST-HOC ONLY","Identity"],ylim=(0,10),ylabel="Official HOTA",
        title="A. Full 25 videos / 425 expressions")
    for j,v in enumerate(vals):ax.text(j,v+.12,f"{v:.4f}",ha="center")
    ax.text(.03,.08,"Count comes from the executed Identity run.\nNot an independently deployable threshold.",transform=ax.transAxes,fontsize=9)

    ax=axes[0,1];names=["mean_margin","max_margin","positive_fraction","visual_mean_margin"]
    xx=np.arange(len(names));a=[query["presence"][n]["AUROC"] for n in names];p=[query["presence"][n]["AUPRC"] for n in names]
    ax.bar(xx-.17,a,width=.34,label="AUROC",color="#2563eb");ax.bar(xx+.17,p,width=.34,label="AUPRC",color="#0d9488")
    ax.axhline(query["presence_random_AUPRC"],color="#64748b",ls=":",label="Random AP")
    ax.set(xticks=xx,xticklabels=["Mean","Max","Positive\nfraction","Query-mean\nvisual control"],ylim=(0,1),
        title="B. Episode presence: offline aggregates",ylabel="Score separation")
    ax.legend(fontsize=8,loc="upper right")

    ax=axes[1,0];names=["visual_roi","pcd_person","background","geometry"]
    vals=[retrieval["features"][n]["top1_same_clothing_label"] for n in names]
    ax.bar(range(4),vals,color=["#64748b","#2563eb","#a78bfa","#94a3b8"])
    ax.axhline(retrieval["features"]["pcd_person"]["random_other_video_gallery"],ls=":",color="#dc2626",label="Random gallery")
    ax.set(xticks=range(4),xticklabels=["Visual ROI","Generic PCD","Background","Geometry"],ylim=(0,1),ylabel="Shared-clothing-label Top1",
        title="C. No-training cross-video retrieval (603 obs.)")
    for j,v in enumerate(vals):ax.text(j,v+.02,f"{100*v:.1f}%",ha="center")
    ax.legend(fontsize=8,loc="upper right")
    ax.text(.03,.07,"PCD - background 95% CI includes zero.\nGT defines relevance; only 5/17 types span >=2 videos.",transform=ax.transAxes,fontsize=9)

    ax=axes[1,1];names=["pcd_query","text_only","background_and_text","geometry_and_text"]
    vals=[qualification["same_sample_source_macro_AUROC"]]+[probe["metrics"][n]["mixed_frame_query_macro_AUROC"] for n in names]
    ax.bar(range(5),vals,color=["#d97706","#2563eb","#94a3b8","#a78bfa","#64748b"])
    ax.axhline(.5,ls=":",color="#dc2626",label="Chance / constant score")
    ax.set(xticks=range(5),xticklabels=["Frozen\nhead","Query PCD\nprobe","Text only","Background\n+ text","Geometry\n+ text"],ylim=(0,1),ylabel="Within-frame mixed-candidate AUROC",
        title="D. Same 69 groups: GT probe, 5-fold VIDEO CV")
    for j,v in enumerate(vals):ax.text(j,v+.02,f"{v:.3f}",ha="center")
    ax.text(.03,.06,"Probe does NOT exceed the same-sample frozen head.\nFixed C=1; diagnostic, not source-free learning.",transform=ax.transAxes,fontsize=9)
    fig.suptitle("V2 -> Dance recoverability diagnosis (not new TTA results)",fontsize=15)
    directory=output/"figures";directory.mkdir(exist_ok=True)
    fig.savefig(directory/"recoverability_summary.png",dpi=180)
    fig.savefig(directory/"recoverability_summary.pdf")
    plt.close(fig)


def qualify_probe(output):
    """Inspect saved OOF readouts, not refit them; same-sample source control."""
    import numpy as np
    from scripts.analyze_pf_full_offline import auroc
    from scripts.pf_diag_common import write_csv
    with (output/"features/probe_oof_scores.csv").open() as stream:rows=list(csv.DictReader(stream))
    source={}
    plan=json.loads((output/"features/sampling_manifest.json").read_text())
    for video in plan["videos"]:
        for sample in video["samples"]:
            with np.load(output/"features/cache"/(video["video"]+"_"+str(sample["frame"])+".npz")) as data:
                for index,track_id in enumerate(data["ids"]):
                    source[(video["video"],sample["frame"],int(track_id))]=data["margin"][index]
    groups={}
    for r in rows:groups.setdefault((r["video"],int(r["frame"]),int(r["query_index"])),[]).append(r)
    mixed=[];ranges={name:0. for name in ("text_only","background_and_text")}
    names=("source","pcd_query","text_only","background_and_text","geometry_and_text")
    for (video,frame,query),items in groups.items():
        labels=np.asarray([int(r["gt_candidate_match"]) for r in items],dtype=bool)
        for name in ranges:ranges[name]=max(ranges[name],float(np.ptp([float(r[name]) for r in items])))
        if not labels.any() or labels.all():continue
        scores={name:[float(r[name]) for r in items] for name in names if name!="source"}
        scores["source"]=[float(source[(video,frame,int(r["track_id"]))][query]) for r in items]
        mixed.append(dict(video=video,frame=frame,query_index=query,candidates=len(items),positives=int(labels.sum()),
            **{name+"_AUROC":auroc(values,labels) for name,values in scores.items()}))
    write_csv(output/"features/probe_mixed_frame_groups.csv",mixed)
    videos=sorted({r["video"] for r in mixed});n=len(mixed)
    all_videos=[v["video"] for v in plan["videos"]]
    per=[np.asarray([r["pcd_query_AUROC"]-r["source_AUROC"] for r in mixed if r["video"]==v]) for v in all_videos]
    boot=[]
    for draw in np.random.default_rng(0).integers(0,25,size=(2000,25)):
        difference=np.concatenate([per[j] for j in draw])
        if len(difference):boot.append(float(difference.mean()))
    result=dict(protocol="OFFLINE_SAVED_READOUT_QUALIFICATION_NO_REFITTING",mixed_frame_query_groups=n,
        eligible_videos=len(videos),eligible_video_ids=videos,
        same_sample_source_macro_AUROC=float(np.mean([r["source_AUROC"] for r in mixed])),
        pcd_probe_macro_AUROC=float(np.mean([r["pcd_query_AUROC"] for r in mixed])),
        pcd_probe_minus_same_sample_source=dict(delta=float(np.mean([r["pcd_query_AUROC"]-r["source_AUROC"] for r in mixed])),
            paired_video_CI95=np.percentile(boot,[2.5,97.5]).tolist(),valid_resamples=len(boot)),
        constant_control_max_within_group_score_range=ranges,
        constant_control_interpretation="mathematically constant within frame/query; raw ~0.5 AUROC can differ at floating-point dot-product roundoff; not real candidate discrimination",
        limits="69 mixed frame-query groups are correlated, not69 independent videos. Pooled OOF AUROC/AP also contain query/scene priors and cross-fold readout calibration differences.")
    atomic_json(output/"features/probe_qualification.json",result)
    return result


def seal(output=OUT):
    output=Path(output).resolve()
    load=lambda name:json.loads((output/name).read_text())
    protocol=load("protocol_audit.json");count=load("countmatched/comparison.json")
    query=load("query_information/summary.json");retrieval=load("features/retrieval_summary.json")
    probe=load("features/linear_probe_summary.json");extraction=load("features/extraction_receipt.json")
    if protocol["status"]!="PASS" or count["episodes"]!=425 or query["episodes"]!=425:raise ValueError("incomplete CPU diagnosis")
    if extraction["frames"]!=75 or extraction["videos"]!=25 or extraction["optimizer_steps"]!=0:raise ValueError("feature protocol differs")
    for fold in probe["fold_details"]:
        if set(fold["train_videos"])&set(fold["test_videos"]):raise ValueError("probe video leakage")
    if len(probe["fold_details"])!=20 or probe["probe_weights_saved"]:raise ValueError("probe incomplete or exported")
    subprocess.run(["git","diff","--exit-code","29f265cf6b2c8d82948043534e0044b1341d5899","HEAD","--","results/positive_free/full"],cwd=ROOT,check=True)
    qualification=qualify_probe(output)
    plot(output,count,query,retrieval,probe,qualification)
    conclusion=dict(status="COMPLETED_STOPPED_NO_NEW_FULL_TTA",interpretation="all entries are offline diagnostics, not new TTA methods",
        identity_count_effect=dict(frozen_HOTA=count["metrics"]["ZERO_UPDATE"]["HOTA"],
            identity_HOTA=count["metrics"]["PF_IDENTITY"]["HOTA"],countmatched_HOTA=count["metrics"]["COUNT_MATCHED_FROZEN"]["HOTA"],
            descriptive_gain_ratio=count["descriptive_gain_ratio"],causal_percentage_established=False,
            independently_deployable_count_rule_found=False),
        query_presence=dict(preset_offline_max=query["presence"]["max_margin"],
            random_AUPRC=query["presence_random_AUPRC"],online_GT_selected_threshold=False),
        frozen_features=dict(generic_PCD_cross_video_clothing_Top1=retrieval["features"]["pcd_person"]["top1_same_clothing_label"],
            random_gallery=retrieval["features"]["pcd_person"]["random_other_video_gallery"],
            probe_same_frame_AUROC=qualification["pcd_probe_macro_AUROC"],
            original_head_same_frame_AUROC=qualification["same_sample_source_macro_AUROC"],
            diagnostic_probe_beats_original_head=False,all17_types_semantically_generalized=False),
        validated_independent_unlabeled_semantic_learning_signal="NOT_ESTABLISHED",
        research_decision="No evidence for another full LoRA TTA run. Some frozen query/appearance signal remains; causal consistency may audit UNKNOWN, but is not a reliable semantic label.")
    atomic_json(output/"diagnostic_conclusion.json",conclusion)
    folder=output/"archives";folder.mkdir(exist_ok=True)
    predictions=sorted((output/"countmatched/COUNT_MATCHED_FROZEN/results").glob("*/*/predict.txt"))
    if len(predictions)!=425:raise ValueError("Count-Matched must have all425 prediction files")
    files=[(p,"countmatched_predictions/"+"/".join(p.parts[-3:])) for p in predictions]
    prediction_archive=archive(folder/"countmatched_425_predictions.tar.gz",files)
    log_files=[]
    for p in output.rglob("*"):
        if p.name=="seal_stdout.txt":continue  # this command's log is still being written
        if p.is_file() and p.suffix in (".txt",".log") and ("stdout" in p.name or "log_rank" in p.name or p.name in (
            "final_tests.txt","protocol_audit_verified.txt","probe_dependency_install.txt","linear_probe_missing_dependency.txt")):
            log_files.append((p,str(p.relative_to(output))))
    log_archive=archive(folder/"diagnostic_complete_logs.tar.gz",log_files)
    result=dict(status="COMPLETE_DIAGNOSTIC_ONLY_NO_NEW_TTA",base_public_commit="29f265cf6b2c8d82948043534e0044b1341d5899",
        original_six_published_result_files_unchanged=True,phases=dict(protocol=True,countmatched425=True,query_information425=True,
        frozen_feature_frames=75,frozen_feature_videos=25,all17_expression_types=True,probe_video_folds=5,probe_fits=20),
        no_new_full_TTA=True,no_source_training=True,no_target_gt_threshold_selection=True,
        fixed_feature_manifest_sha256=extraction["plan_sha256"],checkpoint_sha256=extraction["checkpoint_sha256"],
        actual_feature_source_commit=extraction["experiment_commit"],actual_probe_source_commit=probe["experiment_commit"],
        raw_feature_cache=str(output/"features/cache"),raw_feature_cache_git_tracked=False,
        probe_qualification=qualification,
        archives=dict(countmatched_predictions=prediction_archive,complete_diagnostic_logs=log_archive),
        excluded_from_archives=["target GT","target images","checkpoint","margin streams","raw feature arrays","probe weights"],
        stop="diagnosis completed; do not automatically start another full TTA run")
    atomic_json(output/"delivery_manifest.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="archives"},indent=2))
    print(json.dumps({k:{p:v for p,v in a.items() if p!="entries"} for k,a in result["archives"].items()},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output-root",type=Path,default=OUT)
    seal(parser.parse_args().output_root)
