#!/usr/bin/env python3
"""POST-HOC DIAGNOSTIC: Frozen top-k, k from actual sealed Identity outputs.

Run --stage predict and --stage evaluate as separate processes. No deployable
TTA method, no GT-defined counts, no new inference, no threshold optimization.
"""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.pf_diag_common import DATA,TRACKS,FROZEN,IDENTITY,OUT,protocol_pass,official_metrics,official_video_values,write_csv
from flex_tta.runtime.pf_receipts import EpisodeStore,atomic_json
from flex_tta.dance_io import load_tracker_lines

METHOD="COUNT_MATCHED_FROZEN"


def topk_ids(ids,scores,k):
    return set(sorted(ids,key=lambda i:(-scores[i],i))[:k])


def generate(output=OUT):
    output=Path(output)
    root=output/"countmatched";method_root=root/METHOD
    source_audit=json.loads((FROZEN/"ZERO_UPDATE/full_audit.json").read_text())
    identity_audit=json.loads((IDENTITY/"PF_IDENTITY/full_audit.json").read_text())
    if source_audit["status"]!="PASS" or identity_audit["valid_episodes"]!=425:
        raise ValueError("only sealed full source/Identity streams are allowed")
    access=dict(target_annotation_open_attempts=[])
    def no_gt(event,values):
        if event!="open" or not isinstance(values[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(values[0])).absolute()
        if path.is_relative_to(DATA) and ("gt_template" in path.parts or "expression" in path.parts or path.name=="labels.json"):
            access["target_annotation_open_attempts"].append(str(path))
            raise PermissionError("Count-Matched prediction cannot open target annotations")
    sys.addaudithook(no_gt)
    records=[];audit_episodes=[];started=time.perf_counter()
    for video in sorted({e["video"] for e in source_audit["episodes"]}):
        tracks=load_tracker_lines(TRACKS,video)
        for old in (e for e in source_audit["episodes"] if e["video"]==video):
            expression=old["expression"]
            relative=Path("results")/video/expression
            src=FROZEN/"ZERO_UPDATE"/relative
            ident=IDENTITY/"PF_IDENTITY"/relative
            store=EpisodeStore(method_root,video,expression,
                dict(protocol="POST_HOC_DIAGNOSTIC",rule="Frozen top-k with Identity actual k",frozen=str(FROZEN),identity=str(IDENTITY)),resume=True)
            receipt=store.completed()
            if receipt is None:
                start=time.perf_counter();source_count=selected=frames=candidates=0
                id_selected=Counter(int(float(line.split(",")[0])) for line in (ident/"predict.txt").read_text().splitlines())
                pp,mp=store.begin();mp.symlink_to((src/"candidate_margins.jsonl.gz").resolve())
                with pp.open("w") as dest,gzip.open(src/"candidate_margins.jsonl.gz","rt") as s,gzip.open(ident/"candidate_margins.jsonl.gz","rt") as it:
                    observed=[]
                    for a,b in zip(s,it,strict=True):
                        a,b=json.loads(a),json.loads(b)
                        t,ids=a["frame"],a["ids"]
                        if b["frame"]!=t or set(b["ids"])!=set(ids) or set(ids)!=set(tracks[t]):
                            raise ValueError("candidate universe differs")
                        k=sum(v>0 for v in b["adapted"])
                        if k!=id_selected[t]:raise ValueError("Identity actual selected count differs from sealed scores")
                        scores=dict(zip(ids,a["adapted"]))
                        chosen=topk_ids(ids,scores,k)
                        for i in sorted(chosen):dest.write(tracks[t][i]+"\n")
                        observed.append(t);frames+=1;candidates+=len(ids);selected+=len(chosen)
                        source_count+=sum(v>0 for v in scores.values())
                if observed!=sorted(tracks):raise ValueError("incomplete count-matched episode")
                receipt=store.seal(pp,mp,dict(full_frame_coverage=True,wall_seconds=time.perf_counter()-start,
                    stats=dict(frames=frames,candidate_rows=candidates,selected_rows=selected,optimizer_steps=0),
                    counter_delta=dict(frames=frames,candidates=candidates,selected=selected,source_selected=source_count,optimizer_steps=0)))
            records.append(receipt)
            pred=store.prediction.read_text().splitlines()
            audit_episodes.append(dict(video=video,expression=expression,frames=receipt["stats"]["frames"],
                candidates=receipt["stats"]["candidate_rows"],selected_rows=len(pred),
                selected_ids=len({r.split(",")[1] for r in pred}),wall_seconds=receipt["wall_seconds"]))
    if len(records)!=425 or len({(e["video"],e["expression_id"]) for e in records})!=425:
        raise ValueError("count-matched requires425 unique complete expressions")
    audit=dict(status="PASS",method=METHOD,protocol="POST_HOC_DIAGNOSTIC_NOT_DEPLOYABLE_TTA",valid_episodes=425,unique_videos=25,
        duplicated_episodes=0,missing_episodes=0,target_gt_opened=False,tracker_geometry_changed=False,tracker_ids_changed=False,
        per_frame_counts_equal_actual_identity=True,episodes=audit_episodes,
        **{k:sum(e[k] for e in audit_episodes) for k in ("frames","candidates","selected_rows","selected_ids")})
    if audit["selected_rows"]!=identity_audit["selected_rows"]:raise ValueError("total output counts differ from Identity")
    atomic_json(method_root/"full_audit.json",audit)
    atomic_json(method_root/"run_metadata.json",dict(status="complete",method=METHOD,protocol="POST_HOC_DIAGNOSTIC",
        episodes=records,sequences=425,videos=sorted({e["video"] for e in records}),gpu_inference_calls=0,
        optimizer_steps=0,wall_seconds=time.perf_counter()-started,file_access_audit=access))
    print(json.dumps({k:v for k,v in audit.items() if k!="episodes"},indent=2))


def evaluate(output=OUT,run_trackeval=True):
    from scripts.analyze_pf_full_offline import analyze
    from scripts.analyze_pf_video_uncertainty import paired_clusters
    output=Path(output);protocol_pass(output)
    root=output/"countmatched";folder=root/METHOD
    if json.loads((folder/"full_audit.json").read_text())["valid_episodes"]!=425:raise ValueError("seal all predictions first")
    if run_trackeval:
        with (root/"evaluation_stdout.txt").open("w") as log:
            subprocess.run([sys.executable,str(ROOT/"scripts/evaluate_flexhook_dance_tta.py"),"--output-root",str(root),
                "--dataset-root",str(DATA),"--vendor-root",str(ROOT/"vendor/FlexHook"),"--methods",METHOD],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        a=analyze(root,METHOD,FROZEN,DATA,TRACKS,root/"offline_attribution.json")
    else:
        a=json.loads((root/"offline_attribution.json").read_text())
    paths={"ZERO_UPDATE":ROOT/"results/positive_free/full/zero_update","COUNT_MATCHED_FROZEN":folder/"results", "PF_IDENTITY":ROOT/"results/positive_free/full/pf_identity"}
    metrics={m:official_metrics(p) for m,p in paths.items()}
    vals={m:official_video_values(p/"pedestrian_detailed.csv") for m,p in paths.items()}
    counts=np.random.default_rng(0).multinomial(25,np.ones(25)/25,size=2000)
    comparisons={}
    for x,y in ((METHOD,"ZERO_UPDATE"),("PF_IDENTITY",METHOD)):
        if vals[x][0]!=vals[y][0]:raise ValueError("video groups differ")
        d=float(paired_clusters(vals[x][1],vals[y][1],np.ones(25)))
        comparisons[x+"_vs_"+y]=dict(delta_hota=d,paired_25_video_CI95=np.percentile(paired_clusters(vals[x][1],vals[y][1],counts),[2.5,97.5]).tolist())
    identity_gain=metrics["PF_IDENTITY"]["HOTA"]-metrics["ZERO_UPDATE"]["HOTA"]
    count_gain=metrics[METHOD]["HOTA"]-metrics["ZERO_UPDATE"]["HOTA"]
    shared=changed=count_total=0
    for prediction in sorted((folder/"results").glob("*/*/predict.txt")):
        video,expression=prediction.parts[-3:-1]
        def selections(path):
            out={}
            for line in path.read_text().splitlines():
                f=line.split(",");out.setdefault(int(float(f[0])),set()).add(int(float(f[1])))
            return out
        cm=selections(prediction);im=selections(IDENTITY/"PF_IDENTITY/results"/video/expression/"predict.txt")
        for t in set(cm)|set(im):
            c,i=cm.get(t,set()),im.get(t,set())
            if len(c)!=len(i):raise ValueError("count match failed during output-set comparison")
            shared+=len(c&i);count_total+=len(c);changed+=int(c!=i)
    result=dict(protocol="POST_HOC_DIAGNOSTIC_NOT_DEPLOYABLE_TTA",episodes=425,videos=25,metrics=metrics,
        identity_gain=identity_gain,count_conditioned_gain=count_gain,identity_minus_countmatched=metrics["PF_IDENTITY"]["HOTA"]-metrics[METHOD]["HOTA"],
        descriptive_gain_ratio=count_gain/identity_gain,comparisons=comparisons,
        ratio_limit="not a causal percentage or fixed-threshold result; k is conditioned on the actual adapted run, HOTA is nonlinear",gpu_calls=0,
        selection_agreement=dict(selected_rows_per_method=count_total,shared_selected_rows=shared,
            shared_fraction=shared/count_total,changed_frames=changed,exchanged_rows_each_direction=count_total-shared),
        attribution={k:v for k,v in a.items() if k!="episodes"})
    atomic_json(root/"comparison.json",result)
    public=root/"official";public.mkdir(exist_ok=True)
    for name in ("pedestrian_summary.txt","pedestrian_detailed.csv"):
        shutil.copy2(folder/"results"/name,public/name)
    shutil.copy2(folder/"trackeval.log",public/"trackeval_stdout.txt")
    audit=json.loads((folder/"full_audit.json").read_text())
    atomic_json(root/"full_audit.json",audit)
    metadata=json.loads((folder/"run_metadata.json").read_text())
    atomic_json(root/"episode_receipts.json",dict(protocol=metadata["protocol"],raw_root=str(folder),
        wall_seconds=metadata["wall_seconds"],file_access_audit=metadata["file_access_audit"],
        episodes=[{k:e[k] for k in ("video","expression_id","prediction_path","prediction_sha256","margins_path","margins_sha256","stats")} for e in metadata["episodes"]]))
    write_csv(root/"official_comparison.csv",[dict(method=m,**{k:v[k] for k in ("HOTA","DetA","AssA","DetRe","IDF1","CLR_TP","CLR_FP","Dets")}) for m,v in metrics.items()])
    print(json.dumps({k:v for k,v in result.items() if k!="attribution"},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("predict","evaluate","summarize"),required=True)
    p.add_argument("--output-root",type=Path,default=OUT)
    a=p.parse_args()
    if a.stage=="predict":generate(a.output_root.resolve())
    else:evaluate(a.output_root.resolve(),run_trackeval=a.stage=="evaluate")
