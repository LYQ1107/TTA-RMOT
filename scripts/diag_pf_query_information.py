#!/usr/bin/env python3
"""Offline all425 real-query score diagnostics. No new GPU inference/thresholds.

Episode aggregates and centering use full sealed videos: diagnostic only, not
causal online decision rules. Other real queries are not assumed negative;
all-other, GT-disjoint and nonempty-GT-disjoint swaps are reported separately.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import rankdata
from scripts.pf_diag_common import OUT,TRACKS,load_source_video,candidate_labels,protocol_pass,write_csv
from scripts.analyze_pf_full_offline import auroc
from flex_tta.dance_io import dance_videos
from flex_tta.runtime.pf_receipts import atomic_json


def average_precision(scores,labels):
    scores=np.asarray(scores);y=np.asarray(labels,dtype=bool)
    if not y.any():return None
    order=np.argsort(-scores,kind="stable");s=scores[order];y=y[order]
    end=np.r_[np.flatnonzero(np.diff(s)),len(s)-1]
    tp=np.cumsum(y)[end];recall=tp/y.sum();precision=tp/(end+1)
    return float(np.sum(np.diff(np.r_[0,recall])*precision))


def binary_summary(scores,labels):
    labels=np.asarray(labels,dtype=bool)
    return dict(n=len(labels),positives=int(labels.sum()),prevalence=float(labels.mean()),
                AUROC=auroc(scores,labels),AUPRC=average_precision(scores,labels))


def decompose(scores):
    mu=scores.mean();visual=scores.mean(axis=1,keepdims=True);query=scores.mean(axis=0,keepdims=True)
    interaction=scores-visual-query+mu
    total=float(np.square(scores-mu).sum())
    ss=dict(candidate_visual=float(scores.shape[1]*np.square(visual-mu).sum()),
            query_marginal=float(scores.shape[0]*np.square(query-mu).sum()),interaction=float(np.square(interaction).sum()))
    return visual,query,interaction,dict(total_ss=total,**ss)


def rank_vector(values):
    return rankdata(values,method="average")


def frame_metrics(scores,positive):
    n,q=scores.shape
    ranks=rankdata(scores,method="average",axis=0)
    order=np.argsort(-scores,axis=0,kind="stable")
    visual=scores.mean(axis=1)
    vranks=rank_vector(visual);vtop=int(np.argmax(visual))
    result=[]
    for query in range(q):
        y=positive[:,query];p=int(y.sum())
        if not p:
            result.append(None);continue
        own=int(y[order[0,query]]);wrong=np.flatnonzero(np.arange(q)!=query)
        disjoint=np.asarray([j for j in wrong if not np.any(y&positive[:,j])],dtype=int)
        hard=np.asarray([j for j in disjoint if positive[:,j].any()],dtype=int)
        row=dict(valid_frames=1,mixed_frames=int(p<n),correct_top1=float(own),random_top1=p/n,
                 visual_top1=float(y[vtop]),correct_recall_at1=float(own/p),
                 correct_positive_rank_percentile=float(((ranks[y,query]-1)/max(n-1,1)).mean()) if n>1 else .5,
                 all_other_top1=float(y[order[0,wrong]].mean()),
                 all_other_positive_rank_percentile=float(((ranks[y][:,wrong]-1)/max(n-1,1)).mean()) if n>1 else .5,
                 disjoint_frames=int(bool(len(disjoint))),hard_disjoint_frames=int(bool(len(hard))))
        for name,others in (("disjoint",disjoint),("hard_disjoint",hard)):
            row[name+"_top1"]=float(y[order[0,others]].mean()) if len(others) else 0.
            row[name+"_correct_top1"]=float(own) if len(others) else 0.
            row[name+"_random_top1"]=p/n if len(others) else 0.
        if p<n:
            def auc_from_ranks(r):return float((r[y].sum()-p*(p+1)/2)/(p*(n-p)))
            row.update(mixed_correct_top1=float(own),mixed_random_top1=p/n,mixed_visual_top1=float(y[vtop]),
                mixed_source_AUROC=auc_from_ranks(ranks[:,query]),mixed_visual_AUROC=auc_from_ranks(vranks),
                mixed_other_AUROC=float(np.mean([auc_from_ranks(ranks[:,j]) for j in wrong])))
        result.append(row)
    centered=ranks-ranks.mean(axis=0,keepdims=True)
    norms=np.linalg.norm(centered,axis=0)
    valid=np.triu((norms[:,None]>0)&(norms[None,:]>0),1)
    corr=(centered.T@centered)/np.maximum(norms[:,None]*norms[None,:],1e-12)
    counts=np.asarray(list(Counter(order[0]).values()))
    sensitivity=dict(query_pair_spearman_sum=float(corr[valid].sum()),valid_query_pairs=int(valid.sum()),
                     top1_disagreement=1-float(np.sum(counts*(counts-1))/(q*(q-1))))
    return result,sensitivity


def add(total,row):
    if row is not None:
        for k,v in row.items():total[k]=total.get(k,0)+v


def finalize(total):
    n=total.get("valid_frames",0);mixed=total.get("mixed_frames",0)
    out={k:total.get(k,0) for k in ("valid_frames","mixed_frames","disjoint_frames","hard_disjoint_frames")}
    for k,v in total.items():
        if k in out:continue
        denominator=(mixed if k.startswith("mixed_") else total.get("hard_disjoint_frames",0) if k.startswith("hard_disjoint_")
                     else total.get("disjoint_frames",0) if k.startswith("disjoint_") else n)
        out[k]=v/denominator if denominator else None
    return out


def analyze(output=OUT):
    output=Path(output);protocol=protocol_pass(output);started=time.perf_counter()
    episodes=[];videos=[];totals={};all_scores=[];all_labels=[];visual_scores=[];query_scores=[];interaction_scores=[]
    raw_video_records=[];swap_rows=[];query_variations=[]
    for video in dance_videos(TRACKS):
        bank=load_source_video(video);scores=bank["scores"];labels,gt_counts=candidate_labels(bank)
        visual,query,interaction,ss=decompose(scores)
        all_scores.append(scores.ravel());all_labels.append(labels.ravel())
        visual_scores.append(np.broadcast_to(visual,scores.shape).ravel())
        query_scores.append(np.broadcast_to(query,scores.shape).ravel());interaction_scores.append(interaction.ravel())
        query_variations.extend(scores.std(axis=1).tolist())
        per_query=[{} for _ in range(17)];sense=dict(query_pair_spearman_sum=0.,valid_query_pairs=0,top1_disagreement=0.,frames=0)
        maxes=[];imax=[]
        for t,a,b in bank["slices"]:
            rows,sensitivity=frame_metrics(scores[a:b],labels[a:b])
            for j,row in enumerate(rows):add(per_query[j],row)
            for k,v in sensitivity.items():sense[k]+=v
            sense["frames"]+=1
            maxes.append(scores[a:b].max(axis=0));imax.append(interaction[a:b].max(axis=0))
        for j,(expression,sentence) in enumerate(bank["expressions"]):
            summary=finalize(per_query[j]);add(totals,per_query[j])
            item=dict(video=video,expression=expression,sentence=sentence,query_present=bool(gt_counts[j]),gt_observations=gt_counts[j],
                candidate_observations=len(scores),mean_margin=float(scores[:,j].mean()),p90_margin=float(np.quantile(scores[:,j],.9)),
                max_margin=float(scores[:,j].max()),mean_frame_max_margin=float(np.mean(np.asarray(maxes)[:,j])),
                positive_fraction=float(np.mean(scores[:,j]>0)),visual_mean_margin=float(scores.mean()),
                interaction_mean_frame_max=float(np.asarray(imax)[:,j].mean()))
            episodes.append(item)
            swap_rows.append(dict(video=video,expression=expression,query_present=item["query_present"],**summary))
        record=dict(video=video,observations=len(scores),frames=len(bank["slices"]),score_decomposition=ss,
            query_pair_mean_spearman=sense["query_pair_spearman_sum"]/max(sense["valid_query_pairs"],1),
            mean_top1_cross_query_disagreement=sense["top1_disagreement"]/sense["frames"],
            valid_query_pairs=sense["valid_query_pairs"],source_candidate=binary_summary(scores.ravel(),labels.ravel()),
            visual_bias_candidate=binary_summary(np.broadcast_to(visual,scores.shape).ravel(),labels.ravel()),
            query_bias_candidate=binary_summary(np.broadcast_to(query,scores.shape).ravel(),labels.ravel()))
        videos.append(record);raw_video_records.append({e:row for (e,_),row in zip(bank["expressions"],per_query)})
        print(json.dumps(dict(event="source_video_analyzed",video=video,observations=len(scores))),flush=True)
    # A query-score prior from OTHER videos: no GT and no held-out video scores.
    mean_by_query={q:np.mean([e["mean_margin"] for e in episodes if e["expression"]==q]) for q in {e["expression"] for e in episodes}}
    for e in episodes:e["leave_video_out_query_score_bias"]=(25*mean_by_query[e["expression"]]-e["mean_margin"])/24
    metrics=("mean_margin","p90_margin","max_margin","mean_frame_max_margin","positive_fraction", "visual_mean_margin","interaction_mean_frame_max","leave_video_out_query_score_bias")
    y=np.asarray([e["query_present"] for e in episodes]);video_ids=sorted({e["video"] for e in episodes})
    group=[np.asarray([i for i,e in enumerate(episodes) if e["video"]==v]) for v in video_ids]
    rng=np.random.default_rng(0);draws=rng.integers(0,25,size=(2000,25))
    presence={};per_video=[];per_expression=[]
    for name in metrics:
        values=np.asarray([e[name] for e in episodes]);r=binary_summary(values,y)
        aucs=[];aps=[]
        for draw in draws:
            ix=np.concatenate([group[j] for j in draw]);a=auroc(values[ix],y[ix]);p=average_precision(values[ix],y[ix])
            if a is not None:aucs.append(a)
            if p is not None:aps.append(p)
        r.update(video_cluster_AUROC_CI95=np.percentile(aucs,[2.5,97.5]).tolist(),video_cluster_AUPRC_CI95=np.percentile(aps,[2.5,97.5]).tolist())
        presence[name]=r
        for video,ix in zip(video_ids,group):per_video.append(dict(video=video,statistic=name,**binary_summary(values[ix],y[ix])))
        for expression in sorted(mean_by_query):
            ix=np.asarray([i for i,e in enumerate(episodes) if e["expression"]==expression])
            per_expression.append(dict(expression=expression,statistic=name,**binary_summary(values[ix],y[ix])))
    target=np.concatenate(all_labels)
    candidate={n:binary_summary(np.concatenate(xs),target) for n,xs in
               (("source",all_scores),("query_independent_visual_bias",visual_scores),("query_marginal_bias",query_scores),("double_centered_interaction",interaction_scores))}
    historic=json.loads((ROOT/"results/positive_free/full/zero_update/offline_attribution.json").read_text())
    if not np.isclose(candidate["source"]["AUROC"],historic["source_auroc"],atol=1e-12):raise ValueError("full source score/GT labels no longer reproduce sealed AUROC")
    ss={k:sum(v["score_decomposition"][k] for v in videos) for k in videos[0]["score_decomposition"]}
    variations=np.asarray(query_variations)
    query_swap=finalize(totals)
    # Paired VIDEO resampling for within-frame correct-vs-other text discrimination.
    effects={}
    for first,second in (("correct_top1","all_other_top1"),("mixed_source_AUROC","mixed_other_AUROC"),("mixed_source_AUROC","mixed_visual_AUROC"),
                         ("hard_disjoint_correct_top1","hard_disjoint_top1")):
        per=[]
        for records in raw_video_records:
            total={}
            for row in records.values():add(total,row)
            per.append(total)
        den="mixed_frames" if first.startswith("mixed_") else "hard_disjoint_frames" if first.startswith("hard_disjoint_") else "valid_frames"
        def delta(ids):
            denominator=sum(per[i].get(den,0) for i in ids)
            return sum(per[i].get(first,0)-per[i].get(second,0) for i in ids)/denominator if denominator else np.nan
        boot=np.asarray([delta(draw) for draw in draws]);valid=boot[np.isfinite(boot)]
        point=delta(range(25))
        effects[first+"_minus_"+second]=dict(delta=float(point) if np.isfinite(point) else None,
            video_cluster_CI95=np.percentile(valid,[2.5,97.5]).tolist() if len(valid) else None,
            valid_video_bootstrap_resamples=len(valid))
    result=dict(status="COMPLETE_25_VIDEOS_425_EXPRESSIONS",gpu_calls=0,optimizer_steps=0,protocol="OFFLINE_POST_HOC_NO_THRESHOLD_SELECTION",
        episodes=425,videos=25,observations=len(target),query_present=int(y.sum()),query_empty=int((~y).sum()),
        presence=presence,presence_random_AUPRC=float(y.mean()),candidate=candidate,query_swap=query_swap,query_swap_effects=effects,
        within_video_variance_fractions={k:v/ss["total_ss"] for k,v in ss.items() if k!="total_ss"},
        candidate_cross_query_margin_std_quantiles={str(q):float(np.quantile(variations,q)) for q in (.1,.5,.9)},
        score_aggregate_limit="whole sealed episode aggregates/centering are offline; not causal online presence rules; fixed statistics, no GT-fitted threshold",
        visual_control_definition="row mean over17 real queries; query-independent score bias, NOT a separately trained visual model",
        swap_limit="other real query may describe same target; separate all-other, disjoint, and nonempty-disjoint GT candidate sets; GT only offline",
        rank_limit="within-frame AUROC only mixed positive/negative frames; no claim from dense all-target frames or query-only prior",
        wall_seconds=time.perf_counter()-started,per_video=videos)
    destination=output/"query_information"
    atomic_json(destination/"summary.json",result)
    write_csv(destination/"episode_presence_scores.csv",episodes)
    # Some queries have no positive GT, so their ranking fields are intentionally empty.
    columns=list(dict.fromkeys(k for r in swap_rows for k in r))
    write_csv(destination/"per_episode_text_sensitivity.csv",[{k:r.get(k) for k in columns} for r in swap_rows])
    write_csv(destination/"per_video_presence.csv",per_video)
    write_csv(destination/"per_expression_presence.csv",per_expression)
    atomic_json(destination/"per_expression_text_sensitivity.json",{
        expression:finalize({k:sum(records[expression].get(k,0) for records in raw_video_records)
                             for k in totals}) for expression in sorted(mean_by_query)})
    print(json.dumps({k:v for k,v in result.items() if k!="per_video"},indent=2))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-root",type=Path,default=OUT)
    a=p.parse_args();analyze(a.output_root.resolve())
