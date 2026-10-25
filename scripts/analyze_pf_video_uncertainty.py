#!/usr/bin/env python3
"""Paired video-cluster diagnostics from sealed official results, never online GT."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor/FlexHook/TrackEval"))
from trackeval.metrics import HOTA, Identity
from flex_tta.runtime.pf_receipts import atomic_json

ORDER = ("ZERO_UPDATE", "RELATIVE_ONLY", "PF_RANK", "PF_RANK_GUARD", "PF_IDENTITY", "PF_HYBRID")


def unpack(row, metric):
    result = {f: float(row[f]) for f in metric.float_fields+metric.integer_fields}
    for f in metric.float_array_fields+metric.integer_array_fields:
        result[f] = np.asarray([float(row[f+"___"+str(int(100*x))]) for x in metric.array_labels])
    return result


def sufficient(video_stats):
    """Official combine_sequences: AssA is TP-weighted, never mean video HOTA."""
    numerator = np.stack([s["AssA"]*s["HOTA_TP"] for s in video_stats])
    denominator = np.stack([s["HOTA_TP"]+s["HOTA_FN"]+s["HOTA_FP"] for s in video_stats])
    return numerator, denominator


def pooled_hota(values, weights):
    num, den = values
    return 100*np.sqrt((weights@num)/np.maximum(weights@den, 1)).mean(axis=-1)


def paired_clusters(adapted, baseline, counts):
    return pooled_hota(adapted, counts)-pooled_hota(baseline, counts)


def load_method(folder):
    report = json.loads((folder/"metrics.json").read_text())
    if report["full_audit"]["status"] != "PASS" or report["full_audit"]["valid_episodes"] != 425:
        raise ValueError("only complete, sealed, officially evaluated methods can be analyzed")
    with (folder/"pedestrian_detailed.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    episodes = {r["seq"]:r for r in rows if r["seq"] != "COMBINED"}
    if len(episodes) != 425:
        raise ValueError("missing or duplicated official episodes")
    hota, identity = HOTA({"PRINT_CONFIG":False}), Identity({"PRINT_CONFIG":False})
    hrows = {key:unpack(row, hota) for key,row in episodes.items()}
    videos = sorted({key.split("+")[0] for key in episodes})
    if len(videos) != 25:
        raise ValueError("video-cluster analysis requires the entire 25-video benchmark")
    per_video = [hota.combine_sequences({key:r for key,r in hrows.items() if key.startswith(v+"+")}) for v in videos]
    values = sufficient(per_video)
    combined = next(r for r in rows if r["seq"] == "COMBINED")
    actual = float(pooled_hota(values, np.ones(25)))
    if not np.isclose(actual, 100*float(combined["HOTA___AUC"]), atol=1e-8, rtol=0):
        raise ValueError("cluster sufficient statistics fail exact official HOTA pooling")
    attribution = json.loads((folder/"offline_attribution.json").read_text())
    gt = {r["video"]+"+"+r["expression"]:r["gt_observations"] for r in attribution["episodes"]}
    groups = {}
    for name, present in (("query_present",True),("query_empty",False)):
        selected = {k:r for k,r in episodes.items() if bool(gt[k]) == present}
        h = hota.combine_sequences({k:hrows[k] for k in selected})
        ident = identity.combine_sequences({k:unpack(r,identity) for k,r in selected.items()})
        groups[name] = dict(episodes=len(selected), HOTA=100*float(np.mean(h["HOTA"])) if present else None,
                            DetRe=100*float(np.mean(h["DetRe"])) if present else None,
                            IDF1=100*float(ident["IDF1"]) if present else None,
                            official_CLEAR_FP=sum(int(float(r["CLR_FP"])) for r in selected.values()))
    return dict(report=report, videos=videos, values=values, groups=groups)


def analyze(publish, samples=2000, seed=0, require_six=False):
    publish = Path(publish)
    methods = {m:load_method(publish/m.lower()) for m in ORDER if (publish/m.lower()/"metrics.json").exists()}
    if require_six and len(methods) != 6:
        raise ValueError("final uncertainty analysis requires six complete methods")
    base = methods["ZERO_UPDATE"]
    if any(m["videos"] != base["videos"] for m in methods.values()):
        raise ValueError("methods use different video clusters")
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(25, np.ones(25)/25, size=samples)
    pairs = [(m,"ZERO_UPDATE") for m in methods if m != "ZERO_UPDATE"]
    if "PF_HYBRID" in methods:
        pairs += [("PF_HYBRID",m) for m in ("PF_RANK","PF_IDENTITY")]
    comparisons = {}
    for a,b in pairs:
        va,vb = methods[a]["values"], methods[b]["values"]
        delta = float(paired_clusters(va,vb,np.ones(25)))
        bootstrap = paired_clusters(va,vb,counts)
        individual = paired_clusters(va,vb,np.eye(25))
        leave_one_out = paired_clusters(va,vb,np.ones((25,25))-np.eye(25))
        video_rows = [dict(video=v, delta_hota=float(d), delta_hota_without_video=float(loo),
                           leave_one_out_influence=float(delta-loo))
                      for v,d,loo in zip(base["videos"],individual,leave_one_out)]
        ci = np.percentile(bootstrap,[2.5,97.5]).tolist()
        comparisons[a+"_vs_"+b] = dict(delta_hota=delta, paired_video_cluster_CI95=ci,
            CI_excludes_zero=bool(ci[0] > 0 or ci[1] < 0),
            improved_videos=int((individual > 1e-9).sum()), declined_videos=int((individual < -1e-9).sum()),
            tied_videos=int((np.abs(individual) <= 1e-9).sum()),
            most_influential_videos=sorted(video_rows,key=lambda r:abs(r["leave_one_out_influence"]),reverse=True)[:5],
            per_video=video_rows,
            query_present_delta_hota=methods[a]["groups"]["query_present"]["HOTA"]-methods[b]["groups"]["query_present"]["HOTA"])
    costs = {}
    for method,item in methods.items():
        report = item["report"]
        stats = report.get("reused_rank_stats", report["aggregate_stats"])
        opportunities = stats.get("update_opportunities",0)
        costs[method] = dict(**report["cost"], trainable_parameters_per_replica=stats.get("trainable_parameters",0),
            optimizer_updates=stats.get("optimizer_steps",0), update_statistics_reused_from_rank="reused_rank_stats" in report,
            update_opportunities=opportunities,
            effective_evidence_coverage=(opportunities-stats.get("skipped_no_evidence",0)-stats.get("skipped_multi_batch",0))/opportunities if opportunities else None,
            accepted_update_fraction=stats.get("optimizer_steps",0)/opportunities if opportunities else None)
    result = dict(status="COMPLETE_6_METHODS" if len(methods)==6 else "INTERIM", completed_methods=list(methods),
        cluster_unit="video; all 17 correlated expressions and frames stay together", clusters=25,
        bootstrap_samples=samples, bootstrap_seed=seed, interval="paired nonparametric percentile 95%",
        pooling="official TP-weighted HOTA sufficient statistics, checked against unrounded COMBINED CSV",
        caveat="descriptive development-set video uncertainty, not a blind-test significance claim or training-seed uncertainty; no independent-frame sample assumption",
        empty_group_hota_definition="N/A: no scored GT; compare FP and nonempty episode rate, not HOTA gain",
        query_group_official_metrics={m:r["groups"] for m,r in methods.items()}, compute_summary=costs, comparisons=comparisons)
    atomic_json(publish/"video_uncertainty.json",result)
    lines = ["# Video-cluster and query-group analysis", "",
        f"{result['status']}; {len(methods)}/6 methods. Paired bootstrap: 25 whole videos, {samples} resamples, seed {seed}.", "",
        "Descriptive development-set uncertainty, not frame-independent significance or training-seed uncertainty.", "",
        "| Comparison | ΔHOTA | Video-cluster 95% interval | Up / down / tie videos | Query-Present ΔHOTA |",
        "|---|---:|---:|---:|---:|"]
    for name,r in comparisons.items():
        low,high=r["paired_video_cluster_CI95"]
        lines.append(f"| {name} | {r['delta_hota']:+.6f} | [{low:+.6f}, {high:+.6f}] | {r['improved_videos']} / {r['declined_videos']} / {r['tied_videos']} | {r['query_present_delta_hota']:+.6f} |")
    lines += ["", "Query-Empty HOTA is not meaningful without GT; FP and nonempty prediction rates are reported separately.",
        "All 425 sequences remain in the primary evaluation. HOTA is pooled, not a mean of per-video HOTA.",
        "Per-video deltas and leave-one-video-out influence are in `results/positive_free/full/video_uncertainty.json`; HOTA deltas are not additive video contributions."]
    (ROOT/"docs/PF_VIDEO_GROUP_ANALYSIS_20261025.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(status=result["status"],comparisons={k:{f:v for f,v in r.items() if f not in {"per_video","most_influential_videos"}} for k,r in comparisons.items()}),indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--publish-root",type=Path,default=ROOT/"results/positive_free/full")
    p.add_argument("--samples",type=int,default=2000)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--require-six",action="store_true")
    args=p.parse_args()
    analyze(args.publish_root,args.samples,args.seed,args.require_six)


if __name__=="__main__": main()
