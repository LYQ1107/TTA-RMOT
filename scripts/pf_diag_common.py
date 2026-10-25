"""Shared sealed-score I/O for offline diagnostics, never an online TTA input."""
from contextlib import ExitStack
import csv
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from flex_tta.dance_io import load_expressions,load_tracker_lines

OUT=ROOT/"results/diagnostics/pf_recoverability_20261025"
DATA=Path("/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance")
TRACKS=Path("/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs")
FROZEN=ROOT/"results/runs/pf_zero_completebank_full_20261025_merged"
IDENTITY=ROOT/"results/runs/pf_identity_completebank_full_20261025_merged"


def protocol_pass(output=OUT):
    audit=json.loads((Path(output)/"protocol_audit.json").read_text())
    if audit["status"]!="PASS" or audit["episodes"]!=425:
        raise ValueError("resolve protocol audit before diagnostic evaluation")
    return audit


def load_source_video(video,dataset=DATA,frozen=FROZEN):
    """[fixed MOT observations,17 real queries], including track final rows."""
    expressions=load_expressions(Path(dataset),video)
    frames=[];ids=[];scores=[];slices=[]
    with ExitStack() as stack:
        streams=[stack.enter_context(gzip.open(Path(frozen)/"ZERO_UPDATE/results"/video/e/"candidate_margins.jsonl.gz","rt")) for e,_ in expressions]
        for lines in zip(*streams,strict=True):
            rows=[json.loads(line) for line in lines]
            t=rows[0]["frame"];order=rows[0]["ids"]
            if any(r["frame"]!=t or r["ids"]!=order for r in rows):
                raise ValueError("the17 real-query score streams are not candidate aligned")
            begin=len(ids)
            frames.extend([t]*len(order));ids.extend(order)
            scores.extend(np.asarray([r["adapted"] for r in rows],dtype=np.float32).T)
            slices.append((t,begin,len(ids)))
    return dict(video=video,expressions=expressions,frame=np.asarray(frames),ids=np.asarray(ids),
                scores=np.asarray(scores,dtype=np.float64),slices=slices)


def candidate_labels(bank,dataset=DATA,tracker=TRACKS):
    """Independent offline-only IoU>=.5 labels. Not used to generate predictions."""
    from scripts.analyze_pf_full_offline import load_eval_gt,overlaps
    from scripts.diagnose_flexhook_v2_dance import load_tracker_boxes
    boxes=load_tracker_boxes(tracker,bank["video"])
    labels=np.zeros(bank["scores"].shape,dtype=bool)
    counts=[]
    for q,(expression,_) in enumerate(bank["expressions"]):
        gt=load_eval_gt(dataset,bank["video"],expression)
        counts.append(sum(len(v) for v in gt.values()))
        for t,a,b in bank["slices"]:
            if t not in gt:continue
            matrix=overlaps([boxes[t][int(i)] for i in bank["ids"][a:b]],[box for _,box in gt[t]])
            labels[a:b,q]=(matrix>=.5).any(axis=1)
    return labels,counts


def official_metrics(folder):
    headers,values=(Path(folder)/"pedestrian_summary.txt").read_text().splitlines()[:2]
    return dict(zip(headers.split(),map(float,values.split())))


def official_video_values(csv_path):
    from scripts.analyze_pf_video_uncertainty import unpack,sufficient
    from trackeval.metrics import HOTA
    with Path(csv_path).open() as stream:
        rows=list(csv.DictReader(stream))
    hota=HOTA({"PRINT_CONFIG":False})
    by={r["seq"]:unpack(r,hota) for r in rows if r["seq"]!="COMBINED"}
    videos=sorted({s.split("+")[0] for s in by})
    if len(by)!=425 or len(videos)!=25:raise ValueError("official diagnostic must cover25/425")
    pooled=[hota.combine_sequences({k:r for k,r in by.items() if k.startswith(v+"+")}) for v in videos]
    return videos,sufficient(pooled)


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
