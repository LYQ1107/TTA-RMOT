#!/usr/bin/env python3
"""Offline GT-only developmental diagnostic, STRICTLY separated from TTA.

`--i-allow-offline-gt` is required. Reads only sealed source logits, tracker
rows and GT templates, never writes TTA state. Do not tune final test methods
from this result and claim it is an independently held-out confirmation.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from flex_tta.diagnostics.recoverability import RankedCandidate, rank_recoverability


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i-allow-offline-gt",action="store_true")
    p.add_argument("--source-result",type=Path,required=True)
    p.add_argument("--dataset-root",type=Path,required=True)
    p.add_argument("--tracker-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(argv)
    if not args.i_allow_offline_gt:
        p.error("This is a GT-based development diagnostic; add --i-allow-offline-gt")
    from flex_tta.dance_io import dance_videos,load_expressions,score_frames
    from scripts.diagnose_flexhook_v2_dance import load_gt,load_tracker_boxes,iou
    scores=json.loads(args.source_result.read_text(encoding="utf-8"))
    frames=[]
    sequences=0
    for video in dance_videos(args.tracker_root):
        bank=load_tracker_boxes(args.tracker_root,video)
        for expression_id,sentence in load_expressions(args.dataset_root,video):
            reference=load_gt(args.dataset_root,video,expression_id)
            margins=score_frames(scores,video,sentence)
            for frame_id, gt_rows in reference.items():
                frame_scores=margins.get(frame_id,{})
                rowset=bank.get(frame_id,{})
                frames.append([
                    RankedCandidate(tid,score,any(iou(box,gt_box)>=0.5 for _,gt_box in gt_rows))
                    for tid,score in frame_scores.items() for box in [rowset.get(tid)] if box is not None
                ])
            sequences+=1
    result=rank_recoverability(frames)
    result.update({"videos":25,"sequences":sequences,"protocol":"offline GT analysis, not target-TTA supervision"})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
    return 0

if __name__=="__main__":raise SystemExit(main())
