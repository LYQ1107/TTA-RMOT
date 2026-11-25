#!/usr/bin/env python3
"""Offline protocol audit against original expression/label/GT assets; no rewrites."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import zipfile
import zlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.dance_io import dance_videos,load_expressions,load_tracker_lines
from flex_tta.runtime.pf_receipts import atomic_json

DATA=Path("/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance")
TRACKS=Path("/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs")
OUT=ROOT/"results/diagnostics/pf_recoverability_20261025"


def normalized_label_xywh(box,width,height):
    """Official labels.json stores normalized TOP-LEFT xywh, not center xywh."""
    return tuple(v*s for v,s in zip(box,(width,height,width,height)))


def audit(dataset=DATA,tracker=TRACKS,output=OUT):
    from PIL import Image
    dataset,tracker,output=map(Path,(dataset,tracker,output))
    labels=json.loads((dataset/"labels.json").read_text())
    videos=dance_videos(tracker)
    archive=Path("/data2/user/TTA-RMOT-assets/flexhook/Refer-Dance.zip")
    issues=[]
    episodes=[]
    archive_checks=0
    marks,classes=Counter(),Counter()
    with zipfile.ZipFile(archive) as original:
        names={n.removeprefix("Refer-Dance/"):info for n,info in ((i.filename,i) for i in original.infolist())}
        def check_archive(path):
            nonlocal archive_checks
            relative=path.relative_to(dataset).as_posix()
            info=names.get(relative)
            if info is None:
                issues.append(dict(kind="not_in_original_archive",path=relative));return
            content=path.read_bytes()
            if len(content)!=info.file_size or zlib.crc32(content)&0xffffffff!=info.CRC:
                issues.append(dict(kind="differs_from_original_archive",path=relative))
            archive_checks+=1
        check_archive(dataset/"labels.json")
        check_archive(dataset/"seqmap.txt")
        expected_seq=set()
        for video in videos:
            image_root=dataset/"DanceTrack/training/image_02"/video
            frames=sorted(int(p.stem) for p in image_root.glob("*.png"))
            if frames!=list(range(1,len(frames)+1)):
                issues.append(dict(kind="non_contiguous_one_based_images",video=video))
            with Image.open(image_root/"000001.png") as im:
                width,height=im.size
            tracks=load_tracker_lines(tracker,video)
            if min(tracks)<1 or max(tracks)>len(frames):
                issues.append(dict(kind="tracker_frame_outside_images",video=video))
            for expression,sentence in load_expressions(dataset,video):
                expected_seq.add(video+"+"+expression)
                ep=dataset/"expression"/video/(expression+".json")
                gp=dataset/"gt_template"/video/expression/"gt.txt"
                check_archive(ep);check_archive(gp)
                payload=json.loads(ep.read_text())
                expected={}
                for tid,observations in labels[video].items():
                    for frame,a in observations.items():
                        if sentence not in a["expression_raw"]:continue
                        expected[(int(frame),int(tid))]=normalized_label_xywh(a["bbox"],width,height)
                actual={}
                raw_rows=0
                coordinate_error=0.
                for line in gp.read_text().splitlines():
                    f=line.split(",");raw_rows+=1
                    t,i,mark,cls=int(float(f[0])),int(float(f[1])),int(float(f[6])),int(float(f[7]))
                    marks[mark]+=1;classes[cls]+=1
                    if not 1<=t<=len(frames):issues.append(dict(kind="gt_frame_outside_images",video=video,expression=expression,frame=t))
                    if not mark or cls!=1:continue
                    key=(t,i)
                    if key in actual:issues.append(dict(kind="duplicate_gt_key",video=video,expression=expression,key=key))
                    actual[key]=tuple(map(float,f[2:6]))
                    if i not in payload["label"].get(str(t),[]):
                        issues.append(dict(kind="gt_id_not_in_expression_label",video=video,expression=expression,key=key))
                missing=set(expected)-set(actual);extra=set(actual)-set(expected)
                if missing or extra:
                    issues.append(dict(kind="labels_gt_membership_mismatch",video=video,expression=expression,missing=len(missing),extra=len(extra)))
                for key in set(expected)&set(actual):
                    coordinate_error=max(coordinate_error,max(abs(a-b) for a,b in zip(expected[key],actual[key])))
                # Six-decimal normalized annotations imply <=2e-6*dimension pixel rounding.
                if coordinate_error>2e-6*max(width,height)+1e-6:
                    issues.append(dict(kind="labels_gt_bbox_mismatch",video=video,expression=expression,max_error=coordinate_error))
                history=ROOT/"results/runs/pf_zero_completebank_full_20261025_merged/ZERO_UPDATE/results"/video/expression/"gt.txt"
                if history.resolve()!=gp.resolve():
                    issues.append(dict(kind="historical_gt_symlink_mismatch",video=video,expression=expression))
                episodes.append(dict(video=video,expression=expression,sentence=sentence,raw_gt_rows=raw_rows,
                    scored_gt_rows=len(actual),labels_expected_rows=len(expected),query_present=bool(actual),
                    max_bbox_quantization_error=coordinate_error,image_frames=len(frames),
                    tracker_first_frame=min(tracks),tracker_last_frame=max(tracks),frame_offset=0))
        for path in (dataset/"seqmap.txt",ROOT/"vendor/FlexHook/seqmaps/dance.txt"):
            actual=set(path.read_text().splitlines())
            if actual!=expected_seq:issues.append(dict(kind="seqmap_mismatch",path=str(path),missing=len(expected_seq-actual),extra=len(actual-expected_seq)))
    result=dict(status="PASS" if not issues else "PROTOCOL_DISCREPANCY_PRESERVE_HISTORY",videos=len(videos),episodes=len(episodes),
        query_present=sum(e["query_present"] for e in episodes),query_empty=sum(not e["query_present"] for e in episodes),
        gt_observations=sum(e["scored_gt_rows"] for e in episodes),original_archive=str(archive),
        archive_metadata_crc_verified_files=archive_checks,gt_marks=dict(marks),gt_classes=dict(classes),
        labels_bbox_format="normalized top-left xywh; not center xywh",
        frame_convention="images/tracker/GT all one-based, no offset; exact frame-ID + GT-ID membership and normalized boxes checked",
        filtering="official evaluator: nonzero mark, pedestrian class1; distractor preprocessing exists but actual GT classes audited",
        source_links=["https://github.com/dyhBUPT/iKUN","https://github.com/buptLwz/FlexHook/blob/bd1acc38634b28525d54dc6e0fcb38335f0029f9/datasets/DATASET.md"],
        official_vendor_commit="bd1acc38634b28525d54dc6e0fcb38335f0029f9",history_modified=False,issues=issues,per_episode=episodes)
    atomic_json(output/"protocol_audit.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="per_episode"},indent=2))
    if issues:raise RuntimeError("report protocol impact before any evaluation-protocol changes")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root",type=Path,default=DATA)
    p.add_argument("--tracker-root",type=Path,default=TRACKS)
    p.add_argument("--output-root",type=Path,default=OUT)
    a=p.parse_args();audit(a.dataset_root,a.tracker_root,a.output_root)
