"""Fixed candidate sampling from legal sentences, predicted MOT and Frozen scores."""
from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import shutil

from flex_tta.runtime.dance_episode import DanceEpisode
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from .semantic_parser import parse_expression

ROOT=Path(__file__).resolve().parents[2]
FROZEN=ROOT/'results/runs/pf_zero_completebank_full_20261025_merged/ZERO_UPDATE'


def select_candidates(ids,margins,past_lengths,seed,key,limit=4):
    if not ids:return []
    selected=[];reasons={}
    def add(track,reason):
        if track not in reasons:selected.append(track);reasons[track]=[]
        reasons[track].append(reason)
    add(min(ids,key=lambda i:(-margins[i],i)),'HIGHEST_FROZEN_MARGIN')
    add(min(ids,key=lambda i:(abs(margins[i]),i)),'CLOSEST_TO_ZERO')
    remaining=sorted(set(ids)-set(selected))
    rng=random.Random(int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()[:16],16))
    if remaining:add(rng.choice(remaining),'DETERMINISTIC_RANDOM')
    remaining=sorted(set(ids)-set(selected))
    while remaining and len(selected)<limit:
        add(min(remaining,key=lambda i:(-past_lengths[i],i)),'LONGEST_CAUSAL_TRACK')
        remaining=sorted(set(ids)-set(selected))
    return [(i,reasons[i]) for i in selected[:limit]]


def build(runtime_manifest,output,stage,quantiles,seed=17,limit=4):
    output,stage=Path(output).resolve(),Path(stage).resolve()
    if output.exists():raise FileExistsError('preserve fixed manifest; use a new output for a changed protocol')
    rows=[json.loads(s) for s in Path(runtime_manifest).read_text().splitlines()]
    if len(rows)!=425 or len({r['video_id'] for r in rows})!=25:raise ValueError('require all25/425 legal runtime inputs')
    output.parent.mkdir(parents=True,exist_ok=True);stage.mkdir(parents=True,exist_ok=True)
    videos=defaultdict(list)
    for r in rows:videos[r['video_id']].append(r)
    samples=[];clips={};image_sources={};used_images=set();parser_counts=Counter();zero=[]
    for video,expressions in sorted(videos.items()):
        first=expressions[0];source_images=Path(first['image_root']);dataset=source_images.parents[3]
        ep=DanceEpisode(dataset,Path(first['predicted_tracks']).parent,video)
        anchors=sorted({ep.frames[round((len(ep.frames)-1)*q)] for q in quantiles})
        image_sources[video]=source_images
        track_dest=stage/'tracker_outputs'/f'{video}.txt';track_dest.parent.mkdir(parents=True,exist_ok=True)
        if not track_dest.exists():shutil.copy2(first['predicted_tracks'],track_dest)
        elif file_sha(track_dest)!=file_sha(first['predicted_tracks']):raise ValueError('staged tracker differs')
        used_images.add((video,ep.frames[0]))
        for anchor in anchors:
            index=ep.frames.index(anchor);clip=ep.frames[max(0,index-12):index+1];clips[f'{video}/{anchor}']=clip
            for t in clip:
                used_images.add((video,t))
                for _,history in ep.current_candidates(t):
                    used_images.update((video,o.frame_id) for o in history)
        for r in sorted(expressions,key=lambda x:x['expression_id']):
            parsed=parse_expression(r['query_text']);parser_counts[parsed.kind]+=1
            stream=FROZEN/'results'/video/r['expression_id']/'candidate_margins.jsonl.gz'
            scores={}
            with gzip.open(stream,'rt') as handle:
                for line in handle:
                    item=json.loads(line)
                    if item['frame'] in anchors:scores[item['frame']]=dict(zip(item['ids'],item['adapted']))
            for t in anchors:
                ids=list(ep.observations_by_frame[t]);margin=scores[t]
                if set(ids)!=set(margin):raise ValueError('Frozen candidate bank is not aligned to immutable MOT')
                chosen=select_candidates(ids,margin,{i:ep.track_index[i][t]+1 for i in ids},seed,f'{video}/{r["expression_id"]}/{t}',limit)
                if not chosen:
                    zero.append(dict(video=video,query_id=r['expression_id'],frame=t,reason='NO_PREDICTED_CANDIDATE'))
                for track,reasons in chosen:
                    obs=ep.observations_by_frame[t][track];history=ep.history(track,t)
                    key=f'{video}/{r["expression_id"]}/{t}/{track}'
                    samples.append(dict(sample_id=hashlib.sha256(key.encode()).hexdigest()[:24],video_id=video,
                        query_id=r['expression_id'],query_text=r['query_text'],frame_id=t,track_id=track,
                        bbox_xyxy=list(obs.bbox_xyxy),raw_mot_line=obs.raw_line,frozen_margin=margin[track],
                        selection_reasons=reasons,history=[asdict(o) for o in history],
                        expression_type=parsed.kind,attributes=[asdict(a) for a in parsed.attributes],parser_notes=list(parsed.notes)))
        print(json.dumps(dict(video=video,anchors=anchors,samples_so_far=len(samples))),flush=True)
    for video,t in sorted(used_images):
        source=image_sources[video]/f'{t:06d}.png'
        target=stage/'DanceTrack/training/image_02'/video/source.name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            try:os.link(source,target)
            except OSError:shutil.copy2(source,target)
    with output.open('w') as handle:
        for sample in samples:handle.write(json.dumps(sample,sort_keys=True)+'\n')
    staged=stage/output.name;shutil.copy2(output,staged);staged.chmod(0o444)
    atomic_json(stage/(output.stem+'_clips.json'),clips)
    receipt=dict(protocol='DEVELOPMENT_TEACHER_PROBE',target_GT_read=False,manifest=str(output),manifest_sha256=file_sha(output),
        stage=str(stage),samples=len(samples),videos=len(videos),video_queries=len(rows),anchor_quantiles=quantiles,
        expected_upper_bound=len(rows)*len(quantiles)*limit,zero_candidate_queries=zero,images=len(used_images),
        parser_counts=dict(parser_counts),seed=seed,frame_offset=0,source_runtime_manifest_sha256=file_sha(runtime_manifest),
        selection_is_GT_blind=True,history_is_causal=True,readonly_staging_on_completion=True)
    atomic_json(output.with_suffix('.receipt.json'),receipt)
    # Do not chmod hard-linked images: their original dataset permissions must be preserved.
    # Runtime kernel policy grants stage READ only and its own output READ/WRITE.
    for folder in sorted((p for p in stage.rglob('*') if p.is_dir()),reverse=True):folder.chmod(0o555)
    stage.chmod(0o555)
    return receipt
