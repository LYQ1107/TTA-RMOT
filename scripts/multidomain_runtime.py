"""Official-only generality experiment inputs/seals; no GT payload IO."""
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_tta.mot_interface.trackers import merge_kitti_class_splits
RUN=ROOT/'results/runs/multidomain_generality_20261101'
PUBLIC=ROOT/'results/multidomain_generality'
VENDOR=ROOT/'vendor/FlexHook'
OLD_SOURCE=ROOT/'results/runs/xquery_decode_20261101/source'
SOURCE=Path('/data3/user/TTA-RMOT-assets/flexhook/refer-kitti-v2')
TRACKS=VENDOR/'tracker_outputs/Temp-NeuralSORT-kitti2'
CHECKPOINTS={
 'V2-best':(Path('/data3/user/TTA-RMOT-assets/flexhook/SOTA_ckpts/refer-kitti-v2-best.pth'),'71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613'),
 'Dance-best':(Path('/data3/user/TTA-RMOT-assets/flexhook/SOTA_ckpts/refer-dance-best.pth'),'fbb5df7339a1201f34300c105df81e893a5edbb8ef939b7f6eaf8780348eac6c')}


def require_plan():
    p=json.loads((RUN/'prereg_amendment_publication.json').read_text())
    if not p['remote_verified'] or file_sha(PUBLIC/'PREREG.md')!=p['prereg_sha256']:
        raise ValueError('Publish amendment before downloads/inference')
    return p


def guard():
    attempts=[]
    runtime=(RUN/'runtime/kitti/expression').resolve()
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(runtime):return # independently exported sentence-only files
        if p.name in ('gt.txt','labels.json') or p.name.endswith('_labels.json') or any(k in p.parts for k in ('gt_template','gt_templates','expression','annotations_v1')):
            attempts.append(str(p));raise PermissionError('Generality inference/decoder must not open GT: '+str(p))
    sys.addaudithook(audit);return attempts


def prepare_kitti():
    require_plan();attempts=guard();old=json.loads((OLD_SOURCE/'metadata.json').read_text())
    if file_sha(OLD_SOURCE/'FROZEN.npy')!=old['frozen_array_sha256']:raise ValueError('Historical frozen array changed')
    videos={};episodes=[];offset=0;history_mask=[];history_indices=[];groups=[];group_offset=0
    runtime=RUN/'runtime/kitti';runtime.mkdir(parents=True,exist_ok=True)
    images=runtime/'KITTI/training/image_02';images.parent.mkdir(parents=True,exist_ok=True)
    if not images.exists():images.symlink_to(SOURCE/'KITTI/training/image_02',target_is_directory=True)
    text_files=[]
    for video,prior in old['videos'].items():
        rows=merge_kitti_class_splits(TRACKS/video/'car/predict.txt',TRACKS/video/'pedestrian/predict.txt')
        obs=[(r.frame_id-1,r.track_id) for r in rows]
        oldpos={tuple(o):j for j,o in enumerate(prior['observations'])}
        index=[oldpos.get(o,-1) for o in obs];mask=np.asarray(index)>=0
        queries=[q for q in old['episodes'] if q['video_id']==video]
        frames=np.asarray(obs)[:,0];unique,local=np.unique(frames,return_inverse=True)
        videos[video]=dict(observations=obs,lines=[','.join(map(str,r.values)) for r in rows],
            image_root=str(SOURCE/'KITTI/training/image_02'/video),historical_positions=index,
            tracker_files=[dict(path=str(TRACKS/video/c/'predict.txt'),sha256=file_sha(TRACKS/video/c/'predict.txt')) for c in ('car','pedestrian')],
            full_observations=len(obs),historical_observations=int(mask.sum()),physical_tracks=len({r.track_id for r in rows}))
        for q in queries:
            new=dict(video_id=video,expression_id=q['expression_id'],query_text=q['query_text'],
                start=offset,stop=offset+len(rows),historical_start=q['start'],historical_stop=q['stop'],
                frame_count=len(list((SOURCE/'KITTI/training/image_02'/video).glob('*.png'))))
            episodes.append(new)
            history_indices.extend(q['start']+i for i in index if i>=0)
            history_mask.extend(mask);groups.extend(local+group_offset);group_offset+=len(unique)
            offset+=len(rows)
            p=runtime/'expression'/video/(q['expression_id']+'.json')
            payload=dict(sentence=q['query_text'])
            if p.exists() and json.loads(p.read_text())!=payload:raise ValueError('Changed runtime sentence')
            if not p.exists():atomic_json(p,payload)
            text_files.append(dict(path=str(p),sha256=file_sha(p),allowed_fields=['sentence']))
    if offset!=2425382 or sum(history_mask)!=2355528 or offset-sum(history_mask)!=69854:
        raise ValueError('KITTI complete/history cardinalities differ from amendment')
    dest=RUN/'kitti';dest.mkdir(exist_ok=True)
    np.save(dest/'historical.npy',np.asarray(history_mask,dtype=bool))
    np.save(dest/'historical_indices.npy',np.asarray(history_indices,dtype=np.int64))
    np.save(dest/'groups.npy',np.asarray(groups,dtype=np.int32))
    meta=dict(episodes=episodes,videos=videos,candidate_query_rows=offset,group_count=group_offset,
        full_candidate_observations=sum(v['full_observations'] for v in videos.values()),
        historical_candidate_query_rows=2355528,missing_tail_candidate_query_rows=69854,
        frame_rule='MOT frame = image index + 1; retain original raw MOT rows',
        runtime_root=str(runtime),text_files=text_files,GT_open_attempts=len(attempts),
        old_metadata_sha256=file_sha(OLD_SOURCE/'metadata.json'),old_frozen_sha256=old['frozen_array_sha256'])
    atomic_json(dest/'metadata.json',meta)
    print(json.dumps(dict(event='KITTI_RUNTIME_PREPARED',queries=len(episodes),full=offset,missing=69854,GT_open_attempts=len(attempts))),flush=True)


if __name__=='__main__':prepare_kitti()
