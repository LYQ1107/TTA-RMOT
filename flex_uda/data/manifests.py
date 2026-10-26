"""Create text/image/predicted-track inputs without target membership labels."""
import ast
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[2]
ASSETS=Path('/data2/user/TTA-RMOT-assets/flexhook')
DANCE=ASSETS/'refer-dance-official/Refer-Dance'
SOURCE=Path('/data3/user/TTA-RMOT-assets/flexhook/refer-kitti-v2')
VAL_TRACKS=ASSETS/'runs/dance_detr_neuralsort_official/tracker_outputs'
TRAIN_TRACKS=ASSETS/'runs/dance_train_detr_neuralsort_official/tracker_outputs'


def official_splits():
    tree=ast.parse((ROOT/'vendor/FlexHook/data/utils.py').read_text())
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='VIDEOS' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('official VIDEOS split not found')


def write_jsonl(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')


def read_jsonl(path):
    with Path(path).open() as stream:return [json.loads(line) for line in stream if line.strip()]


def extract_training_images(archive,root=DANCE):
    """Read only PNG image payloads from the already-downloaded official ZIP."""
    train=set(official_splits()['dance']['train']);counts={v:dict(extracted=0,reused=0) for v in sorted(train)}
    prefix='Refer-Dance/DanceTrack/training/image_02/'
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            if not item.filename.startswith(prefix) or not item.filename.endswith('.png'):continue
            relative=Path(item.filename[len('Refer-Dance/'):]);video=relative.parts[-2]
            if video not in train:continue
            target=Path(root)/relative
            if target.exists():
                if target.stat().st_size!=item.file_size:raise ValueError(f'existing image differs: {target}')
                counts[video]['reused']+=1;continue
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_suffix('.png.uda-part')
            # ZipFile validates CRC on read. No target labels/GT are extracted.
            with source.open(item) as src,temporary.open('wb') as dst:shutil.copyfileobj(src,dst)
            temporary.replace(target);counts[video]['extracted']+=1
    if any(not sum(c.values()) for c in counts.values()):raise ValueError('archive lacks some official training videos')
    return dict(archive=str(archive),only_target_training_images=True,target_annotation_payloads_read=0,videos=counts)


def build_target_manifests(directory,root=DANCE,train_tracks=TRAIN_TRACKS,val_tracks=VAL_TRACKS):
    split=official_splits()['dance'];train=set(split['train']);val=set(split['val'])
    if len(train)!=40 or len(val)!=25 or train&val or set(split['test'])!=val:raise ValueError('official Dance split differs')
    rows={};inventory=[]
    for group,videos,tracker in (('train',train,train_tracks),('val',val,val_tracks)):
        current=[]
        for video in sorted(videos):
            images=Path(root)/'DanceTrack/training/image_02'/video
            frames=sorted(int(p.stem) for p in images.glob('*.png'))
            if not frames or frames!=list(range(frames[0],frames[-1]+1)):raise ValueError(f'missing/noncontiguous images: {video}')
            expressions=sorted((Path(root)/'expression'/video).glob('*.json'))
            if not expressions:raise ValueError(f'no input expressions: {video}')
            for path in expressions:
                # Trusted preparation strips every field except input text.
                # Neither membership nor presence is inspected or exported.
                sentence=json.loads(path.read_text())['sentence']
                if not isinstance(sentence,str) or not sentence.strip():raise ValueError(f'invalid input expression: {path}')
                current.append(dict(video_id=video,expression_id=path.stem,query_text=sentence,
                    image_root=str(images),frame_indices=dict(start=frames[0],stop=frames[-1]+1,step=1),
                    frame_count=len(frames),predicted_tracks=str(Path(tracker)/(video+'.txt'))))
            track=Path(tracker)/(video+'.txt')
            inventory.append(dict(split=group,video_id=video,images=len(frames),expressions=len(expressions),
                predicted_tracks=str(track),predicted_tracks_present=track.is_file()))
        rows[group]=current
    if len(rows['val'])!=425:raise ValueError('Val must contain all425 actual expressions')
    directory=Path(directory)
    write_jsonl(directory/'dance_train_uda.jsonl',rows['train'])
    write_jsonl(directory/'dance_val_runtime.jsonl',rows['val'])
    write_jsonl(directory/'dance_val_evaluation.jsonl',[dict(video_id=r['video_id'],expression_id=r['expression_id'],
        gt_path=str(Path(root)/'gt_template'/r['video_id']/r['expression_id']/'gt.txt')) for r in rows['val']])
    return dict(train_videos=40,val_videos=25,train_val_video_overlap=[],train_episodes=len(rows['train']),
        val_episodes=len(rows['val']),inventory=inventory,
        target_runtime_fields=sorted(rows['train'][0]),target_membership_fields_exported=False,
        ready_for_formal_training=all(r['predicted_tracks_present'] for r in inventory))
