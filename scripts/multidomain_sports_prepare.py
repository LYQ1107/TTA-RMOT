"""Audit official Sports assets and independently export a text-only mix facade.

This CPU preparation process extracts only annotation['language']; no referring
membership/GT values are copied or used. Inference opens the facade, not the
original annotations or SportsMOT_labels.json. Vendor code remains unchanged.
"""
from collections import Counter, defaultdict
import configparser
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.multidomain_runtime import RUN, PUBLIC, VENDOR, require_plan
from scripts.multidomain_baidu_resume import DEST
from flex_tta.runtime.pf_receipts import atomic_json, file_sha

PACKAGE = DEST / 'FlexHook'
LAMOT = PACKAGE / 'datasets/LaMOT-main'
IMAGES = Path('/data2/user/LocateMOT_FullDataV2_spill/datasets/SportsMOT/val')
TRACKS = PACKAGE / 'tracker_outputs/McByte-SportsMOT'


def link_new(path, target):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() and not path.is_symlink():
        path.symlink_to(target, target_is_directory=target.is_dir())
    if path.resolve() != target.resolve():
        raise ValueError('Preserve conflicting layout link: ' + str(path))


def write_once(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Preserve conflicting runtime export: ' + str(path))
    else:
        atomic_json(path, value)


def main():
    require_plan()
    provenance = json.loads((PUBLIC / 'downloads.json').read_text())
    known = {x['path']: x for x in provenance['files'] if x.get('sha256')}
    checkpoint = PACKAGE / 'SOTA_ckpts/LaMOT-best.pth'
    required = [checkpoint, LAMOT/'sequences/val/SportsMOT_labels.json',
        LAMOT/'sequences/val/SportsMOT_RESOLUTION.json', LAMOT/'sequences/val/TAO_frame2file.json']
    for path in required:
        if not path.is_file() or str(path) not in known or file_sha(path) != known[str(path)]['sha256']:
            raise ValueError('Official downloaded asset missing/unverified: ' + str(path))
    seqmap = VENDOR / 'seqmaps/SportsMOT.txt'
    seqs = [x.strip() for x in seqmap.read_text().splitlines() if x.strip()]
    videos = sorted({x.split('+', 1)[0] for x in seqs})
    if len(seqs) != 466 or len(set(seqs)) != 466 or len(videos) != 45:
        raise ValueError('Official Sports seqmap cardinality mismatch')
    exported = RUN / 'runtime/sports'
    resolution = json.loads((LAMOT/'sequences/val/SportsMOT_RESOLUTION.json').read_text())
    originals = LAMOT / 'annotations_v1/val/SportsMOT'
    # The stock mix test parser only uses labels.keys(). Empty objects carry no GT.
    write_once(exported/'sequences/val/SportsMOT_labels.json', {v:{} for v in videos})
    write_once(exported/'sequences/val/SportsMOT_RESOLUTION.json', resolution)
    write_once(exported/'sequences/val/TAO_frame2file.json', {})  # unused for SportsMOT
    link_new(exported/'sequences/val/SportsMOT-sequences', IMAGES)
    mapping = {}
    source_records = []
    raw_github = DEST.parent.parent / 'lamot_github/extracted/LaMOT-main/annotations_v1/val/SportsMOT'
    for path in sorted(originals.glob('*.json')):
        if str(path) not in known or file_sha(path) != known[str(path)]['sha256']:
            raise ValueError('Unverified official expression file: ' + str(path))
        text = json.loads(path.read_text())['language']
        if not isinstance(text, str) or not text:
            raise ValueError('Official language is not a nonempty sentence')
        video = path.stem.rsplit('_',1)[0]
        key = video + '+' + text.replace(' ', '-')  # stock test_utils_mix.py:106
        if key in mapping:
            raise ValueError('Duplicate video/text annotation; do not silently deduplicate')
        mapping[key] = dict(video=video, expression_id=key.split('+',1)[1], query_text=text,
            annotation_file=path.name)
        runtime = exported / 'annotations_v1/val/SportsMOT' / path.name
        write_once(runtime, dict(language=text))
        github = raw_github / path.name
        source_records.append(dict(path=str(path), sha256=known[str(path)]['sha256'],
            runtime_text=str(runtime), runtime_text_sha256=file_sha(runtime),
            fields_exported=['language'], original_label_fields_used=False,
            github_equivalent=github.is_file() and file_sha(github)==known[str(path)]['sha256']))
    if set(mapping) != set(seqs):
        mismatch = dict(missing=sorted(set(seqs)-set(mapping)), extra=sorted(set(mapping)-set(seqs)))
        atomic_json(PUBLIC/'sports_annotation_mismatch.json', mismatch)
        raise ValueError('Official expression/seqmap mismatch; inspect sports_annotation_mismatch.json')
    banks = {}; episodes = []; groups = []; historical = []; offset = 0; group_offset = 0
    tracker_records = []
    for video in videos:
        path = TRACKS / (video+'.txt')
        if str(path) not in known or file_sha(path) != known[str(path)]['sha256']:
            raise ValueError('Missing/unverified official McByte file: ' + str(path))
        raw = [s.strip() for s in path.read_text().splitlines() if s.strip()]
        rows = [list(map(float, s.split(','))) for s in raw]
        if not rows or any(len(r)!=10 or not np.isfinite(r).all() or r[0]<1 or r[1]<0 or r[4]<=0 or r[5]<=0 for r in rows):
            raise ValueError('Invalid official MOT rows: ' + video)
        ordered = sorted(zip(rows,raw), key=lambda r:(int(r[0][0]),int(r[0][1])))
        obs = [(int(r[0]),int(r[1])) for r,s in ordered]
        if len(set(obs)) != len(obs):
            raise ValueError('Duplicate official (frame, ID) observation')
        image_root = IMAGES/video/'img1'
        images = sorted(int(p.stem) for p in image_root.glob('*.jpg') if p.stem.isdigit())
        ini = configparser.ConfigParser(); ini.read(IMAGES/video/'seqinfo.ini')
        nframes = ini.getint('Sequence','seqLength')
        if images != list(range(1,nframes+1)) or any(t>nframes for t,i in obs):
            raise ValueError('Missing/invalid SportsMOT image coverage: ' + video)
        with Image.open(image_root/f'{obs[0][0]:06d}.jpg') as image:
            if list(image.size) != resolution[video]:
                raise ValueError('Official resolution differs from actual image')
        tails = {i:t for t,i in obs}
        hist = np.array([t!=tails[i] for t,i in obs])
        frames = np.array(obs)[:,0]
        unique, starts, local = np.unique(frames, return_index=True, return_inverse=True)
        stops = np.r_[starts[1:], len(obs)]
        banks[video] = dict(observations=obs, lines=[s for r,s in ordered],
            image_root=str(image_root), image_extension='.jpg', width=resolution[video][0],
            height=resolution[video][1], frame_count=nframes, full_observations=len(obs),
            historical_observations=int(hist.sum()), physical_tracks=len(tails),
            tracker_path=str(path), tracker_sha256=file_sha(path))
        for key in (s for s in seqs if s.split('+',1)[0] == video):
            query = mapping[key]
            q = dict(video_id=video, expression_id=query['expression_id'], query_text=query['query_text'],
                frame_count=nframes, start=offset, stop=offset+len(obs),
                groups=[[int(t),offset+int(a),offset+int(b),group_offset+j]
                    for j,(t,a,b) in enumerate(zip(unique,starts,stops))])
            episodes.append(q); groups.extend(local+group_offset); historical.extend(hist)
            offset += len(obs); group_offset += len(unique)
        tracker_records.append(dict(video=video, path=str(path), sha256=file_sha(path),
            full_observations=len(obs), historical_observations=int(hist.sum()), tracks=len(tails), images=nframes))
    folder = RUN/'sports';folder.mkdir(exist_ok=True)
    np.save(folder/'groups.npy', np.array(groups,dtype=np.int32))
    np.save(folder/'historical.npy', np.array(historical,dtype=bool))
    meta = dict(episodes=episodes,videos=banks,candidate_query_rows=offset,group_count=group_offset,
        full_candidate_observations=sum(v['full_observations'] for v in banks.values()),
        historical_candidate_query_rows=int(sum(historical)),
        missing_tail_candidate_query_rows=int(len(historical)-sum(historical)),
        frame_rule='MOT/image both 1-based; no ID/box/frame edits; full last observation added',
        runtime_root=str(exported),text_files=source_records,
        category_tokens='Stock mix: class=0; no category token added; original language unchanged',
        tokenizer='Canonical RoBERTa vocab+merges, same as diag_attrib v2',
        referring_labels_copied=0,GT_labels_used=0)
    write_once(folder/'metadata.json', meta)
    # Official package layout for post-seal independent evaluation, separate from runtime facade.
    layout = DEST.parent / 'layout/LaMOT-main'
    for relative in ['annotations_v1/val/SportsMOT','sequences/val/SportsMOT_labels.json',
            'sequences/val/SportsMOT_RESOLUTION.json','sequences/val/TAO_frame2file.json', 'gt_templates/SportsMOT']:
        target = LAMOT/relative
        if target.exists(): link_new(layout/relative, target)
    link_new(VENDOR/'tracker_outputs/McByte-SportsMOT', TRACKS)
    assets = dict(status='CPU_ASSETS_VERIFIED_REQUIRES_STRICT_LOAD_AND_SMOKE',
        checkpoint=dict(path=str(checkpoint),sha256=file_sha(checkpoint),bytes=checkpoint.stat().st_size),
        official_annotated_episodes=466,official_videos=45,seqmap_sha256=file_sha(seqmap),
        raw_annotation_files=len(source_records),github_byte_identical=sum(x['github_equivalent'] for x in source_records),
        annotation_mismatches=[],tracker_records=tracker_records,
        full_candidate_observations=meta['full_candidate_observations'],full_candidate_query_rows=offset,
        images=sum(r['images'] for r in tracker_records),runtime_root=str(exported),
        runtime_metadata_sha256=file_sha(folder/'metadata.json'),
        GT_used_by_model=0,projection_process_only=True,vendor_code_changed=False)
    atomic_json(PUBLIC/'sports_assets.json',assets)
    print(json.dumps({k:v for k,v in assets.items() if k!='tracker_records'}), flush=True)


if __name__ == '__main__':
    main()
