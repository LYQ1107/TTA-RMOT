"""Legal own-source training data only; never imported by target exporters."""
import ast
from collections import defaultdict
import json
from pathlib import Path
import zipfile
from flex_uda.presence.runtime import ROOT, RUN
from flex_tta.runtime.pf_receipts import file_sha

SPORTS_ARCHIVE = Path('/data2/user/TTA-RMOT-assets/flexhook/lamot_20261101/lamot_github/LaMOT-main.zip')
SPORTS_IMAGES = Path('/data2/user/LocateMOT_FullDataV2_spill/datasets/SportsMOT/train')


def splits():
    tree = ast.parse((ROOT / 'vendor/FlexHook/data/utils.py').read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'VIDEOS' for t in n.targets))


def source_banks(target):
    """One pure numerical GT bank and literal query list per training video.

    Keep all GT candidates, including ones with no referring descriptions.
    Their absence for a query is a genuine zero rather than an omitted unit.
    """
    if target == 'SportsMOT':
        return sports_banks()
    dataset = 'kitti-2' if target == 'KITTI-V2' else 'dance'
    root = ROOT / 'vendor/FlexHook/datasets' / ('refer-kitti-v2' if target == 'KITTI-V2' else 'refer-dance')
    labels = json.loads((root / 'labels.json').read_text())
    result = {}
    for video in sorted(splits()[dataset]['train']):
        image_root = root / ('KITTI/training/image_02' if target == 'KITTI-V2' else 'DanceTrack/training/image_02') / video
        files = sorted(image_root.glob('*.png'))
        if not files:
            raise FileNotFoundError(f'No source images: {image_root}')
        from PIL import Image
        with Image.open(files[0]) as im: width, height = im.size
        by_frame, membership = defaultdict(dict), {}
        for track, frames in labels[video].items():
            for frame, item in frames.items():
                t, i = int(frame), int(track)
                x, y, w, h = item['bbox']
                by_frame[t][i] = [x*width, y*height, w*width, h*height]
                membership[t, i] = set(item.get('expression_raw', []))
        queries = [dict(expression_id=p.stem, query_text=json.loads(p.read_text())['sentence'])
                   for p in sorted((root / 'expression' / video).glob('*.json'))]
        result[video] = make_bank(video, image_root, files, by_frame, queries, membership, target)
    return result


def make_bank(video, image_root, files, by_frame, queries, membership, target):
    observations, lines = [], []
    for t, objects in sorted(by_frame.items()):
        for i, box in sorted(objects.items()):
            observations.append([t, i])
            lines.append(','.join(map(str, [t, i, *box, 1, -1, -1, -1])))
    return dict(video_id=video, image_root=str(image_root.resolve()), observations=observations,
        lines=lines, image_frames=[int(p.stem) for p in files], queries=queries,
        membership=membership, target=target)


def sports_banks():
    query_map = defaultdict(list)
    membership = defaultdict(lambda: defaultdict(set))
    # Existing, previously downloaded official GitHub archive contains Train
    # as well as Val. No new download or modification of the sealed Val view.
    with zipfile.ZipFile(SPORTS_ARCHIVE) as archive:
        names = sorted(n for n in archive.namelist() if '/annotations_v1/train/SportsMOT/' in n and n.endswith('.json'))
        for name in names:
            payload = json.loads(archive.read(name))
            video = Path(name).stem.rsplit('_', 1)[0]
            query_map[video].append(dict(expression_id=Path(name).stem, query_text=payload['language']))
            for frame, tracks in payload['targets'].items():
                for i in tracks:
                    membership[video][int(frame), int(i)].add(payload['language'])
    result = {}
    for video, queries in sorted(query_map.items()):
        base = SPORTS_IMAGES / video
        files = sorted(p for p in (base / 'img1').glob('*.jpg') if p.stem.isdigit())
        if not files or not (base / 'gt/gt.txt').is_file():
            raise FileNotFoundError(f'Incomplete source SportsMOT train: {base}')
        by_frame = defaultdict(dict)
        for line in (base / 'gt/gt.txt').read_text().splitlines():
            fields = list(map(float, line.split(',')))
            by_frame[int(fields[0])][int(fields[1])] = fields[2:6]
        observed = {(t, i) for t, rows in by_frame.items() for i in rows}
        missing = set(membership[video]) - observed
        if missing:
            raise ValueError(f'Official Sports source referring IDs have no boxes: {video}, {sorted(missing)[:3]}')
        result[video] = make_bank(video, base / 'img1', files, by_frame, queries, membership[video], 'SportsMOT')
    return result


def counts_and_storage(banks, target):
    stride = 3 if target == 'KITTI-V2' else 10
    rows = units = 0
    counts = []
    for video, bank in banks.items():
        frames = bank['image_frames'][::stride]
        sizes = defaultdict(int)
        for t, _ in bank['observations']: sizes[t] += 1
        nq = len(bank['queries']); n = sum(sizes[t] for t in frames)
        units += len(frames)*nq; rows += n*nq
        counts.append(dict(video=video, images=len(bank['image_frames']), queries=nq,
                           units=len(frames)*nq, feature_rows=n*nq))
    return dict(target=target, videos=len(banks), queries=sum(len(b['queries']) for b in banks.values()),
        units=units, feature_rows=rows, pcd_bytes=rows*768*2, estimated_bytes=rows*(768*2+4+12+1)+units*24,
        stride=stride, per_video=counts)
