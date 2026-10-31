"""Train-only preparation; raw referring objects are never deserialized.

This module has no model/GT-evaluator imports. Preparation runs separately from
teacher/student workers; those workers consume only its stripped runtime files.
"""
from collections import Counter
import configparser
import json
from pathlib import Path

from .manifests import DANCE, official_splits, write_jsonl
from flex_tta.runtime.pf_receipts import atomic_json, file_sha

ORIGINAL = Path('/data2/user/LocateMOT/data/DanceTrack/train')
ASSETS = Path('/data2/user/TTA-RMOT-assets/train2val/20261031')


def sentence_only(path):
    """Decode only top-level keys and sentence, lexically skip all other values.

    The trusted text-export process necessarily opens the mixed JSON container,
    but never interprets label/frame/target-ID values or exports them to runtime.
    """
    text = Path(path).read_text()
    decoder = json.JSONDecoder()
    i = text.index('{') + 1
    sentence = None
    schema = {}
    while i < len(text):
        while text[i] in ' \t\n\r,':
            i += 1
        if text[i] == '}':
            break
        key, i = decoder.raw_decode(text, i)
        while text[i].isspace():
            i += 1
        if text[i] != ':':
            raise ValueError('invalid expression JSON separator')
        i += 1
        while text[i].isspace():
            i += 1
        schema[key] = {'[': 'array', '{': 'object', '"': 'string'}.get(text[i], 'scalar')
        if key == 'sentence':
            sentence, i = decoder.raw_decode(text, i)
            continue
        depth = 0
        quoted = escaped = False
        while i < len(text):
            char = text[i]
            if quoted:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in '[{':
                depth += 1
            elif char in ']}':
                if depth == 0:
                    break
                depth -= 1
            elif char == ',' and depth == 0:
                break
            i += 1
    if not isinstance(sentence, str) or not sentence.strip():
        raise ValueError('missing sentence text: ' + str(path))
    return sentence, schema


def geometry_rows(path):
    """Read only the six authorized MOT fields; ignore mark/class/visibility."""
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                fields = line.strip().split(',', 6)[:6]
                if len(fields) != 6:
                    raise ValueError('incomplete MOT geometry')
                yield fields


def inventory(root=DANCE, original=ORIGINAL):
    root, original = Path(root), Path(original)
    splits = official_splits()['dance']
    videos = []
    sentences, schemas = {}, Counter()
    for video in sorted(splits['train']):
        directory = original / video
        image_dir = root / 'DanceTrack/training/image_02' / video
        png_ids = {int(p.stem) for p in image_dir.glob('*.png')}
        jpg_ids = {int(p.stem) for p in (directory / 'img1').glob('*.jpg')}
        cfg = configparser.ConfigParser()
        cfg.read(directory / 'seqinfo.ini')
        expected = cfg.getint('Sequence', 'seqLength', fallback=0)
        gt = directory / 'gt/gt.txt'
        count = 0
        frames, tracks = set(), set()
        invalid = 0
        if gt.is_file():
            for fields in geometry_rows(gt):
                frame, track = map(int, fields[:2])
                x, y, w, h = map(float, fields[2:])
                frames.add(frame)
                tracks.add(track)
                count += 1
                invalid += int(frame not in png_ids or frame not in jpg_ids or w <= 0 or h <= 0)
        expressions = sorted((root / 'expression' / video).glob('*.json'))
        for path in expressions:
            sentence, schema = sentence_only(path)
            sentences.setdefault(path.stem, set()).add(sentence)
            schemas[json.dumps(schema, sort_keys=True)] += 1
        videos.append(dict(video_id=video, image_root=str(image_dir), original_image_root=str(directory / 'img1'),
            original_gt_path=str(gt), resolved_original_gt_path=str(gt.resolve()), gt_present=gt.is_file(),
            original_gt_sha256=file_sha(gt) if gt.is_file() else None,
            expected_frames=expected, png_frames=len(png_ids), original_jpg_frames=len(jpg_ids),
            missing_png_frames=len(set(range(1, expected + 1)) - png_ids),
            missing_original_frames=len(set(range(1, expected + 1)) - jpg_ids),
            gt_frames=len(frames), gt_rows=count, tracks=len(tracks), invalid_geometry_or_frame=invalid,
            expression_files=len(expressions)))
    # Val is metadata-only: filenames and seqmap, no Val JSON or GT contents.
    seqmap = Path(__file__).resolve().parents[2] / 'vendor/FlexHook/seqmaps/dance.txt'
    pairs = [line.split('+', 1) for line in seqmap.read_text().splitlines() if '+' in line]
    own = all((root / 'expression' / video / (query + '.json')).is_file() for video, query in pairs)
    val_stems = {query for _, query in pairs}
    overlap = sorted(set(sentences) & val_stems)
    complete = (len(videos) == 40 and all(r['gt_present'] and r['expected_frames'] > 0 and
        r['png_frames'] == r['original_jpg_frames'] == r['expected_frames'] and
        not r['missing_png_frames'] and not r['missing_original_frames'] and
        not r['invalid_geometry_or_frame'] and r['expression_files'] > 0 for r in videos))
    return dict(status='PASS_P0' if complete else 'STOP_P0_INCOMPLETE_ASSETS',
        dataset_root=str(root), original_train_root=str(original), videos=videos,
        train_videos=len(videos), train_frames=sum(r['png_frames'] for r in videos),
        candidate_box_rows=sum(r['gt_rows'] for r in videos), candidate_tracks=sum(r['tracks'] for r in videos),
        train_video_query_pairs=sum(r['expression_files'] for r in videos), train_unique_texts=len(set().union(*sentences.values())),
        expression_schema=dict(schemas), train_video_val_video_overlap=sorted(set(splits['train']) & set(splits['val'])),
        val_seqmap_sha256=file_sha(seqmap), val_pairs=len(pairs), val_videos=len({v for v, q in pairs}),
        val_unique_query_stems=len(val_stems), all_val_pairs_have_own_video_expression=own,
        pairing_rule='OWN_VIDEO_EXPRESSION_FILES_NO_CROSS_VIDEO_TEXT_LOOKUP',
        shared_train_val_expression_stems=overlap, shared_stem_count=len(overlap),
        val_expression_contents_opened=0, referring_labels_deserialized=0,
        text_export_process_opens_mixed_train_expression_containers=True,
        note='8 shared filenames establish lexical overlap; no claim that all Val expressions are unseen.')


def convert_geometry(source, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('preserve existing candidates: ' + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w') as handle:
        for fields in geometry_rows(source):
            handle.write(','.join(fields + ['1', '-1', '-1', '-1']) + '\n')
    # Numeric and textual first-six-field identity, not approximate box matching.
    actual = [line.split(',')[:6] for line in destination.read_text().splitlines()]
    if actual != list(geometry_rows(source)):
        raise RuntimeError('candidate conversion changed frame/ID/box/order')
    return dict(path=str(destination), sha256=file_sha(destination), rows=len(actual), first_six_columns_exact=True)


def prepare(p0, output=ASSETS):
    if p0['status'] != 'PASS_P0':
        raise RuntimeError('P0 missing-asset stop condition')
    output = Path(output)
    receipt = output / 'candidate_receipt.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        for row in saved['videos']:
            if file_sha(row['path']) != row['sha256']:
                raise ValueError('frozen Train candidates changed')
        return saved
    runtime, records = [], []
    root = Path(p0['dataset_root'])
    for video in p0['videos']:
        vid = video['video_id']
        target = output / 'tracker_outputs' / (vid + '.txt')
        record = convert_geometry(video['original_gt_path'], target)
        records.append(dict(video_id=vid, tracks=video['tracks'], source=video['original_gt_path'], **record))
        for expression in sorted((root / 'expression' / vid).glob('*.json')):
            text, _ = sentence_only(expression)
            runtime.append(dict(video_id=vid, expression_id=expression.stem, query_text=text,
                image_root=video['image_root'], frame_count=video['png_frames'],
                frame_indices=dict(start=1, stop=video['expected_frames'] + 1, step=1),
                predicted_tracks=str(target), candidate_provenance='ORIGINAL_DANCETRACK_TRAIN_GT_GEOMETRY_ID_ONLY'))
    write_jsonl(output / 'runtime_train.jsonl', runtime)
    result = dict(status='SEALED_TRAIN_GEOMETRY_ONLY_CANDIDATES', videos=records,
        video_query_pairs=len(runtime), candidate_rows=p0['candidate_box_rows'], tracks=p0['candidate_tracks'],
        candidate_query_combinations=sum(r['gt_rows'] * r['expression_files'] for r in p0['videos']),
        runtime_path=str(output / 'runtime_train.jsonl'), runtime_sha256=file_sha(output / 'runtime_train.jsonl'),
        referring_labels_used=False, visibility_mark_class_used=False, interpolation_or_filtering=False,
        train_GT_geometry_is_not_predicted_NeuralSORT=True,
        limitation='Train uses oracle geometry/physical IDs by user protocol; Val retains predicted DETR+NeuralSORT.')
    atomic_json(receipt, result)
    return result
