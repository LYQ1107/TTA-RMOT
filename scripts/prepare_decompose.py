"""GT-free expression taxonomy, fixed paired probes and pre-call cost decision."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
import configparser
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decompose_common import *
from flex_uda.data.manifests import read_jsonl, write_jsonl


def prepare(publication):
    RUN.mkdir(parents=True, exist_ok=True)
    if file_sha(PUBLIC / 'PREREG.md') != PREREG_SHA:
        raise ValueError('Literal preregistration changed')
    proof = dict(remote_commit=publication, remote_verified=True, prereg_sha256=PREREG_SHA,
                 branch='research/decompose-rank-select-20261102', publication_verified_before_new_computation=True)
    atomic_json(RUN / 'prereg_publication.json', proof)
    require_plan()
    access = guard()
    datasets = {t: load_target(t) for t in TARGETS}
    mapping_path = OLD / 'source_query_categories.json'
    category = json.loads(mapping_path.read_text())['mapping']
    old_plan = json.loads((COUNT / 'probe_plan.json').read_text())
    receipt = json.loads((COUNT / 'probe/2B/receipt.json').read_text())
    if receipt['model_revision'] != REVISION or file_sha(receipt['outputs_path']) != receipt['outputs_sha256']:
        raise ValueError('Changed original Dance T1 outputs')
    if file_sha(old_plan['manifest_path']) != old_plan['manifest_sha256']:
        raise ValueError('Changed original Dance probes')
    for k, p in [('original_backend_sha256', 'yestrack_backend.py'), ('original_visual_builder_sha256', 'visual_input_builder.py')]:
        if file_sha(ROOT / 'flex_uda/teacher' / p) != receipt[k]:
            raise ValueError('Original T1 code changed')
    dance = read_jsonl(old_plan['manifest_path'])
    for r in dance:
        r['target'] = 'Dance'
    fps_config = json.loads((ROOT / 'configs/teacher/decompose_fps.json').read_text())
    fps, fps_sources = {}, {}
    image_roots = []
    taxonomy = []
    videos = []
    for target, (meta, arrays, groups, records) in datasets.items():
        for q in meta['episodes']:
            taxonomy.append(dict(target=target, video=q['video_id'], expression=q['expression_id'],
                                 query_text=q['query_text'], types='|'.join(types(q['query_text']))))
        by_video = video_queries(meta)
        for v, queries in by_video.items():
            bank = meta['videos'][v]
            if target == 'Dance':
                image_root = Path(queries[0]['image_root'])
            else:
                image_root = Path(bank['image_root'])
            image_roots.append(image_root)
            if target == 'SportsMOT':
                p = image_root.parent / 'seqinfo.ini'
                cfg = configparser.ConfigParser(); cfg.read(p)
                rate = float(cfg['Sequence']['frameRate'])
                origin = dict(path=str(p), sha256=file_sha(p), field='Sequence.frameRate')
            else:
                rate = fps_config[target]['fps']; origin = fps_config[target]
            delta = max(1, int(np.floor(rate * .25 + .5)))
            fps[target + '/' + v] = dict(fps=rate, delta=delta, source=origin)
            videos.append(dict(target=target, video_id=v, image_root=str(image_root),
                               image_offset=-1 if target == 'KITTI-V2' else 0,
                               extension='.jpg' if target == 'SportsMOT' else '.png',
                               observations=physical(meta, target, v), fps=rate, delta=delta))
    isolation = sandbox([*image_roots, receipt['outputs_path'], old_plan['manifest_path']], [RUN, PUBLIC])
    write_csv(PUBLIC / 'expression_types.csv', taxonomy)
    taxonomy_seal = dict(path=str(PUBLIC / 'expression_types.csv'), sha256=file_sha(PUBLIC / 'expression_types.csv'),
                         rows=len(taxonomy), before_any_new_evaluation=True, uses_only_text=True)
    by_key = {(r['target'], r['video_id']): r for r in videos}

    def probes(anchor_fractions):
        rows = []
        rng = np.random.RandomState(0)
        sampling = []
        for target in ('KITTI-V2', 'SportsMOT'):
            meta = datasets[target][0]
            by_video = video_queries(meta)
            selected_videos = sorted(by_video) if target == 'KITTI-V2' else sorted(by_video)[::3]
            for v in selected_videos:
                qs = by_video[v]
                if target == 'KITTI-V2' and len(qs) > 60:
                    strata = {c: [q for q in qs if category[v + '+' + q['expression_id']] == c] for c in ('car', 'pedestrian')}
                    car_n = int(np.floor(60 * len(strata['car']) / len(qs) + .5))
                    quota = dict(car=car_n, pedestrian=60-car_n)
                    selected = [stratum[i] for c, stratum in strata.items() for i in rng.choice(len(stratum), quota[c], replace=False)]
                    sampling.append(dict(video=v, available={c: len(s) for c, s in strata.items()}, quota=quota))
                    qs = sorted(selected, key=lambda q: q['expression_id'])
                video = by_key[target, v]
                frame_rows = defaultdict(list)
                for c in video['observations']:
                    frame_rows[c['frame_id']].append(c)
                available = sorted(frame_rows)
                count = by_video[v][0]['frame_count']
                for j, frac in enumerate(anchor_fractions):
                    wanted = int(np.floor(1 + frac * (count - 1) + .5))
                    t = min(available, key=lambda f: (abs(f-wanted), f))
                    p = Path(video['image_root']) / f"{t + video['image_offset']:06d}{video['extension']}"
                    rows.append(dict(target=target, video_id=v, frame_id=t, probe_frame_index=j,
                                     anchor_fraction=frac, wanted_frame=wanted, image_path=str(p), image_sha256=file_sha(p),
                                     candidates=sorted(frame_rows[t], key=lambda c: c['track_id']),
                                     queries=[dict(query_id=q['expression_id'], query_text=q['query_text']) for q in qs]))
        return rows + dance, sampling

    full, sampling = probes([.05, .17, .29, .41, .53, .65, .77, .89])
    budget_old = receipt['budgets']['T1_CROP_YESNO']
    seconds_per_pair = budget_old['forward_path_seconds'] / budget_old['new_logical_calls']
    # Pre-call token-work estimate: four crops for CLIP; full-frame vision token
    # area relative to the fixed 320-square crop. No GT or new teacher outputs.
    from PIL import Image
    def estimate(frames):
        cost = Counter()
        calls = Counter()
        for f in frames:
            n = len(f['queries']) * len(f['candidates'])
            with Image.open(f['image_path']) as im:
                scale = 1024 / max(im.size)
                area = round(im.width*scale) * round(im.height*scale)
            for method, multiplier in [('CROP', 1), ('CLIP', 4), ('FRAME_BOX', area/(320*320))]:
                calls[method] += n
                cost[method] += n * seconds_per_pair * multiplier
        return dict(logical_calls=dict(calls), estimated_forward_GPU_hours={k: v/3600 for k, v in cost.items()},
                    total_estimated_forward_GPU_hours=sum(cost.values())/3600)
    initial = estimate(full)
    fallback = initial['total_estimated_forward_GPU_hours'] > 6
    frames, sampling = probes([.05, .30, .55, .80]) if fallback else (full, sampling)
    final = estimate(frames)
    p = RUN / 'probe/manifest.jsonl'; p.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(p, frames)
    runtime = RUN / 'probe/videos.json'
    atomic_json(runtime, videos)
    records = [r for _, _, _, r in datasets.values() for r in r.values()]
    plan = dict(status='FIXED_PROBE_AND_TAXONOMY_SEALED_BEFORE_NEW_CALLS', prereg_publication=proof,
                expression_types=taxonomy_seal, margin_banks=records, manifest_path=str(p), manifest_sha256=file_sha(p),
                videos_path=str(runtime), videos_sha256=file_sha(runtime), fps=fps,
                stratified_sampling=sampling, numpy_seed=0, anchor_rounding='floor(x+.5); nearest active frame, ties earlier',
                dataset_counts={t: dict(videos=len({f['video_id'] for f in frames if f['target']==t}),
                    frames=sum(f['target']==t for f in frames), candidate_query_pairs=sum(len(f['queries'])*len(f['candidates']) for f in frames if f['target']==t)) for t in TARGETS},
                initial_budget=initial, final_budget=final, reduce_8_to_4=fallback,
                estimate_basis=dict(old_receipt=str(COUNT/'probe/2B/receipt.json'), seconds_per_CROP_pair=seconds_per_pair,
                    CLIP_multiplier=4, FRAME_BOX_multiplier='resized RGB area / 320**2; estimated visual-token work, not a measured new forward',
                    includes_reused_Dance_CROP_equivalent=True),
                dance_reused_CROP=dict(outputs_path=receipt['outputs_path'], outputs_sha256=receipt['outputs_sha256'],
                    original_receipt_path=str(COUNT/'probe/2B/receipt.json'), calls=6426),
                original_backend_sha256=receipt['original_backend_sha256'],
                original_visual_builder_sha256=receipt['original_visual_builder_sha256'],
                GT_open_attempts=access, isolation=isolation, student_updates=0, teacher_updates=0)
    atomic_json(RUN/'probe/plan.json', plan); atomic_json(PUBLIC/'probe_sets.json', plan)
    print(json.dumps(dict(event='DECOMPOSE_PROBES_SEALED', dataset_counts=plan['dataset_counts'],
                         initial_budget=initial, final_budget=final, reduce_8_to_4=fallback, GT_open_attempts=len(access))), flush=True)


if __name__ == '__main__':
    p = ArgumentParser(); p.add_argument('--publication', required=True); a = p.parse_args()
    prepare(a.publication)
