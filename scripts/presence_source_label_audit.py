"""Independent source-Train labels/cache check after the same-domain failure.

No target adaptation process imports this file. No GPU/model/teacher calls.
The earlier pre-fixed first-video audit is preserved, not overwritten.
"""
import json
from argparse import ArgumentParser
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.presence.source import source_banks, SPORTS_ARCHIVE
from flex_uda.presence.runtime import RUN, PUBLIC, IN_DOMAIN
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.diag_attrib_v2_common import write_csv


def expression_labels(target, video):
    if target != 'SportsMOT':
        name = 'refer-kitti-v2' if target == 'KITTI-V2' else 'refer-dance'
        directory = ROOT / 'vendor/FlexHook/datasets' / name / 'expression' / video
        return {p.stem:{(int(t),int(i)) for t, ids in json.loads(p.read_text())['label'].items() for i in ids}
                for p in sorted(directory.glob('*.json'))}
    result = {}
    with zipfile.ZipFile(SPORTS_ARCHIVE) as archive:
        for name in archive.namelist():
            if '/annotations_v1/train/SportsMOT/' not in name or not name.endswith('.json'): continue
            if Path(name).stem.rsplit('_',1)[0] != video: continue
            raw = json.loads(archive.read(name))
            result[Path(name).stem] = {(int(t),int(i)) for t, ids in raw['targets'].items() for i in ids}
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--sports-label-root',type=Path)
    parser.add_argument('--output-prefix',default='source_label_full_audit')
    args = parser.parse_args()
    rows, problems = [], []
    for target, model in IN_DOMAIN.items():
        for video, bank in source_banks(target).items():
            raw = expression_labels(target, video)
            root = RUN / 'features/source_train' / target / model / 'videos' / video
            metadata = json.loads((root / 'metadata.json').read_text())
            observations = [tuple(map(int,o)) for o in np.load(root / 'observations.npy')]
            label_path = (args.sports_label_root / 'videos' / video / 'labels.npy'
                          if target == 'SportsMOT' and args.sports_label_root else root / 'labels.npy')
            cached = np.load(label_path,mmap_mode='r').reshape(len(metadata['queries']),len(observations))
            all_observed = {tuple(o) for o in bank['observations']}
            positive_raw = positive_loader = 0
            mismatch_queries = symmetric_difference = cache_mismatches = 0
            positive_queries = 0
            for qi, query in enumerate(metadata['queries']):
                expression = query['expression_id']; text = query['query_text']
                expected = raw[expression] & all_observed
                key = query[bank.get('membership_query_key','query_text')]
                actual = {o for o in all_observed if key in bank['membership'].get(o,())}
                difference = expected ^ actual
                mismatch_queries += bool(difference); symmetric_difference += len(difference)
                positive_raw += len(expected); positive_loader += len(actual); positive_queries += bool(expected)
                sampled_expected = np.fromiter((o in expected for o in observations),dtype=bool,count=len(observations))
                mismatches = int((sampled_expected != cached[qi]).sum()); cache_mismatches += mismatches
                if difference or mismatches:
                    problems.append(dict(target=target,video=video,expression=expression,
                        loader_symmetric_difference=len(difference),cached_mismatching_observations=mismatches,
                        first_loader_differences=sorted(difference)[:5]))
            rows.append(dict(target=target,model=model,video=video,queries=len(metadata['queries']),
                positive_queries=positive_queries,positive_pairs_expression_label=positive_raw,
                positive_pairs_loader=positive_loader,mismatching_queries=mismatch_queries,
                symmetric_difference=symmetric_difference,cached_rows=cached.size,
                cached_mismatching_observations=cache_mismatches))
    write_csv(PUBLIC / (args.output_prefix+'.csv'),rows)
    result = dict(status='PASS_ALL_SOURCE_TRAIN_LABELS_AND_CACHES' if not problems else 'FAIL_SOURCE_LABELS_STOP_AND_INVESTIGATE',
        videos=len(rows),queries=sum(r['queries'] for r in rows),cached_rows=sum(r['cached_rows'] for r in rows),
        mismatching_queries=sum(r['mismatching_queries'] for r in rows),
        cached_mismatching_observations=sum(r['cached_mismatching_observations'] for r in rows),
        scope='legal own-source TRAIN only; independent literal expression targets vs loader text membership and sealed cache labels',
        earlier_first_video_audit_preserved=True,new_GPU_or_teacher_calls=0,problems=problems)
    atomic_json(PUBLIC / (args.output_prefix+'.json'),result)
    print(json.dumps(result,indent=2),flush=True)
    if problems: raise RuntimeError('Source label mismatch; preserve old results and investigate before any new method run')


if __name__ == '__main__': main()
