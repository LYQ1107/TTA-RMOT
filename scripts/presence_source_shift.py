"""Offline source-only diagnosis: GT-candidate train vs tracker validation.

Read exported own-source labels, not a target adaptation process. These
descriptive statistics never set model thresholds, classes or hyperparameters.
"""
import json
from argparse import ArgumentParser
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.presence.runtime import RUN, PUBLIC, IN_DOMAIN
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.diag_attrib_v2_common import write_csv


def support(root,labels_root=None):
    hist = np.zeros(11, dtype=np.int64)
    candidate_counts, positive_counts, phrase_set = [], [], set()
    raw_count_sum = overflow = units = videos = 0
    for directory in sorted((root / 'videos').iterdir()):
        metadata = json.loads((directory / 'metadata.json').read_text())
        frames = np.asarray(metadata['frames'])
        observations = np.load(directory / 'observations.npy')
        label_path = (Path(labels_root) / 'videos' / directory.name / 'labels.npy') if labels_root else directory / 'labels.npy'
        labels = np.load(label_path, mmap_mode='r').reshape(len(metadata['queries']), len(observations))
        unique, starts, counts = np.unique(observations[:, 0], return_index=True, return_counts=True)
        positive = np.zeros((len(labels), len(frames)), dtype=np.int64)
        candidates = np.zeros(len(frames), dtype=np.int64)
        if len(unique):
            positions = np.searchsorted(frames, unique)
            if not np.array_equal(frames[positions], unique): raise ValueError('Candidate frame outside source feature units')
            positive[:, positions] = np.add.reduceat(labels, starts, axis=1, dtype=np.int64)
            candidates[positions] = counts
        hist += np.bincount(positive.clip(max=10).ravel(), minlength=11)
        raw_count_sum += int(positive.sum()); overflow += int((positive > 10).sum())
        units += positive.size; videos += 1
        positive_counts.extend(positive[positive > 0].tolist())
        candidate_counts.extend(candidates.tolist())
        phrase_set.update(' '.join(q['query_text'].lower().split()) for q in metadata['queries'])
    positive_units = units-int(hist[0])
    return dict(videos=videos, units=units, positive_units=positive_units,
        positive_unit_fraction=positive_units/units, count_histogram_0_to_9_ge10=hist.tolist(),
        positive_candidate_count_mean=raw_count_sum/positive_units,
        positive_candidate_count_p50=float(np.median(positive_counts)),
        positive_candidate_count_p90=float(np.percentile(positive_counts,90)),
        positive_candidate_count_over10_units=overflow, candidate_count_mean=float(np.mean(candidate_counts)),
        candidate_count_p50=float(np.median(candidate_counts)), candidate_count_p90=float(np.percentile(candidate_counts,90)),
        unique_exact_phrases=len(phrase_set), export_receipt_sha256=file_sha(root / 'export_receipt.json'))


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--sports-label-root',type=Path)
    parser.add_argument('--output-prefix',default='source_support_shift')
    args=parser.parse_args()
    rows, detail = [], {}
    for target, model in IN_DOMAIN.items():
        for split, root in [('TRAIN_GT_CANDIDATES', RUN / 'features/source_train' / target / model),
                            ('SOURCE_VALIDATION_TRACKER', RUN / 'features' / target / model)]:
            corrected=args.sports_label_root if target=='SportsMOT' and split=='TRAIN_GT_CANDIDATES' else None
            result = support(root,labels_root=corrected)
            detail[target+'/'+split] = dict(model=model, source_domain=target, path=str(root), **result)
            rows.append(dict(source_domain=target, model=model, split=split,
                **{k:v for k,v in result.items() if k != 'count_histogram_0_to_9_ge10'}))
    write_csv(PUBLIC / (args.output_prefix+'.csv'), rows)
    atomic_json(PUBLIC / (args.output_prefix+'.json'), dict(ORACLE_GT_DIAGNOSTIC_ONLY=True,
        labels_scope='each head own-source train and own-source official validation only',
        no_new_model_or_teacher_calls=True, no_hyperparameters_selected=True, records=detail,
        corrected_sports_labels_root=str(args.sports_label_root) if args.sports_label_root else None))
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__': main()
