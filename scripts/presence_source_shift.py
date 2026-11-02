"""Offline source-only diagnosis: GT-candidate train vs tracker validation.

Read exported own-source labels, not a target adaptation process. These
descriptive statistics never set model thresholds, classes or hyperparameters.
"""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.presence.runtime import RUN, PUBLIC, IN_DOMAIN
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.diag_attrib_v2_common import write_csv


def support(root):
    hist = np.zeros(11, dtype=np.int64)
    candidate_counts, positive_counts, phrase_set = [], [], set()
    raw_count_sum = overflow = units = videos = 0
    for directory in sorted((root / 'videos').iterdir()):
        metadata = json.loads((directory / 'metadata.json').read_text())
        frames = np.asarray(metadata['frames'])
        observations = np.load(directory / 'observations.npy')
        labels = np.load(directory / 'labels.npy', mmap_mode='r').reshape(len(metadata['queries']), len(observations))
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
    rows, detail = [], {}
    for target, model in IN_DOMAIN.items():
        for split, root in [('TRAIN_GT_CANDIDATES', RUN / 'features/source_train' / target / model),
                            ('SOURCE_VALIDATION_TRACKER', RUN / 'features' / target / model)]:
            result = support(root)
            detail[target+'/'+split] = dict(model=model, source_domain=target, path=str(root), **result)
            rows.append(dict(source_domain=target, model=model, split=split,
                **{k:v for k,v in result.items() if k != 'count_histogram_0_to_9_ge10'}))
    write_csv(PUBLIC / 'source_support_shift.csv', rows)
    atomic_json(PUBLIC / 'source_support_shift.json', dict(ORACLE_GT_DIAGNOSTIC_ONLY=True,
        labels_scope='each head own-source train and own-source official validation only',
        no_new_model_or_teacher_calls=True, no_hyperparameters_selected=True, records=detail))
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__': main()
