"""Separate legal own-source validation labels; never imported by runtime."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.presence.runtime import RUN, PUBLIC, IN_DOMAIN, target_metadata
from flex_tta.runtime.pf_receipts import atomic_json, file_sha


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--target', choices=list(IN_DOMAIN), required=True)
    a = p.parse_args()
    from scripts.multidomain_diagnostics import target_root
    from scripts.analyze_pf_full_offline import load_eval_gt, overlaps, matched_gt
    meta = target_metadata(a.target)
    base = RUN / 'features' / a.target / IN_DOMAIN[a.target]
    receipt = json.loads((base / 'export_receipt.json').read_text())
    if receipt['status'] != 'TARGET_FEATURES_COMPLETE' or receipt['GT_open_attempts']:
        raise ValueError('Seal independent frozen features before source validation labels')
    gt_root = target_root(a.target); rows = []
    for video, bank in meta['videos'].items():
        folder = base / 'videos' / video
        local = json.loads((folder / 'metadata.json').read_text())
        obs = np.load(folder / 'observations.npy')
        n, nq = len(obs), len(local['queries'])
        labels = np.zeros(n*nq, dtype=bool)
        counts = np.zeros((nq, len(local['frames'])), dtype='int16')
        frame_indices = {t: np.flatnonzero(obs[:, 0] == t) for t in local['frames']}
        boxes = np.asarray([list(map(float, line.split(',')))[2:6] for line in bank['lines']])
        boxes[:, 2:] += boxes[:, :2]
        for qi, query in enumerate(local['queries']):
            gt = load_eval_gt(gt_root, video, query['expression_id'])
            for fi, t in enumerate(local['frames']):
                # Source KITTI image index0, official evaluation MOT/GT frame1.
                truth = gt.get(t+(a.target == 'KITTI-V2'), [])
                counts[qi, fi] = len(truth)
                ids = frame_indices[t]
                if len(ids) and truth:
                    _, matched = matched_gt(overlaps(boxes[ids], [b for _, b in truth]), list(range(len(ids))))
                    if matched: labels[qi*n + ids[list(matched)]] = True
        np.save(folder / 'labels.npy', labels); np.save(folder / 'GT_counts.npy', counts)
        rows.append(dict(video=video, units=counts.size, GT_detections=int(counts.sum()),
                         candidate_positive_rows=int(labels.sum()), labels_sha256=file_sha(folder / 'labels.npy'),
                         counts_sha256=file_sha(folder / 'GT_counts.npy')))
    atomic_json(PUBLIC / (a.target + '_source_validation.json'), dict(target=a.target,
        legal_source_validation_only=True, candidate_matching='IoU>=0.5 one-to-one existing v2 matcher', videos=rows))


if __name__ == '__main__': main()
