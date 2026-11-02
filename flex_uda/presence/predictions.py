"""Immutable candidate selection shared by presence and active decoders."""
import json
from pathlib import Path
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha


def seal_mask(folder, meta, mask, *, target, model, method, GT_open_attempts, extra=None):
    folder = Path(folder)
    entries = []
    for query in meta['episodes']:
        video, expression = query['video_id'], query['expression_id']
        bank = meta['videos'][video]
        selected = np.flatnonzero(mask[query['start']:query['stop']])
        text = ''.join(bank['lines'][i]+'\n' for i in selected)
        path = folder / 'results' / video / expression / 'predict.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_text() != text: raise ValueError('Do not overwrite a different prediction: ' + str(path))
        else:
            path.write_text(text)
        entries.append(dict(video=video, expression=expression, prediction_path=str(path),
                            prediction_sha256=file_sha(path), selected_rows=len(selected)))
    receipt = dict(status='SEALED_BEFORE_GT', target=target, model=model, method=method,
        episodes=entries, episode_count=len(entries), prediction_rows=sum(r['selected_rows'] for r in entries),
        immutable_MOT=True, GT_open_attempts=GT_open_attempts, **(extra or {}))
    atomic_json(folder / 'predictions_seal.json', receipt)
    return receipt


def verify_predictions(folder):
    folder = Path(folder)
    receipt = json.loads((folder / 'predictions_seal.json').read_text())
    if receipt['GT_open_attempts']: raise ValueError('Prediction-generation GT isolation failed')
    for row in receipt['episodes']:
        if file_sha(row['prediction_path']) != row['prediction_sha256']:
            raise ValueError('Prediction changed after seal')
    return receipt
