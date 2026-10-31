"""Strict Train→Val input policy; independent audit plus kernel read whitelist."""
import os
from pathlib import Path
import sys

from flex_uda.data.manifests import DANCE, ROOT, SOURCE, official_splits


def install_train_only_audit(pseudo_directory, output, extra_read=()):
    """Install before loading the target cache, not after reading its contents."""
    pseudo = Path(pseudo_directory).resolve()
    output = Path(output).resolve()
    train = set(official_splits()['dance']['train'])
    val = set(official_splits()['dance']['val'])
    allowed = [pseudo, output, *(Path(p).resolve() for p in extra_read)]
    attempts = []

    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if any(path.is_relative_to(p) for p in allowed):
            return
        reason = None
        parts = set(path.parts)
        if parts & val or path.stem in val:
            reason = 'VAL_VIDEO_INPUT'
        elif path.name in ('dance_val_runtime.jsonl', 'dance_val_evaluation.jsonl'):
            reason = 'VAL_EXPRESSION_MANIFEST'
        elif path.is_relative_to(ROOT / 'results/runs/diag_fit_20261030_v1') or path.is_relative_to(ROOT / 'results/diag_fit'):
            reason = 'OLD_DFIT_ARTIFACT'
        elif path.is_relative_to(Path('/data2/user/TTA-RMOT-assets/psr/val_v1')):
            reason = 'OLD_VAL_TEACHER_CACHE'
        elif 'gt_template' in parts and not path.is_relative_to(SOURCE):
            reason = 'TARGET_REFERRING_GT'
        elif path.is_relative_to(DANCE) and (path.name == 'labels.json' or 'expression' in parts or 'labels_with_ids' in parts):
            reason = 'RAW_TARGET_ANNOTATION_CONTAINER'
        elif path.is_relative_to(DANCE / 'DanceTrack/training/image_02') and not parts & train:
            reason = 'NON_TRAIN_TARGET_IMAGE'
        if reason:
            attempts.append(dict(path=str(path), reason=reason))
            raise PermissionError('TRAIN_ONLY input policy: ' + reason + ': ' + str(path))
    sys.addaudithook(audit)
    return attempts


def train_read_roots(config, pseudo):
    """Never grant a parent containing Val images, manifests, caches or weights."""
    train = official_splits()['dance']['train']
    if pseudo.split != 'train' or {r['video_id'] for r in pseudo.runtime} != set(train):
        raise PermissionError('Train-only complete official split required')
    return [ROOT / config['data']['source_train_manifest'], ROOT / config['data']['source_val_manifest'],
        *(DANCE / 'DanceTrack/training/image_02' / v for v in train),
        Path(config['data']['frozen_reference_root']), pseudo.directory,
        Path(pseudo.runtime[0]['predicted_tracks']).parent]
