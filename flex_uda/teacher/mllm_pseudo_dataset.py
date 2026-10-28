"""Sealed raw-soft teacher labels, joined by immutable candidate identity."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from flex_tta.runtime.pf_receipts import file_sha
from flex_uda.data.manifests import read_jsonl, official_splits

REVISION = '89644892e4d85e24eaac8bacfd4f463576704203'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def identity(row):
    return (row['video_id'], row['query_id'], int(row['frame_id']), int(row['track_id']))


class MLLMPseudoDataset:
    def __init__(self, directory, *, allow_val_diagnostic=False):
        self.directory = Path(directory).resolve()
        self.seal = json.loads((self.directory/'seal.json').read_text())
        if self.seal['status'] != 'SEALED_RAW_SOFT_TEACHER_CACHE':
            raise ValueError('teacher labels must be sealed before student training')
        split = self.seal['target_split']
        if split == 'val' and not allow_val_diagnostic:
            raise PermissionError('Val labels only authorized for explicit transductive diagnosis')
        if split not in ('train', 'val'):
            raise ValueError('unknown target split')
        self.split = split
        for name, sha in self.seal['files_sha256'].items():
            if file_sha(self.directory/name) != sha:
                raise ValueError('sealed pseudo label data changed: '+name)
        self.samples = read_jsonl(self.directory/'manifest.jsonl')
        labels = read_jsonl(self.directory/'soft_labels.jsonl')
        self.runtime = read_jsonl(self.directory/'runtime.jsonl')
        runtime = {(r['video_id'], r['expression_id']): r for r in self.runtime}
        if len(runtime) != len(self.runtime):
            raise ValueError('duplicate runtime video/query')
        allowed = set(official_splits()['dance'][split])
        if {r['video_id'] for r in self.runtime} != allowed:
            raise ValueError('incomplete/contaminated target split')
        if len({s['sample_id'] for s in self.samples}) != len(self.samples):
            raise ValueError('duplicate sampled identity')
        lookup = {r['sample_id']: r for r in labels}
        if len(lookup) != len(labels) or set(lookup) != {s['sample_id'] for s in self.samples}:
            raise ValueError('duplicate/missing/extra teacher sample IDs')
        self.rows = []
        self.by_window = defaultdict(list)
        for sample in self.samples:
            label = lookup[sample['sample_id']]
            if identity(sample) != identity(label) or sample['bbox_xyxy'] != label['bbox_xyxy']:
                raise ValueError('teacher frame/ID/box identity mismatch')
            if label['teacher_revision'] != REVISION or label['manifest_sha256'] != self.seal['files_sha256']['manifest.jsonl']:
                raise ValueError('teacher model/manifest version mismatch')
            p = label['teacher_prob']
            if not all(math.isfinite(label[k]) for k in ('teacher_yes_logit','teacher_no_logit','teacher_prob')) or not 0 <= p <= 1:
                raise ValueError('invalid teacher probability/logits')
            row = dict(sample, **{k:v for k,v in label.items() if k not in sample})
            key = (row['video_id'], row['query_id'])
            if key not in runtime or row['query_text'] != runtime[key]['query_text']:
                raise ValueError('teacher query text/runtime mismatch')
            if row['cache_key'] != digest(row['cache_identity']):
                raise ValueError('identity-qualified cache key mismatch')
            self.rows.append(row)
            self.by_window[identity(row)[:3]].append(row)
        # Retain the full video/query/anchor schedule even if no candidates exist.
        self.windows = read_jsonl(self.directory/'windows.jsonl')
        if len({(r['video_id'],r['query_id'],r['frame_id']) for r in self.windows}) != len(self.windows):
            raise ValueError('duplicate training window')
        self.groups = []
        for window in self.windows:
            key = (window['video_id'],window['query_id'],window['frame_id'])
            group = dict(runtime[key[:2]], frame=key[2], samples=self.by_window.get(key, []))
            if len(group['samples']) != window['sampled_candidates']:
                raise ValueError('teacher window coverage mismatch')
            self.groups.append(group)
        if sum(len(g['samples']) for g in self.groups) != len(self.rows):
            raise ValueError('samples outside registered time windows')
        for path, sha in self.seal['tracker_files_sha256'].items():
            if file_sha(Path(path)) != sha:
                raise ValueError('fixed MOT changed after teacher generation')

    def __len__(self):
        return len(self.rows)
