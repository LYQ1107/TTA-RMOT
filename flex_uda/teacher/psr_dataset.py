"""Sealed direct+weak PSR labels, grouped by a fixed optimizer sampling slot."""
from collections import defaultdict
import json
from pathlib import Path
from flex_uda.data.manifests import read_jsonl,official_splits
from flex_tta.runtime.pf_receipts import file_sha
from .mllm_pseudo_dataset import REVISION,identity


class PSRPseudoDataset:
    def __init__(self,directory,*,allow_val_diagnostic=False,expanded_kind='full'):
        self.directory=Path(directory).resolve();self.seal=json.loads((self.directory/'seal.json').read_text())
        if self.seal['status']!='SEALED_PSR_DIRECT_AND_WEAK_CACHE':raise ValueError('seal teacher and expansion before training')
        self.split=self.seal['target_split']
        if self.split=='val' and not allow_val_diagnostic:raise PermissionError('explicit Val same-set authorization required')
        for name,sha in self.seal['files_sha256'].items():
            if file_sha(self.directory/name)!=sha:raise ValueError('sealed PSR cache changed: '+name)
        self.runtime=read_jsonl(self.directory/'runtime.jsonl');self.samples=read_jsonl(self.directory/'manifest.jsonl')
        self.rows=read_jsonl(self.directory/'soft_labels.jsonl')
        self.expanded=[] if expanded_kind=='none' else read_jsonl(self.directory/('expanded.jsonl' if expanded_kind=='full' else 'expanded_geometry.jsonl'))
        self.windows=read_jsonl(self.directory/'windows.jsonl')
        if {r['video_id'] for r in self.runtime}!=set(official_splits()['dance'][self.split]):raise ValueError('incomplete or contaminated split')
        if len(self.samples)!=len(self.rows) or {r['sample_id'] for r in self.samples}!={r['sample_id'] for r in self.rows}:raise ValueError('teacher manifest mismatch')
        sample_by={r['sample_id']:r for r in self.samples};runtime={(r['video_id'],r['expression_id']):r for r in self.runtime}
        by=defaultdict(list);expanded_by=defaultdict(list)
        for row in self.rows:
            sample=sample_by[row['sample_id']]
            if identity(sample)!=identity(row) or sample['bbox_xyxy']!=row['bbox_xyxy'] or row['teacher_revision']!=REVISION:
                raise ValueError('teacher identity/model mismatch')
            if not 0<=row['teacher_prob']<=1:raise ValueError('invalid raw soft probability')
            by[row['window_id']].append(row)
        direct_ids=set(sample_by);seen=set()
        for row in self.expanded:
            if row['supervision_source']!='EXPANDED_WEAK' or row['sample_id'] in direct_ids or identity(row) in seen:raise ValueError('duplicated or disguised expanded coverage')
            if not all(i in direct_ids for i in row['parent_sample_ids']):raise ValueError('missing direct evidence parents')
            seen.add(identity(row));expanded_by[row['window_id']].append(row)
        self.groups=[]
        for window in self.windows:
            group=dict(runtime[window['video_id'],window['query_id']],frame=window['frame_id'],window_id=window['window_id'],
                samples=by[window['window_id']],expanded_samples=expanded_by[window['window_id']])
            if len(group['samples'])!=window['sampled_candidates']:raise ValueError('incomplete target window')
            self.groups.append(group)
        for path,sha in self.seal['tracker_files_sha256'].items():
            if file_sha(Path(path))!=sha:raise ValueError('fixed MOT changed')

    def __len__(self):return len(self.rows)
