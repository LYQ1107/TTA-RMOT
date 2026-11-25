"""Sealed direct+weak PSR labels, grouped by a fixed optimizer sampling slot."""
from collections import defaultdict
import json
from pathlib import Path
from flex_uda.data.manifests import read_jsonl,official_splits
from flex_tta.runtime.pf_receipts import file_sha
from .mllm_pseudo_dataset import REVISION,identity,digest


class PSRPseudoDataset:
    def __init__(self,directory,*,allow_val_diagnostic=False,expanded_kind='full'):
        self.directory=Path(directory).resolve();self.seal=json.loads((self.directory/'seal.json').read_text())
        if self.seal['status']!='SEALED_PSR_DIRECT_AND_WEAK_CACHE':raise ValueError('seal teacher and expansion before training')
        self.split=self.seal['target_split']
        if self.split not in ('train','val'):raise ValueError('unknown target split')
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
            if row['manifest_sha256']!=self.seal['files_sha256']['manifest.jsonl'] or row['cache_key']!=digest(row['cache_identity']):
                raise ValueError('teacher manifest or identity-qualified cache mismatch')
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


def parent_weight_rows(direct,opportunities):
    """Replay actual judged endpoints; never fabricate a middle-frame label."""
    by_id={r['sample_id']:r for r in direct};rows=[]
    for weak in opportunities:
        for parent_id in weak['parent_sample_ids']:
            parent=by_id[parent_id]
            rows.append(dict(parent,sample_id='PARENT_REPLAY:'+weak['sample_id']+':'+parent_id,
                supervision_source='DIRECT_PARENT_WEIGHT_CONTROL_NOT_NEW_LABEL',
                original_direct_sample_id=parent_id,control_opportunity_id=weak['sample_id']))
    return rows


class PSRParentWeightDataset(PSRPseudoDataset):
    """Post-hoc geometry mechanism control: .25 extra mean on its two parents."""
    def __init__(self,directory,*,allow_val_diagnostic=False):
        super().__init__(directory,allow_val_diagnostic=allow_val_diagnostic,expanded_kind='geometry')
        opportunities=self.expanded
        self.expanded=parent_weight_rows(self.rows,opportunities)
        by=defaultdict(list)
        for row in self.expanded:by[row['window_id']].append(row)
        for group in self.groups:group['expanded_samples']=by[group['window_id']]
        self.parent_weight_control=dict(kind='POST_HOC_MECHANISM_CONTROL_NOT_PREREGISTERED_HEADLINE',
            geometry_opportunities=len(opportunities),parent_replay_terms=len(self.expanded),
            unique_added_weak_labels=0,extra_teacher_judgments=0,extra_loss_coefficient=.25,
            normalization='mean over both actual judged endpoints for each available geometry opportunity',
            new_target_frames_or_labels=False,target_GT_used=False,
            limitation='tests opportunity-matched positive weighting; not a universal matched-gradient control')


class PairedUnlabeledDataset:
    """Matched EMA baseline reads sampling inputs, NEVER Qwen pseudo labels."""
    def __init__(self,directory,*,allow_val_diagnostic=False):
        self.directory=Path(directory).resolve();self.runtime=read_jsonl(self.directory/'runtime.jsonl')
        self.samples=read_jsonl(self.directory/'manifest.jsonl');self.windows=read_jsonl(self.directory/'windows.jsonl')
        split_videos={r['video_id'] for r in self.runtime};splits=official_splits()['dance']
        self.split='val' if split_videos==set(splits['val']) else 'train' if split_videos==set(splits['train']) else None
        if self.split is None or self.split=='val' and not allow_val_diagnostic:raise ValueError('explicit complete runtime split required')
        self.rows=[dict(r,valid=True,state='UNKNOWN',teacher_prob=.5,supervision_source='UNLABELED_VISUAL_INPUT_NO_MLLM') for r in self.samples]
        by=defaultdict(list)
        for row in self.rows:by[row['window_id']].append(row)
        runtime={(r['video_id'],r['expression_id']):r for r in self.runtime}
        self.expanded=[];self.groups=[dict(runtime[w['video_id'],w['query_id']],frame=w['frame_id'],window_id=w['window_id'],samples=by[w['window_id']],expanded_samples=[]) for w in self.windows]
        self.seal=dict(files_sha256={'manifest.jsonl':file_sha(self.directory/'manifest.jsonl'),'soft_labels.jsonl':None},valid_soft_samples=len(self.rows),
            note='valid means observed visual inputs, not external teacher labels; placeholder probability is unused by EMA loss')
