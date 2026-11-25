"""Unlabeled target inputs; never reuse the official label-backed train loader."""
from pathlib import Path
from .manifests import read_jsonl,official_splits

RUNTIME_FIELDS=frozenset(('video_id','expression_id','query_text','image_root','frame_indices','frame_count','predicted_tracks'))


class TargetDanceDataset:
    def __init__(self,manifest,*,split='train',require_complete=True):
        self.rows=read_jsonl(manifest);self.split=split
        allowed=set(official_splits()['dance'][split])
        if not self.rows:raise ValueError('empty target input manifest')
        for row in self.rows:
            if set(row)!=RUNTIME_FIELDS:raise ValueError('target runtime manifest must contain only the input whitelist')
            if row['video_id'] not in allowed:raise ValueError('target train/evaluation video contamination')
        if require_complete and {r['video_id'] for r in self.rows}!=allowed:raise ValueError('incomplete target video coverage')
        if len({(r['video_id'],r['expression_id']) for r in self.rows})!=len(self.rows):raise ValueError('duplicate video-expression inputs')

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=dict(self.rows[index]);path=Path(row['predicted_tracks'])
        if not path.is_file():raise FileNotFoundError('unlabeled predicted MOT candidates required; never fall back to GT: '+str(path))
        from flex_tta.dance_io import load_tracker_lines
        row['predicted_observations']=load_tracker_lines(path.parent,row['video_id'])
        return row
