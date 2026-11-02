"""Memory-mapped frame/query sets with candidate-synchronous augmentation."""
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class FeatureUnits(Dataset):
    def __init__(self, root, *, labels=True, validation=False, shard_subdir='videos'):
        self.root, self.validation, self.labels = Path(root), validation, labels
        self.units, self.shards, self.arrays = [], [], OrderedDict()
        self.positive_units = 0
        for folder in sorted((self.root / shard_subdir).iterdir()):
            if not (folder / 'seal.json').is_file():
                raise ValueError('Unsealed feature shard: ' + str(folder))
            meta = json.loads((folder / 'metadata.json').read_text())
            obs = np.load(folder / 'observations.npy', mmap_mode='r')
            n = len(obs)
            frames, starts, nums = np.unique(obs[:, 0], return_index=True, return_counts=True)
            ranges = {int(t): (int(a), int(a+c)) for t, a, c in zip(frames, starts, nums)}
            y = np.load(folder / 'labels.npy', mmap_mode='r') if labels else None
            if validation and not (folder / 'GT_counts.npy').exists():
                raise ValueError('Prepare independent own-source validation labels first')
            shard = len(self.shards); self.shards.append((folder, meta))
            for qi in range(len(meta['queries'])):
                for fi, frame in enumerate(meta['frames']):
                    start, stop = ranges.get(frame, (0, 0))
                    # Training and validation both count positive CANDIDATES.
                    # GT_counts.npy additionally records total scene GT, which
                    # may exceed candidate support. It is offline diagnosis,
                    # not an instruction to select extra false candidates.
                    count = int(y[qi*n+start:qi*n+stop].sum()) if labels else -1
                    self.units.append((shard, qi, frame, qi*n+start, qi*n+stop, start, stop, count))
                    self.positive_units += count > 0

    def __len__(self): return len(self.units)

    def arrays_for(self, shard):
        if shard not in self.arrays:
            folder, _ = self.shards[shard]
            names = ['pcd', 'margin', 'boxes', 'sentence'] + (['labels'] if self.labels else [])
            self.arrays[shard] = {name: np.load(folder / (name+'.npy'), mmap_mode='r') for name in names}
            # These are read-only mmaps, not materialized feature copies.
            # Keep their small handles open: random frame/query batches would
            # otherwise reopen several NPY files for nearly every unit.
        self.arrays.move_to_end(shard)
        return self.arrays[shard]

    def __getitem__(self, index):
        shard, qi, frame, a, b, ba, bb, count = self.units[index]
        ar = self.arrays_for(shard)
        result = dict(pcd=torch.from_numpy(ar['pcd'][a:b].astype('float32')),
                      margin=torch.from_numpy(ar['margin'][a:b].copy()),
                      box=torch.from_numpy(ar['boxes'][ba:bb].copy()),
                      sent=torch.from_numpy(ar['sentence'][qi].copy()), count=count, index=index)
        if self.labels: result['y'] = torch.from_numpy(ar['labels'][a:b].copy())
        return result


def collate_units(rows, *, augment=False):
    result = []
    for row in rows:
        row = dict(row)
        if augment:
            keep = torch.rand(len(row['margin'])) >= .1
            for k in ('pcd', 'margin', 'box', 'y'): row[k] = row[k][keep]
            row['count'] = int(row['y'].sum())
            row['margin'] = row['margin'] + .1*torch.randn_like(row['margin'])
            row['pcd'] = torch.nn.functional.dropout(row['pcd'], p=.1, training=True)
        result.append(row)
    lengths = torch.tensor([len(r['margin']) for r in result])
    n = int(lengths.max())
    batch = {k: pad_sequence([r[k] for r in result], batch_first=True) for k in ('pcd', 'margin', 'box')}
    batch.update(sent=torch.stack([r['sent'] for r in result]),
                 count=torch.tensor([r['count'] for r in result]),
                 padding_mask=torch.arange(n)[None] >= lengths[:, None],
                 index=torch.tensor([r['index'] for r in result]))
    return batch


def head_loss(outputs, count, pos_weight):
    presence, logits = outputs
    truth = count > 0
    bce = torch.nn.functional.binary_cross_entropy_with_logits(presence, truth.float(), pos_weight=pos_weight)
    ce = (torch.nn.functional.cross_entropy(logits[truth], count[truth].clamp_max(10))
          if truth.any() else logits.sum()*0)
    return bce + .5*ce, bce, ce
