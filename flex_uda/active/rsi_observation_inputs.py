"""Unlabelled visual inputs, followed by explicitly requested teacher data.

Keeping these steps separate prevents action features from accidentally
including unrequested answers in a finite-library curriculum experiment.
"""
from collections import defaultdict
import numpy as np
import torch

from flex_uda.active.joint_observation_fit import probabilities


class RSIVisualInputs:
    def __init__(self, video, actions):
        from scripts.a100_joint_observation_collect import teacher_pool
        self.video, self.actions = video, actions
        self.pools = teacher_pool(video)
        nq, nt, d = len(video.queries), len(video.tracks), video.dimension
        self.pool_values = np.zeros((nq, nt, 3, d), dtype='float32')
        self.pool_present = np.zeros((nt, 3), dtype=bool)
        for ti, rows in enumerate(self.pools):
            self.pool_present[ti, :len(rows)] = True
            for qi in range(nq):
                self.pool_values[qi, ti, :len(rows)] = video.dense_rows(np.full(len(rows), qi), rows)
        self.coarse = np.zeros((nq, nt, d), dtype='float32')
        self.norm_sum, self.norm_square, self.norm_rows = np.zeros(d), np.zeros(d), 0
        for ti in range(nt):
            context = video.sample_rows[ti, video.present[ti]]
            extra = [s for s, row in enumerate(self.pools[ti]) if row not in set(context)]
            values = np.concatenate((video.raw_bags[:, ti, video.present[ti]],
                                     self.pool_values[:, ti, extra]), axis=1).astype('float64')
            self.coarse[:, ti] = values.mean(1)
            hist = np.square(values[..., 768:1080]).reshape(nq, values.shape[1], 3, 104)
            flags = values[..., 1080:1083]
            self.coarse[:, ti, 768:1080] = np.sqrt(hist.sum(1)/np.maximum(flags.sum(1)[..., None], 1)).reshape(nq, 312)
            self.norm_sum += values.sum((0, 1))
            self.norm_square += np.square(values).sum((0, 1))
            self.norm_rows += values.shape[0]*values.shape[1]

    def arrays(self, uids):
        rows = [self.actions[uid] for uid in uids]
        keys = np.asarray([[r[k] for k in ('local_query', 'local_track', 'row_A', 'row_B')] for r in rows], dtype=int)
        qi, ti, ra, rb = keys.T
        return dict(anchors=self.video.raw_bags[qi, ti], present=self.video.present[ti],
                    first=self.video.dense_rows(qi, ra), second=self.video.dense_rows(qi, rb))

    def teacher_dataset(self, received):
        """Only actually requested rows enter solver/ordinary supervision."""
        seen, grouped, endpoint = set(), defaultdict(list), defaultdict(list)
        for row in received:
            uid = row['uid']
            if uid in seen:
                continue
            seen.add(uid)
            assert uid in self.actions
            key = tuple(int(row[k]) for k in ('local_query', 'local_track', 'row_A', 'row_B'))
            qi, ti, ra, rb = key
            p = np.asarray(row['joint_target'], dtype='float64')
            assert p.shape == (4,) and np.isfinite(p).all() and np.all(p >= 0)
            p /= p.sum()
            a, b = row['shared_binary_target']
            binary = np.asarray([a*(1-b), (1-a)*b, a*b, (1-a)*(1-b)]) if ra != rb else p.copy()
            grouped[key].append((p, binary, uid))
            endpoint[qi, ti, ra].append(p[0]+p[2])
            if rb != ra:
                endpoint[qi, ti, rb].append(p[1]+p[2])
        keys = sorted(grouped)
        assert keys
        # The same visual input has one arithmetic view-mixture target for
        # both structured learning and adequate ordinary proper-risk fitting.
        representative = [grouped[k][0][2] for k in keys]
        arrays = self.arrays(representative)
        bags = defaultdict(list)
        for (qi, ti, _), opinions in endpoint.items():
            bags[qi, ti].append(float(np.mean(opinions)))
        bag_keys = sorted(bags)
        arrays.update(joint_target=np.stack([np.mean([p[0] for p in grouped[k]], axis=0) for k in keys]),
                      binary_target=np.stack([np.mean([p[1] for p in grouped[k]], axis=0) for k in keys]),
                      coarse_inputs=np.stack([self.coarse[q, t] for q, t in bag_keys]),
                      coarse_target=np.asarray([np.mean(bags[k]) for k in bag_keys]),
                      report=dict(requested_bundles=len(seen), unique_visual_pairs=len(keys),
                                  coarse_bags=len(bag_keys), unrequested_teacher_rows=0))
        return arrays


def predict_joint(model, arrays, mean, std, device='cpu', batch=1024):
    values = []
    with torch.no_grad():
        for begin in range(0, len(arrays['first']), batch):
            stop = begin+batch
            data = {}
            for key in ('anchors', 'present', 'first', 'second'):
                rows = arrays[key][begin:stop]
                if key != 'present':
                    rows = ((rows-mean)/std).astype('float32')
                    if key == 'anchors':
                        rows[~arrays['present'][begin:stop]] = 0
                data[key] = torch.as_tensor(rows, device=device)
            values.append(probabilities(model, data).cpu().numpy())
    return np.concatenate(values)


def probe_cross_entropy(model, received, visual, mean, std, device='cpu'):
    if not received:
        return None
    uids = [r['uid'] for r in received]
    logp = predict_joint(model, visual.arrays(uids), mean, std, device)
    target = np.asarray([r['joint_target'] for r in received], dtype='float64')
    target /= target.sum(-1, keepdims=True)
    ce = -(target*logp).sum(-1)
    query = np.asarray([r['local_query'] for r in received])
    # A query with many probe views must not dominate the feedback signal.
    return float(np.mean([ce[query == q].mean() for q in np.unique(query)]))
