"""v8 evidence-derived BCE pools and exactly balanced slow updates."""
import time
import numpy as np
import torch
import torch.nn.functional as F
from flex_uda.active.v7_slow import SlowAdapterV7
from flex_uda.active.adapt import sampled_frames, concatenate_prepared


def training_pools(state, track_ids, episode, *, exclusive=False):
    pseudo = state.pseudo_labels()
    positive, negative = {}, {}
    for q, ti in zip(*np.where(pseudo >= 0)):
        key = (int(q), int(ti))
        if pseudo[q, ti] == 1: positive[key] = 'DIRECT_POS'
        else: negative[key] = 'DIRECT_NEG'
    if exclusive:
        for _, ti in positive:
            for q in range(len(state.a)):
                key = (q, ti)
                if key not in positive and key not in negative:
                    negative[key] = 'EXCLUSIVE_NEG'
    for q in np.flatnonzero(state.absent & ~state.low_tentative):
        for ti in np.flatnonzero(~state.frozen_tracks):
            key = (int(q), int(ti))
            if key not in positive and key not in negative:
                negative[key] = 'ABSENT_NEG'
    samples = {k: [] for k in ('DIRECT_POS', 'DIRECT_NEG', 'EXCLUSIVE_NEG', 'ABSENT_NEG')}
    for mapping in (positive, negative):
        for (q, ti), source in sorted(mapping.items()):
            if state.frozen_tracks[ti]: continue
            samples[source].extend((q, ti, int(frame), source)
                                   for frame in sampled_frames(episode, track_ids[ti]))
    npos = len(samples['DIRECT_POS'])
    nneg = sum(len(v) for k, v in samples.items() if k != 'DIRECT_POS')
    counts = {k: sum(s == k for s in positive.values())+sum(s == k for s in negative.values()) for k in samples}
    receipt = dict(distinct_positive_pairs=len(positive), distinct_negative_pairs=len(negative),
                   pair_pools=counts, frame_pools={k: len(v) for k, v in samples.items()},
                   training_pool_positive=npos, training_pool_negative=nneg,
                   training_pool_count_unit='sampled query-track-frame',
                   balanced_gate_pass=len(positive) >= 8 and len(negative) >= 8 and npos >= 32 and nneg >= 32,
                   binary_positive_weight=1., none_class_weight=1., loss_type='BINARY_BCE',
                   derived_pair_precedence='DIRECT_NEG > EXCLUSIVE_NEG > ABSENT_NEG; no duplicate pairs')
    return samples, receipt


def balanced_batch(pools, rng):
    def choose(pool, n):
        return [pool[i] for i in rng.choice(len(pool), n, replace=len(pool) < n)]
    direct = pools['DIRECT_NEG']
    derived = pools['EXCLUSIVE_NEG']+pools['ABSENT_NEG']
    negatives = (choose(direct, 8)+choose(derived, 8) if direct and derived
                 else choose(direct or derived, 16))
    result = choose(pools['DIRECT_POS'], 16)+negatives
    assert sum(x[3] == 'DIRECT_POS' for x in result) == 16
    assert len(result) == 32
    return result


class SlowAdapterV8(SlowAdapterV7):
    def slow_step(self, state, track_ids, texts, *, exclusive=False, steps=50, batch_size=32):
        from flex_uda.presence.features import margins_and_features
        from flex_uda.model.flexhook_uda import prepared_inputs
        from flex_uda.diag_fit import inherited_amp_backoff
        assert batch_size == 32
        pools, counts = training_pools(state, track_ids, self.cache.episode, exclusive=exclusive)
        common = dict(counts, samples=sum(map(len, pools.values())),
                      actual_optimized_positive_samples=0, actual_optimized_negative_samples=0,
                      actual_optimizer_steps=0, optimizer_updates=0, GPU_seconds=0., records=[])
        if not counts['balanced_gate_pass']:
            return dict(common, status='SKIP_INSUFFICIENT_BALANCE')
        started = time.perf_counter(); records = []; start_updates = self.updates
        for step in range(steps):
            selected = balanced_batch(pools, self.rng)
            prepared = concatenate_prepared([
                prepared_inputs(self.cache, [(track_ids[ti], self.cache.episode.history(track_ids[ti], frame))], texts[q], frame)
                for q, ti, frame, _ in selected])
            labels = torch.tensor([int(source == 'DIRECT_POS') for _, _, _, source in selected],
                                  dtype=torch.float32, device=self.cache.device)
            assert int(labels.sum()) == 16 and len(labels) == 32
            retry = 0
            while True:
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(self.cache.device.type, dtype=torch.float16, enabled=self.scaler.is_enabled()):
                    margin, _ = margins_and_features(self.model, prepared, [(prepared['text'], prepared['mask'])])
                    loss = F.binary_cross_entropy_with_logits(margin[:, 0], labels)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite v8 balanced BCE objective')
                self.scaler.scale(loss).backward(); self.scaler.unscale_(self.optimizer)
                norm = torch.nn.utils.clip_grad_norm_(self.parameters, 5.)
                if not torch.isfinite(norm):
                    retry, _ = inherited_amp_backoff(self.scaler, retry)
                    records.append(dict(step=step, numerical_retry=retry, optimizer_updated=False))
                    continue
                self.scaler.step(self.optimizer); self.scaler.update(); self.updates += 1
                records.append(dict(step=step+1, loss=float(loss.detach()), gradient_norm=float(norm),
                    optimizer_updated=True, actual_positive_samples=16, actual_negative_samples=16,
                    binary_positive_weight=1., none_class_weight=1.,
                    sample_sources={source: sum(x[3] == source for x in selected) for source in pools}))
                break
        if self.cache.device.type == 'cuda': torch.cuda.synchronize(self.cache.device)
        updates = self.updates-start_updates
        return dict(common, status='SLOW_UPDATED', optimizer_updates=updates, actual_optimizer_steps=updates,
                    actual_optimized_positive_samples=16*updates, actual_optimized_negative_samples=16*updates,
                    GPU_seconds=time.perf_counter()-started, records=records,
                    trainable_parameters=sum(p.numel() for p in self.parameters))
