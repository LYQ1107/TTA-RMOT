"""v6 LoRA optimizer/sampling with the registered v7 class balance and counts."""
import time
import numpy as np
import torch
import torch.nn.functional as F
from flex_uda.active.adapt import SlowAdapter, sampled_frames, concatenate_prepared
from flex_uda.active.v7 import balanced_weights


class SlowAdapterV7(SlowAdapter):
    def slow_step(self, state, track_ids, texts, *, exclusive=False, steps=50, batch_size=32):
        from flex_uda.presence.features import margins_and_features
        from flex_uda.model.flexhook_uda import prepared_inputs
        from flex_uda.diag_fit import inherited_amp_backoff
        from flex_uda.active.v6 import exclusive_training_logits
        episode = self.cache.episode; pseudo = state.pseudo_labels(); samples = []
        if exclusive:
            for ti, track in enumerate(track_ids):
                if state.frozen_tracks[ti]: continue
                positives = np.flatnonzero(pseudo[:, ti] == 1)
                if len(positives): label = int(max(positives, key=lambda q: (state.posterior[q, ti], -q)))
                elif np.all(pseudo[:, ti] == 0): label = len(texts)
                else: continue
                samples.extend((ti, frame, label) for frame in sampled_frames(episode, track))
        else:
            for qi, ti in zip(*np.where(pseudo >= 0)):
                samples.extend((int(ti), frame, int(qi)) for frame in sampled_frames(episode, track_ids[ti]))
        positive = sum(label < len(texts) if exclusive else pseudo[label, ti] == 1 for ti, _, label in samples)
        negative = len(samples)-positive
        weights = balanced_weights(positive, negative, exclusive=exclusive)
        common = dict(samples=len(samples), training_pool_positive=int(positive), training_pool_negative=int(negative),
            training_pool_count_unit='sampled track-frame target/none' if exclusive else 'sampled query-track-frame',
            actual_optimized_positive_samples=0, actual_optimized_negative_samples=0,
            actual_optimizer_steps=0, optimizer_updates=0, GPU_seconds=0., **weights)
        if not samples: return dict(common, status='SKIP_NO_CONFIDENT_TRACKS')
        if not exclusive and negative == 0:
            return dict(common, status='SKIP_ZERO_EFFECTIVE_WEIGHT_POSITIVE_ONLY')
        started = time.perf_counter(); records = []; start_updates = self.updates
        actual_positive = actual_negative = 0
        all_text = [self.cache._text(t) for t in texts]
        ce_weights = torch.ones(len(texts)+1, device=self.cache.device)
        ce_weights[-1] = weights['none_class_weight']
        for step in range(steps):
            selected = [samples[i] for i in self.rng.choice(len(samples), batch_size, replace=len(samples) < batch_size)]
            parts = [prepared_inputs(self.cache, [(track_ids[ti], episode.history(track_ids[ti], frame))],
                     texts[0] if exclusive else texts[label], frame) for ti, frame, label in selected]
            prepared = concatenate_prepared(parts)
            query_text = all_text if exclusive else [(prepared['text'], prepared['mask'])]
            labels = torch.tensor([label if exclusive else int(pseudo[label, ti]) for ti, _, label in selected], device=self.cache.device)
            npos = int((labels < len(texts) if exclusive else labels == 1).sum())
            nneg = batch_size-npos; retry = 0
            while True:
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(self.cache.device.type, dtype=torch.float16, enabled=self.scaler.is_enabled()):
                    margin, _ = margins_and_features(self.model, prepared, query_text)
                    if exclusive:
                        margin = exclusive_training_logits(margin, pseudo, [ti for ti, _, _ in selected])
                        # Per-example mean, matching binary BCE normalization;
                        # the frequency weight is not normalized away in an all-none batch.
                        loss = F.cross_entropy(torch.cat([margin, torch.zeros_like(margin[:, :1])], 1),
                                               labels, weight=ce_weights, reduction='none').mean()
                    else:
                        loss = F.binary_cross_entropy_with_logits(margin[:, 0], labels.float(),
                            pos_weight=margin.new_tensor(weights['binary_positive_weight']))
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite v7 balanced LoRA objective')
                self.scaler.scale(loss).backward(); self.scaler.unscale_(self.optimizer)
                norm = torch.nn.utils.clip_grad_norm_(self.parameters, 5.)
                if not torch.isfinite(norm):
                    retry, _ = inherited_amp_backoff(self.scaler, retry)
                    records.append(dict(step=step, numerical_retry=retry, optimizer_updated=False))
                    continue
                self.scaler.step(self.optimizer); self.scaler.update(); self.updates += 1
                actual_positive += npos; actual_negative += nneg
                records.append(dict(step=step+1, loss=float(loss.detach()), gradient_norm=float(norm),
                    optimizer_updated=True, actual_positive_samples=npos, actual_negative_samples=nneg, **weights))
                break
        if self.cache.device.type == 'cuda': torch.cuda.synchronize(self.cache.device)
        result = dict(common, status='SLOW_UPDATED', optimizer_updates=self.updates-start_updates,
            actual_optimizer_steps=self.updates-start_updates, actual_optimized_positive_samples=actual_positive,
            actual_optimized_negative_samples=actual_negative, GPU_seconds=time.perf_counter()-started,
            records=records, trainable_parameters=sum(p.numel() for p in self.parameters))
        assert actual_positive+actual_negative == batch_size*result['optimizer_updates']
        return result
