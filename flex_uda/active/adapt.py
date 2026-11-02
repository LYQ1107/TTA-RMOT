"""Fast calibration and slow, causal-input PCD-LoRA adaptation; no GT IO."""
from collections import defaultdict
import time
import numpy as np
import torch
import torch.nn.functional as F


def fast_step(state, presence_logits, *, temperature=1., bias=0., slope_reference=None):
    pseudo = state.pseudo_labels()
    valid = pseudo >= 0
    a = torch.tensor(state.a, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(state.slope if slope_reference is None else slope_reference, dtype=torch.float64, requires_grad=True)
    z = torch.tensor(state.z, dtype=torch.float64)
    target = torch.tensor(pseudo.clip(0), dtype=torch.float64)
    valid_tensor = torch.tensor(valid)
    log_t = torch.tensor(np.log(temperature), dtype=torch.float64, requires_grad=True)
    beta = torch.tensor(bias, dtype=torch.float64, requires_grad=True)
    confirmed = state.confirmed_present | state.absent
    p_logit = torch.tensor(np.asarray(presence_logits).max(axis=1), dtype=torch.float64)
    e_target = torch.tensor(state.confirmed_present, dtype=torch.float64)
    confirmed_tensor = torch.tensor(confirmed)
    if not valid.any() and not confirmed.any():
        return dict(status='SKIP_NO_CONFIDENT_EVIDENCE', pseudo_positive=0, pseudo_negative=0,
                    temperature=temperature, bias=bias, iterations=0)
    parameters = [a, b] if valid.any() else []
    if confirmed.any(): parameters += [log_t, beta]
    optimizer = torch.optim.LBFGS(parameters, max_iter=50, line_search_fn='strong_wolfe')
    calls = 0
    def closure():
        nonlocal calls
        calls += 1; optimizer.zero_grad()
        loss = a.sum()*0
        if valid.any():
            loss = F.binary_cross_entropy_with_logits((a[:, None]+b*z)[valid_tensor], target[valid_tensor]) + .1*a.square().mean()
        if confirmed.any():
            loss = loss + F.binary_cross_entropy_with_logits((p_logit/log_t.exp()+beta)[confirmed_tensor], e_target[confirmed_tensor])
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite fast-calibration objective')
        loss.backward()
        return loss
    before = float(closure().detach()); optimizer.step(closure); after = float(closure().detach())
    if valid.any(): state.replace_prior(intercept=a.detach().numpy(), slope=float(b.detach()))
    return dict(status='FAST_UPDATED', loss_before=before, loss_after=after, closure_calls=calls,
        pseudo_positive=int((pseudo == 1).sum()), pseudo_negative=int((pseudo == 0).sum()),
        confirmed_presence_queries=int(confirmed.sum()), temperature=float(log_t.detach().exp()),
        bias=float(beta.detach()), slope=state.slope, a_l2_definition='0.1*mean(a_q**2)', iterations=50)


def sampled_frames(episode, track_id, cap=8):
    frames = [o.frame_id for o in episode.observations_by_track[track_id]]
    return [frames[i] for i in np.unique(np.linspace(0, len(frames)-1, min(cap, len(frames)), dtype=int))]


def concatenate_prepared(parts):
    return dict(outputs=[torch.cat([p['outputs'][i] for p in parts]) for i in range(len(parts[0]['outputs']))],
        text=torch.cat([p['text'] for p in parts]), mask=torch.cat([p['mask'] for p in parts]),
        positions=torch.cat([p['positions'] for p in parts]), boxes=torch.cat([p['boxes'] for p in parts]),
        track_ids=[i for p in parts for i in p['track_ids']])


class SlowAdapter:
    def __init__(self, model, cache, *, rank=4, alpha=8, lr=2e-4, seed=0):
        from flex_tta.adaptation.pcd import PCDLoRATTA
        self.model, self.cache = model, cache
        self.adapter = PCDLoRATTA(model, rank=rank, layer_indices=(-2, -1), projections=('wq', 'wv'), alpha=alpha).to(cache.device)
        self.parameters = [p for p in self.adapter.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.parameters, lr=lr, weight_decay=.0001)
        self.scaler = torch.amp.GradScaler('cuda', enabled=cache.device.type == 'cuda')
        self.rng = np.random.RandomState(seed)
        self.updates = 0

    def reset_video(self):
        self.adapter.reset_episode()
        self.optimizer.state.clear()
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.cache.device.type == 'cuda')

    def set_cache(self, cache): self.cache = cache

    def slow_step(self, state, track_ids, texts, *, exclusive=False, steps=50, batch_size=32):
        from flex_uda.presence.features import margins_and_features
        from flex_uda.model.flexhook_uda import prepared_inputs
        from flex_uda.diag_fit import inherited_amp_backoff
        episode = self.cache.episode; pseudo = state.pseudo_labels(); samples = []
        if exclusive:
            for ti, track in enumerate(track_ids):
                positives = np.flatnonzero(pseudo[:, ti] == 1)
                if len(positives):
                    label = int(max(positives, key=lambda q: (state.posterior[q, ti], -q)))
                elif np.all(pseudo[:, ti] == 0): label = len(texts)
                else: continue
                samples.extend((ti, frame, label) for frame in sampled_frames(episode, track))
        else:
            for qi, ti in zip(*np.where(pseudo >= 0)):
                samples.extend((int(ti), frame, int(qi)) for frame in sampled_frames(episode, track_ids[ti]))
        if not samples:
            return dict(status='SKIP_NO_CONFIDENT_TRACKS', optimizer_updates=0, GPU_seconds=0., samples=0)
        started = time.perf_counter(); records = []; start_updates = self.updates
        all_text = [self.cache._text(t) for t in texts]
        for step in range(steps):
            selected = [samples[i] for i in self.rng.choice(len(samples), batch_size, replace=len(samples)<batch_size)]
            parts = [prepared_inputs(self.cache, [(track_ids[ti], episode.history(track_ids[ti], frame))],
                     texts[0] if exclusive else texts[label], frame) for ti, frame, label in selected]
            prepared = concatenate_prepared(parts)
            query_text = all_text if exclusive else [(prepared['text'], prepared['mask'])]
            labels = torch.tensor([label if exclusive else int(pseudo[label, ti]) for ti, frame, label in selected], device=self.cache.device)
            retry = 0
            while True:
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(self.cache.device.type, dtype=torch.float16, enabled=self.scaler.is_enabled()):
                    margin, _ = margins_and_features(self.model, prepared, query_text)
                    loss = (F.cross_entropy(torch.cat([margin, torch.zeros_like(margin[:, :1])], 1), labels)
                            if exclusive else F.binary_cross_entropy_with_logits(margin[:, 0], labels.float()))
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite slow LoRA objective')
                self.scaler.scale(loss).backward(); self.scaler.unscale_(self.optimizer)
                norm = torch.nn.utils.clip_grad_norm_(self.parameters, 5.)
                if not torch.isfinite(norm):
                    retry, _ = inherited_amp_backoff(self.scaler, retry)
                    records.append(dict(step=step, numerical_retry=retry, optimizer_updated=False))
                    continue
                self.scaler.step(self.optimizer); self.scaler.update(); self.updates += 1
                records.append(dict(step=step+1, loss=float(loss.detach()), gradient_norm=float(norm), optimizer_updated=True))
                break
        torch.cuda.synchronize(self.cache.device)
        return dict(status='SLOW_UPDATED', optimizer_updates=self.updates-start_updates,
                    samples=len(samples), GPU_seconds=time.perf_counter()-started, records=records,
                    trainable_parameters=sum(p.numel() for p in self.parameters))

    @torch.no_grad()
    def rescore(self, track_ids, texts):
        from flex_uda.presence.features import margins_and_features
        from flex_uda.model.flexhook_uda import prepared_inputs
        episode = self.cache.episode; by_frame = defaultdict(list)
        for ti, track in enumerate(track_ids):
            for frame in sampled_frames(episode, track): by_frame[frame].append((ti, track))
        score = np.zeros((len(texts), len(track_ids))); counts = np.zeros(len(track_ids), dtype=int)
        all_text = [self.cache._text(t) for t in texts]
        for frame, candidates in sorted(by_frame.items()):
            for a in range(0, len(candidates), 8):
                selected = candidates[a:a+8]
                prepared = prepared_inputs(self.cache, [(i, episode.history(i, frame)) for ti, i in selected], texts[0], frame)
                ti = [i for i, _ in selected]
                for q in range(0, len(texts), 8):
                    margin, _ = margins_and_features(self.model, prepared, all_text[q:q+8])
                    score[q:q+len(all_text[q:q+8]), ti] += margin.float().cpu().numpy().T
                counts[ti] += 1
        return score/counts[None]
