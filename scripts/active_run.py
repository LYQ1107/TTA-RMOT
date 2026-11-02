"""GT-free active video stream. Requires the published source-presence gate."""
from argparse import ArgumentParser
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.special import expit
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.presence.runtime import RUN, PUBLIC, IN_DOMAIN, target_metadata, sealed_margins
from flex_uda.active.belief import BeliefState
from flex_uda.active.reliability import Reliability
from flex_uda.active.acquisition import questions, keyframe, exclusive_group, expression_types, round_budget
from flex_uda.active.adapt import fast_step
from flex_tta.runtime.pf_receipts import atomic_json, append_json, file_sha

PAIRS = {'C1': ('V2-best', 'Dance'), 'C2': ('Dance-best', 'KITTI-V2'),
         'C3': ('V2-best', 'SportsMOT'), 'C4': ('Dance-best', 'SportsMOT'),
         'C5': ('LaMOT-best', 'KITTI-V2'), 'C6': ('LaMOT-best', 'Dance')}
VARIANTS = ('FULL', 'RANDOM_ACQ', 'MINUS_PRES', 'MINUS_DS', 'MINUS_ROUTE', 'MINUS_FAST', 'MINUS_SLOW', 'RESET_LORA')


def video_spec(target, video, bank):
    observations = []
    # Use internal candidate frame indices throughout; only KITTI raw output
    # lines have the additional evaluation-frame+1 conversion.
    for (t, i), line in zip(bank['observations'], bank['lines']):
        observations.append(dict(frame_id=t, track_id=i, bbox_xywh=list(map(float, line.split(',')))[2:6]))
    return dict(target=target, video_id=video, image_root=bank['image_root'], observations=observations,
                image_offset=0, extension='.jpg' if target == 'SportsMOT' else '.png', delta={'KITTI-V2': 1, 'Dance': 2, 'SportsMOT': 3}[target])


def video_prior(meta, bank, video, model_name, target, margins, *, temperature=1., bias=0.):
    folder = RUN / 'features' / target / model_name / 'videos' / video
    local = json.loads((folder / 'metadata.json').read_text())
    from flex_uda.active.execution import active_paths
    decoded = active_paths().stage_a / target / model_name
    presence = np.load(decoded / 'presence_logits.npy', mmap_mode='r')
    counts = np.load(decoded / 'count_probs.npy', mmap_mode='r')
    qrows = [q for q in meta['episodes'] if q['video_id'] == video]
    tids = sorted({i for t, i in bank['observations']})
    track_index = {i: k for k, i in enumerate(tids)}
    obs = np.asarray(bank['observations']); tracks = defaultdict(list)
    for j, (t, i) in enumerate(bank['observations']): tracks[i].append(j)
    mean = np.stack([[margins[q['start']+np.asarray(tracks[i])].mean() for i in tids] for q in qrows])
    # FeatureUnits/standalone decode order is sorted video, then query, then
    # actual image frame. Resolve this index without rebuilding a whole-dataset
    # Python unit list separately for each video in the stream.
    offset = 0
    for directory in sorted((RUN / 'features' / target / model_name / 'videos').iterdir()):
        previous = json.loads((directory / 'metadata.json').read_text())
        if previous['video_id'] == video: break
        offset += len(previous['queries'])*len(previous['frames'])
    size = len(qrows)*len(local['frames'])
    frame_presence = np.asarray(presence[offset:offset+size]).reshape(len(qrows), -1).astype(float)
    expected = (np.asarray(counts[offset:offset+size])@np.arange(counts.shape[-1])).reshape(len(qrows), -1)
    frame_index = {t: i for i, t in enumerate(local['frames'])}
    calibrated = expit(frame_presence/temperature+bias)
    return dict(queries=qrows, tracks=tids, track_positions=tracks, track_means=mean,
        lengths=[len(tracks[i]) for i in tids], expected=np.median(calibrated*expected, axis=1),
        max_presence=calibrated.max(axis=1), presence_logits=frame_presence,
        frame_presence=calibrated, frame_index=frame_index, width=local['width'], height=local['height'])


def run(pair, rho, variant, *, smoke_videos=0, pilot_videos=0):
    import torch
    from flex_uda.presence.runtime import episode_from_bank, cache_for, load_model, runtime_guard
    from flex_uda.active.adapt import SlowAdapter
    from flex_uda.active.teacher_client import TeacherClient
    from flex_uda.presence.predictions import seal_mask
    from flex_uda.active.execution import authorize_active
    execution,publication=authorize_active()
    model_name, target = PAIRS[pair]
    torch.manual_seed(0); np.random.seed(0)
    torch.cuda.set_per_process_memory_fraction(min(.9, (16 << 30)/torch.cuda.get_device_properties(0).total_memory))
    attempts = runtime_guard(); started = time.perf_counter()
    meta = target_metadata(target); margins, margin_seal = sealed_margins(target, model_name)
    out = execution.active / pair / variant / ('rho_'+str(rho))
    if smoke_videos: out = out / ('smoke_'+str(smoke_videos))
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'predictions_seal.json').exists():
        print(json.dumps(dict(status='ALREADY_SEALED_NO_RERUN', path=str(out))), flush=True); return
    specs = [video_spec(target, v, b) for v, b in meta['videos'].items()]
    runtime = out / 'teacher_runtime.json'; atomic_json(runtime, specs)
    teacher = TeacherClient(runtime, RUN / 'teacher/raw.sqlite', out / 'teacher_server.log', ROOT)
    ds = Reliability(); slope = 1.; temperature = 1.; bias = 0.
    slow = None; model = tokenizer = transform = None; device = torch.device('cuda:0')
    if variant != 'MINUS_SLOW':
        model, tokenizer, transform, loaded = load_model(target, model_name, out / 'model_log', device)
    ordered = sorted(meta['videos']); selected_videos = ordered[:smoke_videos] if smoke_videos else ordered
    if pilot_videos: selected_videos = selected_videos[:pilot_videos]
    checkpoint_paths = sorted(out.glob('stream_*.pt'))
    start_index = 0
    if checkpoint_paths:
        saved = torch.load(checkpoint_paths[-1], map_location='cpu', weights_only=False)
        ds = saved['ds']; slope, temperature, bias = saved['slope'], saved['temperature'], saved['bias']
        start_index = saved['next_video']
    else: saved = None
    slow_seconds = rescore_seconds = 0.
    try:
        for vi, video in enumerate(selected_videos):
            if vi < start_index: continue
            bank = meta['videos'][video]
            prior = video_prior(meta, bank, video, model_name, target, margins, temperature=temperature, bias=bias)
            texts = [q['query_text'] for q in prior['queries']]
            group = exclusive_group(texts, target)
            spec = next(s for s in specs if s['video_id'] == video)
            tracks = defaultdict(list)
            for o in spec['observations']: tracks[o['track_id']].append(o)
            asked = defaultdict(set); accepted_questions = []; round_logs = []
            budget = int(np.floor(rho*sum(n >= 3 for n in prior['lengths'])*len(texts)))
            from flex_uda.active.baselines import video_rng
            random = video_rng(video)
            rescore_before, slow_before = rescore_seconds, slow_seconds
            rescore_work=0
            rescore_rows=len(texts)*sum(min(8,n) for n in prior['lengths'])
            if model is not None:
                episode = episode_from_bank(video, bank, target, transform); cache = cache_for(model, tokenizer, episode, device)
                if slow is None:
                    slow = SlowAdapter(model, cache)
                    if saved:
                        if variant != 'RESET_LORA':
                            slow.adapter.load_state_dict(saved['adapter']); slow.optimizer.load_state_dict(saved['optimizer'])
                            slow.scaler.load_state_dict(saved['scaler'])
                        # RESET changes the adapter state, not the registered
                        # sampling/RNG sequence after a completed-video resume.
                        slow.rng.set_state(saved['slow_rng'])
                        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state(saved['cuda_rng'])
                else: slow.set_cache(cache)
                if variant == 'RESET_LORA': slow.reset_video()
                elif vi > 0:
                    begin_rescore = time.perf_counter()
                    prior['track_means'] = slow.rescore(prior['tracks'], texts)
                    rescore_seconds += time.perf_counter()-begin_rescore
                    rescore_work+=rescore_rows
            state = BeliefState(prior['track_means'], prior['expected'], prior['max_presence'], slope=slope,
                                use_presence=variant != 'MINUS_PRES', track_lengths=prior['lengths'])
            ever_confirmed = state.pseudo_labels() >= 0
            initial_confirmed_pairs = int(ever_confirmed.sum())
            identities = state.prior > 0
            teacher_before = dict(teacher.stats)
            carry = 0
            for r in range(4):
                values = ds.values if variant != 'MINUS_DS' else {v: (.8, .8) for v in ds.values}
                possible = questions(state, texts, prior['lengths'], values, r,
                                     only_crop=variant == 'MINUS_ROUTE', use_presence=variant != 'MINUS_PRES')
                if variant == 'RANDOM_ACQ': random.shuffle(possible)
                allowance = round_budget(budget, r)+carry
                requests = []; chosen = []; used_pairs = set()
                for question in (possible if allowance > 0 else []):
                    q, ti = question.query, question.track
                    if (q, ti) in used_pairs: continue
                    obs = keyframe(tracks[prior['tracks'][ti]], asked[q, ti], prior['width'], prior['height'])
                    if obs is None: continue
                    used_pairs.add((q, ti)); asked[q, ti].add(obs['frame_id'])
                    chosen.append(question)
                    requests.append(dict(target=target, video_id=video, frame_id=obs['frame_id'], track_id=prior['tracks'][ti],
                        query_id=prior['queries'][q]['expression_id'], query_text=texts[q], view=question.view))
                    if len(chosen) >= allowance: break
                if allowance == 0: requests, chosen = [], []
                before = state.pseudo_labels()
                answers = teacher.ask_many(requests) if requests else []
                for question, answer in zip(chosen, answers):
                    q, ti = question.query, question.track; view = answer['effective_view']
                    yes = answer['yes_probability'] >= .5
                    probability_before = float(state.posterior[q, ti])
                    state.observe(q, ti, yes, view, values[view], presence_question=question.kind == 'PRESENCE')
                    ds.add((video, prior['queries'][q]['expression_id'], prior['tracks'][ti]), view, yes, identities[q, ti])
                    accepted_questions.append(dict(answer, round=r+1, kind=question.kind, information_gain=question.information_gain,
                        identity_vote=bool(identities[q, ti]), posterior_probability_before=probability_before,
                        posterior_probability_after=float(state.posterior[q, ti])))
                carry = allowance-len(answers)
                if variant != 'MINUS_DS': ds.fit()
                after_teacher = state.pseudo_labels(); fast = dict(status='DISABLED')
                if variant != 'MINUS_FAST':
                    s, c = ds.values['IDENTITY'] if variant != 'MINUS_DS' else (.8, .8)
                    reference = (np.log(s/(1-c))-np.log((1-s)/c))/(2*np.log(4))
                    fast = fast_step(state, prior['presence_logits'], temperature=temperature, bias=bias, slope_reference=reference)
                    temperature, bias = fast['temperature'], fast['bias']
                    state.presence_max = expit(prior['presence_logits']/temperature+bias).max(axis=1)
                slow_log = dict(status='DISABLED', optimizer_updates=0, GPU_seconds=0.)
                if slow is not None:
                    slow_log = slow.slow_step(state, prior['tracks'], texts, exclusive=group)
                    slow_seconds += slow_log['GPU_seconds']
                    if slow_log['optimizer_updates']:
                        begin_rescore = time.perf_counter()
                        rescored = slow.rescore(prior['tracks'], texts)
                        rescore_seconds += time.perf_counter()-begin_rescore
                        rescore_work+=rescore_rows
                        state.replace_prior(scores=rescored)
                        if variant != 'MINUS_FAST':
                            fast_after = fast_step(state, prior['presence_logits'], temperature=temperature, bias=bias)
                            temperature, bias = fast_after['temperature'], fast_after['bias']
                            state.presence_max = expit(prior['presence_logits']/temperature+bias).max(axis=1)
                            fast['after_slow'] = fast_after
                counts = state.pseudo_labels()
                newly_ever = int(((counts >= 0) & ~ever_confirmed).sum())
                ever_confirmed |= counts >= 0
                asked_mask = np.zeros_like(counts, dtype=bool)
                for (q, ti), frames in asked.items(): asked_mask[q, ti] = bool(frames)
                row = dict(video=video, video_index=vi, round=r+1, allocated=allowance, actual_questions=len(answers),
                    presence_questions=sum(q.kind == 'PRESENCE' for q in chosen), identity_questions=sum(q.kind == 'IDENTITY' for q in chosen),
                    new_confirmed_pairs_teacher=int(((before < 0) & (after_teacher >= 0)).sum()),
                    new_confirmed_pairs_final=int(((before < 0) & (counts >= 0)).sum()),
                    new_distinct_confirmed_pairs=newly_ever,
                    initial_confirmed_pairs=initial_confirmed_pairs,
                    additional_distinct_confirmed_pairs=int(ever_confirmed.sum())-initial_confirmed_pairs,
                    confident_pairs_never_asked=int(((counts >= 0) & ~asked_mask).sum()),
                    confirmed_pairs=int((counts >= 0).sum()), positive_pairs=int((counts == 1).sum()),
                    absent_queries=int(state.absent.sum()), fast=fast, slow=slow_log, ds=ds.receipt())
                round_logs.append(row); append_json(out / 'rounds.jsonl', row)
                print(json.dumps({k:v for k,v in row.items() if k not in ('slow', 'fast', 'ds')}), flush=True)
            posterior = state.posterior
            selected = posterior > .5
            if group:
                best = posterior.argmax(axis=0)
                selected &= np.arange(len(texts))[:, None] == best[None]
            mask = np.zeros((len(texts), len(bank['observations'])), dtype=bool)
            calibrated_presence = expit(prior['presence_logits']/temperature+bias)
            track_index = {track: i for i, track in enumerate(prior['tracks'])}
            for qi, text in enumerate(texts):
                motion = 'MOTION' in expression_types(text) and variant != 'MINUS_PRES'
                for j, (frame, track) in enumerate(bank['observations']):
                    mask[qi, j] = selected[qi, track_index[track]] and (not motion or calibrated_presence[qi, prior['frame_index'][frame]] >= .5)
            directory = out / 'videos' / video; directory.mkdir(parents=True, exist_ok=True)
            np.save(directory / 'mask.npy', mask)
            np.savez(directory / 'belief.npz', prior=state.prior, evidence=state.evidence, posterior=posterior,
                absent=state.absent, confirmed_present=state.confirmed_present, low_presence=state.low,
                expected_counts=state.expected, presence_max=state.presence_max, intercept=state.a, slope=state.slope,
                temperature=temperature, bias=bias, tracks=np.asarray(prior['tracks']),
                queries=np.asarray([q['expression_id'] for q in prior['queries']]))
            atomic_json(directory / 'questions.json', accepted_questions)
            atomic_json(directory / 'receipt.json', dict(video=video, budget=budget, questions=len(accepted_questions),
                unspent_budget=budget-len(accepted_questions), exclusive_group=group, short_tracks=sum(n < 3 for n in prior['lengths']),
                initial_confident_pairs=initial_confirmed_pairs,
                selected_rows=int(mask.sum()), mask_sha256=file_sha(directory / 'mask.npy'), rounds=round_logs,
                teacher_delta={k: teacher.stats[k]-teacher_before.get(k, 0) for k in ('logical_calls', 'new_calls', 'cache_hits', 'forward_seconds', 'equivalent_forward_seconds')},
                slow_GPU_seconds=slow_seconds-slow_before, rescore_GPU_seconds=rescore_seconds-rescore_before,
                rescore_candidate_query_rows=rescore_work))
            slope = state.slope
            checkpoint = dict(next_video=vi+1, ds=ds, slope=slope, temperature=temperature, bias=bias,
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                adapter=slow.adapter.state_dict() if slow else None, optimizer=slow.optimizer.state_dict() if slow else None,
                scaler=slow.scaler.state_dict() if slow else None, slow_rng=slow.rng.get_state() if slow else None)
            path = out / f'stream_{vi:03d}.pt'
            temporary = path.with_suffix('.partial')
            torch.save(checkpoint, temporary); temporary.replace(path)
            saved = checkpoint
        teacher_receipt = teacher.close(); teacher = None
        if smoke_videos:
            atomic_json(out / 'smoke_receipt.json', dict(status='REAL_VIDEO_SMOKE_COMPLETE_NOT_FULL_RMOT', videos=selected_videos,
                teacher=teacher_receipt, LoRA_GPU_seconds=slow_seconds, rescore_GPU_seconds=rescore_seconds, GT_open_attempts=len(attempts)))
            return
        if pilot_videos and len(selected_videos) < len(ordered):
            video_receipts = [json.loads((out / 'videos' / video / 'receipt.json').read_text()) for video in selected_videos]
            sessions=[json.loads(p.read_text()) for p in (out / 'teacher_sessions').glob('*.json')]
            teacher_receipt=dict(teacher_receipt,logical_calls=sum(v['questions'] for v in video_receipts),
                attempted_logical_calls=sum(s['logical_calls'] for s in sessions),
                **{key:sum(s[key] for s in sessions) for key in ('new_calls','cache_hits','forward_seconds','equivalent_forward_seconds')},
                peak_allocated_bytes=max(s['peak_allocated_bytes'] for s in sessions),
                all_session_wall_seconds=sum(s['process_wall_seconds'] for s in sessions),
                prior_session_resume_included=True)
            receipt=dict(status='REAL_PILOT_COMPLETE_RESUMABLE_NOT_FULL_RMOT',
                pair=pair, target=target, model=model_name, rho=rho, variant=variant,
                videos=selected_videos, total_videos=len(ordered), video_receipts=video_receipts,
                teacher=teacher_receipt, LoRA_GPU_seconds=sum(v['slow_GPU_seconds'] for v in video_receipts),
                rescore_GPU_seconds=sum(v['rescore_GPU_seconds'] for v in video_receipts),
                GT_open_attempts=len(attempts), wall_seconds=time.perf_counter()-started,
                resume='same output directory and stream_NNN.pt; no healthy video rerun',
                no_full_predictions_seal=True,stage_A_publication=publication,
                execution_manifest_sha256=file_sha(execution.manifest_path) if execution.manifest else None)
            previous=out / 'pilot_receipt.json'
            if previous.exists():
                old=json.loads(previous.read_text())
                atomic_json(out / 'pilot_receipts' / f"first_{len(old['videos']):03d}.json",old)
            atomic_json(out / 'pilot_receipts' / f'first_{len(selected_videos):03d}.json',receipt)
            atomic_json(previous,receipt)
            return
        mask = np.zeros(len(margins), dtype=bool)
        video_receipts = [json.loads((out / 'videos' / video / 'receipt.json').read_text()) for video in ordered]
        sessions = [json.loads(p.read_text()) for p in (out / 'teacher_sessions').glob('*.json')]
        teacher_receipt = dict(teacher_receipt,
            logical_calls=sum(v['questions'] for v in video_receipts),
            attempted_logical_calls=sum(s['logical_calls'] for s in sessions),
            new_calls=sum(s['new_calls'] for s in sessions), cache_hits=sum(s['cache_hits'] for s in sessions),
            forward_seconds=sum(s['forward_seconds'] for s in sessions),
            equivalent_forward_seconds=sum(s['equivalent_forward_seconds'] for s in sessions),
            peak_allocated_bytes=max(s['peak_allocated_bytes'] for s in sessions),
            prior_session_resume_included=True)
        for video in ordered:
            value = np.load(out / 'videos' / video / 'mask.npy')
            qrows = [q for q in meta['episodes'] if q['video_id'] == video]
            for qi, q in enumerate(qrows): mask[q['start']:q['stop']] = value[qi]
        seal_mask(out, meta, mask, target=target, model=model_name, method=variant, GT_open_attempts=len(attempts),
            extra=dict(pair=pair, rho=rho, teacher=teacher_receipt,
                LoRA_GPU_seconds=sum(v['slow_GPU_seconds'] for v in video_receipts),
                rescore_GPU_seconds=sum(v['rescore_GPU_seconds'] for v in video_receipts), margin_input=margin_seal,
                video_transductive=True, strict_frame_causal=False, ordered_video_stream=True,
                wall_seconds=time.perf_counter()-started,stage_A_publication=publication,
                execution_manifest_sha256=file_sha(execution.manifest_path) if execution.manifest else None))
    finally:
        if teacher is not None and teacher.child.poll() is None: teacher.close()


def main():
    p = ArgumentParser(description=__doc__); p.add_argument('--pair', choices=list(PAIRS), required=True)
    p.add_argument('--rho', type=float, choices=[.1, .3, 1.], default=.3)
    p.add_argument('--variant', choices=VARIANTS, default='FULL'); p.add_argument('--smoke-videos', type=int, default=0)
    p.add_argument('--pilot-videos', type=int, default=0,
                   help='Run the first N videos in the FINAL directory, then stop for cost audit; resume preserves their state')
    a = p.parse_args()
    if a.pilot_videos and a.smoke_videos: p.error('Choose an independent smoke OR a resumable final-path pilot')
    from flex_uda.active.receipts import record_attempt
    from flex_uda.active.execution import active_paths
    folder = active_paths().active / a.pair / a.variant / ('rho_'+str(a.rho))
    if a.smoke_videos: folder = folder / ('smoke_'+str(a.smoke_videos))
    with record_attempt(folder, phase='pilot' if a.pilot_videos else 'smoke' if a.smoke_videos else 'full'):
        run(a.pair, a.rho, a.variant, smoke_videos=a.smoke_videos, pilot_videos=a.pilot_videos)


if __name__ == '__main__': main()
