"""Unchanged frozen three-view teacher, plus a separate-environment client."""
import json
from pathlib import Path
import subprocess
import sys
import time
from flex_uda.teacher.teacher_cache import TeacherCache, request_key
from flex_tta.runtime.pf_receipts import file_sha

REVISION = '89644892e4d85e24eaac8bacfd4f463576704203'


class LocalTeacher:
    """Lives exclusively inside the existing Qwen environment/process."""
    def __init__(self, videos, cache_path, root):
        import yaml
        import torch
        from flex_uda.teacher.yestrack_backend import FrozenYesTrack
        from flex_uda.teacher.decompose_inputs import PairedViews
        self.root = Path(root)
        self.config = yaml.safe_load((self.root / 'configs/teacher/base_qwen2b.yaml').read_text())
        self.prompts = yaml.safe_load((self.root / 'configs/teacher/prompts_v1.yaml').read_text())
        if self.config['revision'] != REVISION or self.config.get('lora_path') is not None:
            raise ValueError('Only the fixed, adapter-free base teacher is authorized')
        torch.cuda.set_per_process_memory_fraction(min(.9, (20 << 30) / torch.cuda.get_device_properties(0).total_memory))
        self.backend = FrozenYesTrack(self.config)
        self.views = PairedViews(videos, self.backend.upstream)
        self.cache = TeacherCache(cache_path)
        self.stats = dict(logical_calls=0, new_calls=0, cache_hits=0, forward_seconds=0.,
                          equivalent_forward_seconds=0., batches=0)
        self.image_shas = {}

    def prepare(self, request):
        from flex_uda.teacher.decompose_inputs import images_sha, CLIP_PROMPT, FRAME_PROMPT
        target, video, t, track, text, view = [request[k] for k in ('target', 'video_id', 'frame_id', 'track_id', 'query_text', 'view')]
        key = target, video; obs = self.views.observations[key, t, track]
        full = self.views.frame(key, t)[0]
        x1, y1, x2, y2 = obs.bbox_xyxy
        coord = f'{(x1+x2)/2/full.width:.3f} {(y1+y2)/2/full.height:.3f}'
        fallback = False; clip_info = None
        if view == 'CLIP':
            images, clip_info = self.views.clip(key, t, track)
            if clip_info['fallback']: fallback = True
            else: prompt = CLIP_PROMPT.format(expression=text)
        if view == 'CROP' or fallback:
            images = [self.views.patch(key, obs)]
            prompt = self.prompts['official_single'].format(coord=coord, sentence=text)
        elif view == 'FRAME_BOX':
            images = [self.views.frame_box(key, t, track)]
            prompt = FRAME_PROMPT.format(expression=text)
        spec = self.views.video_specs[key]
        image_key = target, video, t
        if image_key not in self.image_shas:
            path = Path(spec['image_root']) / f"{t+spec['image_offset']:06d}{spec['extension']}"
            self.image_shas[image_key] = file_sha(path)
        info = dict(revision=REVISION, weight=self.backend.weight_sha,
            code=file_sha(self.root / 'flex_uda/teacher/decompose_inputs.py'),
            target=target, video=video, frame=t, track=track, query=request['query_id'],
            image=self.image_shas[image_key], visual=images_sha(images), prompt=prompt,
            bbox=list(obs.bbox_xyxy), method='CROP' if fallback else view)
        return request_key(info), images, prompt, dict(request, input=info, clip=clip_info,
            effective_view='CROP' if fallback else view, clip_fallback=fallback)

    def ask_many(self, requests):
        responses = [None]*len(requests); pending = []
        for i, request in enumerate(requests):
            cache_key, images, prompt, info = self.prepare(request)
            old = self.cache.get_or_claim(cache_key)
            self.stats['logical_calls'] += 1
            if old is not None:
                self.stats['cache_hits'] += 1
                self.stats['equivalent_forward_seconds'] += old['batch_forward_seconds']/old['batch_size']
                responses[i] = dict(old, cache_hit=True, requested_view=request['view'],
                    clip=info['clip'], clip_fallback=info['clip_fallback'], effective_view=info['effective_view'])
            else:
                pending.append((i, cache_key, images, prompt, info))
        for a in range(0, len(pending), 8):
            rows = pending[a:a+8]
            raw = self.backend.forward([r[2] for r in rows], [r[3] for r in rows])
            self.stats['forward_seconds'] += raw[0]['batch_forward_seconds']; self.stats['batches'] += 1
            self.stats['new_calls'] += len(rows)
            for (i, key, images, prompt, info), result in zip(rows, raw):
                response = dict(info, **result, request_id=key, cache_hit=False, requested_view=info['view'])
                self.stats['equivalent_forward_seconds'] += result['batch_forward_seconds']/result['batch_size']
                self.cache.put(key, response); responses[i] = response
        return responses

    def close(self):
        self.backend.seal_frozen()
        return dict(self.stats, cache=self.cache.stats(), teacher_weight_sha256=self.backend.weight_sha,
                    backend_unmodified=True, revision=REVISION, model_load_seconds=self.backend.load_seconds,
                    peak_allocated_bytes=__import__('torch').cuda.max_memory_allocated())


class TeacherClient:
    def __init__(self, runtime_path, cache_path, log_path, root):
        self.root = Path(root); self.log = Path(log_path).open('a')
        self.started = time.perf_counter(); self.session_id = str(time.time_ns())
        self.session_root = Path(log_path).parent / 'teacher_sessions'
        python = '/data2/user/TTA-RMOT-assets/teacher-p1/env/bin/python'
        command = [python, str(self.root / 'scripts/active_teacher_worker.py'), '--runtime', str(runtime_path), '--cache', str(cache_path)]
        self.child = subprocess.Popen(command, cwd=self.root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=self.log, text=True, bufsize=1)
        self.stats = dict(logical_calls=0, new_calls=0, cache_hits=0, forward_seconds=0., equivalent_forward_seconds=0.)
        ready = self.read()
        if ready['status'] != 'READY': raise RuntimeError('Frozen teacher did not become ready')

    def read(self):
        for line in self.child.stdout:
            if line.startswith('ACTIVE_RESPONSE\t'): return json.loads(line.split('\t', 1)[1])
            self.log.write(line); self.log.flush()
        raise RuntimeError('Teacher process exited; inspect ' + self.log.name)

    def ask_many(self, requests):
        self.child.stdin.write(json.dumps(dict(command='ASK', requests=requests))+'\n'); self.child.stdin.flush()
        response = self.read()
        if response['status'] != 'ANSWERS': raise RuntimeError('Teacher failed: ' + str(response))
        self.stats = response['costs']
        return response['answers']

    def close(self):
        self.child.stdin.write(json.dumps(dict(command='CLOSE'))+'\n'); self.child.stdin.flush()
        response = self.read(); code = self.child.wait(); self.log.close()
        if code or response['status'] != 'SEALED': raise RuntimeError('Teacher final frozen-weight audit failed')
        from flex_tta.runtime.pf_receipts import atomic_json
        receipt = dict(response['receipt'], process_wall_seconds=time.perf_counter()-self.started,
                       process_wall_is_not_GPU_utilization_integral=True)
        atomic_json(self.session_root / (self.session_id+'.json'), receipt)
        return receipt
