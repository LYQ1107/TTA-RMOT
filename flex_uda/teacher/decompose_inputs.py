"""New paired temporal/context inputs; the existing T1 crop code is unchanged."""
from bisect import bisect_right
from collections import defaultdict
import hashlib
from pathlib import Path
from types import SimpleNamespace
from PIL import Image, ImageDraw
from flex_uda.teacher.visual_input_builder import VisualInputBuilder

CLIP_PROMPT = 'These images show the same object at consecutive moments, from earliest to latest. Does this object match the description: "{expression}"? Answer Yes or No.'
FRAME_PROMPT = 'Look at the object inside the red box. Does it match the description: "{expression}"? Answer Yes or No.'


class PairedViews(VisualInputBuilder):
    def __init__(self, videos, upstream):
        super().__init__('.', upstream, 320, .2)
        self.video_specs = {(v['target'], v['video_id']): v for v in videos}
        self.tracks = defaultdict(list)
        self.observations = {}
        for key, spec in self.video_specs.items():
            for r in spec['observations']:
                x, y, w, h = r['bbox_xywh']
                ob = SimpleNamespace(frame_id=r['frame_id'], track_id=r['track_id'], bbox_xyxy=(x,y,x+w,y+h))
                self.observations[key, ob.frame_id, ob.track_id] = ob
                self.tracks[key, ob.track_id].append(ob)
        for obs in self.tracks.values():
            obs.sort(key=lambda ob: ob.frame_id)

    def frame(self, video, frame):
        key = video, frame
        if key in self.frame_cache:
            value = self.frame_cache.pop(key); self.frame_cache[key] = value
            self.cache_counts['frame_hits'] += 1
            return value
        spec = self.video_specs[video]
        path = Path(spec['image_root']) / f"{frame+spec['image_offset']:06d}{spec['extension']}"
        with Image.open(path) as im:
            value = [im.convert('RGB'), None]
        self.frame_cache[key] = value; self.cache_counts['frame_decodes'] += 1
        while len(self.frame_cache) > self.frame_limit:
            self.frame_cache.popitem(last=False)
        return value

    def clip(self, video, t, track):
        delta = self.video_specs[video]['delta']
        track_obs = self.tracks[video, track]
        times = [o.frame_id for o in track_obs]
        selected = []
        for wanted in (t-3*delta, t-2*delta, t-delta, t):
            j = bisect_right(times, wanted)-1
            if j >= 0:
                selected.append(track_obs[j])
        if any(o.frame_id > t for o in selected):
            raise ValueError('Future frame in causal clip')
        # Preserve replacement multiplicity; a repeated historical image remains
        # a repeated crop, not fabricated motion evidence.
        frames = [o.frame_id for o in selected]
        return [self.patch(video, o) for o in selected], dict(frames=frames, usable_crops=len(frames),
                     unique_frames=len(set(frames)), fallback=len(frames)<2, delta=delta)

    def frame_box(self, video, t, track):
        full = self.frame(video,t)[0]
        ratio = 1024 / max(full.size)
        im = full.resize((round(full.width*ratio), round(full.height*ratio)), Image.Resampling.BICUBIC)
        box = self.observations[video,t,track].bbox_xyxy
        ImageDraw.Draw(im).rectangle([x*ratio for x in box], outline=(255,0,0), width=4)
        return im


def images_sha(images):
    h = hashlib.sha256()
    for im in images:
        h.update(str(im.size).encode()); h.update(im.tobytes())
    return h.hexdigest()
