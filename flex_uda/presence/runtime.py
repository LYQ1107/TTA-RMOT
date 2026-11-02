"""Official complete candidates and exact float32 vendor geometry, no GT IO."""
from collections import OrderedDict, defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from flex_tta.runtime.dance_episode import DanceEpisode, Observation
from flex_tta.runtime.frozen_features import FrozenFeatureCache
from flex_tta.runtime.pf_receipts import file_sha
from scripts.multidomain_runtime import ROOT, RUN as OLD, PUBLIC as OLD_PUBLIC, checkpoint_spec

RUN = ROOT / 'results/runs/presence_active_20261102'
PUBLIC = ROOT / 'results/presence_active'
TARGET_KEY = {'KITTI-V2': 'kitti', 'Dance': 'dance', 'SportsMOT': 'sports'}
IN_DOMAIN = {'KITTI-V2': 'V2-best', 'Dance': 'Dance-best', 'SportsMOT': 'LaMOT-best'}


def target_metadata(target):
    return json.loads((OLD / TARGET_KEY[target] / 'metadata.json').read_text())


def sealed_margins(target, name):
    rows = json.loads((OLD_PUBLIC / 'margin_paths.json').read_text())['files']
    record = next(r for r in rows if r['target'] == target and r['model'] == name and r['coverage'] == 'complete')
    if file_sha(record['path']) != record['sha256']:
        raise ValueError('Changed sealed margin bank')
    return np.load(record['path'], mmap_mode='r'), record


class OfficialGeometryEpisode(DanceEpisode):
    def _image(self, frame_id):
        if frame_id not in self.image_cache:
            with Image.open(self.image_root / f'{frame_id:06d}{self.extension}') as im:
                self.image_cache[frame_id] = self.transform(im.convert('RGB'))
            while len(self.image_cache) > self.image_cache_size:
                self.image_cache.popitem(last=False)
        self.image_cache.move_to_end(frame_id)
        return self.image_cache[frame_id]

    def geometry_inputs(self, candidates, device):
        positions, boxes, ids = [], [], []
        for track, history in candidates:
            b, p = [], []
            for obs in history:
                x1, y1, x2, y2 = obs.bbox_xyxy
                value = torch.tensor([x1/self.width, y1/self.height, x2/self.width, y2/self.height], dtype=torch.float32)
                value[2:] -= value[:2]
                value[:2] += value[2:] / 2
                b.append(value)
                p.append(self._grid(obs.bbox_xyxy))
            positions.append(torch.stack(p)); boxes.append(torch.stack(b)); ids.append(track)
        return torch.stack(positions).to(device), torch.stack(boxes).to(device), ids


def episode_from_bank(video, bank, target, transform):
    episode = OfficialGeometryEpisode.__new__(OfficialGeometryEpisode)
    episode.video, episode.image_root = video, Path(bank['image_root'])
    episode.extension = '.jpg' if target == 'SportsMOT' else '.png'
    episode.observations_by_frame, episode.observations_by_track = defaultdict(dict), defaultdict(list)
    for line, (frame, track) in zip(bank['lines'], bank['observations']):
        x, y, w, h = list(map(float, line.split(',')))[2:6]
        obs = Observation(int(frame), int(track), (x, y, x+w, y+h), line)
        episode.observations_by_frame[int(frame)][int(track)] = obs
        episode.observations_by_track[int(track)].append(obs)
    episode.frames = sorted(episode.observations_by_frame)
    episode.track_index = {i: {o.frame_id: j for j, o in enumerate(rows)} for i, rows in episode.observations_by_track.items()}
    episode.image_cache, episode.image_cache_size = OrderedDict(), 12
    with Image.open(episode.image_root / f'{episode.frames[0]:06d}{episode.extension}') as im:
        episode.width, episode.height = im.size
    episode.transform = transform
    return episode


def load_model(target, name, output, device):
    import compat.sitecustomize
    from transformers import RobertaTokenizerFast
    from scripts.multidomain_kitti_scores import config_for
    from scripts.multidomain_sports_smoke import config as sports_config
    checkpoint, expected = checkpoint_spec(name)
    if file_sha(checkpoint) != expected:
        raise ValueError('Official checkpoint hash changed')
    cfg = sports_config(checkpoint, output) if target == 'SportsMOT' else config_for(checkpoint, output, target)
    from models import build_model
    from data.mydataloader import get_transform
    model = build_model(cfg)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['model']
    strict = model.load_state_dict(state, strict=True)
    del state
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = RobertaTokenizerFast.from_pretrained(cfg.ROBERTA_PATH)
    if tokenizer.encode('person', add_special_tokens=False) != [5970]:
        raise ValueError('Canonical tokenizer regression')
    return model, tokenizer, get_transform('test', cfg), dict(checkpoint=str(checkpoint), checkpoint_sha256=expected,
                  strict_load=str(strict), target_config=cfg.DATA.DATASET, tokenizer_person=[5970], precision='FP32')


def cache_for(model, tokenizer, episode, device):
    return FrozenFeatureCache(model, episode, tokenizer, device, max_bytes=512 << 20)
