"""Bounded, causal reuse of *frozen* FlexHook encoders; decode is unchanged."""
from __future__ import annotations

from collections import OrderedDict

import torch


class FrozenFeatureCache:
    def __init__(self, model, episode, tokenizer, device, *, max_bytes=4 << 30):
        self.model, self.episode = model, episode
        self.tokenizer, self.device = tokenizer, device
        self.max_bytes = int(max_bytes)
        self.visual = OrderedDict()
        self.text = {}
        self.bytes = 0
        self.stats = dict(backbone_calls=0, backbone_images=0, text_encoder_calls=0,
                          visual_hits=0, visual_misses=0, peak_cache_bytes=0)

    def _visual(self, frame, current_frame):
        if frame > current_frame:
            raise ValueError("future frame requested by frozen cache")
        if frame in self.visual:
            self.stats["visual_hits"] += 1
            self.visual.move_to_end(frame)
            return self.visual[frame]
        self.stats["visual_misses"] += 1
        with torch.no_grad():
            image = self.episode._image(frame).unsqueeze(0).to(self.device)
            backbone = self.model.backbone
            outputs = (backbone.forward_features(image) if hasattr(backbone, "forward_features")
                       else backbone(image))
            if isinstance(outputs, dict):
                outputs = list(outputs.values())
            value = tuple(x.detach() for x in outputs)
        self.stats["backbone_calls"] += 1
        self.stats["backbone_images"] += 1
        size = sum(x.numel() * x.element_size() for x in value)
        while self.visual and self.bytes + size > self.max_bytes:
            _, old = self.visual.popitem(last=False)
            self.bytes -= sum(x.numel() * x.element_size() for x in old)
        if size <= self.max_bytes:
            self.visual[frame] = value
            self.bytes += size
            self.stats["peak_cache_bytes"] = max(self.stats["peak_cache_bytes"], self.bytes)
        return value

    def _text(self, sentence):
        if sentence not in self.text:
            with torch.no_grad():
                enc = self.tokenizer([sentence], padding="max_length", truncation=True,
                                     max_length=25, return_tensors="pt")
                ids = enc["input_ids"].to(self.device)
                mask = enc["attention_mask"].to(self.device)
                encoder = self.model.text_encoder
                if hasattr(encoder, "encode_text_cpany"):
                    hidden = encoder.encode_text_cpany(ids)
                    mask = mask[:, :25]
                else:
                    hidden = encoder(ids, mask).last_hidden_state
                self.text[sentence] = (hidden.detach().unsqueeze(1), mask.unsqueeze(1))
            self.stats["text_encoder_calls"] += 1
        return self.text[sentence]

    def margin(self, candidates, sentence, current_frame):
        # Only requested past/current frames are retrieved, even when frozen
        # features from a preceding expression still reside in the LRU.
        frame_features = {}
        for _, history in candidates:
            for obs in history:
                if obs.frame_id not in frame_features:
                    frame_features[obs.frame_id] = self._visual(obs.frame_id, current_frame)
        history_features = [frame_features[obs.frame_id] for _, hs in candidates for obs in hs]
        outputs = [torch.cat([row[layer] for row in history_features], dim=0)
                   for layer in range(len(history_features[0]))]
        positions, boxes, ids = self.episode.geometry_inputs(candidates, self.device)
        text, mask = self._text(sentence)
        batch = len(ids)
        text = text.expand(batch, -1, -1, -1).contiguous()
        mask = mask.expand(batch, -1, -1).contiguous()
        output, _ = self.model.decode(outputs, text, mask, positions, boxes)
        if output.ndim == 4:
            output = output[:, -1]
        if output.ndim != 3 or output.shape[1:] != (1, 2):
            raise RuntimeError(f"unexpected cached decode shape: {output.shape}")
        return output[:, 0, 1] - output[:, 0, 0], ids
