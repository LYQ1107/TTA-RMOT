"""A non-invasive adapter around the pinned TempRMOT source checkout.

The source checkout is intentionally loaded from a configured read-only
reference path.  This project owns the hook and snapshot contract; the source
model remains the official frozen implementation.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from rmot_tta.types import FrameSnapshot


@dataclass(frozen=True)
class TempRMOTConfig:
    source_root: Path
    checkpoint: Path
    text_root: Path
    device: str = "cuda:0"
    dataset_file: str = "e2e_rmot"
    hist_len: int = 4
    num_queries: int = 300
    image_height: int = 800
    image_width: int = 1536


@dataclass
class WrapperFrameResult:
    track_instances: Any
    snapshot: FrameSnapshot
    source_output: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


class _RedirectTokenizer:
    def __init__(self, original, text_root: Path):
        self.original = original
        self.text_root = text_root

    def from_pretrained(self, path, *args, **kwargs):
        path = str(self.text_root) if not Path(str(path)).exists() else path
        return self.original.from_pretrained(path, *args, **kwargs)


class _RedirectModel:
    def __init__(self, original, text_root: Path):
        self.original = original
        self.text_root = text_root

    def from_pretrained(self, path, *args, **kwargs):
        path = str(self.text_root) if not Path(str(path)).exists() else path
        return self.original.from_pretrained(path, *args, **kwargs)


class TempRMOTWrapper:
    """Load one frozen source model and expose causal frame snapshots."""

    def __init__(self, config: TempRMOTConfig, *, _model: torch.nn.Module | None = None):
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("TempRMOTWrapper requested CUDA but CUDA is unavailable")
        self.source_root = config.source_root.resolve()
        self._source_modules = self._import_source_modules()
        self.model = _model if _model is not None else self._build_model()
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.text_cache: dict[tuple[str, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._capture: dict[str, Any] | None = None
        self._install_text_cache()
        self._install_snapshot_hook()
        self._assert_frozen()

    def _import_source_modules(self) -> dict[str, Any]:
        source = str(self.source_root)
        shim = "/data2/user/reference_envs/temprmot/shims"
        source_site = "/data2/user/reference_envs/temprmot/site-packages"
        source_ops = str(self.source_root / "models" / "ops")
        for path in (shim, source_site, source_ops):
            if path not in sys.path:
                sys.path.insert(0, path)
        if source not in sys.path:
            sys.path.insert(0, source)
        main_module = importlib.import_module("main")
        trans_module = importlib.import_module("models.transrmot_pro")
        return {"main": main_module, "trans": trans_module}

    def _build_args(self) -> Any:
        parser = self._source_modules["main"].get_args_parser()
        args = parser.parse_args([])
        # These values match the public Refer-KITTI release configuration.  No
        # training branch is entered by the wrapper.
        updates = {
            "meta_arch": "temp_rmot", "dataset_file": self.config.dataset_file,
            "device": str(self.device), "with_box_refine": True, "two_stage": False,
            "num_feature_levels": 4, "hidden_dim": 256, "dim_feedforward": 1024,
            "nheads": 8, "num_queries": self.config.num_queries, "enc_layers": 6,
            "dec_layers": 6, "dec_n_points": 4, "enc_n_points": 4, "dropout": 0.0,
            "aux_loss": True, "query_interaction_layer": "QIM", "update_query_pos": True,
            "merger_dropout": 0.0, "random_drop": 0.1, "fp_ratio": 0.3,
            "extra_track_attn": False, "sample_mode": "random_interval", "sample_interval": 1,
            "sampler_steps": [60, 80, 90], "sampler_lengths": [4, 4, 4, 4],
            "hist_len": self.config.hist_len, "tracking": False, "use_checkpoint": False,
            "memory_bank_type": None, "num_workers": 0, "rmot_path": str(self.config.source_root),
        }
        for name, value in updates.items():
            setattr(args, name, value)
        return args

    def _build_model(self) -> torch.nn.Module:
        trans_module = self._source_modules["trans"]
        # The public source hard-codes the author's old RoBERTa path.  Redirect
        # only that constructor call to the verified local copy; no source file
        # or global environment path is modified.
        trans_module.RobertaTokenizerFast = _RedirectTokenizer(trans_module.RobertaTokenizerFast, self.config.text_root)
        trans_module.RobertaModel = _RedirectModel(trans_module.RobertaModel, self.config.text_root)
        args = self._build_args()
        model, _criterion, _postprocessors = self._source_modules["main"].build_model(args)
        package = torch.load(self.config.checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(package, dict) or "model" not in package:
            raise ValueError(f"checkpoint has no model state: {self.config.checkpoint}")
        missing, unexpected = model.load_state_dict(package["model"], strict=False)
        unexpected = [key for key in unexpected
                      if not key.endswith(("total_params", "total_ops"))
                      and key != "text_encoder.embeddings.position_ids"]
        self.load_report = {"missing_keys": list(missing), "unexpected_keys": list(unexpected)}
        if missing or unexpected:
            raise RuntimeError(f"TempRMOT checkpoint load mismatch: {self.load_report}")
        return model

    def _install_text_cache(self) -> None:
        model = self.model
        # TempRMOT was released against Transformers versions where the fast
        # tokenizer exposed ``batch_encode_plus``.  Transformers 5 keeps the
        # same call contract through ``__call__`` but removed that alias.
        # Bridge only this narrow API difference; the source model and
        # checkpoint remain otherwise untouched.
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None and not hasattr(tokenizer, "batch_encode_plus"):
            tokenizer.batch_encode_plus = tokenizer.__call__
        raw = type(model).forward_text.__get__(model, type(model))

        def cached(text_queries, device):
            key = tuple(str(value) for value in text_queries)
            if key not in self.text_cache:
                with torch.no_grad():
                    value = raw(text_queries, device)
                self.text_cache[key] = tuple(item.detach().clone() for item in value)
            return tuple(item.to(device=device) for item in self.text_cache[key])

        model.forward_text = cached

    @staticmethod
    def _restore_tracker_update(model: torch.nn.Module) -> None:
        tracker = model.track_base
        tracker.update = type(tracker).update.__get__(tracker, type(tracker))

    def _install_snapshot_hook(self) -> None:
        model = self.model
        self._restore_tracker_update(model)
        raw_update = type(model.track_base).update.__get__(model.track_base, type(model.track_base))

        def hooked(track_instances):
            raw_update(track_instances)
            if self._capture is not None:
                self._capture["snapshot"] = self._snapshot_from_instances(
                    track_instances,
                    episode_id=self._capture["episode_id"],
                    video_id=self._capture["video_id"],
                    expression_id=self._capture["expression_id"],
                    query_text=self._capture["query_text"],
                    frame_index=self._capture["frame_index"],
                    image_size=self._capture["image_size"],
                )

        model.track_base.update = hooked

    def _snapshot_from_instances(self, track_instances, *, episode_id: str, video_id: str,
                                 expression_id: str, query_text: str, frame_index: int,
                                 image_size: tuple[int, int]) -> FrameSnapshot:
        features = track_instances.output_embedding.detach().clone().float()
        source_logits = track_instances.cache_pred_refers.detach().clone().float().reshape(-1)
        source_probs = source_logits.sigmoid()
        objectness = track_instances.cache_pred_logits.detach().clone().float()[..., 0].sigmoid()
        boxes_cxcywh = track_instances.pred_boxes.detach().clone().float()
        boxes = torch.stack((
            boxes_cxcywh[:, 0] - 0.5 * boxes_cxcywh[:, 2],
            boxes_cxcywh[:, 1] - 0.5 * boxes_cxcywh[:, 3],
            boxes_cxcywh[:, 0] + 0.5 * boxes_cxcywh[:, 2],
            boxes_cxcywh[:, 1] + 0.5 * boxes_cxcywh[:, 3],
        ), dim=-1)
        h, w = image_size
        boxes = boxes * boxes.new_tensor([w, h, w, h])
        track_ids = track_instances.obj_idxes.detach().clone().long()
        disappear = track_instances.disappear_time.detach().clone().long()
        valid = track_ids >= 0
        uids = tuple(f"{episode_id}:{expression_id}:{frame_index}:{int(track_id)}:{index}"
                     for index, track_id in enumerate(track_ids.detach().cpu().tolist()))
        snapshot = FrameSnapshot(
            episode_id=episode_id, video_id=video_id, expression_id=expression_id,
            query_text=query_text, frame_index=int(frame_index), candidate_uid=uids,
            features=features, source_logits=source_logits, source_probs=source_probs,
            boxes_xyxy=boxes, objectness=objectness, pred_track_id=track_ids,
            disappear_time=disappear, image_size=(int(h), int(w)), valid_mask=valid,
        )
        head = self.model.STReasoner.track_ref
        recomputed = F.linear(features, head.weight.detach().float(), head.bias.detach().float()).reshape(-1)
        if not torch.allclose(recomputed, source_logits, atol=2e-4, rtol=2e-4):
            max_error = float((recomputed - source_logits).abs().max().detach().cpu())
            raise AssertionError(f"snapshot source head mismatch: max_error={max_error}")
        return snapshot

    def _assert_frozen(self) -> None:
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise AssertionError("source TempRMOT model has trainable parameters")

    @property
    def source_head(self) -> tuple[torch.Tensor, torch.Tensor]:
        head = self.model.STReasoner.track_ref
        return head.weight.detach().cpu().clone(), head.bias.detach().cpu().clone()

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "source_root": str(self.source_root),
            "source_commit": _git_commit(self.source_root),
            "checkpoint": str(self.config.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(self.config.checkpoint),
            "text_root": str(self.config.text_root.resolve()),
            "load_report": getattr(self, "load_report", {}),
            "device": str(self.device),
            "source_parameter_count": sum(parameter.numel() for parameter in self.model.parameters()),
            "trainable_parameter_count": sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad),
        }

    def reset_runtime(self) -> None:
        self.model.clear()
        self.text_cache.clear()

    def infer_frame(self, image_tensor: torch.Tensor, sentence: str, image_size: tuple[int, int],
                    track_instances: Any, frame_index: int, episode_id: str, expression_id: str,
                    video_id: str = "") -> WrapperFrameResult:
        image_tensor = image_tensor.to(self.device).float()
        self._capture = {
            "episode_id": episode_id, "video_id": video_id, "expression_id": expression_id,
            "query_text": sentence, "frame_index": int(frame_index), "image_size": tuple(image_size),
        }
        try:
            with torch.no_grad():
                output = self.model.inference_single_image(
                    [image_tensor], [sentence], tuple(image_size), track_instances,
                )
        finally:
            capture = self._capture
            self._capture = None
        snapshot = capture.get("snapshot") if capture is not None else None
        if snapshot is None:
            raise RuntimeError("TempRMOT snapshot hook did not observe RuntimeTrackerBase.update")
        result_instances = output["track_instances"]
        # Verify that the post-QIM referring probability is still the source
        # probability for every physical ID visible in both views.
        final_ids = result_instances.obj_idxes.detach().cpu().tolist()
        final_probs = result_instances.refers.detach().float().cpu().tolist()
        final_by_id = {int(track_id): float(probability) for track_id, probability in zip(final_ids, final_probs) if int(track_id) >= 0}
        for track_id, probability, valid in zip(snapshot.pred_track_id.cpu().tolist(), snapshot.source_probs.cpu().tolist(), snapshot.valid_mask.cpu().tolist()):
            if valid and int(track_id) in final_by_id:
                if abs(float(probability) - final_by_id[int(track_id)]) > 2e-4:
                    raise AssertionError(f"source refers changed across QIM for physical ID {track_id}")
        return WrapperFrameResult(result_instances, snapshot, output)

    def clone(self) -> "TempRMOTWrapper":
        """Make an independent frozen runtime for one query probe."""
        cloned_model = copy.deepcopy(self.model)
        self._restore_tracker_update(cloned_model)
        clone = TempRMOTWrapper(self.config, _model=cloned_model)
        clone.reset_runtime()
        return clone
