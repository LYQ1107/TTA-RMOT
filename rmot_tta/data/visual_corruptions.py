"""Deterministic, geometry-preserving visual shifts for ShiftRMOT.

The benchmark stores no transformed image copies.  A row records a shift name,
severity, and seed basis; this module materializes the RGB image only at
runtime.  The transforms intentionally avoid geometric changes so the original
KITTI boxes remain valid for evaluation.
"""
from __future__ import annotations

import hashlib
from typing import Any

import cv2
import numpy as np


_SEVERITY_SCALE = {"light": 0.5, "medium": 1.0}


def _seed(seed_basis: str) -> int:
    digest = hashlib.sha256(str(seed_basis).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32)


def visual_condition_registry() -> dict[str, Any]:
    """Return the public transform registry used by the manifest builder."""
    return {
        "format": "shift-rmot-visual-conditions-v1",
        "geometry_preserving": True,
        "conditions": [
            {"shift_id": "identity", "severity": "identity", "family": "A0", "description": "original RGB image"},
            {"shift_id": "fog_reduced_contrast", "severity": "light", "family": "visual", "description": "white veil and reduced contrast"},
            {"shift_id": "fog_reduced_contrast", "severity": "medium", "family": "visual", "description": "stronger white veil and reduced contrast"},
            {"shift_id": "low_light_sensor_noise", "severity": "light", "family": "visual", "description": "underexposure with sensor noise"},
            {"shift_id": "low_light_sensor_noise", "severity": "medium", "family": "visual", "description": "strong underexposure with sensor noise"},
            {"shift_id": "motion_blur", "severity": "light", "family": "visual", "description": "horizontal motion blur"},
            {"shift_id": "motion_blur", "severity": "medium", "family": "visual", "description": "strong horizontal motion blur"},
            {"shift_id": "jpeg_compression", "severity": "light", "family": "visual", "description": "moderate JPEG recompression"},
            {"shift_id": "jpeg_compression", "severity": "medium", "family": "visual", "description": "strong JPEG recompression"},
        ],
    }


def _fog_reduced_contrast(image: np.ndarray, severity: str) -> np.ndarray:
    scale = _SEVERITY_SCALE[severity]
    alpha = 0.18 + 0.28 * scale
    contrast = 1.0 - (0.16 + 0.22 * scale)
    shifted = image.astype(np.float32) * contrast + 255.0 * alpha
    return np.clip(shifted, 0.0, 255.0).astype(np.uint8)


def _low_light_sensor_noise(image: np.ndarray, severity: str, seed_basis: str) -> np.ndarray:
    scale = _SEVERITY_SCALE[severity]
    rng = np.random.default_rng(_seed(seed_basis))
    brightness = 0.68 - 0.24 * scale
    sigma = 4.0 + 10.0 * scale
    shifted = image.astype(np.float32) * brightness
    shifted += rng.normal(0.0, sigma, size=image.shape).astype(np.float32)
    return np.clip(shifted, 0.0, 255.0).astype(np.uint8)


def _motion_blur(image: np.ndarray, severity: str) -> np.ndarray:
    kernel_size = 7 if severity == "light" else 13
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)


def _jpeg_compression(image: np.ndarray, severity: str) -> np.ndarray:
    quality = 62 if severity == "light" else 35
    encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1]
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape != image.shape:
        raise RuntimeError("JPEG shift changed image shape or failed to decode")
    return decoded


def apply_visual_shift(image: np.ndarray, shift_id: str = "identity", severity: str = "identity",
                       seed_basis: str = "") -> np.ndarray:
    """Apply one registered shift to an RGB uint8 image.

    ``seed_basis`` is part of the row-level protocol, so repeated evaluation of
    the same condition is bitwise deterministic for stochastic noise.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3 image, got {image.shape}")
    if image.dtype != np.uint8:
        raise TypeError(f"expected uint8 image, got {image.dtype}")
    if shift_id == "identity":
        return image.copy()
    if severity not in _SEVERITY_SCALE:
        raise ValueError(f"unsupported visual severity: {severity}")
    if shift_id == "fog_reduced_contrast":
        shifted = _fog_reduced_contrast(image, severity)
    elif shift_id == "low_light_sensor_noise":
        shifted = _low_light_sensor_noise(image, severity, seed_basis)
    elif shift_id == "motion_blur":
        shifted = _motion_blur(image, severity)
    elif shift_id == "jpeg_compression":
        shifted = _jpeg_compression(image, severity)
    else:
        raise ValueError(f"unsupported visual shift: {shift_id}")
    if shifted.shape != image.shape:
        raise AssertionError(f"visual shift changed shape from {image.shape} to {shifted.shape}")
    return shifted
