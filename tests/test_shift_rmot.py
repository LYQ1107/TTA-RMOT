from __future__ import annotations

import numpy as np

from rmot_tta.data.language_rewrites import generate_safe_rewrites
from rmot_tta.data.visual_corruptions import apply_visual_shift


def test_visual_shifts_are_deterministic_and_geometry_preserving():
    image = np.arange(32 * 48 * 3, dtype=np.uint8).reshape(32, 48, 3)
    for shift_id, severity in (
        ("identity", "identity"), ("fog_reduced_contrast", "medium"),
        ("low_light_sensor_noise", "medium"), ("motion_blur", "medium"),
        ("jpeg_compression", "medium"),
    ):
        first = apply_visual_shift(image, shift_id, severity, "unit-seed")
        second = apply_visual_shift(image, shift_id, severity, "unit-seed")
        assert first.shape == image.shape
        assert first.dtype == np.uint8
        assert np.array_equal(first, second)


def test_language_rewrite_whitelist_does_not_change_direction_polarity_or_unsafe_words():
    assert generate_safe_rewrites("people on the left side")
    assert generate_safe_rewrites("cars on the left side") == []
    for row in generate_safe_rewrites("people located on the right"):
        assert "left" not in row["query_text"]
        assert "right" in row["query_text"]
        assert row["semantic_equivalence_status"] == "conservative_equivalent"
