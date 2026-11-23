from __future__ import annotations

import numpy as np

from rmot_tta.data.shift_loader import _expand
from rmot_tta.data.visual_corruptions import apply_visual_shift


def _condition(*, family: str, query: str, visual: str, severity: str, seed: str) -> dict:
    return {
        "dataset": "shift_rmot_v2",
        "video_id": "0005",
        "base_expression_id": "base",
        "condition_id": f"base__{family}",
        "runtime_expression_id": f"base__{family}",
        "condition_family": family,
        "query_text": query,
        "original_query_text": "people on the left",
        "frame_indices": [1, 2],
        "runtime_frame_indices": [0, 1, 2, 3],
        "evaluation_frame_indices": [1, 2],
        "image_root": "/tmp/shift-rmot-test-images",
        "image_suffix": ".png",
        "expression_path": "/tmp/shift-rmot-test-expression.json",
        "label_root": "/tmp/shift-rmot-test-labels",
        "label_format": "cxcywh",
        "visual_shift": visual,
        "visual_severity": severity,
        "shift_seed_basis": seed,
    }


def test_runtime_uses_full_video_axis_but_evaluation_uses_label_range():
    condition = _condition(family="A1", query="people on the left", visual="low_light_sensor_noise",
                           severity="medium", seed="ShiftRMOT-v2|global_seed=1|video=0005|shift=low_light_sensor_noise|severity=medium")
    runtime = _expand(condition, evaluator=False)
    evaluation = _expand(condition, evaluator=True)
    assert [row["frame_index"] for row in runtime] == [0, 1, 2, 3]
    assert [row["frame_index"] for row in evaluation] == [1, 2]
    assert all(row["frame_role"] == "runtime_video_axis" for row in runtime)
    assert all(row["frame_role"] == "evaluation_label_range" for row in evaluation)


def test_visual_seed_is_shared_by_a1_and_a3_but_not_expression_text():
    seed = "ShiftRMOT-v2|global_seed=20261021|video=0005|shift=low_light_sensor_noise|severity=medium"
    a1 = _expand(_condition(family="A1", query="people on the left", visual="low_light_sensor_noise",
                            severity="medium", seed=seed), evaluator=False)
    a3 = _expand(_condition(family="A3", query="people located on the left", visual="low_light_sensor_noise",
                            severity="medium", seed=seed), evaluator=False)
    assert [row["shift_seed_basis"] for row in a1] == [row["shift_seed_basis"] for row in a3]
    image = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
    for left, right in zip(a1, a3):
        left_image = apply_visual_shift(image, left["visual_shift"], left["visual_severity"], left["shift_seed_basis"])
        right_image = apply_visual_shift(image, right["visual_shift"], right["visual_severity"], right["shift_seed_basis"])
        assert np.array_equal(left_image, right_image)


def test_identity_pixels_are_shared_by_a0_and_a2():
    seed = "ShiftRMOT-v2|global_seed=20261021|video=0005|shift=identity|severity=identity"
    a0 = _expand(_condition(family="A0", query="people on the left", visual="identity",
                            severity="identity", seed=seed), evaluator=False)
    a2 = _expand(_condition(family="A2", query="people located on the left", visual="identity",
                            severity="identity", seed=seed), evaluator=False)
    image = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
    for left, right in zip(a0, a2):
        left_image = apply_visual_shift(image, left["visual_shift"], left["visual_severity"], left["shift_seed_basis"])
        right_image = apply_visual_shift(image, right["visual_shift"], right["visual_severity"], right["shift_seed_basis"])
        assert np.array_equal(left_image, right_image)
