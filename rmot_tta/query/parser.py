"""Rule-only parser for the small Refer-KITTI language subset.

This is not a general semantic parser.  Unknown, nested, OR, and negated
forms remain explicitly unsupported instead of being rewritten by string
replacement.
"""
from __future__ import annotations

import re

from .query_spec import QuerySpec


_APPEARANCE = ("black", "red", "silver", "light color", "light-colored", "white")
_SPATIAL = ("left", "right", "front", "behind", "horizon")
_DYNAMIC = ("moving", "parking", "turning", "braking", "same direction", "counter direction")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower().rstrip("."))


def parse_query(text: str) -> QuerySpec:
    original = str(text).strip()
    clean = _clean(original)
    if not clean:
        return QuerySpec(original, None, None, (), (), "UNKNOWN", None, None, unsupported_reason="empty query")
    unsupported: str | None = None
    if re.search(r"\b(or|either|neither|not|more than|less than|between)\b", clean):
        unsupported = "logical_or_negation_or_comparison_not_in_safe_subset"
    constraints: list[str] = []
    kinds: list[str] = []
    for phrase in _APPEARANCE:
        if phrase in clean:
            constraints.append(phrase)
            kinds.append("appearance")
    for phrase in _SPATIAL:
        if re.search(rf"\b{re.escape(phrase)}\b", clean):
            constraints.append(phrase)
            kinds.append("spatial")
    for phrase in _DYNAMIC:
        if phrase in clean:
            constraints.append(phrase)
            kinds.append("dynamic")
    if not constraints and unsupported is None:
        unsupported = "no_supported_slot"
    subject_match = re.search(r"\b(car|cars|vehicle|vehicles|person|people|pedestrian|pedestrians)\b", clean)
    subject = subject_match.group(1) if subject_match else None
    reference_object = "camera" if "camera" in clean or "ours" in clean else None
    if "same direction" in clean or "counter direction" in clean:
        reference_frame = "ego_direction"
    elif any(slot in clean for slot in ("left", "right", "front", "behind")):
        reference_frame = "image_or_ego_unspecified"
    else:
        reference_frame = None
    if any(kind == "dynamic" for kind in kinds):
        temporal_scope = "current_frame_short_history"
    else:
        temporal_scope = "current_frame"
    logical_structure = "AND" if len(set(constraints)) > 1 else "ATOM"
    if unsupported is not None:
        logical_structure = "UNKNOWN"
    return QuerySpec(
        original_text=original,
        subject=subject,
        reference_object=reference_object,
        constraints=tuple(dict.fromkeys(constraints)),
        constraint_types=tuple(dict.fromkeys(kinds)),
        logical_structure=logical_structure,
        reference_frame=reference_frame,
        temporal_scope=temporal_scope,
        supported_rules=tuple(f"slot:{slot}" for slot in dict.fromkeys(constraints)),
        unsupported_reason=unsupported,
    )
