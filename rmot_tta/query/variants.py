"""Conservative equivalent/counterfactual query generation."""
from __future__ import annotations

import re

from .parser import parse_query
from .query_spec import QuerySpec, QueryVariant, Relation


def _canonical_spacing(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _counterfactual(text: str, spec: QuerySpec) -> QueryVariant | None:
    # These substitutions preserve the slot and are mutually exclusive only
    # for the explicitly spatial/directional pairs below.  Color and class
    # words are intentionally not treated as complements.
    pairs = (
        (r"\bleft\b", "right", "image_left_vs_right"),
        (r"\bright\b", "left", "image_right_vs_left"),
        (r"same direction", "counter direction", "ego_same_vs_counter_direction"),
        (r"counter direction", "same direction", "ego_counter_vs_same_direction"),
    )
    for pattern, replacement, rule_id in pairs:
        if re.search(pattern, text, flags=re.IGNORECASE):
            changed = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
            if changed != text:
                return QueryVariant(
                    query_type="q_cf",
                    text=_canonical_spacing(changed),
                    relation=Relation.MUTUALLY_EXCLUSIVE,
                    rule_id=rule_id,
                    changed_slot="spatial_relation",
                    explanation="same object/time/reference frame cannot occupy both registered directional slots",
                )
    return None


def generate_variants(text: str) -> tuple[QuerySpec, tuple[QueryVariant, ...]]:
    spec = parse_query(text)
    if spec.unsupported_reason is not None:
        q0 = QueryVariant("q0", spec.original_text, Relation.UNKNOWN, "original", None, "original query")
        return spec, (q0,)
    # Whitespace normalization is the only universally safe equivalent in the
    # first release.  It is useful for auditing while making no semantic claim
    # about car/vehicle, color, or action synonyms.
    eq_text = _canonical_spacing(spec.original_text)
    q_eq = QueryVariant("q_eq", eq_text, Relation.EQUIVALENT, "canonical_spacing", None,
                        "whitespace and terminal punctuation normalization only")
    variants = [QueryVariant("q0", spec.original_text, Relation.EQUIVALENT, "original", None, "original query"), q_eq]
    counterfactual = _counterfactual(spec.original_text, spec)
    if counterfactual is not None:
        variants.append(counterfactual)
    return spec, tuple(variants)
