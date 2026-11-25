"""A conservative whitelist of referring-query rewrites.

The source expressions contain many class, color, activity, and quantifier
phrases.  This first benchmark release rewrites only local directional
prepositions or removes a redundant positional participle.  It deliberately
does not touch ``car/vehicle`` synonyms, motion/parking words, quantifiers,
negation, or left/right polarity.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any


_UNSAFE = re.compile(
    r"\b(?:no|not|without|all|both|each|any|one|two|three|first|second|"
    r"moving|parked|car|cars|automobile|automobiles|vehicle|vehicles)\b",
    flags=re.IGNORECASE,
)


def _token_hash(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", text.lower())
    return hashlib.sha256("\x1f".join(tokens).encode("utf-8")).hexdigest()


def _canonical(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _candidate_rules(text: str) -> list[tuple[str, str, str]]:
    if _UNSAFE.search(text):
        return []
    rules: list[tuple[str, str, str]] = []
    replacements = (
        (r"\bon the (left|right) side\b", r"to the \1 side", "direction_on_to_side"),
        (r"\bto the (left|right) side\b", r"on the \1 side", "direction_to_on_side"),
        (r"\blocated on the (left|right)\b", r"on the \1", "drop_located_direction"),
        (r"\bpositioned on the (left|right)\b", r"on the \1", "drop_positioned_direction"),
    )
    for pattern, replacement, rule_id in replacements:
        changed = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
        changed = _canonical(changed)
        if changed != _canonical(text):
            rules.append((rule_id, changed, "conservative_local_direction"))
    return rules


def generate_safe_rewrites(text: str) -> list[dict[str, Any]]:
    """Generate deduplicated, auditable equivalent query variants."""
    original = _canonical(text)
    original_hash = _token_hash(original)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rule_id, rewritten, semantic_rule in _candidate_rules(original):
        if rewritten in seen or rewritten == original:
            continue
        seen.add(rewritten)
        rewritten_hash = _token_hash(rewritten)
        rows.append({
            "variant_id": rule_id,
            "query_text": rewritten,
            "rewrite_rule": semantic_rule,
            "original_token_hash": original_hash,
            "rewrite_token_hash": rewritten_hash,
            "token_hash_equal": original_hash == rewritten_hash,
            "semantic_equivalence_status": "conservative_equivalent",
            "review_status": "rule_whitelist_reviewed",
        })
    return rows
