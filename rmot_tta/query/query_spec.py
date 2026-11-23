"""Structured, deliberately conservative query metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Relation(str, Enum):
    EQUIVALENT = "EQUIVALENT"
    MUTUALLY_EXCLUSIVE = "MUTUALLY_EXCLUSIVE"
    ENTAILMENT = "ENTAILMENT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class QueryVariant:
    query_type: str
    text: str
    relation: Relation
    rule_id: str
    changed_slot: str | None
    explanation: str


@dataclass(frozen=True)
class QuerySpec:
    original_text: str
    subject: str | None
    reference_object: str | None
    constraints: tuple[str, ...]
    constraint_types: tuple[str, ...]
    logical_structure: str
    reference_frame: str | None
    temporal_scope: str | None
    supported_rules: tuple[str, ...] = ()
    unsupported_reason: str | None = None
    variants: tuple[QueryVariant, ...] = field(default_factory=tuple)
