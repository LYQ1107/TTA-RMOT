from dataclasses import dataclass, field
from enum import Enum


class EvidenceState(str, Enum):
    SUPPORT_POS='SUPPORT_POS'
    SUPPORT_NEG='SUPPORT_NEG'
    UNKNOWN='UNKNOWN'


@dataclass(frozen=True)
class Attribute:
    kind: str
    text: str
    view: str


@dataclass(frozen=True)
class ParsedExpression:
    original: str
    kind: str
    attributes: tuple[Attribute,...]
    notes: tuple[str,...]=()


@dataclass
class EvidenceDecision:
    state: EvidenceState
    probability: float
    reasons: list[str]=field(default_factory=list)
    features: dict=field(default_factory=dict)
