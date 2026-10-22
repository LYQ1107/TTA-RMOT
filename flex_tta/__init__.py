"""Tracker-agnostic FlexHook integration and source-free score adapters.

The official FlexHook model remains in ``vendor/FlexHook`` at its pinned
commit.  This package owns only the interface around tracker candidate banks,
causal score adaptation, and evidence/metric bookkeeping.
"""

from .types import TrackerCandidate, TrackRow

__all__ = ["TrackerCandidate", "TrackRow"]
