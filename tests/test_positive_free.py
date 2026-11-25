import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flex_tta.adaptation.positive_free import (
    CausalRankMemory, CausalRelativeSelector, RankConfig, RankPair,
    RelativeConfig, positive_free_rank_loss,
)


def test_no_future_or_same_frame_access():
    mem = CausalRankMemory(RankConfig(min_history=2, history=4))
    mem.observe(1, {1: 2.0, 2: 0.0})
    with pytest.raises(ValueError):
        mem.observe(1, {1: 3.0})
    with pytest.raises(ValueError):
        mem.stable_pairs(1, [1, 2])
    assert not mem.stable_pairs(2, [1, 2])


def test_stable_historical_pairs_no_binary_pseudolabels():
    cfg = RankConfig(min_history=3, history=5, max_gap=2, min_separation=0.4)
    mem = CausalRankMemory(cfg)
    for t in [1, 2, 3]:
        # Both candidates can be negative; only relative order is retained.
        mem.observe(t, {7: -9.0, 8: -10.0, 9: -10.1})
    pairs = mem.stable_pairs(4, [7, 8, 9])
    assert any((p.higher, p.lower) == (7, 8) for p in pairs)
    assert not any((p.higher, p.lower) == (8, 9) for p in pairs)


def test_rank_flip_and_stale_histories_skipped():
    cfg = RankConfig(min_history=3, history=5, max_gap=2, min_separation=0.4)
    mem = CausalRankMemory(cfg)
    mem.observe(1, {1: 1.0, 2: -1.0})
    mem.observe(2, {1: -1.0, 2: 1.0})
    mem.observe(3, {1: 1.0, 2: -1.0})
    assert not mem.stable_pairs(4, [1, 2])
    assert not mem.stable_pairs(20, [1, 2])


def test_positive_free_loss_autograd_does_not_create_labels():
    cfg = RankConfig(min_history=2, rank_weight=1.0, anchor_weight=0, retention_weight=0)
    student = torch.tensor([-2.0, -1.0], requires_grad=True)
    source = torch.tensor([-2.0, -1.0])
    loss, record = positive_free_rank_loss(student, source, [1, 2], [RankPair(2, 1, 1.0, 3)], cfg)
    assert record["stable_pairs"] == 1
    loss.backward()
    assert torch.isfinite(student.grad).all()
    assert student.grad[0] > 0 and student.grad[1] < 0
    loss2, r2 = positive_free_rank_loss(student, source, [1, 2], [], cfg)
    assert r2["stable_pairs"] == 0
    assert math.isclose(float(loss2.detach()), 0)


def test_relative_selector_can_select_multiple_targets():
    selector = CausalRelativeSelector(RelativeConfig(min_history=2, median_mad_multiplier=0.8))
    rows = {1: -4.0, 2: -4.1, 3: -11.0, 4: -11.2, 5: -11.3}
    assert selector.step(1, rows) == set()
    assert selector.step(2, rows) == set()
    assert selector.step(3, rows) == {1, 2}


def test_empty_set_not_forced_and_known_positive_retained():
    sel = CausalRelativeSelector(RelativeConfig(min_history=1))
    assert sel.step(1, {1: -11.0, 2: -11.0}) == set()
    assert sel.step(2, {1: 0.1, 2: -1.1}) == {1}
    with pytest.raises(ValueError):
        sel.step(2, {1: 1.0})


def test_independent_episode_reset_prevents_memory_leak():
    selector = CausalRelativeSelector(RelativeConfig(min_history=1, median_mad_multiplier=0.8))
    rows = {1: -4.0, 2: -4.1, 3: -11.0, 4: -11.2, 5: -11.3}
    selector.step(1, rows)
    assert selector.step(2, rows)
    selector.reset()
    assert selector.step(1, rows) == set()


def test_rank_config_validation():
    with pytest.raises(ValueError):
        RankConfig(history=1, min_history=2)
    with pytest.raises(ValueError):
        RankConfig(rank_temperature=0)


def test_no_target_gt_dependency():
    # Positive-free code works using only score dictionaries, track IDs and torch.
    mem = CausalRankMemory(RankConfig(min_history=2))
    mem.observe(4, {5: -12, 7: -14})
    mem.observe(5, {5: -11, 7: -14})
    assert mem.stable_pairs(6, [5, 7])
