from flex_tta.adaptation.score import CausalStabilityTTA, FrozenScoreAdapter
from flex_tta.mot_interface.trackers import merge_kitti_class_splits
from flex_tta.plugin import FlexHookPluginConfig, FlexHookReferringPlugin
from flex_tta.types import TrackRow, TrackerCandidate


def test_frozen_adapter_is_stateless():
    adapter = FrozenScoreAdapter()
    first = adapter.step(0, {3: 0.4})
    second = adapter.step(1, {3: -0.2})
    assert first.adapted_margins == {3: 0.4}
    assert second.adapted_margins == {3: -0.2}
    assert second.pseudo_updates == 0


def test_causal_adapter_uses_only_persistent_previous_frame():
    adapter = CausalStabilityTTA(lr=1e-2)
    initial = adapter.step(0, {1: 2.0, 2: -2.0})
    updated = adapter.step(1, {1: 2.0, 2: -2.0})
    gap = adapter.step(3, {1: 2.0})
    assert initial.scale == 1.0
    assert updated.pseudo_updates == 2
    assert updated.scale != 1.0
    assert gap.pseudo_updates == 0


def test_kitti_class_merge_serializes_pedestrian_offset(tmp_path):
    car = tmp_path / "car.txt"
    pedestrian = tmp_path / "pedestrian.txt"
    car.write_text("1,1,10,20,30,40,1,-1,-1,-1\n", encoding="utf-8")
    pedestrian.write_text("1,1,11,21,31,41,1,-1,-1,-1\n", encoding="utf-8")
    rows = merge_kitti_class_splits(car, pedestrian)
    assert [row.track_id for row in rows] == [1, 2]
    assert rows[1].values[1] == 2


def test_plugin_keeps_tracker_geometry_and_identity():
    row = TrackRow(1, 7, (1.0, 7.0, 2.0, 3.0, 4.0, 5.0))
    candidate = TrackerCandidate("0005", 0, 7, row, 0.5)
    plugin = FlexHookReferringPlugin(FlexHookPluginConfig(adapter="frozen"))
    _, selected = plugin.step(0, [candidate])
    assert selected[0].track_id == 7
    assert selected[0].row.values == row.values
