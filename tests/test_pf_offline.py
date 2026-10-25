"""Small assignment checks; no real GT or experiment is opened by these tests."""
import numpy as np
from scripts.analyze_pf_full_offline import matched_gt, overlaps, auroc, load_eval_gt, query_presence_summary, observable_id_switches


def test_assignment_counts_duplicates_as_false_positives():
    gt, candidates = matched_gt(np.array([[.9], [.8]]), [0, 1])
    assert gt == {0} and candidates == {0}


def test_assignment_maximizes_valid_match_cardinality_first():
    gt, candidates = matched_gt(np.array([[.99, .51], [.51, .49]]), [0, 1])
    assert len(gt) == len(candidates) == 2


def test_rank_metric_and_empty_boxes():
    assert auroc([0., 1., 2.], [0, 1, 1]) == 1.
    assert overlaps([(0., 0., 1., 1.)], []).shape == (1, 0)


def test_offline_gt_counts_only_official_scored_pedestrians(tmp_path):
    path = tmp_path / "gt_template/video/expression/gt.txt"
    path.parent.mkdir(parents=True)
    path.write_text("1,1,0,0,10,20,1,1,1\n1,2,0,0,10,20,0,1,1\n1,3,0,0,10,20,1,2,1\n")
    assert load_eval_gt(tmp_path, "video", "expression") == {1: [(1, (0., 0., 10., 20.))]}


def test_query_presence_keeps_empty_episodes_and_distinguishes_fp_definitions():
    fields = ("gt_observations", "baseline_matched_gt", "adapted_matched_gt", "baseline_false_positive_rows",
              "adapted_false_positive_rows", "recovered_gt", "lost_gt", "new_false_positive_rows",
              "removed_false_positive_rows", "baseline_selected_rows", "adapted_selected_rows")
    present = dict.fromkeys(fields, 0)
    present.update(gt_observations=10, baseline_matched_gt=2, adapted_matched_gt=3,
                   baseline_selected_rows=2, adapted_selected_rows=4, adapted_false_positive_rows=1)
    empty = dict.fromkeys(fields, 0)
    empty.update(adapted_selected_rows=5, adapted_false_positive_rows=5, new_false_positive_rows=4)
    result = query_presence_summary([present, empty, dict.fromkeys(fields, 0)])
    assert result["query_present"]["adapted_GT_recall"] == .3
    assert result["query_empty"]["episodes"] == 2
    assert result["query_empty"]["adapted_nonempty_prediction_rate"] == .5
    assert result["query_empty"]["adapted_GT_recall"] is None
    assert result["query_empty"]["net_false_positive_change"] == 5
    assert result["query_empty"]["new_false_positive_rows"] == 4


def test_video_bootstrap_pools_counts_not_macro_hota():
    from scripts.analyze_pf_video_uncertainty import sufficient, pooled_hota, paired_clusters
    rows = [dict(AssA=np.array([1.]), HOTA_TP=np.array([1.]), HOTA_FP=np.array([0.]), HOTA_FN=np.array([0.])),
            dict(AssA=np.array([.25]), HOTA_TP=np.array([9.]), HOTA_FP=np.array([0.]), HOTA_FN=np.array([0.]))]
    values = sufficient(rows)
    assert np.isclose(pooled_hota(values,np.ones(2)),100*np.sqrt(3.25/10))
    assert not np.isclose(pooled_hota(values,np.ones(2)),75.)
    counts=np.array([[2,0],[0,2],[1,1]])
    assert np.allclose(paired_clusters(values,values,counts),0.)


def test_id_switch_diagnostic_does_not_invent_labels_for_unknown_or_stale_ids():
    previous={1:(2,10),2:(2,None),3:(0,20)}
    assert observable_id_switches(previous,{1:11,2:12,3:21},3) == (1,1)
    assert observable_id_switches(previous,{1:10},3) == (1,0)
