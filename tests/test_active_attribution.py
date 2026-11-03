import numpy as np
from scripts.active_attribution import selection_changes


def test_removed_duplicate_FP_can_change_which_unchanged_row_matches():
    # Before: row0 wins. Removing it makes the unchanged row1 a TP.
    change=selection_changes(np.array([[.9],[.8]]),[0,1],[1])
    assert change['removed_false_positive_rows']==0
    assert change['unchanged_rows_removed_FP']==1
    assert change['baseline_false_positive_rows']==1
    assert change['adapted_false_positive_rows']==0
    assert change['recovered_gt']==change['lost_gt']==0


def test_new_TP_can_displace_existing_TP_without_recovering_GT():
    change=selection_changes(np.array([[.8],[.9]]),[0],[0,1])
    assert change['new_false_positive_rows']==0
    assert change['unchanged_rows_new_FP']==1
    assert change['recovered_gt']==0
    assert change['adapted_false_positive_rows']==1


def test_recovery_loss_and_empty_GT():
    change=selection_changes(np.eye(2),[0],[1])
    assert change['recovered_gt']==1 and change['lost_gt']==1
    empty=selection_changes(np.empty((2,0)),[],[0,1])
    assert empty['new_false_positive_rows']==2 and empty['adapted_matched_gt']==0
