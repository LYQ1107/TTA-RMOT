import pytest

from scripts.active_v6_close import fp_residual


@pytest.mark.parametrize('new,removed,residual,net', [(121396, 76666, 7, 44737),
                                                  (44094, 221161, -272, -177339)])
def test_fp_accounting_includes_unchanged_row_reassignment(new, removed, residual, net):
    row = dict(new_false_positive_rows=new, removed_false_positive_rows=removed,
               unchanged_rows_new_FP=max(residual, 0), unchanged_rows_removed_FP=max(-residual, 0),
               net_FP_change=net)
    assert fp_residual(row) == residual


def test_fp_accounting_does_not_silently_drop_reassignment():
    with pytest.raises(AssertionError):
        fp_residual(dict(new_false_positive_rows=20, removed_false_positive_rows=10,
                        unchanged_rows_new_FP=1, unchanged_rows_removed_FP=0, net_FP_change=10))
