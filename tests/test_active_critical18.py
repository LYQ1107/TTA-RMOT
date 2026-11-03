from collections import Counter


def test_additional18_is_exact_registered_scope_without_repeating_completed_ten():
    from scripts.active_prepare_critical18 import additional_jobs
    selected=additional_jobs()
    assert Counter(j['pair'] for j in selected)=={'C2':4,'C4':5,'C5':4,'C6':5}
    assert all(j['rho']==.3 for j in selected)
    assert all(j['pair'] not in ('C2','C5') for j in selected if j['method']=='MINUS_SLOW')
