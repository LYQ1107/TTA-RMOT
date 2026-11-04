import pytest

from scripts.active_v6_report import rescore_seconds


def test_frozen_yestrack_has_no_model_rescore_phase():
    assert rescore_seconds({'LoRA_GPU_seconds':0.},'YESTRACK_BUDGET_PRIOR')==0.
    assert rescore_seconds({'rescore_GPU_seconds':1.25},'YESTRACK_BUDGET_PRIOR')==1.25


@pytest.mark.parametrize('method',['FULL','MINUS_SLOW'])
def test_current_model_methods_require_measured_frame_cache_time(method):
    assert rescore_seconds({'rescore_GPU_seconds':3.5},method)==3.5
    with pytest.raises(KeyError):
        rescore_seconds({},method)
