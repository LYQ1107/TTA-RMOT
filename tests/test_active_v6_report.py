import pytest

from scripts.active_v6_report import rescore_seconds,adaptation_coverage


def test_frozen_yestrack_has_no_model_rescore_phase():
    assert rescore_seconds({'LoRA_GPU_seconds':0.},'YESTRACK_BUDGET_PRIOR')==0.
    assert rescore_seconds({'rescore_GPU_seconds':1.25},'YESTRACK_BUDGET_PRIOR')==1.25


@pytest.mark.parametrize('method',['FULL','MINUS_SLOW'])
def test_current_model_methods_require_measured_frame_cache_time(method):
    assert rescore_seconds({'rescore_GPU_seconds':3.5},method)==3.5
    with pytest.raises(KeyError):
        rescore_seconds({},method)


def test_coverage_reports_skips_and_the_final_after_slow_calibration():
    fast=dict(slope=1.,temperature=1.,bias=0.,identity={'status':'SKIP'},presence={'status':'SKIP'})
    updated=dict(slope=.17,temperature=.6,bias=-.25,identity={'status':'MAP_UPDATED'},presence={'status':'MAP_UPDATED'})
    receipts=[dict(video='first',rounds=[dict(positive_pairs=9,slow={'optimizer_updates':50},fast=dict(fast,after_slow=updated))]),
              dict(video='second',rounds=[dict(positive_pairs=0,slow={'optimizer_updates':0},fast=fast)])]
    rows,summary=adaptation_coverage(receipts,'C1','FULL_v6')
    assert rows[0]['final_w']==.17 and rows[0]['identity_fit_calls']==1
    assert rows[0]['new_video_single_answer_positive_upper_bound']>.95
    assert rows[1]['initial_w']==.17 and rows[1]['new_video_single_answer_positive_upper_bound']<.95
    assert summary['new_videos_single_answer_cannot_certify_positive']==1
    assert summary['optimizer_updates']==50 and summary['zero_slow_update_videos']==1
    assert summary['rounds_without_positive_pseudo_pairs']==1 and summary['videos']==2
    negative_only=[dict(video='negative_only',rounds=[dict(positive_pairs=0,slow={'optimizer_updates':50},fast=fast)])]
    _,negative=adaptation_coverage(negative_only,'C3','FULL_v6')
    assert negative['zero_positive_optimizer_updates']==50 and negative['zero_positive_update_rounds']==1
