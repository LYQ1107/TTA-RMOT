import json
import numpy as np


def test_offline_presence_distinguishes_false_absence_and_unobservable_target(tmp_path):
    from scripts.report_active import saved_presence_decisions
    root=tmp_path / 'videos/v';root.mkdir(parents=True)
    np.savez(root / 'belief.npz',tracks=[1,2],queries=['visible','missed','empty'],
        expected_counts=[.5,.5,.5],low_presence=[True,True,True],
        confirmed_present=[True,False,False],absent=[True,True,True],
        posterior=np.zeros((3,2)))
    np.save(root / 'mask.npy',np.zeros((3,6),bool))
    (root / 'questions.json').write_text(json.dumps([dict(query_id='visible',track_id=1)]))
    meta=dict(videos=dict(v=dict(observations=[(t,i) for t in (1,2,3) for i in (1,2)])))
    zero=np.zeros(6,bool)
    truth={('v','visible'):dict(labels=np.array([1,0,1,0,1,0],bool),counts=np.ones(3)),
        ('v','missed'):dict(labels=zero,counts=np.ones(3)),
        ('v','empty'):dict(labels=zero,counts=np.zeros(3))}
    rows=saved_presence_decisions(tmp_path,meta,truth,'C1','FULL',.3)
    assert rows[0]['false_absence_with_retrievable_GT'] and rows[0]['teacher_contradiction']
    assert rows[0]['matched_candidate_positive_tracks_majority']==1
    assert rows[0]['confident_negative_GT_FN']==1 and rows[0]['confident_negative_GT_FN_unqueried']==0
    assert rows[1]['confident_negative_GT_FN']==0
    assert rows[2]['confident_negative_GT_TN']==2
    assert all(row['final_belief_not_optimizer_sample_labels'] for row in rows)
    assert rows[1]['query_present_GT'] and not rows[1]['false_absence_with_retrievable_GT']
    assert not rows[2]['query_present_GT'] and rows[2]['selected_prediction_rows']==0
    assert all(row['ORACLE_GT_DIAGNOSTIC_ONLY'] and not row['GT_feedback_to_runtime'] for row in rows)
    from scripts.report_active import final_confidence_summary
    summary=final_confidence_summary(rows)[0]
    assert summary['confident_negative_GT_FN']==1 and summary['confident_negative_GT_FN_unqueried']==0
    assert summary['GTpositive_high_negative_rate']==1.0
    assert summary['high_negative_GT_error_rate']==1/6
    assert summary['high_positive_GT_precision'] is None
