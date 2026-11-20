import json


def test_only_finished_official_receipts_are_published(tmp_path):
    from scripts.active_stage_snapshot import completed_evaluations
    from flex_tta.runtime.pf_receipts import file_sha
    job=dict(pair='C2',method='FULL',rho=.3)
    pending=dict(pair='C4',method='FULL',rho=.3)
    model=tmp_path/'model';model.mkdir()
    seal=model/'predictions_seal.json';seal.write_text('{}')
    metrics=dict(path=str(model),prediction_seal_sha256=file_sha(seal),GT_open_attempts=0,HOTA=12.3)
    receipt=tmp_path/'evaluation_C2_FULL.json'
    receipt.write_text(json.dumps(dict(job,exit_code=0,metrics=metrics)))
    completed,failed=completed_evaluations(dict(jobs=[job,pending]),tmp_path)
    assert len(completed)==1 and completed[0]['pair']=='C2' and not failed
    assert completed[0]['HOTA']==12.3
