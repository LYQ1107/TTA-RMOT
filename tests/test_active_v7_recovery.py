import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from flex_tta.runtime.pf_receipts import file_sha
from scripts.active_continue import reuse_evaluations
from scripts.active_v7_storage_recovery import verify_copy
from scripts.active_v7_report import sealed_diagnostic


def setup_evaluation(tmp_path,target='Dance'):
    e=SimpleNamespace(active=tmp_path/'active');j=dict(pair='C1',method='FULL',rho=.3)
    f=e.active/'C1/FULL/rho_0.3';f.mkdir(parents=True)
    (f/'predictions_seal.json').write_text('{}')
    m=dict(target=target,prediction_seal_sha256=file_sha(f/'predictions_seal.json'),GT_open_attempts=0)
    (f/'offline_summary.json').write_text(json.dumps(m))
    src=tmp_path/'old';dst=tmp_path/'new';src.mkdir();dst.mkdir()
    p=src/'evaluation_C1_FULL.json';p.write_text(json.dumps(dict(j,exit_code=0,metrics=m)))
    return e,j,f,src,dst,p


def test_reuse_closed_evaluation_without_evaluator(tmp_path):
    e,j,f,src,dst,p=setup_evaluation(tmp_path)
    rows=reuse_evaluations(e,[j],src,dst)
    row=rows['C1','FULL',.3]
    assert row['reused_closed_official_evaluation'] and row['original_receipt_sha256']==file_sha(p)
    assert json.loads((dst/p.name).read_text())['metrics']==json.loads((f/'offline_summary.json').read_text())


def test_reuse_rejects_changed_prediction(tmp_path):
    e,j,f,src,dst,p=setup_evaluation(tmp_path)
    (f/'predictions_seal.json').write_text('{"different":true}')
    with pytest.raises(ValueError,match='prediction seal'):reuse_evaluations(e,[j],src,dst)


def test_reuse_ignores_failed_evaluation(tmp_path):
    e,j,f,src,dst,p=setup_evaluation(tmp_path)
    x=json.loads(p.read_text());x['exit_code']=1;p.write_text(json.dumps(x))
    assert reuse_evaluations(e,[j],src,dst)=={}


def test_reuse_sports_requires_same_full_prediction(tmp_path):
    e,j,f,src,dst,p=setup_evaluation(tmp_path,'SportsMOT')
    s=f/'sensitivity_418';s.mkdir()
    (s/'predictions_seal.json').write_text(json.dumps(dict(original_full466_seal_sha256='different')))
    with pytest.raises(ValueError,match='sensitivity'):reuse_evaluations(e,[j],src,dst)


def test_copy_verification_checks_bytes(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    (a/'x').write_bytes(b'checkpoint');(b/'x').write_bytes(b'checkpoint')
    assert verify_copy(a,b)[0]['sha256']==file_sha(a/'x')
    (b/'x').write_bytes(b'different!')
    with pytest.raises(ValueError,match='copy changed'):verify_copy(a,b)


def test_closed_diagnostic_is_not_rewritten(tmp_path):
    e,j,f,src,dst,p=setup_evaluation(tmp_path)
    before=(f/'offline_summary.json').read_bytes()
    def forbidden(*args,**kwargs):raise AssertionError('Closed diagnostics must be reused')
    result=sealed_diagnostic(f,'Dance','offline_summary.json',forbidden,tmp_path/'public')
    assert result['target']=='Dance' and (f/'offline_summary.json').read_bytes()==before


def test_new_diagnostic_writes_only_public_cache(tmp_path):
    e,j,f,src,dst,p=setup_evaluation(tmp_path)
    (f/'offline_summary.json').unlink()
    def compute(folder,target,*,persist):
        assert persist is False
        return dict(target=target,prediction_seal_sha256=file_sha(folder/'predictions_seal.json'))
    public=tmp_path/'public'
    result=sealed_diagnostic(f,'Dance','offline_summary.json',compute,public)
    assert not (f/'offline_summary.json').exists()
    assert json.loads(next((public/'offline_diagnostics').iterdir()).read_text())==result
