"""Targeted rank/count, full support and GT-separation regression tests."""
import json
import subprocess
import sys
import numpy as np
import pytest
from scripts.multidomain_runtime import RUN,OLD_SOURCE
from scripts.diag_attrib_v2_common import rank_within_groups
from scripts.diag_pf_countmatched import topk_ids


def test_numeric_id_ties_and_exact_counts():
    ids=[1,2,10];values=np.array([1.,1.,1.,-2.,3.]);groups=np.array([0,0,0,1,1])
    r=rank_within_groups(values,groups,2)
    assert set(np.flatnonzero(r[:3]<2))=={ids.index(i) for i in topk_ids(ids,dict(zip(ids,values[:3])),2)}
    assert (r<np.array([2,1])[groups]).tolist()==[True,True,False,False,True]


def test_kitti_full_support_and_historical_bit_identity():
    meta=json.loads((RUN/'kitti/metadata.json').read_text())
    mask=np.load(RUN/'kitti/historical.npy');idx=np.load(RUN/'kitti/historical_indices.npy')
    assert len(mask)==2425382 and int(mask.sum())==2355528 and int((~mask).sum())==69854
    full=np.load(RUN/'kitti/V2-best/complete.npy',mmap_mode='r');old=np.load(OLD_SOURCE/'FROZEN.npy',mmap_mode='r')
    assert np.array_equal(full[mask],old[idx]) and np.isfinite(full).all()
    assert len(meta['episodes'])==862 and len(meta['videos'])==4


def test_prediction_process_blocks_gt_reads():
    code="from scripts.multidomain_runtime import guard; a=guard();\ntry: open('/data3/user/TTA-RMOT-assets/flexhook/refer-kitti-v2/labels.json')\nexcept PermissionError: assert len(a)==1\nelse: raise AssertionError('GT was opened')"
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_mix_runtime_only_allows_independent_text_facade(tmp_path):
    runtime=tmp_path/'runtime';runtime.mkdir()
    safe=runtime/'SportsMOT_labels.json';safe.write_text('{"v_test":{}}')
    forbidden=tmp_path/'original'/'SportsMOT_labels.json';forbidden.parent.mkdir()
    forbidden.write_text('{"never_read":true}')
    program=f"""from scripts.multidomain_runtime import guard
from pathlib import Path
a=guard([Path({str(runtime)!r})])
assert Path({str(safe)!r}).read_text() == '{{"v_test":{{}}}}'
try: Path({str(forbidden)!r}).read_text()
except PermissionError: assert len(a)==1
else: raise AssertionError('Original Sports GT was opened')
"""
    result=subprocess.run([sys.executable,'-c',program],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_cross_source_outputs_cannot_overwrite_completed_pairs():
    from scripts.multidomain_diagnostics import base
    assert base('KITTI-V2','Dance-best')==base('KITTI-V2')
    assert base('Dance','V2-best')==base('Dance')
    assert base('KITTI-V2','LaMOT-best')!=base('KITTI-V2')
    assert base('SportsMOT','V2-best')!=base('SportsMOT','Dance-best')


def test_official_gt_completion_preserves_source_rows_and_requires_ids():
    from scripts.multidomain_complete_sports_gt import select_original_rows
    raw=['1,8,11.25,12,13,14,1,1,0.5','1,2,21,22,23,24,1,1,1','2,8,31,32,33,34,1,1,1']
    assert select_original_rows(raw,{'1':[8,2],'2':[]})==raw[:2]
    with pytest.raises(ValueError,match='absent'):
        select_original_rows(raw,{'1':[999]})


def test_evaluation_dispatch_keeps_source_separate_from_method_list(monkeypatch):
    from scripts import multidomain_diagnostics as diag
    calls=[]
    monkeypatch.setattr(diag,'evaluate',lambda target,names=None,source_model=None:
        calls.append((target,names,source_model)))
    diag.run_phase('evaluate','KITTI-V2','LaMOT-best')
    assert calls==[('KITTI-V2',None,'LaMOT-best')]


def test_exported_video_statistics_reproduce_pooled_hota_exactly():
    from scripts.report_multidomain_generality import export_hota_stats
    from scripts.analyze_pf_video_uncertainty import pooled_hota
    stats=(np.array([[3.125,2.75],[7.25,5.625]]),np.array([[13.,11.],[23.,21.]]))
    packed=json.loads(json.dumps(export_hota_stats(dict(videos=['v1','v2'],stats=stats))))
    restored=tuple(np.asarray(packed[k]) for k in ('assA_TP_numerator','TP_FN_FP_denominator'))
    weights=np.array([[1,1],[2,0],[0,2]])
    assert np.array_equal(pooled_hota(stats,weights),pooled_hota(restored,weights))
