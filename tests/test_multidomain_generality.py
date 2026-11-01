"""Targeted rank/count, full support and GT-separation regression tests."""
import json
import subprocess
import sys
import numpy as np
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
