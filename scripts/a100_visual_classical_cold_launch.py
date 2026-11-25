"""Durable physical cold-clip run, followed by isolated official evaluation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.a100_visual_classical_prepare import OUT
from flex_tta.runtime.pf_receipts import atomic_json


if __name__=='__main__':
    begin=time.perf_counter();name='C4_first_full_clip_seed0';directory=OUT/'cold_live'/name
    with (OUT/'supervisor/recovery_cold_full_clip.log').open('a',buffering=1) as log:
        code=subprocess.call([sys.executable,'scripts/a100_visual_classical_cold_slice.py','--name',name],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    assert code==0,('cold generation failed',code)
    from scripts.a100_mechanism_offline import evaluate
    receipt=json.loads((directory/'completion.json').read_text());assert receipt['GT_open_attempts']==0
    results={arm:evaluate(directory/arm,'C4',subset=True) for arm in ('SOFT_KD','COHORT','HARD_KD')}
    atomic_json(directory/'official_evaluation.json',dict(status='PASS_REAL_COLD_CLIP_AND_POST_SEAL_OFFICIAL',
        results=results,wall_seconds=time.perf_counter()-begin,
        scope='one full deterministically selected C4 clip, not whole-C4 HOTA or cost',
        shared_prefix_physically_measured_once=True,no_independent_cold_arm_replication=True))
