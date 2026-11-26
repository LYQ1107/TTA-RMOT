"""Additional strong/common-decoder baselines, without regenerating inference."""
import concurrent.futures
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_evaluate_group import job


if __name__=='__main__':
    for pair in ('C4','C1'):
        while not (OUT/'strict_prefix'/pair/'completion.json').is_file():time.sleep(15)
    tasks=[('strict_prefix',pair,method) for method in ('COVER__FRAME_STRUCTURED','COVER__DIRECT_NATIVE','COVER__LOCAL_GRAPH') for pair in ('C4','C1')]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(job,tasks))
    atomic_json(OUT/'strict_prefix/additional_baselines.json',dict(status='PASS_ADDITIONAL_STRONG_BASELINES',results=results))
