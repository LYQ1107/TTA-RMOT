"""Task-local two-GPU assignment; retired idle cards cannot start new work."""
import json
import os
import time
from flex_uda.presence.runtime import RUN


def admitted_device(minimum_free_mib=0):
    from scripts.select_gpu import query
    assignment = RUN / 'resources/current.json'
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES')
    while True:
        if assignment.exists():
            selected = json.loads(assignment.read_text())['selected']
            if uuid not in {g['uuid'] for g in selected}:
                raise RuntimeError('TASK_GPU_RETIRED: no new CUDA context on '+str(uuid))
        current = next(g for g in query() if g['uuid'] == uuid)
        if current['free_mib'] >= minimum_free_mib: return current
        time.sleep(20)
