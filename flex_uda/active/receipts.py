"""Wall-time ledger for real worker attempts, including resumable failures."""
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import time
from flex_tta.runtime.pf_receipts import atomic_json


@contextmanager
def record_attempt(folder, *, phase):
    directory = Path(folder) / 'worker_attempts'; directory.mkdir(parents=True, exist_ok=True)
    path = directory / (str(time.time_ns())+'.json'); start = time.perf_counter()
    receipt = dict(status='RUNNING', phase=phase, pid=os.getpid(),
        physical_gpu=os.environ.get('PRESENCE_PHYSICAL_GPU'), gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
        wall_time_is_not_utilization_integral=True)
    atomic_json(path, receipt)
    try:
        yield
        receipt['status'] = 'RETURNED_SUCCESSFULLY'
    except BaseException as exc:
        receipt.update(status='FAILED_RETAINED_FOR_RECOVERY', error=repr(exc))
        raise
    finally:
        receipt['wall_seconds'] = time.perf_counter()-start
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            receipt['parent_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        receipt['teacher_peak_is_reported_in_its_separate_receipt'] = True
        atomic_json(path, receipt)
