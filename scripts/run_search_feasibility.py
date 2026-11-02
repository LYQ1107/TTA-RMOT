"""One blocking CPU parent per phase; never polls GPUs or experiment processes."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.search_feasibility_common import PUBLIC, RUN, dump, require_prereg


def main(phase):
    require_prereg(); start = time.perf_counter()
    dest = RUN / 'launchers' / (phase + '_' + str(time.time_ns())); dest.mkdir(parents=True)
    if phase == 'finalize':
        snapshot = RUN / 'offline/initial_aggregation'; snapshot.mkdir(exist_ok=True)
        for name in ('summary.csv','verdicts.json','bootstrap.json','costs.json','validation.json'):
            if not (snapshot/name).exists(): shutil.copy2(PUBLIC/name,snapshot/name)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1')
    names = (['prepare_search_feasibility', 'simulate_search_feasibility'] if phase == 'search' else
        ['summarize_search_feasibility', 'validate_search_feasibility', 'report_search_feasibility'] if phase == 'finalize' else
        ['evaluate_search_feasibility', 'summarize_search_feasibility', 'validate_search_feasibility', 'report_search_feasibility'])
    results = []
    for name in names:
        begin = time.perf_counter(); log = dest / (name + '.log')
        with log.open('w') as stream:
            process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'scripts' / (name + '.py'))],
                cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                stream.write(line); stream.flush(); print(line.rstrip(), flush=True)
            code = process.wait()
        row = dict(stage=name, exit_code=code, wall_seconds=time.perf_counter()-begin, log=str(log))
        results.append(row); dump(dest / 'status.json', dict(phase=phase, stages=results))
        if code: raise RuntimeError('Stop at first actionable failure: ' + str(log))
    dump(dest / 'completion.json', dict(phase=phase, stages=results, wall_seconds=time.perf_counter()-start,
                                       teacher_calls=0, model_calls=0, GPU_hours=0))


if __name__ == '__main__':
    p = ArgumentParser(); p.add_argument('--phase', choices=('search', 'offline', 'finalize'), required=True)
    main(p.parse_args().phase)
