"""GT-free CPU simulation under a kernel runtime-file read whitelist."""
import gzip
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.search_feasibility_common import *
from scripts import decompose_common as dc


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    require_prereg(); attempts = dc.guard(); start = time.perf_counter()
    prepared = json.loads((RUN / 'runtime/seal.json').read_text()); path = checked(prepared['runtime'])
    if prepared['GT_open_attempts']: raise PermissionError('Unsafe runtime preparation')
    dest = RUN / 'search'; dest.mkdir(parents=True, exist_ok=True)
    isolation = dc.sandbox([path], [dest])
    total = lists = 0; output = dest / 'predictions.jsonl.gz'
    with gzip.open(output, 'wt') as stream:
        for r in json_rows(path):
            lists += 1
            for model in MODELS:
                order = ranking(r['margins'][model], r['track_ids'])
                margins = [r['margins'][model][i] for i in order]
                for verifier in TEACHERS:
                    answers = [r['probabilities'][verifier][i] >= .5 for i in order]
                    runs = suite(answers, margins, r['subseed'])
                    pba = next(x for x in runs if x['method'] == 'PBA_0.2')
                    out = dict(list_id=r['list_id'], target=r['target'], video=r['video'], model=model,
                        verifier=verifier, track_order=[r['track_ids'][i] for i in order], runs=runs,
                        C_dense=dense_conflict(answers), C_online=online_conflict(pba['asked_positions'], pba['answers']))
                    stream.write(json.dumps(out, separators=(',', ':'), allow_nan=False) + '\n'); total += len(runs)
            if lists % 500 == 0: print(json.dumps(dict(event='SEARCH_PROGRESS', lists=lists, runs=total)), flush=True)
    if attempts: raise PermissionError('Search attempted GT access')
    result = dict(status='GT_FREE_SEARCH_SEALED_BEFORE_GT', predictions=record(output), runtime=prepared['runtime'],
        lists=lists, runs=total, rankers=3, teacher_verifiers=3, methods=list(METHODS), GT_open_attempts=0,
        isolation=isolation, teacher_calls=0, model_calls=0, optimizer_updates=0, GPU_hours=0,
        CPU_wall_seconds=time.perf_counter()-start, cache=boundary_search.cache_info()._asdict())
    dump(dest / 'seal.json', result)
    print(json.dumps(dict(event='GT_FREE_SEARCH_SEALED', lists=lists, runs=total, GT_open_attempts=0,
                         wall_seconds=result['CPU_wall_seconds'])), flush=True)


if __name__ == '__main__': main()
