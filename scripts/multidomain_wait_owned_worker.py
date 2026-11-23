"""One blocking completion wait for the surviving owned inference worker."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.multidomain_runtime import RUN,PUBLIC,require_plan
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def main():
    p=ArgumentParser();p.add_argument('--pid',type=int,required=True);p.add_argument('--model',required=True)
    p.add_argument('--launcher',type=Path,required=True);a=p.parse_args();require_plan()
    command=Path('/proc',str(a.pid),'cmdline').read_bytes().replace(b'\0',b' ').decode()
    if str(ROOT/'scripts/multidomain_kitti_scores.py') not in command or '--worker '+a.model not in command:
        raise ValueError('Wait target is not the owned experiment worker')
    seal=RUN/'kitti'/a.model/'scores_seal.json'
    print(json.dumps(dict(event='WAIT_OWNED_WORKER_WITHOUT_RESTART',pid=a.pid,model=a.model)),flush=True)
    while not seal.exists():
        try:os.kill(a.pid,0)
        except ProcessLookupError:raise RuntimeError('Owned worker exited without score seal; inspect original log')
        state=Path('/proc',str(a.pid),'stat').read_text().split(') ',1)[1][0]
        if state=='Z':raise RuntimeError('Owned worker became zombie without score seal')
        time.sleep(15)
    r=json.loads(seal.read_text())
    if r['status']!='COMPLETE_SCORES_SEALED' or r['GT_open_attempts'] or file_sha(r['complete_path'])!=r['complete_sha256']:
        raise ValueError('Surviving worker completion failed integrity')
    record=dict(status='COMPLETE_AFTER_ORCHESTRATOR_SIGTERM_WORKER_SURVIVED',jobs=[
        dict(phase='full',model=a.model,physical_gpu=int(r['physical_gpu']),exit_code=0,pid=a.pid,
            wall_seconds=r['wall_seconds'],wall_measurement='worker-entry to seal; supervising shell lifetime not used',
            stdout=str(a.launcher/(a.model+'_full.txt')))],
        supervisor_interrupted_but_worker_not_restarted=True,score_seal_path=str(seal),score_seal_sha256=file_sha(seal))
    atomic_json(a.launcher/'launcher_status.json',record)
    print(json.dumps(dict(event='OWNED_WORKER_SEAL_VERIFIED',model=a.model,complete_rows=r['complete_rows'],seconds=r['wall_seconds'])),flush=True)


if __name__=='__main__':main()
