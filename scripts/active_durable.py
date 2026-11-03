"""Wait once for the existing queue without binding GPUs to a terminal session.

The detached manager records its own child exit code. An observer can attach
without launching another queue; all resource/protocol checks remain inside
active_continue and active_launch.
"""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json


def observe(state,manager=None):
    launch=json.loads((state/'launch.json').read_text())
    with (state/'coordinator.log').open() as stream:
        while True:
            text=stream.read()
            if text:print(text,end='',flush=True)
            if (state/'completion.json').exists():
                result=json.loads((state/'completion.json').read_text())
                rest=stream.read()
                if rest:print(rest,end='',flush=True)
                print(json.dumps(dict(event='DURABLE_QUEUE_FINISHED',**result)),flush=True)
                return result['exit_code']
            # A missing observation handle is not a reason to restart work.
            # Here the independently registered manager PID is checked inside
            # this single blocking observer, never by agent-level polling.
            try:os.kill(launch['manager_pid'],0)
            except ProcessLookupError:
                raise RuntimeError('Manager exited without a completion receipt: '+str(state))
            if manager is not None and manager.poll() is not None:
                raise RuntimeError('Manager exited without a completion receipt: '+str(state))
            time.sleep(2)


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--jobs-json',type=Path)
    p.add_argument('--state-dir',type=Path)
    p.add_argument('--manager',action='store_true')
    p.add_argument('--attach',type=Path)
    a=p.parse_args()
    if a.attach:return observe(a.attach)
    if not a.jobs_json:p.error('Supply the same published --jobs-json plan')
    if a.manager:
        result=subprocess.run([sys.executable,str(ROOT/'scripts/active_continue.py'),
            '--jobs-json',str(a.jobs_json)],cwd=ROOT)
        atomic_json(a.state_dir/'completion.json',dict(exit_code=result.returncode,
            manager_pid=os.getpid(),finished_time_ns=time.time_ns()))
        return result.returncode
    from flex_uda.active.execution import authorize_active
    execution,_=authorize_active()
    state=execution.launchers/('durable_'+str(time.time_ns()));state.mkdir()
    with (state/'coordinator.log').open('x') as stream:
        manager=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--manager',
            '--state-dir',str(state),'--jobs-json',str(a.jobs_json)],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(state/'launch.json',dict(manager_pid=manager.pid,plan=str(a.jobs_json),
        launched_time_ns=time.time_ns(),terminal_session_independent=True))
    print(json.dumps(dict(event='DURABLE_QUEUE_STARTED',manager_pid=manager.pid,state_dir=str(state))),flush=True)
    return observe(state,manager)


if __name__=='__main__':sys.exit(main())
