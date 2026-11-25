#!/usr/bin/env python3
"""Stream a recovery log into the existing blocking experiment output pipe."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--log',type=Path,required=True)
    p.add_argument('--status',type=Path,required=True);p.add_argument('--controller-pid',type=int,required=True)
    a=p.parse_args()
    with a.log.open() as handle:
        while True:
            line=handle.readline()
            if line:
                print('RECOVERY_SHARD02 '+line.rstrip(),flush=True);continue
            status=json.loads(a.status.read_text()) if a.status.exists() else {}
            if status.get('status','').startswith(('COMPLETE_','FAILED_')):
                print('RECOVERY_SHARD02 '+json.dumps(dict(controller_status=status['status'])),flush=True);break
            stat=Path(f'/proc/{a.controller_pid}/stat')
            if not stat.exists() or stat.read_text().rpartition(')')[2].split()[0]=='Z':
                print('RECOVERY_SHARD02 CONTROLLER_EXITED_CHECK_SAVED_STATUS',flush=True);break
            time.sleep(1)


if __name__=='__main__':main()
