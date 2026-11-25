"""Attach to an existing durable queue, compact transport only (never launch)."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.active_durable import observe


class EventStream:
    def __init__(self, output, interval=30):
        self.output=output;self.interval=interval;self.buffer='';self.rows=0
        self.last=None;self.last_emit=time.monotonic()

    def emit(self,line):
        self.output.write(line+'\n');self.output.flush()

    def progress(self,force=False):
        if self.rows and (force or time.monotonic()-self.last_emit>=self.interval):
            self.emit(json.dumps(dict(event='OBSERVER_PROGRESS',suppressed_rows=self.rows,
                last_unlabelled_row=self.last,raw_log_retained=True,
                parallel_rows_not_attributed_to_a_method=True)))
            self.rows=0;self.last_emit=time.monotonic()

    def write(self,text):
        self.buffer+=text
        while '\n' in self.buffer:
            line,self.buffer=self.buffer.split('\n',1)
            try:row=json.loads(line)
            except (json.JSONDecodeError,TypeError):
                if line:self.emit(line)
                continue
            if isinstance(row,dict) and row.get('event') and row['event']!='V6_FRAME_CACHE_PROGRESS':
                self.progress(force=True);self.emit(line)
            else:
                self.rows+=1
                self.last={k:row[k] for k in ('video','video_index','round','actual_questions','event','completed_frames','total_frames') if k in row} if isinstance(row,dict) else None
                self.progress()
        return len(text)

    def flush(self):
        self.output.flush()

    def finish(self):
        if self.buffer:self.emit(self.buffer);self.buffer=''
        self.progress(force=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attach',type=Path,required=True);a=p.parse_args()
    launch=json.loads((a.attach/'launch.json').read_text())
    print(json.dumps(dict(event='OBSERVER_ATTACHED',state_dir=str(a.attach),
        manager_pid=launch['manager_pid'],queue_restarted=False,
        raw_log=str(a.attach/'coordinator.log'))),flush=True)
    writer=EventStream(sys.stdout)
    try:
        with redirect_stdout(writer):return observe(a.attach)
    finally:writer.finish()


if __name__=='__main__':sys.exit(main())
