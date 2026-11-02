"""Private stdio service: original Qwen env, fixed crops/prompts, no GT IO."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def send(value): print('ACTIVE_RESPONSE\t'+json.dumps(value), flush=True)


def main():
    p = ArgumentParser(description=__doc__); p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True); a = p.parse_args()
    from scripts.multidomain_runtime import guard
    access = guard()
    videos = json.loads(a.runtime.read_text())
    from flex_uda.active.teacher_client import LocalTeacher
    teacher = LocalTeacher(videos, a.cache, ROOT)
    send(dict(status='READY'))
    for line in sys.stdin:
        request = json.loads(line)
        if request['command'] == 'CLOSE':
            result = teacher.close()
            if access: raise RuntimeError('Teacher opened target GT')
            send(dict(status='SEALED', receipt=dict(result, GT_open_attempts=len(access))))
            return
        answers = teacher.ask_many(request['requests'])
        if access: raise RuntimeError('Teacher opened target GT')
        send(dict(status='ANSWERS', answers=answers, costs=teacher.stats))
    raise RuntimeError('Caller exited without requesting the frozen-weight seal')


if __name__ == '__main__': main()
