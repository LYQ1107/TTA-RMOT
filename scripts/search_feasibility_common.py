"""Fixed CPU search primitives. No model, teacher, image or GT imports/IO."""
from functools import lru_cache
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'results/search_feasibility'
RUN = ROOT / 'results/runs/search_feasibility_20261102'
TARGETS = ('KITTI-V2', 'SportsMOT', 'Dance')
MODELS = ('V2-best', 'Dance-best', 'LaMOT-best')
IN_DOMAIN = {'KITTI-V2': 'V2-best', 'SportsMOT': 'LaMOT-best', 'Dance': 'Dance-best'}
TEACHERS = ('TEACHER_CROP', 'TEACHER_CLIP', 'TEACHER_FRAMEBOX')
METHODS = ('DENSE', 'BISECT_EXACT', 'PBA_0.1', 'PBA_0.2', 'PBA_0.3',
           'PBA_PRESENCE_0.1', 'PBA_PRESENCE_0.2', 'PBA_PRESENCE_0.3',
           'RANDOM_SAME_BUDGET', 'NATIVE')
RAW_SHAS = ('acc506f5a7c2959e8322ee568d5d0e249f401e5a343b662effb24a2eb34a01f7',
            'c1448ce28167c0edd325b278794814e877e43528cfc9ec38292ca42eba686a97')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def record(path):
    return dict(path=str(path), sha256=sha(path), bytes=Path(path).stat().st_size)


def checked(rec):
    if sha(rec['path']) != rec['sha256']: raise ValueError('Changed sealed input: ' + rec['path'])
    return Path(rec['path'])


def require_prereg():
    proof = json.loads((PUBLIC / 'prereg_publication.json').read_text())
    if not proof['remote_verified'] or sha(PUBLIC / 'PREREG.md') != proof['prereg_sha256']:
        raise PermissionError('Publish unchanged preregistration before computation')
    return proof


def json_rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        for line in stream:
            if line.strip(): yield json.loads(line)


def csv_rows(path, rows):
    rows = iter(rows); first = next(rows)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'wt', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first), lineterminator='\n')
        writer.writeheader(); writer.writerow(first); writer.writerows(rows)


def list_key(target, video, frame, query):
    return json.dumps([target, video, int(frame), query], ensure_ascii=False, separators=(',', ':'))


def sub_seed(key):
    return int.from_bytes(hashlib.sha256(('0|' + key).encode()).digest()[:4], 'big')


def ranking(margins, track_ids):
    return sorted(range(len(track_ids)), key=lambda j: (-margins[j], track_ids[j]))


def online_conflict(positions, answers):
    # Each later answer contributes at most once, even if many earlier answers disagree.
    count = 0
    for k, (i, y) in enumerate(zip(positions, answers)):
        count += any((j < i and not a and y) or (j > i and a and not y)
                     for j, a in zip(positions[:k], answers[:k]))
    return count / len(positions) if positions else 0.


def dense_conflict(answers):
    p, n = sum(answers), len(answers) - sum(answers)
    if not p or not n: return None
    before_no = inversions = 0
    for y in answers:
        if y: inversions += before_no
        else: before_no += 1
    return inversions / (p * n)


@lru_cache(maxsize=100000)
def boundary_search(answers, kind, epsilon='0.2', presence=False):
    """One-indexed question positions; integer likelihoods preserve exact ties."""
    n = len(answers); asked = []
    if kind == 'BISECT_EXACT':
        lo, hi = 0, n
        while lo != hi:
            i = (lo + hi + 1) // 2; asked.append(i)
            if answers[i-1]: lo = i
            else: hi = i-1
        return lo, tuple(asked), None
    high, low = {'0.1': (9, 1), '0.2': (4, 1), '0.3': (7, 3)}[epsilon]
    weights = [n if presence else 1] + [1] * n
    used = set()
    while max(weights) * 10 < sum(weights) * 9 and len(used) < n:
        total = sum(weights); tail = total - weights[0]
        best = None
        for i in range(1, n+1):
            if i not in used:
                candidate = (abs(2 * tail - total), i)
                if best is None or candidate < best: best = candidate
            tail -= weights[i]
        i = best[1]; asked.append(i); used.add(i); y = answers[i-1]
        weights = [w * (high if ((b >= i) == y) else low) for b, w in enumerate(weights)]
    b = max(range(n+1), key=lambda b: weights[b])
    return b, tuple(asked), weights[b] / sum(weights)


def suite(answers, margins_sorted, seed):
    """Use only cached Boolean answers. No adaptive access to unasked answers."""
    answers = tuple(bool(y) for y in answers); n = len(answers); result = []
    pba_budget = None
    for method in METHODS:
        posterior = None
        if method == 'DENSE':
            asked = list(range(1, n+1)); selected = [i for i in asked if answers[i-1]]; b = len(selected)
        elif method == 'NATIVE':
            asked = []; selected = [i+1 for i, m in enumerate(margins_sorted) if m > 0]; b = len(selected)
        elif method == 'RANDOM_SAME_BUDGET':
            asked = (np.random.RandomState(seed).permutation(n)[:pba_budget] + 1).tolist()
            b = min(range(n+1), key=lambda b: sum((i <= b) != answers[i-1] for i in asked))
            selected = list(range(1, b+1))
        else:
            kind = 'BISECT_EXACT' if method == 'BISECT_EXACT' else 'PBA'
            b, asked, posterior = boundary_search(answers, kind, method.rsplit('_', 1)[-1], 'PRESENCE' in method)
            asked = list(asked); selected = list(range(1, b+1))
            if method == 'PBA_0.2': pba_budget = len(asked)
        answer_trace = [answers[i-1] for i in asked]
        result.append(dict(method=method, m=len(asked), b_hat=b, selected_positions=selected,
                           asked_positions=asked, answers=answer_trace, posterior_max=posterior))
    return result


def verify_search():
    seal = json.loads((RUN / 'search/seal.json').read_text())
    if seal['status'] != 'GT_FREE_SEARCH_SEALED_BEFORE_GT' or seal['GT_open_attempts']:
        raise ValueError('Complete GT-free search before independent GT evaluation')
    checked(seal['predictions']); checked(seal['runtime'])
    return seal
