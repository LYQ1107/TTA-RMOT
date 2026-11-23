"""Remove only JSON whitespace outside strings from two legacy statistics.

No statistic encoding, number spelling, key, ordering, type or string changes.
The immutable published Git version and a local byte backup are retained.
This is publication housekeeping, not a scientific transformation.
"""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BACKUP = Path('/var/tmp/liuyeqiang-tta-rmot-a100-20261122/publication_whitespace_originals')
FILES = ('docs/a100_visual_classical_20261121/three_seed_clipped_sufficient_statistics.json',
         'docs/a100_visual_classical_20261121/three_seed_task_valid_sufficient_statistics.json')


def tokens(data):
    result = bytearray(); quoted = escaped = False
    for byte in data:
        if quoted:
            result.append(byte)
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            result.append(byte); quoted = True
        elif byte not in b' \t\r\n':
            result.append(byte)
    assert not quoted and not escaped
    return bytes(result)


def equal_types(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_types(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(equal_types(x, y) for x, y in zip(a, b))
    return a == b


def main():
    ref = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    rows = []
    assert tokens(b' { "s": "a b", "v": 1e-3 } \n') == b'{"s":"a b","v":1e-3}'
    for name in FILES:
        path = ROOT/name
        assert path.resolve().is_relative_to(ROOT) and not path.is_symlink()
        original = path.read_bytes()
        assert original == subprocess.check_output(['git', 'show', ref+':'+name])
        destination = BACKUP/name; destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert destination.read_bytes() == original
        else:
            destination.write_bytes(original)
        compact = tokens(original)+b'\n'
        assert tokens(original) == tokens(compact)
        assert equal_types(json.loads(original), json.loads(compact))
        path.write_bytes(compact)
        rows.append(dict(file=name, original_published_ref=ref, original_local_backup=str(destination),
            before_bytes=len(original), after_bytes=len(compact),
            before_sha256=hashlib.sha256(original).hexdigest(),
            after_sha256=hashlib.sha256(compact).hexdigest(),
            all_nonwhitespace_tokens_and_string_bytes_identical=True,
            all_values_and_numeric_types_identical=True))
    report = dict(status='PASS_WHITESPACE_ONLY_LEGACY_STATISTICS_PUBLICATION', files=rows,
        saved_bytes=sum(r['before_bytes']-r['after_bytes'] for r in rows),
        raw_aggregate_encoding_unchanged=True, original_commits_and_NAS_unchanged=True,
        GT_features_predictions_models_read=0,
        reason='Keep current necessary code and exact experiment evidence below100MB; no statistic repacking or result selection.',
        restore='Use gitshow at the immutable original ref to a NEW LOCAL destination; the compact publication has identical JSON tokens and values.')
    path = ROOT/'docs/legacy_archives/whitespace_only_statistics_20261122.json'
    path.write_text(json.dumps(report, separators=(',', ':'))+'\n')
    print(json.dumps(dict(status=report['status'], saved_bytes=report['saved_bytes'])))


if __name__ == '__main__':
    main()
