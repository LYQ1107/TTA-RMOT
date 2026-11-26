"""Small numerical/protocol sanity checks, not experiment acceptance proxies."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.diag_attrib_audit import expected_rows, prediction


def test_native_mot_comparison_handles_official_float_format_and_order(tmp_path):
    path = tmp_path / 'predict.txt'
    path.write_text('2.0,7.0,1.0,2.0,3.0,4.0,1.0,-1.0,-1.0,-1.0\n'
                    '1.0,2.0,5.0,6.0,7.0,8.0,1.0,-1.0,-1.0,-1.0\n')
    tracks = {1: {2: '1,2,5,6,7,8,1,-1,-1,-1'}, 2: {7: '2,7,1,2,3,4,1,-1,-1,-1'}}
    assert prediction(path) == expected_rows(tracks, [(1, 2), (2, 7)])
    # A genuine geometry or ID change must still fail equality.
    assert prediction(path) != expected_rows({1: {2: '1,2,5,6,7,9,1,-1,-1,-1'}, **{2: tracks[2]}}, [(1, 2), (2, 7)])


def test_duplicate_candidate_is_not_silently_collapsed(tmp_path):
    path = tmp_path / 'predict.txt'
    path.write_text('1,2,5,6,7,8,1,-1,-1,-1\n' * 2)
    with pytest.raises(ValueError, match='duplicate'):
        prediction(path)


def test_gt_access_audit_catches_direct_and_symlink_reads(tmp_path):
    dataset = tmp_path / 'target'
    dataset.mkdir()
    truth = dataset / 'gt.txt'
    truth.write_text('not a runtime input')
    alias = tmp_path / 'ordinary_name.txt'
    alias.symlink_to(truth)
    code = ('from pathlib import Path; import scripts.diag_attrib_audit as a; '
            f'a.DATA=Path({str(dataset)!r}); attempts=a.deny_gt(); '
            '\nfor p in ' + repr([str(truth), str(alias)]) + ':\n'
            ' try: Path(p).read_text()\n'
            ' except PermissionError: pass\n'
            ' else: raise AssertionError("GT read unexpectedly succeeded")\n'
            'assert len(attempts)==2\n')
    result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
