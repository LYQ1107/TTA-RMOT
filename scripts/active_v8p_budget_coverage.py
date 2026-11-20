"""Descriptive, GT-free coverage counts from already sealed v8P questions.

This does not simulate, select, evaluate or change any prediction. Repeated
pair questions are different from repeated cache keys: another frame/view of
the same (query, physical track) consumes another logical teacher call.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.presence.runtime import runtime_guard, target_metadata
from scripts.active_run import PAIRS

PUBLIC = ROOT / 'results/presence_active/redesign_v8p'
METHODS = ('MINUS_SLOW', 'YESTRACK_BUDGET_PRIOR', 'UNIFORM_K')
KINDS = ('GLANCE', 'PRESENCE', 'PRESENCE_COVER', 'IDENTITY', 'POSITIVE_RECHECK')


def source_folder(manifest, pair, method, rho):
    if rho == .3:
        if method != 'MINUS_SLOW':
            return Path(manifest['prior_v5_active_root']) / pair / method / 'rho_0.3'
        refs = json.loads((PUBLIC / 'inputs.json').read_text())['reused_v8']
        for ref in refs:
            if ref['pair'] == pair:
                return Path(ref['folder'])
    return Path(manifest['active_root']) / pair / method / ('rho_' + str(rho))


def summarize(manifest, pair, method, rho):
    folder = source_folder(manifest, pair, method, rho)
    seal_path = folder / 'predictions_seal.json'
    seal = json.loads(seal_path.read_text())  # required before reading questions
    assert seal['GT_open_attempts'] == 0
    target = PAIRS[pair][1]
    meta = target_metadata(target)
    counts = Counter()
    kinds = Counter()
    for video, bank in sorted(meta['videos'].items()):
        tracks = Counter(int(obs[1]) for obs in bank['observations'])
        queries = {q['expression_id'] for q in meta['episodes'] if q['video_id'] == video}
        counts['all_pairs'] += len(queries) * len(tracks)
        counts['eligible_pairs_n_ge_3'] += len(queries) * sum(n >= 3 for n in tracks.values())
        directory = folder / 'videos' / video
        receipt = json.loads((directory / 'receipt.json').read_text())
        if method == 'MINUS_SLOW':
            questions = json.loads((directory / 'questions.json').read_text())
            assert len(questions) == receipt['questions']
            physical = {(a['query_id'], int(a['track_id'])) for a in questions}
            assert all(q in queries and t in tracks for q, t in physical)
            identity = {(a['query_id'], int(a['track_id'])) for a in questions
                        if a['kind'] in ('IDENTITY', 'POSITIVE_RECHECK')}
            counts['logical_calls'] += len(questions)
            counts['unique_asked_pairs'] += len(physical)
            counts['identity_unique_pairs'] += len(identity)
            kinds.update(a['kind'] for a in questions)
        else:
            assert receipt['all_pairs'] == len(queries) * len(tracks)
            # Registered rho<=1 controls ask at most once per covered pair.
            assert receipt['actual_calls'] == receipt['covered_pairs']
            counts['logical_calls'] += receipt['actual_calls']
            counts['unique_asked_pairs'] += receipt['covered_pairs']
    assert counts['logical_calls'] == seal['teacher']['logical_calls']
    row = dict(pair=pair, target=target, method=method, rho=rho, **counts)
    row.update(repeated_pair_calls=counts['logical_calls'] - counts['unique_asked_pairs'],
               repeat_pair_fraction=(counts['logical_calls'] - counts['unique_asked_pairs']) /
                                    counts['logical_calls'],
               coverage_all_pairs=counts['unique_asked_pairs'] / counts['all_pairs'],
               calls_per_unique_pair=counts['logical_calls'] / counts['unique_asked_pairs'],
               unasked_all_pairs=counts['all_pairs'] - counts['unique_asked_pairs'],
               identity_unique_pairs=counts['identity_unique_pairs'] if method == 'MINUS_SLOW' else '',
               **{'calls_' + k: kinds[k] if method == 'MINUS_SLOW' else '' for k in KINDS},
               path=str(folder), prediction_seal_sha256=file_sha(seal_path))
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only', choices=list(PAIRS))
    a = p.parse_args()
    opened = runtime_guard()
    manifest = json.loads((PUBLIC / 'execution.json').read_text())
    pairs = [a.check_only] if a.check_only else list(PAIRS)
    rhos = (1.,) if a.check_only else (.1, .3, 1.)
    methods = ('MINUS_SLOW', 'UNIFORM_K') if a.check_only else METHODS
    rows = [summarize(manifest, pair, method, rho)
            for pair in pairs for rho in rhos for method in methods]
    assert not opened
    if a.check_only:
        print(json.dumps(dict(rows=rows, GT_open_attempts=0, files_written=False)), flush=True)
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (PUBLIC / 'budget_coverage.csv').open('w') as output:
        writer = csv.DictWriter(output, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(PUBLIC / 'budget_coverage_inputs.json', dict(
        rows=len(rows), GT_open_attempts=0, teacher_calls=0, model_calls=0,
        descriptive_only=True, affects_acceptance_gate=False,
        all_pairs_definition='query_count times all physical tracker IDs, including length<3',
        eligible_pairs_definition='query_count times physical tracks of length>=3',
        unique_definition='distinct (video, query ID, physical track ID), any question kind',
        repeated_pair_not_duplicate_input='another frame/view of the same pair is another logical call',
        controls_definition='at most one question per covered pair; YT may cover short tracks',
        inputs=[{k: r[k] for k in ('pair', 'method', 'rho', 'path', 'prediction_seal_sha256')}
                for r in rows]))
    print(json.dumps(dict(status='SEALED_BUDGET_COVERAGE_CLOSED', rows=len(rows),
                          GT_open_attempts=0, new_teacher_or_model_calls=0)), flush=True)


if __name__ == '__main__':
    main()
