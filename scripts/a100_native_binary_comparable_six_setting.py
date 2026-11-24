"""Prospective effective-view correction with exact unaffected-setting reuse.

Train only C2/C5, preserve the original Trust study, and reuse the four
settings where the critic's full metadata input is provably unchanged.
All 318 output records seal before this study reads target annotations.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts import a100_native_binary_six_setting as original
from scripts import a100_native_binary_trust_six_setting as parent

PAIRS, PUBLIC = original.PAIRS, original.PUBLIC
PARENT_OUT, PARENT_PLAN = parent.OUT, parent.PLAN
OUT = PARENT_OUT.parent/'native_binary_comparable_quality_20261124'
PLAN = PUBLIC/'native_binary_comparable_quality_registration_20261124.json'
STATE = OUT/'supervisor/state.json'
PRIMARY, RECIPES, METHODS, COMPARATORS = parent.PRIMARY, parent.RECIPES, parent.METHODS, parent.COMPARATORS
AFFECTED, REUSED = ('C2', 'C5'), ('C1', 'C3', 'C4', 'C6')
CROSS_NAMES = ('CORRECTED_AT_PARENT_COUNT', 'PARENT_AT_CORRECTED_COUNT')
EXPECTED = 318  # 312 own methods + four new count masks + two parent primaries.


def check_plan():
    plan = json.loads(PLAN.read_text())
    parent.check_plan()
    assert plan['neural_seeds'] == [0] and plan['primary'] == PRIMARY
    assert plan['recipes'] == RECIPES and plan['methods'] == list(METHODS)
    assert plan['official_records'] == EXPECTED
    assert plan['parent_registration_sha256'] == file_sha(PARENT_PLAN)
    for name, sha in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == sha, name
    audit = json.loads((PUBLIC/'native_binary_effective_views_20261124.json').read_text())
    assert file_sha(PUBLIC/'native_binary_effective_views_20261124.json') == plan['metadata_audit_sha256']
    assert all(audit['settings'][p]['probe_effective_view_mismatches'] == 0 for p in REUSED)
    assert sum(audit['settings'][p]['probe_effective_view_mismatches'] for p in AFFECTED) == 80
    return plan


def bind_scope():
    original.OUT, original.PRIMARY, original.RECIPES = OUT, PRIMARY, RECIPES
    original.METHODS, original.COMPARATORS = METHODS, COMPARATORS
    original.registration = check_plan


def worker(pair):
    assert pair in AFFECTED
    check_plan(); bind_scope()
    from flex_uda.active import native_binary_curriculum as curriculum
    from flex_uda.active.rsi_probe_conditioning import comparable_paid_probe
    from flex_uda.active.rsi_policy_distribution import recorded_selection, gated_update, trust_update
    base_probe, base_credit = curriculum.probe_measure, curriculum.progress_reward
    credit = {}

    def probe(model, paid, *args):
        eligible, conditioning = comparable_paid_probe(paid)
        value, summary = base_probe(model, eligible, *args)
        if summary is not None:
            summary = dict(summary, comparable_paid_probe=conditioning)
        return value, summary

    def measure(*args):
        credit.clear(); credit.update(base_credit(*args)); return dict(credit)

    def update(*args, **kwargs):
        return gated_update(trust_update, credit, *args, **kwargs)

    curriculum.probe_measure, curriculum.progress_reward = probe, measure
    curriculum.select_round, curriculum.update_challenger = recorded_selection, update
    original.worker(pair)


def counts(pair):
    assert pair in AFFECTED
    bind_scope(); original.counts(pair)


def amendment_counts(pair):
    """Original float64 ranking and ID tie break across the two cohorts."""
    from flex_uda.a100.mechanism import metadata, generation_guard
    from flex_uda.presence.predictions import seal_mask
    guard = generation_guard()
    root, before = OUT/pair/'seed0', PARENT_OUT/pair/'seed0'
    destination = root/'probe_amendment_counts'
    if (destination/'completion.json').is_file():
        return
    source, target = PAIRS[pair]; md = metadata(target)
    masks = {name: np.zeros(md['candidate_query_rows'], dtype=bool) for name in CROSS_NAMES}
    for video, bank in sorted(md['videos'].items()):
        obs = np.asarray(bank['observations'])
        queries = [q for q in md['episodes'] if q['video_id'] == video]
        folders = {'CORRECTED': root, 'PARENT': before}
        scores = {n: np.load(p/'videos'/video/(PRIMARY+'_scores.npy'), mmap_mode='r') for n, p in folders.items()}
        bits = {n: np.load(p/PRIMARY/'videos'/video/'mask.npy') for n, p in folders.items()}
        for frame in np.unique(obs[:, 0]):
            ids = np.flatnonzero(obs[:, 0] == frame)
            for qi, query in enumerate(queries):
                order = {n: ids[np.lexsort((obs[ids, 1], -s[qi, ids]))] for n, s in scores.items()}
                k = {n: int(b[qi, ids].sum()) for n, b in bits.items()}
                for n in folders:
                    assert np.array_equal(np.sort(order[n][:k[n]]), ids[bits[n][qi, ids]])
                for name, rank, donor in ((CROSS_NAMES[0], 'CORRECTED', 'PARENT'),
                                          (CROSS_NAMES[1], 'PARENT', 'CORRECTED')):
                    masks[name][query['start']+order[rank][:k[donor]]] = True
    methods = []
    for name, mask in masks.items():
        seal = seal_mask(destination/name, md, mask, target=target, model=source, method=name,
            GT_open_attempts=len(guard), extra=dict(counts_equal_per_query_frame=True,
                counts_from_GT=False, diagnostic_not_deployment=True,
                CountSelf_identical_for_both_sources=True, parent_cohort=str(PARENT_OUT), primary=PRIMARY))
        methods.append(dict(method=name, rows=seal['prediction_rows']))
    assert not guard
    atomic_json(destination/'completion.json', dict(methods=methods, GT_open_attempts=0,
        CountSelf_identical_for_both_sources=True))


def raw_ordinary_gate():
    fits = 0; maximum = 0.
    for pair in PAIRS:
        raw = json.loads((OUT/pair/'seed0/completion.json').read_text())
        assert raw['GT_open_attempts'] == 0
        for video in raw['receipts']:
            for name, report in video['fits'].items():
                if RECIPES[name]['family'] == 'structured':
                    continue
                for fit in report['fits']:
                    assert fit['teacher_fit_gate_passed'] and fit['excess_risk'] <= .005
                    fits += 1; maximum = max(maximum, fit['excess_risk'])
    atomic_json(OUT/'ordinary_fit_gate_before_GT.json', dict(fits=fits, maximum_excess_risk=maximum,
        GT_open_attempts=0, distinct_risk_units_preserved=True))


def known_gpu_workers():
    known = set()
    for path in (PARENT_OUT/'supervisor/state.json', original.STATE):
        if path.is_file():
            for job in json.loads(path.read_text()).get('active', {}).values():
                known.add(int(job['pid']))
    return known


def main():
    plan = check_plan(); start = time.perf_counter()
    pending = list(AFFECTED); active = {}; finished = []; failure = None

    def state(status):
        view = {}
        for gpu, (proc, handle, job) in active.items():
            p = Path(job['folder'])/'progress.json'
            view[str(gpu)] = dict(job, progress=json.loads(p.read_text()) if p.is_file() else {})
        atomic_json(STATE, dict(status=status, pid=os.getpid(), active=view, pending=pending,
            finished=finished, failure=failure, wall_seconds=time.perf_counter()-start,
            neural_seeds=[0], goal_status='ACTIVE'))
        print(json.dumps(dict(status=status, active=view, pending=pending, failure=failure)), flush=True)

    STATE.parent.mkdir(parents=True, exist_ok=True)
    while pending or active:
        for gpu, (proc, handle, job) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            handle.close(); del active[gpu]
            if code == 0 and (Path(job['folder'])/'completion.json').is_file():
                finished.append(job)
            else:
                failure = dict(job=job, exit_code=code, error_tail=Path(job['log']).read_text()[-2400:])
        for pair in list(pending):
            if (OUT/pair/'seed0/completion.json').is_file():
                pending.remove(pair); finished.append(dict(pair=pair, reused_completed_correction=True))
        if pending and not failure:
            # Assignment preflight only: share the user's known small-head
            # jobs; do not allocate alongside any unknown resident process.
            known = known_gpu_workers(); candidates = []
            for gpu in (0, 1, 2, 3):
                if gpu in active:
                    continue
                used = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                    '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
                pids = {int(p) for p in subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                    '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).split()}
                if used < 8192 and pids <= known:
                    candidates.append((len(pids), used, gpu, sorted(pids)))
            for _, used, gpu, residents in sorted(candidates):
                if not pending:
                    break
                pair = pending.pop(0); log = OUT/'supervisor'/(pair+'_seed0.log')
                handle = log.open('a', buffering=1)
                proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--pair', pair], cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                             OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2'))
                job = dict(pair=pair, pid=proc.pid, physical_GPU=gpu, folder=str(OUT/pair/'seed0'),
                    log=str(log), preflight_memory_MiB=used, preserved_known_resident_PIDs=residents)
                active[gpu] = proc, handle, job
        state('PRESERVING_HEALTHY_CORRECTION_WORKERS_AFTER_FAILURE' if failure else 'RUNNING_AFFECTED_SETTINGS_EFFECTIVE_VIEW_CORRECTION_NO_GT')
        if failure and not active:
            raise RuntimeError(json.dumps(failure))
        if pending or active:
            time.sleep(30)
    # Reuse is justified by a no-op on every possible reserved probe, not
    # by a favorable HOTA, checkpoint, confidence, or approximate tolerance.
    while not all((PARENT_OUT/p/'seed0/countmatched/completion.json').is_file() for p in PAIRS):
        state('WAITING_PARENT_COUNTS_FOR_EXACT_UNAFFECTED_REUSE_NO_GT')
        time.sleep(30)
    reuse = {}
    for pair in REUSED:
        source, destination = PARENT_OUT/pair, OUT/pair
        if destination.is_symlink():
            assert destination.resolve() == source.resolve()
        else:
            assert not destination.exists(); destination.symlink_to(source, target_is_directory=True)
        reuse[pair] = dict(parent_path=str(source), raw_completion_sha256=file_sha(source/'seed0/completion.json'),
            count_completion_sha256=file_sha(source/'seed0/countmatched/completion.json'),
            reason='All reserved effective views match; the new critic filter returns its original input object.')
    atomic_json(OUT/'exact_reuse.json', reuse)
    raw_ordinary_gate()
    state('ALL_CORRECTED_RAW_SEALED_GENERATING_DUAL_AND_AMENDMENT_COUNTS_NO_GT')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(counts, AFFECTED)); list(pool.map(amendment_counts, AFFECTED))
    tasks, new_tasks, inherited = [], [], {}
    for pair in PAIRS:
        root = OUT/pair/'seed0'
        for folder in (root, root/'countmatched'):
            doc = json.loads((folder/'completion.json').read_text()); assert doc['GT_open_attempts'] == 0
            for method in doc['methods']:
                task = (str(folder/method['method']), pair); tasks.append(task)
                if pair in AFFECTED:
                    new_tasks.append(task)
                else:
                    inherited[task[0]] = str((PARENT_OUT/pair/'seed0'/folder.relative_to(root))/method['method'])
        if pair in AFFECTED:
            extra = root/'probe_amendment_counts'; doc = json.loads((extra/'completion.json').read_text())
            assert doc['GT_open_attempts'] == 0
            new_tasks.extend((str(extra/r['method']), pair) for r in doc['methods'])
            tasks.extend((str(extra/r['method']), pair) for r in doc['methods'])
            parent_primary = str(PARENT_OUT/pair/'seed0'/PRIMARY)
            tasks.append((parent_primary, pair)); inherited[parent_primary] = parent_primary
    assert len(tasks) == len(set(tasks)) == EXPECTED and len(new_tasks) == 108 and len(inherited) == 210
    atomic_json(OUT/'all_predictions_sealed_before_GT.json', dict(methods=EXPECTED, raw_methods=108,
        CountSelf_aliases=108, new_official_evaluations=108, exactly_reused_official_records=210,
        GT_open_attempts=0, primary=PRIMARY, cohort=OUT.name, source_sha256=plan['source_sha256']))
    state('ALL_318_CORRECTION_RECORDS_SEALED_ISOLATED_OFFICIAL_EVALUATION')
    from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        results = dict(pool.map(evaluate_one, new_tasks))
    while not (PARENT_OUT/'official.json').is_file():
        state('WAITING_COMPLETED_PARENT_OFFICIAL_FOR_EXACT_REUSE')
        time.sleep(30)
    parent_official = json.loads((PARENT_OUT/'official.json').read_text())
    assert parent_official['methods'] == 312
    for alias, source in inherited.items():
        results[alias] = parent_official['results'][source]
    assert len(results) == EXPECTED
    atomic_json(OUT/'official.json', dict(methods=EXPECTED, results=results, exactly_reused_records=210,
        new_official_evaluations=108, parent_official_sha256=file_sha(PARENT_OUT/'official.json')))
    state('ANALYZING_COMPLETE_EFFECTIVE_VIEW_CORRECTION')
    for script in ('scripts/a100_native_binary_comparable_analyze.py', 'scripts/a100_native_binary_comparable_statistics_reproduce.py'):
        subprocess.run([sys.executable, '-u', script], cwd=ROOT, check=True,
            env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2'))
    state('PASS_COMPARABLE_PROBE_ALL_SIX_QUALITY_REPRODUCED_GOAL_ACTIVE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--pair', choices=AFFECTED)
    args = parser.parse_args()
    worker(args.pair) if args.pair else main()
