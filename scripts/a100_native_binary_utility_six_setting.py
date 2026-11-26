"""Prospective utility-readout intervention on frozen, causal native heads.

No new fitting, teacher answers, backbone inference or manual-label input.
The original learned model supplies K; a conditional set model supplies the
ranking. All six settings seal before isolated official evaluation. Existing
studies, primary methods and checkpoints are never changed.
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
from scripts import a100_native_binary_comparable_six_setting as parent
from scripts import a100_native_binary_six_setting as original

PAIRS, PUBLIC = parent.PAIRS, parent.PUBLIC
OUT = parent.OUT.parent/'native_binary_utility_quality_20261124'
PLAN = PUBLIC/'native_binary_utility_quality_registration_20261124.json'
STATE = OUT/'supervisor/state.json'
PARENT_L = parent.PRIMARY
PARENT_F = 'FULL_CONDITIONAL_REFERENCE_FROZEN_FINAL'
ORDINARY = tuple(n for n in parent.METHODS if n not in (PARENT_L, PARENT_F))
PRIMARY = 'UTILITY_LEARNED__COUNT_LEARNED'
U_FF = 'UTILITY_FROZEN__COUNT_FROZEN'
U_FL = 'UTILITY_FROZEN__COUNT_LEARNED'
U_LF = 'UTILITY_LEARNED__COUNT_FROZEN'
READOUTS = {PRIMARY: ('LEARNED', PARENT_L), U_FF: ('FROZEN', PARENT_F),
            U_FL: ('FROZEN', PARENT_L), U_LF: ('LEARNED', PARENT_F)}
READOUTS.update({'UTILITY_LEARNED__COUNT_'+n: ('LEARNED', n) for n in ORDINARY})
EXPECTED = 324  # 120 new masks/evals and 204 exact parent records.


def source_root(pair):
    # Four unaffected settings eventually become exact parent symlinks.
    # Their completed originals can already supply the same frozen weights.
    base = parent.OUT if pair in parent.AFFECTED else parent.PARENT_OUT
    return base/pair/'seed0'


def parent_paths(pair):
    root = parent.OUT/pair/'seed0'
    paths = {'PARENT__'+n: root/n for n in parent.METHODS}
    paths.update({'PARENT__'+n+'__AT_LEARNED_COUNT':
        root/'countmatched'/(n+'__AT_'+PARENT_L+'_COUNT') for n in ORDINARY})
    assert len(paths) == 34
    return paths


def contrast_pairs():
    result = {
        'READOUT_AT_LEARNED_COUNT': (PRIMARY, 'PARENT__'+PARENT_L),
        'READOUT_AT_FROZEN_COUNT': (U_FF, 'PARENT__'+PARENT_F),
        'RSI_RAW': (PRIMARY, U_FF),
        'RSI_AT_LEARNED_COUNT': (PRIMARY, U_FL),
        'RSI_AT_FROZEN_COUNT': (U_LF, U_FF)}
    for other in ORDINARY:
        result['RAW_vs_'+other] = (PRIMARY, 'PARENT__'+other)
        result['AT_LEARNED_COUNT_vs_'+other] = (PRIMARY, 'PARENT__'+other+'__AT_LEARNED_COUNT')
        result['AT_OTHER_COUNT_vs_'+other] = ('UTILITY_LEARNED__COUNT_'+other, 'PARENT__'+other)
    assert len(result) == 53
    return result


def check_plan():
    plan = json.loads(PLAN.read_text())
    parent.check_plan()
    assert plan['neural_seeds'] == [0] and plan['primary'] == PRIMARY
    assert plan['readouts'] == {n: list(v) for n, v in READOUTS.items()}
    assert plan['contrasts'] == {n: list(v) for n, v in contrast_pairs().items()}
    assert plan['official_records'] == EXPECTED
    assert plan['parent_registration_sha256'] == file_sha(parent.PLAN)
    for name, sha in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == sha, name
    return plan


def project_many(scores, donors, observations):
    """Sort each utility only once for all model-donated cardinalities."""
    from flex_uda.active.conditional_set_readout import at_donor_counts
    shape = scores['LEARNED'].shape
    assert all(v.shape == shape and np.isfinite(v).all() for v in scores.values())
    assert all(v.shape == shape and v.dtype == bool for v in donors.values())
    masks = {name: np.zeros(shape, dtype=bool) for name in READOUTS}
    order = np.argsort(observations[:, 0], kind='stable')
    cuts = np.flatnonzero(np.diff(observations[order, 0]))+1
    for ids in np.split(order, cuts):
        ks = {name: value[:, ids].sum(1) for name, value in donors.items()}
        for qi in range(shape[0]):
            ranks = {p: ids[np.lexsort((observations[ids, 1], -u[qi, ids]))] for p, u in scores.items()}
            for name, (policy, donor) in READOUTS.items():
                masks[name][qi, ranks[policy][:int(ks[donor][qi])]] = True
        for name, (_, donor) in READOUTS.items():
            assert np.array_equal(masks[name][:, ids].sum(1), ks[donor])
    # Independent simple implementation verifies the optimized sorting on
    # the same actual input, not a mirrored synthetic implementation test.
    for name in (PRIMARY, U_FF, U_FL, U_LF):
        policy, donor = READOUTS[name]
        assert np.array_equal(masks[name], at_donor_counts(scores[policy], donors[donor], observations))
    return masks


def utilities(video, model, mean, std, destination):
    import torch
    n, nq = len(video.obs), len(video.queries)
    result = np.lib.format.open_memmap(destination, mode='w+', dtype='float64', shape=(nq, n))
    with torch.no_grad():
        for qi in range(nq):
            for begin in range(0, n, 1024):
                ids = np.arange(begin, min(begin+1024, n)); cols = video.columns[ids]
                anchors = ((video.raw_bags[qi, cols]-mean)/std).astype('float32')
                present = video.present[cols]
                anchors[~present] = 0
                current = ((video.dense_rows(np.full(len(ids), qi), ids)-mean)/std).astype('float32')
                value = model.utility(torch.as_tensor(anchors, device='cuda:0'),
                    torch.as_tensor(present, device='cuda:0'), torch.as_tensor(current, device='cuda:0'))
                result[qi, ids] = value.double().cpu().numpy()
    result.flush(); assert np.isfinite(result).all()
    return result


def worker(pair):
    check_plan()
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution = configure_before_cuda()
    import torch
    from flex_uda.a100.mechanism import metadata, generation_guard
    from flex_uda.active.joint_observation_inputs import JointObservationVideo, use_verified_local_features
    from flex_uda.active.native_binary_models import make_model
    from scripts.a100_visual_classical_strict_prefix import immediate_predictions
    from flex_uda.presence.predictions import seal_mask
    guard = generation_guard(); torch.set_num_threads(2)
    torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.10)
    cache = use_verified_local_features()
    source = source_root(pair); raw = json.loads((source/'completion.json').read_text())
    assert raw['GT_open_attempts'] == 0
    assert {r['method'] for r in raw['methods']} == set(parent.METHODS)
    root = OUT/pair/'seed0'; root.mkdir(parents=True, exist_ok=True)
    if (root/'completion.json').is_file():
        return
    student, target = PAIRS[pair]; md = metadata(target); receipts = []
    started = time.perf_counter()
    for vi, video_id in enumerate(sorted(md['videos'])):
        local = root/'videos'/video_id; local.mkdir(parents=True, exist_ok=True)
        done = local/'completion.json'
        if done.is_file():
            receipt = json.loads(done.read_text()); assert receipt['GT_open_attempts'] == 0
            receipts.append(receipt); continue
        stamp = time.perf_counter(); video = JointObservationVideo(pair, video_id)
        scores, checkpoints = {}, {}
        for policy in ('LEARNED', 'FROZEN'):
            path = source/'videos'/video_id/'policies'/policy/'closed.pt'
            saved = torch.load(path, map_location='cpu', weights_only=False)['checkpoint']
            assert saved['recipe'] == parent.RECIPES['FULL_CONDITIONAL_REFERENCE_'+policy]
            model = make_model(saved['recipe'], 1101, saved['width']).cuda().eval()
            model.load_state_dict(saved['model'])
            scores[policy] = utilities(video, model, saved['mean'], saved['std'], local/(policy+'_utility.npy'))
            checkpoints[policy] = dict(path=str(path), sha256=file_sha(path),
                normalization_is_original_past_and_current_prefix=True)
            del model
        donors = {n: np.load(source/n/'videos'/video_id/'mask.npy') for n in parent.METHODS}
        masks = project_many(scores, donors, video.obs)
        changes = {}
        for name, bits in masks.items():
            folder = root/name/'videos'/video_id; folder.mkdir(parents=True, exist_ok=True)
            np.save(folder/'mask.npy', bits); immediate_predictions(root/name, video, bits)
            changes[name] = int((bits ^ donors[READOUTS[name][1]]).sum()//2)
        torch.cuda.synchronize()
        receipt = dict(video=video_id, model_checkpoints=checkpoints,
            rank_replacements_relative_to_count_donor=changes,
            all_counts_equal_per_query_frame=True, counts_from_manual_GT=False,
            GT_open_attempts=len(guard), new_fits=0, new_teacher_calls=0, new_backbone_calls=0,
            wall_seconds=time.perf_counter()-stamp, prepaid_replay_not_cold_deployment=True)
        assert not guard; atomic_json(done, receipt); receipts.append(receipt)
        progress = dict(stage='FROZEN_PREFIX_UTILITY_READOUT_NO_GT', pair=pair, video=video_id,
            completed_videos=vi+1, total_videos=len(md['videos']))
        atomic_json(root/'progress.json', progress); print(json.dumps(progress), flush=True)
        del scores, masks, donors, video; torch.cuda.empty_cache()
    methods = []
    for name in READOUTS:
        mask = np.zeros(md['candidate_query_rows'], dtype=bool)
        for video_id in sorted(md['videos']):
            bits = np.load(root/name/'videos'/video_id/'mask.npy')
            for qi, query in enumerate(q for q in md['episodes'] if q['video_id'] == video_id):
                mask[query['start']:query['stop']] = bits[qi]
        seal = seal_mask(root/name, md, mask, target=target, model=student, method=name,
            GT_open_attempts=len(guard), extra=dict(seed=0, frozen_source_cohort=str(source),
                count_donor=READOUTS[name][1], utility_policy=READOUTS[name][0],
                counts_equal_per_query_frame=True, counts_from_GT=False,
                ordinary_count_donor_is_diagnostic=name not in (PRIMARY, U_FF),
                conditional_set_readout_not_original_pair_network_MAP=True))
        methods.append(dict(method=name, rows=seal['prediction_rows']))
    assert not guard
    atomic_json(root/'completion.json', dict(status='SEALED_ALL_UTILITY_READOUTS_NO_GT', pair=pair,
        methods=methods, receipts=receipts, GT_open_attempts=0, execution=execution,
        source_raw_completion_sha256=file_sha(source/'completion.json'),
        verified_cache_files=cache['files'], wall_seconds=time.perf_counter()-started,
        new_fits=0, new_teacher_calls=0, new_backbone_calls=0))


def assignment_candidates(active):
    known = set()
    for path in (original.STATE, parent.parent.STATE, parent.STATE):
        if path.is_file():
            known.update(int(j['pid']) for j in json.loads(path.read_text()).get('active', {}).values())
    known.update(proc.pid for proc, _, _ in active.values())
    candidates = []
    for gpu in (0, 1, 2, 3):
        if gpu in active:
            continue
        used = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used',
            '--format=csv,noheader,nounits'], text=True).strip())
        pids = {int(p) for p in subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).split()}
        if used < 8192 and pids <= known:
            candidates.append((len(pids), used, gpu, sorted(pids)))
    return sorted(candidates)


def main():
    plan = check_plan(); start = time.perf_counter()
    pending = ['C1', 'C6', 'C2', 'C5', 'C3', 'C4']; active = {}; finished = []; failure = None

    def state(status):
        jobs = {}
        for gpu, (_, _, job) in active.items():
            path = Path(job['folder'])/'progress.json'
            jobs[str(gpu)] = dict(job, progress=json.loads(path.read_text()) if path.is_file() else {})
        atomic_json(STATE, dict(status=status, pid=os.getpid(), active=jobs, pending=pending,
            finished=finished, failure=failure, wall_seconds=time.perf_counter()-start,
            neural_seeds=[0], goal_status='ACTIVE'))
        print(json.dumps(dict(status=status, active={g: dict(pair=j['pair'], pid=j['pid'],
            progress=j['progress']) for g, j in jobs.items()}, pending=pending, failure=failure)), flush=True)

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
                pending.remove(pair); finished.append(dict(pair=pair, reused_completed=True))
        ready = [p for p in pending if (source_root(p)/'completion.json').is_file()]
        if ready and len(active) < 2 and not failure:
            for _, used, gpu, residents in assignment_candidates(active):
                if not ready or len(active) >= 2:
                    break
                pair = ready.pop(0); pending.remove(pair)
                log = OUT/'supervisor'/(pair+'_seed0.log'); handle = log.open('a', buffering=1)
                proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--pair', pair], cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                             OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2'))
                active[gpu] = proc, handle, dict(pair=pair, pid=proc.pid, physical_GPU=gpu,
                    folder=str(OUT/pair/'seed0'), log=str(log), preflight_memory_MiB=used,
                    preserved_known_resident_PIDs=residents)
        state('PRESERVING_HEALTHY_READOUT_WORKERS_AFTER_FAILURE' if failure else 'RUNNING_FROZEN_PREFIX_UTILITY_READOUTS_NO_GT')
        if failure and not active:
            raise RuntimeError(json.dumps(failure))
        if pending or active:
            time.sleep(30)
    while not (parent.OUT/'all_predictions_sealed_before_GT.json').is_file():
        state('WAITING_PARENT_ALL_COUNT_OUTPUTS_BEFORE_UTILITY_EVALUATION_NO_GT'); time.sleep(30)
    tasks, references = [], {}
    for pair in PAIRS:
        root = OUT/pair/'seed0'; raw = json.loads((root/'completion.json').read_text())
        assert raw['GT_open_attempts'] == 0
        tasks.extend((str(root/r['method']), pair) for r in raw['methods'])
        for name, path in parent_paths(pair).items():
            assert json.loads((path/'predictions_seal.json').read_text())['GT_open_attempts'] == 0
            references[pair+'/'+name] = str(path)
    assert len(tasks) == len(set(tasks)) == 120 and len(references) == 204
    atomic_json(OUT/'all_predictions_sealed_before_GT.json', dict(methods=EXPECTED,
        new_official_evaluations=120, exactly_reused_records=204, GT_open_attempts=0,
        inherited_paths=references, source_sha256=plan['source_sha256']))
    state('ALL_324_UTILITY_RECORDS_SEALED_ISOLATED_OFFICIAL_EVALUATION')
    from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        results = dict(pool.map(evaluate_one, tasks))
    while not (parent.OUT/'official.json').is_file():
        state('WAITING_COMPLETE_PARENT_OFFICIAL_FOR_EXACT_RECORD_REUSE'); time.sleep(30)
    official = json.loads((parent.OUT/'official.json').read_text()); assert official['methods'] == 318
    for alias, source in references.items():
        results[alias] = official['results'][source]
    assert len(results) == EXPECTED
    atomic_json(OUT/'official.json', dict(methods=EXPECTED, results=results,
        new_official_evaluations=120, exactly_reused_records=204,
        parent_official_sha256=file_sha(parent.OUT/'official.json')))
    state('ANALYZING_COMPLETE_UTILITY_READOUT_STUDY')
    for script in ('scripts/a100_native_binary_utility_analyze.py',
                   'scripts/a100_native_binary_utility_statistics_reproduce.py'):
        subprocess.run([sys.executable, '-u', script], cwd=ROOT, check=True,
            env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2'))
    state('PASS_ALL_SIX_UTILITY_READOUT_QUALITY_REPRODUCED_GOAL_ACTIVE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--pair', choices=PAIRS)
    args = parser.parse_args(); worker(args.pair) if args.pair else main()
