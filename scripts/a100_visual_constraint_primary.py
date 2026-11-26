"""Image-grounded correspondence learning with immutable teacher guidance.

One complete primary versus nine same9352D adequate ordinary opponents.
No new module-ablation matrix. The learned observer is not the Qwen teacher.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.a100_six_direction import OUT as SIX

OUT = SIX/'visual_constraint_primary_20261125'
DATA = SIX/'appearance_learning_common_20261125'
PUBLIC = ROOT/'docs/a100_six_direction_20261122'
PLAN = OUT/'registration.json'
STEM = 'visual_constraint_primary_20261125'
PRIMARY = 'VISUAL_CONSTRAINT_FEEDBACK'
ORDINARY = ('MEAN_SOFT_KD', 'MAJORITY_HARD_KD', 'CONFIDENCE_SOFT_KD',
            'CROP_SOFT_KD', 'DS_FUSED_SOFT_KD', 'GAUSSIAN_FUSED_SOFT_KD',
            'PATTERN_FUSED_SOFT_KD')
COMPARATORS = (*ORDINARY, 'MEAN_SOFT_POOLED', 'MAJORITY_HARD_POOLED')
METHODS = (PRIMARY, *COMPARATORS)
SOURCES = ('scripts/a100_visual_constraint_primary.py',
           'flex_uda/active/visual_constraint_learning.py',
           'flex_uda/active/appearance_learning_inputs.py',
           'scripts/a100_appearance_learning_prepare.py',
           'scripts/a100_primary_statistics_common.py',
           'scripts/a100_semantic_pattern_teacher_statistics.py',
           'scripts/a100_joint_teacher_noise_pilot.py',
           'flex_uda/active/joint_teacher_noise.py',
           'flex_uda/active/semantic_teacher_patterns.py',
           'flex_uda/active/view_evidence.py',
           'flex_uda/active/factor_rsi_learning.py',
           'flex_uda/active/joint_observation_inputs.py',
           'flex_uda/active/observation_learning_inputs.py',
           'flex_uda/active/native_binary_inputs.py',
           'scripts/a100_factor_rsi_primary_quality.py',
           'flex_uda/active/deterministic_observation_runtime.py',
           'flex_uda/a100/evaluation_reuse.py',
           'flex_uda/a100/mechanism.py', 'flex_uda/presence/predictions.py')


def prepare():
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.active.visual_constraint_learning import probe_mask
    guard = generation_guard()
    import numpy as np
    inputs = json.loads((DATA/'data_preparation.json').read_text())
    assert (DATA/'independent_dense_alignment.json').is_file()
    cases = []
    for case in inputs['cases']:
        case = dict(case)
        assert file_sha(DATA/case['pair']/'teacher_inputs.npz') == case['data_sha256']
        keys = json.loads((DATA/case['pair']/'row_keys.json').read_text())['keys']
        hold = probe_mask(keys, case['evaluated_video'])
        case.update(probe_rows=int(hold.sum()),
                    probe_split_sha256=hashlib.sha256(hold.tobytes()).hexdigest())
        cases.append(case)
    references = {}
    for case in cases:
        for cohort in ('joint_teacher_noise_primary_20261125', 'semantic_pattern_teacher_primary_20261125'):
            path = SIX/cohort/case['pair']/'initial_observer.json'
            references[str(path)] = file_sha(path)
    plan = dict(status='PREREGISTERED_COMPLETE_VISUAL_CONSTRAINT_LEARNING_PRIMARY',
        primary=PRIMARY, comparators=COMPARATORS, cases=cases,
        source_sha256={name: file_sha(ROOT/name) for name in SOURCES},
        data_preparation_sha256=file_sha(DATA/'data_preparation.json'),
        dense_alignment_sha256=file_sha(DATA/'independent_dense_alignment.json'),
        initial_fused_teacher_reference_sha256=references,
        failure_constraints=[
            'Teacher probabilities and pseudo targets cannot be rewritten by the observer.',
            'No splitting of main-video labels; auxiliary probes are used by every final learner.',
            'Every ordinary opponent is refit adequately on exactly the same new9352D features.',
            'Image relations require local appearance AND query-conditioned features AND observable state; official tracks do not force permanent semantic membership.',
            'Feedback is an actual differentiable virtual update evaluated on withheld external teacher answers, not model agreement.',
            'All84 raw/dual-count predictions must be sealed before this-study GT evaluation.'
        ],
        human_visual_inspiration='Compare how objects look while checking what is changing in the scene; retain a learned relation only as weak evidence, never equate similar appearance with identical query membership.',
        classical_ideas='Local manifold energy as a learning regularizer; bilevel experimental feedback and gradient alignment to learn a correspondence curriculum. These ideas are prior work, not claimed inventions.',
        theoretical_limit='For nonnegative fixed weights, quadratic graph energy is nonnegative and its null space is constant on connected components. Our robust Huber energy is also nonnegative; this does NOT establish semantic homophily or an HOTA guarantee.',
        meta_prior_work='Ren et al.2018 uses CLEAN validation. Here probes are noisy frozen-teacher responses, so its clean-label guarantees do not apply. https://proceedings.mlr.press/v80/ren18a.html',
        graph='Within each first video, eight image-only DINO nearest neighbours plus previous/next provided-track observations. Per-query edge weights also gate query-conditioned PCD, colour and18D physical state. No cross-query edges or new track IDs.',
        state='Existing12D kinematics plus6D geometry/time; standard deviation normalization uses only unlabeled observations. No parsed query creates hard equivalence or truth.',
        observer='10->32 tanh->1, multiplier2sigmoid; inputs6 image/state distances and4 detached current-student summaries. Learned multiplier never generates a semantic target.',
        solver='9352->128 SiLU->128 SiLU->1, identical capacity/init for every ordinary head.',
        fitting='Adam lr.001 seed0; S500bootstrap+1500meta-stage+4000all-guidance final steps. Auxiliary whole-track probe groups20percent selected by SHA; main-video evidence never withheld. A150 updates via one virtual SGD step lr.001; actualS usesAdam, so the virtual step is an approximation.',
        regularization='Graph strength.1, Huber delta1,256 sampled edges/step, observer magnitude prior.01; meta objective is virtual probe-CE change divided by virtual rate.001 plus prior. One fixed recipe, no GT/seed/parameter grid.',
        ordinary_minimum_steps=6000, ordinary_maximum_steps=16000,
        ordinary_proper_excess_KL_limit=.005,
        fairness='All ordinary learners receive all original teacher probabilities, same100target-collection auxiliary events and same DINO descriptors; main final fit also uses all. Ordinary sees probe responses from the start.',
        primary_objective='Original immutable mean-soft teacher BCE plus learned robust visual correspondence energy; primary need not interpolate noisy teacher targets.',
        RSI_limit='Actual small observer<->RMOT student feedback; Qwen itself stays frozen. This is NOT an implemented Qwen co-evolution experiment.',
        teacher_evolution_next='A trainable Qwen adapter remains a distinct proposed direction; do not train it by treating student confidence as truth or call this observer a Qwen update.',
        execution='GPU0/1/3 complete primary and five ordinary heads; GPU2 shared fraction.10 trains Gaussian/Pattern fused ordinary heads. All existing GPU2 processes retained.',
        raw_records=30, dual_count_records=54, official_statistical_records=84,
        paired_effects=81, CountSelf_exact_aliases=30,
        success_screen='Unchanged strict screen: raw AND BOTH CountMatched effects positive against ALL9 controls in EVERY first-video case. No post-hoc highest-control promotion.',
        all84_outputs_sealed_before_this_study_GT=True, no_new_module_ablation_matrix=True,
        neural_seeds=[0], new_teacher_GPU_calls=0, source_features_prepaid=True,
        all_data_DEV=True, same_target_collection_auxiliary_transduction=True,
        prior_evaluation_exposure_disclosed=True,
        no_full_six_setting_three_contribution_or_fresh_cost_claim=True, goal_status='ACTIVE')
    assert not guard
    atomic_json(PLAN, plan); atomic_json(PUBLIC/(STEM+'_registration.json'), plan)
    print(json.dumps(dict(status=plan['status'], probes=[(c['pair'], c['probe_rows']) for c in cases],
                          pins=len(SOURCES))), flush=True)


def registration():
    plan = json.loads(PLAN.read_text())
    assert file_sha(DATA/'data_preparation.json') == plan['data_preparation_sha256']
    assert file_sha(DATA/'independent_dense_alignment.json') == plan['dense_alignment_sha256']
    for name, sha in plan['source_sha256'].items(): assert file_sha(ROOT/name) == sha, name
    for name, sha in plan['initial_fused_teacher_reference_sha256'].items(): assert file_sha(name) == sha, name
    for case in plan['cases']:
        assert file_sha(DATA/case['pair']/'teacher_inputs.npz') == case['data_sha256']
        assert file_sha(DATA/case['pair']/'row_keys.json') == case['row_keys_sha256']
    return plan


def graph_inputs(case, device):
    import numpy as np
    import torch
    from flex_uda.active.joint_observation_inputs import JointObservationVideo
    from flex_uda.active.factor_rsi_learning import Inputs
    from flex_uda.active.appearance_learning_inputs import augment_dense, bank, DIMENSION
    from scripts.a100_joint_teacher_noise_pilot import pack
    folder = OUT/case['pair']/'graph'; folder.mkdir(parents=True, exist_ok=True)
    final = folder/'receipt.json'
    if final.exists():
        receipt = json.loads(final.read_text()); assert receipt['registration_sha256'] == file_sha(PLAN)
        for name, sha in receipt['output_sha256'].items(): assert file_sha(folder/name) == sha, name
        return np.load(folder/'features.npy', mmap_mode='r'), dict(np.load(folder/'edges.npz'))
    started = time.perf_counter()
    video = JointObservationVideo(case['pair'], case['evaluated_video']); inputs = Inputs(video)
    n, nq = len(video.obs), len(video.queries)
    image = bank(video.target, video.video)
    physical = np.asarray([image.by_observation[tuple(map(int, r))] for r in video.obs])
    appearance = image.normalized(physical)/np.sqrt(768)
    a = torch.as_tensor(appearance, device=device)
    all_neighbours = []
    for begin in range(0, n, 256):
        distance = (2-2*(a[begin:begin+256]@a.T)).clamp(min=0)
        distance[torch.arange(len(distance), device=device), torch.arange(begin, begin+len(distance), device=device)] = torch.inf
        all_neighbours.append(torch.argsort(distance, dim=1, stable=True)[:, :min(8, n-1)].cpu().numpy())
    neighbours = np.concatenate(all_neighbours)
    left = np.repeat(np.arange(n), neighbours.shape[1]); right = neighbours.flatten()
    temporal_left, temporal_right = [], []
    for track in video.tracks:
        rows = np.flatnonzero(video.obs[:, 1] == track)
        rows = rows[np.argsort(video.obs[rows, 0], kind='stable')]
        temporal_left.extend(np.r_[rows[:-1], rows[1:]])
        temporal_right.extend(np.r_[rows[1:], rows[:-1]])
    packed = np.unique(np.r_[left*n+right, np.asarray(temporal_left, dtype='int64')*n+temporal_right])
    left, right = packed//n, packed % n
    assert (left != right).all() and len(np.unique(left)) == n
    dino = np.square(appearance[left]-appearance[right]).sum(1)
    state = np.c_[video.trajectory, video.geometry].astype('float64')
    state = (state-state.mean(0))/state.std(0).clip(min=.01)
    state_distance = np.square(state[left]-state[right]).mean(1)
    same = (video.obs[left, 1] == video.obs[right, 1]).astype('float32')
    gap = np.abs(video.obs[left, 0]-video.obs[right, 0])/max(int(np.ptp(video.obs[:, 0])), 1)
    dense = np.lib.format.open_memmap(folder/'features.npy', mode='w+', dtype='float32', shape=(n*nq, DIMENSION))
    edge_lists, contexts, weights, scales = [], [], [], []
    for qi, query in enumerate(video.queries):
        for begin in range(0, n, 512):
            rows = np.arange(begin, min(begin+512, n))
            records = [dict(local_query=qi, local_track=int(video.columns[r]), frame_id=int(video.obs[r, 0]),
                            track_id=int(video.obs[r, 1])) for r in rows]
            dense[qi*n+rows] = augment_dense(pack(inputs.arrays(records)), video, rows)
        raw = video.dense_rows(np.full(n, qi), np.arange(n))
        pcd = raw[:, :768]/np.maximum(np.linalg.norm(raw[:, :768], axis=1, keepdims=True), 1e-8)
        pd = np.square(pcd[left]-pcd[right]).sum(1)
        cd = np.square(raw[left, 768:1080]-raw[right, 768:1080]).mean(1)
        distance = np.stack((dino, pd, state_distance, cd), 1)
        scale = np.median(distance, axis=0).clip(min=1e-6)
        standardized = (distance/scale).clip(max=20).astype('float32')
        context = np.c_[standardized, same, gap].astype('float32')
        weight = np.exp(-standardized.sum(1)).astype('float32')
        edge_lists.append(np.c_[qi*n+left, qi*n+right]); contexts.append(context); weights.append(weight)
        scales.append(scale.tolist())
        print(json.dumps(dict(status='VISUAL_GRAPH_QUERY_INPUTS_COMPLETE', pair=case['pair'],
                              query=qi+1, total_queries=nq, physical_edges=len(left))), flush=True)
    dense.flush()
    edges = np.concatenate(edge_lists); context = np.concatenate(contexts); weight = np.concatenate(weights)
    assert np.isfinite(dense).all() and np.isfinite(context).all() and (weight > 0).all()
    np.savez(folder/'edges.npz', left=edges[:, 0], right=edges[:, 1], context=context, weight=weight)
    # Check actual paid main-video rows against the final inference pack.
    keys = json.loads((DATA/case['pair']/'row_keys.json').read_text())['keys']
    with np.load(DATA/case['pair']/'teacher_inputs.npz') as saved: paid = saved['features']
    lookup = {q['expression_id']: i for i, q in enumerate(video.queries)}
    selected = [i for i, k in enumerate(keys) if k[0] == video.video]
    positions = np.asarray([lookup[keys[i][1]]*n+video.positions[int(keys[i][3]), int(keys[i][2])] for i in selected])
    assert np.array_equal(dense[positions], paid[selected]), 'Actual train/graph/inference9352D agreement'
    anchored = np.unique(positions)
    receipt = dict(status='COMPLETE_UNLABELED_VISUAL_CONSTRAINT_GRAPH', pair=case['pair'],
        physical_observations=n, queries=nq, nodes=n*nq, edges=len(edges),
        main_paid_nodes=len(anchored), paid_nodes_with_outgoing_edges=int(np.isin(anchored, edges[:, 0]).sum()),
        full9352D_paid_alignment_passed=True, per_query_unlabeled_distance_scales=scales,
        registration_sha256=file_sha(PLAN), wall_seconds=time.perf_counter()-started,
        output_sha256={name: file_sha(folder/name) for name in ('features.npy', 'edges.npz')},
        GT_open_attempts=0, teacher_responses_used_to_construct_edges=0,
        similar_appearance_not_semantic_truth=True, no_permanent_TrackID_membership_constraint=True)
    atomic_json(final, receipt)
    return dense, dict(np.load(folder/'edges.npz'))


def ordinary_targets(probability, pair):
    import numpy as np
    from scripts.a100_joint_teacher_noise_pilot import ordinary_targets as original
    from flex_uda.active.joint_teacher_noise import CorrelatedTeacher
    from flex_uda.active.semantic_teacher_patterns import SemanticPatterns
    _, targets, _ = original(probability)
    full = np.ones(len(probability), bool)
    for cohort, kind, model in (
            ('joint_teacher_noise_primary_20261125', 'GAUSSIAN_FUSED_SOFT_KD', CorrelatedTeacher),
            ('semantic_pattern_teacher_primary_20261125', 'PATTERN_FUSED_SOFT_KD', SemanticPatterns)):
        initial = json.loads((SIX/cohort/pair/'initial_observer.json').read_text())
        targets[kind] = (full, model.from_state(initial['state']).posterior(probability), np.ones(len(probability)))
    return targets


def fit_ordinary(case, name, device):
    import numpy as np
    import torch
    from flex_uda.active.visual_constraint_learning import make_solver, cpu_state, save
    from flex_uda.active.joint_teacher_noise import fit_solver
    folder = OUT/case['pair']/'ordinary'; folder.mkdir(parents=True, exist_ok=True)
    path = folder/(name+'.pt'); model = make_solver(device)
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert saved['registration_sha256'] == file_sha(PLAN) and saved['data_sha256'] == case['data_sha256']
        assert saved['fit']['adequate_passed']
        model.load_state_dict(saved['model'])
        return model.eval(), saved['fit']
    with np.load(DATA/case['pair']/'teacher_inputs.npz') as data:
        features, probability = data['features'], data['probability']
    selected, target, weight = ordinary_targets(probability, case['pair'])[name]
    fit = fit_solver(model, features[selected], target, device, steps=6000, weight=weight, adequate=True)
    save(path, dict(model=cpu_state(model), fit=fit, registration_sha256=file_sha(PLAN),
                    data_sha256=case['data_sha256'], target_sha256=hashlib.sha256(target.tobytes()+weight.tobytes()).hexdigest(), seed=0))
    print(json.dumps(dict(status='SAME9352D_ADEQUATE_ORDINARY', pair=case['pair'], method=name,
                          steps=fit['steps'], excess_KL=fit['excess_KL'])), flush=True)
    return model.eval(), fit


def auxiliary_worker():
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    configure_before_cuda()
    import torch
    from flex_uda.a100.mechanism import generation_guard
    guard = generation_guard(); plan = registration()
    torch.set_num_threads(2); torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.10)
    for case in plan['cases']:
        for name in ORDINARY[-2:]:
            model, _ = fit_ordinary(case, name, 'cuda:0')
            del model
    assert not guard
    atomic_json(OUT/'GPU2_complete.json', dict(status='SIX_SAME_FEATURE_FUSION_KD_HEADS_COMPLETE_NO_GT', GT_open_attempts=0))


def worker(pair):
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution = configure_before_cuda()
    import numpy as np
    import torch
    from scipy.special import logsumexp
    from flex_uda.active.joint_observation_inputs import JointObservationVideo, use_verified_local_features
    from flex_uda.active.visual_constraint_learning import train
    from flex_uda.a100.mechanism import generation_guard, PAIRS
    from flex_uda.presence.predictions import seal_mask
    from scripts.a100_factor_rsi_primary_quality import subset_metadata, projection
    plan = registration(); guard = generation_guard()
    case = next(c for c in plan['cases'] if c['pair'] == pair)
    folder = OUT/pair; folder.mkdir(parents=True, exist_ok=True)
    if (folder/'main_completion.json').exists(): return
    started = time.perf_counter()
    torch.set_num_threads(2); torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.35)
    use_verified_local_features()
    dense, graph = graph_inputs(case, 'cuda:0')
    with np.load(DATA/pair/'teacher_inputs.npz') as data: features, probability = data['features'], data['probability']
    keys = json.loads((DATA/pair/'row_keys.json').read_text())['keys']
    primary, learned = train(features, probability, keys, case['evaluated_video'], dense, graph,
                             folder/'primary', 'cuda:0', file_sha(PLAN), case['data_sha256'])
    heads, fits = {PRIMARY: primary}, {}
    for name in ORDINARY[:5]: heads[name], fits[name] = fit_ordinary(case, name, 'cuda:0')
    for name in ORDINARY[-2:]:
        path = folder/'ordinary'/(name+'.pt')
        while not path.exists():
            failure = OUT/'GPU2_failed.json'
            if failure.exists(): raise RuntimeError(failure.read_text())
            time.sleep(2)
        heads[name], fits[name] = fit_ordinary(case, name, 'cuda:0')
    video = JointObservationVideo(pair, case['evaluated_video']); n, nq = len(video.obs), len(video.queries)
    scores = {name: np.empty((nq, n), dtype='float32') for name in heads}
    with torch.no_grad():
        for begin in range(0, len(dense), 512):
            x = torch.as_tensor(np.asarray(dense[begin:begin+512]), device='cuda:0')
            for name, model in heads.items(): scores[name].reshape(-1)[begin:begin+len(x)] = model(x).cpu().numpy()
    for original, pooled in (('MEAN_SOFT_KD', 'MEAN_SOFT_POOLED'), ('MAJORITY_HARD_KD', 'MAJORITY_HARD_POOLED')):
        z = scores[original].astype('float64'); result = np.empty_like(z)
        for ti in range(len(video.tracks)):
            rows = np.flatnonzero(video.columns == ti)
            value = logsumexp(-np.logaddexp(0, -z[:, rows]), axis=1)-logsumexp(-np.logaddexp(0, z[:, rows]), axis=1)
            result[:, rows] = value[:, None]
        scores[pooled] = result.astype('float32')
    bits = {name: values > 0 for name, values in scores.items()}
    metadata = subset_metadata(case['target'], video.video); tasks = []
    for name in METHODS:
        path = folder/'predictions'/name; path.mkdir(parents=True, exist_ok=True)
        assert np.isfinite(scores[name]).all()
        np.save(path/'scores.npy', scores[name]); np.save(path/'mask.npy', bits[name])
        assert np.array_equal(projection(scores[name], bits[name], video.obs), bits[name])
        seal_mask(path, metadata, bits[name].flatten(), target=case['target'], model=PAIRS[pair][0], method=name,
                  GT_open_attempts=len(guard), extra=dict(seed=0, full_raw_logits_retained=True,
                  matched_data_sha256=case['data_sha256'], Qwen_frozen=True, same_DINO_features=True))
        tasks.append(dict(pair=pair, method=name, folder=str(path)))
    for rank, count in [(PRIMARY, name) for name in COMPARATORS]+[(name, PRIMARY) for name in COMPARATORS]:
        name = rank+'__AT_'+count+'_COUNT'; path = folder/'predictions'/name
        selected = projection(scores[rank], bits[count], video.obs)
        seal_mask(path, metadata, selected.flatten(), target=case['target'], model=PAIRS[pair][0], method=name,
                  GT_open_attempts=len(guard), extra=dict(rank_method=rank, count_method=count,
                  counts_equal_per_query_frame=True, counts_from_GT=False))
        tasks.append(dict(pair=pair, method=name, folder=str(path)))
    assert len(tasks) == 28 and not guard
    atomic_json(folder/'main_completion.json', dict(status='SEALED_ALL28_VISUAL_CONSTRAINT_OUTPUTS_NO_GT',
        tasks=tasks, GT_open_attempts=0, ordinary_fits=fits, execution=execution,
        primary_teacher_excess_CE=learned['teacher_excess_CE'],
        observer_parameter_change=learned['observer_parameter_change'], teacher_targets_unchanged=True,
        wall_seconds=time.perf_counter()-started, not_full_cold_deployment=True))


def drain(process, log):
    with log.open('a', buffering=1) as stream:
        for line in process.stdout: stream.write(line); print(line, end='', flush=True)


def main():
    plan = registration(); supervisor = OUT/'supervisor'; supervisor.mkdir(parents=True, exist_ok=True)
    pending = [(str(c['pair']), gpu, ['--worker', c['pair']]) for gpu, c in zip((0, 1, 3), plan['cases'])
               if not (OUT/c['pair']/'main_completion.json').exists()]
    if not (OUT/'GPU2_complete.json').exists(): pending.append(('GPU2', 2, ['--auxiliary-worker']))
    for label, gpu, _ in pending:
        residents = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'], text=True).strip()
        used = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
        total = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.total', '--format=csv,noheader,nounits'], text=True).strip())
        if gpu == 2: assert total-used >= 8192, (gpu, used, total)
        else: assert not residents and used < 512, (gpu, residents, used)
        atomic_json(supervisor/(label+'_preflight.json'), dict(GPU=gpu, residents=residents, used_MiB=used,
                    all_existing_processes_preserved=True, user_authorized_sharing=gpu == 2))
    jobs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as readers:
        for label, gpu, arguments in pending:
            proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), *arguments], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
            future = readers.submit(drain, proc, supervisor/(label+'.log')); jobs.append((label, proc, future))
            atomic_json(supervisor/(label+'_launch.json'), dict(pid=proc.pid, GPU=gpu, role=label))
            print(json.dumps(dict(status='STARTED_VISUAL_CONSTRAINT_PRIMARY_ROLE', role=label, GPU=gpu, pid=proc.pid)), flush=True)
        failed = []
        for label, proc, future in jobs:
            code = proc.wait(); future.result()
            if code: failed.append(dict(role=label, exit_code=code))
    if failed: raise RuntimeError(failed)
    from flex_uda.presence.predictions import verify_predictions
    tasks = []
    for case in plan['cases']:
        done = json.loads((OUT/case['pair']/'main_completion.json').read_text()); assert done['GT_open_attempts'] == 0
        tasks.extend(done['tasks'])
    assert len(tasks) == 84
    for task in tasks: verify_predictions(task['folder'])
    atomic_json(OUT/'all84_sealed_before_GT.json', dict(tasks=tasks, GT_open_attempts=0, registration_sha256=file_sha(PLAN)))
    print('ALL84_VISUAL_CONSTRAINT_OUTPUTS_SEALED_ISOLATED_EVALUATION', flush=True)
    from scripts.a100_joint_teacher_noise_pilot import evaluate_one
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool: completed = list(pool.map(evaluate_one, tasks))
    results = {c['pair']: {} for c in plan['cases']}
    for task, result in completed: results[task['pair']][task['method']] = result
    comparisons = []
    for pair, values in results.items():
        primary = values[PRIMARY]['HOTA_unrounded']
        for other in COMPARATORS:
            reference = values[other]['HOTA_unrounded']
            comparisons.append(dict(pair=pair, comparator=other, primary_HOTA=primary, ordinary_HOTA=reference,
                raw_delta_HOTA=primary-reference,
                primary_at_ordinary_count_delta_HOTA=values[PRIMARY+'__AT_'+other+'_COUNT']['HOTA_unrounded']-reference,
                ordinary_at_primary_count_delta_HOTA=primary-values[other+'__AT_'+PRIMARY+'_COUNT']['HOTA_unrounded']))
    eligible = all(min(r['raw_delta_HOTA'], r['primary_at_ordinary_count_delta_HOTA'], r['ordinary_at_primary_count_delta_HOTA']) > 0 for r in comparisons)
    atomic_json(OUT/'quality_results.json', dict(status='COMPLETE_VISUAL_CONSTRAINT_PRIMARY', results=results,
        comparisons=comparisons, candidate_eligible_for_larger_screen=eligible, all_data_DEV=True,
        no_six_setting_three_contribution_or_fresh_cost_claim=True, goal_status='ACTIVE'))
    from scripts.a100_primary_statistics_common import collect, reproduce
    collect(OUT, PUBLIC, STEM, 84); reproduce(PUBLIC, STEM, 84, 81)
    print(json.dumps(dict(status='COMPLETE_VISUAL_CONSTRAINT_PRIMARY_ALL84_REPRODUCED', comparisons=comparisons)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker'); parser.add_argument('--auxiliary-worker', action='store_true'); args = parser.parse_args()
    try:
        if args.prepare: prepare()
        elif args.worker: worker(args.worker)
        elif args.auxiliary_worker: auxiliary_worker()
        else: main()
    except Exception as error:
        if args.worker: atomic_json(OUT/args.worker/'failed.json', dict(error=repr(error)))
        if args.auxiliary_worker: atomic_json(OUT/'GPU2_failed.json', dict(error=repr(error)))
        raise
