"""Honor the user's single-training-seed amendment without changing learners.

Adopt healthy seed0 workers, reuse sealed predictions and official statistics,
and evaluate only the six seed0 conditions. No training job is launched here.
The original three-seed registrations and completed records remain intact.
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
from flex_uda.a100.mechanism import PAIRS
from scripts import a100_conditional_deterministic_repair as original
from scripts.a100_six_direction import OUT as SIX, PUBLIC

STEM = 'conditional_seed0_scope'
OUT = SIX/'conditional_seed0_scope_20261123'
STATE = OUT/'supervisor/state.json'
REGISTRATION = PUBLIC/(STEM+'_registration_v1.json')
AMENDMENT = SIX.parent/'supervisor/single_seed_user_amendment_20261123.json'
SUCCESS = 'PASS_SINGLE_SEED_SIX_SETTINGS_OFFICIAL_ANALYSIS_GOAL_ACTIVE'
INTERACTION = ('FULL_FINAL', 'CONDITIONAL_BALANCED_FINAL',
    'DIRECT_PROFILE_FINAL', 'DIRECT_BALANCED_FINAL')


def conditions():
    return [dict(pair=p, seed=0, folder=str(original.OUT/p/'seed0')) for p in PAIRS]


def check_registration():
    plan = json.loads(REGISTRATION.read_text())
    assert plan['seeds'] == [0] and plan['primary'] == original.PRIMARY
    assert plan['settings'] == {p:list(v) for p, v in PAIRS.items()}
    assert plan['contrasts'] == {k:list(v) for k, v in original.CONTRASTS.items()}
    for name, sha in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == sha, name
    original.check_registration()
    return plan


def live(pid):
    path = Path('/proc')/str(pid)/'stat'
    return path.is_file() and path.read_text().rsplit(')', 1)[1].split()[0] != 'Z'


def state(status, **extra):
    atomic_json(STATE, dict(status=status, pid=os.getpid(), seeds=[0],
        goal_status='ACTIVE', **extra))


def adopt_seed0_workers():
    """Wait on existing owners; never restart fits or launch additional seeds."""
    amendment = json.loads(AMENDMENT.read_text())
    adopted = {j['pair']:j for j in amendment['preserved_seed0_workers']}
    assert all(j['seed'] == 0 for j in adopted.values())
    while True:
        complete, active = [], {}
        for record in conditions():
            if (Path(record['folder'])/'completion.json').is_file():
                complete.append(record)
            else:
                owner = adopted[record['pair']]
                assert live(owner['pid']), ('Resolve existing seed0 worker failure', owner)
                active[str(owner['physical_GPU'])] = owner
        state('ADOPTING_EXISTING_SEED0_WORKERS_NO_NEW_FITS', completed=len(complete),
            finished=complete, active=active, pending=[], failure=None)
        if len(complete) == 6:
            return
        time.sleep(15)


def seal_scope():
    # The previous parent studies already sealed all their outputs without GT.
    # Their remaining seed17/42 evaluation and aggregation are intentionally cut.
    for root, expected in original.PARENTS:
        proof = json.loads((root/'all_predictions_sealed_before_GT.json').read_text())
        assert proof['methods'] == expected and proof['GT_open_attempts'] == 0
    state('GENERATING_ONLY_SEED0_DUAL_COUNTS_NO_GT', completed=6, active={})
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(original.counts, conditions()))
    tasks = []
    for record in conditions():
        root = Path(record['folder'])
        for folder in (root, root/'countmatched'):
            assert json.loads((folder/'completion.json').read_text())['GT_open_attempts'] == 0
        for ref in (original.PRECISION, original.STANDARD):
            folder = ref/record['pair']/'seed0'
            raw = json.loads((folder/'completion.json').read_text())
            counts = json.loads((folder/'countmatched/completion.json').read_text())
            assert raw['GT_open_attempts'] == counts['GT_open_attempts'] == 0
            assert not any(counts['CountSelf_changed_query_files'].values())
        paths = original.statistical_paths(record)
        assert len(paths) == 146
        tasks.extend((str(folder), record['pair']) for folder in paths.values())
    assert len(tasks) == len(set(tasks)) == 876
    atomic_json(OUT/'all_predictions_sealed_before_GT.json', dict(methods=876,
        own_methods=588, fixed_reference_methods=288, conditions=6, seeds=[0],
        GT_open_attempts=0, primary=original.PRIMARY,
        parent_outputs_previously_sealed=[n for _, n in original.PARENTS],
        supersedes_three_seed_evaluation_scope_only=True))
    return tasks


def interaction(values):
    full, conditional, profile, direct = values
    return dict(factorization_given_profile=full-profile,
        factorization_without_profile=conditional-direct,
        profile_given_factorization=full-conditional,
        profile_without_factorization=profile-direct,
        interaction=full-conditional-profile+direct)


def analyze():
    from scripts.a100_visual_classical_analyze import read_stats
    from scripts.a100_six_direction_analyze import comparison
    from scripts.analyze_pf_video_uncertainty import pooled_hota
    from scripts.a100_visual_classical_sparse_statistics_reproduce import pack_statistic_rows
    data, statistics, summary, error = {}, {}, {}, 0.
    for record in conditions():
        root = Path(record['folder'])
        slug = record['pair']+'_seed0'
        raw = json.loads((root/'completion.json').read_text())
        counts = json.loads((root/'countmatched/completion.json').read_text())
        data[slug] = {name:read_stats(folder) for name, folder in original.statistical_paths(record).items()}
        for name, row in data[slug].items():
            num, den = row['parent_stats']
            error = max(error, abs(float(pooled_hota((num, den), np.ones(len(num))))-row['HOTA']))
            statistics[slug+'/'+name] = dict(parents=row['parent_videos'], HOTA=row['HOTA'],
                numerator=num.tolist(), denominator=den.tolist())
        summary[slug] = dict(raw_HOTA={n:data[slug][n]['HOTA'] for n in original.METHODS},
            target_fit_failures=raw['target_fit_failures'],
            CountSelf_changed_query_files=counts['CountSelf_changed_query_files'],
            exact_rank_cutoff_ties=counts['actual_exact_rank_cutoff_ties'])
    settings, decomposition = {}, {}
    for pair in PAIRS:
        rows = data[pair+'_seed0']
        raw = interaction([rows[n]['HOTA'] for n in INTERACTION])
        matched = interaction([rows[n+'__AT_FULL_FINAL_COUNT']['HOTA'] for n in INTERACTION])
        settings[pair] = dict(neural_seeds=[0],
            mean_HOTA={n:rows[n]['HOTA'] for n in original.METHODS},
            comparisons={label:comparison([rows[a]], [rows[b]])
                for label, (a, b) in original.CONTRASTS.items()},
            factorization_profile_interaction=dict(per_seed=[raw['interaction']],
                mean=raw['interaction'], CI95=None))
        decomposition[pair] = dict(neural_seed=0, raw=raw, at_full_output_count=matched,
            raw_minus_common_count_interaction=raw['interaction']-matched['interaction'], CI95=None)
    assert error < 1e-8 and len(statistics) == 876
    result = dict(status=SUCCESS, primary=original.PRIMARY, neural_seeds=[0],
        conditions=summary, repeated=settings, HOTA_parent_sum_max_error=error,
        repeated_field_compatibility_only_no_multiple_training_seeds=True,
        no_across_neural_seed_uncertainty_estimated=True,
        uncertainty_scope='Paired original-parent-video bootstrap at fixed seed0; four-video KITTI has no population CI; bootstrap draws require no new fits.',
        common_count_four_arm_description=decomposition,
        raw_minus_count_is_not_unique_causal_quantity_effect=True,
        all_own_and_required_parent_outputs_sealed_before_this_GT=True,
        all_data_development=True, independent_blind_validation=False,
        manual_target_GT_in_adaptation=False, quality_replay_not_cold_cost=True,
        latent_factors_not_semantically_identified=True,
        paired_ordinary_deterministic_methods=list(original.OWN[-4:]),
        other_ordinary_references_frozen_not_all_paired_execution_ablation=True,
        execution_repair_not_a_new_semantic_innovation=True, goal_status='ACTIVE')
    atomic_json(OUT/'analysis.json', result)
    atomic_json(OUT/'parent_statistics_readable.json', statistics)
    (PUBLIC/(STEM+'_analysis.json')).write_text(json.dumps(result, separators=(',', ':'))+'\n')
    (PUBLIC/(STEM+'_parent_statistics.json')).write_text(json.dumps(pack_statistic_rows(statistics), separators=(',', ':'))+'\n')


def main():
    check_registration()
    if STATE.is_file() and json.loads(STATE.read_text()).get('status') == SUCCESS:
        return
    adopt_seed0_workers()
    tasks = seal_scope()
    reused = sum((Path(folder)/'official_completion.json').is_file() for folder, _ in tasks)
    state('EVALUATING_ONLY_MISSING_SEED0_OWN_AND_REFERENCE_METHODS', completed=6,
        active={}, official_expected=876, official_reused_at_start=reused)
    from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        # The official evaluator immediately returns existing valid completions.
        results = dict(pool.map(evaluate_one, tasks))
    atomic_json(OUT/'official.json', dict(methods=876, own_methods=588,
        fixed_reference_methods=288, reused_at_start=reused, results=results))
    state('ALL_SEED0_OFFICIAL_COMPLETE_ANALYZING', completed=6, active={})
    analyze()
    subprocess.run([sys.executable,
        'scripts/a100_six_setting_statistics_reproduce.py', '--stem', STEM],
        cwd=ROOT, check=True)
    state(SUCCESS, completed=6, active={}, official_expected=876,
        independent_arithmetic_reproduction=True, scientific_goal_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        check_registration()
        print(json.dumps(dict(status='PASS_SINGLE_SEED_AMENDMENT_SOURCE_CHECK',
            settings=6, seeds=[0], no_fits_or_GT_or_new_workers=True)))
    else:
        try:
            main()
        except Exception as error:
            state('FAILED_SINGLE_SEED_SCOPE_REQUIRES_ROOT_CAUSE', error=repr(error))
            raise
