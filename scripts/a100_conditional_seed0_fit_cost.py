"""Report actual retained seed0 fit phases, never call them cold deployment."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from scripts import a100_conditional_seed0_scope as scope


def main():
    scope.check_registration()
    arms = (*scope.original.CORE, *scope.original.PAIRED)
    summaries, widths = {}, Counter()
    prefixes = actual = reused = fit_receipts = 0
    maximum = 0.
    for record in scope.conditions():
        raw = json.loads((Path(record['folder'])/'completion.json').read_text())
        assert raw['GT_open_attempts'] == 0 and not raw['target_fit_failures']
        assert raw['no_new_teacher_or_backbone_calls']
        rows = {a:dict(reference_fit_wall_seconds=0., actual_unique_fit_wall_seconds=0.,
            actual_fits=0, reused_fits=0, risk_continuations=0, maximum_objective_excess=0.) for a in arms}
        condition_actual = 0
        for video in raw['receipts']:
            assert video['GT_open_attempts'] == 0 and video['current_and_past_only']
            assert video['quality_replay_not_cold_cost']
            assert video['execution']['deterministic_algorithms']
            assert video['execution']['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
            assert set(video['fits']) == set(arms)
            video_actual = 0
            for arm, fit in video['fits'].items():
                assert fit['target_fit_gate_pass'] and not fit['target_GT_or_HOTA_used']
                assert fit['deterministic_execution'] and fit['source_recipe_unchanged']
                assert fit['final_CE_recomputed_from_actual_float32_deployed_head']
                excess = float(fit['metrics']['objective_excess'])
                assert excess <= .005 and float(fit['metrics']['target_KL']) <= .005
                wall = float(fit['wall_seconds'])
                assert wall >= 0
                owner = fit.get('identical_own_prefix_fit_reused_from')
                rows[arm]['reference_fit_wall_seconds'] += wall
                rows[arm]['maximum_objective_excess'] = max(rows[arm]['maximum_objective_excess'], excess)
                if owner is None:
                    rows[arm]['actual_fits'] += 1
                    rows[arm]['actual_unique_fit_wall_seconds'] += wall
                    video_actual += 1
                else:
                    assert arms.index(owner) < arms.index(arm)
                    assert wall == video['fits'][owner]['wall_seconds']
                    rows[arm]['reused_fits'] += 1
                rows[arm]['risk_continuations'] += bool(fit['teacher_risk_continuations'])
                widths[str(fit['width'])] += 1
                maximum = max(maximum, excess)
            assert video_actual == video['actual_head_fits']
            assert len(arms)-video_actual == video['exact_identical_fit_reuse']
            condition_actual += video_actual
        assert condition_actual == raw['actual_head_fits']
        n = len(raw['receipts'])
        prefixes += n
        fit_receipts += n*len(arms)
        actual += condition_actual
        reused += n*len(arms)-condition_actual
        summaries[record['pair']] = dict(neural_seed=0, video_prefixes=n, per_arm=rows)
    assert prefixes == 148 and fit_receipts == 1628
    result = dict(status='PASS_SIX_SETTINGS_SEED0_RETAINED_FIT_PHASES', seeds=[0],
        conditions=summaries, video_prefixes=prefixes, logical_fit_receipts=fit_receipts,
        actual_unique_fits=actual, reused_identical_fit_receipts=reused, widths=dict(widths),
        maximum_objective_excess=maximum,
        actual_unique_fit_wall_seconds_sum=sum(r['actual_unique_fit_wall_seconds']
            for setting in summaries.values() for r in setting['per_arm'].values()),
        all_original_teacher_risk_and_execution_receipt_gates_pass=True,
        not_independent_recomputation_from_every_saved_head=True,
        no_new_fits_predictions_teacher_backbone_or_GT_HOTA_reads=True,
        phase_includes_optimization_continuations_and_deployed_head_check=True,
        shared_inputs_teacher_and_GPU_initialization_prepaid=True,
        multiple_outputs_generated_jointly_no_equal_cost_split=True,
        retained_valid_prefixes_only_failed_attempts_and_cache_preparation_excluded=True,
        not_GPU_kernel_timing_or_controlled_speed_comparison=True,
        end_to_end_cold_cost=False, goal_status='ACTIVE')
    atomic_json(scope.OUT/'fit_cost_phases.json', result)
    (scope.PUBLIC/'conditional_seed0_fit_cost.json').write_text(json.dumps(result, separators=(',', ':'))+'\n')
    print(json.dumps({k:result[k] for k in ('status','video_prefixes','logical_fit_receipts',
        'actual_unique_fits','reused_identical_fit_receipts','maximum_objective_excess',
        'actual_unique_fit_wall_seconds_sum','end_to_end_cold_cost')}))


if __name__ == '__main__':
    main()
