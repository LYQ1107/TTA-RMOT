"""Explain completed old-joint results using their existing official tables.

No GT opens, new predictions, matching or HOTA evaluation. This is descriptive
association evidence, not an additive causal attribution to temporal flips.
"""
import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.a100_joint_observation_six_setting import OUT, PRIMARY, PUBLIC, PAIRS


def main():
    official = json.loads((OUT/'official.json').read_text())
    reproduced = json.loads((PUBLIC/'joint_observation_quality_seed0_reproduction.json').read_text())
    assert official['methods'] == reproduced['statistical_records'] == 582
    assert reproduced['maximum_absolute_error'] < 1e-8
    baseline = 'COARSE_UNIT_SOFT_DESIGN_FINAL'; settings = {}
    for pair in PAIRS:
        root = OUT/pair/'seed0'; methods = {
            'PRIMARY': root/PRIMARY, 'ORDINARY': root/baseline,
            'PRIMARY_AT_ORDINARY_COUNT': root/'countmatched'/(PRIMARY+'__AT_'+baseline+'_COUNT'),
            'ORDINARY_AT_PRIMARY_COUNT': root/'countmatched'/(baseline+'__AT_'+PRIMARY+'_COUNT')}
        rows = {}
        for name, folder in methods.items():
            receipt = official['results'][str(folder)]
            path = folder/'results/pedestrian_detailed.csv'
            if (folder/'official_equivalent.json').is_file():
                path = Path(json.loads((folder/'official_equivalent.json').read_text())[
                    'identical_prediction_reference'])/'results/pedestrian_detailed.csv'
            with path.open() as stream:
                row = next(r for r in csv.DictReader(stream) if r['seq'] == 'COMBINED')
            metrics = {n: 100*float(row[n+'___AUC']) for n in ('HOTA', 'DetA', 'AssA')}
            assert abs(metrics['HOTA']-receipt['HOTA_unrounded']) < 1e-8
            rows[name] = dict(**metrics, IDSW=receipt['metrics']['IDSW'], Dets=receipt['metrics']['Dets'],
                complete_official_detail_sha256=file_sha(path))
        assert rows['PRIMARY_AT_ORDINARY_COUNT']['Dets'] == rows['ORDINARY']['Dets']
        assert rows['ORDINARY_AT_PRIMARY_COUNT']['Dets'] == rows['PRIMARY']['Dets']
        differences = {}
        for scope, a, b in (
            ('RAW', 'PRIMARY', 'ORDINARY'),
            ('AT_ORDINARY_COUNT', 'PRIMARY_AT_ORDINARY_COUNT', 'ORDINARY'),
            ('AT_PRIMARY_COUNT', 'PRIMARY', 'ORDINARY_AT_PRIMARY_COUNT')):
            differences[scope] = {m: rows[a][m]-rows[b][m] for m in ('HOTA', 'DetA', 'AssA', 'IDSW', 'Dets')}
        settings[pair] = dict(source_target=PAIRS[pair], methods=rows, differences=differences)
    result = dict(status='COMPLETE_OLD_JOINT_OFFICIAL_ASSOCIATION_DIAGNOSIS', settings=settings,
        cohort=OUT.name, fixed_display_ordinary=baseline, primary=PRIMARY,
        all_582_evaluated_and_reproduced_before_this_diagnosis=True,
        all_six_primary_AssA_lower_at_both_donor_counts=True,
        manual_GT_opened=False, new_models_or_predictions_or_evaluations=False,
        official_DetA_AssA_are_unrounded_AUC_percent=True,
        HOTA_is_mean_over_thresholds_not_sqrt_of_the_two_published_means=True,
        IDSW_is_CLEAR_at_IoU_point5_not_HOTA_AssA=True,
        flips_and_IoU_point5_candidate_membership_not_additive_HOTA_causes=True,
        old_joint_teacher_supervision_differs_from_current_native_binary_studies=True,
        no_parameters_selected_and_all_data_DEVELOPMENT=True, goal_status='ACTIVE')
    assert all(s['differences'][scope]['AssA'] < 0 for s in settings.values()
        for scope in ('AT_ORDINARY_COUNT', 'AT_PRIMARY_COUNT'))
    atomic_json(PUBLIC/'joint_observation_official_association_20261124.json', result)
    print(json.dumps(dict(status=result['status'], at_ordinary_count={p: s['differences']['AT_ORDINARY_COUNT']
        for p, s in settings.items()})), flush=True)


if __name__ == '__main__':
    main()
