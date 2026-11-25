"""Independent NumPy-only reproduction; no Torch, GT or saved predictions."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.a100_visual_classical_sparse_statistics_reproduce import resolve_record, pooled, recompute
STEM = 'native_binary_utility_quality_seed0_20261124'


def main(directory):
    directory = Path(directory); analysis = json.loads((directory/(STEM+'_analysis.json')).read_text())
    assert analysis['cohort'] == 'native_binary_utility_quality_20261124'
    filename = STEM+'_parent_statistics.json'; cache = {}; errors = []; records = effects = 0
    for pair, setting in analysis['settings'].items():
        data = {n: resolve_record(directory, filename, pair+'_seed0/'+n, cache) for n in setting['HOTA']}
        assert len(data) == 54
        for name, row in data.items():
            value = float(pooled(row, np.ones(len(row['parents']))))
            errors.extend((abs(value-row['HOTA']), abs(value-setting['HOTA'][name])))
            records += 1
        for effect in setting['comparisons'].values():
            first, second = effect['methods']; again = recompute([data[first]], [data[second]])
            errors.append(abs(again['delta_HOTA']-effect['delta_HOTA']))
            if len(data[first]['parents']) >= 5:
                errors.extend(np.abs(np.asarray(again['fixed_neural_seeds_parent_CI95'])-
                    effect['fixed_neural_seeds_parent_CI95']).tolist())
            effects += 1
    maximum = max(errors)
    assert records == 324 and effects == 318 and maximum < 1e-8
    result = dict(status='PASS_INDEPENDENT_ALL_SIX_UTILITY_READOUT_STATISTICS_REPRODUCTION',
        statistical_records=records, paired_effects=effects, maximum_absolute_error=float(maximum),
        neural_seeds=[0], no_GT_models_Torch_GPU_predictions_or_refits=True,
        exact_parent_reuse_not_independent_replication=True,
        fixed_output_parent_bootstrap_not_additional_neural_seeds=True)
    (directory/(STEM+'_reproduction.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--directory',
        default=str(ROOT/'docs/a100_six_direction_20261122'))
    main(parser.parse_args().directory)
