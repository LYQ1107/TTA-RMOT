"""Independently reproduce public joint-study numbers using NumPy only."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.a100_visual_classical_sparse_statistics_reproduce import resolve_record,pooled,recompute

STEM='native_binary_quality_seed0_20261124'


def main(directory):
    directory=Path(directory);analysis=json.loads((directory/(STEM+'_analysis.json')).read_text())
    filename=STEM+'_parent_statistics.json';cache={};errors=[];records=effects=factorials=0
    primary=analysis['primary']
    for pair,setting in analysis['settings'].items():
        names=set(setting['raw_HOTA'])
        for effect in setting['comparisons'].values():names.update(effect['methods'])
        data={name:resolve_record(directory,filename,pair+'_seed0/'+name,cache) for name in names}
        assert len(data)==121
        for name,row in data.items():
            value=float(pooled(row,np.ones(len(row['parents']))));errors.append(abs(value-row['HOTA']))
            if name in setting['raw_HOTA']:errors.append(abs(value-setting['raw_HOTA'][name]))
            records+=1
        for effect in setting['comparisons'].values():
            first,second=effect['methods'];again=recompute([data[first]],[data[second]])
            errors.append(abs(again['delta_HOTA']-effect['delta_HOTA']))
            if len(data[first]['parents'])>=5:
                errors.extend(np.abs(np.asarray(again['fixed_neural_seeds_parent_CI95'])-
                                     effect['fixed_neural_seeds_parent_CI95']).tolist())
            effects+=1
        for scope,items in setting['factorial'].items():
            for effect in items.values():
                weights=effect['method_weights'];used={}
                for name,weight in weights.items():
                    key=name if scope=='raw' or name==primary else name+'__AT_'+primary+'_COUNT'
                    used[key]=weight
                n=len(data[next(iter(used))]['parents'])
                point=sum(w*float(pooled(data[k],np.ones(n))) for k,w in used.items())
                errors.append(abs(point-effect['delta_HOTA']))
                if n>=5:
                    rng=np.random.default_rng(0);counts=rng.multinomial(n,np.ones(n)/n,size=10000)
                    draws=sum(w*pooled(data[k],counts) for k,w in used.items())
                    errors.extend(np.abs(np.percentile(draws,[2.5,97.5])-effect['CI95']).tolist())
                else:assert effect['CI95'] is None
                factorials+=1
    maximum=max(errors);assert records==726 and maximum<1e-8
    result=dict(status='PASS_INDEPENDENT_NATIVE_BINARY_PUBLIC_STATISTICS_REPRODUCTION',
        statistical_records=records,paired_effects=effects,factorial_effects=factorials,
        maximum_absolute_error=float(maximum),neural_seeds=[0],
        no_raw_predictions_GT_models_Torch_GPU_or_refits=True,
        fixed_output_parent_bootstrap_not_additional_neural_seeds=True,
        aggregate_reproduction_does_not_establish_new_blind_generalization=True)
    (directory/(STEM+'_reproduction.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',default=str(ROOT/'docs/a100_six_direction_20261122'))
    main(parser.parse_args().directory)
