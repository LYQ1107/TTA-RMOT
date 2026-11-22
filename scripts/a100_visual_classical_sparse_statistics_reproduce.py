"""Reproduce published phase statistics with NumPy only, no raw assets or GPU.

This verifies arithmetic, not independent generalization or correction for
development iteration. Official HOTA uses pooled TP-weighted statistics.
"""
import argparse
import base64
import json
from pathlib import Path
import zlib
import numpy as np


def pooled(row, weights):
    return 100*np.sqrt((weights @ np.asarray(row['numerator'])) /
        np.maximum(weights @ np.asarray(row['denominator']), 1)).mean(axis=-1)


def methods(label):
    if label.startswith('COUNT_OWN_'):
        first, second = label.removeprefix('COUNT_OWN_').split('_vs_')
        return first+'__AT_'+first+'_COUNT', second+'__AT_'+first+'_COUNT'
    if label.startswith('COUNT_BASELINE_'):
        first, second = label.removeprefix('COUNT_BASELINE_').split('_vs_')
        return first+'__AT_'+second+'_COUNT', second+'__AT_'+second+'_COUNT'
    return label.split('_vs_')


def recompute(left, right):
    parents = left[0]['parents']
    assert all(a['parents'] == b['parents'] == parents for a, b in zip(left, right))
    n = len(parents)
    rng = np.random.default_rng(0)
    weights = rng.multinomial(n, np.ones(n)/n, size=10000)
    draws = np.stack([pooled(a, weights)-pooled(b, weights) for a, b in zip(left, right)])
    point = np.mean([pooled(a, np.ones(n))-pooled(b, np.ones(n)) for a, b in zip(left, right)])
    result = dict(delta_HOTA=float(point),
        fixed_neural_seeds_parent_CI95=np.percentile(draws.mean(0), [2.5, 97.5]).tolist())
    if len(left) > 1:
        seed_weights = rng.multinomial(len(left), np.ones(len(left))/len(left), size=10000)
        result['parent_and_neural_seed_sensitivity_CI95'] = np.percentile(
            (seed_weights*draws.T).sum(1)/len(left), [2.5, 97.5]).tolist()
    return result


def statistic_file(directory,filename,cache):
    path=(directory/filename).absolute();key=str(path)
    if key not in cache:cache[key]=json.loads(path.read_text())
    expand_statistic_archive(cache[key])
    return path,cache[key]


def expand_statistic_archive(statistics):
    """One shared lossless aggregate-row payload, no extra method records."""
    for row in statistics.values():
        if 'parent_rows_archive_zlib_base64' in row and 'parent_rows' not in row:
            decoded=json.loads(zlib.decompress(base64.b64decode(row['parent_rows_archive_zlib_base64'])))
            for name,values in decoded.items():
                restored=[]
                for value in values:
                    if isinstance(value,dict) and 'float64_le_shuffled' in value:
                        raw=base64.b64decode(value['float64_le_shuffled'])
                        bytes_=np.frombuffer(raw,dtype=np.uint8).reshape(8,-1).T.copy()
                        value=bytes_.reshape(-1).view('<f8').reshape(2,-1).tolist()
                    restored.append(value)
                statistics[name]['parent_rows']=restored


def parent_rows(row):
    """Decode only aggregate-count JSON, preserving its original numbers."""
    if 'parent_rows_zlib_base64' in row and 'parent_rows' not in row:
        row['parent_rows']=json.loads(zlib.decompress(base64.b64decode(row['parent_rows_zlib_base64'])))
    return row.get('parent_rows')


def pack_statistic_rows(statistics):
    """Lossless aggregate storage; HOTA, parents and record links stay readable."""
    expand_statistic_archive(statistics)
    result={};plain={};arrays={}
    for name,row in statistics.items():
        values=parent_rows(row)
        if values is None and 'numerator' in row:values=list(zip(row['numerator'],row['denominator']))
        if values is None:result[name]=row;plain[name]=row;continue
        base={k:v for k,v in row.items() if k not in ('parent_rows','parent_rows_zlib_base64',
            'parent_rows_archive_owner','parent_rows_archive_zlib_base64','parent_rows_archive_codec','numerator','denominator')}
        arrays[name]=values;plain[name]=dict(base,parent_rows=values)
        encoded=base64.b64encode(zlib.compress(json.dumps(values,separators=(',',':')).encode(),9)).decode()
        result[name]=dict(base,parent_rows_zlib_base64=encoded)
    if arrays:
        owner=next(iter(arrays));bundle={}
        for name,row in plain.items():
            bundle[name]=dict({k:v for k,v in row.items() if k!='parent_rows'},parent_rows_archive_owner=owner) if name in arrays else row
        payload=json.dumps(arrays,separators=(',',':')).encode()
        bundle[owner]['parent_rows_archive_zlib_base64']=base64.b64encode(zlib.compress(payload,9)).decode()
        if len(json.dumps(bundle))<len(json.dumps(result)):result=bundle
        binary={}
        for name,values in arrays.items():
            encoded=[]
            for value in values:
                if isinstance(value,list) and all(isinstance(cell,float) for vector in value for cell in vector):
                    matrix=np.asarray(value,dtype='<f8')
                    shuffled=matrix.view(np.uint8).reshape(-1,8).T.copy().tobytes()
                    value=dict(float64_le_shuffled=base64.b64encode(shuffled).decode())
                encoded.append(value)
            binary[name]=encoded
        packed_binary={name:dict(row) for name,row in bundle.items()}
        packed_binary[owner]['parent_rows_archive_zlib_base64']=base64.b64encode(zlib.compress(json.dumps(binary,separators=(',',':')).encode(),9)).decode()
        packed_binary[owner]['parent_rows_archive_codec']='json-f64le-byte-shuffle-v1'
        if len(json.dumps(packed_binary))<len(json.dumps(result)):result=packed_binary
    if len(json.dumps(plain))<len(json.dumps(result)):result=plain
    return result


def resolve_parent_row(directory,filename,name,index,cache):
    path,payload=statistic_file(directory,filename,cache);row=payload[name]
    if 'reference' in row:
        ref=row['reference']
        return resolve_parent_row(path.parent,ref['file'],ref['record'],index,cache)
    rows=parent_rows(row)
    if rows is None:return [row['numerator'][index],row['denominator'][index]]
    value=rows[index]
    if isinstance(value,dict):
        ref=value['reference']
        return resolve_parent_row(path.parent,ref.get('file',path.name),ref['record'],ref['row'],cache)
    return value


def resolve_record(directory,filename,name,cache):
    path,payload=statistic_file(directory,filename,cache);row=payload[name]
    if 'reference' in row:
        ref=row['reference']
        return resolve_record(path.parent,ref['file'],ref['record'],cache)
    values=parent_rows(row)
    if values is not None:
        rows=[resolve_parent_row(path.parent,path.name,name,i,cache) for i in range(len(values))]
        return dict({k:v for k,v in row.items() if k not in ('parent_rows','parent_rows_zlib_base64',
            'parent_rows_archive_owner','parent_rows_archive_zlib_base64','parent_rows_archive_codec')},
            numerator=[r[0] for r in rows],denominator=[r[1] for r in rows])
    return row


def share_identical_statistics(statistics,filename):
    """Preserve every number/key; share identical records and parent rows."""
    result={};seen={};count=0
    for name,row in statistics.items():
        if 'reference' in row:result[name]=row;continue
        value=json.dumps(row,sort_keys=True,separators=(',',':'))
        if value in seen:
            result[name]=dict(reference=dict(file=filename,record=seen[value]));count+=1
        else:result[name]=row;seen[value]=name
    packed={};parents={}
    for name,row in result.items():
        if 'reference' in row or 'parent_rows' in row:packed[name]=row;continue
        rows=[]
        for index,(numerator,denominator) in enumerate(zip(row['numerator'],row['denominator'])):
            value=[numerator,denominator];key=json.dumps(value,separators=(',',':'))
            if key in parents:
                owner,position=parents[key];rows.append(dict(reference=dict(record=owner,row=position)))
            else:parents[key]=(name,index);rows.append(value)
        packed[name]=dict({k:v for k,v in row.items() if k not in ('numerator','denominator')},parent_rows=rows)
    if len(json.dumps(packed,separators=(',',':')))<len(json.dumps(result,separators=(',',':'))):result=packed
    return pack_statistic_rows(result),count


def main(directory, stem='sparse_observation'):
    statistics = json.loads((directory / (stem+'_parent_statistics.json')).read_text())
    references=0;external={str((directory/(stem+'_parent_statistics.json')).absolute()):dict(statistics)}
    encoded_records=sum('parent_rows_zlib_base64' in row or 'parent_rows_archive_owner' in row for row in statistics.values())
    expand_statistic_archive(statistics)
    parent_row_references=sum(sum(isinstance(value,dict) for value in (parent_rows(row) or [])) for row in statistics.values())
    for name,row in list(statistics.items()):
        references+=int('reference' in row)
        statistics[name]=resolve_record(directory,stem+'_parent_statistics.json',name,external)
    report = json.loads((directory / (stem+'_analysis.json')).read_text())
    maximum_error, effects = 0., 0
    def compare(a, b, name):
        nonlocal maximum_error
        error = float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        assert error < 1e-8, (name, error)
        maximum_error = max(maximum_error, error)
    for name, row in statistics.items():
        compare(pooled(row, np.ones(len(row['parents']))), row['HOTA'], name)
    for name, row in report['conditions'].items():
        for contrast, published in row['comparisons'].items():
            a, b = methods(contrast)
            computed = recompute([statistics[name+'/'+a]], [statistics[name+'/'+b]])
            for field, value in computed.items():
                compare(value, published[field], name+'/'+contrast+'/'+field)
            effects += 1
    for family, row in report['repeated'].items():
        if 'condition_keys' in row:
            keys = row['condition_keys']
        else:
            kind, pair = family.rsplit('_', 1)
            prefix = 'observation' if kind == 'OBSERVATION' else 'budget0.1'
            keys = [prefix+f'_seed{seed}_{pair}' for seed in row['neural_seeds']]
        for method, value in row['mean_HOTA'].items():
            compare(np.mean([statistics[k+'/'+method]['HOTA'] for k in keys]), value, family+'/'+method)
        for contrast, published in row['comparisons'].items():
            a, b = methods(contrast)
            computed = recompute([statistics[k+'/'+a] for k in keys], [statistics[k+'/'+b] for k in keys])
            for field, value in computed.items():
                compare(value, published[field], family+'/'+contrast+'/'+field)
            effects += 1
    result = dict(status='PASS_PUBLISHED_'+stem.upper()+'_STATISTICS_REPRODUCED',
        HOTA_records=len(statistics), effect_records=effects, maximum_absolute_error=maximum_error,
        bootstrap_samples=10000, bootstrap_seed=0, original_parent_resampling=True,
        neural_seed_sensitivity_reproduced=True, no_raw_assets_or_GPU_required=True,
        existing_public_statistic_references=references,
        shared_parent_row_references=parent_row_references,
        losslessly_encoded_aggregate_records=encoded_records,
        arithmetic_reproduction_is_not_independent_validation=True)
    (directory / (stem+'_statistical_reproduction.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path,
        default=Path(__file__).resolve().parents[1] / 'docs/a100_visual_classical_20261121')
    parser.add_argument('--stem', default='sparse_observation')
    args = parser.parse_args()
    main(args.directory, args.stem)
