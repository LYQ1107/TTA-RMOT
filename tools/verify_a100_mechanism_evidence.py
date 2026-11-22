"""Recompute published20261121 metrics from small JSONs, without data or GPU."""
import argparse
import json
from pathlib import Path
import numpy as np


def load(root,relative):
    return json.loads((root/relative).read_text())


def close(a,b):
    assert np.allclose(a,b,rtol=0,atol=1e-10),(a,b)


def hota(numerator,denominator,weights):
    return 100*np.sqrt((weights@numerator)/np.maximum(weights@denominator,1)).mean(axis=-1)


def verify(public):
    report=load(public,'analysis/official_results.json')
    statistics=load(public,'analysis/official_sufficient_statistics.json')['methods']
    for name,data in statistics.items():
        num,den=np.array(data['numerator']),np.array(data['denominator'])
        pn,pd=np.array(data['parent_numerator']),np.array(data['parent_denominator'])
        close(hota(num,den,np.ones(len(num))),report['methods'][name]['HOTA'])
        close(hota(pn,pd,np.ones(len(pn))),report['methods'][name]['HOTA'])
        assert np.allclose(num.sum(0),pn.sum(0),rtol=1e-12,atol=1e-8)
        close(den.sum(0),pd.sum(0))
    for pair,data in report['comparisons'].items():
        a,b=pair.split('_vs_');sa,sb=statistics[a],statistics[b]
        for unit,prefix in (('original_parent_video','parent_'),('evaluation_clip','')):
            an,ad=np.array(sa[prefix+'numerator']),np.array(sa[prefix+'denominator'])
            bn,bd=np.array(sb[prefix+'numerator']),np.array(sb[prefix+'denominator'])
            n=len(an);weights=np.random.default_rng(0).multinomial(n,np.ones(n)/n,size=10000)
            close(np.percentile(hota(an,ad,weights)-hota(bn,bd,weights),[2.5,97.5]),data[unit]['CI95'])
    prior=load(public,'analysis/prior_parent_sensitivity.json');previous={}
    old=public.parent/'a100_three_stage_20261120'
    for name in prior['points']:
        path='stage2/'+('countmatched/' if '__' in name else '')+name+'/official_video_pooling.json'
        row=load(old,path);num=np.array(row['numerator_AssA_times_TP']);den=np.array(row['denominator_TP_plus_FN_plus_FP'])
        parents=sorted({v.rsplit('_c',1)[0] for v in row['videos']})
        grouped=[np.stack([x[[i for i,v in enumerate(row['videos']) if v.rsplit('_c',1)[0]==p]].sum(0) for p in parents]) for x in (num,den)]
        previous[name]=dict(evaluation_clip=(num,den),original_parent_video=tuple(grouped))
    for pair,data in prior['comparisons'].items():
        a,b=pair.split('_vs_')
        for unit in ('original_parent_video','evaluation_clip'):
            an,ad=previous[a][unit];bn,bd=previous[b][unit];n=len(an)
            weights=np.random.default_rng(0).multinomial(n,np.ones(n)/n,size=10000)
            close(np.percentile(hota(an,ad,weights)-hota(bn,bd,weights),[2.5,97.5]),data[unit]['CI95'])
    rows=load(public,'analysis/bank_labeled_rows.json')['rows']
    bank=load(public,'analysis/bank_results.json')
    manifest=load(public,'evidence_manifest.json')['samples']
    assert len(rows)==len(manifest)==300
    for pair,data in bank['pairs'].items():
        for split in ('calibration','evaluation'):
            selected=[r for r in rows if r['pair']==pair and r['split']==split]
            truth=np.array([r['truth'] for r in selected])
            for method,expected in data['splits'][split]['methods'].items():
                p=np.array([r['posterior'][method] for r in selected])
                close(((p>.5)==truth).mean(),expected['accuracy'])
                close(((p-truth)**2).mean(),expected['Brier'])
        cal={r['parent_video'] for r in rows if r['pair']==pair and r['split']=='calibration'}
        ev={r['parent_video'] for r in rows if r['pair']==pair and r['split']=='evaluation'}
        assert not cal.intersection(ev)
        selected=[r for r in rows if r['pair']==pair and r['split']=='evaluation']
        groups=sorted(ev);index=np.array([groups.index(r['parent_video']) for r in selected])
        truth=np.array([r['truth'] for r in selected])
        for comparison in data['comparisons']:
            a=np.array([r['posterior'][comparison['first']] for r in selected])
            b=np.array([r['posterior'][comparison['second']] for r in selected])
            delta=((a>.5)==truth).astype(float)-((b>.5)==truth)
            reduction=(b-truth)**2-(a-truth)**2
            close(delta.mean(),comparison['delta_accuracy']);close(reduction.mean(),comparison['Brier_reduction'])
            if comparison['accuracy_CI95'] is None:continue
            weights=np.random.default_rng(0).multinomial(len(groups),np.ones(len(groups))/len(groups),size=10000)[:,index]
            close(np.percentile(weights@delta/weights.sum(1),[2.5,97.5]),comparison['accuracy_CI95'])
            close(np.percentile(weights@reduction/weights.sum(1),[2.5,97.5]),comparison['Brier_reduction_CI95'])
    ledger=load(public,'analysis/study_cost_ledger.json')
    assert ledger['teacher']['responses']==900 and ledger['teacher']['new_calls']<=900
    assert ledger['historical_original_view_input_parity']['identical_inputs']==300
    assert load(public,'fixed_replay/C4/reference_equivalence.json')['status']=='PASS'
    assert load(public,'fixed_replay/C4/new_original_view_equivalence.json')['all_sealed_prediction_hashes_equal']
    profile=load(public,'profile/completion.json')
    assert all(all(r['bitwise_equal_to_first'].values()) for r in profile['records'])
    return dict(status='PASS_PUBLIC_EVIDENCE_RECOMPUTATION',official_methods=len(statistics),
        official_comparisons=len(report['comparisons']),bank_positions=len(rows),
        prior_stage_parent_sensitivity_comparisons=len(prior['comparisons']),
        bootstrap_samples=10000,bootstrap_seed=0,
        scope='JSON transcription/pooling and grouped uncertainty reproduced; this does not turn development data into an independent test',
        raw_GPU_and_GT_assets_not_required=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--public',type=Path,default=Path(__file__).resolve().parents[1]/'docs/a100_mechanism_20261121')
    p.add_argument('--receipt',type=Path);args=p.parse_args();result=verify(args.public)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True,exist_ok=True);args.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
