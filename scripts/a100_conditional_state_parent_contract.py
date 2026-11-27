"""CPU-only parent-alias, projection, immutable-provider and seal contract."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
DEST=ROOT/'a100_runs/visual_classical_20261121/supervisor/conditional_state_parent_contract_20261127'
OWN=DEST/'registration.json'
SOURCES=('scripts/a100_conditional_state_parent_contract.py','scripts/a100_conditional_state_parent_controls.py',
         'scripts/a100_conditional_state_parent_statistics.py')
PARENT=ROOT/'a100_runs/visual_classical_20261121/six_direction_20261122/conditional_state_20261127/primary/registration.json'


def prepare():
    if OWN.exists():raise FileExistsError(OWN)
    parent=json.loads(PARENT.read_text())
    for name,sha in parent['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    atomic_json(OWN,dict(status='REGISTERED_RETAINED_PARENT_NUMERICAL_CONTRACT',
        source_sha256={n:file_sha(ROOT/n) for n in SOURCES},original48_source_sha256=parent['source_sha256'],
        synthetic_CPU_only=True,no_target_images_teacher_GPU_or_HOTA=True))
    print(json.dumps(dict(status='REGISTERED',sources=3)),flush=True)


def run():
    import numpy as np
    from scripts.a100_conditional_state_parent_controls import CONTROLS,PARENT_ALIAS
    from scripts.a100_conditional_state_parent_statistics import resolve,value,PARENT_ALIAS as STAT_ALIAS
    from scripts.a100_factor_rsi_primary_quality import projection
    from flex_uda.presence.predictions import seal_mask,verify_predictions
    plan=json.loads(OWN.read_text())
    for group in ('source_sha256','original48_source_sha256'):
        for name,sha in plan[group].items():assert file_sha(ROOT/name)==sha,name
    if (DEST/'result.json').exists():raise FileExistsError(DEST/'result.json')
    checks=[]
    assert PARENT_ALIAS==STAT_ALIAS=={'PARENT_SOFT_KD':'UNIFORM_SOFT_KD288','PARENT_POOLED':'UNIFORM_POOLED'}
    assert len(CONTROLS)==3 and (1+2*len(CONTROLS))*148==1036
    checks.append('retained_uniform_not_active_parent_aliases_and1036_output_count')
    obs=np.array([[2,7],[1,3],[1,7],[3,3]])
    scores={'MAIN':np.array([[.3,.2,-.4,.8],[-.1,.4,.6,-.7]]),
        'PARENT_SOFT_KD':np.array([[.1,-.3,.8,-.6],[.7,.5,-.8,.2]]),
        'PARENT_POOLED':np.array([[.6,-.1,.6,-.1],[-.2,.9,-.2,.9]]),
        'PARENT_SOFT_SHARED_MEMORY':np.array([[-.4,.2,.7,.3],[.1,-.7,-.3,.4]])}
    masks={k:v>0 for k,v in scores.items()}
    with tempfile.TemporaryDirectory(prefix='parent-contract-',dir=DEST) as directory:
        root=Path(directory);paths=[]
        md=dict(episodes=[dict(video_id='synthetic',expression_id=str(q),start=q*4,stop=(q+1)*4) for q in range(2)],
            videos={'synthetic':dict(lines=[f'{int(f)},{int(t)},0,0,1,1,1' for f,t in obs])})
        bundle=[('PARENT_SOFT_SHARED_MEMORY',masks['PARENT_SOFT_SHARED_MEMORY'])]
        for rank,count in [('MAIN',c) for c in CONTROLS]+[(c,'MAIN') for c in CONTROLS]:
            actual=projection(scores[rank],masks[count],obs);expected=np.zeros_like(actual)
            for q in range(2):
                for frame in set(obs[:,0]):
                    rows=[i for i,row in enumerate(obs) if row[0]==frame]
                    k=sum(bool(masks[count][q,i]) for i in rows)
                    chosen=sorted(rows,key=lambda i:(-scores[rank][q,i],i))[:k]
                    expected[q,chosen]=True
            assert np.array_equal(actual,expected)
            bundle.append((rank+'__AT_'+count+'_COUNT',actual))
        assert len(bundle)==7
        for name,mask in bundle:
            path=root/name;path.mkdir();seal_mask(path,md,mask.flatten(),target='synthetic',model='fixture',method=name,GT_open_attempts=0)
            receipt=verify_predictions(path);assert receipt['episode_count']==2 and receipt['prediction_rows']==int(mask.sum());paths.append(path)
        checks.append('seven_masks_independent_per_query_frame_topk_and_sealed_MOT_counts')
        first=verify_predictions(paths[0])['episodes'][0];file=Path(first['prediction_path']);original=file.read_bytes();file.write_bytes(original+b'changed\n')
        try:verify_predictions(paths[0])
        except ValueError:checks.append('changed_prediction_detected_after_seal')
        else:raise AssertionError('Changed predictions accepted')
        file.write_bytes(original)
        record=dict(numerator=[3.]*19,denominator=[5.]*19,HOTA=100*np.sqrt(.6),videos=25)
        provider=root/'provider.json';provider.write_text(json.dumps(dict(records={'C1/UNIFORM_SOFT_KD288':record})))
        digest=hashlib.sha256(provider.read_bytes()).hexdigest()
        reference=dict(reference=dict(file=provider.name,key='C1/UNIFORM_SOFT_KD288',sha256=digest))
        actual=resolve(reference,root=root);assert actual==record and abs(value(actual)-100*np.sqrt(.6))<1e-12
        checks.append('exact_SHA_and_key_resolve_original_adequate_parent_statistics')
        provider.write_text(provider.read_text()+' ')
        try:resolve(reference,root=root)
        except ValueError:checks.append('changed_provider_rejected_even_if_JSON_values_same')
        else:raise AssertionError('Changed statistics provider accepted')
        bad=dict(reference=dict(file='provider.json',key='self',sha256='synthetic-loop'))
        try:resolve(bad,root=root,cache={('provider.json','synthetic-loop'):dict(records={'self':bad})})
        except ValueError:checks.append('cyclic_provider_reference_rejected')
        else:raise AssertionError('Reference cycle accepted')
        wrong=dict(reference=dict(file='../outside.json',key='x',sha256='x'))
        try:resolve(wrong,root=root)
        except AssertionError:checks.append('provider_must_be_explicit_public_basename')
        else:raise AssertionError('Non-public provider accepted')
    for group in ('source_sha256','original48_source_sha256'):
        for name,sha in plan[group].items():assert file_sha(ROOT/name)==sha,name
    receipt=dict(status='PASS_RETAINED_PARENT_ALIASES_PROJECTIONS_SEALS_AND_PROVIDERS',checks=checks,
        original48sources_unchanged=True,source_sha256=plan['source_sha256'],synthetic_CPU_only=True,
        target_GT_reads=0,new_teacher_calls=0,neural_fits=0,GPU_calls=0,no_real_HOTA_or_contribution_claim=True,goal_status='ACTIVE')
    atomic_json(DEST/'result.json',receipt);print(json.dumps(dict(status=receipt['status'],checks=len(checks))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else run()
