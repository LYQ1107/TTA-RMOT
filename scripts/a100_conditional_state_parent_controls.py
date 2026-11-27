"""Retained adequate parent KD controls, fixed before current quality access.

Prediction generation from already sealed primary units, with a small frozenR
forward on GPU3 in the original deterministic configuration. This does
not change or restart the four-GPU primary. Each invocation computes a finite
snapshot of ready units; it never polls or creates an observer process. All
supplemental masks must seal before this study's isolated evaluation.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha

PRIMARY_ROOT=ROOT/'a100_runs/visual_classical_20261121/six_direction_20261122/conditional_state_20261127/primary'
OUT=PRIMARY_ROOT.parent/'parent_control_supplement'
DESIGN=OUT/'design_registration.json'
PLAN=OUT/'registration.json'
PUBLIC=ROOT/'docs/a100_six_direction_20261122'
STEM='conditional_state_parent_controls_20261127'
PRIMARY='CONDITIONAL_STATE_MEMORY'
CONTROLS=('PARENT_SOFT_KD','PARENT_POOLED','PARENT_SOFT_SHARED_MEMORY')
PARENT_ALIAS={'PARENT_SOFT_KD':'UNIFORM_SOFT_KD288','PARENT_POOLED':'UNIFORM_POOLED'}
NEW_SOURCES=('scripts/a100_conditional_state_parent_controls.py',
    'scripts/a100_conditional_state_parent_statistics.py',
    'scripts/a100_conditional_state_parent_contract.py')


def prepare():
    from flex_uda.a100.mechanism import generation_guard,KEYS
    guard=generation_guard()
    if PLAN.exists():raise FileExistsError(PLAN)
    design=json.loads(DESIGN.read_text());parent=json.loads((PRIMARY_ROOT/'registration.json').read_text())
    assert file_sha(PRIMARY_ROOT/'registration.json')==design['parent_primary_registration_sha256']
    assert not (PRIMARY_ROOT/'quality_results.json').exists(),'Do not redefine this supplement after current quality'
    contract=ROOT/'a100_runs/visual_classical_20261121/supervisor/conditional_state_parent_contract_20261127/result.json'
    assert json.loads(contract.read_text())['status'].startswith('PASS_')
    sources=dict(parent['source_sha256'])
    for name,sha in sources.items():assert file_sha(ROOT/name)==sha,name
    sources.update({name:file_sha(ROOT/name) for name in NEW_SOURCES})
    inputs=dict(parent['input_sha256']);inputs.update(design['input_sha256'])
    for name,sha in inputs.items():assert file_sha(name)==sha,name
    for target in {c['target'] for c in parent['cases']}:
        p=ROOT/'results/runs/multidomain_generality_20261101'/KEYS[target]/'metadata.json';inputs[str(p)]=file_sha(p)
    old_provider=PUBLIC/'fullscope_visual_six_20261126_statistics.json'
    lookup={(c['pair'],c['video']):c for c in design['cases']}
    cases=[]
    for case in parent['cases']:
        key=case['pair'],case['evaluated_video'];cases.append(dict(case,retained=lookup[key]))
    assert len(cases)==148 and set(CONTROLS)==set(design['controls'])
    plan=dict(status='REGISTERED_RETAINED_PARENT_CONTROLS_BEFORE_GENERATION_OR_NEW_QUALITY',
        primary=PRIMARY,controls=CONTROLS,cases=cases,source_sha256=sources,input_sha256=inputs,
        design_sha256=file_sha(DESIGN),parent_registration_sha256=file_sha(PRIMARY_ROOT/'registration.json'),
        contract_sha256=file_sha(contract),old_statistics_provider_sha256=file_sha(old_provider),
        old_statistics_provider=old_provider.name,primary_statistics_provider='conditional_state_primary_20261127_statistics.json',
        no_new_teacher_fits_seeds_or_candidates=True,CPU_threads=2,R_forward_GPU=3,GPU_memory_fraction=.05,
        inference='Load saved primaryR only on GPU3 under the original deterministic settings, use recorded final channel values and original memory covariates/event scope/protocol. No S/referring model is refitted.',
        design_execution_refinement='The earlier design intendedCPU-only inference. Before numerical execution this registration specifies one small frozenR GPU forward for faithful replay of the original GPU decoder; state inference and CountMatched remainCPU. No new labels, weights, neural fits or teacher calls.',
        precision='Reconstruct existing continued-soft shared-memory probabilities from saved artifacts; require max absolute probability error<=2e-6 and exactly identical threshold masks. Record any mismatch and fail instead of silently changing the comparator.',
        source_scope='Consume only complete sealed original-primary VIDEO units; all original48 sources remain fixed. This separate study generates1036new masks, seals all before its own isolated matching.',
        references='Reuse primary raw and exact original UNIFORM_SOFT_KD288/UNIFORM_POOLED sufficient statistics. Ordinary parent weights already satisfy>=6000steps/properKL.005.',
        new_methods_per_unit=7,total_new_masks=1036,full_statistics_records=60,paired_effects=54,
        statistics='42 own HOTA records and18 immutable provider references, resolve exact JSON+SHA and recompute all60/54 with public NumPy only. Never average videoHOTA.',
        success='Primary raw and BOTH per-query/frame CountMatched summary effects positive versus all3retained controls in every complete setting, ADDITIONAL to original7-control study; no best-control promotion.',
        readiness='--generate-ready computes a finite ready-unit snapshot without waiting, no background observer. --evaluate only after all1036 own seals and original132/126 evaluation+reproduction complete.',
        existing_DEV_only=True,no_module_ablation=True,no_fresh_cold_cost_or_RSI_causal_claim=True,
        goal_status='ACTIVE',goal_complete=False)
    assert not guard;atomic_json(PLAN,plan)
    compact={k:v for k,v in plan.items() if k not in ('cases','input_sha256')}
    compact.update(video_units=148,setting_video_counts={p:sum(c['pair']==p for c in cases) for p in sorted({c['pair'] for c in cases})})
    atomic_json(PUBLIC/(STEM+'_registration.json'),compact)
    print(json.dumps(dict(status=plan['status'],units=148,new_masks=1036,sources=len(sources))),flush=True)


def read_plan():
    plan=json.loads(PLAN.read_text())
    for name,sha in plan['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    for name,sha in plan['input_sha256'].items():assert file_sha(name)==sha,name
    assert file_sha(PRIMARY_ROOT/'registration.json')==plan['parent_registration_sha256']
    plan['registration_sha256']=file_sha(PLAN)
    return plan


def generate_unit(case,plan,guard):
    import numpy as np
    import torch
    from scipy.special import expit
    from flex_uda.a100.mechanism import metadata,PAIRS
    from flex_uda.active.conditional_state_learning import StateObserver,scoped_bound
    from scripts.a100_conditional_state_engineering import source_events
    from scripts.a100_conditional_state_primary import probability_scores
    from scripts.a100_factor_rsi_primary_quality import subset_metadata,projection
    from flex_uda.presence.predictions import verify_predictions,seal_mask
    source=PRIMARY_ROOT/'units'/case['pair']/case['evaluated_video']
    folder=OUT/'units'/case['pair']/case['evaluated_video'];folder.mkdir(exist_ok=True,parents=True)
    done_path=folder/'generation_complete.json'
    if done_path.exists():
        done=json.loads(done_path.read_text());assert done['registration_sha256']==plan['registration_sha256']
        for name,sha in done['input_sha256'].items():assert file_sha(name)==sha,name
        for task in done['tasks']:verify_predictions(task['folder'])
        return done
    started=time.perf_counter();source_done=json.loads((source/'generation_complete.json').read_text())
    assert source_done['registration_sha256']==plan['parent_registration_sha256'] and source_done['GT_open_attempts']==0
    assert len(source_done['tasks'])==22
    fit=source/'primary';receipt=json.loads((fit/'completion.json').read_text())
    assert file_sha(fit/'latest.pt')==receipt['checkpoint_sha256']
    assert file_sha(fit/'memory_inputs.npz')==receipt['memory_inputs_sha256']
    checkpoint=torch.load(fit/'latest.pt',map_location='cpu',weights_only=False)
    assert checkpoint['registration_sha256']==plan['parent_registration_sha256'] and len(checkpoint['rounds'])==4
    old=case['retained'];obs=np.load(case['observations'])
    old_receipt=json.loads(Path(old['old_generation']).read_text())['ordinary_fits']['UNIFORM_SOFT_KD288']
    assert old_receipt['total_steps']>=6000 and old_receipt['proper_teacher_excess_CE']<=.005
    bank=metadata(case['target'])['videos'][case['evaluated_video']]
    assert np.array_equal(obs,np.asarray(bank['observations']))
    nq=len(json.loads(Path(case['temporal_queries']).read_text())['queries']);n=len(obs)
    context=SimpleNamespace(n=n,video=SimpleNamespace(obs=obs,bank=bank))
    events,provenance=source_events(context,case)
    assert len(events)==case['logical_budget']
    with np.load(fit/'memory_inputs.npz') as arrays:change,base=arrays['change'].copy(),arrays['base'].copy()
    assert change.shape==(nq,n,8) and base.shape==(nq,n)
    observer=StateObserver().to('cuda:0');observer.load_state_dict(checkpoint['observer']);observer.eval()
    torch.cuda.synchronize();forward_started=time.perf_counter()
    with torch.no_grad():
        keep=observer(torch.as_tensor(change.reshape(-1,8),device='cuda:0'),torch.as_tensor(base.reshape(-1),device='cuda:0')).cpu().numpy().reshape(nq,n)
        reproduced_channel=observer.reliability().cpu().numpy()
    torch.cuda.synchronize();observer_forward_seconds=time.perf_counter()-forward_started
    channel=np.asarray(checkpoint['rounds'][-1]['channel'],dtype='float32')
    channel_error=float(np.max(np.abs(channel-reproduced_channel)));assert channel_error<=2e-7
    def decode(scores):
        return scoped_bound(expit(scores),keep,obs,events,channel,
            protocol=receipt['protocol'],existence_prior=.5,iterations=8)['posterior']
    soft_source=source/'predictions/SOFT_KD/scores.npy'
    reference_source=source/'predictions/SOFT_SHARED_MEMORY/scores.npy'
    expected_mask=np.load(source/'predictions/SOFT_SHARED_MEMORY/mask.npy')
    reproduced=decode(np.load(soft_source))
    reference=expit(np.load(reference_source).astype('float64'))
    probability_error=float(np.max(np.abs(reproduced-reference)))
    mask_changes=int(np.count_nonzero((reproduced>.5)!=expected_mask))
    if probability_error>2e-6 or mask_changes:
        atomic_json(folder/'precision_failure.json',dict(status='SHARED_R_READOUT_REPRODUCTION_FAILED',probability_error=probability_error,
            changed_threshold_rows=mask_changes,channel_error=channel_error,GT_open_attempts=len(guard)))
        raise FloatingPointError(('Shared-R readout mismatch',case['pair'],case['evaluated_video'],probability_error,mask_changes))
    primary_source=source/'predictions'/PRIMARY
    verify_predictions(primary_source)
    scores={PRIMARY:np.load(primary_source/'scores.npy'),
        'PARENT_SOFT_KD':np.load(old['old_soft_scores']),
        'PARENT_POOLED':np.load(old['old_pool_scores'])}
    masks={PRIMARY:np.load(primary_source/'mask.npy'),
        'PARENT_SOFT_KD':np.load(old['old_soft_mask']),
        'PARENT_POOLED':np.load(old['old_pool_mask'])}
    for name in scores:
        assert scores[name].shape==masks[name].shape==(nq,n)
        assert np.array_equal(scores[name]>0,masks[name])
    scores['PARENT_SOFT_SHARED_MEMORY']=probability_scores(decode(scores['PARENT_SOFT_KD']))
    masks['PARENT_SOFT_SHARED_MEMORY']=scores['PARENT_SOFT_SHARED_MEMORY']>0
    md=subset_metadata(case['target'],case['evaluated_video']);tasks=[]
    def seal(name,mask,extra):
        path=folder/'predictions'/name;path.mkdir(exist_ok=True,parents=True)
        np.save(path/'mask.npy',mask)
        seal_mask(path,md,mask.flatten(),target=case['target'],model=PAIRS[case['pair']][0],method=name,
            GT_open_attempts=len(guard),extra=dict(registration_sha256=plan['registration_sha256'],
                fixed_sealed_inputs=True,no_new_training_teacher_or_seeds=True,**extra))
        tasks.append(dict(pair=case['pair'],video=case['evaluated_video'],method=name,folder=str(path)))
    shared='PARENT_SOFT_SHARED_MEMORY';path=folder/'predictions'/shared;path.mkdir(exist_ok=True,parents=True)
    np.save(path/'scores.npy',scores[shared]);seal(shared,masks[shared],dict(same_primaryR_and_scope=True))
    for rank,count in [(PRIMARY,c) for c in CONTROLS]+[(c,PRIMARY) for c in CONTROLS]:
        name=rank+'__AT_'+count+'_COUNT';mask=projection(scores[rank],masks[count],obs)
        seal(name,mask,dict(rank_method=rank,count_method=count,counts_from_GT=False,counts_equal_per_query_frame=True))
    assert len(tasks)==7 and not guard
    paths=[source/'generation_complete.json',fit/'completion.json',fit/'latest.pt',fit/'memory_inputs.npz',
        soft_source,reference_source,source/'predictions/SOFT_SHARED_MEMORY/mask.npy',
        primary_source/'scores.npy',primary_source/'mask.npy',primary_source/'predictions_seal.json']
    done=dict(status='SEALED_SEVEN_RETAINED_PARENT_CONTROL_OUTPUTS',pair=case['pair'],video=case['evaluated_video'],tasks=tasks,
        registration_sha256=plan['registration_sha256'],parent_registration_sha256=plan['parent_registration_sha256'],
        input_sha256={str(p):file_sha(p) for p in paths},GT_open_attempts=0,
        original_shared_readout_probability_max_error=probability_error,original_shared_threshold_changes=mask_changes,
        channel_max_error=channel_error,teacher_events=len(events),new_training_calls=0,new_teacher_calls=0,new_seeds=0,
        retained_parent_steps=old_receipt['total_steps'],retained_parent_excess_KL=old_receipt['proper_teacher_excess_CE'],
        observer_forward_GPU=3,observer_forward_seconds=observer_forward_seconds,
        wall_seconds=time.perf_counter()-started,references_from_sealed_original_raw_and_pool=True)
    atomic_json(done_path,done)
    print(json.dumps(dict(status=done['status'],pair=case['pair'],video=case['evaluated_video'],outputs=7,
        reference_probability_error=probability_error,reference_mask_changes=mask_changes)),flush=True)
    return done


def generate_ready():
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution=configure_before_cuda()
    import torch
    from flex_uda.a100.mechanism import generation_guard
    guard=generation_guard();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.05,0);plan=read_plan()
    # Finite computational batch, not a filesystem polling/monitor loop.
    ready=[c for c in plan['cases'] if (PRIMARY_ROOT/'units'/c['pair']/c['evaluated_video']/'generation_complete.json').exists()]
    for case in ready:generate_unit(case,plan,guard)
    completed=[];tasks=[]
    for case in plan['cases']:
        path=OUT/'units'/case['pair']/case['evaluated_video']/'generation_complete.json'
        if not path.exists():continue
        done=json.loads(path.read_text());assert done['GT_open_attempts']==0
        completed.append(dict(pair=case['pair'],video=case['evaluated_video']));tasks.extend(done['tasks'])
    assert not guard
    summary=dict(status='COMPLETE_READY_RETAINED_PARENT_GENERATION_BATCH',ready_source_units=len(ready),
        completed_units=len(completed),new_masks=len(tasks),total_units=148,total_masks=1036,
        no_new_teacher_training_GT_or_seeds=True,observer_forward_only_GPU=3,execution=execution,goal_complete=False)
    atomic_json(OUT/'latest_generation_batch.json',summary)
    if len(completed)==148:
        from flex_uda.presence.predictions import verify_predictions
        assert len(tasks)==1036
        for task in tasks:verify_predictions(task['folder'])
        atomic_json(OUT/'all1036_sealed_before_GT.json',dict(status='ALL1036_PARENT_CONTROL_OUTPUTS_SEALED_BEFORE_OWN_GT',
            tasks=tasks,GT_open_attempts=0,registration_sha256=plan['registration_sha256'],parent_registration_sha256=plan['parent_registration_sha256']))
    print(json.dumps(summary),flush=True)


def evaluate():
    plan=read_plan();barrier=json.loads((OUT/'all1036_sealed_before_GT.json').read_text())
    assert barrier['GT_open_attempts']==0 and len(barrier['tasks'])==1036
    assert barrier['registration_sha256']==plan['registration_sha256']
    original_reproduction=json.loads((PUBLIC/'conditional_state_primary_20261127_reproduction.json').read_text())
    assert original_reproduction['status']=='PASS_INDEPENDENT_ALL132_HOTA_AND126_EFFECTS'
    from flex_uda.a100.evaluation_reuse import evaluate as official
    from scripts.a100_conditional_state_parent_statistics import collect,reproduce
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(lambda task:official(task['folder'],task['pair']),barrier['tasks']):pass
    collect();reproduce()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--generate-ready',action='store_true');parser.add_argument('--evaluate',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.generate_ready:generate_ready()
    elif args.evaluate:evaluate()
    else:parser.error('Choose --prepare, --generate-ready or --evaluate')
