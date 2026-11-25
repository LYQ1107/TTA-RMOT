"""New image-evidence/teacher-anchored primary and8 adequate strong controls.

Four GPU roles, seed0. All75 current-study predictions seal before isolated
GT evaluation, then independent75HOTA/72effects. Not an ablation matrix.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.a100_six_direction import OUT as SIX
from scripts.a100_residual_rsi_primary import PLAN as OLD_PLAN,registration as old_registration
from scripts.a100_evidence_visual_inputs import OUT as PIXEL

OUT=SIX/'evidence_visual_primary_20261125'
PLAN=OUT/'registration.json'
PUBLIC=ROOT/'docs/a100_six_direction_20261122'
STEM='evidence_visual_primary_20261125'
PRIMARY='EVIDENCE_VISUAL_RSI'
COMPARATORS=('SOFT_KD','HARD_KD','IPW_KD','SSL_KD','R_DIRECT','SOFT_POOLED','UNIFORM_SOFT_KD','UNIFORM_POOLED')
METHODS=(PRIMARY,*COMPARATORS)
SOURCES=('scripts/a100_evidence_visual_primary.py','flex_uda/active/evidence_visual_learning.py',
    'scripts/a100_evidence_visual_inputs.py','scripts/a100_pixel_continuity_audit.py',
    'scripts/a100_primary_statistics_common.py','scripts/a100_evidence_visual_contract.py')


def prepare():
    import numpy as np
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.active.evidence_visual_learning import make
    guard=generation_guard();assert not PLAN.exists()
    old=old_registration();pairs=json.loads((PIXEL/'completion.json').read_text())
    contract=ROOT/'a100_runs/visual_classical_20261121/supervisor/evidence_visual_contract_20261125.json'
    assert json.loads(contract.read_text())['status']=='PASS_GT_FREE_NEW_VISUAL_LEARNING_CONTRACT'
    cases=[]
    for case,record in zip(old['cases'],pairs['cases']):
        assert case['pair']==record['pair'] and case['evaluated_video']==record['video']
        path=PIXEL/case['pair']/'pixel_pairs.npz';assert file_sha(path)==record['pair_sha256']
        cases.append(dict(case,pixel_pairs_path=str(path),pixel_pairs_sha256=record['pair_sha256'],pixel_pairs=record['pairs']))
    sizes={}
    mean,std=np.zeros(9352,dtype='float32'),np.ones(9352,dtype='float32')
    for key,kw in (('S',{}),('R',dict(outputs=2)),('ordinary288',dict(kind='plain',width=288)),('SSL288',dict(width=288))):
        model=make(mean,std,'cpu',**kw);sizes[key]=sum(p.numel() for p in model.parameters())
    assert sizes['ordinary288']>=sizes['S']+sizes['R']
    sources=dict(old['source_sha256']);sources.update({n:file_sha(ROOT/n) for n in SOURCES})
    plan=dict(status='REGISTERED_NEW_EVIDENCE_ANCHORED_VISUAL_RSI_PRIMARY',primary=PRIMARY,comparators=COMPARATORS,cases=cases,
        source_sha256=sources,original_pool_registration_sha256=file_sha(OLD_PLAN),shared_pixel_completion_sha256=file_sha(PIXEL/'completion.json'),
        numerical_contract_sha256=file_sha(contract),parameter_counts=sizes,
        hypothesis='Weak image-supported content learning, exact external-teacher anchors, and pre-answer student-error-driven observation can improve RMOT beyond adequate larger ordinary KD.',
        human_visual_inspiration='Maintain object content across a short supported image displacement, retain separate query/state context, and inspect what the current understanding is likely to get wrong.',
        classical_ideas='Conditional invariance versus state-dependent decisions, nonnegative proper scoring, variance/covariance anti-collapse and positive-support adaptive experimental design. Existing ideas, not new theorems.',
        candidate_components=['Query-independent content learning from weak pixel-cycle pairs, while full9352Dquery/current-state information remains in a gated neural solver.',
            'External-evidence anchored learning: original teacher probabilities and positive weights; final observed-teacher properKL<=.005. Never optimize old signed point-targets.',
            'Two actual trainableR/S models: R predicts pre-answer STUDENT squared error; currentSgradientgeometry andR predictions change next acquisition; new external answers retrainR.'],
        image_pretext='VICReg-inspired25/25/1 dividedby50, weight.1. No hard different-person negatives. Only content features align, never full query membership scores.',
        prior_work=[dict(url='https://arxiv.org/abs/2105.04906',scope='VICReg variance/invariance/covariance regularization is credited prior work; no code copied.'),
            dict(url='https://arxiv.org/html/2511.16043',scope='Dynamic dual-role curriculum inspiration; majority reward is not binary semantic truth.'),
            dict(url='https://arxiv.org/html/2610.10782',scope='Actor-dependent verifiable curriculum inspiration, not its seed labels or a universal scaling theorem.')],
        evidence_limit='Pixel cycles can follow background/repeatedtexture; they are weak image pretext, not certified identity or semantic groundtruth. No permanent TrackIDmembership, source masks, hand query parser or GT gating.',
        teacher='OriginalQwenT0 and all original answers frozen. R is a small learned visual observer, not a secretly fine-tuned Qwen. No response rewriting or synthetic semantic labels.',
        training='Allseed0/Adam.0001/clip10/cyclic512; boundedsoftsign8raw/(8+absraw), polynomial derivative. Saux500+500/500/500/4000; Raux500+four500. Rimagepretext; Scontentalignment. Finalteacher-onlyprojection available equally to primary andSSL controls untilproperKL.005,min6000,max16000. No GT checkpoint/rate selection.',
        capacity='Main R/Swidth128 with separate768->128->64imagecontent. Ordinaryflatwidth288 has at least combinedR+Sparametercount. SSLcontrolwidth288 has same raw9352/pixelpairs and VICRegpretext, adequate teacher projection.',
        acquisition='Same declared104/4068/95finite pools and perquery56/1891/43logical budgets; fourfrontloaded stages; withreplacement.20uniform positive support, repeatedactions explicit. No assertion of risk unbiasedness or support outsidepool.',
        auxiliary='Same100target-collection3viewaux events for everylearner; transductive. Imagepairs from unlabeled first-video frames only, sameinputs available to all controls.',
        controls={'SOFT_KD':'Adequate larger288flat soft teacher BCE, sameactual questions and phase order.',
            'HARD_KD':'Adequate larger288flat originalteacher probabilities thresholdedat.5, includingaux.',
            'IPW_KD':'Sameadequate largerhead, positive recorded1/(n*pi) queryweights, targets unchanged.',
            'SSL_KD':'Larger visual architecture, sameall pixelpairs andVICReg pretext, directteacherBCE, no dual-model loop. Finalprojection keeps original adequacy.',
            'R_DIRECT':'Actual finalobserver meanhead, never promoted posthoc.',
            'SOFT_POOLED':'Arithmeticprobability TrackIDpooling of adequate sameQAsoftKD, retain even for timevarying queries.',
            'UNIFORM_SOFT_KD':'Distinctuniform sameperquerybudget can get more unique original facts; adequate288head.',
            'UNIFORM_POOLED':'Same uniformhead with arithmeticprobability TrackIDpooling; avoid artificiallyweak uniformcomparison.'},
        failure_constraints=['Old signed-risk knownteacher direction conflicts203/1811C2; never recreate that target loss.',
            'Old boundedtanhC4 saturation reproduced; new declared dampednumerics andpolynomialcap forALLmodels, no loweredadequacy.',
            'Old graphobserver selectivitynotproved; currentimagepretextdoesnotassume semanticedgehomophily.',
            'Old contextpilotworsenedKITTI; no training on its supposedbetterteacher.',
            'Do not drop originalquery modifiers, freeze whole-trackmembership, treat modelagreement as independenttruth, or split away halfmainlabels.'],
        neural_seeds=[0],GPU_roles={'C1':0,'C2':1,'C4':3,'UNIFORM':2},GPU2_fraction=.10,existing_GPU2_services_preserved=True,
        all_data_DEV=True,new_Qwen_or_frozen_backbone_calls=0,prepaid_replay_not_fresh_cold_cost=True,
        new_main_raw_records=27,new_dual_count_records=48,official_statistical_records=75,paired_effects=72,CountSelf_exact_aliases=27,
        all75sealed_before_current_study_GT=True,no_prepositive_module_ablation_matrix=True,
        success_screen='Unchanged strict exploratory screen: raw and BOTH CountMatched effects positive versus ALL8controls in EVERYone-video case. No posthoc promotion.',
        no_fullsix_threecontribution_or_RSI_success_claim=True,goal_status='ACTIVE')
    assert not guard
    atomic_json(PLAN,plan);atomic_json(PUBLIC/(STEM+'_registration.json'),plan)
    print(json.dumps(dict(status=plan['status'],pins=len(sources),parameter_counts=sizes,records=75)),flush=True)


def registration():
    plan=json.loads(PLAN.read_text())
    for n,sha in plan['source_sha256'].items():assert file_sha(ROOT/n)==sha,n
    for case in plan['cases']:
        for n,sha in case['pool_output_sha256'].items():assert file_sha(Path(case['pool_folder'])/n)==sha
        assert file_sha(case['auxiliary_path'])==case['auxiliary_sha256']
        assert file_sha(case['pixel_pairs_path'])==case['pixel_pairs_sha256']
    assert file_sha(PIXEL/'completion.json')==plan['shared_pixel_completion_sha256']
    return plan


def uniform_worker():
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    configure_before_cuda()
    import torch
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.active.residual_rsi_learning import uniform_draws
    from flex_uda.active.evidence_visual_learning import fit_control
    guard=generation_guard();plan=registration();torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.10)
    for case in plan['cases']:
        folder=OUT/case['pair'];(folder/'ordinary').mkdir(parents=True,exist_ok=True)
        path=folder/'uniform_draws.json'
        if path.exists():draws=json.loads(path.read_text())['draws']
        else:
            draws,receipt=uniform_draws(case,'cuda:0');atomic_json(path,dict(draws=draws,broker=receipt))
        model,fit=fit_control(case,draws,'UNIFORM_SOFT_KD',folder/'ordinary','cuda:0');del model
        print(json.dumps(dict(status='EVIDENCE_VISUAL_ADEQUATE_UNIFORM',pair=case['pair'],fit=fit)),flush=True)
    assert not guard
    atomic_json(OUT/'GPU2_complete.json',dict(status='ALL3_UNIFORM288_CONTROLS_COMPLETE',GT=0))


def worker(pair):
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution=configure_before_cuda()
    import numpy as np
    import torch
    from scipy.special import logsumexp
    from flex_uda.a100.mechanism import generation_guard,PAIRS
    from flex_uda.active.joint_observation_inputs import JointObservationVideo,use_verified_local_features
    from flex_uda.active.factor_rsi_learning import Inputs
    from flex_uda.active.full_appearance_learning_inputs import augment_dense
    from flex_uda.active.evidence_visual_learning import train_primary,fit_control,MeanHead,tensors
    from flex_uda.presence.predictions import seal_mask
    from scripts.a100_factor_rsi_primary_quality import subset_metadata,projection
    plan=registration();guard=generation_guard();case=next(c for c in plan['cases'] if c['pair']==pair)
    folder=OUT/pair;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'main_completion.json').exists():return
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.35)
    use_verified_local_features();started=time.perf_counter()
    solver,observer,learned=train_primary(case,folder/'primary','cuda:0',file_sha(PLAN))
    (folder/'ordinary').mkdir(exist_ok=True);heads={PRIMARY:solver,'R_DIRECT':MeanHead(observer)};fits={}
    for name in ('SOFT_KD','HARD_KD','IPW_KD','SSL_KD'):
        heads[name],fits[name]=fit_control(case,learned['draws'],name,folder/'ordinary','cuda:0')
    while not (folder/'ordinary/UNIFORM_SOFT_KD.pt').exists():
        if (OUT/'GPU2_failed.json').exists():raise RuntimeError((OUT/'GPU2_failed.json').read_text())
        time.sleep(2)
    uniform=json.loads((folder/'uniform_draws.json').read_text())['draws']
    heads['UNIFORM_SOFT_KD'],fits['UNIFORM_SOFT_KD']=fit_control(case,uniform,'UNIFORM_SOFT_KD',folder/'ordinary','cuda:0')
    video=JointObservationVideo(pair,case['evaluated_video']);inputs=Inputs(video);n,nq=len(video.obs),len(video.queries)
    scores={name:np.empty((nq,n),dtype='float32') for name in heads}
    with torch.no_grad():
        for qi in range(nq):
            for begin in range(0,n,512):
                rows=np.arange(begin,min(begin+512,n))
                records=[dict(local_query=qi,local_track=int(video.columns[r]),frame_id=int(video.obs[r,0]),track_id=int(video.obs[r,1])) for r in rows]
                a=inputs.arrays(records);base=np.column_stack((a['bags'].reshape(len(rows),-1),a['present'],a['current'],a['coarse'])).astype('float32')
                x=tensors(augment_dense(base,video,rows),'cuda:0')
                for name,head in heads.items():scores[name][qi,rows]=head(x).cpu().numpy()
    for source,name in (('SOFT_KD','SOFT_POOLED'),('UNIFORM_SOFT_KD','UNIFORM_POOLED')):
        z=scores[source].astype('float64');pooled=np.empty_like(z)
        for ti in range(len(video.tracks)):
            rows=np.flatnonzero(video.columns==ti)
            pooled[:,rows]=(logsumexp(-np.logaddexp(0,-z[:,rows]),axis=1)-logsumexp(-np.logaddexp(0,z[:,rows]),axis=1))[:,None]
        scores[name]=pooled.astype('float32')
    masks={name:values>0 for name,values in scores.items()};metadata=subset_metadata(case['target'],video.video);tasks=[]
    for name in METHODS:
        path=folder/'predictions'/name;path.mkdir(parents=True,exist_ok=True)
        assert np.isfinite(scores[name]).all() and np.array_equal(projection(scores[name],masks[name],video.obs),masks[name])
        np.save(path/'scores.npy',scores[name]);np.save(path/'mask.npy',masks[name])
        seal_mask(path,metadata,masks[name].flatten(),target=case['target'],model=PAIRS[pair][0],method=name,GT_open_attempts=len(guard),
            extra=dict(seed=0,same_raw9352_and_pixels=True,teacher_targets_unchanged=True,prepaid_replay_not_fresh_cost=True))
        tasks.append(dict(pair=pair,method=name,folder=str(path)))
    for rank,count in [(PRIMARY,n) for n in COMPARATORS]+[(n,PRIMARY) for n in COMPARATORS]:
        name=rank+'__AT_'+count+'_COUNT';path=folder/'predictions'/name;selected=projection(scores[rank],masks[count],video.obs)
        seal_mask(path,metadata,selected.flatten(),target=case['target'],model=PAIRS[pair][0],method=name,GT_open_attempts=len(guard),
            extra=dict(rank_method=rank,count_method=count,counts_equal_per_query_frame=True,counts_from_GT=False))
        tasks.append(dict(pair=pair,method=name,folder=str(path)))
    assert len(tasks)==25 and not guard
    atomic_json(folder/'main_completion.json',dict(status='SEALED_ALL25_EVIDENCE_VISUAL_OUTPUTS_NO_GT',tasks=tasks,ordinary_fits=fits,
        parameter_changes=learned['parameter_changes'],rounds=learned['rounds'],primary_proper_teacher_KL=learned['proper_known_teacher_KL'],
        logical_draws=len(learned['draws']),unique_teacher_actions=learned['broker']['unique_teacher_bundles'],execution=execution,
        teacher_targets_unchanged=True,GT_open_attempts=0,worker_wall_seconds=time.perf_counter()-started))


def drain(proc,path):
    with path.open('a',buffering=1) as stream:
        for line in proc.stdout:stream.write(line);print(line,end='',flush=True)


def main():
    plan=registration();sup=OUT/'supervisor';sup.mkdir(parents=True,exist_ok=True)
    pending=[(c['pair'],gpu,['--worker',c['pair']]) for c,gpu in zip(plan['cases'],(0,1,3)) if not (OUT/c['pair']/'main_completion.json').exists()]
    if not (OUT/'GPU2_complete.json').exists():pending.append(('UNIFORM',2,['--uniform-worker']))
    for label,gpu,_ in pending:
        owners=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True).strip()
        used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        total=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).strip())
        if gpu==2:assert total-used>=8192,(used,total)
        else:assert not owners and used<512,(gpu,used,owners)
        atomic_json(sup/(label+'_preflight.json'),dict(GPU=gpu,existing_residents=owners,used_MiB=used,all_preserved=True))
    jobs=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for label,gpu,args in pending:
            proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),*args],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
            future=pool.submit(drain,proc,sup/(label+'.log'));jobs.append((label,proc,future))
            atomic_json(sup/(label+'_launch.json'),dict(pid=proc.pid,GPU=gpu));print(json.dumps(dict(status='STARTED_NEW_EVIDENCE_VISUAL_PRIMARY',role=label,GPU=gpu,pid=proc.pid)),flush=True)
        failed=[]
        for label,proc,future in jobs:
            code=proc.wait();future.result()
            if code:failed.append(dict(role=label,exit_code=code))
    if failed:raise RuntimeError(failed)
    from flex_uda.presence.predictions import verify_predictions
    tasks=[]
    for case in plan['cases']:
        doc=json.loads((OUT/case['pair']/'main_completion.json').read_text());assert doc['GT_open_attempts']==0;tasks.extend(doc['tasks'])
    assert len(tasks)==75
    for task in tasks:verify_predictions(task['folder'])
    atomic_json(OUT/'all75_sealed_before_GT.json',dict(tasks=tasks,GT_open_attempts=0,registration_sha256=file_sha(PLAN)))
    print('ALL75_EVIDENCE_VISUAL_OUTPUTS_SEALED_ISOLATED_EVALUATION',flush=True)
    from scripts.a100_joint_teacher_noise_pilot import evaluate_one
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:completed=list(pool.map(evaluate_one,tasks))
    results={c['pair']:{} for c in plan['cases']}
    for task,result in completed:results[task['pair']][task['method']]=result
    comparisons=[]
    for pair,values in results.items():
        primary=values[PRIMARY]['HOTA_unrounded']
        for other in COMPARATORS:
            reference=values[other]['HOTA_unrounded']
            comparisons.append(dict(pair=pair,comparator=other,primary_HOTA=primary,ordinary_HOTA=reference,raw_delta_HOTA=primary-reference,
                primary_at_ordinary_count_delta_HOTA=values[PRIMARY+'__AT_'+other+'_COUNT']['HOTA_unrounded']-reference,
                ordinary_at_primary_count_delta_HOTA=primary-values[other+'__AT_'+PRIMARY+'_COUNT']['HOTA_unrounded']))
    eligible=all(min(r['raw_delta_HOTA'],r['primary_at_ordinary_count_delta_HOTA'],r['ordinary_at_primary_count_delta_HOTA'])>0 for r in comparisons)
    atomic_json(OUT/'quality_results.json',dict(status='COMPLETE_NEW_EVIDENCE_VISUAL_PRIMARY',results=results,comparisons=comparisons,
        candidate_eligible_for_larger_screen=eligible,all_data_DEV=True,no_fullsix_threecontribution_or_fresh_cost_success=True,goal_status='ACTIVE'))
    from scripts.a100_primary_statistics_common import collect,reproduce
    collect(OUT,PUBLIC,STEM,75);reproduce(PUBLIC,STEM,75,72)
    print(json.dumps(dict(status='COMPLETE_NEW_EVIDENCE_VISUAL_PRIMARY_ALL75_REPRODUCED',comparisons=comparisons)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker');p.add_argument('--uniform-worker',action='store_true');a=p.parse_args()
    try:
        if a.prepare:prepare()
        elif a.worker:worker(a.worker)
        elif a.uniform_worker:uniform_worker()
        else:main()
    except Exception as error:
        if a.worker:atomic_json(OUT/a.worker/'failed.json',dict(error=repr(error)))
        if a.uniform_worker:atomic_json(OUT/'GPU2_failed.json',dict(error=repr(error)))
        raise
