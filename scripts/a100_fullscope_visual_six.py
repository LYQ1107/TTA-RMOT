"""Full6setting study: all-frame support, all-ROI learning, unique QA.

Four persistent GPUworkers retain originalT0, share only exact-input teacher
cache, and never expose unrequested answers. All4588video outputs precede
this study's GT. This is a new unified primary, not a promoted old control.
"""
import argparse
import concurrent.futures
import fcntl
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
from scripts.a100_full_temporal_support import OUT as SUPPORT
from scripts.a100_evidence_visual_primary import registration as previous_registration

OUT=SIX/'fullscope_visual_six_20261126'
PLAN=OUT/'registration.json'
PUBLIC=ROOT/'docs/a100_six_direction_20261122'
STEM='fullscope_visual_six_20261126'
PRIMARY='FULL_SCOPE_VISUAL_RSI'
COMPARATORS=('SOFT_KD288','HARD_KD288','LURE_KD288','SSL_KD288','R_DIRECT',
    'SOFT_POOLED','UNIFORM_SOFT_KD288','UNIFORM_POOLED','FLAT_KD128','SAME_ARCH_KD128')
METHODS=(PRIMARY,*COMPARATORS)
EXTRA_SOURCES=('scripts/a100_fullscope_visual_six.py','flex_uda/active/fullscope_visual_runtime.py',
    'scripts/a100_fullscope_visual_contract.py','scripts/a100_full_temporal_support.py',
    'flex_uda/active/teacher_client.py','flex_uda/teacher/yestrack_backend.py',
    'flex_uda/teacher/teacher_cache.py','flex_uda/teacher/decompose_inputs.py',
    'flex_uda/teacher/visual_input_builder.py','configs/teacher/base_qwen2b.yaml','configs/teacher/prompts_v1.yaml')


def prepare():
    from flex_uda.a100.mechanism import generation_guard,PAIRS
    guard=generation_guard();assert not PLAN.exists()
    parent=previous_registration();support=json.loads((SUPPORT/'completion.json').read_text());assert support['video_units']==148
    contract=ROOT/'a100_runs/visual_classical_20261121/supervisor/fullscope_visual_contract_20261126.json'
    assert json.loads(contract.read_text())['status']=='PASS_FULL_SCOPE_INPUT_AND_SAMPLING_CONTRACT_NO_GT'
    cases=[]
    for unit in support['video_units_records']:
        pair,video=unit['pair'],unit['video'];aux=SIX/'fullsix_learning_common_20261125'/pair/'AUX/teacher_inputs.npz'
        queries=SUPPORT/pair/video/'queries.json';observations=SUPPORT/pair/video/'observations.npy'
        assert file_sha(queries)==unit['output_sha256']['queries.json'] and file_sha(observations)==unit['output_sha256']['observations.npy']
        cases.append(dict(pair=pair,target=unit['target'],evaluated_video=video,temporal_queries=str(queries),
            temporal_queries_sha256=file_sha(queries),observations=str(observations),observations_sha256=file_sha(observations),
            auxiliary_path=str(aux),auxiliary_sha256=file_sha(aux),logical_budget=unit['logical_budget'],
            pixel_pairs_path=str(OUT/'shared_pixels'/unit['target']/video/'pixel_pairs.npz')))
    assert len(cases)==148 and {c['pair'] for c in cases}==set(PAIRS)
    sources=dict(parent['source_sha256']);sources.update({n:file_sha(ROOT/n) for n in EXTRA_SOURCES})
    plan=dict(status='REGISTERED_NEW_FULL_SCOPE_VISUAL_SIX_SETTING_PRIMARY',primary=PRIMARY,comparators=COMPARATORS,cases=cases,
        source_sha256=sources,full_support_completion_sha256=file_sha(SUPPORT/'completion.json'),mathematical_input_contract_sha256=file_sha(contract),
        hypothesis='Allphysical-image content learning and temporallyunrestricted distinct externalverification can repair sparse-scope representation transfer and outperform adequate same-information teacherKD across all6settings.',
        prior_failures=['Previous75+21complete, scientificscreenfalse. C4primary69.627977losttosamearchitectureKD70.190925; no oldcontrolpromotion.',
          'SimagepairMSEgrew .04278->.14584/.01278->.21218/.01082->.04309 despite Rimproving. Scontentalignmentonlycovered11/44/19physicalROIs outof4371/1064/11168.',
          'Oldactualrequestscover1or2frames/track, <=3; C2canonical17positivequeries absentfromwholefiniteactionpool. Newactionspaceincludesallprovidedframes, noGTselection.',
          'Withreplacementwasted7/72/1logicalqueries; newsamplingwithoutreplacement ensures equal distinctquestionbudgets. Oldsignedpoint-targets remain rejected.'],
        candidate_contributions=['Conditionallyseparatedobjectcontent andquery/state neuralsolver: imagepretext is not fullsemanticmembership equality.',
          'Temporallyunrestricted activevision: allprovidedframes, softdiminishing-return trackcoverage, positiveexploration overremainingactions, distinctquestions.',
          'ActualR/Sfeedback: externalteacher mean/residual forecasts combine withCURRENT Sprobability through residual+(mean-S)^2; no stale studenterror critic. Rupdates external/image knowledge BEFORE same-round S update, including finalround. Contenttransfer coversALLphysicalROIs.'],
        not_yet_proved_original_or_independent_contributions=True,
        classical_thought='Observability/temporalaliasing, conditionalinvariance, diminishingreturnscoverage, properlosses, and positiveLURE surveyweights. Priorideas credited; no fitted-model,HOTA or semanticguarantee.',
        implementation='Lazyq*Nobs+obsindex; streamed9352Dfeatures. Normalizerfrom4096deterministic evenlyspaced unlabeledactions, sharedallmodels. Contentloss averages physicalimages directly, avoiding query-expanded tensors.',
        sampling='Withinquery fixedoriginalquota, fourfrontloadedstages; NOreplacement. .2uniformonremaining plus.8Rerror*Scurrentlast-layerJacobian/(1+questions_on_track). Selectionjournal saved BEFORE stageanswers; batch4teacher; labels do not affect otherquestions withinstage.',
        risk_control='PrimaryandLURE_KD288usecorrectpositiveno-replacementLUREweightsperquery/prefix. With.2uniformremaining,weights<=5. Legacy helpermodeIPW_KD is explicitly remapped; never use1/(N*pi)as if replacement. Originalteacher targets unchanged; no signedtargets. Primaryfinalbothweighted/unweightedproperKL<=.005.',
        critic_theory='E[(Y-pS)^2|X]=Var(Y|X)+(E[Y|X]-pS)^2 foridealcalibration. Rlearnedprequentialresidual is only a proxy, not certifiedvariance or truth. The criterion tracksCURRENTS, not pastSerrors alone; lastlayerJacobian approximates gradientsecondmoment. No optimalacquisition/HOTA guarantee.',
        teacher='OriginalQwen3VL2B9a13380... unchanged. FreshGPUforwards for exact-cachemisses, persistentT0onceperGPUworker. Sharedcache only exactimage/prompt/bbox/query/view/weight keys; eachpolicygetsONLYitsaskedanswers.',
        supervision='Onlytargetimages/taskqueries/originalteacher. Same100targetcollection3viewauxiliaryforALLmodels, explicitlyprepaidtransductive. No targetmanualGTadaptation, parameters, masks, or questionchoices.',
        training='Allseed0/Adam.0001/clip10/softsign8, aux500+500/500/500/4000. R500perstage BEFORE S; S/.1contentalignment fullphysicalROIs; R/.1VICRegprior onweakimagepairs. RmeanproperBCEandpre-answer RresidualMSE; initialauxresidualusesafterSbootstraperrorasdeclaredprior. Primaryfinalbothweighted/unweightedproperKL.005; ordinaryandSSLoriginal.005,min6000/max16000. No extra seeds or grid.',
        weak_pixels='Same12observed-adjacenttriplets/video, >=3same-IDcyclepoints and>=halfvalidpoints. Background/repeatedtexture maypass. Fewerthan2pairs gives explicitzeroimagepretext forR/SSL, NEVER skipvideo or inventsemantics.',
        all_models='PrimaryS/Rwidth128. Flatordinary288hasmoreparameters thanS+Rcombined; include directflat128andexactSarchitecture128, strongSSL288, hard, positiveLURE, finalR, soft/uniformTrackIDmean. Alloriginalraw9352/imageinformationavailable.',
        uniform='Strongdistinctobject-balancedbaseline:randomtrackorderwithoneuniformunseenframe/track before revisiting; sameperqueryquotas, sameviewroute. No unbiased-risk claimforuniformsampling.',
        per_video_adaptation='Independentvideounits, sameunlabelledcurrent-video normalization andsameauxforall. No futurevideoanswers enterlearners; exact-inputcachemayreuseanindependentlyrequestedanswer fromanotherunit, never grantsunrequesteddata.',
        GPUs=[0,1,2,3],GPU_memory_fraction=.35,original_GPU2_services_preserved=True,teacher_batch_size=4,
        planned_video_units=148,per_video_outputs=31,all_video_outputs=4588,
        official_setting_statistics=186,paired_effects=180,
        all4588sealed_and_four_teacher_final_weight_seals_before_current_study_GT=True,
        success_screen='Fixedprimary raw andBOTHCountMatched effects positive versusALL10controls inEACHcomplete cross-domain setting. Preserve previousfirst-video failures, no posthocbestcontrolpromotion.',
        evaluation='OfficialTrackEvalper-videoquerysequences, exactTPweightedAssAandTP/FP/FNsummed acrossallvideos; NEVERmeanofvideoHOTAs. PublicNumPyreproduction of186HOTAand180effects required.',
        cost_scope='Realnewteacher/imageIO/assembly/learning/predictiontimes measured. FrontPCD/DINOfeatures andauxiliaryQAareprepaid, teacher/cache shared acrosscollection; NOT a same-execution fullcolddeployment or iso-computeproof.',
        all_data_DEV=True,no_module_ablation_matrix=True,no_fullsix_threecontribution_or_freshcost_success_claim=True,goal_status='ACTIVE')
    assert not guard
    atomic_json(PLAN,plan)
    atomic_json(PUBLIC/(STEM+'_registration.json'),{k:v for k,v in plan.items() if k!='cases'})
    print(json.dumps(dict(status=plan['status'],cases=148,pins=len(sources),outputs=4588)),flush=True)


def registration():
    plan=json.loads(PLAN.read_text())
    for n,sha in plan['source_sha256'].items():assert file_sha(ROOT/n)==sha,n
    return plan


def pixels(case):
    import cv2
    import numpy as np
    from flex_uda.a100.mechanism import metadata
    from flex_uda.active.full_appearance_learning_inputs import FullAppearanceBank
    import scripts.a100_pixel_continuity_audit as old
    folder=Path(case['pixel_pairs_path']).parent;folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'prepare.lock').open('a')
    with lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt=folder/'completion.json'
        if receipt.exists():
            done=json.loads(receipt.read_text());assert file_sha(Path(case['pixel_pairs_path']))==done['pixel_pairs_sha256'];return done
        target,video=case['target'],case['evaluated_video']
        frames=np.unique(np.asarray(metadata(target)['videos'][video]['observations'])[:,0]);assert len(frames)>=3
        starts=np.unique(np.linspace(0,len(frames)-3,min(12,len(frames)-2)).astype(int))
        triplets=[frames[i:i+3].astype(int).tolist() for i in starts]
        old.OUT=folder/'raw'
        summary=old.worker(dict(target=target,video=video,triplets=triplets))
        record=json.loads((old.OUT/target/'completion.json').read_text());bank=FullAppearanceBank(target,video)
        left=[];right=[];weight=[]
        for triplet in record['triplets']:
            for unit in triplet['units']:
                valid,good=unit['cycle_valid'],unit['same_ID_supported_points']
                if good<3 or 2*good<valid:continue
                rows=[bank.by_observation.get((int(f),unit['track'])) for f in triplet['frames']]
                if any(r is None for r in rows) or not bank.valid[rows].all():continue
                features=bank.normalized(np.asarray(rows))
                for j in (1,2):left.append(features[0]);right.append(features[j]);weight.append(good/valid)
        np.savez(case['pixel_pairs_path'],left=np.asarray(left,dtype='float32').reshape(-1,768),
            right=np.asarray(right,dtype='float32').reshape(-1,768),weight=np.asarray(weight,dtype='float32'))
        done=dict(pairs=len(left),pretext_available=len(left)>=2,source_pixel_summary=summary,
            pixel_pairs_sha256=file_sha(Path(case['pixel_pairs_path'])),no_GT_teacher_or_backbone_calls=True)
        atomic_json(receipt,done);return done


def run_unit(case,dispatcher,gpu):
    import numpy as np
    import torch
    from scipy.special import logsumexp
    from flex_uda.a100.mechanism import generation_guard,PAIRS
    from flex_uda.active.fullscope_visual_runtime import Context,train,uniform_draws,teacher_table,fit,safe_content_loss
    import flex_uda.active.evidence_visual_learning as core
    from flex_uda.presence.predictions import seal_mask
    from scripts.a100_factor_rsi_primary_quality import subset_metadata,projection
    guard=generation_guard();pair,video=case['pair'],case['evaluated_video'];folder=OUT/'units'/pair/video
    folder.mkdir(parents=True,exist_ok=True);receipt=folder/'generation_complete.json'
    if receipt.exists():
        done=json.loads(receipt.read_text());assert done['registration_sha256']==file_sha(PLAN);return
    assert file_sha(Path(case['temporal_queries']))==case['temporal_queries_sha256']
    assert file_sha(Path(case['auxiliary_path']))==case['auxiliary_sha256']
    started=time.perf_counter();pixel=pixels(case);context=Context(case,folder,dispatcher,gpu)
    core.data=lambda unused:context.data()
    core.teacher_table=lambda features,query,draws,quotas,auxiliary,ay,mode='SOFT_KD':teacher_table(context,draws,mode)
    core.fit=lambda model,x,y,w,steps,device,optimizer=None,**kw:fit(context,model,x,y,w,steps,device,optimizer,**kw)
    core.content_loss=safe_content_loss
    s,r,learned=train(context,folder/'primary','cuda:0',file_sha(PLAN))
    heads={PRIMARY:s,'R_DIRECT':core.MeanHead(r)};fits={}
    original_make=core.make
    configs=(('SOFT_KD288','SOFT_KD','plain',288),('HARD_KD288','HARD_KD','plain',288),
        ('LURE_KD288','IPW_KD','plain',288),('SSL_KD288','SSL_KD','visual',288),
        ('FLAT_KD128','SOFT_KD','plain',128),('SAME_ARCH_KD128','SOFT_KD','visual',128))
    for name,mode,kind,width in configs:
        destination=folder/'ordinary'/name;destination.mkdir(parents=True,exist_ok=True)
        def factory(mean,std,device,kind=kind,width=width,outputs=1,_kind=kind,_width=width):
            return original_make(mean,std,device,kind=_kind,width=_width,outputs=outputs)
        core.make=factory
        try:head,rec=core.fit_control(case,learned['draws'],mode,destination,'cuda:0')
        finally:core.make=original_make
        rec.update(mode=name,width=width,architecture=kind,correct_positive_no_replacement_LURE=name=='LURE_KD288')
        saved=torch.load(destination/(mode+'.pt'),map_location='cpu',weights_only=False);saved['receipt']=rec;core.save(destination/(mode+'.pt'),saved)
        heads[name]=head;fits[name]=rec
    uniform=uniform_draws(context);destination=folder/'ordinary/UNIFORM_SOFT_KD288';destination.mkdir(parents=True,exist_ok=True)
    heads['UNIFORM_SOFT_KD288'],fits['UNIFORM_SOFT_KD288']=core.fit_control(case,uniform,'UNIFORM_SOFT_KD',destination,'cuda:0')
    n,nq=context.n,context.nq;scores={name:np.empty((nq,n),dtype='float32') for name in heads}
    with torch.no_grad():
        for q in range(nq):
            for begin in range(0,n,512):
                rows=np.arange(q*n+begin,q*n+min(begin+512,n));x=core.tensors(context.features[rows],'cuda:0')
                for name,head in heads.items():scores[name][q,begin:begin+len(rows)]=head(x).cpu().numpy()
    for source,name in (('SOFT_KD288','SOFT_POOLED'),('UNIFORM_SOFT_KD288','UNIFORM_POOLED')):
        z=scores[source].astype('float64');pooled=np.empty_like(z)
        for t in range(len(context.video.tracks)):
            rows=np.flatnonzero(context.video.columns==t)
            pooled[:,rows]=(logsumexp(-np.logaddexp(0,-z[:,rows]),axis=1)-logsumexp(-np.logaddexp(0,z[:,rows]),axis=1))[:,None]
        scores[name]=pooled.astype('float32')
    masks={name:values>0 for name,values in scores.items()};md=subset_metadata(case['target'],video);tasks=[]
    for name in METHODS:
        path=folder/'predictions'/name;path.mkdir(parents=True,exist_ok=True)
        assert np.isfinite(scores[name]).all()
        np.save(path/'scores.npy',scores[name]);np.save(path/'mask.npy',masks[name])
        seal_mask(path,md,masks[name].flatten(),target=case['target'],model=PAIRS[pair][0],method=name,GT_open_attempts=len(guard),
            extra=dict(seed=0,all_frames_available=True,all_physical_content_learning=True,teacher_targets_unchanged=True))
        tasks.append(dict(pair=pair,video=video,method=name,folder=str(path)))
    for rank,count in [(PRIMARY,c) for c in COMPARATORS]+[(c,PRIMARY) for c in COMPARATORS]:
        name=rank+'__AT_'+count+'_COUNT';path=folder/'predictions'/name
        mask=projection(scores[rank],masks[count],context.video.obs)
        seal_mask(path,md,mask.flatten(),target=case['target'],model=PAIRS[pair][0],method=name,GT_open_attempts=len(guard),
            extra=dict(rank_method=rank,count_method=count,counts_equal_per_query_frame=True,counts_from_GT=False))
        tasks.append(dict(pair=pair,video=video,method=name,folder=str(path)))
    assert len(tasks)==31 and not guard
    atomic_json(receipt,dict(status='SEALED_ALL31_FULL_SCOPE_VIDEO_OUTPUTS_NO_GT',tasks=tasks,GT_open_attempts=0,
        registration_sha256=file_sha(PLAN),ordinary_fits=fits,primary_teacher_KL=learned['proper_known_teacher_KL'],
        active_broker=context.primary.receipt(complete=True),uniform_broker=context.uniform.receipt(complete=True),
        physical_images=len(context.images),conceptual_actions=len(context.features),pixel_pairs=pixel['pairs'],
        content_loss_missing=not pixel['pretext_available'],GPU=gpu,wall_seconds=time.perf_counter()-started,
        teacher_final_weight_seal_deferred_to_worker_exit=True,prepaid_front_and_aux_not_fresh_cold_cost=True))
    print(json.dumps(dict(status='FULL_SCOPE_VIDEO_GENERATION_COMPLETE',pair=pair,video=video,GPU=gpu,questions=len(learned['draws']),outputs=31)),flush=True)


def worker(gpu):
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution=configure_before_cuda()
    import torch
    from flex_uda.active.fullscope_visual_runtime import TeacherDispatcher
    plan=registration();torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    dispatcher=TeacherDispatcher(ROOT,OUT/'teacher_cache/actual_inputs.sqlite')
    (OUT/'teacher_cache').mkdir(parents=True,exist_ok=True)
    ordered=sorted(plan['cases'],key=lambda c:(-c['logical_budget'],c['pair'],c['evaluated_video']))
    try:
        for case in ordered:
            folder=OUT/'units'/case['pair']/case['evaluated_video'];folder.mkdir(parents=True,exist_ok=True)
            if (folder/'generation_complete.json').exists():continue
            with (folder/'unit.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if (folder/'generation_complete.json').exists():continue
                atomic_json(folder/'assignment.json',dict(GPU=gpu,pid=os.getpid(),source_registration_sha256=file_sha(PLAN)))
                run_unit(case,dispatcher,gpu)
    finally:
        sealed=dispatcher.close()
        receipt=dict(teacher=sealed,execution=execution,GPU=gpu,peak_allocated_bytes=torch.cuda.max_memory_allocated())
        atomic_json(OUT/'supervisor'/f'GPU{gpu}_teacher_session_{os.getpid()}.json',receipt)
        final=OUT/'supervisor'/f'GPU{gpu}_teacher_final.json'
        if sealed is not None or not final.exists():atomic_json(final,receipt)


def drain(proc,path):
    with path.open('a',buffering=1) as stream:
        for line in proc.stdout:stream.write(line);print(line,end='',flush=True)


def evaluate_and_collect(plan,tasks):
    import csv
    import numpy as np
    from scripts.a100_joint_teacher_noise_pilot import evaluate_one
    from scripts.a100_semantic_pattern_teacher_statistics import value,write
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:completed=list(pool.map(evaluate_one,tasks))
    records={}
    for task,result in completed:
        key=task['pair']+'/'+task['method'];row=records.setdefault(key,dict(numerator=np.zeros(19),denominator=np.zeros(19),episodes=0,videos=0))
        detail=Path(task['folder'])/'results/pedestrian_detailed.csv';assert file_sha(detail)==result['detailed_sha256']
        with detail.open() as stream:episodes=[r for r in csv.DictReader(stream) if r['seq']!='COMBINED']
        for e in episodes:
            for i,threshold in enumerate(range(5,100,5)):
                tp=float(e['HOTA_TP___'+str(threshold)])
                row['numerator'][i]+=float(e['AssA___'+str(threshold)])*tp
                row['denominator'][i]+=tp+float(e['HOTA_FP___'+str(threshold)])+float(e['HOTA_FN___'+str(threshold)])
        row['episodes']+=len(episodes);row['videos']+=1
    assert len(records)==186
    for key,row in records.items():
        row['numerator']=row['numerator'].tolist();row['denominator']=row['denominator'].tolist();row['HOTA']=value(row)
        pair=key.split('/')[0];assert row['videos']==sum(c['pair']==pair for c in plan['cases'])
    comparisons=[]
    for pair in sorted({c['pair'] for c in plan['cases']}):
        def v(name):return records[pair+'/'+name]['HOTA']
        for other in COMPARATORS:
            comparisons.append(dict(pair=pair,comparator=other,primary_HOTA=v(PRIMARY),ordinary_HOTA=v(other),raw_delta_HOTA=v(PRIMARY)-v(other),
                primary_at_ordinary_count_delta_HOTA=v(PRIMARY+'__AT_'+other+'_COUNT')-v(other),
                ordinary_at_primary_count_delta_HOTA=v(PRIMARY)-v(other+'__AT_'+PRIMARY+'_COUNT')))
    eligible=all(min(r['raw_delta_HOTA'],r['primary_at_ordinary_count_delta_HOTA'],r['ordinary_at_primary_count_delta_HOTA'])>0 for r in comparisons)
    analysis=dict(status='COMPLETE_ALL6_FULL_SCOPE_PRIMARY',comparisons=comparisons,candidate_eligible_for_larger_screen=eligible,
        all_data_DEV=True,no_threecontribution_or_fullcold_success_claim=True,goal_status='ACTIVE')
    write(PUBLIC/(STEM+'_statistics.json'),dict(records=records,whole_cross_domain_settings=True))
    write(PUBLIC/(STEM+'_analysis.json'),analysis);atomic_json(OUT/'quality_results.json',analysis)
    # Independent parse of published sufficient statistics and contrasts.
    records=json.loads((PUBLIC/(STEM+'_statistics.json')).read_text())['records']
    public_analysis=json.loads((PUBLIC/(STEM+'_analysis.json')).read_text())
    comparisons=public_analysis['comparisons']
    expected={p+'/'+m for p in {c['pair'] for c in plan['cases']} for m in (
        *METHODS,*[PRIMARY+'__AT_'+c+'_COUNT' for c in COMPARATORS],*[c+'__AT_'+PRIMARY+'_COUNT' for c in COMPARATORS])}
    assert set(records)==expected and len(records)==186 and len(comparisons)==60
    assert all(abs(value(row)-row['HOTA'])<1e-8 for row in records.values())
    errors=[]
    for row in comparisons:
        pair,other=row['pair'],row['comparator']
        def v(name):return value(records[pair+'/'+name])
        errors.extend([abs(v(PRIMARY)-v(other)-row['raw_delta_HOTA']),abs(v(PRIMARY+'__AT_'+other+'_COUNT')-v(other)-row['primary_at_ordinary_count_delta_HOTA']),
            abs(v(PRIMARY)-v(other+'__AT_'+PRIMARY+'_COUNT')-row['ordinary_at_primary_count_delta_HOTA'])])
    assert len(errors)==180 and max(errors)<1e-8
    reproduced=all(min(r['raw_delta_HOTA'],r['primary_at_ordinary_count_delta_HOTA'],r['ordinary_at_primary_count_delta_HOTA'])>0 for r in comparisons)
    assert reproduced==public_analysis['candidate_eligible_for_larger_screen']
    write(PUBLIC/(STEM+'_reproduction.json'),dict(status='PASS_ALL186_HOTA_AND180_EFFECTS',records=186,effects=180,
        maximum_absolute_error=max(errors),raw_statistics_recompute='float64TPweightedAssAsum,nevermeanvideoHOTA',eligibility_reproduced=eligible,
        independent_published_JSON_and_NumPy_contrast_reproduction=True,all_data_DEV=True,goal_status='ACTIVE'))
    print(json.dumps(dict(status='COMPLETE_FULL_SCOPE_SIX_SETTING_PRIMARY',comparisons=comparisons)),flush=True)


def main():
    plan=registration();sup=OUT/'supervisor';sup.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for gpu in (0,1,2,3):
        owners=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True).strip()
        used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        total=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).strip())
        if gpu==2:assert total-used>=18*1024,(gpu,used,total)
        else:assert not owners and used<512,(gpu,owners,used)
        atomic_json(sup/f'GPU{gpu}_preflight.json',dict(GPU=gpu,residents=owners,used_MiB=used,all_existing_services_preserved=True))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for gpu in (0,1,2,3):
            proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker',str(gpu)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
            future=pool.submit(drain,proc,sup/f'GPU{gpu}.log');jobs.append((gpu,proc,future))
            atomic_json(sup/f'GPU{gpu}_launch.json',dict(GPU=gpu,pid=proc.pid))
            print(json.dumps(dict(status='STARTED_FULL_SCOPE_SIX_GPU_WORKER',GPU=gpu,pid=proc.pid)),flush=True)
        failed=[]
        for gpu,proc,future in jobs:
            code=proc.wait();future.result()
            if code:failed.append([gpu,code])
    assert not failed,failed
    from flex_uda.presence.predictions import verify_predictions
    tasks=[]
    for case in plan['cases']:
        folder=OUT/'units'/case['pair']/case['evaluated_video'];done=json.loads((folder/'generation_complete.json').read_text())
        assert done['GT_open_attempts']==0 and done['registration_sha256']==file_sha(PLAN);tasks.extend(done['tasks'])
    assert len(tasks)==4588
    for task in tasks:verify_predictions(task['folder'])
    for gpu in (0,1,2,3):assert (sup/f'GPU{gpu}_teacher_final.json').exists()
    atomic_json(OUT/'all4588_sealed_before_GT.json',dict(tasks=tasks,GT_open_attempts=0,teacher_final_weight_seals=4,registration_sha256=file_sha(PLAN)))
    print('ALL4588_FULL_SCOPE_OUTPUTS_AND_TEACHER_FINAL_SEALS_BEFORE_ISOLATED_GT',flush=True)
    evaluate_and_collect(plan,tasks)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=int);a=p.parse_args()
    if a.prepare:prepare()
    elif a.worker is not None:worker(a.worker)
    else:main()
