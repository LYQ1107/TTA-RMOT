"""New matched joint/binary teacher cohort; no adaptation or target labels.

Two observation policies share an input union. Every learner later receives
only its policy's data and its past/current video prefix. Collection cost is
research preparation, not the cold cost of deploying one method.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, metadata, generation_guard
from flex_uda.active.observation_learning_inputs import ObservationVideo
from scripts.a100_six_direction import OUT as SIX, PUBLIC
from scripts.a100_observation_design_collect import TEACHER_SHA, PREP

OUT = SIX/'joint_observation_cohort_20261123'
PLAN = PUBLIC/'joint_observation_collection_plan.json'
STATE = OUT/'supervisor/state.json'
POLICIES = ('DESIGN', 'UNIFORM')
MODES = ('JOINT_AB', 'JOINT_BA', 'SHARED_A', 'SHARED_B')
SUCCESS = 'PASS_MATCHED_JOINT_BINARY_TEACHER_COHORT_GOAL_ACTIVE'


def identity(case):
    fields = ('target','video','query_id','query_text','track_id','frame_A','frame_B','view')
    return hashlib.sha256(json.dumps({k:case[k] for k in fields},sort_keys=True,separators=(',',':')).encode()).hexdigest()


def teacher_pool(v):
    """Three real observations, preferentially disjoint from context anchors."""
    pools = []
    for ti in range(len(v.tracks)):
        rows = np.flatnonzero(v.columns == ti)
        anchors = set(map(int,v.sample_rows[ti,v.present[ti]]))
        available = np.asarray([int(r) for r in rows if int(r) not in anchors],dtype=int)
        selected = available[np.unique(np.linspace(0,len(available)-1,min(3,len(available))).round().astype(int))].tolist() if len(available) else []
        while len(selected) < min(3,len(rows)):
            choices = [int(r) for r in rows if int(r) not in selected]
            chosen = max(choices,key=lambda r:(min(abs(int(v.obs[r,0])-int(v.obs[s,0])) for s in selected) if selected else 0,-r))
            selected.append(chosen)
        pools.append(np.asarray(sorted(selected),dtype=int))
    return pools


def group_inputs(target, video):
    cases, own = {}, {}
    for pair, (_, domain) in PAIRS.items():
        if domain != target:
            continue
        v = ObservationVideo(pair,video)
        pools = teacher_pool(v)
        feature_cache = {}
        selected = {}
        for policy in POLICIES:
            requests = json.loads((PREP/pair/video/(policy+'_requests.json')).read_text())['requests']
            groups = defaultdict(list); seen = defaultdict(set)
            for request in requests:
                qi,ti = int(request['local_query']),int(request['local_track'])
                assert v.queries[qi]['expression_id'] == request['query_id'] and int(v.tracks[ti]) == request['track_id']
                pool = pools[ti]
                first = int(pool[np.abs(v.obs[pool,0]-request['frame_id']).argmin()])
                candidates = [int(r) for r in pool if int(r) != first]
                # Prefer a new unordered pair over repeating the opposite order.
                fresh = [r for r in candidates if (ti,*sorted((first,r)),request['view']) not in seen[qi]]
                if fresh:
                    candidates = fresh
                if not candidates:
                    second = first
                elif policy == 'UNIFORM':
                    start = int(np.flatnonzero(pool == first)[0])
                    cyclic = np.r_[pool[start+1:],pool[:start]]
                    second = next(int(r) for r in cyclic if int(r) in candidates)
                else:
                    if qi not in feature_cache:
                        rows = np.concatenate(pools)
                        values = v.dense_rows(np.full(len(rows),qi,dtype=int),rows).astype('float64')
                        descriptors = []
                        for block in (values[:,:768],values[:,768:1080]):
                            descriptors.append(block/np.maximum(np.linalg.norm(block,axis=-1,keepdims=True),1e-12))
                        descriptors.append(values[:,-6:].clip(0,1))
                        feature_cache[qi] = dict(zip(map(int,rows),np.c_[tuple(descriptors)]))
                    features = feature_cache[qi]
                    second = max(candidates,key=lambda r:(float(np.square(features[first]-features[r]).sum()),-r))
                signature = (ti,*sorted((first,second)),request['view'])
                if signature in seen[qi]:
                    continue
                seen[qi].add(signature)
                ra,rb = sorted((first,second),key=lambda r:int(v.obs[r,0]))
                case = dict(target=target,video=video,query_id=request['query_id'],query_text=request['query_text'],
                    track_id=request['track_id'],frame_A=int(v.obs[ra,0]),frame_B=int(v.obs[rb,0]),view=request['view'])
                uid = identity(case)
                cases[uid] = case
                groups[qi].append(dict(uid=uid,local_query=qi,local_track=ti,row_A=ra,row_B=rb,
                    first_in_context=bool(ra in v.sample_rows[ti,v.present[ti]]),
                    second_in_context=bool(rb in v.sample_rows[ti,v.present[ti]])))
            selected[policy] = groups
        assert set(selected['DESIGN']) == set(selected['UNIFORM']) == set(range(len(v.queries)))
        # Equal unique bundle quota per query before seeing either answer.
        quotas = {qi:min(len(selected[p][qi]) for p in POLICIES) for qi in selected['DESIGN']}
        assert min(quotas.values()) >= 1
        for policy in POLICIES:
            own[pair+'/'+policy] = [r for qi in sorted(quotas) for r in selected[policy][qi][:quotas[qi]]]
    used = {r['uid'] for rows in own.values() for r in rows}
    return {k:cases[k] for k in sorted(used)},own


def prepare():
    plan = json.loads(PLAN.read_text()); guard = generation_guard()
    # Use the already completed byte-identical cache, not a fresh feature
    # version or another copy of images/checkpoints. NAS mmap page waits had
    # stalled the first CPU preparation after 25 complete video groups.
    from flex_uda.active import observation_learning_inputs as inputs
    from scripts.a100_observation_local_cache import CACHE
    cache = json.loads((CACHE/'completion.json').read_text())
    assert cache['status'] == 'PASS_BYTE_IDENTICAL_HISTORICAL_OBSERVATION_LOCAL_CACHE' and cache['files'] == 592
    inputs.HISTORY = CACHE
    preserved = json.loads((OUT/'NAS_IO_handoff/preserved_inputs.json').read_text())['preserved_group_sha256']
    assert all(file_sha(OUT/name) == sha for name,sha in preserved.items())
    reused = 0
    counts = Counter(); groups = []; membership = Counter(); single = Counter()
    for target in sorted({t for _,t in PAIRS.values()}):
        for video in sorted(metadata(target)['videos']):
            path = OUT/'inputs'/target/video/'manifest.json'
            if path.is_file():
                saved = json.loads(path.read_text());assert saved['GT_open_attempts'] == 0
                cases,own = saved['cases'],saved['own'];reused += 1
            else:
                cases,own = group_inputs(target,video)
                atomic_json(path,dict(cases=cases,own=own,GT_open_attempts=0))
            assert all(identity(c) == uid for uid,c in cases.items())
            assert {r['uid'] for rows in own.values() for r in rows} == set(cases)
            expected = sum(1 if c['frame_A'] == c['frame_B'] else 4 for c in cases.values())
            groups.append(dict(target=target,video=video,cases=len(cases),model_examples=expected,manifest_sha256=file_sha(path)))
            for owner,rows in own.items():
                counts[owner] += len(rows)
                membership[owner] += sum(int(r['first_in_context'])+int(r['second_in_context']) for r in rows)
                single[owner] += sum(cases[r['uid']]['frame_A'] == cases[r['uid']]['frame_B'] for r in rows)
            if len(groups) % 5 == 0 or len(groups) == 74:
                progress = dict(status='PREPARING_MATCHED_JOINT_INPUTS_EXACT_LOCAL_IO',pid=os.getpid(),
                    completed=len(groups),total=74,reused_groups=reused,goal_status='ACTIVE')
                atomic_json(STATE,progress);print(json.dumps(progress),flush=True)
    assignments = [[] for _ in range(plan['worker_shards'])];loads = [0]*len(assignments)
    for group in sorted(groups,key=lambda g:(-g['model_examples'],g['target'],g['video'])):
        i = min(range(len(loads)),key=lambda i:(loads[i],i));assignments[i].append(group);loads[i] += group['model_examples']
    for i,rows in enumerate(assignments):
        atomic_json(OUT/f'shard{i}/manifest.json',dict(groups=sorted(rows,key=lambda g:(g['target'],g['video'])),expected_model_examples=loads[i]))
    assert len(groups) == 74 and not guard
    assert all(file_sha(OUT/name) == sha for name,sha in preserved.items())
    for pair in PAIRS:
        assert counts[pair+'/DESIGN'] == counts[pair+'/UNIFORM']
    audit = dict(status='PASS_UNLABELLED_MATCHED_JOINT_COHORT_INPUTS',target_videos=74,
        per_setting_policy_unique_bundles=dict(counts),per_setting_policy_context_member_endpoints=dict(membership),
        per_setting_policy_singleton_bundles=dict(single),unique_cases=sum(g['cases'] for g in groups),
        expected_unique_model_examples=sum(loads),shard_model_examples=loads,
        equal_unique_bundle_quota_per_query=True,singletons_use_one_real_binary_input_not_fake_two_events=True,
        preserved_pre_handoff_groups=len(preserved),reused_group_manifests=reused,
        exact_local_historical_feature_cache_SHA=cache['aggregate_sha256'],
        cache_preparation_not_method_deployment_cost=True,
        no_GT_teacher_answers_or_method_predictions_used=True,new_GPU_or_teacher_or_head_fits=0,
        context_relation_fix_shared_by_all_future_methods=True,goal_status='ACTIVE')
    atomic_json(OUT/'preparation.json',audit)
    atomic_json(PUBLIC/'joint_observation_collection_input_audit.json',audit)
    print(json.dumps(audit),flush=True)


def visual(client,case,frame):
    request = dict(target=case['target'],video_id=case['video'],query_id=case['query_id'],query_text=case['query_text'],
        track_id=case['track_id'],frame_id=frame,view=case['view'])
    _,images,_,info = client.prepare(request)
    key = case['target'],case['video']
    frames = info['clip']['frames'] if info['effective_view'] == 'CLIP' else [frame]
    full = client.views.frame(key,frame)[0]
    boxes = [client.views.observations[key,int(f),case['track_id']].bbox_xyxy for f in frames]
    moments = [dict(frame=int(f),center_x=round((b[0]+b[2])/(2*full.width),3),
        center_y=round((b[1]+b[3])/(2*full.height),3),width=round((b[2]-b[0])/full.width,3),
        height=round((b[3]-b[1])/full.height,3)) for f,b in zip(frames,boxes)]
    assert len(moments) == len(images)
    return images,dict(moments=moments,effective_view=info['effective_view'],visual_sha256=info['input']['visual'])


def prompt_context(case,first,second=None):
    text = 'Crops show the candidate target. A full-frame image marks the candidate with a red box. '
    for role,item in zip(('A','B'),(first,second)):
        if item is not None:
            text += f'Group {role} contains the next {len(item[0])} images, in chronological order. '
            text += 'Their frame and normalized candidate position/size records are '+json.dumps(item[1]['moments'],separators=(',',':'))+'. '
    return text+'Judge each group at its OWN LAST moment using its temporal context if needed. The complete description is: '+json.dumps(case['query_text'])+'. '


def worker(shard):
    guard = generation_guard(); plan = json.loads(PLAN.read_text())
    for name,sha in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == sha,name
    import torch
    from scripts.active_run import video_spec
    from flex_uda.a100.runtime import teacher
    root = OUT/f'shard{shard}'; start = time.perf_counter();torch.set_num_threads(4)
    torch.cuda.set_device(0)
    groups = json.loads((root/'manifest.json').read_text())['groups']
    md = {t:metadata(t) for t in {g['target'] for g in groups}}
    specs = [video_spec(g['target'],g['video'],md[g['target']]['videos'][g['video']]) for g in groups]
    atomic_json(root/'progress.json',dict(phase='LOADING_FROZEN_TEACHER',videos=0,total_videos=len(groups)))
    client = teacher(specs,root/'unused_cache.sqlite');assert client.backend.weight_sha == TEACHER_SHA
    load_seconds = time.perf_counter()-start
    atomic_json(root/'load_receipt.json',dict(teacher_load_and_setup_seconds=load_seconds,teacher_weight_sha256=TEACHER_SHA))
    tokens = [client.backend.processor.tokenizer.encode(s,add_special_tokens=False) for s in 'ABCD']
    assert all(len(t) == 1 for t in tokens)
    ids = [t[0] for t in tokens]; captured = []
    def hook(module,args,kwargs,output):
        z = output.logits[:,-1,:].detach().float();sub = z[:,ids]
        probabilities = sub.softmax(-1).cpu().tolist()
        masses = (sub.logsumexp(-1)-z.logsumexp(-1)).exp().cpu().tolist()
        for p,m,top in zip(probabilities,masses,z.argmax(-1).cpu().tolist()):
            captured.append(dict(choice_probability=p,choice_vocabulary_mass=m,top1_is_choice=top in ids))
    handle = client.backend.model.register_forward_hook(hook,with_kwargs=True)
    new_examples = reused_examples = 0; forward_wall = 0.; videos = []
    try:
        for vi,group in enumerate(groups):
            target,video = group['target'],group['video']
            manifest_path = OUT/'inputs'/target/video/'manifest.json'
            assert file_sha(manifest_path) == group['manifest_sha256']
            data = json.loads(manifest_path.read_text());cases = data['cases']
            folder = root/'videos'/target/video;checkpoint = folder/'answers_checkpoint.json'
            answers = json.loads(checkpoint.read_text())['answers'] if checkpoint.is_file() else {}
            assert set(answers) <= set(cases)
            reused_examples += sum(len(r['responses']) for r in answers.values())
            cache = {};begin = time.perf_counter()
            ordered = sorted(cases,key=lambda uid:(cases[uid]['view'],cases[uid]['track_id'],cases[uid]['frame_A'],cases[uid]['frame_B'],uid))
            for ci,uid in enumerate(ordered):
                if uid in answers:
                    continue
                case = cases[uid];parts = []
                for role in ('A','B'):
                    key = (case['track_id'],case['frame_'+role],case['view'])
                    if key not in cache:
                        if len(cache) >= 32:cache.pop(next(iter(cache)))
                        cache[key] = visual(client,case,case['frame_'+role])
                    parts.append(cache[key])
                a,b = parts;requests = []
                modes = MODES if case['frame_A'] != case['frame_B'] else ('SINGLE',)
                for mode in modes:
                    first,second = (b,a) if mode == 'JOINT_BA' else (a,b)
                    images = first[0]+second[0] if mode != 'SINGLE' else a[0]
                    prompt = prompt_context(case,first,second if mode != 'SINGLE' else None)
                    if mode.startswith('JOINT'):
                        prompt += 'Which statement is correct? A: only group A matches. B: only group B matches. C: both groups match. D: neither group matches. Answer with only one letter: A, B, C, or D.'
                    else:
                        role = mode[-1] if mode != 'SINGLE' else 'A'
                        prompt += f'Does group {role} match the description at its own last moment? Answer Yes or No.'
                    requests.append(dict(mode=mode,images=images,prompt=prompt))
                captured.clear()
                raw = client.backend.forward([r['images'] for r in requests],[r['prompt'] for r in requests])
                assert len(raw) == len(captured) == len(requests)
                forward_wall += raw[0]['batch_forward_seconds'];new_examples += len(raw)
                responses = {r['mode']:{**response,**choice,'prompt':r['prompt']} for r,response,choice in zip(requests,raw,captured)}
                if modes == ('SINGLE',):
                    p = responses['SINGLE']['yes_probability_fp32'];joint = [0.,0.,p,1-p];binary = [p,p]
                else:
                    ab = np.asarray(responses['JOINT_AB']['choice_probability'])
                    ba = np.asarray(responses['JOINT_BA']['choice_probability'])[[1,0,2,3]]
                    joint = ((ab+ba)/2).tolist()
                    binary = [responses['SHARED_'+s]['yes_probability_fp32'] for s in ('A','B')]
                assert np.isfinite(joint).all() and abs(sum(joint)-1) < 1e-5
                answers[uid] = dict(joint_target=joint,shared_binary_target=binary,responses=responses,visual_A=a[1],visual_B=b[1])
                if ci % 32 == 0 or ci+1 == len(ordered):
                    atomic_json(checkpoint,dict(answers=answers,GT_open_attempts=0))
                    atomic_json(root/'progress.json',dict(phase='COLLECTING_MATCHED_JOINT_BINARY',videos=vi,total_videos=len(groups),
                        current_target=target,current_video=video,current_cases=len(answers),current_total_cases=len(cases),
                        session_new_examples=new_examples,session_reused_examples=reused_examples))
            assert set(answers) == set(cases)
            for owner,rows in data['own'].items():
                pair,policy = owner.split('/')
                selected = [dict(r,case=cases[r['uid']],joint_target=answers[r['uid']]['joint_target'],
                    shared_binary_target=answers[r['uid']]['shared_binary_target']) for r in rows]
                atomic_json(OUT/'policy_answers'/pair/video/(policy+'_answers.json'),dict(cases=selected,
                    only_this_policy_requested_responses=True,teacher_weight_sha256=TEACHER_SHA,GT_open_attempts=0))
            videos.append(dict(target=target,video=video,cases=len(cases),examples=sum(len(r['responses']) for r in answers.values()),
                service_including_IO_and_checkpoints_seconds=time.perf_counter()-begin,
                joint_top1_choice=sum(r['top1_is_choice'] for a in answers.values() for m,r in a['responses'].items() if m.startswith('JOINT')),
                joint_examples=sum(m.startswith('JOINT') for a in answers.values() for m in a['responses'])))
            atomic_json(folder/'completion.json',dict(status='SEALED_MATCHED_JOINT_BINARY_VIDEO',**videos[-1],GT_open_attempts=0))
    finally:
        handle.remove()
    frozen = client.backend.seal_frozen();assert not guard
    atomic_json(root/'completion.json',dict(status='SEALED_MATCHED_JOINT_BINARY_SHARD',videos=videos,
        actual_new_model_examples=new_examples,reused_model_examples=reused_examples,cache_hits=0,
        teacher_load_and_setup_seconds=load_seconds,model_forward_wall_seconds=forward_wall,
        total_worker_wall_seconds=time.perf_counter()-start,physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),
        frozen=frozen,GT_open_attempts=0,not_single_method_cold_cost=True,goal_status='ACTIVE'))


def main(prepare_only=False):
    plan = json.loads(PLAN.read_text())
    for name,sha in plan['source_sha256'].items():assert file_sha(ROOT/name) == sha,name
    if not (OUT/'preparation.json').is_file():prepare()
    if prepare_only:return
    pending = list(range(plan['worker_shards']));active = {};finished = [];failure = None;start = time.perf_counter();last = 0.
    while pending or active:
        for gpu,(proc,log,shard) in list(active.items()):
            code = proc.poll()
            if code is None:continue
            log.close();del active[gpu]
            if code or not (OUT/f'shard{shard}/completion.json').is_file():
                failure = dict(shard=shard,exit_code=code,log_tail=(OUT/f'shard{shard}/worker.log').read_text()[-5000:])
            else:finished.append(shard)
        if pending and not failure:
            memory_text = subprocess.check_output(['nvidia-smi','-i','0,1,2,3','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            memory = {int(s.split(',')[0]):int(s.split(',')[1]) for s in memory_text.strip().splitlines()}
            for gpu in (0,1,2,3):
                if not pending or gpu in active or memory[gpu] >= 512:continue
                shard = pending.pop(0);folder = OUT/f'shard{shard}'
                if (folder/'completion.json').is_file():finished.append(shard);continue
                log = (folder/'worker.log').open('a',buffering=1)
                proc = subprocess.Popen([sys.executable,'-u',str(Path(__file__)), '--shard',str(shard)],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
                active[gpu] = proc,log,shard
        if time.perf_counter()-last > 45 or not active:
            progress = {}
            for gpu,(proc,_,shard) in active.items():
                path = OUT/f'shard{shard}/progress.json'
                progress[str(gpu)] = dict(pid=proc.pid,shard=shard,**(json.loads(path.read_text()) if path.is_file() else {'phase':'INITIALIZING'}))
            status = dict(status='RUNNING_MATCHED_JOINT_BINARY_TEACHER_COHORT',pid=os.getpid(),finished=finished,
                pending=pending,active=progress,failure=failure,wall_seconds=time.perf_counter()-start,goal_status='ACTIVE')
            atomic_json(STATE,status);print(json.dumps(status),flush=True);last = time.perf_counter()
        if failure and not active:raise RuntimeError(json.dumps(failure))
        if pending or active:time.sleep(5)
    shards = [json.loads((OUT/f'shard{i}/completion.json').read_text()) for i in sorted(finished)]
    audit = json.loads((OUT/'preparation.json').read_text())
    assert sum(sum(v['examples'] for v in s['videos']) for s in shards) == audit['expected_unique_model_examples']
    receipt = dict(status=SUCCESS,input_audit=audit,shards=shards,GT_open_attempts=0,
        raw_answers_not_to_Git=True,not_single_method_cold_cost=True,no_adaptation_or_HOTA_yet=True,goal_status='ACTIVE')
    atomic_json(OUT/'completion.json',receipt);atomic_json(PUBLIC/'joint_observation_collection_receipt.json',receipt)
    atomic_json(STATE,dict(status=SUCCESS,pid=os.getpid(),goal_status='ACTIVE'))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--shard',type=int);args=p.parse_args()
    try:
        worker(args.shard) if args.shard is not None else main(args.prepare_only)
    except Exception as error:
        destination = OUT/f'shard{args.shard}/failure.json' if args.shard is not None else STATE
        atomic_json(destination,dict(status='FAILED_MATCHED_JOINT_BINARY_COHORT_REQUIRES_ROOT_CAUSE',error=repr(error),goal_status='ACTIVE'))
        raise
