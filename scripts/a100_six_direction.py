"""Six-direction research inputs and the common, label-free method matrix.

This is a new study, not a restart of the archived v8P queue. The first two
directions reuse their measured 10% stream; the other four get new physical
student/teacher streams. All target annotation access belongs to evaluation.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import PAIRS, metadata

PREVIOUS = ROOT/'a100_runs/visual_classical_20261121'
OUT = PREVIOUS/'six_direction_20261122'
PUBLIC = ROOT/'docs/a100_six_direction_20261122'
PROTOCOL = {'C1':'CHECKLIST', 'C2':'PRESUPPOSED', 'C3':'PRESUPPOSED',
            'C4':'PRESUPPOSED', 'C5':'PRESUPPOSED', 'C6':'CHECKLIST'}
SEEDS = (0, 17, 42)
METHODS = ('KD_FINAL', 'KD_MEAN', 'HARD_FINAL', 'BASE_PROTOCOL',
           'RAW_TEACHER_FILL', 'PROTOCOL_GRAPH', 'VISUAL_ONLY', 'LOCAL_JOINT', 'JOINT')


def source(pair):
    names = {'C4':'C4_minimal_single_KD_rho0p1_20261122',
             'C1':'C1_key_rho0p1_mean_pair_20261122'}
    return PREVIOUS/'cold_full_c4'/names[pair] if pair in names else OUT/'evidence'/pair


def questions(pair, video, bank, queries, pcd, prior, tracks, anchors):
    from flex_uda.active.visual_classical import visual_graph, acquisition_order
    target = PAIRS[pair][1]; obs = np.asarray(bank['observations'])
    lengths = np.asarray([(obs[:,1] == t).sum() for t in tracks]); eligible = np.flatnonzero(lengths >= 3)
    budget = int(np.floor(.1*len(queries)*len(eligible)))
    quota = np.full(len(queries), budget//len(queries)); quota[:budget%len(queries)] += 1
    requests = []
    for qi, q in enumerate(queries):
        _, distance = visual_graph(pcd[qi])
        seed = int(hashlib.sha256((video+str(qi)).encode()).hexdigest()[:8], 16)
        order = acquisition_order(prior[qi], distance, eligible, 'COVER', np.random.default_rng(seed))
        for ti in order[:quota[qi]]:
            requests.append(dict(target=target, video_id=video, query_id=q['expression_id'], query_text=q['query_text'],
                track_id=int(tracks[ti]), frame_id=int(anchors[ti]), view='CROP', local_query=qi, local_track=int(ti)))
    assert len(requests) == budget
    return requests


def collect(pair):
    """Fresh physical features and teacher QA, never evaluation or adaptation GT."""
    import torch
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.a100.runtime import load_student, source_head, teacher
    from flex_uda.presence.runtime import episode_from_bank
    from flex_tta.runtime.frozen_features import FrozenFeatureCache
    from flex_uda.active.v6_frames import FrameCacheV6
    from scripts.a100_visual_classical_cold_slice import SparsePCD
    from scripts.a100_visual_classical_cold_protocol_final import real_colours
    from scripts.active_run import video_spec
    assert pair in ('C2', 'C3', 'C5', 'C6')
    root = source(pair)
    if (root/'completion.json').is_file(): return
    root.mkdir(parents=True, exist_ok=True); guard = generation_guard(); start = time.perf_counter()
    torch.set_num_threads(4); torch.manual_seed(0); device = torch.device('cuda:0'); torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model_name, target = PAIRS[pair]; m = metadata(target); videos = sorted(m['videos'])
    ledger_path = root/'phase_ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(phases=[], attempts=[])
    ledger['attempts'].append(dict(started_unix=time.time(), physical_GPU=__import__('os').environ.get('CUDA_VISIBLE_DEVICES')))
    def phase(name, stamp, video=None):
        row = dict(phase=name, seconds=time.perf_counter()-stamp, video=video)
        ledger['phases'].append(row); atomic_json(ledger_path, ledger)
        print(json.dumps(dict(event='SIX_DIRECTION_EVIDENCE_PHASE', pair=pair, **row)), flush=True)
    t = time.perf_counter(); student, tok, transform, loaded = load_student(root/'student_log', device,
        model_name=model_name, target=target); phase('student_load_and_integrity', t)
    t = time.perf_counter(); client = teacher([video_spec(target,v,b) for v,b in m['videos'].items()], root/'raw.sqlite')
    phase('teacher_load_and_integrity', t); receipts = []
    for vi, video in enumerate(videos):
        folder = root/'source'/video
        if (folder/'receipt.json').exists():
            receipts.append(json.loads((folder/'receipt.json').read_text())); continue
        begin = time.perf_counter(); folder.mkdir(parents=True, exist_ok=True); bank = m['videos'][video]
        queries = [q for q in m['episodes'] if q['video_id'] == video]; obs = np.asarray(bank['observations'])
        sink = SparsePCD(obs, len(queries)); tracks = sink.tracks
        score_frames = np.unique(obs[np.concatenate(sink.samples),0]); ids = np.flatnonzero(np.isin(obs[:,0],score_frames))
        scoring_bank = dict(bank, observations=obs[ids].tolist(), lines=[bank['lines'][i] for i in ids])
        def receive(qa, local_ids, feature): sink(qa, ids[local_ids], feature)
        t = time.perf_counter(); episode = episode_from_bank(video, bank, target, transform)
        cache = FrozenFeatureCache(student, episode, tok, device, max_bytes=512<<20)
        fc = FrameCacheV6(student, cache, scoring_bank, queries, score_frames, source_head(model_name), folder,
            store_pcd=False, pcd_sink=receive)
        fc.score(model_step=0); pcd = sink.mean(); prior,_,_ = fc.decode(1.,0.); fc.close()
        assert fc.tracks == list(map(int, tracks)); phase('student_three_observations_original_temporal_context', t, video)
        t = time.perf_counter(); colours, anchors = real_colours(bank, obs, sink.samples,
            '.jpg' if target == 'SportsMOT' else '.png', compute_rgb=True)
        np.savez_compressed(folder/'sparse_inputs.npz', pcd=pcd, colours=colours, prior=prior, tracks=tracks, anchors=anchors)
        requests = questions(pair, video, bank, queries, pcd, prior, tracks, anchors)
        atomic_json(folder/'questions_before_answers.json', dict(requests=requests, GT_open_attempts=len(guard)))
        phase('real_RGB_and_COVER_acquisition', t, video)
        t = time.perf_counter(); responses = []
        for offset in range(0,len(requests),8):
            batch = requests[offset:offset+8]
            answers = client.ask_many([{k:v for k,v in row.items() if k not in ('local_query','local_track')} for row in batch])
            responses += [dict(answer, local_query=row['local_query'], local_track=row['local_track']) for row,answer in zip(batch,answers)]
        atomic_json(folder/'answers.json', dict(answers=responses, logical_calls=len(responses), GT_open_attempts=len(guard)))
        phase('real_teacher_service_including_IO', t, video)
        receipt = dict(video=video, index=vi, logical_teacher_calls=len(responses), queries=len(queries), tracks=len(tracks),
            source_frames=len(score_frames), full_frames=len(np.unique(obs[:,0])), original_four_observation_context_preserved=True,
            GT_open_attempts=len(guard), wall_seconds=time.perf_counter()-begin)
        atomic_json(folder/'receipt.json',receipt); receipts.append(receipt)
        del fc,cache,episode,sink; torch.cuda.empty_cache()
        print(json.dumps(dict(event='SIX_DIRECTION_EVIDENCE_VIDEO', pair=pair, completed=vi+1, total=len(videos))),flush=True)
    t = time.perf_counter(); teacher_receipt = client.close(); phase('teacher_final_integrity',t)
    assert not guard
    ledger['attempts'][-1]['actual_evidence_worker_seconds'] = time.perf_counter()-start; atomic_json(ledger_path,ledger)
    atomic_json(root/'completion.json',dict(status='SEALED_SIX_DIRECTION_EVIDENCE_NO_GT',pair=pair,source=loaded,
        teacher=teacher_receipt,receipts=receipts,logical_teacher_calls=sum(r['logical_teacher_calls'] for r in receipts),
        GT_open_attempts=0,initial_features_prepaid=False,initial_QA_prepaid=False,
        evidence_preparation_cost_not_single_method_deployment=True,all_data_development=True))


def run(pair, seed):
    import torch
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.active.confidence_teacher import fit_head
    from flex_uda.active.acquisition import exclusive_group
    from flex_uda.active.topology_selection import graph_matrices, apply_graph
    from flex_uda.active.protocol_calibration import global_pool, contract_graph, nonempty_baseline
    from flex_uda.active.appearance_runtime import empty_appearance_history, regularize_appearance, commit_appearance_history
    from scripts.a100_visual_classical_strict_prefix import plain, immediate_predictions
    from flex_uda.presence.predictions import seal_mask
    guard = generation_guard(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    root = OUT/'quality'/pair/f'seed{seed}'
    if (root/'completion.json').is_file(): return
    root.mkdir(parents=True,exist_ok=True); start = time.perf_counter(); model,target = PAIRS[pair]; m = metadata(target)
    total=np.zeros(768); square=np.zeros(768); rows=0; bx=[]; by=[]; history=empty_appearance_history(); receipts=[]
    state_path=root/'stream.pt'; next_video=0
    if state_path.exists():
        saved=torch.load(state_path,map_location='cpu',weights_only=False)
        total,square,rows,bx,by,history,receipts,next_video=[saved[k] for k in ('total','square','rows','bx','by','history','receipts','next_video')]
    for vi,(video,bank) in enumerate(sorted(m['videos'].items())):
        if vi<next_video: continue
        stamp=time.perf_counter(); queries=[q for q in m['episodes'] if q['video_id']==video]; texts=[q['query_text'] for q in queries]
        folder=source(pair)/'source'/video
        with np.load(folder/'sparse_inputs.npz') as a: pcd,colours,prior,tracks=[a[k] for k in ('pcd','colours','prior','tracks')]
        answers=np.full(prior.shape,np.nan)
        for a in json.loads((folder/'answers.json').read_text())['answers']:
            answers[int(a['local_query']),int(a['local_track'])]=a['yes_probability']
        current=pcd.reshape(-1,768).astype('float32'); total+=current.sum(0,dtype=float)
        square+=np.square(current.astype(float)).sum(0); rows+=len(current)
        mean=total/rows; std=np.maximum(np.sqrt(np.maximum(square/rows-mean**2,0)),1e-3)
        known=np.isfinite(answers); bx.append(pcd[known]); by.append(answers[known]); fits={}
        if pair in ('C1','C4'):
            previous=PREVIOUS/'confidence_teacher'/pair/f'seed{seed}'/'videos'/video
            with np.load(previous/'coarse_scores.npz') as a:
                assert np.array_equal(a['tracks'],tracks)
                scores={k:a[k].copy() for k in ('KD_FINAL','KD_MEAN','HARD_FINAL')}
            fits=json.loads((previous/'receipt.json').read_text())['fits']
        else:
            x=np.concatenate(bx); raw_y=np.concatenate(by); y=np.clip(raw_y,.02,.98)
            scores={}
            for label,soft in (('KD',True),('HARD',False)):
                heads,fit,checkpoint=fit_head(x,np.zeros(len(y)),y,current,np.zeros(prior.size),mean,std,
                    sample_weight=1.+3.*np.abs(2.*raw_y-1.),soft=soft,seed=seed)
                fits[label]=fit; torch.save(checkpoint,root/(label+'_latest_head.pt'))
                scores[label+'_FINAL']=heads['CONVERGED'].reshape(prior.shape)
                if soft:scores['KD_MEAN']=.5*(heads['400']+heads['CONVERGED']).reshape(prior.shape)
        unary=scores['KD_FINAL']; lengths=np.asarray([(np.asarray(bank['observations'])[:,1]==t).sum() for t in tracks])
        t=time.perf_counter(); geometry,local,glob,valid,supported=graph_matrices(pcd,colours,texts)
        pooled=global_pool(unary,answers,glob,valid,supported)
        selected=contract_graph(unary,pooled,PROTOCOL[pair]); local_p=apply_graph(unary,answers,geometry,local,supported)
        scores['BASE_PROTOCOL']=nonempty_baseline(unary,PROTOCOL[pair])
        scores['RAW_TEACHER_FILL']=np.where(known,answers,unary)
        scores['PROTOCOL_GRAPH']=selected
        scores['VISUAL_ONLY'],inputs,cal=regularize_appearance(unary,colours,texts,lengths,answers,tracks,video,history)
        scores['LOCAL_JOINT'],_,_=regularize_appearance(local_p,colours,texts,lengths,answers,tracks,video,history)
        scores['JOINT'],_,_=regularize_appearance(selected,colours,texts,lengths,answers,tracks,video,history)
        operation_seconds=time.perf_counter()-t
        if pair in ('C1','C4'):
            with np.load(PREVIOUS/'protocol_calibration'/pair/f'seed{seed}'/'videos'/video/'coarse_scores.npz') as a:
                assert np.array_equal(scores['JOINT'],a['PROTOCOL_JOINT']),('C1/C4 exact recipe replay',pair,seed,video)
        obs=np.asarray(bank['observations']); lookup={int(t):i for i,t in enumerate(tracks)}; columns=np.asarray([lookup[int(t)] for _,t in obs])
        for name in METHODS:
            mask=plain(scores[name][:,columns],exclusive_group(texts,target)); destination=root/name/'videos'/video
            destination.mkdir(parents=True,exist_ok=True); np.save(destination/'mask.npy',mask)
            immediate_predictions(root/name,type('Context',(),dict(bank=bank,queries=queries))(),mask)
        local_root=root/'videos'/video; local_root.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(local_root/'coarse_scores.npz',tracks=tracks,**scores)
        receipt=dict(video=video,index=vi,fits=fits,current_teacher_examples=int(known.sum()),teacher_examples=sum(map(len,by)),
            reused_neural_fit=pair in ('C1','C4'),protocol=PROTOCOL[pair],GT_open_attempts=len(guard),calibration=cal,
            classical_seconds=operation_seconds,wall_seconds=time.perf_counter()-stamp)
        atomic_json(local_root/'receipt.json',receipt); receipts.append(receipt); commit_appearance_history(history,inputs,answers)
        saved=dict(total=total,square=square,rows=rows,bx=bx,by=by,history=history,receipts=receipts,next_video=vi+1)
        temporary=state_path.with_suffix('.tmp'); torch.save(saved,temporary); temporary.replace(state_path)
        print(json.dumps(dict(event='SIX_DIRECTION_QUALITY_VIDEO',pair=pair,seed=seed,completed=vi+1,total=len(m['videos']))),flush=True)
    records=[]
    for name in METHODS:
        mask=np.zeros(m['candidate_query_rows'],bool)
        for video in sorted(m['videos']):
            bits=np.load(root/name/'videos'/video/'mask.npy')
            for qi,q in enumerate(q for q in m['episodes'] if q['video_id']==video):mask[q['start']:q['stop']]=bits[qi]
        seal=seal_mask(root/name,m,mask,target=target,model=model,method='SIX_DIRECTION_'+name,GT_open_attempts=len(guard),
            extra=dict(seed=seed,teacher_fraction=.1,task_protocol=PROTOCOL[pair],source_logit_zero=True,
                same_shared_converged_ordinary_head=True,current_video_transductive=True,no_future_answers=True,
                quality_replay_not_single_method_cold_cost=True))
        records.append(dict(method=name,rows=seal['prediction_rows']))
    assert not guard
    atomic_json(root/'completion.json',dict(status='SEALED_SIX_DIRECTION_METHODS',pair=pair,seed=seed,methods=records,
        receipts=receipts,GT_open_attempts=0,logical_teacher_calls=sum(map(len,by)),new_teacher_calls=0,
        new_head_fits=0 if pair in ('C1','C4') else 2*len(receipts),source_pool=str(source(pair)),
        wall_seconds=time.perf_counter()-start,goal_status='ACTIVE'))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=tuple(PAIRS),required=True)
    p.add_argument('--collect',action='store_true');p.add_argument('--seed',type=int,default=0);a=p.parse_args()
    if a.collect:
        # A failed source registration may be repaired independently while
        # healthy workers finish. Serialise only the same evidence stream;
        # a later group-controller resume must not duplicate its teacher QA.
        import fcntl
        folder=source(a.pair);folder.mkdir(parents=True,exist_ok=True)
        with (folder/'collection.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);collect(a.pair)
    else:
        import fcntl
        folder=OUT/'quality'/a.pair/f'seed{a.seed}';folder.mkdir(parents=True,exist_ok=True)
        with (folder/'condition.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);run(a.pair,a.seed)
