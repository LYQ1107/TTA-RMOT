"""Actual one-anchor/one-query student, three-view frozen teacher and fast fit."""
import time
PROCESS_STARTED = time.perf_counter()
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.runtime import OUTPUT, metadata, source_head, load_student, teacher, video_spec


def main():
    from flex_uda.presence.runtime import episode_from_bank, cache_for, runtime_guard
    from flex_uda.model.flexhook_uda import prepared_inputs
    from flex_uda.presence.features import margins_and_features
    from flex_uda.presence.residual import head_from_checkpoint
    from flex_uda.presence.cardinality import expected_iou_utilities
    from flex_uda.active.v6 import selection_fractions
    from flex_uda.active.v8p import belief_state, PRESUPPOSED
    from flex_uda.active.v8 import fast_step_v8
    opened = runtime_guard()
    out = OUTPUT/'stage1'
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(0); np.random.seed(0)
    m = metadata(); video = sorted(m['videos'])[0]; full = m['videos'][video]
    # Query selected by sentence alone: first alphabetical clothing-specific query.
    query = sorted([q for q in m['episodes'] if q['video_id']==video and 'clothes' in q['query_text']],key=lambda q:q['expression_id'])[0]
    anchor = min(f for f,t in full['observations'])+9
    indices = [i for i,(f,t) in enumerate(full['observations']) if f==anchor]
    bank = dict(full, observations=[full['observations'][i] for i in indices], lines=[full['lines'][i] for i in indices])
    atomic_json(out/'slice.json',dict(video=video,frame=anchor,query_id=query['expression_id'],query_text=query['query_text'],
        candidate_tracks=[t for f,t in bank['observations']],selection='first sorted video; first clothing query; tenth frame; no GT',
        student_unique_image_count=1,student_history_padding='repeat anchor as prescribed for an episode start',
        teacher_clip='real past observations at t-9,t-6,t-3,t',upstream_candidates='fixed official tracking candidates'))
    student_device=torch.device('cuda:0'); teacher_device=torch.device('cuda:1')
    model,tok,transform,loaded=load_student(out/'model_log',student_device)
    ep=episode_from_bank(video,bank,'SportsMOT',transform); cache=cache_for(model,tok,ep,student_device)
    torch.cuda.synchronize(student_device); t=time.perf_counter()
    candidates=list(ep.current_candidates(anchor))
    with torch.no_grad():
        prepared=prepared_inputs(cache,candidates,query['query_text'],anchor)
        margin,pcd=margins_and_features(model,prepared,[(prepared['text'][:1],prepared['mask'][:1])])
    torch.cuda.synchronize(student_device); feature_seconds=time.perf_counter()-t
    values=margin[:,0].float().cpu().numpy()[None]
    assert np.isfinite(values).all() and bool(pcd.isfinite().all())
    np.savez(out/'student.npz',margin=values,pcd=pcd.float().cpu().numpy(),tracks=prepared['track_ids'])
    head=head_from_checkpoint(torch.load(source_head(),map_location='cpu',weights_only=False)).eval()
    with torch.no_grad():
        h=head(margin=torch.from_numpy(values.astype('float64')),padding_mask=torch.zeros(values.shape,dtype=torch.bool))
        util=expected_iou_utilities(h['candidate_probabilities'],torch.zeros(values.shape,dtype=torch.bool)).numpy()[:,None,:]
    raw=h['presence'].numpy().reshape(1,1); raw_k=util.argmax(-1)
    tracks=prepared['track_ids']
    fractions,prob,_=selection_fractions(bank['observations'],tracks,{anchor:0},values,util,raw,raw_k=raw_k)
    lengths=[sum(t==track for f,t in full['observations']) for track in tracks]
    state=belief_state(PRESUPPOSED,values,fractions,prob.max(1),track_lengths=lengths)
    before=state.posterior.copy(); params_before=dict(intercept=state.a.tolist(),slope=state.slope)
    spec=video_spec(video,full)
    spec['observations']=[r for r in spec['observations'] if anchor-9<=r['frame_id']<=anchor]
    t=time.perf_counter(); client=teacher([spec],out/'teacher.sqlite',str(teacher_device)); teacher_init=time.perf_counter()-t
    requests=[dict(target='SportsMOT',video_id=video,frame_id=anchor,track_id=track,query_id=query['expression_id'],query_text=query['query_text'],view=view)
              for track in tracks for view in ('CROP','CLIP','FRAME_BOX')]
    t=time.perf_counter();answers=client.ask_many(requests);teacher_request_seconds=time.perf_counter()-t
    assert all(not a['cache_hit'] for a in answers), 'Stage1 must start from fresh local cache'
    assert all(not a['clip_fallback'] and a['clip']['unique_frames']>=2 for a in answers if a['requested_view']=='CLIP')
    atomic_json(out/'teacher_answers.json',answers)
    for a in answers:
        state.observe(0,tracks.index(a['track_id']),a['yes_probability']>=.5,a['effective_view'],(.8,.8))
    t=time.perf_counter(); fast=fast_step_v8(state,raw);fast_seconds=time.perf_counter()-t
    t=time.perf_counter(); seal=client.close();teacher_seal_seconds=time.perf_counter()-t
    np.savez(out/'belief.npz',before=before,after=state.posterior,asked_mask=state.asked_mask,pseudo=state.pseudo_labels(),evidence=state.evidence)
    changed=float(np.max(np.abs(before-state.posterior)))
    receipt=dict(status='PASS' if changed>0 and fast['identity']['status']=='MAP_UPDATED' else 'EVIDENCE_UPDATED_FAST_FIT_REQUIRES_REVIEW',
        student=loaded,student_feature_seconds=feature_seconds,student_margin_shape=list(values.shape),student_pcd_shape=list(pcd.shape),
        student_peak_allocated_bytes=torch.cuda.max_memory_allocated(student_device),
        teacher=seal,teacher_initialization_seconds=teacher_init,teacher_request_wall_seconds=teacher_request_seconds,
        teacher_final_seal_seconds=teacher_seal_seconds,fast=fast,fast_fit_seconds=fast_seconds,
        params_before=params_before,params_after=dict(intercept=state.a.tolist(),slope=state.slope),max_belief_change=changed,
        logical_calls=len(answers),actual_teacher_calls=client.stats['new_calls'],cache_hits=client.stats['cache_hits'],
        views={v:sum(a['effective_view']==v for a in answers) for v in ('CROP','CLIP','FRAME_BOX')},
        GT_open_attempts=len(opened),physical_GPUs=[0,1],python=sys.version,
        process_wall_seconds=time.perf_counter()-PROCESS_STARTED,
        cold_definition='new process/model load; shared OS page cache not flushed')
    assert receipt['logical_calls']==seal['logical_calls']==seal['new_calls']
    assert np.isclose(sum(a['batch_forward_seconds']/a['batch_size'] for a in answers),seal['forward_seconds'])
    assert seal['cache_hits']==0 and seal['cache']['pending_leases']==0
    assert seal['cache']['complete_raw_responses']==len(answers)
    assert fast['identity']['optimizer_success'] and fast['identity']['objective_after']<fast['identity']['objective_before']
    assert receipt['params_before']!=receipt['params_after'] and not opened
    receipt['cost_receipt_audit']='PASS'
    atomic_json(out/'receipt.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
