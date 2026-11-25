"""Stage 0b/0d: sealed-v7 answers -> v8 EM/counterfactuals, no GT access."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.active.v8 import PresenceEM,BeliefStateV8
from flex_uda.presence.runtime import runtime_guard,target_metadata
from flex_uda.presence.predictions import verify_predictions
from scripts.active_run import PAIRS

PUBLIC=ROOT/'results/presence_active/redesign_v8/amendments_A2_A3'


def rel_from(receipt):
    return {v:(x['sensitivity'],x['specificity']) for v,x in receipt['reliability'].items()}


def replay():
    attempts=runtime_guard();start=time.perf_counter()
    exact=json.loads((PUBLIC/'stage0_v7_reproduction.json').read_text())
    assert exact['status']=='PASS' and exact['max_absolute_error']<1e-9
    prior=json.loads((ROOT/'results/presence_active/redesign_v7/execution.json').read_text())
    base=Path(prior['active_root']);records=[];summaries=[];inputs=[]
    for pair,(model,target) in PAIRS.items():
        source=base/pair/'MINUS_SLOW/rho_0.3';verify_predictions(source)
        meta=target_metadata(target);em=PresenceEM()
        rel={v:(.8,.8) for v in ('CROP','CLIP','FRAME_BOX')}
        for vi,(video,bank) in enumerate(sorted(meta['videos'].items())):
            directory=source/'videos'/video
            receipt=json.loads((directory/'receipt.json').read_text())
            questions=json.loads((directory/'questions.json').read_text())
            with np.load(directory/'belief.npz') as b:
                tracks=b['tracks'].tolist();qnames=b['queries'].tolist()
                qid={q:i for i,q in enumerate(qnames)};tid={t:i for i,t in enumerate(tracks)}
                obs=np.asarray(bank['observations'])
                lengths=[int((obs[:,1]==t).sum()) for t in tracks]
                before=em.receipt()
                state=BeliefStateV8(np.zeros_like(b['fractions']),b['fractions'],b['presence_max'],
                    track_lengths=lengths,rho_hat=em.rho,pi_strata=em.pi.copy())
                # The old initial thirds are preserved, not recomputed from
                # the saved end-of-video f. No new GLANCE answers are invented.
                state.strata=b['prior_strata'].copy();state.recompute_presence(rel)
                old_absent=b['absent'].copy();old_probability=b['presence_probability'].copy()
                weak=b['weak_presence_logits'].copy()
                state.weak_presence_logits=weak;state.recompute_presence(rel)
                for r in range(1,5):
                    for answer in (a for a in questions if a['round']==r):
                        q,t=qid[answer['query_id']],tid[answer['track_id']]
                        view=answer['effective_view']
                        state.observe(q,t,answer['yes_probability']>=.5,view,rel[view],
                                      presence_question=answer['kind']=='PRESENCE')
                    rel=rel_from(receipt['rounds'][r-1]['ds'])
                    state.recompute_presence(rel)
                em.add_video(video,state);fit=em.fit(rel)
                output=PUBLIC/'gt_free_replay'/pair/video
                output.mkdir(parents=True,exist_ok=True)
                np.savez(output/'counterfactual.npz',queries=np.asarray(qnames),tracks=np.asarray(tracks),
                    prior_strata=state.strata,presence_probability=state.presence_probability,
                    absent=state.absent,low_tentative=state.low_tentative,output_absent=state.output_absence(),
                    confirmed_present=state.confirmed_present,weak_presence_logits=state.weak_presence_logits,
                    covered_strata=state.covered_strata,needed_strata=state.needed_strata,
                    old_absent=old_absent,old_presence_probability=old_probability,
                    presence_logLR_unclipped_sum=state.presence_logLR_unclipped_sum,
                    presence_logLR_clipped_sum=state.presence_logLR_clipped_sum)
                per_query=[]
                for q,name in enumerate(qnames):
                    per_query.append(dict(query=name,probability=float(state.presence_probability[q]),
                        absent=bool(state.absent[q]),low_tentative=bool(state.low_tentative[q]),
                        output_absent=bool(state.output_absence()[q]),old_absent=bool(old_absent[q]),
                        covered_strata=np.flatnonzero(state.covered_strata[q]).tolist(),
                        decision_source='EVIDENCE' if state.absent[q] else
                            'PRIOR_BUDGET_EXHAUSTED' if state.output_absence()[q] else 'NOT_REJECTED'))
                entry=dict(pair=pair,video=video,video_index=vi,target=target,model=model,
                    before=before,after=fit,queries=per_query,
                    counterfactual_path=str(output/'counterfactual.npz'),
                    counterfactual_sha256=file_sha(output/'counterfactual.npz'),
                    original_directory=str(directory),questions_sha256=file_sha(directory/'questions.json'),
                    belief_sha256=file_sha(directory/'belief.npz'),receipt_sha256=file_sha(directory/'receipt.json'),
                    approximate_counterfactual=True,new_GLANCE_answers_invented=False,
                    weak_score_protocol='stored v7 final weak ranks fixed throughout the video; saved per-round DS states',
                    all_answer_kinds_used_for_EM=True,GT_open_attempts=len(attempts))
                atomic_json(output/'em.json',entry);records.append(entry)
                print(json.dumps(dict(event='V8_OFFLINE_EM_VIDEO',pair=pair,video=video,
                    rho=em.rho,pi=em.pi.tolist(),iterations=fit['iterations'],converged=fit['converged'],
                    objective_trace_monotone=fit['objective_trace_monotone'])),flush=True)
                if not fit['converged']:
                    result=dict(status='STOP_EM_NONCONVERGENCE',records=records,GT_open_attempts=len(attempts))
                    atomic_json(PUBLIC/'stage0_offline_em.json',result)
                    return result
        summaries.append(dict(pair=pair,target=target,model=model,**em.receipt()))
        inputs.append(dict(pair=pair,predictions_seal=str(source/'predictions_seal.json'),
                           sha256=file_sha(source/'predictions_seal.json')))
    assert not attempts
    result=dict(status='PASS',summaries=summaries,records=records,inputs=inputs,
                GT_open_attempts=0,GPU_calls=0,teacher_calls=0,wall_seconds=time.perf_counter()-start,
                diagnostic_counterfactual_only=True,new_RMOT_predictions_generated=False)
    path=PUBLIC/'stage0_offline_em.json';atomic_json(path,result)
    atomic_json(PUBLIC/'stage0_gt_free_seal.json',dict(status='SEALED_BEFORE_GT',path=str(path),
        sha256=file_sha(path),inputs=inputs,GT_open_attempts=0,records=len(records)))
    return result


if __name__=='__main__':
    result=replay()
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','summaries')}),flush=True)
    if result['status']!='PASS':raise SystemExit(1)
