"""Teacher-only test of the within-object relation assumption, not HOTA.

Current-video object/query units are held out jointly across views. Their first
half frames supply the nuisance intercept; their last half is predicted. The
shared slope sees past videos and other current units only. No candidate
predictions, manual GT, new teacher calls or parameter choice are produced.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import expit
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import generation_guard
from flex_uda.active.relational_transport import fit
from scripts.a100_six_direction import PUBLIC
from scripts.a100_relation_transport import PREP

OUT=PREP.parent/'teacher_predictive_check'


def fold(row):
    unit=json.dumps([row['video'],row['query'],row['track']],separators=(',',':'))
    return int(hashlib.sha256(unit.encode()).hexdigest()[:8],16)%3


def score(rows,beta):
    grouped=defaultdict(list);result=[]
    for row in rows:grouped[tuple(row['group'])].append(row)
    for group,values in grouped.items():
        frames=sorted({r['frame'] for r in values})
        if len(frames)<2:continue
        cut=frames[max(1,len(frames)//2)];anchor=[r for r in values if r['frame']<cut];held=[r for r in values if r['frame']>=cut]
        w=np.asarray([r['weight'] for r in anchor],float);w/=w.sum()
        logit_mean=float(w@np.asarray([r['teacher_logit'] for r in anchor]))
        reference=float(w@np.asarray([r['x'] for r in anchor]))
        raw_mean=float(w@np.asarray([r['probability'] for r in anchor]))
        for row in held:
            result.append(dict(group=group,frame=row['frame'],target=row['probability'],weight=row['weight'],
                dx=row['x']-reference,WITHIN_RELATION=float(expit(logit_mean+beta*(row['x']-reference))),
                LOGIT_CONSTANT=float(expit(logit_mean)),RAW_CONSTANT=raw_mean))
    return result


def metrics(records):
    if not records:return dict(rows=0,object_view_groups=0)
    y=np.asarray([r['target'] for r in records]);w=np.asarray([r['weight'] for r in records])
    result=dict(rows=len(records),object_view_groups=len({tuple(r['group']) for r in records}),models={})
    for name in ('WITHIN_RELATION','LOGIT_CONSTANT','RAW_CONSTANT'):
        p=np.clip([r[name] for r in records],1e-7,1-1e-7)
        ce=-y*np.log(p)-(1-y)*np.log(1-p)
        result['models'][name]=dict(teacher_soft_CE=float(np.average(ce,weights=w)),
            teacher_probability_MSE=float(np.average((p-y)**2,weights=w)))
    return result


def main():
    start=time.perf_counter();guard=generation_guard()
    plan=json.loads((PUBLIC/'relation_teacher_predictive_registration.json').read_text())
    assert plan['source_sha256']==file_sha(Path(__file__))
    prepared=json.loads((PREP/'completion.json').read_text());result={};all_raw={}
    for pair,record in prepared['pairs'].items():
        past=[];heldout=[];receipts=[]
        for receipt in record['prefix_fits']:
            video=receipt['video'];rows=json.loads((PREP/pair/video/'teacher_rows.json').read_text())['rows']
            if rows:
                for k in range(3):
                    training=past+[r for r in rows if fold(r)!=k];testing=[r for r in rows if fold(r)==k]
                    beta,optimizer=fit(training);predictions=score(testing,beta);heldout.extend(predictions)
                    receipts.append(dict(video=video,heldout_fold=k,training_rows=len(training),heldout_input_rows=len(testing),beta=beta,
                        excited_training_groups=optimizer['excited_groups'],KKT_residual=optimizer['KKT_residual'],metrics=metrics(predictions)))
            past.extend(rows)
        moving=[r for r in heldout if abs(r['dx'])>.05]
        result[pair]=dict(all_heldout=metrics(heldout),state_change_over_0p05=metrics(moving),fold_receipts=receipts)
        all_raw[pair]=heldout
    assert not guard
    report=dict(status='TEACHER_ONLY_WITHIN_RELATION_PREDICTIVE_CHECK_COMPLETE',pairs=result,
        new_teacher_encoder_or_GPU_calls=0,neural_fits=0,manual_target_GT_open_attempts=0,
        slope_fits=sum(len(v['fold_receipts']) for v in result.values()),candidate_predictions_or_parameters_updated=False,
        first_half_intercept_last_half_targets=True,heldout_unit='video/query/providedTrackID jointly across views',
        no_future_video_training=True,current_video_transductive=True,
        interpretation='Teacher-response predictive fidelity only. The constants are mathematical nuisance-only controls, not ordinary RMOT distillation opponents. Not semantic correctness or HOTA evidence.',
        no_CI_claim_for_correlated_rows=True,all_data_development=True,wall_seconds=time.perf_counter()-start,goal_status='ACTIVE')
    OUT.mkdir(parents=True,exist_ok=True);atomic_json(OUT/'private_heldout_predictions.json',all_raw);atomic_json(OUT/'completion.json',report)
    compact=dict(report,pairs={p:{k:v for k,v in r.items() if k!='fold_receipts'} for p,r in result.items()})
    (PUBLIC/'relation_teacher_predictive_check.json').write_text(json.dumps(compact,separators=(',',':'))+'\n')
    print(json.dumps(compact),flush=True)


if __name__=='__main__':main()
