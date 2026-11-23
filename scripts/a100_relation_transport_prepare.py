"""Teacher-only causal-prefix preparation of the candidate relation field.

This fits a scalar relation to paid QA; no candidate predictions or GT metrics
are generated. Final acceptance still requires fair per-frame distillation,
same-unary ablations, CountMatched, all six settings and actual deployment cost.
"""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import PAIRS,metadata,history,generation_guard
from flex_uda.active.relational_transport import horizontal_direction,evidence_rows,fit,MU,HUBER_DELTA,MAX_LOGIT_SHIFT
from scripts.a100_six_direction import OUT as SIX,PUBLIC

OUT=SIX/'relational_transport/teacher_prepare'


def main():
    plan=json.loads((PUBLIC/'relation_transport_teacher_prepare_registration.json').read_text())
    assert plan['source_sha256']==file_sha(Path(__file__))
    assert plan['relation_module_sha256']==file_sha(ROOT/'flex_uda/active/relational_transport.py')
    guard=generation_guard();start=time.perf_counter();results={}
    for pair,(_,target) in PAIRS.items():
        m=metadata(target);past=[];receipts=[];actual_calls=eligible_queries=0
        for video,bank in sorted(m['videos'].items()):
            queries=[q for q in m['episodes'] if q['video_id']==video]
            directions=np.asarray([horizontal_direction(q['query_text']) for q in queries],int)
            eligible_queries+=int(np.count_nonzero(directions))
            records=json.loads((history(pair)/'videos'/video/'questions.json').read_text());actual_calls+=len(records)
            if directions.any():
                first_frame=min(int(f) for f,_ in bank['observations']);extension='.jpg' if target=='SportsMOT' else '.png'
                with Image.open(Path(bank['image_root'])/f'{first_frame:06d}{extension}') as image:width,height=image.size
            else:width,height=1,1
            rows,dedup=evidence_rows(records,queries,width,video)
            past.extend(rows);beta,receipt=fit(past)
            receipt.update(video=video,eligible_queries=int(np.count_nonzero(directions)),current_rows=len(rows),
                prefix_rows=len(past),actual_question_records=len(records),deduplication=dedup,
                no_future_video_data=True,GT_open_attempts=len(guard))
            destination=OUT/pair/video;destination.mkdir(parents=True,exist_ok=True)
            atomic_json(destination/'teacher_rows.json',dict(rows=rows,query_directions=directions.tolist(),width=width,height=height,
                no_target_manual_labels=True,rows_not_to_Git=True))
            atomic_json(destination/'prefix_fit.json',receipt);receipts.append(receipt)
        results[pair]=dict(source=PAIRS[pair][0],target=target,total_queries=len(m['episodes']),eligible_queries=eligible_queries,
            actual_historical_question_records=actual_calls,prefix_fits=receipts,last_beta=receipts[-1]['beta'],
            new_teacher_encoder_or_GPU_calls=0,neural_fits=0,manual_target_labels_used=False)
    assert not guard
    result=dict(status='TEACHER_ONLY_RELATION_PREFIX_FITS_PREPARED',pairs=results,
        hyperparameters=dict(mu=MU,Huber_delta=HUBER_DELTA,future_logit_shift_cap=MAX_LOGIT_SHIFT),
        mathematical_unit_tests='tests/test_relational_transport.py',predictions_generated=0,
        target_GT_open_attempts=0,new_teacher_encoder_or_GPU_calls=0,neural_fits=0,
        interpretation='A teacher-only candidate parameter fit and optimization receipt, not task improvement or validation of three innovations.',
        next_gate='Fair per-frame ordinary distillation must receive the same geometry and QA; raw/count outputs must be sealed before new GT evaluation. No posthoc best method selection.',
        all_data_development=True,independent_validation=False,wall_seconds=time.perf_counter()-start,goal_status='ACTIVE')
    atomic_json(OUT/'completion.json',result)
    compact=dict(result,pairs={p:{k:v for k,v in r.items() if k!='prefix_fits'} for p,r in results.items()})
    (PUBLIC/'relation_transport_teacher_prepare.json').write_text(json.dumps(compact,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps(compact,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
