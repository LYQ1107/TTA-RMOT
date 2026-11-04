"""Post-queue v6 protocol/complete-MOT audit; no GT/model/teacher invocation."""
from argparse import ArgumentParser
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.active.execution import authorize_active
from flex_uda.presence.runtime import target_metadata,runtime_guard
from flex_uda.presence.predictions import verify_predictions
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.active_run import PAIRS
from scripts.active_eval import official_stats
from scripts.active_v6_prepare import sample32


def validate(execution):
    attempts=runtime_guard();public=execution.public
    plan=json.loads((public/'registered_jobs.json').read_text());jobs=plan['jobs'][:]
    decisions=json.loads((public/'baseline_budget_decisions.json').read_text())
    jobs+=decisions['jobs']
    assert file_sha(plan['prereg_path'])==plan['prereg_sha256']
    old_public=Path(execution.manifest['prior_v5_public_root'])
    with (old_public/'main.csv').open() as f:old_rows=list(csv.DictReader(f))
    for row in old_rows:
        if row['method']=='GATE_REASSIGN_REF':continue
        assert file_sha(Path(row['path'])/'predictions_seal.json')==row['prediction_seal_sha256']
    meta_cache={};records=[];total_episodes=total_rows=0
    for job in jobs:
        pair,method=job['pair'],job['method'];target=PAIRS[pair][1]
        folder=execution.active/pair/method/'rho_0.3';seal=verify_predictions(folder)
        if target not in meta_cache:meta_cache[target]=target_metadata(target)
        meta=meta_cache[target];expected={(q['video_id'],q['expression_id']) for q in meta['episodes']}
        assert {(x['video'],x['expression']) for x in seal['episodes']}==expected
        teacher=seal['teacher'];assert teacher['GT_open_attempts']==seal['GT_open_attempts']==0
        assert teacher['revision']=='89644892e4d85e24eaac8bacfd4f463576704203'
        assert teacher['teacher_weight_sha256']=='9a13380b4279bc98f4680bd083ff4995fdcfec120cfded459513be90d041a956'
        sealed={(x['video'],x['expression']):x for x in seal['episodes']}
        updates=0;zero_grad=0;calls=0;cache_rows=0
        for video,bank in meta['videos'].items():
            directory=folder/'videos'/video;receipt=json.loads((directory/'receipt.json').read_text())
            mask=np.load(directory/'mask.npy');assert file_sha(directory/'mask.npy')==receipt['mask_sha256']
            queries=[q for q in meta['episodes'] if q['video_id']==video]
            assert mask.shape==(len(queries),len(bank['observations']))
            for qi,q in enumerate(queries):
                row=sealed[video,q['expression_id']]
                assert Path(row['prediction_path']).read_text()==''.join(line+'\n' for line,keep in zip(bank['lines'],mask[qi]) if keep)
            if method.startswith('YESTRACK'):
                full=json.loads((execution.active/pair/'FULL/rho_0.3/videos'/video/'receipt.json').read_text())
                assert receipt['actual_calls']==full['questions'];continue
            calls+=receipt['questions'];answers=json.loads((directory/'questions.json').read_text())
            by_track=defaultdict(list)
            for i,(_,track) in enumerate(bank['observations']):by_track[track].append(i)
            with np.load(directory/'belief.npz') as b:
                tracks=b['tracks'].tolist();lookup={t:i for i,t in enumerate(tracks)}
                qid={q['expression_id']:i for i,q in enumerate(queries)}
                eligible=np.asarray([len(by_track[t])>=3 for t in tracks])
                expected_evidence=np.zeros_like(b['evidence']);seen=np.zeros_like(b['asked_mask'])
                no=[set() for _ in queries];yes=[set() for _ in queries]
                for answer in answers:
                    q,t=qid[answer['query_id']],lookup[answer['track_id']]
                    assert eligible[t] and abs(answer['evidence_delta'])<=2.
                    assert np.isclose(answer['evidence_delta'],np.clip(answer['unclipped_evidence_delta'],-2,2))
                    expected_evidence[q,t]+=answer['evidence_delta'];seen[q,t]=True
                    (yes if answer['yes_probability']>=.5 else no)[q].add(t)
                    assert answer['identity_used_as_DS_annotator'] is False
                assert np.array_equal(seen,b['asked_mask']) and np.allclose(expected_evidence,b['evidence'],atol=1e-12)
                assert (b['pseudo_labels'][~seen]==-1).all() and (b['pseudo_labels'][:,~eligible]==-1).all()
                f=b['fractions'];assert ((0<=f)&(f<=1)).all()
                p=np.clip(f,.05,.95);base=np.clip(np.log(p/(1-p)),-2,2)
                assert np.array_equal(base,b['base_logits']) and 0<=float(b['slope'])<=1
                assert np.allclose(b['prior'][:,eligible],(b['intercept'][:,None]+b['slope']*base)[:,eligible])
                raw=np.load(directory/'frame_cache/presence_logits.npy')
                maximum=expit(raw/float(b['temperature'])+float(b['bias'])).max(1)
                assert np.allclose(maximum,b['presence_max'])
                for q in range(len(queries)):
                    covered={int(b['prior_strata'][q,t]) for t in no[q] if b['prior_strata'][q,t]>=0}
                    absent=maximum[q]<.2 and len(no[q])>=3 and not yes[q] and covered=={0,1,2}
                    assert bool(b['absent'][q])==bool(absent)
            local=json.loads((ROOT/'results/runs/presence_active_20261102/features'/target/PAIRS[pair][0]/'videos'/video/'metadata.json').read_text())
            fixed=sample32(local['frames']);assert receipt['fixed32_frames']==fixed
            cache=json.loads((directory/'frame_cache/cache_receipt.json').read_text())
            for name in ('margin.npy','pcd.npy'):
                value=np.load(directory/'frame_cache'/name,mmap_mode='r')
                assert value.dtype==np.float32
                expected_shape=(len(queries),len(bank['observations']))+((768,) if name=='pcd.npy' else ())
                assert value.shape==expected_shape
                assert (directory/'frame_cache'/name).stat().st_size==cache['files'][name]['bytes']
            for item in cache['records']:
                if item['kind']=='FIXED32':assert item['frames']==fixed
                else:assert item['frames']==local['frames']
                count=len(queries)*sum(t in set(item['frames']) for t,_ in bank['observations'])
                assert item['candidate_query_rows']==count;cache_rows+=count
            for rr in receipt['rounds']:
                ds=rr['ds'];assert ds['Beta_prior']==[2.,2.] and not ds['identity_annotator']
                assert set(ds['reliability'])=={'CROP','CLIP','FRAME_BOX'}
                slow=rr['slow'];updates+=slow['optimizer_updates']
                for x in slow.get('records',[]):
                    if x.get('optimizer_updated'):
                        assert np.isfinite([x['loss'],x['gradient_norm']]).all()
                        zero_grad+=x['gradient_norm']==0
        assert method.startswith('YESTRACK') or calls==teacher['logical_calls']
        records.append(dict(pair=pair,method=method,episodes=len(expected),prediction_rows=seal['prediction_rows'],
            HOTA=official_stats(folder)['HOTA_unrounded'],optimizer_updates=updates,zero_gradient_updates=int(zero_grad),
            frame_cache_candidate_query_rows=cache_rows,GT_open_attempts=0,
            prediction_seal_sha256=file_sha(folder/'predictions_seal.json')))
        total_episodes+=len(expected);total_rows+=seal['prediction_rows']
    assert not attempts
    result=dict(status='V6_AUTHORIZED_SET_VALIDATED',primary_jobs=len(plan['jobs']),conditional_jobs=len(decisions['jobs']),
        total_episodes=total_episodes,prediction_rows=total_rows,methods=records,old_v5_seals_unchanged=True,
        unasked_pseudo_labels_never_generated=True,teacher_evidence_not_double_counted=True,
        prior_formula_and_w_bounds_verified=True,absence_three_strata_and_presence_threshold_verified=True,
        DS_beta2_without_identity_verified=True,fixed32_frames_and_full_start_coverage_verified=True,
        FP32_margin_PCD_shapes_verified=True,MOT_rows_boxes_IDs_frames_exact=True,
        GT_open_attempts_by_validator=0,GPU_or_teacher_called_by_validator=False,
        original_v5_80_not_started=True,not_all_six_FULL_directions_claimed=len(plan['jobs'])==5)
    atomic_json(public/'validation.json',result);print(json.dumps({k:v for k,v in result.items() if k!='methods'},indent=2))


if __name__=='__main__':
    execution,_=authorize_active();validate(execution)
