"""Closed v8P protocol audit: no model, teacher or GT reads."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.active.execution import authorize_active
from flex_uda.presence.runtime import runtime_guard,target_metadata
from flex_uda.presence.predictions import verify_predictions
from flex_uda.active.v8 import BeliefStateV8
from scripts.active_run import PAIRS


def validate(execution,plan_path=None):
    attempts=runtime_guard();plan=json.loads((Path(plan_path) if plan_path else execution.public/'registered_jobs.json').read_text())
    records=[]
    for job in plan['jobs']:
        path=execution.active/job['pair']/job['method']/('rho_'+str(job['rho']));seal=verify_predictions(path)
        target=PAIRS[job['pair']][1];meta=target_metadata(target)
        assert seal['version']=='v8P' and seal['GT_open_attempts']==0
        sealed={(x['video'],x['expression']):x for x in seal['episodes']}
        assert {(x['video'],x['expression']) for x in seal['episodes']}=={(q['video_id'],q['expression_id']) for q in meta['episodes']}
        if job['method'] in ('YESTRACK_BUDGET_PRIOR','UNIFORM_K'):
            source=execution.active/job['pair']/'MINUS_SLOW'/('rho_'+str(job['rho']))
            assert seal['budget_source_seal_sha256']==file_sha(source/'predictions_seal.json')
            for video in sorted(meta['videos']):
                d=path/'videos'/video
                r=json.loads((d/'receipt.json').read_text())
                original=json.loads((source/'videos'/video/'receipt.json').read_text())
                assert r['actual_calls']==original['questions']
                assert file_sha(d/'mask.npy')==r['mask_sha256']
            records.append(dict(job,episodes=seal['episode_count'],prediction_rows=seal['prediction_rows'],
                prediction_seal_sha256=file_sha(path/'predictions_seal.json'),
                budget_match_exact_per_video=True,GT_open_attempts=0))
            continue
        rows=updates=0
        for video,bank in sorted(meta['videos'].items()):
            directory=path/'videos'/video;receipt=json.loads((directory/'receipt.json').read_text())
            questions=json.loads((directory/'questions.json').read_text())
            assert len(questions)==receipt['questions']<=receipt['budget']
            pairs=[(a['query_id'],a['track_id'],a['round']) for a in questions]
            assert len(set(pairs))==len(pairs)
            assert all(a['kind']!='IDENTITY' or not a['low_tentative_before'] for a in questions)
            with np.load(directory/'belief.npz') as b:
                assert np.all(b['pseudo_labels'][~b['asked_mask']]==-1)
                assert np.all(b['pseudo_labels'][:,np.asarray([sum(t==i for _,t in bank['observations'])<3 for i in b['tracks']])]==-1)
                posterior=expit(b['prior']+b['evidence']);posterior[b['absent']]=0
                assert np.allclose(posterior,b['posterior'],atol=1e-12,rtol=0)
                assert 0<=b['slope']<=1 and np.max(np.abs(b['prior']))<=2
                assert np.all(b['covered_strata'][b['absent']]>=b['needed_strata'][b['absent']])
                assert not np.any(b['absent'] & b['low_tentative'])
            mask=np.load(directory/'mask.npy');rows+=int(mask.sum())
            queries=[q for q in meta['episodes'] if q['video_id']==video]
            assert mask.shape==(len(queries),len(bank['lines']))
            for q,query in enumerate(queries):
                text=''.join(line+'\n' for line,keep in zip(bank['lines'],mask[q]) if keep)
                assert Path(sealed[video,query['expression_id']]['prediction_path']).read_text()==text
            with np.load(directory/'belief.npz') as b:assert not mask[b['output_absent']].any()
            for rr in receipt['rounds']:
                slow=rr['slow'];steps=[s for s in slow.get('records',[]) if s.get('optimizer_updated')]
                assert len(steps)==slow['optimizer_updates'];updates+=len(steps)
                for s in steps:
                    assert s['actual_positive_samples']==s['actual_negative_samples']==16
                    assert s['binary_positive_weight']==1
                    assert sum(s['sample_sources'].values())==32
                    assert np.isfinite([s['loss'],s['gradient_norm']]).all()
            if not seal['LoRA_enabled']:
                assert receipt['slow_GPU_seconds']==receipt['rescore_GPU_seconds']==0
                assert not (directory/'frame_cache').exists()
                em=receipt['EB']['after']
                assert em['converged']
                if not receipt['presence_inference']:
                    with np.load(directory/'belief.npz') as preset:
                        assert np.all(preset['presence_probability']==1)
                        assert not preset['output_absent'].any() and not preset['low_tentative'].any()
                    assert em['iterations']==0
                    assert all(q['kind'] in ('IDENTITY','POSITIVE_RECHECK') for q in questions)
                elif job['method']!='MINUS_EB':
                    assert em['maximum_iterations']==5000 and em['tolerance']==1e-7
            if job['method']=='SLOW_REPLAY':
                source=execution.active/job['pair']/'MINUS_SLOW'/('rho_'+str(job['rho']))/'videos'/video
                assert json.loads((source/'em_video_start.json').read_text())==json.loads((directory/'em_video_start.json').read_text())
                original=json.loads((source/'questions.json').read_text())
                original_receipt=json.loads((source/'receipt.json').read_text())
                assert [r['ds'] for r in receipt['rounds']]==[r['ds'] for r in original_receipt['rounds']]
                keys=('frame_id','track_id','query_id','effective_view','yes_probability','yes_logit','no_logit','round','kind')
                assert len(questions)==len(original)
                assert all(tuple(a[k] for k in keys)==tuple(b[k] for k in keys) for a,b in zip(questions,original))
                for r in range(5):
                    with np.load(source/f'presence_round{r}.npz') as a,np.load(directory/f'presence_round{r}.npz') as b:
                        assert all(a[k].tobytes()==b[k].tobytes() for k in BeliefStateV8.snapshot_fields)
        assert rows==seal['prediction_rows']
        records.append(dict(job,episodes=seal['episode_count'],prediction_rows=rows,optimizer_updates=updates,
            prediction_seal_sha256=file_sha(path/'predictions_seal.json'),GT_open_attempts=0))
    assert not attempts
    observed_updates=sum(r.get('optimizer_updates',0) for r in records)
    observed_replays=sum(r['method']=='SLOW_REPLAY' for r in records)
    result=dict(status='PASS',jobs=len(records),records=records,GT_open_attempts=0,
        asked_only_labels=True,observed_optimizer_updates=observed_updates,
        balanced_16_16_actual_updates='CHECKED_EVERY_OBSERVED_UPDATE' if observed_updates else 'NOT_EVALUATED_NO_UPDATES',
        full_coverage=True,observed_replay_jobs=observed_replays,
        fixed_replay_presence_byte_exact=True if observed_replays else 'NOT_EVALUATED_NO_REPLAY',
        GPU_or_teacher_calls=0,
        audit_output_note='Do not interpret vacuous frozen-model checks as real LoRA/replay evidence; numeric protocol is unchanged')
    atomic_json(execution.public/('validation_'+plan['tier']+'.json'),result);return result


if __name__=='__main__':
    execution,_=authorize_active();result=validate(execution)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
