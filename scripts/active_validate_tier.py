"""Post-queue complete-candidate audit; reads no GT and runs no model."""
from argparse import ArgumentParser
import csv
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_uda.active.execution import authorize_active
from flex_uda.presence.runtime import target_metadata,runtime_guard
from flex_uda.presence.predictions import verify_predictions
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.active_eval import official_stats
from scripts.active_plan import jobs as full_jobs


def key(job):return job['pair'],job['method'],float(job['rho'])


def main():
    parser=ArgumentParser(description=__doc__);parser.add_argument('--durable-dir',type=Path,required=True)
    args=parser.parse_args();p,_=authorize_active();opens=runtime_guard()
    plan=json.loads((p.public/'registered_jobs.json').read_text())
    start=json.loads((p.public/'execution_started.json').read_text())
    assert file_sha(p.manifest_path)==start['manifest_sha256']
    assert file_sha(p.public/'registered_jobs.json')==start['plan_sha256']
    assert file_sha(plan['prereg_path'])==plan['prereg_sha256']
    previous=json.loads(Path(p.manifest['continuation_of']).read_text())
    changed={'public_root','launcher_root','registered_tier','allowed_registered_jobs','prereg_path','prereg_sha256',
             'continuation_of','full_matrix_authorized','user_authorization','prior_ten_results_immutable'}
    assert all(p.manifest[k]==value for k,value in previous.items() if k not in changed)
    terminal=json.loads((args.durable_dir/'completion.json').read_text())
    done=json.loads((Path(start['coordinator'])/'completion.json').read_text())
    assert terminal['exit_code']==done['launcher_exit_code']==0 and done['status']=='COMPLETE'
    assert len(done['evaluated_jobs'])==done['expected_jobs']==len(plan['jobs'])==18 and not done['evaluation_errors']
    assert all(j['exit_code']==0 for j in done['evaluated_jobs'])
    assert {key(j) for j in done['evaluated_jobs']}=={key(j) for j in plan['jobs']}
    launcher=Path(start['initial_workers'][0]['log']).parent
    launched=json.loads((launcher/'completion.json').read_text())['jobs']
    assert len(launched)==18 and all(j['exit_code']==0 for j in launched)
    assert {key(j) for j in launched}=={key(j) for j in plan['jobs']}
    physical=sorted({j['physical_gpu'] for j in launched});assert len(physical)<=2
    allowed=plan['prior_completed_jobs']+plan['jobs'];assert len({key(j) for j in allowed})==28
    assert all({k:j[k] for k in ('pair','method','rho')} in full_jobs() for j in allowed)
    for j in plan['prior_completed_jobs']:assert file_sha(j['path'])==j['sha256']
    with (p.public/'main.csv').open() as f: rows=list(csv.DictReader(f))
    meta_by_target={};methods=[];episode_total=prediction_total=0
    for job in allowed:
        pair,method,rho=key(job)
        row=next(r for r in rows if r['method']!='GATE_REASSIGN_REF' and key(r)==(pair,method,rho))
        folder=Path(row['path']);seal=verify_predictions(folder);teacher=seal['teacher']
        assert seal['method']==method and seal['pair']==pair and seal['rho']==rho
        assert seal['GT_open_attempts']==teacher['GT_open_attempts']==0 and teacher['backend_unmodified']
        assert teacher['revision']=='89644892e4d85e24eaac8bacfd4f463576704203'
        assert teacher['teacher_weight_sha256']=='9a13380b4279bc98f4680bd083ff4995fdcfec120cfded459513be90d041a956'
        target=seal['target']
        if target not in meta_by_target:meta_by_target[target]=target_metadata(target)
        meta=meta_by_target[target];expected={(q['video_id'],q['expression_id']) for q in meta['episodes']}
        assert len(expected)==len(seal['episodes'])==seal['episode_count']
        assert {(e['video'],e['expression']) for e in seal['episodes']}==expected
        sealed={(e['video'],e['expression']):e for e in seal['episodes']}
        gradient=[];updates=0
        for video,bank in meta['videos'].items():
            directory=folder/'videos'/video;receipt=json.loads((directory/'receipt.json').read_text())
            mask=np.load(directory/'mask.npy')
            assert file_sha(directory/'mask.npy')==receipt['mask_sha256']
            queries=[q for q in meta['episodes'] if q['video_id']==video]
            assert mask.shape==(len(queries),len(bank['observations']))
            for qi,q in enumerate(queries):
                entry=sealed[video,q['expression_id']]
                selected=np.flatnonzero(mask[qi])
                assert Path(entry['prediction_path']).read_text()==''.join(bank['lines'][i]+'\n' for i in selected)
                assert entry['selected_rows']==len(selected)
            if method.startswith('YESTRACK_BUDGET'):
                full=json.loads((p.active/pair/'FULL'/f'rho_{rho}'/'videos'/video/'receipt.json').read_text())
                assert receipt['actual_calls']==full['questions']
            for rr in receipt.get('rounds',[]):
                slow=rr['slow'];updates+=slow['optimizer_updates']
                for g in slow.get('records',[]):
                    if g.get('optimizer_updated'):
                        assert math.isfinite(g['loss']) and math.isfinite(g['gradient_norm'])
                        gradient.append(g['gradient_norm'])
        assert updates==len(gradient)
        exact=official_stats(folder)['HOTA_unrounded']
        assert abs(exact-float(row['HOTA_unrounded']))<1e-8
        methods.append(dict(pair=pair,method=method,rho=rho,model=seal['model'],target=target,videos=len(meta['videos']),
            episodes=seal['episode_count'],prediction_rows=seal['prediction_rows'],HOTA=exact,
            prediction_seal_sha256=file_sha(folder/'predictions_seal.json'),GT_open_attempts=0,
            optimizer_updates=updates,finite_gradient_updates=len(gradient),zero_gradient_updates=sum(g==0 for g in gradient),
            min_preclip_gradient_norm=min(gradient) if gradient else None,max_preclip_gradient_norm=max(gradient) if gradient else None))
        episode_total+=seal['episode_count'];prediction_total+=seal['prediction_rows']
    assert episode_total==15806 and not opens
    equivalent=[]
    for first,second in (('C1','C6'),('C3','C4')):
        a=json.loads((p.active/first/'UNIFORM_K/rho_0.3/predictions_seal.json').read_text())
        b=json.loads((p.active/second/'UNIFORM_K/rho_0.3/predictions_seal.json').read_text())
        equal={(e['video'],e['expression']):e['prediction_sha256'] for e in a['episodes']}=={
            (e['video'],e['expression']):e['prediction_sha256'] for e in b['episodes']}
        equivalent.append(dict(first=first,second=second,method='UNIFORM_K',byte_equivalent=equal))
    result=dict(status='CRITICAL28_VALIDATED_FULL108_NOT_COMPLETE',completed_current_tier=18,
        old_completed_reused=10,total_completed=28,full_matrix=108,remaining_not_authorized=80,
        total_episodes=episode_total,prediction_rows=prediction_total,methods=methods,
        actual_physical_GPUs=physical,new_launcher_jobs=18,new_failed_launcher_jobs=0,
        prior_ten_seals_unchanged=True,method_hyperparameters_and_source_heads_unchanged=True,
        MOT_boxes_IDs_frames_exact_from_masks=True,per_video_YESTRACK_budgets_match_FULL=True,
        official_HOTA_reconstructed_from_detailed_stats=True,GT_open_attempts_by_validator=len(opens),
        model_teacher_or_GT_called_by_validator=False,gradient_norms_are_pre_clipping=True,
        uniform_same_target_output_equivalence=equivalent,
        legacy_interrupted_session_closing_audits_missing_remain_disclosed=True)
    atomic_json(p.public/'phase_validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='methods'},indent=2))


if __name__=='__main__':main()
