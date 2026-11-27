"""Read-only phase accounting for a frozen completed-prefix receipt snapshot.

No models, predictions, GT or teacher inference. Durations are recorded phase
wall times and include CPU work; summing parallel unit times is NOT elapsed
experiment time or GPU kernel time. Prepaid assets are explicitly excluded.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha

PRIMARY=ROOT/'a100_runs/visual_classical_20261121/six_direction_20261122/conditional_state_20261127/primary'
DEST=ROOT/'a100_runs/visual_classical_20261121/supervisor/conditional_state_prefix_cost_20261127'
OWN=DEST/'registration.json'


def prepare():
    from flex_uda.a100.mechanism import generation_guard
    guard=generation_guard()
    if OWN.exists():raise FileExistsError(OWN)
    plan=json.loads((PRIMARY/'registration.json').read_text())
    paths={PRIMARY/'registration.json',Path(__file__)};units=[]
    for case in plan['cases']:
        folder=PRIMARY/'units'/case['pair']/case['evaluated_video'];receipt=folder/'generation_complete.json'
        if not receipt.exists():continue
        attempts=sorted((folder/'primary').glob('attempt_*.json'))
        paths.add(receipt);paths.update(attempts)
        units.append(dict(pair=case['pair'],video=case['evaluated_video'],receipt=str(receipt),attempts=[str(p) for p in attempts]))
    assert units and not guard
    for name,sha in plan['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    atomic_json(OWN,dict(status='REGISTERED_COMPLETED_PREFIX_COST_RECEIPTS',units=units,
        inputs={str(p):file_sha(p) for p in sorted(paths)},original48_source_sha256=plan['source_sha256'],
        exclude='All original source/PCD/DINO extraction, frozenQwen loading/paid QA and inherited6k KD are prepaid; no claim that their full deployment cost is zero.',
        scope='Current recorded phase wall time, NOT pure GPU, fresh deployment, equal-time accuracy, or total elapsed time of concurrent units.',
        compare='Only completed receipt snapshot; additive phase percentages reported only when interval sums are consistent and one primary attempt is recorded.',
        no_new_models_teacher_predictions_GT_or_seeds=True))
    print(json.dumps(dict(status='REGISTERED_PREFIX_COST_AUDIT',units=len(units))),flush=True)


def run():
    from flex_uda.a100.mechanism import generation_guard
    guard=generation_guard();reg=json.loads(OWN.read_text())
    if (DEST/'result.json').exists():raise FileExistsError(DEST/'result.json')
    for name,sha in reg['inputs'].items():assert file_sha(name)==sha,name
    for name,sha in reg['original48_source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    rows=[]
    for unit in reg['units']:
        doc=json.loads(Path(unit['receipt']).read_text());assert doc['GT_open_attempts']==0 and doc['new_teacher_calls']==0
        primary=doc['primary_fit'];rounds=primary['rounds'];ordinary=doc['ordinary_fits']
        s=sum(r['solver_seconds'] for r in rounds);r=sum(row['observer_seconds'] for row in rounds)
        primary_wall=float(primary['wall_seconds']);ordinary_wall=float(doc['ordinary_phase_wall_seconds'])
        ordinary_loops=sum(p['wall_seconds'] for fit in ordinary.values() for p in fit['phase_fits'])
        prediction=float(doc['prediction_seal_phase_wall_seconds']);unit_wall=float(doc['current_attempt_wall_seconds'])
        residuals=[primary_wall-s-r,ordinary_wall-ordinary_loops,unit_wall-primary_wall-ordinary_wall-prediction]
        raw=[s,r,primary_wall,ordinary_wall,ordinary_loops,prediction,unit_wall]
        assert all(x>=0 for x in raw)
        additive=len(unit['attempts'])==1 and min(residuals)>=-.01
        row=dict(pair=unit['pair'],video=unit['video'],primary_attempt_files=len(unit['attempts']),
            solver_loop_seconds=s,observer_loop_seconds=r,primary_total_wall_seconds=primary_wall,
            ordinary_loop_seconds=ordinary_loops,ordinary_outer_wall_seconds=ordinary_wall,
            prediction_and_all22_seals_seconds=prediction,recorded_unit_current_attempt_seconds=unit_wall,
            primary_other_seconds=residuals[0],ordinary_other_seconds=residuals[1],between_phase_other_seconds=residuals[2],
            additive_intervals_consistent=additive,
            interpretation='Single recorded attempt and additive-consistent intervals' if additive else 'Components may span attempts; do not construct a single-attempt percentage from this row')
        rows.append(row)
    settings={}
    for pair in sorted({x['pair'] for x in rows}):
        own=[x for x in rows if x['pair']==pair];valid=[x for x in own if x['additive_intervals_consistent']]
        result=dict(units=len(own),additive_consistent_units=len(valid),median_recorded_unit_seconds=statistics.median(x['recorded_unit_current_attempt_seconds'] for x in own))
        if valid:
            total=sum(x['recorded_unit_current_attempt_seconds'] for x in valid)
            keys=['solver_loop_seconds','observer_loop_seconds','primary_other_seconds','ordinary_loop_seconds',
                  'ordinary_other_seconds','prediction_and_all22_seals_seconds','between_phase_other_seconds']
            pieces={k:sum(x[k] for x in valid) for k in keys}
            assert abs(sum(pieces.values())-total)<1e-6*max(1,total)
            result.update(sum_parallel_unit_wall_seconds_not_total_elapsed=total,
                phase_accounting_seconds=pieces,phase_fraction_of_summed_unit_wall={k:v/total for k,v in pieces.items()})
        settings[pair]=result
    assert not guard
    for name,sha in reg['inputs'].items():assert file_sha(name)==sha,name
    result=dict(status='COMPLETE_READ_ONLY_PREFIX_PHASE_RECEIPT_RECONCILIATION',units=len(rows),settings=settings,rows=rows,
        input_receipts_unchanged=True,original48sources_unchanged=True,new_teacher_calls=0,
        new_neural_fits_or_predictions=0,GT_open_attempts=0,not_fresh_cold_deployment_or_iso_compute=True,
        concurrent_unit_time_sum_is_not_elapsed_or_GPU_time=True,
        remainder='Includes model/feature IO and assembly, repeated neural scoring, HMM/ELBO inference, source checks and checkpointing; cannot label it pure NAS time.',
        prediction_scope='Three ordinary head inference, pooling, shared-memory inference, CountMatched and22-output serialization, not a single deployed method.',
        prepaid_scope=reg['exclude'],goal_status='ACTIVE',time_unix=time.time())
    atomic_json(DEST/'result.json',result)
    print(json.dumps({k:result[k] for k in ['status','units','settings','new_teacher_calls','GT_open_attempts']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else run()
