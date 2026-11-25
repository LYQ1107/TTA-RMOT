"""Summarize all registered primary head-fit phases, without reading GT.

Refuse partial conditions. These are observed replay fitting costs, not
single-method cold deployments, GPU kernel times, or end-to-end costs.
"""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_conditional_six_setting import ARMS, conditions, OUT, PUBLIC

NAME = 'conditional_learning_cost'


def distribution(values):
    values = np.asarray(values, dtype=float)
    return dict(min=float(values.min()), mean=float(values.mean()), max=float(values.max()))


def main():
    records = conditions()
    assert len(records) == 18
    missing = [r['pair']+f"_seed{r['seed']}" for r in records
        if not (Path(r['folder'])/'completion.json').is_file()]
    if missing:
        print(json.dumps(dict(status='NOT_READY_ALL18_PRIMARY_RAW_CONDITIONS_REQUIRED', missing=missing)))
        return 3
    summaries = {}
    all_rows = []
    total_prefixes = 0
    for record in records:
        condition = json.loads((Path(record['folder'])/'completion.json').read_text())
        assert condition['status'] == 'SEALED_CONDITIONAL_OBSERVATION_RAW_NO_GT'
        assert condition['GT_open_attempts'] == 0 and condition['new_teacher_or_student_calls'] == 0
        assert not condition['target_fit_failures']
        rows = []
        video_phases = []
        for video in condition['receipts']:
            assert set(video['fits']) == set(ARMS)
            assert video['GT_open_attempts'] == 0 and video['quality_replay_not_deployment_cost']
            assert video['normalization_current_and_past_anchors_only']
            actual = 0
            actual_wall = 0.
            for arm, fit in video['fits'].items():
                assert fit['target_fit_gate_pass'] and not fit['target_GT_or_HOTA_used']
                assert fit['final_CE_recomputed_from_actual_float32_deployed_head']
                excess = float(fit['metrics']['objective_excess'])
                assert excess <= .005+1e-12
                seconds = float(fit['wall_seconds'])
                assert np.isfinite(seconds) and seconds >= 0
                owner = fit.get('identical_own_prefix_fit_reused_from')
                if owner is not None:
                    assert owner in video['fits'] and ARMS.index(owner) < ARMS.index(arm)
                    assert not video['fits'][owner].get('identical_own_prefix_fit_reused_from')
                    assert seconds == float(video['fits'][owner]['wall_seconds'])
                else:
                    actual += 1
                    actual_wall += seconds
                rows.append(dict(pair=record['pair'], seed=record['seed'], video=video['video'], arm=arm,
                    reference_fit_wall_seconds=seconds, actual_unique_fit=owner is None,
                    identical_fit_owner=owner, steps=int(fit['steps']), width=int(fit['width']),
                    continuation_blocks=len(fit['teacher_risk_continuations']),
                    teacher_examples=int(fit['teacher_examples']), objective_excess=excess))
            assert actual == video['actual_head_fits']
            assert len(ARMS)-actual == video['exact_identical_fit_reuse']
            prediction = float(video['prediction_and_output_wall_seconds'])
            wait = float(video['dense_shared_feature_wait_seconds'])
            overhead = float(video['wall_seconds'])-actual_wall-prediction-wait
            assert overhead >= -1e-6, (record['pair'], record['seed'], video['video'], overhead)
            video_phases.append(dict(video=video['video'], actual_unique_fit_wall_seconds=actual_wall,
                all24_outputs_prediction_and_write_seconds=prediction, shared_RGB_wait_seconds=wait,
                remaining_input_normalization_setup_seconds=max(overhead, 0.),
                valid_video_receipt_wall_seconds=float(video['wall_seconds'])))
        assert sum(r['actual_unique_fit'] for r in rows) == condition['actual_head_fits']
        total_prefixes += len(video_phases)
        slug = record['pair']+f"_seed{record['seed']}"
        summaries[slug] = dict(videos=len(video_phases),
            per_arm={arm:dict(reference_fit_wall_seconds=sum(r['reference_fit_wall_seconds'] for r in rows if r['arm']==arm),
                actual_unique_fit_wall_seconds=sum(r['reference_fit_wall_seconds'] for r in rows if r['arm']==arm and r['actual_unique_fit']),
                actual_fits=sum(r['actual_unique_fit'] for r in rows if r['arm']==arm),
                reused_fit_receipts=sum(not r['actual_unique_fit'] for r in rows if r['arm']==arm),
                AdamW_steps_in_reference_receipts=sum(r['steps'] for r in rows if r['arm']==arm),
                receipts_with_teacher_risk_continuation=sum(r['continuation_blocks']>0 for r in rows if r['arm']==arm),
                maximum_objective_excess=max(r['objective_excess'] for r in rows if r['arm']==arm)) for arm in ARMS},
            shared_phase_sums={k:sum(r[k] for r in video_phases) for k in video_phases[0] if k!='video'})
        all_rows.extend(rows)
    assert total_prefixes == 444 and len(all_rows) == 5328
    repeated = {}
    for pair in ('C1','C2','C3','C4','C5','C6'):
        seeds = [summaries[pair+f'_seed{s}'] for s in (0,17,42)]
        repeated[pair] = {arm:dict(
            reference_fit_wall_seconds=distribution([r['per_arm'][arm]['reference_fit_wall_seconds'] for r in seeds]),
            actual_fits=sum(r['per_arm'][arm]['actual_fits'] for r in seeds),
            reused_fit_receipts=sum(r['per_arm'][arm]['reused_fit_receipts'] for r in seeds),
            receipts_with_teacher_risk_continuation=sum(r['per_arm'][arm]['receipts_with_teacher_risk_continuation'] for r in seeds),
            maximum_objective_excess=max(r['per_arm'][arm]['maximum_objective_excess'] for r in seeds)) for arm in ARMS}
    result = dict(status='PASS_ALL18_REGISTERED_PRIMARY_HEAD_FIT_COST_PHASES', conditions=summaries,
        repeated=repeated, valid_video_prefixes=total_prefixes, fit_receipts=len(all_rows),
        actual_unique_fits=sum(r['actual_unique_fit'] for r in all_rows),
        reused_fit_receipts=sum(not r['actual_unique_fit'] for r in all_rows),
        actual_unique_fit_wall_seconds_sum=sum(r['reference_fit_wall_seconds'] for r in all_rows if r['actual_unique_fit']),
        widths=dict(Counter(str(r['width']) for r in all_rows)),
        scope=[
            'Wall times are original fit-function phases, including actual optimization/continuation and deployed-head checking; not GPU kernel times.',
            'Reference arm times include copies of measured exactly identical own-prefix fits. Unique execution totals exclude those copies.',
            'Prediction/output phases serve all24 variants together; no equal split or single-method latency claim.',
            'Video receipts span input/normalization/setup plus unique fits/output/waits. Source/teacher weights and neural features were prepaid.',
            'Retained prefix receipts survive recovery. A completion wall time may describe only a resumed attempt; it is not the whole original run.',
            'Lost incomplete-video retries, failed-attempt overhead and exact local-cache preparation are separate recovery work; these sums do not include them.',
            'Do not add the old failed-controller attempt wall time to these retained prefixes: they overlap.',
            'No quality/HOTA/GT read, parameter selection, new fit or prediction. All18 raw conditions required; queue completion does not establish research success.',
            'A fresh paired deployment with matched input version, real source/teacher forwards and one method per process remains required.'
        ], target_GT_or_HOTA_read=False, new_teacher_student_or_head_calls=0,
        end_to_end_cold_cost=False, goal_status='ACTIVE')
    atomic_json(OUT/'learning_cost_phases.json', dict(result, per_prefix_rows=all_rows))
    (PUBLIC/(NAME+'.json')).write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps({k:result[k] for k in ('status','valid_video_prefixes','fit_receipts','actual_unique_fits','reused_fit_receipts','actual_unique_fit_wall_seconds_sum')}))
    return 0


if __name__ == '__main__': raise SystemExit(main())
