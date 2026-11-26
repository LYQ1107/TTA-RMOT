"""Reproduce same-CPU policy intervention facts, never RMOT/GT quality."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_rsi_policy_distribution_pilot import OUT,PUBLIC,VIDEOS,check_plan

BRANCHES=('CPU_ORIGINAL_SGD','INFO_GATED_SGD','INFO_GATED_TRUST')


def tensors_distance(a,b):
    assert a.keys()==b.keys()
    return dict(byte_identical=all(torch.equal(a[k],b[k]) for k in a),
                parameter_L2=float(torch.sqrt(sum((a[k].double()-b[k].double()).square().sum() for k in a))))


def describe(row):
    return {k:row['case'][k] for k in ('query_id','track_id','frame_id','view')}


def main():
    check_plan();results=[];maximum_KL=maximum_ordinary=0.
    for name in BRANCHES:
        done=json.loads((OUT/name/'completion.json').read_text())
        assert done['GT_open_attempts']==0 and not done['RMOT_predictions_generated']
    for video in VIDEOS:
        raw={};rows={};checkpoint={}
        for name in BRANCHES:
            local=OUT/name/video;raw[name]=json.loads((local/'receipt.json').read_text())
            paid=json.loads((local/'paid.json').read_text())['paid'];seq=json.loads((local/'sequence.json').read_text())['sequence']
            checkpoint[name]=torch.load(local/'closed.pt',map_location='cpu',weights_only=False)
            ordinary=raw[name]['ordinary_all_paid_teacher_fit']
            assert all(v['teacher_fit_gate_passed'] and v['excess_risk']<=.005 for v in ordinary.values())
            maximum_ordinary=max(maximum_ordinary,*[v['excess_risk'] for v in ordinary.values()])
            rows[name]=dict(paid={r['uid']:r for r in paid},order=[r['uid'] for r in seq],
                probes=[[r['uid'] for r in seq if r['role']=='probe' and r['stage']==stage] for stage in range(3)])
        reference=BRANCHES[0];base=rows[reference];base_checkpoint=checkpoint[reference]
        result=dict(video=video,branches={})
        for name in BRANCHES:
            row=rows[name];report=raw[name];ck=checkpoint[name]
            assert row['probes']==base['probes']
            assert report['teacher_budget']['per_query_count']==raw[reference]['teacher_budget']['per_query_count']
            for uid in row['paid'].keys()&base['paid'].keys():assert row['paid'][uid]['probability']==base['paid'][uid]['probability']
            assert np.array_equal(ck['solver']['mean'],base_checkpoint['solver']['mean'])
            assert np.array_equal(ck['solver']['std'],base_checkpoint['solver']['std'])
            a,b=row['paid'].keys()-base['paid'].keys(),base['paid'].keys()-row['paid'].keys()
            tracks=lambda d:{(x['local_query'],x['local_track']) for x in d.values()}
            points=lambda d:{(x['local_query'],x['local_track'],x['case']['frame_id']) for x in d.values()}
            updates=[r['actor_update'] for r in report['rounds']]
            KL=max([u.get('mean_recorded_state_KL',0.) for u in updates])
            maximum_KL=max(maximum_KL,KL)
            assert KL<=.01+1e-8
            if name.startswith('INFO_'):
                assert all(u['actor_parameter_L2_change']==0 for u in updates if u['skipped_zero_information_context'])
            result['branches'][name]=dict(budget=report['teacher_budget']['unique_teacher_bundles'],
                replacement_questions=len(a),same_question_order=row['order']==base['order'],
                changed_query_track_pairs=len(tracks(row['paid'])^tracks(base['paid'])),
                changed_visual_nodes=len(points(row['paid'])^points(base['paid'])),
                added=[describe(row['paid'][u]) for u in sorted(a)],removed=[describe(base['paid'][u]) for u in sorted(b)],
                solver_vs_original_CPU=tensors_distance(ck['solver']['model'],base_checkpoint['solver']['model']),
                observer_vs_original_CPU=tensors_distance(ck['actor']['model'],base_checkpoint['actor']['model']),
                conditional_teacher_probe_CE_after=[u['teacher_probe_CE_after'] for u in updates],
                rewards=[u['reward'] for u in updates],
                zero_information_skipped=[u.get('skipped_zero_information_context',False) for u in updates],
                max_mean_recorded_decision_state_KL=KL,
                actual_parameter_changes=[u['actor_parameter_L2_change'] for u in updates],
                ordinary_teacher_excess_KL={k:v['excess_risk'] for k,v in report['ordinary_all_paid_teacher_fit'].items()})
        assert all(row['solver_vs_original_CPU']['byte_identical'] for row in result['branches'].values()) if video==VIDEOS[0] else True
        results.append(result)
    result=dict(status='PASS_SAME_CPU_POLICY_INTERVENTION_FACTS_REPRODUCED_NOT_RMOT_QUALITY',
        rows=results,neural_seeds=[0],same_CPU_and_two_threads=True,
        same_inputs_normalization_budget_fixed_probes=True,
        maximum_ordinary_teacher_excess_KL=maximum_ordinary,maximum_observer_recorded_state_KL=maximum_KL,
        target_manual_GT_or_HOTA_read=False,RMOT_predictions_generated=False,actual_new_teacher_GPU_calls=0,
        existing_GPU_original_pilot_comparisons_context_only_not_pure_causal=True,
        interpretation='Bounded policy update can change future observations; zero-information baseline noise can be removed. Two videos do not establish RMOT benefit or RSI success.',
        all_data_DEVELOPMENT=True,goal_complete=False,original_live_six_setting_primary_unchanged=True)
    atomic_json(OUT/'same_CPU_causal_analysis.json',result)
    atomic_json(PUBLIC/'rsi_policy_distribution_same_CPU_analysis_20261124.json',result)
    print(json.dumps(dict(status=result['status'],max_KL=maximum_KL,max_ordinary_KL=maximum_ordinary,
        replacements={r['video']:{n:d['replacement_questions'] for n,d in r['branches'].items()} for r in results})),flush=True)


if __name__=='__main__':main()
