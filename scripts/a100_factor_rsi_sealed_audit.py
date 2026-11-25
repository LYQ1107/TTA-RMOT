"""Independent all-39 seal/quantity/ordinary-adequacy audit, without GT."""
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def counts(path):
    return Counter(int(float(line.split(',')[0])) for line in Path(path).read_text().splitlines() if line.strip())


def main():
    import torch
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.presence.predictions import verify_predictions
    from flex_tta.runtime.pf_receipts import atomic_json,file_sha
    from scripts.a100_factor_rsi_primary_quality import OUT,PRIMARY,COMPARATORS,registration
    plan,q=registration();guard=generation_guard()
    barrier=json.loads((OUT/'all_39_predictions_sealed_before_GT.json').read_text())
    assert barrier['GT_open_attempts']==0 and len(barrier['tasks'])==39
    assert barrier['quality_registration_sha256']==file_sha(OUT/'quality_registration.json')
    seals={};summary=[];projected_query_files=0;fits=0
    for task in barrier['tasks']:
        key=task['pair'],task['method'];assert key not in seals
        seals[key]=verify_predictions(task['folder'])
    for case in plan['cases']:
        pair=case['pair'];folder=OUT/pair
        done=json.loads((folder/'all_case_predictions_complete.json').read_text())
        assert done['GT_open_attempts']==0 and not any(done['CountSelf_changed_rows'].values())
        ordinary={}
        for name in COMPARATORS:
            state=torch.load(folder/(name+'.pt'),map_location='cpu',weights_only=False)
            assert state['name']==name and state['width'] in (128,256)
            assert state['teacher_data_sha256']==file_sha(folder/'paid_answers.json')
            assert len(state['reports'])==5
            assert state['reports']==done['ordinary_fits'][name]
            assert state['hard']==('HARD' in name) and state['whole_only']==('_WHOLE_' in name)
            maximum=0.;total_steps=0
            for i,r in enumerate(state['reports']):
                assert r['teacher_fit_gate_passed'] and r['best_maximum_per_role_excess_KL']<=.005
                assert r['total_actual_steps']>=(4000 if i==4 else 500)
                assert r['seed']==0 and not r['manual_GT_selection']
                assert max(v['excess_KL'] for v in r['roles'].values())<=.005
                maximum=max(maximum,r['best_maximum_per_role_excess_KL']);total_steps+=r['total_actual_steps'];fits+=1
            ordinary[name]=dict(width=state['width'],prefixes=5,maximum_excess_KL=maximum,
                total_registered_optimizer_steps=total_steps,checkpoint_sha256=file_sha(folder/(name+'.pt')))
        query_files=0
        for (p,name),seal in seals.items():
            if p!=pair or 'count_method' not in seal:continue
            donor=seals[p,seal['count_method']]
            by={(r['video'],r['expression']):r for r in donor['episodes']}
            assert len(by)==len(seal['episodes'])==case['query_count']
            for row in seal['episodes']:
                original=by[row['video'],row['expression']]
                assert counts(row['prediction_path'])==counts(original['prediction_path']), (p,name,row['expression'])
                projected_query_files+=1;query_files+=1
        summary.append(dict(pair=pair,methods=13,ordinary=ordinary,projected_query_files_checked=query_files))
    assert fits==60 and len(seals)==39 and not guard
    receipt=dict(status='PASS_ALL39_SEALS_60_ORDINARY_PREFIX_GATES_AND_TEXT_LEVEL_COUNTS',cases=summary,
        actual_prediction_seals=39,ordinary_final_heads=12,ordinary_prefix_gates=fits,
        exact_projected_query_files=projected_query_files,no_GT_reads=True,
        no_new_models_teacher_calls_fits_predictions_or_seeds=True,
        CountSelf_all_masks_exact=True,target_GT_not_used_for_model_selection=True,
        no_HOTA_quality_claim=True,source_sha256=file_sha(Path(__file__)))
    atomic_json(OUT/'independent_sealed_prediction_audit.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
