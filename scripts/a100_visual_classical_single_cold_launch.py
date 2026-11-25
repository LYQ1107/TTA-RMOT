"""Two actual independent single-method cold processes on the same GPU3.

The fixed order is declared; OS cache and concurrent host work are retained.
Each process really reloads source/teacher, runs the whole sparse C4 stream,
fits one bounded-soft head per video and writes only its own prediction.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT


def main():
    started=time.perf_counter();directory=OUT/'supervisor';state=directory/'single_cold_state.json';jobs=[];records={}
    if (OUT/'analysis/independent_single_cold_C4.json').is_file():return
    for method,tag in [('CLIP_KD','cold_single_kd'),('CLIP_COHORT','cold_single_cohort')]:
        name='C4_key_single_'+method+'_seed0';root=OUT/'cold_full_c4'/name
        command=[sys.executable,'scripts/a100_visual_classical_cold_full_launch.py','--name',name,'--gpu','3',
            '--perception-mode','track-three','--skip-own-queue','--methods',method,'--state-tag',tag,'--analysis-stem',name]
        log_path=directory/('recovery_strong_field_cold_full_'+tag+'_supervisor.log')
        with log_path.open('a',buffering=1) as log:
            process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
            jobs.append(dict(method=method,pid=process.pid,physical_GPU=3,log=str(log_path)))
            atomic_json(state,dict(status='RUNNING_INDEPENDENT_SINGLE_METHOD_COLD',pid=os.getpid(),jobs=jobs))
            code=process.wait();jobs[-1]['exit_code']=code
        if code:raise RuntimeError('Single-method cold process failed: '+str(log_path))
        receipt=json.loads((root/'completion.json').read_text());evaluation=json.loads((root/'official_evaluation.json').read_text())
        process_receipt=json.loads((root/'worker_process_receipt.json').read_text())
        assert receipt['single_method_execution'] and receipt['single_uninterrupted_attempt']
        assert receipt['GT_open_attempts']==receipt['prepaid_artifact_reads']==receipt['teacher']['cache_hits']==0
        assert receipt['teacher']['new_calls']==receipt['logical_teacher_calls']==3508
        assert [r['method'] for r in receipt['methods']]==[method]
        assert receipt['protocol']['actual_trained_label_types']==['CLIP']
        records[method]=dict(root=str(root),HOTA=evaluation['HOTA'][method],actual_worker_process_seconds=process_receipt['wall_seconds'],
            actual_internal_worker_seconds=receipt['actual_worker_wall_seconds'],teacher=receipt['teacher'],
            method_cost=receipt['cost_by_method'][method],protocol=receipt['protocol'])
        print(json.dumps(dict(event='INDEPENDENT_SINGLE_COLD_METHOD_COMPLETE',method=method,HOTA=records[method]['HOTA'],
            actual_worker_process_seconds=process_receipt['wall_seconds'],new_Qwen_calls=receipt['teacher']['new_calls'],cache_hits=0)),flush=True)
    kd=Path(records['CLIP_KD']['root']);full=Path(records['CLIP_COHORT']['root']);questions_same=True;pcd_equal=True;prior_equal=True;maximum_p_delta=0.
    for folder in sorted((kd/'source').iterdir()):
        video=folder.name
        a=json.loads((folder/'questions_before_answers.json').read_text())['requests']
        b=json.loads((full/'source'/video/'questions_before_answers.json').read_text())['requests']
        questions_same &= a==b
        answers_a=json.loads((folder/'answers.json').read_text())['answers'];answers_b=json.loads((full/'source'/video/'answers.json').read_text())['answers']
        lookup_a={(r['local_query'],r['local_track']):r['yes_probability'] for r in answers_a}
        lookup_b={(r['local_query'],r['local_track']):r['yes_probability'] for r in answers_b}
        if lookup_a.keys()==lookup_b.keys():maximum_p_delta=max(maximum_p_delta,max(abs(lookup_a[k]-lookup_b[k]) for k in lookup_a))
        with np.load(folder/'sparse_inputs.npz') as a,np.load(full/'source'/video/'sparse_inputs.npz') as b:
            pcd_equal &= np.array_equal(a['pcd'],b['pcd']);prior_equal &= np.array_equal(a['prior'],b['prior'])
    from scripts.a100_visual_classical_analyze import read_stats
    from scripts.a100_mechanism_analyze import compare_official
    contrast=compare_official(read_stats(full/'methods/CLIP_COHORT'),read_stats(kd/'methods/CLIP_KD'))
    result=dict(status='PASS_INDEPENDENT_SINGLE_METHOD_COLD_COMPARISON',methods=records,comparison=contrast,
        same_physical_GPU=3,source_features_bitwise_equal=bool(pcd_equal),source_priors_bitwise_equal=bool(prior_equal),
        same_requested_questions=bool(questions_same),maximum_raw_teacher_probability_difference=maximum_p_delta,
        exact_same_teacher_probabilities=maximum_p_delta==0 and questions_same,
        each_method_really_loaded_both_models=True,only_one_head_type_per_video_and_one_method_output=True,
        head_refit_for_each_video_with_only_its_available_prefix=True,
        budget_per_method=3508,cache_hits_each=0,teacher_and_feature_payloads_not_prepaid=True,
        observed_process_wall_not_component_estimate=True,order=['CLIP_KD','CLIP_COHORT'],
        order_not_counterbalanced=True,OS_cache_retained=True,shared_host_NAS_concurrent_work=True,
        all_development_data=True,candidates_provided_upstream_generation_not_remeasured=True,
        total_supervisor_wall_includes_separate_GT_evaluations=time.perf_counter()-started)
    atomic_json(OUT/'analysis/independent_single_cold_C4.json',result)
    atomic_json(state,dict(status='PASS_INDEPENDENT_SINGLE_METHOD_COLD_COMPARISON',pid=os.getpid(),jobs=jobs))
    print(json.dumps(dict(event='INDEPENDENT_SINGLE_COLD_COMPLETE',comparison=contrast,same_questions=questions_same,
        raw_teacher_probability_delta=maximum_p_delta)),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(OUT/'supervisor/single_cold_state.json',dict(status='FAILED_SINGLE_METHOD_COLD_REQUIRES_ROOT_CAUSE',pid=os.getpid()))
        raise
