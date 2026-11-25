#!/usr/bin/env python3
"""Full425 final checkpoints only after Table A is verified on GitHub."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.launch_uda_workers import audit_gpus
from scripts.run_diag_fit_training import child
from scripts.run_train2val_training import RUN,PUBLIC,ARMS


def run():
    gate=json.loads((RUN/'fit_table_publication_gate.json').read_text())
    if not gate['remote_HEAD_verified'] or file_sha(PUBLIC/'TABLE_A.md')!=gate['FIT_TABLE_sha256']:
        raise PermissionError('push and verify Table A before Val inference')
    completion=json.loads((RUN/'full_completion.json').read_text())
    names=[a for a in ARMS if completion['arms'][a]['status']=='COMPLETE']
    attempt=RUN/'inference_launchers'/str(time.time_ns());attempt.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    while True:
        try:chosen,gpus,processes=audit_gpus(len(names),minimum_free_mib=20480);break
        except RuntimeError as e:
            print(json.dumps(dict(event='T2V_INFERENCE_RESOURCE_WAIT',reason=str(e))),flush=True);time.sleep(20)
    atomic_json(attempt/'admission.json',dict(selected=chosen,all_gpus=gpus,processes=processes,maximum_task_GPUs=4,sharing_authorized=True))
    def worker(arm,gpu):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],UDA_PHYSICAL_GPU=str(gpu['physical_gpu']),
            OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
        record=child([sys.executable,ROOT/'scripts/evaluate_mllm_distill.py','--checkpoint',RUN/'students'/arm/'checkpoint_final_step.pt',
            '--output-root',RUN/'evaluation','--method',arm],env,RUN/'logs/inference'/(arm+'.txt'),
            dict(arm=arm,phase='inference',physical_gpu=gpu['physical_gpu']))
        if record['exit_code']:raise RuntimeError('inference failed; preserve healthy425 episode receipts: '+arm)
        folder=RUN/'evaluation'/arm;metadata=json.loads((folder/'run_metadata.json').read_text())
        if metadata['valid_episodes']!=425 or metadata['selected_step']!=8973 or metadata['target_GT_open_attempts']:
            raise ValueError('incomplete or invalid full Val predictions')
        # Do not rewrite run_metadata after sealing; costs belong in sidecars.
        atomic_json(folder/'inference_costs.json',record)
        return arm,dict(run_metadata_path=str(folder/'run_metadata.json'),run_metadata_sha256=file_sha(folder/'run_metadata.json'),
            checkpoint_sha256=metadata['checkpoint_sha256'],episodes=metadata['episodes'])
    sealed={}
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        for future in as_completed([pool.submit(worker,a,g) for a,g in zip(names,chosen)]):
            name,record=future.result();sealed[name]=record
    result=dict(status='ALL_TRAINED_ARMS_COMPLETE425_SEALED_BEFORE_GT',methods=sealed,
        prior_Table_A_publication=gate,wall_seconds=time.perf_counter()-start,target_GT_open_attempts=0)
    atomic_json(RUN/'predictions_seal.json',result);atomic_json(PUBLIC/'predictions_seal.json',result)
    print(json.dumps(dict(event='T2V_ALL_VAL_PREDICTIONS_SEALED',methods=names)),flush=True)


if __name__=='__main__':run()
