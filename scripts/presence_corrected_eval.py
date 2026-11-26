"""Post-selection CPU evaluation of the Sports source-label repair only."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_uda.presence.runtime import RUN,PUBLIC,IN_DOMAIN
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.presence_correct_sports_labels import HEAD_ROOT
from scripts.diag_attrib_v2_common import write_csv


def destination(target,variant):
    base = 'stage_A_label_v2' if variant == 'default' else 'stage_A_source_lr1e5_label_v2'
    return RUN / base / target / 'LaMOT-best'


def collect(selection,selection_sha):
    from scripts.pf_diag_common import official_metrics
    from scripts.presence_finish_a import references
    rows,costs,artifacts,selected_rows = [],[],[],[]
    for target,variant in [('SportsMOT','default'),('SportsMOT','source_lr1e5'),
                           ('KITTI-V2','default'),('Dance','default')]:
        for method in ('PRES_TOPK','PRES_TOPK_SMOOTH'):
            folder = destination(target,variant) / method
            row = json.loads((folder / 'offline_summary.json').read_text())
            rows.append(dict(row,source_label_version='EXPRESSION_ID_V2',source_variant=variant,
                own_domain=target == 'SportsMOT'))
            artifacts.append(dict(target=target,variant=variant,method=method,path=str(folder),
                prediction_seal_sha256=file_sha(folder / 'predictions_seal.json'),
                detailed_sha256=file_sha(folder / 'results/pedestrian_detailed.csv')))
    for variant in ('default','source_lr1e5'):
        path = HEAD_ROOT / variant / 'LaMOT-best/training_receipt.json'
        receipt = json.loads(path.read_text())
        costs.append(dict(model='LaMOT-best',source_domain='SportsMOT',variant=variant,
            learning_rate=receipt['learning_rate'],source_label_version='EXPRESSION_ID_V2',
            wall_seconds=receipt['wall_seconds'],GPU_hours=receipt['wall_seconds']/3600,
            epochs=receipt['epochs'],optimizer_updates=receipt['optimizer_updates'],
            peak_allocated_bytes=receipt['peak_allocated_bytes'],parameters=receipt['parameters'],
            path=str(path),sha256=file_sha(path)))
    for target,model in IN_DOMAIN.items():
        choice = selection['models'][model]
        if model == 'LaMOT-best': folder = destination(target,choice['selected']) / 'PRES_TOPK'
        else:
            base = 'stage_A' if choice['selected'] == 'default' else 'stage_A_source_lr1e5'
            folder = RUN / base / target / model / 'PRES_TOPK'
        row = json.loads((folder / 'offline_summary.json').read_text())
        seal = json.loads((folder / 'predictions_seal.json').read_text())
        if seal['head_checkpoint_sha256'] != choice['checkpoint_sha256']:
            raise ValueError('Corrected selection/checkpoint mismatch')
        native = official_metrics(dict(references(target,model))['NATIVE'] / 'results')['HOTA']
        selected_rows.append(dict(target=target,model=model,selected=choice['selected'],
            native_HOTA=native,HOTA=row['HOTA'],delta_HOTA=row['HOTA']-native,path=str(folder)))
    failures = [r['target'] for r in selected_rows if r['delta_HOTA'] < -1]
    gate = dict(status='FAIL_SELECTED_SOURCE_HEADS_STOP_BEFORE_BC' if failures else 'PASS_SELECTED_SOURCE_HEADS',
        failed_targets=failures,criterion='Each source-loss-selected own-source PRES_TOPK HOTA >= native HOTA -1',
        selection_sha256=selection_sha,source_label_correction_sha256=file_sha(PUBLIC / 'source_label_correction_v2.json'),
        source_labels_all_101_videos_verified=True,selected_heads=selection['models'],
        own_domain_metrics=selected_rows,published_commit=None,
        new_teacher_calls=0,target_LoRA_optimizer_updates=0)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    write_csv(PUBLIC / 'source_label_fix_v2.csv',[{k:r.get(k) for k in fields} for r in rows])
    write_csv(PUBLIC / 'source_label_fix_v2_costs.csv',costs)
    write_csv(PUBLIC / 'stage_A_selected_same_domain_v2.csv',selected_rows)
    atomic_json(PUBLIC / 'stage_A_selected_gate_v2.json',gate)
    atomic_json(PUBLIC / 'source_label_fix_v2_artifacts.json',dict(selection_sha256=selection_sha,files=artifacts))
    return gate


def main():
    parser = ArgumentParser(description=__doc__);parser.add_argument('--collect-only',action='store_true');args=parser.parse_args()
    path = PUBLIC / 'source_lr_trial_selection_v2.json'
    while not path.exists(): time.sleep(20)
    sha = file_sha(path);selection=json.loads(path.read_text())
    if not selection['corrected_sports_HOTA_not_computed_at_selection']:
        raise ValueError('Corrected source-only selection must precede new HOTA')
    if not args.collect_only:
        logs = RUN / 'source_label_fix_v2_eval' / str(time.time_ns());logs.mkdir(parents=True)
        env = dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
            PYTHONPATH=f'{ROOT}/compat:{ROOT}:/data2/user/TTA-RMOT-assets/flexhook/pydeps')
        def cell(job):
            target,variant=job
            while not (RUN / 'features' / target / 'LaMOT-best/export_receipt.json').exists():time.sleep(20)
            out=destination(target,variant);checkpoint=HEAD_ROOT / variant / 'LaMOT-best/checkpoint_best_source.pt'
            commands=[['scripts/presence_eval_offline.py','--target',target,'--model','LaMOT-best',
                '--phase','decode','--head-checkpoint',str(checkpoint),'--output-dir',str(out)]]
            for method in ('PRES_TOPK','PRES_TOPK_SMOOTH'):
                command=['scripts/active_eval.py','--target',target,'--folder',str(out / method)]
                if target == 'SportsMOT':command.append('--sports418')
                commands.append(command)
            with (logs / (target+'_'+variant+'.log')).open('a') as handle:
                for command in commands:
                    if subprocess.call([sys.executable,*command],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT):
                        raise RuntimeError('Corrected-source CPU evaluation failed: '+str(handle.name))
            print(json.dumps(dict(event='CORRECTED_SOURCE_HEAD_EVALUATED',target=target,variant=variant)),flush=True)
        jobs=[('SportsMOT','default'),('SportsMOT','source_lr1e5'),('KITTI-V2','default'),('Dance','default')]
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(cell,jobs))
        if file_sha(path) != sha:raise ValueError('Source selection changed during corrected evaluation')
        result=collect(selection,sha);atomic_json(logs / 'completion.json',result)
    else:result=collect(selection,sha)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':main()
