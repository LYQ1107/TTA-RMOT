"""Repair source expression identity without overwriting frozen cache files."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_uda.presence.runtime import RUN,PUBLIC
from flex_tta.runtime.pf_receipts import atomic_json,file_sha

LABEL_ROOT = RUN / 'source_labels_by_expression_v2/SportsMOT/LaMOT-best'
HEAD_ROOT = RUN / 'heads_source_labels_v2'


def prepare():
    from scripts.presence_source_label_audit import expression_labels
    source = RUN / 'features/source_train/SportsMOT/LaMOT-best'
    manifest = LABEL_ROOT / 'labels_manifest.json'
    if manifest.exists(): return json.loads(manifest.read_text())
    records = []
    for directory in sorted((source / 'videos').iterdir()):
        video = directory.name
        metadata = json.loads((directory / 'metadata.json').read_text())
        observations = [tuple(map(int,o)) for o in np.load(directory / 'observations.npy')]
        raw = expression_labels('SportsMOT',video)
        labels = np.asarray([o in raw[q['expression_id']] for q in metadata['queries'] for o in observations],dtype=bool)
        old = np.load(directory / 'labels.npy')
        out = LABEL_ROOT / 'videos' / video; out.mkdir(parents=True,exist_ok=True)
        np.save(out / 'labels.npy',labels)
        records.append(dict(video=video,path=str(out / 'labels.npy'),sha256=file_sha(out / 'labels.npy'),
            old_path=str(directory / 'labels.npy'),old_sha256=file_sha(directory / 'labels.npy'),
            feature_seal_sha256=file_sha(directory / 'seal.json'),rows=len(labels),
            mismatching_observations=int((old != labels).sum())))
    result = dict(status='SOURCE_EXPRESSION_ID_LABEL_CORRECTION_SEALED',source_domain='SportsMOT',model='LaMOT-best',
        files=records,changed_rows=sum(r['mismatching_observations'] for r in records),
        source_feature_export_sha256=file_sha(source / 'export_receipt.json'),
        old_labels_and_frozen_features_not_modified=True,GT_scope='official source Train literal expression targets',
        hyperparameters_changed=False,new_GPU_or_teacher_calls=0)
    atomic_json(manifest,result)
    atomic_json(PUBLIC / 'source_label_correction_v2.json',dict(result,labels_root=str(LABEL_ROOT),manifest_sha256=file_sha(manifest)))
    return result


def train():
    from scripts.select_gpu import query
    registered = {g['uuid'] for g in json.loads((RUN / 'resources/current.json').read_text())['selected']}
    logs = RUN / 'source_label_fix_v2' / str(time.time_ns()); logs.mkdir(parents=True)
    def worker(job):
        variant,rate = job
        destination = HEAD_ROOT / variant / 'LaMOT-best'
        while True:
            available = [g for g in query() if g['uuid'] in registered and g['free_mib'] >= 8192]
            if available:
                gpu = min(available,key=lambda g:(g['compute_processes'],-g['free_mib'],g['index'])); break
            time.sleep(20)
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],PRESENCE_PHYSICAL_GPU=str(gpu['index']),
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
            PYTHONPATH=f'{ROOT}/compat:{ROOT}:/data2/user/TTA-RMOT-assets/flexhook/pydeps')
        command = [sys.executable,str(ROOT / 'scripts/presence_train.py'),'--target','SportsMOT',
            '--learning-rate',str(rate),'--source-label-root',str(LABEL_ROOT),'--output',str(destination)]
        log = logs / (variant+'.log')
        with log.open('a') as handle:
            child = subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            print(json.dumps(dict(event='CORRECTED_SOURCE_HEAD_STARTED',variant=variant,physical_gpu=gpu['index'],
                pid=child.pid,log=str(log))),flush=True)
            for line in child.stdout:
                handle.write(line);handle.flush()
                if line.startswith('{'): print(line,end='',flush=True)
            if child.wait(): raise RuntimeError('Corrected source head failed: '+str(log))
    with ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(worker,[('default',3e-4),('source_lr1e5',1e-5)]))
    previous = PUBLIC / 'source_lr_trial_selection.json'
    selection = json.loads(previous.read_text())
    candidates = []
    for variant in ('default','source_lr1e5'):
        path = HEAD_ROOT / variant / 'LaMOT-best/training_receipt.json'
        receipt = json.loads(path.read_text())
        candidates.append(dict(variant=variant,loss=receipt['best_source_loss'],path=str(path),sha256=file_sha(path),receipt=receipt))
    chosen = min(candidates,key=lambda c:(c['loss'],c['variant'] != 'default')); receipt = chosen['receipt']
    selection['models']['LaMOT-best'] = dict(source_domain='SportsMOT',selected=chosen['variant'],
        learning_rate=receipt['learning_rate'],checkpoint=receipt['checkpoint'],checkpoint_sha256=receipt['checkpoint_sha256'],
        best_source_loss=chosen['loss'],corrected_source_labels=receipt['corrected_source_labels'],
        candidates=[{k:v for k,v in c.items() if k != 'receipt'} for c in candidates])
    selection.pop('trial_HOTA_not_computed_at_selection',None)
    selection.update(previous_selection_sha256=file_sha(previous),
        corrected_sports_HOTA_not_computed_at_selection=True,
        unaffected_V2_Dance_source_choices_unchanged=True,bug_fix_not_new_hyperparameter_search=True)
    atomic_json(PUBLIC / 'source_lr_trial_selection_v2.json',selection)
    atomic_json(logs / 'completion.json',dict(status='CORRECTED_SPORTS_HEADS_TRAINED_SOURCE_LOSS_SELECTION_SEALED'))
    print(json.dumps(dict(status='CORRECTED_SOURCE_LOSS_SELECTION_SEALED_BEFORE_NEW_HOTA',selected=chosen['variant'])),flush=True)


def main():
    parser = ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['prepare','train'],required=True)
    args = parser.parse_args()
    if args.phase == 'prepare':
        result = prepare();print(json.dumps({k:v for k,v in result.items() if k != 'files'}),flush=True)
    else: train()


if __name__ == '__main__': main()
