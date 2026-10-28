#!/usr/bin/env python3
"""Archive all completed predictions/receipts/logs, excluding GT and large caches."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.analyze_mllm_distillation import METHODS
from scripts.snapshot_mllm_stage import snapshot

RUN=ROOT/'results/runs/mllm_val_distill_20261028_v1'
EVAL=ROOT/'results/runs/mllm_val_evaluation_20261028_v1'
PUBLIC=ROOT/'results/mllm_distillation'


def original_receipt(pair):
    path=pair/'completion.json'
    if path.exists():return path
    reference=json.loads((pair/'reference_receipt.json').read_text())
    path=Path(reference['parent_receipt'])
    if file_sha(path)!=reference['parent_receipt_sha256']:raise ValueError('original reference receipt changed')
    return path


def archive(path,files):
    members=[]
    with path.open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as compressed,tarfile.open(fileobj=compressed,mode='w|') as tar:
        for source,name in sorted(files,key=lambda item:item[1]):
            # Materialize data only, never archive symlinks to a private tree.
            info=tar.gettarinfo(str(source.resolve()),arcname=name);info.mtime=0
            info.uid=info.gid=0;info.uname=info.gname=''
            with source.open('rb') as stream:tar.addfile(info,stream)
            members.append(dict(member=name,sha256=file_sha(source),bytes=source.stat().st_size))
    with tarfile.open(path,'r:gz') as tar:
        if len(tar.getnames())!=len(members):raise ValueError('archive member count differs')
        for member in members:
            if hashlib.sha256(tar.extractfile(member['member']).read()).hexdigest()!=member['sha256']:raise ValueError('archive data differs')
    return dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path),bytes=path.stat().st_size,members=members)


def package():
    summary=json.loads((PUBLIC/'full/summary.json').read_text())
    if summary['status']!='COMPLETE_25VAL_TRANSDUCTIVE_FIRST_ROUND':raise ValueError('complete full evaluation first')
    stage=snapshot();logs=PUBLIC/'logs';archives={};candidate_index=[]
    for method in METHODS:
        folder=EVAL/method;audit=json.loads((folder/'full_audit.json').read_text());files=[]
        if audit['valid_episodes']!=425 or audit['unique_videos']!=25:raise ValueError('all425 required')
        for episode in audit['episodes']:
            relative=Path(episode['video'])/episode['expression'];pair=folder/'results'/relative
            for name in ('predict.txt','completion.json'):
                source=original_receipt(pair) if name=='completion.json' else pair/name
                files.append((source,str(relative/name)))
            margin=pair/'candidate_margins.jsonl.gz'
            candidate_index.append(dict(method=method,video=episode['video'],expression=episode['expression'],
                server_path=str(margin.resolve()),sha256=file_sha(margin),bytes=margin.stat().st_size))
        archives[method]=archive(logs/(method+'_all425_predictions_and_receipts.tar.gz'),files)
        if len(archives[method]['members'])!=850:raise ValueError('missing prediction/receipt')
    # These are completed parent/worker/TrackEval logs and provenance, not per-query GT files.
    files=[]
    for path in EVAL.rglob('*'):
        if not path.is_file() or path.is_symlink():continue
        if path.name in ('gt.txt','predict.txt','completion.json','candidate_margins.jsonl.gz'):continue
        if path.suffix in ('.txt','.log') or path.name in ('run_metadata.json','gpu_admission.json','launcher_status.json',
            'whole_video_shards.json','first_launch_epoch.json','official_evaluation_completion.json','controls_sealed.json','reference_metadata.json'):
            files.append((path,str(Path('evaluation')/path.relative_to(EVAL))))
    parent=RUN/'full_evaluation_launcher_stdout.txt'
    if parent.exists():files.append((parent,'training_parent/full_evaluation_launcher_stdout.txt'))
    archives['complete_evaluation_text_logs']=archive(logs/'complete_evaluation_text_logs.tar.gz',files)
    for name in ('whole_video_shards.json','official_evaluation_completion.json','controls_sealed.json'):
        shutil.copy2(EVAL/name,PUBLIC/'full'/name)
    atomic_json(logs/'full_evaluation_archive_index.json',dict(archives=archives,
        no_GT_images_or_weights=True,methods=len(METHODS),episodes_per_method=425,
        large_score_banks='retained on server; every candidate-margin file SHA/path indexed separately'))
    atomic_json(PUBLIC/'full/candidate_score_bank_index.json',candidate_index)
    train=RUN/'MLLM_DISTILL';checkpoints={}
    for name in ('checkpoint_best_source_selected.pt','checkpoint_latest.pt'):
        path=train/name;checkpoints[name]=dict(path=str(path),sha256=file_sha(path),bytes=path.stat().st_size)
    recovery=dict(project='LYQ1107/TTA-RMOT',root=str(ROOT),branch='research/v2dance-mllm-naive-distill-20261028',
        protocol='25Val same-set unlabeled offline diagnosis, NOT40Train-to25Val',
        checkpoints=checkpoints,pseudo_root='/data2/user/TTA-RMOT-assets/mllm-distill/val_pseudo_v1',
        immutable_pseudo_seal_sha256=file_sha(Path('/data2/user/TTA-RMOT-assets/mllm-distill/val_pseudo_v1/seal.json')),
        training_root=str(train),evaluation_root=str(EVAL),python='/home/szgx/anaconda3/envs/dfot/bin/python',
        resume_training_command='python scripts/train_mllm_distill.py --scope full --resume --diagnostic-val-authorized --output '+str(train),
        resume_evaluation_command='python scripts/launch_mllm_evaluation.py --checkpoint '+str(train/'checkpoint_best_source_selected.pt')+' --output-root '+str(EVAL),
        warning='Both runs are already complete. Resume preserves completed work; do not rerun or extend the fixed8973 budget.',
        incomplete_original_formal_assets='40Train matched official DETR+NeuralSORT prediction files are absent; no GT/alternate detector substitute',
        current_gpu_jobs='all current-stage training and four evaluation workers exited0; no new GPU job started')
    atomic_json(PUBLIC/'recovery_index.json',recovery)
    stage.update(status='COMPLETE_25VAL_FIRST_ROUND_AND_FAILURE_ATTRIBUTION',full425_evaluation_complete=True,
        original_formal40Train_to25Val_complete=False,new_full_experiment_started=False,
        scientific_verdict=summary['scientific_verdict'],delivery_index='logs/full_evaluation_archive_index.json')
    atomic_json(PUBLIC/'stage_status.json',stage)
    print(json.dumps(dict(status=stage['status'],prediction_archives=len(METHODS),episodes_each=425,
        evaluation_text_log_members=len(archives['complete_evaluation_text_logs']['members']),
        raw_candidate_score_files_indexed=len(candidate_index)),indent=2))


if __name__=='__main__':package()
