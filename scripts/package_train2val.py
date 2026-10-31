#!/usr/bin/env python3
"""Compact stage records and large-artifact SHA index; never loads weights/GT."""
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.data.train2val import ASSETS
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.run_train2val_training import RUN,PUBLIC


def copy_immutable_metadata(source,destination):
    """Sealed inputs may be0444; reuse the identical published copy on rerun."""
    if destination.exists():
        if file_sha(source)!=file_sha(destination):raise ValueError('immutable metadata changed: '+str(destination))
        return
    shutil.copy2(source,destination)


def package():
    public=PUBLIC/'stages';public.mkdir(parents=True,exist_ok=True)
    records=[]
    for path in sorted(RUN.glob('*_completion.json')):
        shutil.copy2(path,public/path.name)
    for path in [ASSETS/'candidate_receipt.json',ASSETS/'pseudo/manifest_receipt.json',ASSETS/'pseudo/NAIVE/seal.json']:
        if path.exists():
            # The input image/score hash manifest is a small metadata artifact.
            copy_immutable_metadata(path,public/(path.parent.name+'_'+path.name))
    for path in sorted(RUN.rglob('*')):
        if not path.is_file() or any(part in ('evaluation','score_bank','shards') for part in path.parts):continue
        if path.suffix=='.pt' or path.name in ('metrics_train.jsonl','teacher_outputs.jsonl','teacher_fit_after_training.jsonl.gz'):
            records.append(dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=file_sha(path)))
    for path in [ASSETS/'pseudo/NAIVE'/n for n in ('runtime.jsonl','manifest.jsonl','windows.jsonl','soft_labels.jsonl','seal.json')]:
        if path.exists():records.append(dict(path=str(path),bytes=path.stat().st_size,sha256=file_sha(path)))
    teacher=RUN/'teacher_full/YES_FULL'
    for name in ('teacher_outputs.jsonl','receipt.json'):
        path=teacher/name
        if path.exists():records.append(dict(path=str(path),bytes=path.stat().st_size,sha256=file_sha(path)))
    atomic_json(PUBLIC/'server_artifacts.json',dict(large_files_not_in_git=True,root=str(RUN),target_assets=str(ASSETS),files=records))
    costs={};total_process_seconds=0.
    for phase in ('scores','teacher-smoke','teacher'):
        path=RUN/(phase+'_completion.json')
        if not path.exists():continue
        record=json.loads(path.read_text());seconds=sum(w['worker_seconds'] for w in record['workers'])
        costs[phase]=dict(actual_wall_seconds=record['actual_wall_seconds'],GPU_process_seconds=seconds,GPU_process_hours=seconds/3600,
            workers=record['workers']);total_process_seconds+=seconds
    for scope in ('smoke','full'):
        path=RUN/(scope+'_completion.json')
        if not path.exists():continue
        record=json.loads(path.read_text());rows={}
        for arm,r in record['arms'].items():
            attempts=r.get('attempts',[]);seconds=sum(a.get('seconds',0) for a in attempts)
            folder=Path(r.get('path',''))
            receipt=folder/'training_receipt.json'
            rows[arm]=dict(status=r['status'],GPU_process_seconds_including_attempts=seconds,
                GPU_process_hours_including_attempts=seconds/3600,attempts=attempts,
                training_receipt_sha256=file_sha(receipt) if receipt.is_file() else None)
            if receipt.is_file():
                t=json.loads(receipt.read_text());rows[arm].update(peak_allocated_bytes=t['peak_allocated_bytes'],
                    trainable_parameters=t['trainable_parameters'],optimizer_updates=t['steps'],
                    mean_window_exposures=t['mean_target_window_exposures'],
                    optimizer_updates_per_body_second=t['steps']/t['wall_seconds'],
                    target_valid_exposures=t['counters']['target_valid_exposures'],
                    nonzero_target_gradient_update_fraction=t['counters']['target_nonzero_gradient_steps']/t['steps'])
            total_process_seconds+=seconds
        costs['training_'+scope]=dict(actual_wall_seconds=record['actual_wall_seconds'],methods=rows)
    inf=RUN/'predictions_seal.json'
    if inf.exists():
        sealed=json.loads(inf.read_text());rows={}
        for name in sealed['methods']:
            cost=RUN/'evaluation'/name/'inference_costs.json';t=json.loads(cost.read_text())
            seconds=t['seconds'];total_process_seconds+=seconds
            metadata=json.loads((RUN/'evaluation'/name/'run_metadata.json').read_text())
            rows[name]=dict(GPU_process_seconds=seconds,GPU_process_hours=seconds/3600,**t,
                peak_allocated_bytes=metadata.get('peak_allocated_bytes'),optimizer_updates=0,episodes=425,
                physical_video_frames=sum(e['stats']['frames'] for e in metadata['episodes'])//17,
                query_frames=sum(e['stats']['frames'] for e in metadata['episodes']),
                physical_video_FPS=(sum(e['stats']['frames'] for e in metadata['episodes'])/17)/seconds,
                query_frame_FPS=sum(e['stats']['frames'] for e in metadata['episodes'])/seconds)
        costs['inference']=dict(actual_wall_seconds=sealed['wall_seconds'],methods=rows)
    official=RUN/'official_evaluation_completion.json'
    if official.exists():
        r=json.loads(official.read_text());costs['official_CPU']=dict(new_jobs=r['new_official_jobs'],reused_jobs=r['reused_exactly_equivalent_jobs'],
            sum_job_wall_seconds=sum(x['worker_wall_seconds'] for x in r['methods'].values()),
            not_measured_CPU_core_hours=True)
    costs.update(total_new_GPU_process_hours=total_process_seconds/3600,
        interpretation='sum actual admitted child-process elapsed times, including startup and CPU portions; not utilization-integrated GPU energy/time',
        count_controls_inherit_parent_training_and_inference=True,
        historical_references_reused_no_new_GPU=True,maximum_simultaneous_physical_GPUs=4)
    atomic_json(PUBLIC/'costs.json',costs)
    print(json.dumps(dict(event='T2V_ARTIFACTS_PACKAGED',large_artifacts=len(records),new_GPU_process_hours=total_process_seconds/3600)),flush=True)


if __name__=='__main__':package()
