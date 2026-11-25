"""Preserve and relocate only the ENOSPC-interrupted v7 working copy."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha

OLD=Path('/data1/user/TTA-RMOT-assets/presence_active_20261102/redesign_v7')
NEW=Path('/data2/user/TTA-RMOT-assets/v7r_HtKzDo')
PUBLIC=ROOT/'results/presence_active/redesign_v7'


def verify_copy(source,destination):
    paths=lambda root:{str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    names=paths(source)
    if names!=paths(destination):raise ValueError('Storage copy file list mismatch')
    result=[]
    for name in sorted(names):
        a=source/name;b=destination/name
        if a.is_symlink():
            if not b.is_symlink() or os.readlink(a)!=os.readlink(b):raise ValueError('Storage copy symlink mismatch')
            result.append(dict(relative_path=name,symlink=os.readlink(a)))
        else:
            digest=file_sha(a)
            if a.stat().st_size!=b.stat().st_size or digest!=file_sha(b):raise ValueError('Storage copy changed '+name)
            result.append(dict(relative_path=name,bytes=a.stat().st_size,sha256=digest))
    return result


def recover():
    import torch
    record_path=PUBLIC/'storage_recovery.json'
    if record_path.exists():raise ValueError('Storage recovery already recorded; do not switch it twice')
    for pid in (3107662,3107682,3107715,3516077):
        try:os.kill(pid,0)
        except ProcessLookupError:continue
        raise RuntimeError('Original registered process still exists: '+str(pid))
    if shutil.disk_usage(NEW).free<12*(1<<30):raise RuntimeError('Insufficient data2 recovery headroom')
    mappings=[(OLD/'active/C4/SLOW_REPLAY',NEW/'C4'),(OLD/'launchers',NEW/'L')]
    manifests=[]
    for source,destination in mappings:
        manifests.append(dict(source=str(source),destination=str(destination),files=verify_copy(source,destination)))
    atomic_json(PUBLIC/'storage_copy_manifest.json',dict(status='BYTE_EXACT_COPY_VERIFIED',copies=manifests))
    out=NEW/'C4/rho_0.3';checkpoint=sorted(out.glob('stream_*.pt'))[-1]
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['version']=='v7' and saved['next_video']==36 and saved['slow_updates']==7200
    receipts=sorted(out.glob('videos/*/receipt.json'));assert len(receipts)==36
    protected=[dict(relative_path=str(p.relative_to(out)),sha256=file_sha(p)) for p in
        sorted(list(out.glob('stream_*.pt'))+receipts)]
    unfinished=[p for p in (out/'videos').iterdir() if not (p/'receipt.json').exists()]
    assert len(unfinished)==1
    pending=unfinished[0];retained=out/'failed_video_attempts/enospc_20261106'/pending.name
    retained.parent.mkdir(parents=True);pending.rename(retained)
    backups=[]
    for source,destination in mappings:
        backup=source.with_name(source.name+'_enospc_original_20261106')
        if backup.exists() or source.is_symlink():raise ValueError('Original relocation path is not untouched')
        source.rename(backup)
        try:source.symlink_to(destination,target_is_directory=True)
        except BaseException:
            backup.rename(source);raise
        backups.append(dict(logical_path=str(source),retained_original=str(backup),new_target=str(destination)))
    attempts=out/'worker_attempts';original=attempts/'1793900412603928151.json'
    raw=original.read_text();preserved=attempts/'original_running'/original.name
    preserved.parent.mkdir();original.rename(preserved)
    # The empty launcher temp file was created after wait(), but the actual
    # child exit code and perf_counter ledger could not be persisted on ENOSPC.
    terminal=NEW/'L/workers_1793863676143511531/C4_SLOW_REPLAY_completion.json.tmp.3107715'
    elapsed=(terminal.stat().st_mtime_ns-int(original.stem))/1e9
    assert elapsed>0
    row=json.loads(raw);row.update(status='FAILED_RETAINED_FOR_RECOVERY',
        error='ENOSPC on data1; exact child exit code and final GPU phase time not persisted',
        wall_seconds=elapsed,wall_seconds_estimated=True,
        wall_estimate_rule='empty launcher terminal temp mtime minus worker receipt filename time_ns',
        original_running_receipt=str(preserved),original_running_sha256=file_sha(preserved),
        incomplete_failed_rescore_GPU_phases=1,parent_peak_allocated_bytes=None,
        all_task_pids_confirmed_absent=True)
    atomic_json(original,row)
    failed_logs=[NEW/'L/workers_1793863676143511531/C4_SLOW_REPLAY_completion.json',
        NEW/'L/coordinator_1793863672677211187/completion.json',
        NEW/'L/durable_1793863666149837309/completion.json']
    for path in failed_logs:
        atomic_json(path,dict(status='ENOSPC_INTERRUPTED_RETAINED_FOR_RECOVERY',exit_code=None,
            exact_exit_code_unknown=True,observer_exit_code=1,all_task_pids_confirmed_absent=True,
            observation_time_ns=time.time_ns(),failure_receipt=str(original)))
    result=dict(status='ENOSPC_STORAGE_RECOVERY_VERIFIED_NOT_YET_RELAUNCHED',
        original_durable_dir=str(OLD/'launchers/durable_1793863666149837309'),
        original_coordinator=str(OLD/'launchers/coordinator_1793863672677211187'),
        old_observer_exit_code=1,exact_manager_and_child_exit_codes_unknown=True,
        source_volume='/data1',source_available_bytes=0,recovery_volume='/data2',
        free_bytes_after_copy=shutil.disk_usage(NEW).free,storage_aliases=backups,
        copy_manifest_sha256=file_sha(PUBLIC/'storage_copy_manifest.json'),
        checkpoint_path=str(checkpoint),checkpoint_sha256=file_sha(checkpoint),
        next_video=36,completed_videos_preserved=36,slow_updates_restored=7200,
        protected_completed_files=protected,failed_video_retained=str(retained),
        failed_worker_attempt=str(original),failed_worker_wall_seconds_estimate=elapsed,
        incomplete_failed_scoring_GPU_time_unknown=True,previous_complete_methods_preserved=11,
        prereg_and_execution_manifest_unchanged=True,numerical_model_or_teacher_code_changed=False,
        no_foreign_process_stopped=True,original_data_or_checkpoints_deleted=False,
        original_registered_plan=str(PUBLIC/'registered_jobs.json'))
    atomic_json(record_path,result)
    print(json.dumps({k:v for k,v in result.items() if k!='protected_completed_files'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mark-launched',type=Path);p.add_argument('--coordinator',type=Path)
    a=p.parse_args()
    if a.mark_launched:
        record=PUBLIC/'storage_recovery.json';value=json.loads(record.read_text())
        value.update(status='ENOSPC_RESUMED_SAME_REGISTERED_PLAN',resumed_durable_dir=str(a.mark_launched))
        if a.coordinator:value['resumed_coordinator']=str(a.coordinator)
        atomic_json(record,value)
    else:recover()
