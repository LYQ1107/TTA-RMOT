"""v8P controls match each sealed MINUS actual per-video budget."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.presence.runtime import RUN, PUBLIC, target_metadata, sealed_margins, runtime_guard
from flex_uda.active.baselines import yestrack_video, uniform_video
from scripts.active_run import PAIRS, video_spec
from flex_tta.runtime.pf_receipts import atomic_json, file_sha


def run(a):
    from flex_uda.active.execution import authorize_active
    execution,publication=authorize_active()
    assert execution.manifest['version']=='v8P'
    assert dict(pair=a.pair,method=a.method,rho=a.rho) in execution.manifest['allowed_registered_jobs']
    if execution.manifest and execution.manifest.get('pilot_only'):
        raise PermissionError('No new complete baseline is authorized by the small-pilot manifest')
    access = runtime_guard(); begin = time.perf_counter()
    from flex_uda.active.teacher_client import TeacherClient
    from flex_uda.presence.predictions import seal_mask, verify_predictions
    model, target = PAIRS[a.pair]; meta = target_metadata(target)
    margins, reference = sealed_margins(target, model)
    full = execution.active / a.pair / 'MINUS_SLOW' / ('rho_'+str(a.rho))
    dependency=verify_predictions(full)
    out = execution.active / a.pair / a.method / ('original_delta10' if a.method == 'YESTRACK_FULL_D10' else 'rho_'+str(a.rho))
    if (out / 'predictions_seal.json').exists():
        print(json.dumps(dict(status='ALREADY_SEALED_NO_RERUN', path=str(out))), flush=True); return
    out.mkdir(parents=True, exist_ok=True)
    runtime = out / 'teacher_runtime.json'
    atomic_json(runtime, [video_spec(target, video, bank) for video, bank in meta['videos'].items()])
    teacher = TeacherClient(runtime, RUN / 'teacher/raw.sqlite', out / 'teacher_server.log', ROOT)
    mask = margins > 0; receipts = []
    try:
        for video in sorted(meta['videos']):
            bank = meta['videos'][video]
            queries = [q for q in meta['episodes'] if q['video_id'] == video]
            directory = out / 'videos' / video; directory.mkdir(parents=True, exist_ok=True)
            if (directory / 'receipt.json').exists():
                receipt = json.loads((directory / 'receipt.json').read_text())
                if file_sha(directory / 'mask.npy') != receipt['mask_sha256']: raise ValueError('Changed sealed baseline video')
                local = np.load(directory / 'mask.npy')
            else:
                budget = json.loads((full / 'videos' / video / 'receipt.json').read_text())['questions'] if full else None
                local_margins = np.stack([margins[q['start']:q['stop']] for q in queries])
                if a.method == 'UNIFORM_K':
                    from scripts.active_run import video_prior
                    prior = video_prior(meta, bank, video, model, target, margins)
                    assert budget is not None  # use the sealed MINUS actual calls, not a nominal floor
                    local, receipt = uniform_video(target, video, queries, bank, teacher, budget,
                        frame_presence=prior['frame_presence'], frame_index=prior['frame_index'])
                else:
                    local, receipt = yestrack_video(target, video, queries, bank, local_margins, teacher, a.method, full_calls=budget)
                assert receipt['actual_calls']==budget
                receipt['budget_match_exact']=True
                np.save(directory / 'mask.npy', local)
                receipt['mask_sha256'] = file_sha(directory / 'mask.npy')
                atomic_json(directory / 'receipt.json', receipt)
            for qi, q in enumerate(queries): mask[q['start']:q['stop']] = local[qi]
            receipts.append(receipt)
            print(json.dumps({k:v for k,v in receipt.items() if k != 'question_keys'}), flush=True)
        costs = teacher.close(); teacher = None
        sessions = [json.loads(path.read_text()) for path in (out / 'teacher_sessions').glob('*.json')]
        costs = dict(costs, logical_calls=sum(row['actual_calls'] for row in receipts),
            attempted_logical_calls=sum(row['logical_calls'] for row in sessions),
            new_calls=sum(row['new_calls'] for row in sessions), cache_hits=sum(row['cache_hits'] for row in sessions),
            forward_seconds=sum(row['forward_seconds'] for row in sessions),
            equivalent_forward_seconds=sum(row['equivalent_forward_seconds'] for row in sessions),
            peak_allocated_bytes=max(row['peak_allocated_bytes'] for row in sessions),
            prior_session_resume_included=True,
            interrupted_sessions_accounting=[row['interruption_accounting'] for row in sessions if 'interruption_accounting' in row])
        seal_mask(out, meta, mask, target=target, model=model, method=a.method, GT_open_attempts=len(access),
            extra=dict(version='v8P',pair=a.pair,rho=a.rho,teacher=costs,
                budget_source=str(full),budget_source_seal_sha256=file_sha(full/'predictions_seal.json'),
                coverage=receipts, margin_input=reference, LoRA_GPU_seconds=0.,rescore_GPU_seconds=0.,
                video_transductive=a.method != 'YESTRACK_FULL_D10',
                strict_frame_causal=a.method == 'YESTRACK_FULL_D10',
                native_fallback_for_uncovered=a.method.startswith('YESTRACK'), wall_seconds=time.perf_counter()-begin,
                stage_A_publication=publication,
                execution_manifest_sha256=file_sha(execution.manifest_path) if execution.manifest else None))
    finally:
        if teacher is not None and teacher.child.poll() is None: teacher.close()


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--pair', choices=list(PAIRS), required=True)
    p.add_argument('--method', choices=['YESTRACK_BUDGET_PRIOR', 'UNIFORM_K'], required=True)
    p.add_argument('--rho', choices=[.1, .3, 1.], type=float, default=.3)
    a = p.parse_args()
    from flex_uda.active.receipts import record_attempt
    from flex_uda.active.execution import active_paths
    folder = active_paths().active / a.pair / a.method / ('original_delta10' if a.method == 'YESTRACK_FULL_D10' else 'rho_'+str(a.rho))
    with record_attempt(folder, phase='full_baseline'): run(a)


if __name__ == '__main__': main()
