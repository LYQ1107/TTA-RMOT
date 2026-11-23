"""Resume unchanged registered studies with byte-identical local feature I/O.

The wrapper changes only the observation helper's process-local HISTORY
path and child-entry dispatch. Source files, recipes, seeds, prefix state
and evaluation gates remain those of the original registered modules.
"""
import argparse
import faulthandler
import importlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_observation_local_cache import CACHE, FILES

STUDIES = dict(primary='scripts.a100_conditional_six_setting',
    coarse='scripts.a100_conditional_coarse_baseline',
    generic='scripts.a100_conditional_shared_control',
    precision='scripts.a100_conditional_logit_precision')


def main():
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser(); parser.add_argument('--study', choices=tuple(STUDIES), default=os.environ.get('TTA_RECOVERY_STUDY', 'primary'))
    parser.add_argument('--pair'); parser.add_argument('--seed', type=int, default=0); args = parser.parse_args()
    receipt = json.loads((CACHE/'completion.json').read_text()) if (CACHE/'completion.json').is_file() else None
    if receipt is not None:
        assert receipt['status'] == 'PASS_BYTE_IDENTICAL_HISTORICAL_OBSERVATION_LOCAL_CACHE' and receipt['files'] == 592
    from flex_uda.active import observation_learning_inputs as inputs
    original = inputs.HISTORY
    inputs.HISTORY = CACHE
    original_class = inputs.ObservationVideo
    initial_completed = 0
    class WaitingObservationVideo(original_class):
        def __init__(self, pair, video):
            model, target = inputs.PAIRS[pair]
            folder = CACHE/'features'/target/model/'videos'/video
            paths = [folder/name for name in FILES]
            while not all(p.is_file() and p.with_name(p.name+'.copy_receipt.json').is_file() for p in paths):
                cache_state = json.loads((CACHE/'progress.json').read_text())
                if cache_state.get('status', '').startswith('FAILED_'):
                    raise RuntimeError('Resolve exact local feature preparation first')
                try:
                    os.kill(int(cache_state['pid']), 0)
                except ProcessLookupError:
                    raise RuntimeError('Local feature preparation stopped before this video was verified')
                index = sorted(inputs.metadata(target)['videos']).index(video)
                progress = module.OUT/pair/f'seed{args.seed}'/'progress.json'
                atomic_json(progress, dict(completed=max(initial_completed, index), total=len(inputs.metadata(target)['videos']),
                    video=video, stage='WAITING_VERIFIED_LOCAL_VIDEO_CACHE', original_checkpoint_prefix_preserved=initial_completed))
                time.sleep(10)
            for path in paths:
                proof = json.loads(path.with_name(path.name+'.copy_receipt.json').read_text())
                assert path.stat().st_size == proof['bytes'] and proof['source_read_and_independent_local_read_agree']
            super().__init__(pair, video)
    inputs.ObservationVideo = WaitingObservationVideo
    module = importlib.import_module(STUDIES[args.study])
    original_entry = module.__file__
    # The original controller dispatches Path(__file__) to child workers.
    # This process-local assignment keeps those children inside this same
    # explicit I/O wrapper without changing the original source on disk.
    module.__file__ = str(Path(__file__).resolve())
    os.environ['TTA_RECOVERY_STUDY'] = args.study
    print(json.dumps(dict(runtime='EXACT_LOCAL_FEATURE_IO_WRAPPER', study=args.study,
        pair=args.pair, seed=args.seed, original_history=str(original), cache=str(CACHE),
        aggregate_sha256=receipt['aggregate_sha256'] if receipt else None,
        per_video_verified_bytes_available_before_full_cache_completion=True, original_entry=original_entry,
        registered_numerical_files_changed=False, faulthandler_enabled=True)), flush=True)
    if args.pair:
        import fcntl
        root = module.OUT/args.pair/f'seed{args.seed}'; root.mkdir(parents=True, exist_ok=True)
        if (root/'stream.pt').is_file():
            import torch
            checkpoint = torch.load(root/'stream.pt', map_location='cpu', weights_only=False)
            initial_completed = int(checkpoint['next_video'])
            del checkpoint
        with (root/'condition.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX); module.run(args.pair, args.seed)
    else:
        try:
            module.main()
        except Exception as error:
            previous = json.loads(module.STATE.read_text()) if module.STATE.is_file() else {}
            previous.update(status='FAILED_CACHED_REGISTERED_RUNTIME_REQUIRES_ROOT_CAUSE',
                error=repr(error), original_study=args.study, goal_status='ACTIVE')
            atomic_json(module.STATE, previous); raise


if __name__ == '__main__': main()
