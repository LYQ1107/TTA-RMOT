"""Own-source labels -> frozen GT-candidate features (never target labels)."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--target', choices=['KITTI-V2', 'Dance', 'SportsMOT'], required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    import torch
    from flex_uda.presence.source import source_banks
    from flex_uda.presence.runtime import RUN, PUBLIC, IN_DOMAIN, load_model
    from scripts.presence_export_target import export_video
    inventory = json.loads((PUBLIC / 'inventory.json').read_text())
    if inventory['status'] != 'INVENTORY_COMPLETE':
        raise ValueError('Complete source/storage inventory before export')
    own_roots = {'KITTI-V2': ROOT / 'vendor/FlexHook/datasets/refer-kitti-v2',
                 'Dance': ROOT / 'vendor/FlexHook/datasets/refer-dance',
                 'SportsMOT': Path('/data2/user/LocateMOT_FullDataV2_spill/datasets/SportsMOT/train')}
    allowed = [own_roots[a.target].resolve()]
    if a.target in ('KITTI-V2', 'Dance'):
        from flex_uda.presence.source import splits
        dataset = 'kitti-2' if a.target == 'KITTI-V2' else 'dance'
        allowed.extend((own_roots[a.target] / 'expression' / v).resolve()
                       for v in splits()[dataset]['train'])
    if a.target == 'SportsMOT':
        from flex_uda.presence.source import SPORTS_ARCHIVE
        allowed.append(SPORTS_ARCHIVE.resolve())
    from scripts.multidomain_runtime import guard
    access = guard(allowed)
    banks = source_banks(a.target)
    name = IN_DOMAIN[a.target]
    output = RUN / 'features/source_train' / a.target / name
    output.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter(); device = torch.device('cuda:0')
    torch.manual_seed(0)
    torch.cuda.set_per_process_memory_fraction(min(.9, (16384 << 20) / torch.cuda.get_device_properties(0).total_memory))
    model, tokenizer, transform, loaded = load_model(a.target, name, output / 'model_log', device)
    selected = sorted(banks)[:1] if a.smoke else sorted(banks)
    meta = dict(videos=banks, episodes=[dict(q, video_id=v) for v, bank in banks.items() for q in bank['queries']])
    receipts = [export_video(model, tokenizer, transform, device, meta, v, a.target, name, output, None, a.smoke, source=True) for v in selected]
    result = dict(status='SOURCE_SMOKE_PASS' if a.smoke else 'SOURCE_FEATURES_COMPLETE', target=a.target,
        model=name, load=loaded, videos=receipts, target_GT_open_attempts=len(access),
        own_source_GT_permitted=True, GT_candidate_rows_not_predictions=True, physical_gpu=os.environ.get('PRESENCE_PHYSICAL_GPU'),
        wall_seconds=time.perf_counter()-begin, peak_allocated_bytes=torch.cuda.max_memory_allocated())
    atomic_json(output / ('smoke_receipt.json' if a.smoke else 'export_receipt.json'), result)
    print(json.dumps(dict(status=result['status'], target=a.target, seconds=result['wall_seconds'])), flush=True)


if __name__ == '__main__': main()
