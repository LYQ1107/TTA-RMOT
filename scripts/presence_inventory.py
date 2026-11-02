"""Own-source inventory, unit/storage estimates, and prior compute receipts."""
import json
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.presence.source import source_banks, counts_and_storage, SPORTS_ARCHIVE
from flex_uda.presence.runtime import RUN, PUBLIC
from flex_tta.runtime.pf_receipts import atomic_json, file_sha


def main():
    source, failures = [], []
    for target in ('KITTI-V2', 'Dance', 'SportsMOT'):
        try:
            item = counts_and_storage(source_banks(target), target); source.append(item)
            print(json.dumps({k:v for k,v in item.items() if k != 'per_video'}), flush=True)
        except (ValueError, FileNotFoundError) as exc:
            failures.append(dict(target=target, reason=str(exc)))
    target_rows = 3*(2425382+3588445+2968201)
    result = dict(status='INVENTORY_COMPLETE' if not failures else 'MISSING_OR_INCONSISTENT_SOURCE',
        source=source, failures=failures, target_feature_rows=target_rows,
        target_pcd_bytes=target_rows*768*2, total_estimated_bytes=target_rows*(1536+4+12)+sum(s['estimated_bytes'] for s in source),
        storage_root=str(RUN.resolve()), storage_free_bytes=shutil.disk_usage(RUN).free,
        source_sports_annotation_archive=dict(path=str(SPORTS_ARCHIVE), sha256=file_sha(SPORTS_ARCHIVE)),
        kitti_split_note='Vendor VIDEOS[kitti-2][train] has16 videos; historic rehearsal manifest had17 including0017. This head uses the unchanged vendor official train list, not the historical custom manifest.',
        optional_MOT17_VisDrone='Not added: not prepared in the existing layout; primary LaMOT source uses all SportsMOT Train annotations only.',
        sources_are_own_domain_only=True, target_GT_read_by_export=False,
        hyperparameters_changed=False, full_feature_export_started=False)
    atomic_json(PUBLIC / 'inventory.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'source'}, indent=2), flush=True)


if __name__ == '__main__': main()
