#!/usr/bin/env python3
"""Independent Train-only data-preparation process, before model workers."""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.data.train2val import ASSETS, inventory, prepare
from flex_tta.runtime.pf_receipts import atomic_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('p0', 'candidates'))
    args = parser.parse_args()
    public = ROOT / 'results/train2val'
    if args.phase == 'p0':
        result = inventory()
        atomic_json(public / 'p0_inventory.json', result)
        with (public / 'p0_videos.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result['videos'][0]))
            writer.writeheader()
            writer.writerows(result['videos'])
    else:
        result = prepare(json.loads((public / 'p0_inventory.json').read_text()))
        atomic_json(public / 'candidate_receipt.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'videos'}, indent=2))
    if result['status'].startswith('STOP_'):
        raise SystemExit(1)
