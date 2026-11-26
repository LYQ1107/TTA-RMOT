"""Reproduce reverse CountMatched intervals, including published-only inputs."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_repeats_analyze import paired
from scripts.a100_visual_classical_prepare import OUT


def run(directory):
    results={}
    for name,stem in [('raw','three_seed_task_valid'),('bounded','three_seed_clipped')]:
        report=json.loads((directory/(stem+'_results.json')).read_text())
        sufficient=json.loads((directory/(stem+'_sufficient_statistics.json')).read_text());results[name]={}
        for pair,record in report.items():
            def item(seed,key):
                row=sufficient[pair][str(seed)][key]
                return dict(parent_videos=row['parents'],parent_stats=(np.asarray(row['numerator']),np.asarray(row['denominator'])),
                            HOTA=record['per_seed_HOTA'][str(seed)][key])
            contrasts={}
            for count in ('SOFT_KD','HARD_KD','SOFT_KD_400','HARD_KD_400'):
                rank='counts/COHORT__AT_'+count+'_COUNT';reference='counts/'+count+'__AT_'+count+'_COUNT'
                # Raw C4's original count family contains only final soft/hard
                # counts; the bounded study additionally registered early ones.
                if not all(rank in sufficient[pair][str(s)] and reference in sufficient[pair][str(s)] for s in (0,17,42)):
                    continue
                contrasts[count]=paired([item(s,rank) for s in (0,17,42)],[item(s,reference) for s in (0,17,42)])
            results[name][pair]=contrasts
    atomic_json(directory/'three_seed_reverse_counts.json',results)
    print(json.dumps(dict(status='REVERSE_COUNTS_REPRODUCED',directory=str(directory),
        soft_count_mean_effects={name:{p:r['SOFT_KD']['mean_delta_HOTA'] for p,r in pairs.items()} for name,pairs in results.items()})))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--published',action='store_true');a=p.parse_args()
    run(ROOT/'docs/a100_visual_classical_20261121' if a.published else OUT/'analysis')
