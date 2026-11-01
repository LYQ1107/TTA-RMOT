"""GT-free Dance runtime for the remaining official LaMOT source model."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.multidomain_runtime import RUN,PUBLIC,guard,require_plan
from scripts.multidomain_sports_prepare import link_new,write_once
from flex_tta.dance_io import load_tracker_lines
from flex_tta.runtime.pf_receipts import file_sha,atomic_json


def main():
    require_plan();runtime=RUN/'runtime/dance';attempts=guard([runtime])
    source=ROOT/'results/runs/diag_attrib_20261031_v2/score_bank'
    bank=json.loads((source/'bank.json').read_text())
    for name in ('FROZEN','OFFICIAL_DANCE_COMPLETE'):
        if file_sha(source/(name+'.npy'))!=bank['cache_files_sha256'][name+'.npy']:
            raise ValueError('Changed sealed Dance official-model scores')
    data=Path('/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance')
    tracks=Path('/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs')
    link_new(runtime/'DanceTrack/training/image_02',data/'DanceTrack/training/image_02')
    videos={};hist=[]
    for v in sorted({q['video_id'] for q in bank['episodes']}):
        rows=load_tracker_lines(tracks,v);obs=[(t,i) for t,a in rows.items() for i in a]
        tails={i:t for t,i in obs};mask=[t!=tails[i] for t,i in obs]
        queries=[q for q in bank['episodes'] if q['video_id']==v]
        videos[v]=dict(observations=obs,lines=[rows[t][i] for t,i in obs],
            image_root=str(data/'DanceTrack/training/image_02'/v),full_observations=len(obs),
            physical_tracks=len(tails),tracker_path=str(tracks/(v+'.txt')),tracker_sha256=file_sha(tracks/(v+'.txt')))
        for q in queries:
            write_once(runtime/'expression'/v/(q['expression_id']+'.json'),dict(sentence=q['query_text']))
            hist.extend(mask)
    meta=dict(episodes=bank['episodes'],videos=videos,group_count=bank['query_frame_groups'],
        candidate_query_rows=3588445,historical_candidate_query_rows=int(sum(hist)),
        missing_tail_candidate_query_rows=int(len(hist)-sum(hist)),
        frame_rule='Original Dance image/MOT 1-based; boxes and IDs unchanged',GT_open_attempts=len(attempts))
    if len(hist)!=3588445 or sum(hist)!=210393*17 or meta['missing_tail_candidate_query_rows']!=692*17:
        raise ValueError('Dance full/historical coverage differs from sealed protocol')
    folder=RUN/'dance';folder.mkdir(exist_ok=True)
    write_once(folder/'metadata.json',meta);np.save(folder/'historical.npy',np.array(hist,dtype=bool))
    if attempts:raise PermissionError('Runtime preparation attempted target GT')
    print(json.dumps(dict(status='DANCE_RUNTIME_READY',videos=25,episodes=425,full_rows=len(hist),GT_open_attempts=0)),flush=True)


if __name__=='__main__':main()
