"""v9 read-only sealed-input inventory and D3 exact-text reuse; no GT."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.presence.runtime import runtime_guard,target_metadata
from flex_uda.presence.predictions import verify_predictions
from scripts.active_run import PAIRS

PUBLIC=ROOT/'results/presence_active/redesign_v9_offline'
V8_PUBLIC=ROOT/'results/presence_active/redesign_v8'
V8=json.loads((V8_PUBLIC/'execution.json').read_text())
SOURCE=Path(V8['active_root'])
DATA=Path('/data2/user/TTA-RMOT-assets/presence_active_v9_offline_20261106')


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def rel_from(receipt):
    return {v:(x['sensitivity'],x['specificity']) for v,x in receipt['reliability'].items()}


def d3_and_inventory():
    attempts=runtime_guard();inputs=[];video_rows=[];query_rows=[];summaries=[]
    for target in ('Dance','KITTI-V2','SportsMOT'):
        meta=target_metadata(target)
        texts={v:[q for q in meta['episodes'] if q['video_id']==v] for v in sorted(meta['videos'])}
        sets={v:{q['query_text'] for q in qs} for v,qs in texts.items()}
        values=[]
        for video,queries in texts.items():
            other=set().union(*(s for v,s in sets.items() if v!=video))
            reused=[int(q['query_text'] in other) for q in queries]
            reuse=float(np.mean(reused));values.append(reuse)
            video_rows.append(dict(target=target,video=video,queries=len(queries),reuse_v=reuse,
                protocol='CHECKLIST' if reuse>=.8 else 'PRESET',rho_Beta='[2,2]' if reuse>=.8 else '[19,1]'))
            query_rows.extend(dict(target=target,video=video,query=q['expression_id'],
                text=q['query_text'],reuse_q=r) for q,r in zip(queries,reused))
        allq=[r['reuse_q'] for r in query_rows if r['target']==target]
        summaries.append(dict(target=target,videos=len(values),queries=len(allq),
            dataset_mean_reuse=float(np.mean(values)),query_weighted_reuse=float(np.mean(allq)),
            video_reuse_quantiles=np.quantile(values,[0,.1,.25,.5,.75,.9,1]).tolist(),
            E2_reuse_threshold=.8))
    for pair,(model,target) in PAIRS.items():
        path=SOURCE/pair/'MINUS_SLOW/rho_0.3';seal=verify_predictions(path)
        files=[]
        for directory in sorted((path/'videos').iterdir()):
            for name in ('questions.json','receipt.json','belief.npz','em_video_start.json'):
                p=directory/name;files.append(dict(path=str(p),sha256=file_sha(p),bytes=p.stat().st_size))
        arrays=Path(V8['stage_A_root'])/target/model/'presence_logits.npy'
        inputs.append(dict(pair=pair,model=model,target=target,source=str(path),episodes=seal['episode_count'],
            seal_path=str(path/'predictions_seal.json'),seal_sha256=file_sha(path/'predictions_seal.json'),
            presence_logits_path=str(arrays),presence_logits_sha256=file_sha(arrays),files=files))
    applicable=not any(r['target'] in ('KITTI-V2','SportsMOT') and r['dataset_mean_reuse']>=.5 for r in summaries)
    result=dict(status='SEALED_GT_FREE_D3_AND_INPUTS',inputs=inputs,D3=summaries,
        E2_status='APPLICABLE' if applicable else 'NOT_APPLICABLE',
        E2_reason='fixed .5 applicability example and .8 per-video protocol threshold; no threshold adjusted',
        GT_open_attempts=len(attempts),GPU_calls=0,teacher_calls=0,
        manifest_sha256=file_sha(PUBLIC/'manifest.json'),prereg_sha256=file_sha(PUBLIC/'PREREG.md'))
    assert not attempts
    write_csv(PUBLIC/'query_reuse.csv',query_rows);write_csv(PUBLIC/'video_reuse.csv',video_rows)
    atomic_json(PUBLIC/'inventory_D3.json',result)
    atomic_json(PUBLIC/'inputs_seal.json',dict(path=str(PUBLIC/'inventory_D3.json'),
        sha256=file_sha(PUBLIC/'inventory_D3.json'),GT_open_attempts=0))
    print(json.dumps({k:v for k,v in result.items() if k!='inputs'}),flush=True)
    return result


if __name__=='__main__':d3_and_inventory()
