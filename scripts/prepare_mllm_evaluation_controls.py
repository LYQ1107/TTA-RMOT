#!/usr/bin/env python3
"""GT-free baseline copies and two directional count-matched diagnostics."""
from collections import Counter
import argparse
import gzip
import json
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.training import protect_target_annotations
from flex_tta.runtime.pf_receipts import EpisodeStore,atomic_json,file_sha
from scripts.pf_diag_common import TRACKS,FROZEN
from flex_tta.dance_io import load_tracker_lines
from scripts.diag_pf_countmatched import topk_ids


def stage_reference(source,destination,name):
    source,destination=Path(source),Path(destination);audit=json.loads((source/'full_audit.json').read_text())
    if audit['status']!='PASS' or audit['valid_episodes']!=425:raise ValueError('reference needs full425 predictions')
    folder=destination/name;out=folder/'results';out.mkdir(parents=True,exist_ok=True)
    for e in audit['episodes']:
        video=e['video'];expression=e.get('expression',e.get('expression_id'));old=source/'results'/video/expression;new=out/video/expression
        receipt=json.loads((old/'completion.json').read_text())
        if file_sha(old/'predict.txt')!=receipt['prediction_sha256'] or file_sha(old/'candidate_margins.jsonl.gz')!=receipt['margins_sha256']:
            raise ValueError('historical reference is no longer sealed')
        new.mkdir(parents=True,exist_ok=True)
        if not (new/'predict.txt').exists():shutil.copy2(old/'predict.txt',new/'predict.txt')
        if file_sha(new/'predict.txt')!=receipt['prediction_sha256']:raise ValueError('different existing reference copy')
        margin=new/'candidate_margins.jsonl.gz'
        if not margin.exists():margin.symlink_to((old/'candidate_margins.jsonl.gz').resolve())
        if file_sha(margin)!=receipt['margins_sha256']:raise ValueError('different reference score stream')
        atomic_json(new/'reference_receipt.json',dict(parent_receipt=str(old/'completion.json'),parent_receipt_sha256=file_sha(old/'completion.json'),
            prediction_sha256=receipt['prediction_sha256'],margins_sha256=receipt['margins_sha256']))
    atomic_json(folder/'full_audit.json',dict(audit,method=name,reused_reference=str(source)))
    atomic_json(folder/'reference_metadata.json',dict(status='SEALED_REUSED_COMPLETE_REFERENCE',original=str(source),
        valid_episodes=425,current_GPU_calls=0,total_cost_not_zero=True,original_predictions_unchanged=True))


def count_matched(root,name,score_method,count_method):
    root=Path(root);audit=json.loads((root/count_method/'full_audit.json').read_text());started=time.perf_counter();episodes=[];receipts=[];banks={}
    for entry in audit['episodes']:
        video,expression=entry['video'],entry.get('expression',entry.get('expression_id'));relative=Path('results')/video/expression
        if video not in banks:banks[video]=load_tracker_lines(TRACKS,video)
        tracks=banks[video];score_path=root/score_method/relative/'candidate_margins.jsonl.gz'
        counts=Counter(int(float(r.split(',')[0])) for r in (root/count_method/relative/'predict.txt').read_text().splitlines())
        store=EpisodeStore(root/name,video,expression,dict(protocol='POST_HOC_DIAGNOSTIC_NOT_DEPLOYABLE',
            score_source=score_method,per_frame_count_source=count_method,rule='fixed top-k, score descending, numeric ID tie-break, no GT'),resume=True)
        receipt=store.completed()
        if receipt is None:
            pp,mp=store.begin();mp.symlink_to(score_path.resolve());seen=[];rows=selected=0
            with pp.open('w') as dst,gzip.open(score_path,'rt') as stream:
                for line in stream:
                    item=json.loads(line);t=item['frame'];ids=item['ids'];values=item['adapted']
                    if ids!=sorted(tracks[t]):raise ValueError('count-match candidate universe differs')
                    k=counts[t]
                    if k>len(ids):raise ValueError('count exceeds immutable candidate bank')
                    chosen=topk_ids(ids,dict(zip(ids,values)),k)
                    for i in sorted(chosen):dst.write(tracks[t][i]+'\n')
                    seen.append(t);rows+=len(ids);selected+=len(chosen)
            if seen!=sorted(tracks):raise ValueError('count-match frame coverage differs')
            receipt=store.seal(pp,mp,dict(full_frame_coverage=True,wall_seconds=0,
                stats=dict(frames=len(seen),candidate_rows=rows,selected_rows=selected,optimizer_steps=0)))
        receipts.append(receipt);lines=store.prediction.read_text().splitlines()
        episodes.append(dict(video=video,expression=expression,frames=receipt['stats']['frames'],candidates=receipt['stats']['candidate_rows'],
            selected_rows=len(lines),selected_ids=len({r.split(',')[1] for r in lines})))
    if len(episodes)!=425:raise ValueError('all425 count-match episodes required')
    atomic_json(root/name/'full_audit.json',dict(status='PASS',method=name,valid_episodes=425,unique_videos=25,
        missing_episodes=0,duplicated_episodes=0,target_gt_opened=False,tracker_geometry_changed=False,tracker_ids_changed=False,
        episodes=episodes,**{k:sum(e[k] for e in episodes) for k in ('frames','candidates','selected_rows','selected_ids')}))
    atomic_json(root/name/'run_metadata.json',dict(status='SEALED',protocol='POST_HOC_DIAGNOSTIC_NOT_DEPLOYABLE',
        episodes=receipts,count_source=count_method,score_source=score_method,GT_reads=0,GPU_calls=0,
        incremental_CPU_wall_seconds=time.perf_counter()-started,total_cost_includes_parent_models=True))
    print(json.dumps(dict(event='count_matched_sealed',method=name,episodes=425)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();root=a.output_root.resolve()
    sealed=json.loads((root/'MLLM_DISTILL/full_audit.json').read_text())
    if sealed['status']!='PASS' or sealed['valid_episodes']!=425:raise ValueError('seal complete student prediction before controls/evaluation')
    attempts=protect_target_annotations()
    stage_reference(FROZEN/'ZERO_UPDATE',root,'FROZEN')
    stage_reference(ROOT/'results/runs/uda_val_evaluation_20261026_v1/SOURCE_REHEARSAL',root,'SOURCE_REHEARSAL')
    count_matched(root,'FROZEN_AT_MLLM_COUNT','FROZEN','MLLM_DISTILL')
    count_matched(root,'MLLM_AT_FROZEN_COUNT','MLLM_DISTILL','FROZEN')
    atomic_json(root/'controls_sealed.json',dict(status='ALL_GT_FREE_CONTROLS_SEALED',target_GT_open_attempts=attempts,original_results_unchanged=True))
