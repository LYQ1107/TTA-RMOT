#!/usr/bin/env python3
"""Seal complete GT-free P1 predictions; this module never opens target GT."""
import argparse
from datetime import datetime,timezone
import gzip
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha

METHODS=('YES_FULL','YES_ENSEMBLE_BUDGET_MATCHED','STRUCTURED_EVIDENCE',
    'EVIDENCE_VERIFIED','YES_ENSEMBLE_B_PLUS','YESTRACK_TCP_TRP')
OUTPUT=ROOT/'results/teacher_probe'
RUN=ROOT/'results/runs/teacher_probe_p1_20261026_v1'
WEIGHT_SHA='9a13380b4279bc98f4680bd083ff4995fdcfec120cfded459513be90d041a956'


def validate_receipt(receipt,method,manifest_sha,expected):
    if receipt['status']!='SEALED_FIXED_PROBE' or receipt['method']!=method:
        raise ValueError('only complete fixed-probe receipts can be sealed')
    if receipt['samples']!=expected or receipt['manifest_sha256']!=manifest_sha:
        raise ValueError('incomplete/mismatched fixed sample coverage')
    if receipt['GT_access_audit']['target_GT_open_attempts'] or receipt['student_updates']:
        raise ValueError('teacher attempted target annotation access or student update')
    if not receipt['isolation']['kernel_read_whitelist'] or not receipt['isolation']['kernel_stage_write_denied']:
        raise ValueError('kernel input isolation was not enforced')
    frozen=receipt['frozen']
    if (not frozen['frozen_parameters_unchanged'] or frozen['optimizer_updates'] or
        frozen['initial_parameter_sha256']!=WEIGHT_SHA or frozen['final_parameter_sha256']!=WEIGHT_SHA):
        raise ValueError('teacher weights changed or different condition was used')


def compact_record(row):
    d=row['decision']
    if not math.isfinite(d['probability']) or not 0<=d['probability']<=1:
        raise ValueError('teacher produced a nonfinite/out-of-range probability; do not score it')
    return dict(sample_id=row['sample_id'],video_id=row['video_id'],query_id=row['query_id'],
        frame_id=row['frame_id'],track_id=row['track_id'],method=row['method'],condition=row['condition'],
        probability=d['probability'],state=d['state'],reasons=d.get('reasons',[]),
        request_ids=row['request_ids'],features={k:v for k,v in d.get('features',{}).items() if k!='attributes'},
        unverified_ensemble_probability=row.get('unverified_ensemble_probability'))


def validate_seal(output=OUTPUT):
    output=Path(output);record=json.loads((output/'config_hashes.json').read_text())
    if record['status']!='SEALED_BEFORE_TARGET_GT_EVALUATION':raise ValueError('predictions are not sealed')
    for path,sha in record['original_files'].items():
        if file_sha(Path(path))!=sha:raise ValueError('sealed raw prediction/receipt changed')
    for path,sha in record['config_hashes'].items():
        if file_sha(ROOT/path)!=sha:raise ValueError('pre-registered manifest/config/rules changed')
    for filename,key in [('teacher_outputs.jsonl','public_predictions_sha256'),('teacher_cache_index.json','public_cache_index_sha256')]:
        if file_sha(output/filename)!=record[key]:raise ValueError('sealed public outputs changed')
    raw_index=record['raw_response_index']
    if file_sha(Path(raw_index['path']))!=raw_index['sha256']:raise ValueError('sealed raw index changed')
    return record


def seal(run=RUN,output=OUTPUT,condition='T0'):
    run,output=Path(run),Path(output)
    if (output/'config_hashes.json').exists():return validate_seal(output)
    manifest=output/'manifest.jsonl';samples=[json.loads(s) for s in manifest.read_text().splitlines()]
    expected={s['sample_id']:s for s in samples};manifest_sha=file_sha(manifest)
    if len(samples)!=8381 or len(expected)!=8381 or len({s['video_id'] for s in samples})!=25:
        raise ValueError('the pre-registered 25video fixed manifest is required')
    config=ROOT/'configs/teacher/base_qwen2b.yaml'
    hashes={str(p.relative_to(ROOT)):file_sha(p) for p in [config,ROOT/'configs/teacher/prompts_v1.yaml',
        ROOT/'configs/teacher/probe_protocol.yaml',output/'source_calibration_rules_T0.json',
        output/'assets/runtime_versions.json',manifest]}
    result_files={};receipts={};requests={};records=[]
    for method in METHODS:
        folder=run/'full'/condition/method
        receipt=json.loads((folder/'receipt.json').read_text())
        validate_receipt(receipt,method,manifest_sha,len(samples))
        for key,path in [('config_sha256',config),('prompts_sha256',ROOT/'configs/teacher/prompts_v1.yaml'),
            ('protocol_sha256',ROOT/'configs/teacher/probe_protocol.yaml')]:
            if receipt[key]!=file_sha(path):raise ValueError('configuration changed after prediction')
        if receipt['reliability_rules']!=json.loads((output/'source_calibration_rules_T0.json').read_text()):
            raise ValueError('source rules differ between teacher methods')
        predictions=folder/'teacher_outputs.jsonl'
        if receipt['predictions_sha256']!=file_sha(predictions):raise ValueError('prediction hash mismatch')
        seen=set();refs=set()
        with predictions.open() as handle:
            for line in handle:
                row=json.loads(line);sid=row['sample_id']
                if sid in seen or sid not in expected:raise ValueError('duplicated/unregistered sample')
                sample=expected[sid];seen.add(sid)
                if any(row[k]!=sample[k] for k in ('video_id','query_id','frame_id','track_id')):
                    raise ValueError('candidate identity was changed')
                if row['method']!=method or row['condition']!=condition:raise ValueError('method condition mismatch')
                records.append(compact_record(row));refs.update(r for r in row['request_ids'] if r)
        if seen!=set(expected):raise ValueError('incomplete fixed probe')
        result_files[str(predictions)]=file_sha(predictions)
        result_files[str(folder/'receipt.json')]=file_sha(folder/'receipt.json')
        for shard in receipt.get('source_worker_receipts',[]):
            shard_receipt=json.loads(Path(shard['receipt']).read_text())
            validate_receipt(shard_receipt,method,shard_receipt['manifest_sha256'],shard['samples'])
            for field,sha_field in [('receipt','receipt_sha256'),('predictions','predictions_sha256'),('manifest','manifest_sha256')]:
                if file_sha(Path(shard[field]))!=shard[sha_field]:raise ValueError('sealed source shard changed')
                result_files[shard[field]]=shard[sha_field]
        receipts[method]=receipt;requests[method]=sorted(refs)
    # Raw cache is GT-free. Read it only after all prediction coverage checks pass.
    cache=Path('/data2/user/TTA-RMOT-assets/teacher-p1/cache/raw.sqlite')
    db=sqlite3.connect(f'file:{cache}?mode=ro',uri=True)
    index=[];raw_summary={};cache_totals={m:dict(unique_required_forwards=0,equivalent_images=0,equivalent_visual_tokens=0,
        equivalent_language_tokens=0,equivalent_amortized_forward_seconds=0.,raw_non_yes_no=0,
        logical_request_slots=0,logical_images=0,logical_visual_tokens=0,logical_language_tokens=0) for m in METHODS}
    membership={}
    for method,refs in requests.items():
        for ref in refs:membership.setdefault(ref,[]).append(method)
    for ref,methods in sorted(membership.items()):
        found=db.execute('SELECT owner,payload FROM outputs WHERE key=?',(ref,)).fetchone()
        if not found or found[1] is None:raise ValueError('missing persistent raw response')
        owner,payload=found;raw=json.loads(payload);view=raw['input_view']
        if raw['frozen_weight_sha256']!=WEIGHT_SHA:raise ValueError('cache has wrong teacher weight')
        index.append(dict(request_id=ref,owner_pid=owner,methods=methods,variant=view['variant'],
            frames=view.get('frames',[]),images=raw['images'],visual_tokens=raw['visual_tokens'],
            image_sha256=view.get('image_sha256',[]),
            language_tokens=raw['language_tokens'],yes_logit=raw['yes_logit'],no_logit=raw['no_logit'],
            yes_probability=raw['yes_probability'],top_token=raw['top_token'],
            top1_is_yes_no=raw['top1_is_yes_no'],
            payload_sha256=__import__('hashlib').sha256(payload.encode()).hexdigest()))
        raw_summary[ref]=index[-1]
        for method in methods:
            cost=cache_totals[method];cost['unique_required_forwards']+=1
            for field,raw_field in [('equivalent_images','images'),('equivalent_visual_tokens','visual_tokens'),
                ('equivalent_language_tokens','language_tokens')]:cost[field]+=raw[raw_field]
            cost['equivalent_amortized_forward_seconds']+=raw['batch_forward_seconds']/raw['batch_size']
            cost['raw_non_yes_no']+=int(not raw['top1_is_yes_no'])
    db.close()
    by_method={m:{} for m in METHODS}
    for row in records:
        by_method[row['method']][row['sample_id']]=row
        cost=cache_totals[row['method']]
        for ref in row['request_ids']:
            if ref is None:continue
            if any(t>expected[row['sample_id']]['frame_id'] for t in raw_summary[ref]['frames']):
                raise ValueError('future frame in teacher request')
            cost['logical_request_slots']+=1
            for field,raw_field in [('logical_images','images'),('logical_visual_tokens','visual_tokens'),('logical_language_tokens','language_tokens')]:
                cost[field]+=raw_summary[ref][raw_field]
    fairness={}
    for left,right in [('STRUCTURED_EVIDENCE','YES_ENSEMBLE_BUDGET_MATCHED'),('EVIDENCE_VERIFIED','YES_ENSEMBLE_B_PLUS')]:
        mismatches=0
        for sid in expected:
            def signature(row):
                return [None if ref is None else (raw_summary[ref]['image_sha256'],raw_summary[ref]['variant']) for ref in row['request_ids']]
            mismatches+=signature(by_method[left][sid])!=signature(by_method[right][sid])
        fairness[left+'_vs_'+right]=dict(pairwise_visual_or_call_slot_mismatches=mismatches,
            exact_visual_and_call_slots_matched=mismatches==0,
            language_tokens_measured_not_assumed_equal=True)
        if mismatches:raise ValueError('matched-budget methods did not receive identical paired visual inputs')
    public_predictions=output/'teacher_outputs.jsonl'
    with public_predictions.open('w') as handle:
        for row in records:
            # sample_id joins the complete fixed manifest; do not repeat long
            # video/query strings and 15 raw SHA keys in every public row.
            public={k:row[k] for k in ('sample_id','method','condition','probability','state','reasons')}
            public['planned_request_slots']=len(row['request_ids'])
            public['valid_request_slots']=sum(ref is not None for ref in row['request_ids'])
            public['missing_evidence_slots']=sum(ref is None for ref in row['request_ids'])
            if row['unverified_ensemble_probability'] is not None:
                public['unverified_ensemble_probability']=row['unverified_ensemble_probability']
            handle.write(json.dumps(public,sort_keys=True)+'\n')
    cache_index=output/'teacher_cache_index.json'
    raw_index=cache.parent/'full_p1_T0_response_index.jsonl.gz'
    with gzip.open(raw_index,'wt') as handle:
        for row in index:handle.write(json.dumps(row,sort_keys=True)+'\n')
    raw_index_record=dict(path=str(raw_index),sha256=file_sha(raw_index),records=len(index))
    atomic_json(cache_index,dict(protocol='GT_FREE_RAW_RESPONSE_INDEX_NOT_TARGET_LABELS',cache_path=str(cache),
        unique_raw_responses=len(index),by_method=cache_totals,raw_response_index=raw_index_record,
        first8_raw_response_examples=index[:8],
        recovery='join public rows to manifest by sample_id; complete per-pair request_ids/raw logits remain in original sealed method files and this server raw index',
        source_calibration_GT_and_images_not_published=True))
    seal_record=dict(status='SEALED_BEFORE_TARGET_GT_EVALUATION',condition=condition,
        sealed_at_utc=datetime.now(timezone.utc).isoformat(),
        sealing_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        fixed_pairs=len(samples),videos=25,video_queries=425,methods=list(METHODS),
        config_hashes=hashes,original_files=result_files,
        public_predictions_sha256=file_sha(public_predictions),public_cache_index_sha256=file_sha(cache_index),
        raw_predictions_contain_target_GT=False,teacher_target_GT_open_attempts=0,
        teacher_parameter_sha256=WEIGHT_SHA,receipts=receipts,fairness=fairness,raw_response_index=raw_index_record,
        future_frame_requests=0,
        warning='sparse development teacher probe; NOT full-video RMOT HOTA; read-only permission plus hash seal')
    atomic_json(output/'config_hashes.json',seal_record)
    for path in [public_predictions,cache_index,raw_index,*map(Path,result_files)]:path.chmod(0o444)
    print(json.dumps({k:v for k,v in seal_record.items() if k not in ('receipts','original_files')},indent=2))
    return seal_record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,default=RUN)
    p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args();seal(a.run_root,a.output)


if __name__=='__main__':main()
