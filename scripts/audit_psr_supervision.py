#!/usr/bin/env python3
"""GT-free supervision/accounting audit; never determines semantic correctness."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.teacher.psr_dataset import PSRPseudoDataset
from flex_uda.teacher.mllm_pseudo_dataset import identity
from flex_uda.training import protect_target_annotations
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def audit(pseudo_root,output):
    attempts=protect_target_annotations();pseudo_root=Path(pseudo_root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    all_policies={};budgets={}
    for policy in ('NAIVE','PSR','UNIFORM','CONFIDENCE'):
        data=PSRPseudoDataset(pseudo_root/policy,allow_val_diagnostic=True);counts=Counter((r['video_id'],r['query_id']) for r in data.rows)
        budgets[policy]=dict(counts)
        if len(data.runtime)!=425 or len(data.rows)!=8500 or len(data.groups)!=2125 or len(counts)!=425 or set(counts.values())!={20}:raise ValueError('actual matched8500/425 protocol changed')
        for row in data.expanded:
            if row['state']!='POS' or row['frame_id']!=row['anchor_frame'] or row['observed_teacher_frames']!=[row['frame_id']-1,row['frame_id']+1]:raise ValueError('unbounded/disguised temporal supervision')
        direct_ids={identity(r) for r in data.rows};weak_ids={identity(r) for r in data.expanded}
        if direct_ids&weak_ids or len(weak_ids)!=len(data.expanded):raise ValueError('expanded coverage is not new unique observations')
        raw=Path(data.seal['teacher_receipt']).parent/'teacher_outputs.jsonl';required={};raw_recorded={}
        for line in raw.read_text().splitlines():
            row=json.loads(line);response=row['raw_response'];required[response['request_id']]=response;raw_recorded[row['sample_id']]=response
        teacher=dict(required_unique_raw_requests=len(required),logical_candidate_judgments=len(data.rows),logical_images=sum(raw_recorded[r['sample_id']]['images'] for r in data.rows),
            required_visual_tokens=sum(r['visual_tokens'] for r in required.values()),required_language_tokens=sum(r['language_tokens'] for r in required.values()),
            equivalent_amortized_forward_seconds=sum(r['batch_forward_seconds']/r['batch_size'] for r in required.values()),
            actual_this_worker=data.seal['teacher_actual_budget'],cache_sharing_note='Each method requires its own8500logical judgments; across-policy identical inputs may share raw forwards. Cache hits do not remove inherited model cost.')
        result=dict(policy=policy,videos=25,queries=425,registered_windows=2125,actual_direct=8500,valid_soft=data.seal['valid_soft_samples'],
            direct_states=dict(Counter(r['state'] for r in data.rows)),expanded_weak_POS=len(data.expanded),expanded_NEG=0,
            original_manifest_sha256=data.seal['direct_manifest_sha256'],sealed_cache_sha256=file_sha(data.directory/'seal.json'),
            teacher_budget=teacher,new_unique_weak=len(weak_ids),teacher_annotated_GT_correctness='NOT_COMPUTED_BEFORE_ALL_STUDENT_PREDICTIONS_SEALED',
            target_GT_open_attempts=attempts,original_cache=str(data.directory),source_revision=data.seal['teacher_revision'],
            expanded_source_margin_note='Weak cached rows inherit endpoint metadata; any frozen-score comparisons must join the actual middle frame from the Frozen bank, not use parent frozen_margin.',
            expanded_raw_request_note='Raw request provenance is from the explicitly named endpoint parents, not a new teacher call on the middle frame.')
        all_policies[policy]=result
        dest=output/policy;dest.mkdir(exist_ok=True)
        for name in ('seal.json','selection_audit.json','expansion_audit.json'):shutil.copy2(data.directory/name,dest/name)
        atomic_json(dest/'supervision_audit.json',result)
    if any(budgets[m]!=budgets['NAIVE'] for m in budgets):raise ValueError('unmatched per-query budgets')
    result=dict(status='PASS_GT_FREE_SUPERVISION_AND_EXACT_MATCHED_BUDGET',policies=all_policies,
        sampling_manifests_sha256=json.loads((pseudo_root/'manifest_receipt.json').read_text())['manifests_sha256'],
        total_logical_method_slots=34000,actual_per_query_budget=20,Val_same_set_not_Train_to_Val=True,
        raw_teacher_caches_and_images_retained_on_server=True,target_GT_open_attempts=attempts,
        scientific_gain='NOT_ESTABLISHED_BY_THIS_AUDIT')
    atomic_json(output/'summary.json',result);print(json.dumps(dict(status=result['status'],weak_by_policy={m:r['expanded_weak_POS'] for m,r in all_policies.items()})));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pseudo-root',type=Path,default=Path('/data2/user/TTA-RMOT-assets/psr/val_v1'));p.add_argument('--output',type=Path,default=ROOT/'results/psr/supervision');a=p.parse_args();audit(a.pseudo_root,a.output)
