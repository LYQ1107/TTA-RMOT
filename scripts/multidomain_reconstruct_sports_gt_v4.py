"""Authorized v4 GT protocol, separate from the preserved failed v3 gate."""
from collections import Counter
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.multidomain_runtime import (RUN,PUBLIC,require_plan,SPORTS_GT_V4_REVISION,
    SPORTS_GT_V4_STATUS,SPORTS_GT_PROTOCOL)
from scripts.multidomain_complete_sports_gt import numeric_rows,select_original_rows
from scripts.multidomain_diagnostics import base,verify
from scripts.diag_attrib_v2_common import write_new
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def main():
    started=time.perf_counter();publication=require_plan()
    if publication['revision']!=SPORTS_GT_V4_REVISION:
        raise PermissionError('Publish the newly authorized v4 before reconstructing GT')
    old_path=PUBLIC/'sports_GT_completion.json';old_sha=file_sha(old_path)
    if old_sha!='b7ab78fbf469c5ebe17e97f3ffac654f11f641fa9201ecd24f2b0d588cfcf20a':
        raise ValueError('Preserve the actual v3 failure record unchanged')
    old=json.loads(old_path.read_text())
    if old['exact_passed']!=410 or old['exact_failed']!=8:
        raise ValueError('The user-authorized conflict discovery changed')
    for record in old['prediction_inputs']:
        if file_sha(record['path'])!=record['sha256']:raise ValueError('Predictions changed after sealing')
    for model in ('V2-best','Dance-best'):
        record=json.loads((base('SportsMOT',model)/'GT_free_predictions.json').read_text())
        if record['GT_open_attempts']:raise PermissionError('Decoder GT isolation failed')
        for method in record['methods']:verify('SportsMOT',method,model)
    meta=json.loads((RUN/'sports/metadata.json').read_text())
    expected={q['video_id']+'+'+q['expression_id'] for q in meta['episodes']}
    annotations={r['sequence']:r for r in old['annotation_sources']}
    if set(annotations)!=expected or len(expected)!=466:raise ValueError('Keep all466 original sequences')
    raw={}
    for r in old['raw_box_sources']:
        if file_sha(r['path'])!=r['sha256']:raise ValueError('Original SportsMOT boxes changed')
        raw[r['video']]=Path(r['path']).read_text().splitlines()
    originals={r['sequence']:r for r in old['checks']};derived={};comparisons=[];changed=[]
    for key in sorted(expected):
        a=annotations[key]
        if file_sha(a['path'])!=a['sha256']:raise ValueError('Original LaMOT annotation changed')
        payload=json.loads(Path(a['path']).read_text());video=key.split('+',1)[0]
        derived[key]=select_original_rows(raw[video],payload['targets'])
        if key not in originals:continue
        original=originals[key]
        if file_sha(original['official_path'])!=original['official_sha256']:
            raise ValueError('Never overwrite or change the original418 templates')
        left=numeric_rows(Path(original['official_path']).read_text().splitlines())
        right=numeric_rows(derived[key]);exact=left==right
        if exact!=original['all_fields_and_order_numeric_exact']:
            raise ValueError('New difference outside the authorized410/8 conflict')
        extra=sum((Counter(right)-Counter(left)).values())
        if not exact:
            if Counter(left)-Counter(right):raise ValueError('Known official subset relation changed')
            changed.append(key)
        comparisons.append(dict(sequence=key,original_path=original['official_path'],
            original_sha256=original['official_sha256'],original_rows=len(left),v4_rows=len(right),
            all_fields_and_order_numeric_exact=exact,additional_original_annotation_rows=extra))
    if set(changed)!={r['sequence'] for r in old['failures']}:
        raise ValueError('The authorized eight-sequence set changed')
    root=RUN/'eval_assets/SportsMOT_v4_original_annotations';files=[]
    for key in sorted(expected):
        video,expression=key.split('+',1);path=root/'gt_template'/video/expression/'gt.txt'
        write_new(path,''.join(line+'\n' for line in derived[key]))
        original=originals.get(key)
        provenance=('ORIGINAL_TEMPLATE_EXACT_410' if original and original['all_fields_and_order_numeric_exact']
            else 'ORIGINAL_TEMPLATE_DIFFERS_8' if original else 'ORIGINAL_TEMPLATE_MISSING_48')
        files.append(dict(sequence=key,path=str(path),sha256=file_sha(path),rows=len(derived[key]),
            provenance=provenance,annotation=annotations[key],raw_box_video=video))
    for original in originals.values():
        if file_sha(original['official_path'])!=original['official_sha256']:
            raise ValueError('Original GT was modified')
    result=dict(status=SPORTS_GT_V4_STATUS,protocol=SPORTS_GT_PROTOCOL,
        prereg_publication=publication['remote_commit'],prereg_sha256=publication['prereg_sha256'],
        independent_post_seal_CPU_process=True,evaluation_root=str(root),
        prior_v3_failure=dict(path=str(old_path),sha256=old_sha,exact_passed=410,exact_failed=8),
        official_original_templates_unchanged=418,original_templates_exact=410,original_templates_differ=8,
        original_missing_templates=48,total_sequences=len(files),files=files,
        original_template_comparisons=comparisons,raw_box_sources=old['raw_box_sources'],
        original_annotation_sources=old['annotation_sources'],prediction_inputs=old['prediction_inputs'],
        GT_rows=sum(r['rows'] for r in files),empty_sequences=sum(r['rows']==0 for r in files),
        previously_missing_GT_rows=sum(r['rows'] for r in files if r['provenance'].endswith('_48')),
        authorized_eight_additional_rows=sum(r['additional_original_annotation_rows'] for r in comparisons),
        all466_uniform_original_annotation_rule=True,unmodified_official_template_reproduction=False,
        no_frame_box_ID_field_changes=True,GT_feedback_to_model=False,training_updates=0,teacher_calls=0,
        CPU_wall_seconds=time.perf_counter()-started,new_GPU_hours=0)
    atomic_json(PUBLIC/'sports_GT_reconstruction_v4.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','original_template_comparisons',
        'raw_box_sources','original_annotation_sources','prediction_inputs')}),flush=True)


if __name__=='__main__':main()
