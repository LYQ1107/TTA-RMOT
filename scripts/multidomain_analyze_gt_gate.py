"""Read-only attribution of the failed 418-template gate; never repairs GT."""
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.multidomain_runtime import PUBLIC,require_plan
from scripts.multidomain_complete_sports_gt import numeric_rows,select_original_rows
from scripts.diag_attrib_v2_common import write_csv
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def main():
    require_plan();path=PUBLIC/'sports_GT_completion.json';gate=json.loads(path.read_text())
    if gate['status']!='STOP_EXACT_REPRODUCTION_FAILED':raise ValueError('Only inspect the failed gate')
    annotations={r['sequence']:r for r in gate['annotation_sources']}
    raw={r['video']:r for r in gate['raw_box_sources']};rows=[];evidence=[]
    for check in gate['checks']:
        if check['all_fields_and_order_numeric_exact']:continue
        key=check['sequence'];video=key.split('+',1)[0];a=annotations[key];b=raw[video]
        for source in (a,b):
            if file_sha(source['path'])!=source['sha256']:raise ValueError('Original source changed')
        annotation=json.loads(Path(a['path']).read_text())
        original=numeric_rows(Path(check['official_path']).read_text().splitlines())
        reconstructed=numeric_rows(select_original_rows(Path(b['path']).read_text().splitlines(),annotation['targets']))
        if len(original)!=check['official_rows'] or len(reconstructed)!=check['reconstructed_rows']:
            raise ValueError('Mismatch discovery changed')
        left,right=Counter(original),Counter(reconstructed)
        left_ids={tuple(r[:2]):r for r in original};right_ids={tuple(r[:2]):r for r in reconstructed}
        ids=lambda values:dict(sorted(Counter(int(r[1]) for r in values).items()))
        span=lambda values:[int(min(r[0] for r in values)),int(max(r[0] for r in values))] if values else []
        row=dict(sequence=key,video=video,official_rows=len(original),raw_annotation_rows=len(reconstructed),
            common_full_numeric_rows=sum((left&right).values()),
            official_only_rows=sum((left-right).values()),raw_annotation_only_rows=sum((right-left).values()),
            different_values_same_frame_id=sum(left_ids[k]!=right_ids[k] for k in left_ids.keys()&right_ids.keys()),
            same_row_multiset_different_order=left==right and original!=reconstructed,
            official_track_ids=json.dumps(ids(original)),raw_annotation_track_ids=json.dumps(ids(reconstructed)),
            official_frame_span=json.dumps(span(original)),raw_annotation_frame_span=json.dumps(span(reconstructed)))
        rows.append(row);evidence.append(dict(**row,official_template_path=check['official_path'],
            official_template_sha256=file_sha(check['official_path']),annotation=a,original_SportsMOT=b,
            language=annotation['language'],annotation_fields=sorted(annotation)))
    result=dict(status='READ_ONLY_GT_CONFLICT_DIAGNOSIS_NO_REPAIR',gate_path=str(path),gate_sha256=file_sha(path),
        failed_templates=len(rows),affected_videos=len({r['video'] for r in rows}),
        all_official_files_numeric_subsets=all(r['official_only_rows']==0 for r in rows),
        no_shared_frame_id_box_or_field_changes=all(r['different_values_same_frame_id']==0 for r in rows),
        official_rows=sum(r['official_rows'] for r in rows),raw_annotation_rows=sum(r['raw_annotation_rows'] for r in rows),
        official_only_rows=sum(r['official_only_rows'] for r in rows),
        raw_annotation_only_rows=sum(r['raw_annotation_only_rows'] for r in rows),
        evidence=evidence,GT_files_created=0,GT_files_modified=0,formal_SportsMOT_evaluations=0,
        training_updates=0,teacher_calls=0,new_GPU_hours=0,
        interpretation='Observed source/template row-inclusion inconsistency; do not silently choose either as a corrected protocol')
    atomic_json(PUBLIC/'sports_GT_mismatch_analysis.json',result)
    write_csv(PUBLIC/'sports_GT_mismatch_analysis.csv',rows)
    print(json.dumps({k:v for k,v in result.items() if k!='evidence'}),flush=True)


if __name__=='__main__':main()
