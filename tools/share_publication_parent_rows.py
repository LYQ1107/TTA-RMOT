"""Losslessly share identical official HOTA aggregate rows across new reports.

This never opens model features, predictions or GT. Existing published files
remain unchanged; references use the already supported aggregate-row format.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.a100_visual_classical_sparse_statistics_reproduce import resolve_record,pack_statistic_rows

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'
LOCAL=ROOT/'a100_runs/visual_classical_20261121/analysis/cross_study_row_sharing_20261122'
BASES=('sparse_observation','strict_rgb','canonical_pcd','zero_colour')
TARGETS=('semantic_colour','joint_visual','appearance_cold','direct_teacher','direct_teacher_pcd','native_factorial','predictive_topology','native_cold','stable_topology','confidence_teacher','contract_graph','weighted_classical','cue_consensus','coverage_search','optional_transfer','protocol_calibration','pure_components','protocol_cold')
ENCODING=('parent_rows','parent_rows_zlib_base64','parent_rows_archive_owner',
    'parent_rows_archive_zlib_base64','parent_rows_archive_codec','numerator','denominator')


def key(value):
    return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).digest()


def same_types(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(same_types(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(same_types(x,y) for x,y in zip(a,b))
    return a==b


def main(apply=False,receipt_name='aggregate_cross_study_row_regression.json'):
    LOCAL.mkdir(parents=True,exist_ok=True);candidate=LOCAL/'candidate';candidate.mkdir(exist_ok=True)
    for path in PUBLIC.glob('*.json'):
        destination=candidate/path.name
        if not destination.exists():destination.symlink_to(path)
    cache={};seen={};rows=[];expected={};records=0
    for stem in BASES+TARGETS:
        name=stem+'_parent_statistics.json';path=PUBLIC/name
        if not path.is_file():continue
        document=json.loads(path.read_text());resolved={record:resolve_record(PUBLIC,name,record,cache) for record in document}
        published=bool(subprocess.check_output(['git','ls-tree','HEAD','--',str(path.relative_to(ROOT))]))
        if stem in BASES or published:
            # Once a report has been pushed it becomes a read-only provider
            # for later studies; never rewrite it as an unpublished target.
            destination=candidate/name
            if not destination.is_symlink():
                destination.unlink();destination.symlink_to(path)
            for record,value in resolved.items():
                for index,pair in enumerate(zip(value['numerator'],value['denominator'])):
                    seen.setdefault(key(pair),(name,record,index))
            continue
        payload={};shared=0
        for record,raw in document.items():
            value=resolved[record];records+=1;expected[name,record]=value
            if 'reference' in raw:payload[record]=raw;continue
            parent_rows=[]
            for index,pair in enumerate(zip(value['numerator'],value['denominator'])):
                digest=key(pair)
                if digest in seen:
                    filename,owner,position=seen[digest];reference=dict(record=owner,row=position)
                    if filename!=name:reference['file']=filename
                    parent_rows.append(dict(reference=reference));shared+=int(filename!=name)
                else:
                    seen[digest]=(name,record,index);parent_rows.append(list(pair))
            payload[record]=dict({k:v for k,v in raw.items() if k not in ENCODING},parent_rows=parent_rows)
        compact=pack_statistic_rows(payload);encoded=(json.dumps(compact,separators=(',',':'))+'\n').encode();original=path.read_bytes()
        use=encoded if len(encoded)<len(original) else original
        destination=candidate/name
        if destination.is_symlink():destination.unlink()
        destination.write_bytes(use)
        rows.append(dict(file=name,before=len(original),after=len(use),external_parent_row_references_considered=shared))
    actual_cache={}
    for (filename,record),before in expected.items():
        after=resolve_record(candidate,filename,record,actual_cache)
        assert same_types(before,after),(filename,record)
    report=dict(status='PASS_EXACT_CROSS_STUDY_AGGREGATE_ROWS',resolved_records=records,all_values_and_numeric_types_identical=True,
        files=rows,saved_bytes=sum(r['before']-r['after'] for r in rows),manual_GT_features_predictions_opened=0,applied=apply)
    if apply:
        backup=LOCAL/'original';backup.mkdir(exist_ok=True)
        for row in rows:
            path=PUBLIC/row['file'];destination=backup/path.name
            if not destination.exists():destination.write_bytes(path.read_bytes())
            path.write_bytes((candidate/path.name).read_bytes())
        (PUBLIC/receipt_name).write_text(json.dumps(report,indent=2)+'\n')
    (LOCAL/'last_candidate.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true')
    parser.add_argument('--receipt-name',default='aggregate_cross_study_row_regression.json')
    args=parser.parse_args();main(args.apply,args.receipt_name)
