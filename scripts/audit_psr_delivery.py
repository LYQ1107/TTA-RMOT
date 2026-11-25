#!/usr/bin/env python3
"""Check the delivered full predictions/official CSVs and capture actual envs.

CPU only, no model inference and no target annotations. Training receipts must
already be final; this does not inspect or poll active workers.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def environment(python):
    command="""import importlib.metadata as m,json,sys
names=['torch','torchvision','transformers','tokenizers','numpy','scipy','Pillow','timm','yacs','opencv-python','huggingface-hub','safetensors','qwen-vl-utils','einops','scikit-learn']
versions={}
for name in names:
 try:versions[name]=m.version(name)
 except m.PackageNotFoundError:versions[name]=None
print(json.dumps(dict(python=sys.version,executable=sys.executable,packages=versions)))
"""
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    return json.loads(subprocess.check_output([str(python),'-c',command],env=env,text=True))


def audit(public,output):
    public=Path(public).resolve();seal=json.loads((public/'controls_sealed.json').read_text())
    index=json.loads((public/'artifact_index.json').read_text());methods={}
    for method in seal['methods']:
        archive=public/'archives'/(method+'_425_predictions.tar.gz')
        if file_sha(archive)!=index['methods'][method]['sha256']:raise ValueError('archive differs: '+method)
        with tarfile.open(archive,'r:gz') as stream:
            members=stream.getmembers();predictions=[m for m in members if m.name.endswith('/predict.txt')]
            if len(predictions)!=425 or len({m.name.split('/')[0] for m in predictions})!=25:raise ValueError('incomplete archive: '+method)
        with (public/method/'pedestrian_detailed.csv').open() as stream:
            rows=[r for r in csv.DictReader(stream) if r['seq']!='COMBINED']
        if len(rows)!=425 or len({r['seq'].split('+')[0] for r in rows})!=25:raise ValueError('incomplete official CSV: '+method)
        methods[method]=dict(predictions=425,videos=25,official_sequences=425,archive_sha256=file_sha(archive))
    students={}
    for method in seal['student_methods']:
        folder=public/'training'/method;r=json.loads((folder/'training_receipt.json').read_text())
        if r['steps']!=8973 or r['trainable_parameters']!=24576 or r['source_videos_seen']!=17 or r['target_query_pairs_seen']!=425 or r['target_GT_open_attempts'] or not r['frozen_parameters_unchanged']:raise ValueError('invalid final student: '+method)
        students[method]=dict(steps=8973,parameters=24576,source_videos=17,target_queries=425,target_GT_open_attempts=0,
            checkpoint_SHA256=r['checkpoints']['checkpoint_best_source_selected.pt']['sha256'])
    vendor=ROOT/'vendor/FlexHook';patch=ROOT/'results/positive_free/full/manifests/vendor_tracked_source.patch'
    current=subprocess.check_output(['git','diff','--','data/mydataloader.py'],cwd=vendor)
    if current!=patch.read_bytes():raise ValueError('publish the actual vendor patch before claiming reproducibility')
    result=dict(status='PASS_COMPLETE_DELIVERED_PREDICTIONS_OFFICIAL_RESULTS_AND_STUDENTS',methods=methods,students=students,
        target_GT_reads=0,GPU_calls=0,does_not_imply_scientific_success=True,
        vendor=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=vendor,text=True).strip(),
            patch=str(patch.relative_to(ROOT)),patch_sha256=file_sha(patch)),
        environments={name:environment(path) for name,path in (
            ('student_CPU_evaluation','/home/szgx/anaconda3/envs/dfot/bin/python'),
            ('frozen_MLLM','/data2/user/TTA-RMOT-assets/teacher-p1/env/bin/python'))})
    atomic_json(output,result);print(json.dumps(dict(status=result['status'],complete_methods=len(methods),complete_students=len(students),GPU_calls=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--public-root',type=Path,default=ROOT/'results/psr/full')
    p.add_argument('--output',type=Path,default=ROOT/'results/psr/assets/delivery_audit_primary.json')
    a=p.parse_args();audit(a.public_root,a.output)
