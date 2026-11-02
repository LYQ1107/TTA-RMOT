"""Read-only sealed-asset verification before all new diagnostic metrics."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.track_ranking_common import *
from scripts import decompose_common as dc
from scripts.search_feasibility_common import RAW_SHAS


def main():
    os.environ['CUDA_VISIBLE_DEVICES']='';proof=require_plan();start=time.perf_counter()
    inventory=dc.OLD_PUBLIC/'margin_paths.json'
    for cmd in (['git','hash-object',str(inventory)],['git','rev-parse',SOURCE_COMMIT+':results/multidomain_generality/margin_paths.json']):
        if subprocess.check_output(cmd,cwd=ROOT,text=True).strip()!=SOURCE_BLOB:
            raise ValueError('The sealed margin inventory is not the requested 01add887 version')
    files=[r for r in json.loads(inventory.read_text())['files'] if r['coverage']=='complete'];inputs=[record(inventory)];datasets=[]
    if len(files)!=9:raise ValueError('Need exactly nine official complete banks')
    for target in TARGETS:
        meta,arrays,_,records=dc.load_target(target)
        inputs.extend(record(r['path'])|dict(target=target,model=m) for m,r in records.items())
        datasets.append(dict(target=target,videos=len(meta['videos']),queries=len(meta['episodes']),
            complete_candidate_query_observations=len(next(iter(arrays.values()))),
            physical_candidate_observations=sum(len(v['lines']) for v in meta['videos'].values())))
    old=dc.RUN/'probe';probe_plan=json.loads((old/'plan.json').read_text());probe_seal=json.loads((old/'seal.json').read_text())
    checked(dict(path=probe_plan['manifest_path'],sha256=probe_plan['manifest_sha256']))
    inputs.extend([record(old/'plan.json'),record(old/'seal.json'),record(probe_plan['manifest_path'])])
    if probe_seal['status']!='ALL_PAIRED_PROBES_SEALED_BEFORE_GT':raise ValueError('Unsealed teacher cache')
    for i,w in enumerate(probe_seal['workers']):
        checked(dict(path=w['outputs_path'],sha256=RAW_SHAS[i]));inputs.append(record(w['outputs_path']))
    subset=dc.OLD_PUBLIC/'sports_template_inventory.json';inputs.append(record(subset))
    plan=dict(status='ALL_SEALED_INPUTS_VERIFIED_BEFORE_METRICS',role=ROLE,prereg_publication=proof,
        clarifications_public_commit='2bfc9ac160ee8a3ac65c87e0cf1770e4491fce93',
        clarifications=record(PUBLIC/'IMPLEMENTATION_NOTES.md'),inputs=inputs,datasets=datasets,
        margin_source_commit=SOURCE_COMMIT,margin_inventory_git_blob=SOURCE_BLOB,
        new_teacher_calls=0,new_model_calls=0,GPU_hours=0,optimizer_updates=0,CPU_wall_seconds=time.perf_counter()-start)
    dump(RUN/'plan.json',plan);dump(PUBLIC/'input_inventory.json',plan)
    print(json.dumps(dict(event='TRACK_RANKING_INPUTS_VERIFIED',datasets=datasets,GPU_hours=0)),flush=True)


if __name__=='__main__':main()
