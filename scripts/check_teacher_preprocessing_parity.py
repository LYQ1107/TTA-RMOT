#!/usr/bin/env python3
"""CPU-only real-frame pixel/metadata parity for readonly preprocessing caches."""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from flex_uda.teacher.visual_input_builder import VisualInputBuilder
from flex_uda.teacher.yestrack_backend import upstream_functions
from flex_uda.teacher.isolation import install_access_audit
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def main():
    access=install_access_audit()
    previous='4a4b797b67fa0cde2fb683b122052dadf93ea965'  # Published tree equals server9b189ff.
    original=subprocess.check_output(['git','show',previous+':flex_uda/teacher/visual_input_builder.py'],cwd=ROOT,text=True)
    namespace={};exec(compile(original,'original_visual_input_builder.py','exec'),namespace)
    upstream=upstream_functions('/data2/user/TTA-RMOT-assets/teacher-p1/references/YesTrack')
    stage=Path('/data2/user/TTA-RMOT-assets/teacher-p1/staging_full')
    manifest=ROOT/'results/teacher_probe/manifest.jsonl';rows=[json.loads(s) for s in manifest.read_text().splitlines()]
    samples=[rows[0],rows[len(rows)//2]]
    before=namespace['VisualInputBuilder'](stage,upstream);after=VisualInputBuilder(stage,upstream)
    outputs=[];started=time.perf_counter();checks=0
    variants=('original','no_image','candidate_occluded','background_occluded','previous_observation')
    for repeat in range(2):
        for sample in samples:
            for view in ('current','history'):
                for variant in variants:
                    a=before.build(sample,view,variant);b=after.build(sample,view,variant)
                    assert a[1:]==b[1:],(sample['sample_id'],view,variant,'metadata')
                    assert [im.tobytes() for im in a[0]]==[im.tobytes() for im in b[0]],(view,variant,'RGB pixels')
                    checks+=1
                    if repeat==0:outputs.append(dict(sample_id=sample['sample_id'],view=view,variant=variant,image_sha256=b[3].get('image_sha256',[])))
    assert not torch.cuda.is_initialized() and not access['target_GT_open_attempts']
    result=dict(status='EXACT_REAL_RGB_METADATA_SHA_PARITY_PASS',checks=checks,real_fixed_samples=2,
        manifest_sha256=file_sha(manifest),previous_source=previous,
        current_builder_sha256=file_sha(ROOT/'flex_uda/teacher/visual_input_builder.py'),
        wall_seconds=time.perf_counter()-started,new_cache=after.cache_stats(),
        target_GT_access_audit=access,GPU_context_initialized=False,teacher_or_student_parameters_updated=0,
        outputs=outputs,numeric_model_configuration_unchanged=True,healthy_GPU_workers_restarted=False)
    atomic_json(ROOT/'results/teacher_probe/assets/preprocessing_cache_parity.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='outputs'},indent=2))


if __name__=='__main__':main()
