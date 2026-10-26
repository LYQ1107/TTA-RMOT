#!/usr/bin/env python3
"""Frozen, GT-isolated teacher worker. Never import the offline evaluator."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import yaml
import torch
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.teacher.isolation import restrict_filesystem,install_access_audit
from flex_uda.teacher.yestrack_backend import FrozenYesTrack
from flex_uda.teacher.teacher_probe import TeacherProbe
from flex_uda.teacher.teacher_cache import TeacherCache


def main():
    worker_started=time.perf_counter()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--staging-root',type=Path,required=True)
    p.add_argument('--method',required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('/data2/user/TTA-RMOT-assets/teacher-p1/cache/raw.sqlite'))
    p.add_argument('--limit',type=int);p.add_argument('--batch-size',type=int)
    p.add_argument('--memory-budget-gib',type=float,default=16)
    p.add_argument('--reliability-config',type=Path)
    a=p.parse_args();out=a.output_root.resolve();out.mkdir(parents=True,exist_ok=True)
    access=install_access_audit()
    experiment_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    implementation_hashes={str(path.relative_to(ROOT)):file_sha(path) for path in (ROOT/'flex_uda/teacher').glob('*.py')}
    implementation_hashes['scripts/run_teacher_probe.py']=file_sha(Path(__file__))
    config=yaml.safe_load(a.config.read_text());prompts=yaml.safe_load((ROOT/'configs/teacher/prompts_v1.yaml').read_text())
    protocol=yaml.safe_load((ROOT/'configs/teacher/probe_protocol.yaml').read_text())
    rules=dict(positive_threshold=0.9,negative_threshold=0.1,visual_dependence_min=0.03,probability_spread_max=0.2,
        status='PRE_REGISTERED_FUNCTIONALITY_DEFAULT_NOT_SOURCE_CALIBRATED_NOT_P2_READY')
    if a.reliability_config:rules=json.loads(a.reliability_config.read_text())
    manifest_sha=file_sha(a.manifest);rows=[json.loads(s) for s in a.manifest.read_text().splitlines()]
    expected_rows=len(rows)
    if a.limit is not None:rows=rows[:a.limit]
    if config['condition']=='T1':
        if not config.get('lora_path'):raise FileNotFoundError('official T1 missing; do not silently run T0')
    a.cache.parent.mkdir(parents=True,exist_ok=True)
    for name in ('cuda_cache','hf_cache','tmp'): (out/name).mkdir(exist_ok=True)
    os.environ.update(HF_HOME=str(out/'hf_cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        CUDA_CACHE_PATH=str(out/'cuda_cache'),TMPDIR=str(out/'tmp'))
    isolation=restrict_filesystem([a.staging_root,config['model_path'],config['yestrack_source'],
        ROOT/'flex_uda',ROOT/'flex_tta',ROOT/'scripts',ROOT/'configs',ROOT/'vendor',
        Path(sys.prefix),Path(sys.base_prefix),'/home/szgx/anaconda3/envs/vllmedit','/usr','/lib','/lib64','/bin','/etc','/proc','/sys'],
        [out,a.cache.parent,'/dev','/tmp'])
    torch.cuda.set_device(0);torch.manual_seed(protocol['seed'])
    props=torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(a.memory_budget_gib*(1<<30)/props.total_memory,0)
    backend=FrozenYesTrack(config);probe=TeacherProbe(backend,a.staging_root,a.cache,prompts,rules,a.batch_size or config['batch_size'])
    db=sqlite3.connect(out/'progress.sqlite');db.execute('CREATE TABLE IF NOT EXISTS samples (key TEXT PRIMARY KEY,payload TEXT)')
    done={r[0] for r in db.execute('SELECT key FROM samples')};started=time.perf_counter()
    for index,sample in enumerate(rows):
        if sample['sample_id'] in done:continue
        result=probe.evaluate(sample,a.method)
        result.update(sample_id=sample['sample_id'],video_id=sample['video_id'],query_id=sample['query_id'],
            frame_id=sample['frame_id'],track_id=sample['track_id'],method=a.method,condition=config['condition'],
            expression_type=sample['expression_type'],frozen_margin=sample['frozen_margin'])
        db.execute('INSERT INTO samples VALUES (?,?)',(sample['sample_id'],json.dumps(result,sort_keys=True)));db.commit()
        if index%25==0:print(json.dumps(dict(method=a.method,completed=index+1,planned=len(rows),budget=probe.budget.record())),flush=True)
    outputs=out/'teacher_outputs.jsonl'
    with outputs.open('w') as handle:
        for sample in rows:
            record=db.execute('SELECT payload FROM samples WHERE key=?',(sample['sample_id'],)).fetchone()
            if record is None:raise ValueError('incomplete teacher sample coverage')
            handle.write(record[0]+'\n')
    frozen=backend.seal_frozen()
    cached=TeacherCache(a.cache).get(probe.last_refs[0]) if probe.last_refs and probe.last_refs[0] else None
    receipt=dict(status='SEALED_FIXED_PROBE' if len(rows)==expected_rows else 'FUNCTIONALITY_SMOKE_ONLY',
        protocol='DEVELOPMENT_TEACHER_PROBE',method=a.method,condition=config['condition'],manifest_sha256=manifest_sha,
        samples=len(rows),planned_manifest_samples=expected_rows,videos=len({s['video_id'] for s in rows}),
        video_queries=len({(s['video_id'],s['query_id']) for s in rows}),config_sha256=file_sha(a.config),
        prompts_sha256=file_sha(ROOT/'configs/teacher/prompts_v1.yaml'),protocol_sha256=file_sha(ROOT/'configs/teacher/probe_protocol.yaml'),
        predictions_sha256=file_sha(outputs),budget=probe.budget.record(),worker_elapsed_seconds=time.perf_counter()-worker_started,
        inference_and_final_hash_seconds=time.perf_counter()-started,experiment_commit=experiment_commit,
        implementation_sha256=implementation_hashes,
        gpu_uuid=str(props.uuid),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),numeric_dtype='bfloat16',
        quantization=None,student_updates=0,model_load_seconds=backend.load_seconds,frozen=frozen,
        GT_access_audit=access,isolation=isolation,reliability_rules=rules,
        persistent_cache_reopen_check=bool(cached) if probe.last_refs else None,cache=probe.cache.stats())
    atomic_json(out/'receipt.json',receipt);print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
