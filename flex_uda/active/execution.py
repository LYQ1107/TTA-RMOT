"""Explicit Stage-A provenance and isolated outputs for the active experiment."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from flex_uda.presence.runtime import RUN, PUBLIC
from flex_tta.runtime.pf_receipts import file_sha


def active_paths():
    path=os.environ.get('PRESENCE_EXECUTION_MANIFEST')
    if not path:
        return SimpleNamespace(manifest=None, active=RUN / 'active', stage_a=RUN / 'stage_A',
            public=PUBLIC, launchers=RUN / 'active_launchers', gate=PUBLIC / 'stage_A_gate.json')
    manifest=json.loads(Path(path).read_text())
    return SimpleNamespace(manifest=manifest, manifest_path=Path(path),
        active=Path(manifest['active_root']), stage_a=Path(manifest['stage_A_root']),
        public=Path(manifest['public_root']), launchers=Path(manifest['launcher_root']),
        gate=Path(manifest['gate_path']))


def authorize_active():
    paths=active_paths();gate=json.loads(paths.gate.read_text())
    if paths.manifest is None:
        if gate['status']!='PASS_ALL_THREE_SAME_DOMAIN' or not gate.get('published_commit'):
            raise PermissionError('Complete and publish the original Stage-A gate first')
        return paths,gate['published_commit']
    m=paths.manifest
    if file_sha(paths.gate)!=m['gate_sha256']:
        raise PermissionError('Stage-A gate changed after publication')
    if gate['status']!='PASS_SAME_DOMAIN' or not gate['completed_full_A'] or gate['failed_targets']:
        raise PermissionError('All own-source gates and the nine-cell Stage A must pass first')
    if not m['published_commit'] or len(gate['same_domain'])!=3:
        raise PermissionError('Missing published three-source acceptance')
    if file_sha(gate['selection_path'])!=gate['selection_sha256']:
        raise ValueError('Source calibration selection changed')
    for key in ('active','public','launchers'):
        getattr(paths,key).mkdir(parents=True,exist_ok=True)
    return paths,m['published_commit']
