#!/usr/bin/env python3
"""Verify executed training code against actual fetched public git objects."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json


def audit(training,output):
    publication=json.loads((ROOT/'results/psr/assets/publication_map.json').read_text())['snapshots']
    cache={}
    def content(repo,rev,path):
        key=(str(repo),rev,path)
        if key not in cache:
            try:cache[key]=subprocess.check_output(['git','show',rev+':'+path],cwd=repo,stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:cache[key]=None
        return cache[key]
    def digest(repo,rev,path):
        value=content(repo,rev,path)
        return hashlib.sha256(value).hexdigest() if value is not None else None
    result={}
    for p in sorted(Path(training).glob('*/provenance.json')):
        receipt=json.loads((p.parent/'training_receipt.json').read_text())
        if receipt['status']!='COMPLETE_FIXED_COVERAGE_PSR_STUDY':raise ValueError('audit only completed students')
        provenance=json.loads(p.read_text());execution=provenance['experiment_commit'];own={};external={}
        for path,sha in provenance['implementation_sha256'].items():
            if path.startswith('vendor/FlexHook/'):
                rev='bd1acc38634b28525d54dc6e0fcb38335f0029f9'
                if digest(ROOT/'vendor/FlexHook',rev,path.removeprefix('vendor/FlexHook/'))!=sha:raise ValueError('external source mismatch')
                external[path]=dict(vendor_commit=rev,sha256=sha)
            else:
                if digest(ROOT,execution,path)!=sha:raise ValueError('recorded execution tree differs: '+path)
                own[path]=sha
        matches=[(local,public) for local,public in publication.items()
                 if all(digest(ROOT,public,path)==sha for path,sha in own.items())]
        if not matches:raise ValueError('no public snapshot contains actual recorded implementation: '+p.parent.name)
        local,public=matches[0]
        changed=subprocess.check_output(['git','diff','--name-only',execution,public,'--',
            'flex_uda','flex_tta','compat','scripts/run_flexhook_dance_pcd_tta.py'],cwd=ROOT,text=True).splitlines()
        if changed:raise ValueError('runtime packages differ from the matched public snapshot: '+str(changed))
        trees=[subprocess.check_output(['git','rev-parse',r+'^{tree}'],cwd=ROOT,text=True).strip() for r in (local,public)]
        if trees[0]!=trees[1]:raise ValueError('published snapshot tree mismatch')
        result[p.parent.name]=dict(execution_commit=execution,recorded_implementation_sha256=own,
            matching_public_snapshot=public,matching_public_tree=trees[0],external_source=external,
            runtime_package_differences=changed,
            claim='all recorded implementation files match; not a claim that local/public commit identities are identical')
    if not result:raise ValueError('no completed student provenance')
    atomic_json(output,dict(status='PASS_EXECUTION_IMPLEMENTATIONS_AVAILABLE_IN_PUBLIC_HISTORY',methods=result,
        repository='LYQ1107/TTA-RMOT',verification='actual fetched public git objects, SHA256 of recorded files, exact publication tree equality',
        target_GT_reads=0,GPU_calls=0,does_not_imply_scientific_success=True))
    print(json.dumps({m:r['matching_public_snapshot'] for m,r in result.items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-root',type=Path,default=ROOT/'results/psr/full/training')
    p.add_argument('--output',type=Path,default=ROOT/'results/psr/assets/source_revision_audit_primary.json')
    a=p.parse_args();audit(a.training_root,a.output)
