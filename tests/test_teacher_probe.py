import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from flex_uda.teacher.semantic_parser import parse_expression
from flex_uda.teacher.gt_blind_sampler import select_candidates
from flex_uda.teacher.evidence_verifier import confidence_decision,verify
from flex_uda.teacher.teacher_cache import TeacherCache


def test_parse_three_required_conditions_and_unknown():
    p=parse_expression('dacning person with white Tshirt and purple pants')
    assert p.kind=='MIXED' and len(p.attributes)==3 and p.original.startswith('dacning')
    assert parse_expression('someone maybe near something').kind=='UNSUPPORTED'


def test_sampler_deduplicates_and_fills_causal_candidates():
    result=select_candidates([1,2,3,4],{1:0.,2:-1.,3:-2.,4:-3.},{1:1,2:2,3:3,4:4},17,'v/q/f')
    assert len(result)==4 and len({r[0] for r in result})==4
    assert result==select_candidates([1,2,3,4],{1:0.,2:-1.,3:-2.,4:-3.},{1:1,2:2,3:3,4:4},17,'v/q/f')


def test_no_probability_alone_can_make_verified_positive():
    assert confidence_decision(.99,False).state=='UNKNOWN'
    assert verify([],[],.9,.1).state=='UNKNOWN'
    views={'original':dict(yes_probability=.99,top1_is_yes_no=True)}
    result=verify([dict(text='wearing black')],[views],.9,.1)
    assert result.state=='UNKNOWN' and 'TEMPORAL_EVIDENCE_MISSING' in result.reasons


def test_cache_raw_logits_survive_reopen(tmp_path):
    path=tmp_path/'raw.sqlite';cache=TeacherCache(path)
    assert cache.get_or_claim('id') is None
    cache.put('id',dict(yes_logit=2.,no_logit=-2.,yes_probability=.98))
    assert TeacherCache(path).get_or_claim('id')['yes_logit']==2.


def test_landlock_denies_unstaged_payload_and_stage_write(tmp_path):
    good=tmp_path/'stage';bad=tmp_path/'not_staged';good.mkdir();bad.mkdir()
    (good/'image.txt').write_text('legal');(bad/'labels.json').write_text('not legal teacher input')
    code='''
import sys
from pathlib import Path
from flex_uda.teacher.isolation import restrict_filesystem
stage,bad=map(Path,sys.argv[1:])
result=restrict_filesystem([stage,'/usr','/lib','/lib64',sys.prefix,sys.base_prefix],[])
assert (stage/'image.txt').read_text()=='legal'
for path,mode in [(bad/'labels.json','r'),(stage/'new.txt','w')]:
    try:
        with path.open(mode): pass
    except PermissionError: pass
    else: raise AssertionError('kernel permitted forbidden access')
print('KERNEL_ISOLATION_PASS',result['landlock_abi'])
'''
    run=subprocess.run([sys.executable,'-c',code,str(good),str(bad)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    assert 'KERNEL_ISOLATION_PASS' in run.stdout
