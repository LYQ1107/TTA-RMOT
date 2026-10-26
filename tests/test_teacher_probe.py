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


def test_source_stage_includes_shifted_previous_history():
    from types import SimpleNamespace
    from scripts.prepare_teacher_source_calibration import required_history_frames
    class Episode:
        def history(self,track,frame):
            return [SimpleNamespace(frame_id=t) for t in range(frame-3,frame+1)]
    assert required_history_frames(Episode(),7,76)=={72,73,74,75,76}


def test_offline_weighted_metrics_ties_and_zero_acceptance():
    import numpy as np
    from scripts.evaluate_teacher_probe import weighted_binary,aggregate,softmatch_weights
    from scripts.diag_pf_query_information import average_precision
    from scripts.analyze_pf_full_offline import auroc
    score=np.array([.1,.5,.5,.9]);labels=np.array([False,True,False,True])
    ap,auc=weighted_binary(score,labels,np.ones(4))
    assert abs(ap-average_precision(score,labels))<1e-12
    assert abs(auc-auroc(score,labels))<1e-12
    assert np.array_equal(softmatch_weights(np.array([.95,.99]),.78,.02),np.ones(2))
    metric=aggregate('test',[],score,np.array(['UNKNOWN']*4),labels,[])
    assert metric['POS_precision'] is None and metric['accepted_matched_GT']==0


def test_seal_rejects_incomplete_or_GT_touched_receipt():
    import pytest
    from scripts.audit_teacher_leakage import validate_receipt
    with pytest.raises(ValueError,match='complete'):
        validate_receipt(dict(status='FUNCTIONALITY_SMOKE_ONLY'), 'YES_FULL','sha',8381)


def test_sparse_teacher_coverage_counts_distinct_GT_not_duplicate_boxes():
    import numpy as np
    from scripts.evaluate_teacher_probe import aggregate
    group=dict(indices=np.array([0,1]),iou=np.ones((2,1)),gt=[(7,(0,0,10,10))],track_ids=[1,2])
    metric=aggregate('synthetic',[0,1],np.array([.99,.98]),np.array(['SUPPORT_POS']*2),
        np.array([True,True]),[group])
    assert metric['POS_true_candidates']==2 and metric['accepted_matched_GT']==1
    assert metric['one_to_one_POS_precision']==.5 and metric['GT_positive_coverage']==1.


def test_four_shards_keep_complete_videos_and_all_fixed_pairs():
    from scripts.launch_teacher_probe_sharded import whole_video_shards
    rows=[dict(video_id=f'v{v}',sample_id=f'{v}/{t}') for v in range(25) for t in range(5+v)]
    shards=whole_video_shards(rows)
    assert shards==whole_video_shards(rows)
    assert len({r['sample_id'] for shard in shards for r in shard})==len(rows)
    for video in {r['video_id'] for r in rows}:
        assert sum(any(r['video_id']==video for r in shard) for shard in shards)==1


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
