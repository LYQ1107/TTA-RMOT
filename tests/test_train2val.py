import json
from pathlib import Path
import subprocess
import sys

import pytest

from flex_uda.data.train2val import ASSETS,convert_geometry,geometry_rows,sentence_only


def test_sentence_only_skips_nested_referring_payload(tmp_path):
    path=tmp_path/'expression.json'
    path.write_text('{"label":{"2":[7,8],"quoted":"},\\\"[]"},"sentence":"a person","frames":[1,2]}')
    text,schema=sentence_only(path)
    assert text=='a person'
    assert schema=={'label':'object','sentence':'string','frames':'array'}


def test_candidate_conversion_preserves_six_fields(tmp_path):
    source=tmp_path/'gt.txt';dest=tmp_path/'tracks.txt'
    source.write_text('1,7,-2.50,4,10.25,20,0,9,0.01\n2,7,3,4,11,20,1,1,1\n')
    result=convert_geometry(source,dest)
    assert result['rows']==2 and result['first_six_columns_exact']
    assert [r.split(',')[:6] for r in dest.read_text().splitlines()]==list(geometry_rows(source))
    assert all(r.split(',')[6:]==['1','-1','-1','-1'] for r in dest.read_text().splitlines())


def test_actual_all40_conversion():
    receipt=json.loads((ASSETS/'candidate_receipt.json').read_text())
    assert len(receipt['videos'])==40
    for row in receipt['videos']:
        assert [r.split(',')[:6] for r in Path(row['path']).read_text().splitlines()]==list(geometry_rows(row['source']))
    assert receipt['candidate_rows']==348930


def test_psr_dataset_accepts_complete_train_split(tmp_path):
    from flex_uda.data.manifests import official_splits,write_jsonl
    from flex_uda.teacher.mllm_pseudo_dataset import REVISION,digest
    from flex_uda.teacher.psr_dataset import PSRPseudoDataset
    from flex_tta.runtime.pf_receipts import file_sha
    runtime=[];samples=[];windows=[]
    for v in official_splits()['dance']['train']:
        runtime.append(dict(video_id=v,expression_id='query',query_text='a person',predicted_tracks=str(tmp_path/'tracks.txt')))
        sample=dict(sample_id=v,window_id=v,video_id=v,query_id='query',frame_id=2,track_id=1,bbox_xyxy=[1,2,3,4])
        samples.append(sample);windows.append(dict(video_id=v,query_id='query',frame_id=2,window_id=v,sampled_candidates=1))
    write_jsonl(tmp_path/'runtime.jsonl',runtime);write_jsonl(tmp_path/'manifest.jsonl',samples);write_jsonl(tmp_path/'windows.jsonl',windows)
    manifest_sha=file_sha(tmp_path/'manifest.jsonl')
    rows=[dict(s,teacher_prob=.7,valid=True,teacher_revision=REVISION,manifest_sha256=manifest_sha,
        cache_identity={'video':s['video_id']},cache_key=digest({'video':s['video_id']})) for s in samples]
    write_jsonl(tmp_path/'soft_labels.jsonl',rows)
    names=('runtime.jsonl','manifest.jsonl','windows.jsonl','soft_labels.jsonl')
    (tmp_path/'seal.json').write_text(json.dumps(dict(status='SEALED_PSR_DIRECT_AND_WEAK_CACHE',target_split='train',
        files_sha256={n:file_sha(tmp_path/n) for n in names},tracker_files_sha256={})))
    dataset=PSRPseudoDataset(tmp_path,expanded_kind='none')
    assert dataset.split=='train' and len(dataset)==40 and len(dataset.groups)==40


def test_four_arms_preserve_registered_numeric_recipe():
    import yaml
    from scripts.run_train2val_training import configuration,ARMS,ROOT
    for method,seed in ARMS.values():
        path=ROOT/'configs/uda'/('diag_fit_opt.yaml' if method=='DIAG_FIT_OPT' else 'diag_fit_cap.yaml')
        original=yaml.safe_load(path.read_text());actual=configuration(method,seed)
        for key in ('adapter','optimizer','train','selection','loss','teacher','sampling','expansion'):
            assert actual[key]==original[key]


def test_user_verdict_rules_not_extended():
    from scripts.report_train2val import literal_verdict
    assert literal_verdict([20,21,22],[1,5])['literal_user_rule_verdict']=='MLLM 存在性知识可跨视频、跨表达泛化'
    assert literal_verdict([20,21,22],[-1,5])['literal_user_rule_verdict']=='同集增益未能泛化到未见视频和表达'
    assert literal_verdict([20,13.848,22],[1,5])['literal_user_rule_verdict']=='同集增益未能泛化到未见视频和表达'
    assert '未覆盖' in literal_verdict([20,21,22],[-5,-1])['literal_user_rule_verdict']


def test_packaging_reuses_identical_readonly_seal(tmp_path):
    from scripts.package_train2val import copy_immutable_metadata
    source=tmp_path/'source.json';source.write_text('{"sealed":true}\n');source.chmod(0o444)
    target=tmp_path/'published.json'
    copy_immutable_metadata(source,target)
    copy_immutable_metadata(source,target)
    assert target.read_bytes()==source.read_bytes()


@pytest.mark.parametrize('kernel_only',[False,True])
def test_real_train_only_whitelist_blocks_val_and_old_cache(tmp_path,kernel_only):
    # Separate child: never contaminate a real training audit with deliberate
    # denied-open tests. Kernel-only mode bypasses Python audit completely.
    code=r'''
import json,sys
from pathlib import Path
from types import SimpleNamespace
from flex_uda.data.manifests import read_jsonl,ROOT
from flex_uda.data.train2val import ASSETS
from flex_uda.mllm_training import enforce_inputs
from flex_uda.train2val_isolation import install_train_only_audit
output=Path(sys.argv[1]);kernel_only=sys.argv[2]=='True'
pseudo=SimpleNamespace(split='train',runtime=read_jsonl(ASSETS/'runtime_train.jsonl'),directory=ASSETS)
config=dict(input_policy='TRAIN_ONLY_NO_VAL',data=dict(source_train_manifest='manifests/source_kitti_v2_train.jsonl',source_val_manifest='manifests/source_kitti_v2_val.jsonl',frozen_reference_root=str(ASSETS/'frozen_sparse')))
attempts=[] if kernel_only else install_train_only_audit(ASSETS,output)
isolation=enforce_inputs(output,pseudo,config=config)
val=Path('/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance')
paths=[val/'DanceTrack/training/image_02/dancetrack0004/000001.png',
 Path('/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs/dancetrack0004.txt'),
 ROOT/'manifests/dance_val_runtime.jsonl',
 Path('/data2/user/TTA-RMOT-assets/psr/val_v1/NAIVE/soft_labels.jsonl'),
 ROOT/'results/runs/diag_fit_20261030_v1/students/DIAG_FIT_OPT/checkpoint_final_step.pt']
blocked=0
for p in paths:
 try:
  with p.open('rb') as f:f.read(1)
 except PermissionError:blocked+=1
assert blocked==len(paths),(blocked,paths)
assert kernel_only or len(attempts)==len(paths),attempts
print(json.dumps(dict(blocked=blocked,kernel_only=kernel_only,audit_attempts=len(attempts))))
'''
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path),str(kernel_only)],text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout.splitlines()[-1])['blocked']==5
