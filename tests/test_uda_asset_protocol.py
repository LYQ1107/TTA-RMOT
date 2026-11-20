"""Phase0 checks, not a substitute for the requested real GPU UDA smoke."""
import json
import pytest
from flex_uda.data.manifests import official_splits,read_jsonl,write_jsonl,ROOT
from flex_uda.data.target_dance import TargetDanceDataset


def test_official_dance_train_val_disjoint():
    split=official_splits()['dance']
    assert len(set(split['train']))==40 and len(set(split['val']))==25
    assert not set(split['train'])&set(split['val'])
    assert set(split['val'])==set(split['test'])


def test_actual_all_input_expressions_are_retained():
    train=TargetDanceDataset(ROOT/'manifests/dance_train_uda.jsonl')
    val=TargetDanceDataset(ROOT/'manifests/dance_val_runtime.jsonl',split='val')
    assert len(train)==1560 and len(val)==425
    assert len({r['video_id'] for r in train.rows})==40


def test_target_runtime_rejects_membership_or_presence_fields(tmp_path):
    row=read_jsonl(ROOT/'manifests/dance_train_uda.jsonl')[0]
    row['target_membership']=[1]
    path=tmp_path/'bad.jsonl';write_jsonl(path,[row])
    with pytest.raises(ValueError,match='whitelist'):TargetDanceDataset(path,require_complete=False)


def test_training_rejects_val_even_if_images_and_text_exist(tmp_path):
    row=read_jsonl(ROOT/'manifests/dance_val_runtime.jsonl')[0]
    path=tmp_path/'val_as_train.jsonl';write_jsonl(path,[row])
    with pytest.raises(ValueError,match='contamination'):TargetDanceDataset(path,require_complete=False)


def test_missing_candidates_never_fall_back_to_target_gt(tmp_path):
    row=read_jsonl(ROOT/'manifests/dance_train_uda.jsonl')[0]
    row['predicted_tracks']=str(tmp_path/(row['video_id']+'.txt'))
    path=tmp_path/'train.jsonl';write_jsonl(path,[row])
    dataset=TargetDanceDataset(path,require_complete=False)
    with pytest.raises(FileNotFoundError,match='never fall back to GT'):dataset[0]


def test_hidden_target_gt_mutation_cannot_change_runtime_manifest_rows(tmp_path):
    rows=read_jsonl(ROOT/'manifests/dance_train_uda.jsonl')[:1]
    path=tmp_path/'train.jsonl';write_jsonl(path,rows)
    before=TargetDanceDataset(path,require_complete=False).rows
    hidden=tmp_path/'labels.json';hidden.write_text(json.dumps({'membership':[1,2],'presence':True}))
    after=TargetDanceDataset(path,require_complete=False).rows
    hidden.write_text(json.dumps({'membership':[],'presence':False}))
    assert before==after==TargetDanceDataset(path,require_complete=False).rows
