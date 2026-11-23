from flex_uda.active.query_scope import decode,consensus
import json


def description(part):
    return dict(subject='person',subject_span='person',logic='and',uncertain=[],terms=[
        dict(kind='appearance',name='color',value='white',part=part,binding='subject',
             reference='none',time='appearance',negated=False,span='white')])


def test_binding_disagreement_keeps_whole_query_instead_of_a_false_factorization():
    text='white person'
    a=decode(text,json.dumps(description('upper')))
    b=decode(text,json.dumps(description('unspecified')))
    result=consensus(text,a,b)
    assert not result['full_structure_consensus']
    assert result['accepted_structure']['logic']=='opaque'
    assert result['accepted_structure']['original_query_preserved']==text
    assert not result['known_subject_appearance_terms']


def test_invented_evidence_cannot_become_an_attribute_label():
    value=description('upper');value['terms'][0]['span']='blue shirt'
    result=decode('white person',json.dumps(value))
    assert not result['valid_format']
    assert result['terms'][0]['kind']=='opaque'
    assert result['original_query_preserved']=='white person'
