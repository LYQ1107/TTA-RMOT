"""Check metadata retention and controls without a neural model or images."""
from types import SimpleNamespace
from unittest.mock import patch
import copy
from flex_uda.active.state_teacher_inputs import StateAwareTeacher,frame_state,prompt_for
from flex_uda.active.teacher_client import LocalTeacher
from flex_uda.teacher.teacher_cache import request_key


def scene(frames=(2,5,5,11),shift=0):
    key=('target','v');observations={}
    for f in frames:observations[key,f,7]=SimpleNamespace(bbox_xyxy=(10+shift,20,30+shift,60))
    return SimpleNamespace(observations=observations,frame=lambda key,f:[SimpleNamespace(width=100,height=100)])


def test_states_retain_real_offsets_and_repeated_observations():
    request=dict(target='target',video_id='v',track_id=7,frame_id=11)
    rows=frame_state(scene(),request,dict(frames=[2,5,5,11]))
    assert [r['frame_offset'] for r in rows]==[-9,-6,-6,0]
    assert rows[1]==rows[2] and rows[0]['cx']==.2


def test_state_channels_break_spatial_and_time_equivalences_of_ordinal_prompt():
    request=dict(target='target',video_id='v',track_id=7,frame_id=11)
    a=frame_state(scene(),request,dict(frames=[2,5,5,11]))
    b=frame_state(scene(shift=50),request,dict(frames=[2,5,5,11]))
    stretched=copy.deepcopy(a)
    for row in stretched:row['frame_offset']*=10
    text='cars moving on the left'
    assert prompt_for(text,a,'LAST')==prompt_for(text,b,'LAST')
    assert prompt_for(text,a,'POSITION')!=prompt_for(text,b,'POSITION')
    assert prompt_for(text,a,'TIME')!=prompt_for(text,stretched,'TIME')
    assert prompt_for(text,a,'POSITION')==prompt_for(text,stretched,'POSITION')
    assert prompt_for(text,a,'TIME')==prompt_for(text,b,'TIME')


def test_wrapper_preserves_original_images_and_nonclip_exact_control():
    obj=StateAwareTeacher.__new__(StateAwareTeacher);obj.views=scene()
    request=dict(target='target',video_id='v',track_id=7,frame_id=11,query_text='car on the left',view='CLIP')
    images=[object(),object()];info=dict(input=dict(prompt='old',code='old-code'),clip=dict(frames=[2,11]),effective_view='CLIP')
    original=request_key(info['input'])
    with patch.object(LocalTeacher,'prepare',return_value=(original,images,'old',info)):
        key,observed,text,meta=obj.prepare(dict(request,input_variant='ORIGINAL'))
        assert key==original and observed is images and text=='old'
        new_key,observed,text,meta=obj.prepare(dict(request,input_variant='STATE'))
        assert observed is images and new_key!=original and meta['input']['prompt']==text
        assert info['input']['prompt']=='old'
    info['effective_view']='CROP'
    with patch.object(LocalTeacher,'prepare',return_value=(original,images,'old',info)):
        key,observed,text,meta=obj.prepare(dict(request,input_variant='STATE'))
        assert key==original and observed is images and text=='old' and meta['effective_input_variant']=='ORIGINAL'
