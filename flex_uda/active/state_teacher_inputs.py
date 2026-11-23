"""Task-state side channels for the unchanged frozen temporal teacher.

The original CLIP raster/prompt remains an explicit control. Other variants
retain identical image crops while exposing current-time semantics, global
box geometry and/or actual frame offsets. No GT, model update, extra image
or assumed frame rate is introduced. This is a candidate input mechanism.
"""
import copy
import json
from flex_tta.runtime.pf_receipts import file_sha
from flex_uda.active.teacher_client import LocalTeacher
from flex_uda.teacher.teacher_cache import request_key

VARIANTS=('ORIGINAL','LAST','POSITION','TIME','STATE')
LAST_PROMPT=('These images are observations of the same tracked object, ordered from earliest to latest. '
    'Decide whether the object matches the description at the last image, which is the current decision time. '
    'Earlier images provide temporal context. Description: "{expression}".')


def frame_state(views,request,clip):
    key=(request['target'],request['video_id']);frames=clip['frames'];last=int(frames[-1]);rows=[]
    assert last==int(request['frame_id'])
    for frame in frames:
        frame=int(frame);ob=views.observations[key,frame,request['track_id']];image=views.frame(key,frame)[0]
        x1,y1,x2,y2=ob.bbox_xyxy
        rows.append(dict(frame_offset=frame-last,cx=(x1+x2)/(2*image.width),cy=(y1+y2)/(2*image.height),
            width=(x2-x1)/image.width,height=(y2-y1)/image.height))
    return rows


def prompt_for(expression,states,variant):
    assert variant in VARIANTS and variant!='ORIGINAL'
    text=LAST_PROMPT.format(expression=expression)
    if variant in ('POSITION','TIME','STATE'):
        values=[]
        for index,state in enumerate(states):
            row=dict(image=index+1)
            if variant in ('TIME','STATE'):row['source_frame_offset']=int(state['frame_offset'])
            if variant in ('POSITION','STATE'):
                row['global_normalized_box_center_xy_size_wh']=[round(float(state[k]),4) for k in ('cx','cy','width','height')]
            values.append(row)
        text+=' Image metadata in the same order: '+json.dumps(values,separators=(',',':'))+'.'
        if variant in ('POSITION','STATE'):text+=' Box coordinates refer to the entire original frame, not the crop.'
        if variant in ('TIME','STATE'):text+=' Frame offsets are source-frame indices relative to the last image; zero is current. No frame rate is assumed.'
    return text+' Answer Yes or No.'


class StateAwareTeacher(LocalTeacher):
    @classmethod
    def wrap(cls,base):
        obj=cls.__new__(cls);obj.__dict__.update(base.__dict__);return obj

    def prepare(self,request):
        variant=request.get('input_variant','STATE');assert variant in VARIANTS
        standard={k:v for k,v in request.items() if k!='input_variant'}
        original_key,images,prompt,info=super().prepare(standard)
        info=copy.deepcopy(info);info['requested_input_variant']=variant
        if info['effective_view']!='CLIP' or variant=='ORIGINAL':
            info['effective_input_variant']='ORIGINAL';return original_key,images,prompt,info
        states=frame_state(self.views,standard,info['clip']);prompt=prompt_for(request['query_text'],states,variant)
        info['input']['prompt']=prompt;info['input']['code']=file_sha(__file__)
        info['input']['temporal_input_variant']=variant;info['input']['frame_state']=states
        info['effective_input_variant']=variant;info['frame_state']=states
        return request_key(info['input']),images,prompt,info

    def ask_many(self,requests):
        answers=super().ask_many(requests)
        for request,answer in zip(requests,answers):
            variant=request.get('input_variant','STATE');answer['requested_input_variant']=variant
            answer['effective_input_variant']=variant if answer['effective_view']=='CLIP' else 'ORIGINAL'
        return answers
