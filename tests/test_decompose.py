"""Scientific protocol checks for the new fixed diagnostic, not broad CI."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
from flex_uda.teacher.decompose_inputs import PairedViews,CLIP_PROMPT,FRAME_PROMPT
from flex_uda.teacher.visual_input_builder import VisualInputBuilder
from scripts.decompose_common import ROOT,types,rank_within_groups,top_budget,round_budget


def views(tmp_path):
    for t in (1,4,10,14):Image.new('RGB',(1280,720),(20,40,80)).save(tmp_path/f'{t:06d}.png')
    spec=dict(target='test',video_id='v',image_root=str(tmp_path),image_offset=0,extension='.png',fps=12,delta=3,
              observations=[dict(frame_id=t,track_id=7,bbox_xywh=[100,100,200,400]) for t in (1,4,10,14)])
    upstream=dict(crop_with_margin=lambda im,*args:im.crop((90,90,310,510)),
                  xywh_to_xyxy_with_margin=lambda *args:(90,90,310,510))
    return PairedViews([spec],upstream)


def test_taxonomy_normalization_and_word_boundaries():
    assert types('  LEFT team RUNNING in Brown shirt ') == ['MOTION','SPATIAL','APPEARANCE','ROLE']
    assert types('blueberry shadowed') == ['OTHER']
    assert types('person wearing a T-Shirt') == ['APPEARANCE']


def test_crop_path_is_inherited_unmodified(tmp_path):
    assert PairedViews.patch is VisualInputBuilder.patch
    v=views(tmp_path);image=v.patch(('test','v'),v.observations[('test','v'),10,7])
    assert image.size==(320,320)


def test_clip_uses_previous_not_future_observations(tmp_path):
    v=views(tmp_path);images,info=v.clip(('test','v'),10,7)
    assert info['frames']==[1,4,4,10] and len(images)==4
    assert not info['fallback'] and info['unique_frames']==3
    _,early=v.clip(('test','v'),1,7)
    assert early['frames']==[1] and early['fallback']


def test_frame_box_is_fixed_size_and_does_not_change_candidate(tmp_path):
    v=views(tmp_path);before=v.observations[('test','v'),10,7].bbox_xyxy
    image=v.frame_box(('test','v'),10,7)
    assert image.size==(1024,576)
    assert image.getpixel((80,80))==(255,0,0)
    assert v.observations[('test','v'),10,7].bbox_xyxy==before


def test_prompts_are_preregistered_literals():
    assert CLIP_PROMPT.format(expression='red shirt')=='These images show the same object at consecutive moments, from earliest to latest. Does this object match the description: "red shirt"? Answer Yes or No.'
    assert FRAME_PROMPT.format(expression='red shirt')=='Look at the object inside the red box. Does it match the description: "red shirt"? Answer Yes or No.'


def test_v2_ranks_budget_cap_and_ties():
    x=np.asarray([2.,2.,1.,-1.,1.]);groups=np.asarray([0,0,0,1,1])
    assert rank_within_groups(x,groups,2).tolist()==[0,1,2,1,0]
    assert top_budget(x,2)[0].tolist()==[True,True,False,False,False]
    assert round_budget(.5,3)==2 and round_budget(9,3)==3


def test_gt_audit_and_kernel_whitelist(tmp_path):
    allowed=tmp_path/'allowed';allowed.mkdir()
    # /tmp is intentionally a runtime cache/write root. Use a real read-only
    # repository file outside the explicitly allowed code/config directories.
    blocked=ROOT/'AGENTS.md'
    code='''
import json,sys
from pathlib import Path
from scripts.decompose_common import guard,sandbox
attempts=guard()
try:Path('/tmp/labels.json').read_text()
except PermissionError:pass
else:raise AssertionError('GT access must fail')
isolation=sandbox([], [sys.argv[1]])
try:Path(sys.argv[2]).read_text()
except PermissionError:pass
else:raise AssertionError('kernel whitelist must deny a non-GT out-of-scope file too')
print(json.dumps(dict(attempts=len(attempts),kernel=isolation['kernel_read_whitelist'])))
'''
    env=dict(os.environ,PYTHONPATH=str(ROOT))
    p=subprocess.run([sys.executable,'-c',code,str(allowed),str(blocked)],cwd=ROOT,env=env,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert json.loads(p.stdout)==dict(attempts=1,kernel=True)
