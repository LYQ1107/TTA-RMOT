import numpy as np
from scripts.count_structure_common import track_mean,causal_track_mean,em_prior,masks_for_video,frame_mask,topk_ids
from flex_uda.teacher.structure_prompts import parse_som,choice_prompt

def test_original_topk_tie_and_overflow():
    ids=np.array([2,5,7]);m=np.array([1.,1.,-1.]);q=dict(start=0,groups=[[1,0,3,0]])
    for k in (0,1,2,3,100):
        mask=frame_mask(q,ids,m,np.array([k]));assert set(ids[mask])==topk_ids(ids.tolist(),dict(zip(ids,m)),min(3,k))
    assert set(ids[frame_mask(q,ids,m,np.array([1]))])=={2}

def test_track_and_causal_means_no_future():
    inverse=np.array([0,1,0,1]);x=np.array([-1.,-3.,3.,1.])
    assert np.allclose(track_mean(x,inverse,2),[1,-1])
    assert np.allclose(causal_track_mean(x,inverse,2),[-1,-3,1,-1])
    future=x.copy();future[2:]+=100
    assert np.array_equal(causal_track_mean(x,inverse,2)[:2],causal_track_mean(future,inverse,2)[:2])

def test_all_selectors_structural_assignments():
    matrix=np.array([[-1.,-3.,3.,1.],[3.,1.,-1.,-3.]])
    inverse=np.array([0,1,0,1]);m,prior=masks_for_video(matrix,inverse,0.,0.,dict(a=1.,b=0.,pi_s=.5))
    assert len(m)==7 and len(prior)==2
    assert np.all(m['C2_QUERY_ARGMAX'].sum(0)==1)
    assert np.array_equal(m['C2_QUERY_ARGMAX'][0],[False,False,True,True])
    assert np.all(m['C3_QUERY_ARGMAX_TRACK'][0]) and not m['C3_QUERY_ARGMAX_TRACK'][1].any()
    assert np.array_equal(m['C4_C3_SHIFT'][0],[True,False,True,False])
    assert not m['C4_C3_SHIFT'][1].any()

def test_saerens_fixed_point_and_empty_prior():
    p,r=em_prior(np.array([-4.,-3.,3.,4.]),1.,0.,.5)
    assert r['converged'] and abs(r['pi']-.5)<1e-12 and np.allclose(p,1/(1+np.exp(-np.array([-4.,-3.,3.,4.]))))
    p,r=em_prior(np.full(20,-10.),1.,0.,.5)
    assert r['converged'] and r['pi']<1e-6 and not (p>.5).any()

def test_som_unknown_not_negative():
    assert parse_som(' none. ',3)==dict(valid=True,numbers=[],duplicate_numbers=0,error=None)
    assert parse_som('1, 3, 1',3)['numbers']==[1,3]
    assert parse_som('1, 3, 1',3)['duplicate_numbers']==1
    for text in ('people 1 and 2','0','4','1,'):
        r=parse_som(text,3);assert not r['valid'] and r['numbers'] is None

def test_choice_order_and_none_option():
    prompt=choice_prompt([dict(query_text='query'+str(j)) for j in range(17)],'.2 .3')
    assert 'A. query0' in prompt and 'Q. query16' in prompt and 'R. none of these' in prompt

def test_source_tracker_geometry_not_gt_id(tmp_path):
    from PIL import Image
    from scripts.count_structure_source import source_geometry
    root=tmp_path/'source';images=root/'KITTI/training/image_02/0005';images.mkdir(parents=True)
    Image.new('RGB',(100,100)).save(images/'000000.png')
    tracks=tmp_path/'tracks'
    for cls,text in [('car','2,7,10,20,30,40,1,-1,-1,-1\n'),('pedestrian','2,1,50,10,10,20,1,-1,-1,-1\n')]:
        path=tracks/'0005'/cls/'predict.txt';path.parent.mkdir(parents=True);path.write_text(text)
    geom,files=source_geometry('0005',root,tracks)
    assert set(geom)=={1} and set(geom[1])=={7,8}
    assert np.allclose(geom[1][7],[.1,.2,.4,.6]) and len(files)==2

def test_literal_granularity_rule():
    from scripts.count_structure_common import granularity_verdict
    assert granularity_verdict(31.771,0,0)=='视频级存在与数量即可达到同域官方水平'
    assert granularity_verdict(31,31.771,0)=='需要轨迹级判断'
    assert granularity_verdict(31,31,31.771)=='需要帧级存在'
    assert granularity_verdict(31,31,31)=='需要帧级数量'
