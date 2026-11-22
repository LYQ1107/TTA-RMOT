"""Check that sparse frame construction preserves the claimed information."""
import numpy as np
from flex_uda.active.relational_features import extra_features,moments,frame_supervision,references
from scripts.a100_relation_transport import relation_fields


def test_streaming_feature_moments_match_explicit_frame_expansion():
    rng=np.random.default_rng(6);raw=rng.normal(size=(3,2,5));coords=rng.uniform(size=(7,4))
    columns=np.array([0,1,0,0,1,0,1]);directions=np.array([-1,0,1])
    explicit=np.concatenate([np.concatenate((raw[q,columns],extra_features(coords,d)),axis=1) for q,d in enumerate(directions)])
    total,square,n=moments(raw,coords,columns,directions)
    assert n==len(explicit)
    assert np.allclose(total,explicit.sum(0),rtol=1e-12,atol=1e-12)
    assert np.allclose(square,np.square(explicit).sum(0),rtol=1e-12,atol=1e-12)


def test_frame_teacher_preserves_frame_variation_and_deduplicates_requests():
    query=[dict(expression_id='q',query_text='car on the right')]
    first=dict(query_id='q',track_id=7,frame_id=2,effective_view='CROP',yes_probability=.1,
        input=dict(bbox=[10,5,30,25],frame=2,view='CROP'))
    last=dict(query_id='q',track_id=7,frame_id=8,effective_view='FRAME_BOX',yes_probability=.9,
        input=dict(bbox=[60,5,80,25],frame=8,view='FRAME_BOX'))
    keys,p,coords,receipt=frame_supervision([first,last,first],query,np.array([7]),100,50)
    assert keys.tolist()==[[0,0,2],[0,0,8]] and np.array_equal(p,[.1,.9])
    assert np.allclose(coords[:,0],[.2,.7]) and receipt['unique_exact_requests']==2


def test_relation_ablation_keeps_values_and_fallback_while_changing_alignment():
    obs=np.array([[0,7],[1,7],[2,7],[3,7]])
    coords=np.array([[.1,.5,.1,.2],[.3,.5,.1,.2],[.7,.5,.1,.2],[.9,.5,.1,.2]])
    directions=np.array([1,0]);tracks=np.array([7]);base=np.array([[.4],[.8]])
    reference=references([],obs,tracks,directions,coords)
    fields=relation_fields(base,base,coords,np.zeros(4,int),directions,reference,4,obs,tracks,'video')
    full=fields['RELATION_FULL'];shuffled=fields['RELATION_SHUFFLED']
    assert np.array_equal(np.sort(full,axis=1),np.sort(shuffled,axis=1))
    assert not np.array_equal(full[0],shuffled[0])
    assert np.array_equal(full[1],fields['COARSE_DS'][1])
    assert np.allclose(full.mean(1),fields['RELATION_TIME_AVERAGE'].mean(1))
