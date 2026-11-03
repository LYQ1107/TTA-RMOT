from types import SimpleNamespace
import pytest
import torch
from flex_tta.runtime.frozen_features import FrozenFeatureCache
from flex_tta.runtime.host_features import HostBackedFrozenFeatureCache


class Backbone:
    def __init__(self):self.calls=0
    def forward_features(self,image):
        self.calls+=1
        return (image*2,image+.25)


def make_cache(kind,host_frames=2):
    model=SimpleNamespace(backbone=Backbone())
    episode=SimpleNamespace(_image=lambda frame:torch.full((1,2,2),float(frame)))
    # Each frozen value is two float32 tensors of four elements (32 bytes).
    kwargs=dict(max_bytes=32)
    if kind is HostBackedFrozenFeatureCache:kwargs['max_host_bytes']=host_frames*32
    return kind(model,episode,None,torch.device('cpu'),**kwargs)


def test_host_hit_preserves_exact_values_and_avoids_reencoding():
    plain=make_cache(FrozenFeatureCache);host=make_cache(HostBackedFrozenFeatureCache)
    for frame in (1,2,3,1):
        a,b=plain._visual(frame,3),host._visual(frame,3)
        assert all(torch.equal(x,y) and x.dtype==y.dtype for x,y in zip(a,b))
    assert plain.model.backbone.calls==4 and host.model.backbone.calls==3
    assert host.stats['host_hits']==1 and host.host_bytes<=host.max_host_bytes
    assert host.bytes<=host.max_bytes


def test_future_frame_is_rejected_even_when_present_in_host_tier():
    cache=make_cache(HostBackedFrozenFeatureCache)
    cache._visual(2,2);cache._visual(1,2)
    assert 2 in cache.host
    with pytest.raises(ValueError,match='future frame'):cache._visual(2,1)


def test_host_bound_and_disabled_tier_do_not_change_results():
    for host_frames in (0,1):
        plain=make_cache(FrozenFeatureCache);host=make_cache(HostBackedFrozenFeatureCache,host_frames)
        for frame in (1,2,3,1):
            assert all(torch.equal(x,y) for x,y in zip(plain._visual(frame,3),host._visual(frame,3)))
            assert host.host_bytes<=host.max_host_bytes and host.bytes<=host.max_bytes


def test_completed_video_cache_does_not_retain_itself():
    import weakref
    cache=make_cache(HostBackedFrozenFeatureCache)
    cache._visual(1,2);cache._visual(2,2)
    reference=weakref.ref(cache)
    del cache
    assert reference() is None
