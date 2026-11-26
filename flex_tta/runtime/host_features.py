"""Optional exact CPU tier for frozen visual features; not enabled by a runner."""
from collections import OrderedDict
from weakref import proxy
from flex_tta.runtime.frozen_features import FrozenFeatureCache


def tensor_bytes(values):
    return sum(x.numel()*x.element_size() for x in values)


class _OffloadingLRU(OrderedDict):
    def __init__(self,owner):
        super().__init__();self.owner=proxy(owner)

    def popitem(self,last=True):
        frame,value=super().popitem(last=last)
        self.owner._offload(frame,value)
        return frame,value


class HostBackedFrozenFeatureCache(FrozenFeatureCache):
    """Reuse the original encoder and decoder, without quantizing any tensor.

    Only evicted device features are copied to a bounded CPU LRU. This class
    is deliberately opt-in; the current registered active streams still use
    FrozenFeatureCache. Real GPU parity/timing is required before integration.
    """
    def __init__(self,*args,max_host_bytes,**kwargs):
        super().__init__(*args,**kwargs)
        self.max_host_bytes=int(max_host_bytes)
        self.host=OrderedDict();self.host_bytes=0
        self.visual=_OffloadingLRU(self)
        self.stats.update(host_hits=0,host_evictions=0,peak_host_cache_bytes=0)

    def _offload(self,frame,value):
        size=tensor_bytes(value)
        if size>self.max_host_bytes:return
        while self.host and self.host_bytes+size>self.max_host_bytes:
            _,old=self.host.popitem(last=False)
            self.host_bytes-=tensor_bytes(old);self.stats['host_evictions']+=1
        # Explicit copy keeps the same dtype/shape/value, including when CPU
        # tests model the device tier. No compression or pinned-memory claim.
        self.host[frame]=tuple(x.to('cpu',copy=True) for x in value)
        self.host_bytes+=size
        self.stats['peak_host_cache_bytes']=max(self.stats['peak_host_cache_bytes'],self.host_bytes)

    def _visual(self,frame,current_frame):
        if frame>current_frame:raise ValueError('future frame requested by frozen cache')
        if frame in self.visual or frame not in self.host:
            return super()._visual(frame,current_frame)
        value=self.host.pop(frame);size=tensor_bytes(value);self.host_bytes-=size
        value=tuple(x.to(self.device) for x in value)
        self.stats['host_hits']+=1;self.stats['visual_hits']+=1
        while self.visual and self.bytes+size>self.max_bytes:
            _,old=self.visual.popitem(last=False)
            self.bytes-=tensor_bytes(old)
        if size<=self.max_bytes:
            self.visual[frame]=value;self.bytes+=size
            self.stats['peak_cache_bytes']=max(self.stats['peak_cache_bytes'],self.bytes)
        return value
