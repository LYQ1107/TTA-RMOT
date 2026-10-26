"""Frozen encoder reuse and optional UDA feature return, original decode intact."""
import copy
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from flex_tta.adaptation.pcd import PCDLoRATTA
from flex_tta.runtime.frozen_features import FrozenFeatureCache


def attach_adapter(model,device,rank=4):
    adapter=PCDLoRATTA(model,rank=rank,layer_indices=(-2,-1)).to(device)
    return adapter


def prepared_inputs(cache,candidates,sentence,frame):
    with torch.autocast(cache.device.type,enabled=False):
        return _prepared_inputs(cache,candidates,sentence,frame)


def _prepared_inputs(cache,candidates,sentence,frame):
    features={}
    for _,history in candidates:
        for obs in history:
            if obs.frame_id not in features:features[obs.frame_id]=cache._visual(obs.frame_id,frame)
    rows=[features[o.frame_id] for _,hs in candidates for o in hs]
    outputs=[torch.cat([r[layer] for r in rows],dim=0) for layer in range(len(rows[0]))]
    positions,boxes,ids=cache.episode.geometry_inputs(candidates,cache.device)
    text,mask=cache._text(sentence);n=len(ids)
    return dict(outputs=outputs,text=text.expand(n,-1,-1,-1).contiguous(),mask=mask.expand(n,-1,-1).contiguous(),
        positions=positions,boxes=boxes,track_ids=ids)


def forward_uda(model,prepared,*,return_features=True,strong_view=False):
    value={}
    handle=None
    if return_features:
        handle=model.output_ca_qkv[-1].register_forward_hook(lambda _m,_i,out:value.update(pcd=out))
    try:
        # Feature dropout is an augmentation, NOT independent semantic evidence.
        outputs=[F.dropout(x,p=.05,training=True) for x in prepared['outputs']] if strong_view else prepared['outputs']
        logits,regular=model.decode(outputs,prepared['text'],prepared['mask'],prepared['positions'],prepared['boxes'])
    finally:
        if handle is not None:handle.remove()
    if logits.ndim!=4 or logits.shape[-2:]!=(1,2):raise RuntimeError('expected original [N,layers,one_query,2] logits')
    result=dict(all_logits=logits,last_logits=logits[:,-1,0],margin=logits[:,-1,0,1]-logits[:,-1,0,0],
        regular=regular,track_ids=prepared['track_ids'])
    if return_features:result['pcd_features']=value['pcd']
    return result


@torch.no_grad()
def ema_update(teacher,student,decay=.99):
    source=dict(student.named_parameters());changed=0
    for name,p in teacher.named_parameters():
        if source[name].requires_grad:
            p.mul_(decay).add_(source[name].detach(),alpha=1-decay);changed+=p.numel()
    return changed


def frozen_copy(model):
    teacher=copy.deepcopy(model).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    return teacher
