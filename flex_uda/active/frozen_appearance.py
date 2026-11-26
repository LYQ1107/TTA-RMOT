"""Frozen image-only descriptors, shared inputs rather than task labels."""
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np

REPOSITORY=Path('/data3/liuyeqiang/.cache/torch/hub/facebookresearch_dinov2_main')
WEIGHT=Path('/data3/liuyeqiang/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth')
OFFICIAL_WEIGHT='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'
OFFICIAL_BACKBONES_BLOB='af5fd95d38eb33f73a85b92a94a715773a4e70bc'


def load_model(device):
    os.environ['XFORMERS_DISABLED']='1'
    sys.path.insert(0,str(REPOSITORY))
    import torch
    from dinov2.hub.backbones import dinov2_vitb14
    torch.manual_seed(0)
    model=dinov2_vitb14(pretrained=False)
    model.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True),strict=True)
    model.eval().requires_grad_(False)
    return model.to(device)


def imported_source_files():
    from flex_tta.runtime.pf_receipts import file_sha
    result={}
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.is_file() and path.suffix=='.py' and path.is_relative_to(REPOSITORY):
                result[str(path)]=file_sha(path)
    return dict(sorted(result.items()))


def verify_assets():
    import requests
    import torch
    from flex_tta.runtime.pf_receipts import file_sha
    start=time.perf_counter()
    raw=(REPOSITORY/'dinov2/hub/backbones.py').read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    assert blob==OFFICIAL_BACKBONES_BLOB,(blob,OFFICIAL_BACKBONES_BLOB)
    sha=hashlib.sha256();md5=hashlib.md5()
    with WEIGHT.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):sha.update(block);md5.update(block)
    # The inherited HTTP proxy reset this public request; direct HTTPS was
    # separately verified to return the canonical object. TLS stays verified.
    session=requests.Session();session.trust_env=False
    response=session.head(OFFICIAL_WEIGHT,headers={'Accept-Encoding':'identity'},timeout=45)
    response.raise_for_status()
    etag=response.headers.get('ETag','').strip('"')
    length=int(response.headers.get('Content-Length',0))
    assert not length or length==WEIGHT.stat().st_size
    if etag==md5.hexdigest():
        verification='Official HTTPS object ETag equals local MD5; length matches when supplied'
        remote_sha=None
    else:
        remote=hashlib.sha256()
        with session.get(OFFICIAL_WEIGHT,headers={'Accept-Encoding':'identity'},stream=True,timeout=60) as downloaded:
            downloaded.raise_for_status()
            for block in downloaded.iter_content(1<<20):remote.update(block)
        remote_sha=remote.hexdigest();assert remote_sha==sha.hexdigest()
        verification='Full official HTTPS stream SHA256 equals local cached weight; no new weight file written'
    torch.set_num_threads(2)
    model=load_model('cpu')
    with torch.inference_mode():value=model(torch.zeros((1,3,224,224))).numpy()
    assert value.shape==(1,768) and np.isfinite(value).all() and np.linalg.norm(value)>0
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    result=dict(status='PASS_FROZEN_APPEARANCE_ASSETS_AND_CPU_FORWARD',weight_path=str(WEIGHT),
                weight_sha256=sha.hexdigest(),weight_bytes=WEIGHT.stat().st_size,official_weight_url=OFFICIAL_WEIGHT,
                official_object_ETag=etag,official_stream_sha256=remote_sha,weight_origin_check=verification,
                official_factory_Git_blob=blob,external_source_sha256=imported_source_files(),
                CPU_forward_shape=list(value.shape),model_parameters=sum(p.numel() for p in model.parameters()),
                all_weights_frozen=True,new_teacher_answers_or_training_or_RMOT_quality=0,
                pretraining_target_overlap_unknown=True,asset_probe_wall_seconds=time.perf_counter()-start)
    return result


def crop_tensor(image,box):
    """Keep the complete clipped ROI; every method receives these same inputs."""
    from PIL import Image
    import torch
    x,y,w,h=map(float,box)
    left=max(0,int(np.floor(x)));top=max(0,int(np.floor(y)))
    right=min(image.width,int(np.ceil(x+w)));bottom=min(image.height,int(np.ceil(y+h)))
    if right<=left or bottom<=top:return None
    cropped=image.crop((left,top,right,bottom)).resize((224,224),Image.Resampling.BICUBIC)
    array=np.asarray(cropped,dtype='float32')/255.
    cropped.close()
    array=(array-np.asarray([.485,.456,.406],dtype='float32'))/np.asarray([.229,.224,.225],dtype='float32')
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2,0,1)))
