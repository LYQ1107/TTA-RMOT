"""Same legal ROI/history for all methods, paired target/background interventions."""
from collections import OrderedDict
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from flex_tta.runtime.dance_episode import DanceEpisode


class VisualInputBuilder:
    def __init__(self,stage,upstream,image_size=320,margin_ratio=0.2):
        self.stage=Path(stage);self.upstream=upstream;self.image_size=image_size;self.margin_ratio=margin_ratio
        self.episodes={};self.cache=OrderedDict();self.frame_cache=OrderedDict()
        self.patch_limit=1024;self.frame_limit=128
        self.cache_counts=dict(frame_decodes=0,frame_hits=0,patch_hits=0,patch_builds=0,full_frame_mean_computes=0)

    def frame(self,video,frame):
        key=(video,frame)
        if key in self.frame_cache:
            value=self.frame_cache.pop(key);self.frame_cache[key]=value;self.cache_counts['frame_hits']+=1
            return value
        path=self.stage/'DanceTrack/training/image_02'/video/f'{frame:06d}.png'
        with Image.open(path) as image:value=[image.convert('RGB'),None]
        self.cache_counts['frame_decodes']+=1;self.frame_cache[key]=value
        while len(self.frame_cache)>self.frame_limit:self.frame_cache.popitem(last=False)
        return value

    def cache_stats(self):
        return dict(mode='READONLY_RGB_PATCH_SHA_LRU_V2',**self.cache_counts,
            decoded_frame_limit=self.frame_limit,resized_patch_limit=self.patch_limit,
            live_decoded_RGB_bytes=sum(frame[0].width*frame[0].height*3 for frame in self.frame_cache.values()),
            live_patch_RGB_bytes=sum(item[0].width*item[0].height*3 for item in self.cache.values()))

    def episode(self,video):
        if video not in self.episodes:
            self.episodes[video]=DanceEpisode(self.stage,self.stage/'tracker_outputs',video)
        return self.episodes[video]

    def patch(self,video,observation,variant='original'):
        key=(video,observation.frame_id,observation.track_id,variant)
        if key in self.cache:
            value=self.cache.pop(key);self.cache[key]=value;self.cache_counts['patch_hits']+=1
            return value[0].copy()
        frame=self.frame(video,observation.frame_id);full=frame[0]
        x1,y1,x2,y2=observation.bbox_xyxy;box=(x1,y1,x2-x1,y2-y1)
        bounds=self.upstream['xywh_to_xyxy_with_margin'](box,full.size,self.margin_ratio,None,8)
        crop=self.upstream['crop_with_margin'](full,box,self.margin_ratio,None,8)
        if crop is None:raise ValueError('fixed MOT has an invalid visible ROI; do not synthesize negative evidence')
        if variant in ('candidate_occluded','background_occluded'):
            pixels=np.asarray(crop).copy();h,w=pixels.shape[:2]
            local=(max(0,int(x1-bounds[0])),max(0,int(y1-bounds[1])),min(w,int(x2-bounds[0])),min(h,int(y2-bounds[1])))
            lx,ly,rx,ry=local
            body=np.zeros((h,w),dtype=bool)
            bh=max(1,int((ry-ly)*0.55));bw=max(1,int((rx-lx)*0.55))
            cx=(lx+rx)//2;cy=(ly+ry)//2
            body[max(ly,cy-bh//2):min(ry,cy-bh//2+bh),max(lx,cx-bw//2):min(rx,cx-bw//2+bw)]=True
            background=np.ones((h,w),dtype=bool);background[ly:ry,lx:rx]=False
            count=min(int(body.sum()),int(background.sum()))
            if count==0:raise ValueError('no matched background intervention is possible for this ROI')
            mask=np.zeros((h,w),dtype=bool)
            choices=np.flatnonzero(body if variant=='candidate_occluded' else background)[:count]
            mask.flat[choices]=True
            if frame[1] is None:
                frame[1]=np.asarray(full).reshape(-1,3).mean(0).astype(np.uint8)
                self.cache_counts['full_frame_mean_computes']+=1
            pixels[mask]=frame[1]
            crop=Image.fromarray(pixels)
        crop=crop.resize((self.image_size,self.image_size))
        self.cache_counts['patch_builds']+=1
        self.cache[key]=(crop,hashlib.sha256(crop.tobytes()).hexdigest())
        while len(self.cache)>self.patch_limit:self.cache.popitem(last=False)
        return crop.copy()

    def build(self,sample,view,variant='original',frame=None):
        ep=self.episode(sample['video_id']);t=sample['frame_id'] if frame is None else frame
        tid=sample['track_id'];history=ep.history(tid,t)
        if any(o.frame_id>sample['frame_id'] for o in history):raise ValueError('future observation in teacher input')
        observation=history[-1];x1,y1,x2,y2=observation.bbox_xyxy
        coord=f'{(x1+x2)/2/ep.width:.3f} {(y1+y2)/2/ep.height:.3f}'
        if variant=='no_image':return [],[],coord,dict(variant=variant,paired_intervention=True)
        if variant=='previous_observation':
            unique=[o for i,o in enumerate(history) if i==0 or o.frame_id!=history[i-1].frame_id]
            if len(unique)<2:return [],[],coord,dict(variant=variant,missing='TEMPORAL_EVIDENCE_MISSING')
            observation=unique[-2]
            history=ep.history(tid,observation.frame_id)
            x1,y1,x2,y2=observation.bbox_xyxy
            coord=f'{(x1+x2)/2/ep.width:.3f} {(y1+y2)/2/ep.height:.3f}'
        selected=history if view=='history' else [observation]
        images=[self.patch(sample['video_id'],o,variant) for o in selected]
        return images,[o.frame_id for o in selected],coord,dict(variant=variant,view=view,
            frames=[o.frame_id for o in selected],unique_frames=len({o.frame_id for o in selected}),
            boxes=[list(o.bbox_xyxy) for o in selected],
            image_sha256=[self.cache[(sample['video_id'],o.frame_id,o.track_id,variant)][1] for o in selected])
