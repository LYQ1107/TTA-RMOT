"""Official complete-source supervision, with causal four-observation inputs."""
from collections import OrderedDict,defaultdict
import json
from pathlib import Path
import torch
import torchvision.transforms as T
from flex_tta.runtime.dance_episode import DanceEpisode,Observation
from .manifests import SOURCE,read_jsonl


class SourceEpisode(DanceEpisode):
    def __init__(self,video,labels,root=SOURCE):
        self.video=video;self.image_root=Path(root)/'KITTI/training/image_02'/video
        self.observations_by_frame=defaultdict(dict);self.observations_by_track=defaultdict(list)
        self.membership={};self.positive_frames=defaultdict(set)
        for track,frames in labels.items():
            for frame,item in frames.items():
                # Same official eligibility: do not invent negatives for rows
                # lacking usable expression annotations.
                expressions=item.get('expression_raw',[])
                if not expressions:continue
                t=int(frame);i=int(track);self.membership[t,i]=set(expressions)
                for expression in expressions:self.positive_frames[expression].add(t)
                self.observations_by_frame[t][i]=item['bbox']
        self.frames=sorted(self.observations_by_frame)
        self.width,self.height=self._read_resolution()
        for t,rows in self.observations_by_frame.items():
            for i,box in list(rows.items()):
                x,y,w,h=box
                obs=Observation(t,i,(x*self.width,y*self.height,(x+w)*self.width,(y+h)*self.height),'')
                rows[i]=obs;self.observations_by_track[i].append(obs)
        self.observations_by_frame={t:dict(sorted(rows.items())) for t,rows in sorted(self.observations_by_frame.items())}
        self.observations_by_track={i:sorted(rows,key=lambda o:o.frame_id) for i,rows in self.observations_by_track.items()}
        self.track_index={i:{o.frame_id:j for j,o in enumerate(rows)} for i,rows in self.observations_by_track.items()}
        self.image_cache=OrderedDict();self.image_cache_size=12
        self.transform=T.Compose([T.Resize((224,672)),T.ToTensor(),T.Normalize(
            (.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])

    def labels_for(self,candidates,sentence,device):
        return torch.tensor([int(sentence in self.membership[hs[-1].frame_id,i]) for i,hs in candidates],device=device)


class SourceSupervision:
    def __init__(self,manifest,root=SOURCE):
        self.root=Path(root);self.rows=read_jsonl(manifest)
        self.labels=json.loads((self.root/'labels.json').read_text());self.episodes=OrderedDict()

    def episode(self,video):
        if video not in self.episodes:
            self.episodes[video]=SourceEpisode(video,self.labels[video],self.root)
            while len(self.episodes)>2:self.episodes.popitem(last=False)
        self.episodes.move_to_end(video);return self.episodes[video]

    def plan(self,*,validation_queries_per_video=None):
        records=[];counts=[]
        for row in self.rows:
            video=row['video_id'];episode=self.episode(video)
            files=sorted((self.root/'expression'/video).glob('*.json'))
            if validation_queries_per_video and len(files)>validation_queries_per_video:
                files=[files[round(j*(len(files)-1)/(validation_queries_per_video-1))] for j in range(validation_queries_per_video)]
            chosen_tracks=set();missing_positive=0
            for path in files:
                payload=json.loads(path.read_text());sentence=payload['sentence']
                positives=sorted(int(t) for t,ids in payload.get('label',{}).items()
                    if int(t) in episode.observations_by_frame and
                    any(int(i) in episode.observations_by_frame[int(t)] for i in ids))
                frame=positives[len(positives)//2] if positives else episode.frames[len(episode.frames)//2]
                if not positives:missing_positive+=1
                record=dict(video_id=video,expression_id=path.stem,query_text=sentence,frame=frame,
                    confirmed_video_absent=not any(payload.get('label',{}).values()) and not positives,
                    positive_track_ids=[int(i) for i in payload.get('label',{}).get(str(frame),[])])
                records.append(record);chosen_tracks.update(episode.observations_by_frame[frame])
            # Ensure the source plan represents every eligible labelled track,
            # not just the median frame of frequent query paraphrases.
            if validation_queries_per_video is None:
                for i,track in episode.observations_by_track.items():
                    if i in chosen_tracks:continue
                    obs=track[len(track)//2];sentence=sorted(episode.membership[obs.frame_id,i])[0]
                    records.append(dict(video_id=video,expression_id='track_coverage_'+str(i),query_text=sentence,
                        frame=obs.frame_id,confirmed_video_absent=False,extra_source_track_coverage=True,
                        positive_track_ids=[j for j in episode.observations_by_frame[obs.frame_id]
                            if sentence in episode.membership[obs.frame_id,j]]))
                    chosen_tracks.update(episode.observations_by_frame[obs.frame_id])
            counts.append(dict(video_id=video,provided_expressions=len(files),eligible_GT_tracks=len(episode.observations_by_track),
                sampled_GT_tracks=len(chosen_tracks),queries_without_retrievable_source_positive=missing_positive))
        return records,counts
