"""Post-seal GT diagnosis only; never imported by adaptation or selection."""
import json
from collections import Counter,defaultdict
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import PAIRS,metadata,OUT as OLD_DIAG
from scripts.a100_visual_classical_prepare import OUT


def run():
    for pair in ('C4','C1'):
        assert (OUT/'predictions'/pair/'frame_kd_completion.json').is_file()
        assert (OUT/'countmatched'/pair/'completion.json').is_file()
        assert (OUT/'convergence'/pair/'completion.json').is_file()
    output={}
    for pair in ('C4','C1'):
        _,target=PAIRS[pair];m=metadata(target);total=defaultdict(Counter);types=defaultdict(lambda:defaultdict(Counter));queries=[]
        teacher=Counter();source_methods={name:(OUT/'predictions'/pair/('COVER__'+name)) for name in
            ('FULL','NO_GRAPH','NO_CORRECTION','DIRECT_SOFT','FRAME_KD_HARD','FRAME_KD_SOFT','FRAME_KD_SOFT_STRUCTURED')}
        source_methods.update({name:(OUT/'convergence'/pair/('COVER__'+name)) for name in ('KD_HARD_4000','KD_SOFT_4000')})
        responses=[json.loads(line) for line in (OUT/'evidence'/pair/'responses.jsonl').read_text().splitlines()]
        pervideo=defaultdict(list)
        for row in responses:pervideo[row['video_id']].append(row)
        for video,bank in sorted(m['videos'].items()):
            truth=np.load(OLD_DIAG/'offline_GT'/pair/video/'labels.npy')>=0
            union=np.load(OLD_DIAG/'offline_GT'/pair/video/'union_ids.npy')
            obs=np.asarray(bank['observations']);tracks=sorted(set(obs[:,1].tolist()))
            qrows=[q for q in m['episodes'] if q['video_id']==video]
            positions={(int(f),int(t)):i for i,(f,t) in enumerate(obs)}
            for response in pervideo[video]:
                q=int(response['local_query']);i=positions[int(response['frame_id']),int(response['track_id'])]
                y=bool(truth[q,i]);p=float(response['yes_probability']);pred=p>=.5
                teacher['observations']+=1;teacher['positive']+=y;teacher['correct']+=pred==y
                teacher['TP']+=pred and y;teacher['FP']+=pred and not y;teacher['FN']+=not pred and y
            masks={name:np.load(path/'videos'/video/'mask.npy') for name,path in source_methods.items()}
            for q,query in enumerate(qrows):
                y=truth[q];kind=('RELATIVE_LIGHT_DARK' if any(w in query['query_text'].lower() for w in ('light-side','dark-side')) else
                    'COLOR' if any(w in query['query_text'].lower() for w in ('white','black','blue','purple','brown','green','gray')) else 'OTHER')
                for name,mask in masks.items():
                    prediction=mask[q];counts=Counter(TP=int((prediction&y).sum()),FP=int((prediction&~y).sum()),FN=int((~prediction&y).sum()),
                        selected_rows=int(prediction.sum()),queries=1,present=int(y.any()),present_zero=int(y.any() and not prediction.any()))
                    total[name].update(counts);types[kind][name].update(counts)
                    queries.append(dict(video=video,query=query['expression_id'],query_text=query['query_text'],method=name,kind=kind,**dict(counts)))
            for track in tracks:
                ids=np.flatnonzero(obs[:,1]==track);adjacent=(np.diff(obs[ids,0])>0)&(np.diff(obs[ids,0])<=2)
                same=(union[ids[1:]]==union[ids[:-1]])&(union[ids[1:]]>=0)
                for q in range(len(qrows)):
                    y=truth[q,ids];stable=adjacent&same&(y[1:]==y[:-1]);change=adjacent&same&(y[1:]!=y[:-1])
                    for name,mask in masks.items():
                        bits=mask[q,ids];flip=bits[1:]!=bits[:-1]
                        total[name]['stable_identity_membership_transitions']+=int(stable.sum())
                        total[name]['stable_identity_membership_flips']+=int((stable&flip).sum())
                        total[name]['GT_membership_changes_same_visible_ID']+=int(change.sum())
                        total[name]['query_track_pairs_with_both_memberships']+=bool(y.any() and not y.all())
        output[pair]=dict(teacher_point_diagnostic=dict(teacher),methods={k:dict(v) for k,v in total.items()},
            query_types={k:{n:dict(v) for n,v in values.items()} for k,values in types.items()},queries=queries,
            scope='IoU-assigned candidate membership diagnosis, not official HOTA or pure recognition accuracy; GT evaluation only',
            no_method_parameter_selected_in_this_script=True)
    atomic_json(OUT/'diagnosis/v1_membership.json',output)
    print(json.dumps({p:dict(teacher=r['teacher_point_diagnostic'],methods=r['methods']) for p,r in output.items()}),flush=True)


if __name__=='__main__':run()
