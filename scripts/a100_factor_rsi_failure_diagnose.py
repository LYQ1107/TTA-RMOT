"""Post-seal descriptive identity/state diagnosis of the rejected primary.

No new predictions, filters, fits or HOTA evaluations. GT never leaves this
offline process. Temporal annotation semantics and candidate IoU matching
are reported separately; neither is an additive HOTA error decomposition.
"""
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.a100_factor_rsi_primary_quality import OUT,PRIMARY,COMPARATORS,registration


def main():
    from flex_uda.a100.mechanism import metadata
    from scripts.a100_joint_observation_stability import DEST as OLD_DIAG,count_transitions
    plan,q=registration();quality=json.loads((OUT/'quality_results.json').read_text())
    reproduced=json.loads((ROOT/'docs/a100_six_direction_20261122/factor_rsi_primary_pilot_20261125_reproduction.json').read_text())
    assert reproduced['records']==39 and reproduced['maximum_absolute_error']<1e-8
    result={};detail=[]
    for case in plan['cases']:
        pair,target,video=case['pair'],case['target'],case['video'];md=metadata(target);bank=md['videos'][video]
        queries=[r for r in md['episodes'] if r['video_id']==video];obs=np.asarray(bank['observations'])
        cache=OLD_DIAG/'offline_GT'/target/video;receipt=json.loads((cache/'receipt.json').read_text())
        signature=hashlib.sha256(json.dumps(dict(observations=bank['observations'],lines=bank['lines'],
            queries=[r['expression_id'] for r in queries]),separators=(',',':')).encode()).hexdigest()
        assert receipt['candidate_query_signature']==signature
        truths=[];union=defaultdict(set);hashes=[]
        for query in queries:
            path=OUT/pair/'predictions'/PRIMARY/'results'/video/query['expression_id']/'gt.txt'
            data=path.read_bytes();hashes.append(hashlib.sha256(data).hexdigest());truth=defaultdict(set)
            for line in data.decode().splitlines():
                v=line.split(',')
                if int(float(v[6]))==0 or int(float(v[7]))!=1:continue
                t,g=int(float(v[0])),int(float(v[1]));truth[t].add(g);union[t].add(g)
            truths.append(truth)
        assert hashlib.sha256('\n'.join(hashes).encode()).hexdigest()==receipt['ordered_GT_hash_list_sha256']
        labels=np.load(cache/'labels.npy');gids=np.load(cache/'union_ids.npy');positive=labels>=0
        assert positive.shape==(len(queries),len(obs))
        # Raw GT-ID membership conditioned on that ID's visibility in ANY
        # query annotation. This avoids candidate-box IoU threshold flicker.
        visible=defaultdict(list)
        for frame,ids in union.items():
            for gid in ids:visible[gid].append(frame)
        groups=Counter();annotation_edges=Counter()
        for truth in truths:
            for gid,frames in visible.items():
                frames=sorted(frames);membership=np.asarray([gid in truth.get(t,set()) for t in frames])
                hits=int(membership.sum());groups['query_identity_groups']+=1
                groups['positive_groups']+=int(hits>0)
                groups['always_positive_groups']+=int(hits==len(frames))
                groups['mixed_positive_negative_groups']+=int(0<hits<len(frames))
                valid=np.diff(frames)<=2
                annotation_edges['visible_identity_edges']+=int(valid.sum())
                annotation_edges['membership_changes']+=int(((membership[:-1]!=membership[1:])&valid).sum())
        left=[];right=[];purity=[]
        for track in np.unique(obs[:,1]):
            ids=np.flatnonzero(obs[:,1]==track);ids=ids[np.argsort(obs[ids,0],kind='stable')]
            a,b=ids[:-1],ids[1:];gap=obs[b,0]-obs[a,0]
            valid=(gap>0)&(gap<=2)&(gids[a]>=0)&(gids[a]==gids[b])
            left.extend(a[valid]);right.extend(b[valid])
            counts=Counter(int(v) for v in gids[ids] if v>=0)
            if counts:purity.append(dict(track=int(track),matched_rows=sum(counts.values()),purity=max(counts.values())/sum(counts.values())))
        left,right=np.asarray(left,dtype=int),np.asarray(right,dtype=int)
        methods={}
        for name in (PRIMARY,*COMPARATORS):
            root=OUT/pair/'predictions'/name;bits=np.load(root/'mask.npy')
            assert bits.dtype==bool and bits.shape==positive.shape
            row=count_transitions(bits,positive,left,right,np.ones(len(queries),bool))
            with (root/'results/pedestrian_detailed.csv').open() as f:combined=next(r for r in csv.DictReader(f) if r['seq']=='COMBINED')
            methods[name]=dict(row,stable_output_flip_rate=row['stable_output_flips']/max(row['stable_transitions'],1),
                stable_endpoint_error_rate=row['stable_endpoint_errors']/max(2*row['stable_transitions'],1),
                HOTA=quality['results'][pair][name]['HOTA_unrounded'],
                DetA=100*float(combined['DetA___AUC']),AssA=100*float(combined['AssA___AUC']))
        result[pair]=dict(video=video,raw_annotation_membership=dict(groups),annotation_edges=dict(annotation_edges),
            matched_physical_identity_edges=len(left),physical_track_purity=purity,methods=methods,
            same_actual_official_GT_as_verified_old_cache=True,cache=str(cache),GT_file_hash_list_sha256=receipt['ordered_GT_hash_list_sha256'])
        print(json.dumps(dict(pair=pair,annotation_groups=dict(groups),annotation_edges=dict(annotation_edges),
            stable_output_flip_rates={n:r['stable_output_flip_rate'] for n,r in methods.items()})),flush=True)
    receipt=dict(status='POSTSEAL_FACTOR_PRIMARY_IDENTITY_STATE_DIAGNOSIS',settings=result,
        all39_evaluated_and_reproduced_before_current_GT=True,diagnostic_IoU=.5,
        annotation_scope='Per GT-ID visibility in union of query annotations; not full raw detector dataset.',
        no_new_models_teacher_calls_predictions_or_evaluations=True,
        no_GT_labels_thresholds_or_checkpoint_choices_written_to_adaptation=True,
        diagnostic_not_causal_or_HOTA_additive=True,all_data_DEV=True,goal_status='ACTIVE')
    atomic_json(OUT/'failure_identity_state_diagnosis.json',receipt)
    atomic_json(ROOT/'docs/a100_six_direction_20261122/factor_rsi_primary_identity_state_diagnosis_20261125.json',receipt)


if __name__=='__main__':main()
