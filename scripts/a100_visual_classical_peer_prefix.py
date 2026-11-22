"""Fixed-answer teacher-noise audit; exact NO_AUDIT regression is mandatory."""
import argparse
import json
import re
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import expit
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import PAIRS,metadata,FrozenInputs,generation_guard
from flex_uda.active.color_cohort import augmented_features,part_mask
from flex_uda.active.peer_evidence import audit_weights
from flex_uda.active.visual_classical import refine_distilled_prior
from flex_uda.active.acquisition import expression_types,exclusive_group
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_strict_prefix import plain,read_guard,immediate_predictions

POLICIES=('COVER','UNIFORM','FACILITY')
METHODS=('AUDIT','NO_AUDIT','UNIFORM_PRECISION','NO_COHORT','SHUFFLED_PRECISION')


def run(pair):
    guard=generation_guard();visible=set();forbidden,reads=read_guard(pair,visible)
    model,target=PAIRS[pair];m=metadata(target);root=OUT/'peer_prefix'/pair;begin=time.perf_counter();receipts=[]
    if (root/'completion.json').is_file():return
    for vi,video in enumerate(sorted(m['videos'])):
        visible.add(video);started=time.perf_counter();fc=FrozenInputs(pair,video)
        with np.load(OUT/'inputs'/pair/video/'inputs.npz') as a:original=a['features']
        with np.load(OUT/'color_features'/pair/video/'features.npz') as a:colour=a['colours']
        texts=[q['query_text'] for q in fc.queries];features,_,_=augmented_features(original,colour,texts)
        exclusive=exclusive_group(texts,target);first=np.asarray([fc.by_track[t][0] for t in fc.tracks])
        columns=np.asarray([fc.tracks.index(int(t)) for _,t in fc.observations]);audits={};outputs={}
        for policy in POLICIES:
            group='facility_prefix' if policy=='FACILITY' else 'colour_prefix'
            ready=OUT/group/pair/'videos'/video/'prefix_receipt.json'
            while not ready.is_file():time.sleep(5)
            upstream=json.loads(ready.read_text());assert upstream['visible_teacher_videos']==sorted(visible)
            source=OUT/group/pair/(policy+'__RGB_KD')/'videos'/video
            kd=np.load(source/'scores.npy')[:,first];teacher=np.full(kd.shape,np.nan)
            shard=json.loads((OUT/'causal_teacher'/pair/policy/(video+'.json')).read_text())
            for a in shard['answers']:teacher[int(a['local_query']),int(a['local_track'])]=a['yes_probability']
            scores={name:kd.copy() for name in METHODS};audits[policy]=[];outputs[policy]={}
            for qi,text in enumerate(texts):
                requested=part_mask(text)>0
                if not requested.any():continue
                valid=np.all(colour[:,requested].sum(2)>0,axis=1);seen=np.isfinite(teacher[qi])
                if re.search(r'\b(referee|goalkeeper|libero|setter|coach)\b',text.lower()):
                    factor=np.ones(len(seen));no_cohort=factor.copy()
                    record=dict(downweighted=0,supported=0,rows=[],skip_reason='named role cannot be certified by colour cohort alone')
                else:
                    factor,record=audit_weights(features[qi],teacher[qi],valid)
                    no_cohort,_=audit_weights(features[qi],teacher[qi],valid,use_cohorts=False)
                base=np.zeros(len(seen));base[seen]=1+3*np.abs(2*teacher[qi,seen]-1)
                uniform=np.ones(len(seen))
                if base.sum():uniform[:]=(base*factor).sum()/base.sum()
                shuffled=np.ones(len(seen));ids=np.flatnonzero(seen)
                shuffled[ids]=np.random.default_rng(vi*1009+qi).permutation(factor[ids])
                if base.sum():shuffled[ids]*=(base*factor).sum()/max((base*shuffled).sum(),1e-12)
                factors=dict(AUDIT=factor,NO_AUDIT=None,UNIFORM_PRECISION=uniform,NO_COHORT=no_cohort,SHUFFLED_PRECISION=shuffled)
                for name,value in factors.items():scores[name][qi]=refine_distilled_prior(kd[qi],features[qi],teacher[qi],
                    source_weight=.1,node_mask=valid,teacher_weight_factor=value)
                audits[policy].append(dict(query=qi,**record))
            for name,score in scores.items():
                expanded=score[:,columns];mask=plain(expanded,exclusive)
                for qi,q in enumerate(fc.queries):
                    if 'MOTION' in expression_types(q['query_text']):
                        p=expit(fc.raw_presence[qi]);mask[qi]&=np.asarray([p[fc.frame_index[int(f)]]>=.5 for f,_ in fc.observations])
                if name=='NO_AUDIT':
                    reference=np.load(OUT/group/pair/(policy+'__COHORT')/'videos'/video/'mask.npy')
                    assert np.array_equal(mask,reference),'NO_AUDIT must exactly reproduce the sealed cohort control'
                folder=root/(policy+'__'+name);local=folder/'videos'/video;local.mkdir(parents=True,exist_ok=True)
                np.save(local/'mask.npy',mask);np.save(local/'scores.npy',expanded.astype('float32'))
                outputs[policy][name]=immediate_predictions(folder,fc,mask)
        receipt=dict(video=video,visible_teacher_videos=sorted(visible),audits=audits,output_hashes=outputs,
            NO_AUDIT_exact_regression=True,GT_open_attempts=len(guard),wall_seconds=time.perf_counter()-started)
        atomic_json(root/'videos'/video/'receipt.json',receipt);receipts.append(receipt)
        print(json.dumps(dict(event='PEER_AUDIT_VIDEO',pair=pair,video=video,
            downweighted={p:sum(r['downweighted'] for r in rows) for p,rows in audits.items()})),flush=True)
    methods=[]
    for policy in POLICIES:
        for name in METHODS:
            folder=root/(policy+'__'+name);mask=np.zeros(m['candidate_query_rows'],dtype=bool)
            for video in sorted(m['videos']):
                local=np.load(folder/'videos'/video/'mask.npy')
                for qi,q in enumerate([q for q in m['episodes'] if q['video_id']==video]):mask[q['start']:q['stop']]=local[qi]
            source_group='facility_prefix' if policy=='FACILITY' else 'colour_prefix'
            source_seal=json.loads((OUT/source_group/pair/(policy+'__RGB_KD')/'predictions_seal.json').read_text())
            seal=seal_mask(folder,m,mask,target=target,model=model,method='PEER_'+policy+'__'+name,GT_open_attempts=len(guard),
                extra=dict(new_teacher_calls=0,new_neural_updates=0,same_questions_and_head_as_no_audit=True,
                    logical_teacher_calls=source_seal['logical_teacher_calls'],inherits_actual_teacher_KD_fit=True,
                    teacher_acquisition=policy,video_stream_causal=True,current_video_transductive=True,
                    no_future_teacher_feedback=True,uniform_control_matches_total_teacher_precision=True,
                    shuffled_relative_factors_preserve_total_precision=True,not_live_end_to_end_wall=True))
            methods.append(dict(method=policy+'__'+name,rows=seal['prediction_rows']))
    assert not guard and not forbidden
    atomic_json(root/'completion.json',dict(status='SEALED_PEER_TEACHER_NOISE_AUDIT',pair=pair,methods=methods,
        GT_open_attempts=0,forbidden_read_attempts=0,receipts=receipts,teacher_reads=reads,
        wall_seconds=time.perf_counter()-begin,all_NO_AUDIT_masks_identical_to_sealed_controls=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',required=True,choices=['C4','C1']);a=p.parse_args();run(a.pair)
