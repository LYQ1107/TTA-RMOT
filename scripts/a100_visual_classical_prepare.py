"""Freeze unlabelled inputs and first-round protocol before target evaluation."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import FrozenInputs,metadata,HISTORY,generation_guard,PAIRS
from flex_uda.active.visual_classical import visual_graph,acquisition_order

OUT=ROOT/'a100_runs/visual_classical_20261121'


def prepare(pair):
    attempts=generation_guard();begin=time.perf_counter();model,target=PAIRS[pair];m=metadata(target)
    destination=OUT/'inputs'/pair
    if (destination/'completion.json').is_file():
        print(json.dumps(dict(event='INPUTS_ALREADY_READY',pair=pair)),flush=True);return
    rows=[]
    for video in sorted(m['videos']):
        folder=destination/video
        if (folder/'receipt.json').is_file():
            rows.append(json.loads((folder/'receipt.json').read_text()));continue
        start=time.perf_counter();fc=FrozenInputs(pair,video);fraction,_,native=fc.decode()
        p=HISTORY/'features'/target/model/'videos'/video
        pcd=np.load(p/'pcd.npy',mmap_mode='r')
        index=np.load(p/'index.npy',mmap_mode='r')
        # Export is query-major; check its explicit (q,frame,track) index.
        feature_metadata=json.loads((p/'metadata.json').read_text())
        offsets=[];anchors=[];track_features=[];sample_features=[]
        nobs=len(fc.observations)
        nq=len(fc.queries)
        assert pcd.shape==(nq*nobs,768),(pcd.shape,nq,nobs)
        for track in fc.tracks:
            ids=fc.by_track[track]
            samples=ids[np.unique(np.linspace(0,len(ids)-1,min(3,len(ids))).round().astype(int))]
            for q in range(nq):
                assert np.array_equal(index[q*nobs+samples,0],np.full(len(samples),q))
                assert np.array_equal(index[q*nobs+samples,1:],fc.observations[samples])
            offsets.append(samples.tolist())
            boxes=np.asarray([[float(x) for x in fc.bank['lines'][i].split(',')[2:6]] for i in samples])
            anchor=int(samples[np.argmax(boxes[:,2]*boxes[:,3])])
            anchors.append(int(fc.observations[anchor,0]))
            feature=np.stack([np.asarray(pcd[q*nobs+samples],dtype='float32').mean(0) for q in range(nq)])
            track_features.append(feature)
        features=np.stack(track_features,axis=1)
        folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'inputs.npz',features=features,prior=fraction,
            tracks=fc.tracks,lengths=[len(fc.by_track[t]) for t in fc.tracks],anchors=anchors,
            query_ids=[q['expression_id'] for q in fc.queries])
        orders={}
        eligible=np.flatnonzero(np.asarray([len(fc.by_track[t]) for t in fc.tracks])>=3)
        for seed in (0,17,42):
            for method in ('PRIOR','UNIFORM','UNCERTAINTY','COVER'):
                selected=[]
                for q in range(nq):
                    _,distance=visual_graph(features[q])
                    rng=np.random.default_rng(seed+int(__import__('hashlib').sha256((video+str(q)).encode()).hexdigest()[:8],16))
                    selected.append(acquisition_order(fraction[q],distance,eligible,method,rng))
                orders[method+'__'+str(seed)]=selected
        atomic_json(folder/'orders.json',orders)
        row=dict(video=video,pair=pair,queries=nq,tracks=len(fc.tracks),eligible=len(eligible),
            logical_budget=int(np.floor(.3*nq*len(eligible))),source_pcd=str(p/'pcd.npy'),
            source_pcd_shape=list(pcd.shape),sample_observations=offsets,
            input_sha256=file_sha(folder/'inputs.npz'),GT_open_attempts=len(attempts),
            feature_preparation_wall_seconds=time.perf_counter()-start,
            historical_feature_extraction_cost_not_included=True)
        atomic_json(folder/'receipt.json',row);rows.append(row)
        print(json.dumps(dict(event='VISUAL_INPUT_VIDEO',pair=pair,video=video,seconds=row['feature_preparation_wall_seconds'])),flush=True)
    assert not attempts
    atomic_json(destination/'completion.json',dict(status='SEALED_UNLABELLED_INPUTS',pair=pair,videos=rows,
        wall_seconds=time.perf_counter()-begin,GT_open_attempts=0))


def register():
    OUT.mkdir(parents=True,exist_ok=True)
    plan=dict(version='VISUAL_CLASSICAL_V1',source_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),
        pairs=['C4','C1'],target_labels_in_method=False,GT_only_after_all_predictions_sealed=True,
        all_datasets_development=True,independent_blind_validation=False,
        scope='video-transductive target-image/query adaptation; no target labels; frozen source model and frozen teacher',
        budget_fraction=.3,seeds=[0,17,42],first_implementation_seed=0,
        sampling='per-query balanced budget; three frozen PCD observations uniformly spaced per track, largest area among them for teacher',
        acquisition=['PRIOR','UNIFORM','UNCERTAINTY','COVER'],teacher_view='CROP',teacher_threshold=.5,
        methods=['DIRECT_HARD','DIRECT_SOFT','KD_HARD','KD_SOFT','FULL','NO_CORRECTION','NO_GRAPH'],
        same_evidence_controls=True,cover_ablation='FULL inference on uniform and uncertainty acquisitions',
        classical=dict(graph='symmetric self-tuning cosine kNN, k=min(3,n-1)',
            source_precision=.1,teacher_precision='1+3*abs(2*p-1)',teacher_logit_cap=4.,
            teacher_probability_clip=[.02,.98],source_logit_cap=2.,
            graph_strength=1.,parameters_selected_using_GT=False),
        distillation=dict(actual_trainable_head=True,frozen_PCD_features=True,
            architecture='768 ->64 GELU ->1 residual on source logit',optimizer='AdamW',
            learning_rate=.001,weight_decay=.001,steps=400,loss='BCE hard or raw soft teacher targets',
            no_confidence_screening=True,no_target_GT_selection=True),
        protocol=dict(C4='PRESUPPOSED',C1='CHECKLIST',new_early_absence_stop=False),
        comparison=dict(fixed_evidence=True,CountMatched=True,official_HOTA=True,
            original_parent_group_bootstrap=10000,real_wall_cost_separate=True,
            teacher_pool_preparation_not_live_method_cost=True),
        acceptance='full method better than strongest direct-pseudo/KD baseline with transparent module ablations, quantity controls and cross-direction limits',
        physical_GPUs=[0,1,2,3],NAS_write=False)
    path=OUT/'preregistration_v1.json'
    if path.exists():
        frozen=json.loads(path.read_text())
        # A later code commit must not rewrite the already sealed protocol or
        # invalidate its SHA in the collected evidence. Compare scientific
        # fields and retain the original commit as provenance.
        assert {k:v for k,v in frozen.items() if k!='source_commit'}=={k:v for k,v in plan.items() if k!='source_commit'}
    else:atomic_json(path,plan)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=['C4','C1']);a=p.parse_args()
    register()
    if a.pair:prepare(a.pair)
