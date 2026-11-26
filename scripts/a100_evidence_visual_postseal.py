"""Explain both complete studies, never adapt on their GT or choose winners."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.a100_evidence_visual_primary import OUT,PUBLIC,STEM,PRIMARY,METHODS,registration
from scripts.a100_evidence_visual_capacity_controls import OUT as CAPACITY,STEM as CAP_STEM,COMPARATORS


def confusion(truth,prediction):
    return dict(TP=int((truth&prediction).sum()),FP=int((~truth&prediction).sum()),
        FN=int((truth&~prediction).sum()),TN=int((~truth&~prediction).sum()))


def official(folder):
    with (folder/'results/pedestrian_detailed.csv').open() as stream:
        rows=[r for r in csv.DictReader(stream) if r['seq']!='COMBINED']
    return {k:sum(float(r['HOTA_'+k+'___50']) for r in rows) for k in ('TP','FP','FN')}


def main():
    torch.set_num_threads(2);plan=registration()
    for stem,records,effects in ((STEM,75,72),(CAP_STEM,21,18)):
        rep=json.loads((PUBLIC/(stem+'_reproduction.json')).read_text())
        assert rep['status'].startswith('PASS_') and rep['records']==records and rep['effects']==effects
    from flex_uda.a100.mechanism import OUT as DIAGNOSTIC,metadata
    from scripts.a100_observation_evidence_transfer_diagnose import CANONICAL
    from flex_uda.active.evidence_visual_learning import data,make,content_loss
    reports=[]
    for case in plan['cases']:
        pair,video=case['pair'],case['evaluated_video'];folder=OUT/pair
        truth=np.load(DIAGNOSTIC/'offline_GT'/CANONICAL[pair]/video/'labels.npy')>=0
        md=metadata(case['target']);obs=np.asarray(md['videos'][video]['observations'],dtype=int)
        queries=[q for q in md['episodes'] if q['video_id']==video]
        assert truth.shape==(len(queries),len(obs))
        f,qindex,doc,quotas,auxiliary,ay,mean,std=data(case)
        lookup={tuple(map(int,r)):i for i,r in enumerate(obs)};nodes=[]
        for uid in doc['order']:
            a=doc['actions'][uid];r=a['case'];q=a['local_query']
            assert queries[q]['expression_id']==r['query_id']
            nodes.append(q*len(obs)+lookup[int(r['frame_id']),int(r['track_id'])])
        nodes=np.asarray(nodes);pool_truth=truth.flatten()[nodes]
        draws=json.loads((folder/'primary/draws.json').read_text())['draws']
        uniform=json.loads((folder/'uniform_draws.json').read_text())['draws']
        masks={name:np.load(folder/'predictions'/name/'mask.npy') for name in METHODS}
        masks.update({name:np.load(CAPACITY/pair/'predictions'/name/'mask.npy') for name in COMPARATORS})
        records={}
        for name,mask in masks.items():
            path=(CAPACITY if name in COMPARATORS else OUT)/pair/'predictions'/name
            records[name]=dict(canonical=confusion(truth,mask),official_IoU50=official(path),
                positive_queries_zero_output=int((truth.any(1)&~mask.any(1)).sum()))
        coverage={}
        for name,selected in (('primary',draws),('uniform',uniform)):
            unique={r['index']:r for r in selected};asked=np.zeros(len(qindex),dtype=bool);asked[list(unique)]=True
            ty=np.asarray([unique[i]['probability']>.5 for i in unique]);tt=np.asarray([pool_truth[i] for i in unique])
            coverage[name]=dict(logical_draws=len(selected),unique_actions=len(unique),repeated_logical_cache_actions=len(selected)-len(unique),
                teacher_nominal_anchor_counts=confusion(tt,ty),
                known_teacher_class_disagreements=int(sum(masks[PRIMARY].flatten()[nodes[i]]!=(unique[i]['probability']>.5) for i in unique)) if name=='primary' else None,
                positive_pool_queries_never_asked_correct=int(sum(pool_truth[qindex==q].any() and not (pool_truth&asked)[qindex==q].any() for q in quotas)),
                positive_queries_without_correct_action_in_old_pool=int(sum(truth[q].any() and not pool_truth[qindex==q].any() for q in quotas)))
        perquery=[]
        for q,query in enumerate(queries):
            paid=[r for r in {r['index']:r for r in draws}.values() if r['query']==q]
            perquery.append(dict(query_id=query['expression_id'],query_text=query['query_text'],canonical_positive_observations=int(truth[q].sum()),
                unique_paid=len(paid),teacher_Yes_count=sum(r['probability']>.5 for r in paid),
                paid_teacher_Yes_without_canonical_match=sum(r['probability']>.5 and not pool_truth[r['index']] for r in paid),
                primary=confusion(truth[q],masks[PRIMARY][q]),soft=confusion(truth[q],masks['SOFT_KD'][q]),
                uniform=confusion(truth[q],masks['UNIFORM_SOFT_KD'][q])))
        checkpoint=torch.load(folder/'primary/closed.pt',map_location='cpu',weights_only=False)
        final=torch.load(folder/'primary/final.pt',map_location='cpu',weights_only=False)
        with np.load(case['pixel_pairs_path']) as p:pixel={k:torch.from_numpy(p[k].copy()) for k in ('left','right','weight')}
        representations={}
        with torch.no_grad():
            for name,key,weights in (('S_bootstrap','fixed_s',checkpoint),('R_bootstrap','fixed_r',checkpoint),('S_final','solver',final),('R_final','observer',final)):
                outputs=2 if name.startswith('R') else 1;model=make(mean,std,'cpu',outputs=outputs);model.load_state_dict(weights[key])
                a,b=model.app(pixel['left']),model.app(pixel['right'])
                representations[name]=dict(image_pretext=float(content_loss(model,pixel)),
                    positive_pair_MSE=float((a-b).square().mean()),mean_content_std=float(a.std(0).mean()))
        record=dict(pair=pair,methods=records,coverage=coverage,content_diagnostics=representations,
            worst_primary_minus_uniform_FP_queries=sorted(perquery,key=lambda r:r['primary']['FP']-r['uniform']['FP'],reverse=True)[:5],
            primary_teacher_KL=final['proper_known_teacher_KL'],all_original_targets_preserved=final['original_teacher_targets_unchanged'],
            no_signed_semantic_targets=final['no_signed_semantic_targets'])
        reports.append(record);atomic_json(folder/'postseal_query_diagnosis.json',dict(queries=perquery,**record))
        print(json.dumps(dict(status='NEW_VISUAL_PRIMARY_POSTSEAL_CASE_DIAGNOSED',pair=pair,coverage=coverage,
            primary=records[PRIMARY],soft=records['SOFT_KD'],same_arch=records['SAME_ARCH_KD128'],content=representations)),flush=True)
    result=dict(status='COMPLETE_NEW_VISUAL_PRIMARY_AND_CAPACITY_POSTSEAL_DIAGNOSIS',cases=reports,
        source_sha256=file_sha(__file__),new_fits_teacher_backbone_predictions_or_seeds=0,
        GT_only_after_both_studies_complete=True,canonical_not_pure_teacher_semantic_accuracy=True,
        content_pretext_change_not_a_module_quality_ablation=True,
        official_IoU50_not_additive_decomposition_of19threshold_HOTA=True,all_data_DEV=True,goal_status='ACTIVE')
    atomic_json(OUT/'postseal_diagnosis.json',result);atomic_json(PUBLIC/(STEM+'_postseal_diagnosis.json'),result)


if __name__=='__main__':main()
