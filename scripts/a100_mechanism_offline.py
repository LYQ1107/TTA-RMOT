"""Independent offline truth diagnostics. Never imported by teacher collection."""
import argparse
from collections import Counter, defaultdict
import contextlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'vendor/FlexHook/TrackEval'))
import compat.sitecustomize
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, append_json, file_sha
from flex_uda.a100.runtime import relocate
from flex_uda.a100.mechanism import OUT, OLD, PAIRS, HISTORY, metadata, history, FrozenInputs, baseline_mask
from scripts.analyze_pf_full_offline import load_eval_gt, overlaps, matched_pairs


def gt_root(target):
    if target == 'SportsMOT':
        d = json.loads((ROOT / 'results/multidomain_generality/sports_GT_reconstruction_v4.json').read_text())
        return relocate(d['evaluation_root'])
    if target == 'Dance':
        return relocate('/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance')
    return relocate('/data3/user/TTA-RMOT-assets/flexhook/refer-kitti-v2')


def evaluate(folder, pair, oracle=False, *, subset=False):
    from flex_uda.presence.predictions import verify_predictions
    folder = Path(folder)
    if (folder / 'official_completion.json').is_file():
        return json.loads((folder / 'official_completion.json').read_text())
    seal = json.loads((folder / 'predictions_seal.json').read_text()) if oracle else verify_predictions(folder)
    if oracle:
        assert seal['GT_open_attempts'] > 0 and seal['offline_oracle_uses_GT']
        for r in seal['episodes']:
            assert file_sha(r['prediction_path']) == r['prediction_sha256']
    model, target = PAIRS[pair]; m = metadata(target); truth = gt_root(target) / 'gt_template'
    seq_info = {}
    sealed_episodes={(r['video'],r['expression']) for r in seal['episodes']}
    for q in m['episodes']:
        if subset and (q['video_id'],q['expression_id']) not in sealed_episodes:continue
        v, e = q['video_id'], q['expression_id']; seq_info[v+'+'+e] = int(q['frame_count'])
        p = folder / 'results' / v / e / 'gt.txt'
        if not p.exists(): p.symlink_to(truth / v / e / 'gt.txt')
    import trackeval
    cfg = dict(GT_FOLDER=str(truth),TRACKERS_FOLDER=str(folder/'results'),TRACKERS_TO_EVAL=[str(folder/'results')],
        SKIP_SPLIT_FOL=True,GT_LOC_FORMAT='{gt_folder}/{video_id}/{expression_id}/gt.txt',SEQ_INFO=seq_info,PRINT_CONFIG=False)
    ec = dict(USE_PARALLEL=True,NUM_PARALLEL_CORES=2,PRINT_RESULTS=False,PRINT_ONLY_COMBINED=True,PRINT_CONFIG=False,
        TIME_PROGRESS=False,DISPLAY_LESS_PROGRESS=True,OUTPUT_SUMMARY=True,OUTPUT_DETAILED=True,PLOT_CURVES=False,
        BREAK_ON_ERROR=True,RETURN_ON_ERROR=False,LOG_ON_ERROR=str(folder/'trackeval_error.log'))
    begin = time.perf_counter()
    with (folder/'trackeval.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        trackeval.Evaluator(ec).evaluate([trackeval.datasets.MotChallenge2DBox(cfg)],
            [c(dict(PRINT_CONFIG=False,THRESHOLD=.5)) for c in (trackeval.metrics.HOTA,trackeval.metrics.CLEAR,trackeval.metrics.Identity)])
    from scripts.pf_diag_common import official_metrics
    from scripts.active_eval import official_stats
    stats=official_stats(folder); values=official_metrics(folder/'results')
    assert int(values['Dets']) == seal['prediction_rows']
    result=dict(status='PASS',pair=pair,metrics=values,HOTA_unrounded=stats['HOTA_unrounded'],
        offline_oracle_uses_GT=oracle,wall_seconds=time.perf_counter()-begin,
        sealed_episode_subset_only=subset,evaluated_episodes=len(seq_info),
        prediction_seal_sha256=file_sha(folder/'predictions_seal.json'),detailed_sha256=file_sha(folder/'results/pedestrian_detailed.csv'))
    atomic_json(folder/'official_completion.json',result)
    print(json.dumps(dict(event='MECHANISM_OFFICIAL_DONE',folder=str(folder),HOTA=stats['HOTA_unrounded'])),flush=True)
    return result


def audit(pair):
    from flex_uda.active.acquisition import expression_types
    from flex_uda.presence.predictions import seal_mask
    if pair=='C5':
        return audit_cached_kitti()
    begin = time.perf_counter(); model,target=PAIRS[pair]; m=metadata(target)
    folder=OUT/'audit'/pair; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completion.json').is_file():
        print(json.dumps(dict(event='AUDIT_ALREADY_COMPLETE',pair=pair)),flush=True);return
    gtbase=gt_root(target); totals=Counter(); episodes=[]; videos=[]
    masks={name:np.zeros(m['candidate_query_rows'],dtype=bool) for name in ('FRAME_ORACLE','TRACK_MAJORITY_ORACLE')} if pair=='C4' else {}
    for video,bank in sorted(m['videos'].items()):
        fc=FrozenInputs(pair,video); obs=fc.observations; baseline=baseline_mask(pair,video)
        assert baseline.shape == fc.margin.shape
        _,_,initial=fc.decode(); raw=fc.margin>0
        boxes=np.asarray([[float(x) for x in line.split(',')[2:6]] for line in bank['lines']])
        boxes[:,2:]+=boxes[:,:2]
        eval_frames=np.asarray([int(float(line.split(',')[0])) for line in bank['lines']])
        by_frame={int(f):np.flatnonzero(eval_frames==f) for f in np.unique(eval_frames)}
        truths=[load_eval_gt(gtbase,video,q['expression_id']) for q in fc.queries]
        union=defaultdict(dict)
        for truth in truths:
            for frame,rows in truth.items():
                for gid,box in rows: union[frame][gid]=box
        union_ids=np.full(len(obs),-1,dtype=np.int32)
        for frame,ids in by_frame.items():
            g=list(union.get(frame,{}).items())
            mat=overlaps(boxes[ids],[b for _,b in g]); pairs=matched_pairs(mat,list(range(len(ids))))
            for i,j in pairs.items(): union_ids[ids[i]]=g[j][0]
        counts=Counter(); labels=np.full(fc.margin.shape,-1,dtype=np.int32)
        for qi,(q,truth) in enumerate(zip(fc.queries,truths)):
            row=Counter(); row['GT_observations']=sum(len(x) for x in truth.values())
            row['candidate_rows']=len(obs); gt_counts={f:len(g) for f,g in truth.items()}
            query_frames=set(gt_counts)|set(by_frame)
            row['query_frames']=len(query_frames)
            row['GT_positive_frames']=sum(n>0 for n in gt_counts.values())
            for frame,ids in by_frame.items():
                g=truth.get(frame,[]); mat=overlaps(boxes[ids],[b for _,b in g])
                possible=matched_pairs(mat,list(range(len(ids))))
                row['candidate_retrievable_GT']+=len(possible)
                for i,j in possible.items(): labels[qi,ids[i]]=g[j][0]
                for name,bits in [('raw',raw[qi]),('topk',initial[qi]),('v8p',baseline[qi])]:
                    chosen=np.flatnonzero(bits[ids]).tolist(); found=matched_pairs(mat,chosen)
                    row[name+'_matched_GT']+=len(found)
                    row[name+'_selected_rows']+=len(chosen)
                    row[name+'_unmatched_selected_rows']+=len(chosen)-len(found)
                    row[name+'_count_absolute_error']+=abs(len(chosen)-len(g))
            missing_frames=set(gt_counts)-set(by_frame)
            for name in ('raw','topk','v8p'):
                row[name+'_count_absolute_error']+=sum(gt_counts[f] for f in missing_frames)
                row[name+'_retrievable_but_missed']=row['candidate_retrievable_GT']-row[name+'_matched_GT']
            row['GT_not_retrievable']=row['GT_observations']-row['candidate_retrievable_GT']
            row['query_present']=int(row['GT_observations']>0)
            row['present_zero_output']=int(row['query_present'] and not baseline[qi].any())
            row['empty_query_FP']=int(baseline[qi].sum()) if not row['query_present'] else 0
            for track,ids in fc.by_track.items():
                frames=obs[ids,0]; adjacent=(np.diff(frames)>0)&(np.diff(frames)<=2)
                positive=labels[qi,ids]>=0
                same_identity=(union_ids[ids[1:]]==union_ids[ids[:-1]])&(union_ids[ids[1:]]>=0)
                stable=adjacent&(positive[1:]==positive[:-1])&same_identity
                row['adjacent_track_transitions']+=int(adjacent.sum())
                row['stable_membership_identity_transitions']+=int(stable.sum())
                row['physical_ID_visible_identity_changes']+=int((adjacent&(~same_identity)&(union_ids[ids[1:]]>=0)&(union_ids[ids[:-1]]>=0)).sum())
                for name,bits in [('raw',raw[qi,ids]),('topk',initial[qi,ids]),('v8p',baseline[qi,ids])]:
                    flip=bits[1:]!=bits[:-1]
                    row[name+'_all_flips']+=int((flip&adjacent).sum())
                    row[name+'_stable_identity_membership_flips']+=int((flip&stable).sum())
                if masks:
                    positive_count=int(positive.sum())
                    if positive_count*2>len(ids): masks['TRACK_MAJORITY_ORACLE'][q['start']+ids]=True
            if masks: masks['FRAME_ORACLE'][q['start']:q['stop']]=labels[qi]>=0
            episodes.append(dict(pair=pair,video=video,query=q['expression_id'],query_text=q['query_text'],
                query_types=expression_types(q['query_text']),**dict(row)))
            counts.update(row)
        physical=[]; fragments=defaultdict(set)
        for tid,ids in fc.by_track.items():
            values=union_ids[ids]; valid=values[values>=0]; c=Counter(map(int,valid))
            for gid in c: fragments[gid].add(tid)
            physical.append(dict(track=tid,observations=len(ids),GT_matched_observations=len(valid),
                distinct_visible_GT_ids=len(c),purity=max(c.values())/len(valid) if len(valid) else None))
        truthdir=OUT/'offline_GT'/pair/video;truthdir.mkdir(parents=True,exist_ok=True)
        np.save(truthdir/'labels.npy',labels);np.save(truthdir/'union_ids.npy',union_ids)
        np.save(truthdir/'evaluation_frames.npy',eval_frames)
        vrow=dict(video=video,counts=dict(counts),physical_tracks=physical,
            GT_ids_with_multiple_candidate_IDs=sum(len(v)>1 for v in fragments.values()),GT_ids_observed=len(fragments))
        videos.append(vrow);totals.update(counts)
        atomic_json(folder/'videos'/f'{video}.json',vrow)
        print(json.dumps(dict(event='OFFLINE_AUDIT_VIDEO',pair=pair,video=video,
            retrievable=counts['candidate_retrievable_GT'],GT=counts['GT_observations'],
            topk_stable_flips=counts['topk_stable_identity_membership_flips'],v8p_stable_flips=counts['v8p_stable_identity_membership_flips'])),flush=True)
    if masks:
        for name,mask in masks.items():
            dest=folder/'oracles'/name
            receipt=seal_mask(dest,m,mask,target=target,model=model,method=name,GT_open_attempts=len(episodes),
                extra=dict(offline_oracle_uses_GT=True,
                    original_boxes_and_IDs_preserved=True,not_strict_theoretical_upper_bound=True))
            receipt['status']='OFFLINE_ORACLE_USES_GT'
            atomic_json(dest/'predictions_seal.json',receipt)
    total=dict(totals);total['candidate_retrievable_recall']=total['candidate_retrievable_GT']/max(total['GT_observations'],1)
    summary=dict(status='PASS_OFFLINE_DIAGNOSIS',pair=pair,target=target,counts=total,episodes=episodes,videos=videos,
        GT_used_only_offline=True,GT_matching='IoU>=.5 maximum cardinality then IoU; labels are matched query GT IDs, -1 unmatched',
        no_additive_HOTA_error_percentages=True,
        temporal_gate=bool(total['topk_stable_identity_membership_flips']>0),
        final_v8p_stable_output_flips=total['v8p_stable_identity_membership_flips'],
        oracle_evaluation_pending=bool(masks),wall_seconds=time.perf_counter()-begin)
    atomic_json(folder/'completion.json',summary)
    print(json.dumps(dict(event='OFFLINE_AUDIT_COMPLETE',pair=pair,counts=total,seconds=summary['wall_seconds'])),flush=True)


def audit_cached_kitti():
    """Keep the original numerical GT, rather than silently reconstruct KITTI GT.

    The original KITTI gt_template was not backed up. Its sealed complete-bank
    labels and counts were; these support membership/absence diagnosis, not a
    raw-GT physical ID or one-to-one candidate-recall claim.
    """
    pair='C5';m=metadata('KITTI-V2');folder=OUT/'audit'/pair;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completion.json').is_file():return
    begin=time.perf_counter();rows=[];total=Counter()
    for video,bank in sorted(m['videos'].items()):
        source=HISTORY/'features/KITTI-V2/V2-best/videos'/video
        md=json.loads((source/'metadata.json').read_text());obs=np.load(source/'observations.npy')
        assert np.array_equal(obs,np.asarray(bank['observations']))
        queries=[q for q in m['episodes'] if q['video_id']==video]
        assert [q['expression_id'] for q in queries]==[q['expression_id'] for q in md['queries']]
        labels=np.load(source/'labels.npy').reshape(len(queries),-1);counts=np.load(source/'GT_counts.npy')
        mask=baseline_mask(pair,video);assert mask.shape==labels.shape
        truthdir=OUT/'offline_GT'/pair/video;truthdir.mkdir(parents=True,exist_ok=True)
        np.save(truthdir/'labels.npy',np.where(labels,0,-1).astype(np.int32))
        for qi,q in enumerate(queries):
            c=Counter(GT_observations=int(counts[qi].sum()),candidate_rows=len(obs),
                candidate_positive_membership_rows=int(labels[qi].sum()),v8p_selected_rows=int(mask[qi].sum()),
                v8p_selected_positive_membership_rows=int((mask[qi]&labels[qi]).sum()),
                v8p_selected_negative_membership_rows=int((mask[qi]&~labels[qi]).sum()),
                query_present=int(counts[qi].sum()>0),present_zero_output=int(counts[qi].sum()>0 and not mask[qi].any()),
                empty_query_FP=int(mask[qi].sum()) if counts[qi].sum()==0 else 0)
            rows.append(dict(video=video,query=q['expression_id'],**dict(c)));total.update(c)
    result=dict(status='PASS_SEALED_NUMERICAL_GT_DIAGNOSIS',pair=pair,counts=dict(total),episodes=rows,
        GT_used_only_offline=True,truth_source='unchanged original complete-bank V2-best labels.npy and GT_counts.npy; observations and query order exactly verified',
        original_gt_template_missing_in_backup=True,
        raw_annotation_copy_found='/data3/liuyeqiang/nas/A800/data2/user/LocateMOT/data/refer_kitti_v2',
        no_GT_protocol_substitution=True,raw_ID_and_one_to_one_coverage_not_claimed=True,
        temporal_gate=False,wall_seconds=time.perf_counter()-begin)
    atomic_json(folder/'completion.json',result)
    print(json.dumps(dict(event='CACHED_KITTI_AUDIT_COMPLETE',counts=dict(total))),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=list(PAIRS),default='C4')
    p.add_argument('--phase',choices=['audit','oracles'],default='audit');a=p.parse_args()
    assert (OUT/'preregistration.json').is_file()
    if a.phase=='audit': audit(a.pair)
    else:
        results={name:evaluate(OUT/'audit'/a.pair/'oracles'/name,a.pair,oracle=True)
                 for name in ('FRAME_ORACLE','TRACK_MAJORITY_ORACLE')}
        atomic_json(OUT/'audit'/a.pair/'oracle_results.json',results)


if __name__=='__main__':main()
