"""Audit fusion/transport order and prepare a separate per-evidence candidate.

Only frozen paid QA and teacher-only prefix slopes are read. The live v1
heads, targets, outputs and registration stay immutable. No quality result or
target GT is consulted, and new target arrays are not a new HOTA experiment.
"""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import PAIRS,metadata,history,generation_guard
from flex_uda.active.relational_transport import horizontal_direction
from flex_uda.active.object_reference import transport_targets
from flex_uda.active.object_reference_views import reference_frame_supervision,group_canonical_frames
from scripts.a100_relation_transport_prepare import OUT as SLOPE_PREPARED
from scripts.a100_object_reference_prepare import OUT as V1
from scripts.a100_six_direction import OUT as SIX,PUBLIC,source

OUT=SIX/'object_reference/per_evidence_prepare'
REGISTRATION=PUBLIC/'object_reference_order_registration.json'


def main():
    registration=json.loads(REGISTRATION.read_text())
    for name,sha in registration['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    started=time.perf_counter();guard=generation_guard();results={}
    for pair,(_,target) in PAIRS.items():
        m=metadata(target);past=[];receipts=[]
        for video,bank in sorted(m['videos'].items()):
            queries=[q for q in m['episodes'] if q['video_id']==video]
            directions=np.asarray([horizontal_direction(q['query_text']) for q in queries],int)
            if not directions.any():continue
            geometry=json.loads((SLOPE_PREPARED/pair/video/'teacher_rows.json').read_text())
            beta=json.loads((SLOPE_PREPARED/pair/video/'prefix_fit.json').read_text())['beta']
            with np.load(source(pair)/'source'/video/'sparse_inputs.npz') as a:tracks=a['tracks']
            records=json.loads((history(pair)/'videos'/video/'questions.json').read_text())
            past.append(dict(video=video,records=records,queries=queries,tracks=tracks,
                width=geometry['width'],height=geometry['height'],directions=directions))
            blocks=[];frames=[];order_errors=[];eligible_errors=[];multi=eligible_frames=0
            for item in past:
                frame=reference_frame_supervision(item['records'],item['queries'],item['tracks'],
                    item['width'],item['height'],item['directions'],beta)
                block=group_canonical_frames(frame);blocks.append(block);frames.append(frame)
                d=item['directions'][frame['cells'][:,0]]
                # This contrast holds per-request clipping fixed, isolating
                # noncommutation from the separate clipping-order difference.
                mean_first=transport_targets(frame['control_frame_probability'],frame['coords'],d,beta)
                error=np.abs(frame['reference_frame_probability']-mean_first)
                order_errors.extend(error[d!=0]);eligible_frames+=int((d!=0).sum())
                multi+=frame['eligible_multi_view_frame_cells']
                eligible_errors.extend(error[(d!=0)&(error>1e-12)])
            sizes=[len(b['cells']) for b in blocks]
            arrays=dict(reference_target=np.concatenate([b['reference_target'] for b in blocks]),
                unshifted_target=np.concatenate([b['unshifted_target'] for b in blocks]),
                sample_weight=np.concatenate([b['sample_weight'] for b in blocks]),
                cell_query=np.concatenate([b['cells'][:,0] for b in blocks]),
                cell_track=np.concatenate([b['cells'][:,1] for b in blocks]),
                cell_video=np.repeat(np.arange(len(blocks)),sizes))
            with np.load(V1/pair/video/'prefix_targets.npz') as a:
                old={k:a[k] for k in arrays}
            for key in ('sample_weight','cell_query','cell_track','cell_video'):
                assert np.array_equal(arrays[key],old[key]),key
            reference_delta=np.abs(arrays['reference_target']-old['reference_target'])
            control_delta=np.abs(arrays['unshifted_target']-old['unshifted_target'])
            destination=OUT/pair/video;destination.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(destination/'prefix_targets.npz',**arrays)
            receipt=dict(status='SEALED_PER_EVIDENCE_REFERENCE_TARGET_CANDIDATE',video=video,
                ordered_videos=[p['video'] for p in past],group_counts=sizes,
                prefix_reference_groups=sum(sizes),prefix_frame_cells=sum(len(f['cells']) for f in frames),
                beta=beta,reference_position=.5,common_confidence_from_raw_frame_target=True,
                eligible_frame_cells=eligible_frames,eligible_multi_view_frame_cells=multi,
                frame_cells_with_pure_order_difference=len(eligible_errors),
                maximum_pure_order_frame_difference=float(max(order_errors,default=0.)),
                mean_pure_order_frame_difference=float(np.mean(order_errors)) if order_errors else 0.,
                maximum_group_reference_difference_from_V1=float(reference_delta.max()),
                mean_group_reference_difference_from_V1=float(reference_delta.mean()),
                maximum_group_control_difference_from_V1=float(control_delta.max()),
                reference_GPU_float32_labels_identical_to_V1=bool(np.array_equal(
                    arrays['reference_target'].astype('float32'),old['reference_target'].astype('float32'))),
                control_GPU_float32_labels_identical_to_V1=bool(np.array_equal(
                    arrays['unshifted_target'].astype('float32'),old['unshifted_target'].astype('float32'))),
                target_min=float(arrays['reference_target'].min()),target_max=float(arrays['reference_target'].max()),
                all_prefixes_use_only_past_and_current_teacher_answers=True,
                target_arrays_not_to_Git=True,manual_target_GT_open_attempts=len(guard),
                no_current_neural_or_quality_outputs_read=True)
            atomic_json(destination/'completion.json',receipt);receipts.append(receipt)
        results[pair]=dict(source=PAIRS[pair][0],target=target,prefixes=receipts,
            exact_fallback_if_not_eligible=not bool(receipts),
            requires_distinct_reference_head_for_changed_float32_targets=any(
                not r['reference_GPU_float32_labels_identical_to_V1'] or
                not r['control_GPU_float32_labels_identical_to_V1'] for r in receipts))
    assert not guard
    report=dict(status='TEACHER_ONLY_REFERENCE_TRANSPORT_ORDER_AUDITED',pairs=results,
        manual_target_GT_open_attempts=0,new_teacher_encoder_GPU_or_neural_fits=0,
        current_V1_parameters_targets_or_predictions_changed=False,
        conditional_theory='With additive per-view logit state response, per-evidence canonicalization removes that state term before probability fusion. Fusion and nonlinear transport generally do not commute. View bias, varying view sets and probability pooling still limit reconstruction of a single true latent relation.',
        interpretation='A refinement prepared from a mathematical implementation review, not selected from HOTA. New target arrays are not new task-quality evidence.',
        all_data_development=True,goal_status='ACTIVE',wall_seconds=time.perf_counter()-started)
    atomic_json(OUT/'completion.json',report)
    (PUBLIC/'object_reference_transport_order_audit.json').write_text(json.dumps(report,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps(dict(status=report['status'],pairs={p:dict(prefixes=len(r['prefixes']),
        distinct_targets=r['requires_distinct_reference_head_for_changed_float32_targets'],
        final=r['prefixes'][-1] if r['prefixes'] else None) for p,r in results.items()},GT_open_attempts=0),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
