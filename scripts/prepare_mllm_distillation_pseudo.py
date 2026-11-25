#!/usr/bin/env python3
"""Export existing sealed P1 A raw probabilities, not confidence-screened labels."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.data.manifests import read_jsonl,write_jsonl
from flex_uda.teacher.mllm_pseudo_dataset import digest,identity,MLLMPseudoDataset,REVISION
from flex_uda.training import protect_target_annotations
from scripts.audit_teacher_leakage import validate_seal,OUTPUT,RUN,WEIGHT_SHA


def prepare(output):
    output=Path(output).resolve()
    if (output/'seal.json').exists():
        data=MLLMPseudoDataset(output,allow_val_diagnostic=True)
        return dict(status='PRESERVED_COMPLETE_CACHE',samples=len(data),seal_sha256=file_sha(output/'seal.json'))
    if output.exists():raise FileExistsError('preserve an incomplete export and use a new output directory')
    attempts=protect_target_annotations();p1=validate_seal()
    original=RUN/'full/T0/YES_FULL/teacher_outputs.jsonl'
    if file_sha(original)!=p1['original_files'][str(original)]:raise ValueError('P1 A raw predictions changed')
    raw={r['sample_id']:r for r in read_jsonl(original)}
    samples=read_jsonl(OUTPUT/'manifest.jsonl');runtime=read_jsonl(ROOT/'manifests/dance_val_runtime.jsonl')
    lookup={(r['video_id'],r['expression_id']):r for r in runtime}
    manifest=[];labels=[];windows=Counter();image_hashes={};tracker_hashes={};p1_states=Counter()
    for s in samples:
        a=raw.pop(s['sample_id']);r=a['raw_response'];key=identity(s)
        if identity(a)!=key or a['condition']!='T0' or a['method']!='YES_FULL':raise ValueError('P1 identity/condition mismatch')
        if r['frozen_weight_sha256']!=WEIGHT_SHA:raise ValueError('teacher parameter version differs')
        view=r['input_view']
        if view['variant']!='original' or view['view']!='current' or view['frames']!=[s['frame_id']] or view['boxes']!=[s['bbox_xyxy']] or r['images']!=1:
            raise ValueError('not the original single-frame teacher input')
        base=lookup[key[:2]];tracker=Path(base['predicted_tracks'])
        if str(tracker) not in tracker_hashes:tracker_hashes[str(tracker)]=file_sha(tracker)
        image=Path(base['image_root'])/f'{key[2]:06d}.png'
        if str(image) not in image_hashes:image_hashes[str(image)]=file_sha(image)
        m={k:s[k] for k in ('sample_id','video_id','query_id','query_text','frame_id','track_id','bbox_xyxy','raw_mot_line','selection_reasons','frozen_margin')}
        manifest.append(m);windows[key[:3]]+=1;p1_states[a['decision']['state']]+=1
        p=r['yes_probability'];yes=r['yes_logit'];no=r['no_logit']
        if not all(math.isfinite(v) for v in (p,yes,no)) or not 0<=p<=1:raise ValueError('nonfinite teacher output')
        fp32=1/(1+math.exp(max(-700,min(700,no-yes))))
        if abs(p-fp32)>1/256:raise ValueError('probability is inconsistent with original BF16 Yes/No logits')
        cache_identity=dict(model_revision=REVISION,prompt_hash=hashlib.sha256(r['prompt'].encode()).hexdigest(),
            image_sha256=image_hashes[str(image)],roi_sha256=view['image_sha256'][0],bbox_hash=digest(s['bbox_xyxy']),
            video_id=key[0],query_id=key[1],frame_id=key[2],track_id=key[3])
        labels.append(dict(sample_id=s['sample_id'],video_id=key[0],query_id=key[1],frame_id=key[2],track_id=key[3],
            bbox_xyxy=s['bbox_xyxy'],teacher_yes_logit=yes,teacher_no_logit=no,teacher_prob=p,
            valid=bool(r['top1_is_yes_no']),invalid_reason=None if r['top1_is_yes_no'] else 'NON_BINARY_TEACHER_RESPONSE',
            teacher_revision=REVISION,cache_identity=cache_identity,cache_key=digest(cache_identity),
            parent_raw_request_id=r['request_id'],p1_threshold_state_not_used=a['decision']['state'],
            teacher_forward_seconds_amortized=r['batch_forward_seconds']/r['batch_size']))
    if raw:raise ValueError('extra P1 labels without manifest candidates')
    output.mkdir(parents=True)
    write_jsonl(output/'manifest.jsonl',manifest);manifest_sha=file_sha(output/'manifest.jsonl')
    for label in labels:label['manifest_sha256']=manifest_sha
    write_jsonl(output/'soft_labels.jsonl',labels);write_jsonl(output/'runtime.jsonl',runtime)
    write_jsonl(output/'windows.jsonl',[dict(video_id=k[0],query_id=k[1],frame_id=k[2],sampled_candidates=n) for k,n in windows.items()])
    valid=[r for r in labels if r['valid']];by_video=Counter(r['video_id'] for r in valid)
    audit=dict(status='SEALED_RAW_SOFT_TEACHER_CACHE',protocol='TRANSDUCTIVE_VAL_MLLM_DISTILL_DIAGNOSTIC',
        target_split='val',no_Train_to_Val_claim=True,sealed_at_utc=datetime.now(timezone.utc).isoformat(),
        teacher_revision=REVISION,teacher_parameter_sha256=WEIGHT_SHA,model_condition='T0',
        videos=len(by_video),video_queries=len(lookup),sampled_pairs=len(labels),valid_soft_samples=len(valid),
        invalid_soft_samples=len(labels)-len(valid),windows=len(windows),unobserved_candidates_are_NOT_NEG=True,
        raw_probability_including_low_confidence_is_preserved=True,p1_threshold_states_NOT_used=dict(p1_states),
        teacher_prob_above_half=sum(r['teacher_prob']>.5 for r in valid),teacher_prob_below_or_equal_half=sum(r['teacher_prob']<=.5 for r in valid),
        teacher_probability_bins={str(q):sum(r['teacher_prob']<=q for r in valid) for q in (.05,.1,.5,.9,.95)},
        teacher_target_GT_open_attempts=attempts,new_teacher_GPU_calls=0,
        parent_P1_seal=str(OUTPUT/'config_hashes.json'),parent_P1_seal_sha256=file_sha(OUTPUT/'config_hashes.json'),
        parent_P1_raw_predictions=str(original),parent_P1_raw_predictions_sha256=file_sha(original),
        inherited_teacher_required_pairs=len(samples),inherited_teacher_budget=json.loads((OUTPUT/'teacher_cache_index.json').read_text())['by_method']['YES_FULL'],
        files_sha256={n:file_sha(output/n) for n in ('manifest.jsonl','soft_labels.jsonl','runtime.jsonl','windows.jsonl')},
        tracker_files_sha256=tracker_hashes,image_files_sha256=image_hashes,valid_samples_by_video=dict(by_video),
        annotation_policy='runtime whitelist; source supervision allowed; target GT never read',
        note='P1 confidence-UNKNOWN is not invalid raw soft evidence. No hard BCE labels, threshold screening, or default negatives are created.')
    atomic_json(output/'seal.json',audit)
    MLLMPseudoDataset(output,allow_val_diagnostic=True)
    for path in output.iterdir():
        if path.is_file():path.chmod(0o444)
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--diagnostic-val-authorized',action='store_true');a=p.parse_args()
    if not a.diagnostic_val_authorized:raise PermissionError('explicit user-authorized Val diagnostic required')
    result=prepare(a.output);print(json.dumps(result,ensure_ascii=False,indent=2))
