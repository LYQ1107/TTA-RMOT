"""Image-supported content pairs shared with a new primary and all controls.

No semantic-score equality follows from these pairs. All raw9352features
and original teacher responses stay unchanged; only a weak image pretext is
prepared from existing short-gap pixels and the shared frozen ROI bank.
"""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.a100_pixel_continuity_audit import OUT as PIXEL,PLAN as PIXEL_PLAN
from scripts.a100_residual_rsi_inputs import OUT as INPUTS
from scripts.a100_six_direction import OUT as SIX

OUT=SIX/'evidence_visual_inputs_20261125'
PUBLIC=ROOT/'docs/a100_six_direction_20261122'


def main():
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.active.full_appearance_learning_inputs import FullAppearanceBank
    guard=generation_guard()
    completed=json.loads((PIXEL/'completion.json').read_text())
    assert completed['GT_reads']==completed['teacher_calls']==0
    reports=[]
    for pair,target in (('C1','Dance'),('C2','KITTI-V2'),('C4','SportsMOT')):
        record=json.loads((PIXEL/target/'completion.json').read_text())
        video=record['summary']['video'];bank=FullAppearanceBank(target,video)
        left=[];right=[];weights=[];keys=[]
        for triplet in record['triplets']:
            frames=triplet['frames']
            for unit in triplet['units']:
                good=unit['same_ID_supported_points'];valid=unit['cycle_valid'];track=unit['track']
                if good<3 or 2*good<valid:continue
                rows=[bank.by_observation.get((int(frame),int(track))) for frame in frames]
                if any(r is None for r in rows):continue
                if not bank.valid[rows].all():continue
                features=bank.normalized(np.asarray(rows))
                for i in (1,2):
                    left.append(features[0]);right.append(features[i]);weights.append(good/valid)
                    keys.append([video,int(track),int(frames[0]),int(frames[i])])
        assert len(left)>=8
        folder=OUT/pair;folder.mkdir(parents=True,exist_ok=True)
        np.savez(folder/'pixel_pairs.npz',left=np.asarray(left,dtype='float32'),
            right=np.asarray(right,dtype='float32'),weight=np.asarray(weights,dtype='float32'))
        atomic_json(folder/'pair_keys.json',dict(keys=keys,pixel_source_sha256=file_sha(PIXEL/target/'completion.json')))
        inputs=json.loads((INPUTS/pair/'completion.json').read_text())
        assert video==inputs['video']
        reports.append(dict(pair=pair,target=target,video=video,pairs=len(left),
            pair_sha256=file_sha(folder/'pixel_pairs.npz'),keys_sha256=file_sha(folder/'pair_keys.json'),
            rule='>=3same-IDcyclepoints and>=halfvalidpoints; confidenceweight=same/valid. No semantic truth, hard negative, fullscore equality, GT or teacher-dependent pair selection.',
            source_file_sha256=file_sha(PIXEL/target/'completion.json')))
        print(json.dumps(dict(status='EVIDENCE_VISUAL_PAIRS_READY',**reports[-1])),flush=True)
    assert not guard
    result=dict(status='COMPLETE_SHARED_IMAGE_SUPPORTED_CONTENT_PAIRS',cases=reports,
        self_source_sha256=file_sha(__file__),pixel_plan_sha256=file_sha(PIXEL_PLAN),
        GT_open_attempts=0,new_teacher_backbone_or_neural_calls=0,
        pairs_not_certified_physical_identity_or_semantic_membership=True,all_data_DEV=True,goal_status='ACTIVE')
    atomic_json(OUT/'completion.json',result);atomic_json(PUBLIC/'evidence_visual_inputs_20261125.json',result)


if __name__=='__main__':main()
