"""Check common-input nuisance properties without model fitting or GT."""
from pathlib import Path
import sys
import json
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.active.joint_observation_inputs import trajectory_features


def main():
    obs = np.asarray([(t,i) for t in range(5) for i in range(3)],dtype=int)
    t,i = obs.T
    coords = np.c_[.1+.22*i+.01*t*(i+1),.2+.13*i-.006*t*(i+1),
        .08+.02*i+.002*t,.12+.015*i+.001*t]
    base = trajectory_features(obs,coords,1)
    translated = coords.copy();translated[:,:2] += np.c_[.03*np.sin(t),.02*np.cos(t)]
    translated[:,2:] *= np.exp(.1*np.sin(t))[:,None]
    changed = trajectory_features(obs,translated,1)
    common_nuisance_error = float(np.abs(base[:,:8]-changed[:,:8]).max())
    assert common_nuisance_error < 1e-12 and np.abs(base[:,8:10]-changed[:,8:10]).max() > .001
    affine = coords.copy();scale = .8+.03*t
    affine[:,:2] = affine[:,:2]*scale[:,None]+np.c_[.02*t,-.01*t]
    affine[:,2:] *= scale[:,None]
    ranks_size_error = float(np.abs(base[:,:4]-trajectory_features(obs,affine,1)[:,:4]).max())
    assert ranks_size_error < 1e-12
    prefix = t <= 2
    causal_error = float(np.abs(base[prefix]-trajectory_features(obs[prefix],coords[prefix],1)).max())
    assert causal_error == 0.
    keep = ~((t == 1)&(i == 2));irregular = trajectory_features(obs[keep],coords[keep],1)
    row = np.flatnonzero((obs[keep,0] == 2)&(obs[keep,1] == 2))[0]
    assert irregular[row,10] == 1 and irregular[row,11] == 0 and np.all(irregular[row,4:8] == 0)
    report = dict(status='PASS_COMMON_INPUT_NUISANCE_AND_CAUSAL_GEOMETRY_PROPERTIES',
        source_sha256={n:file_sha(ROOT/n) for n in ['scripts/a100_joint_observation_input_probe.py','flex_uda/active/joint_observation_inputs.py']},
        common_translation_and_logsize_shift_residual_error=common_nuisance_error,
        positive_affine_scene_rank_relative_size_error=ranks_size_error,past_current_prefix_error=causal_error,
        unmatched_history_not_claimed_common_camera_residual=True,
        raw_image_motion_also_retained=True,not_true_world_motion_or_camera_identification=True,
        conditions='Properties of provided geometry; common translation and aligned history for velocity residual; positive affine coordinate transform for ranks/relative sizes. Clipping/changed candidate set/parallax not covered.',
        all_ordinary_learners_receive_same_extra_features=True,extra_dimensions=12,
        no_GT_GPU_teacher_fits_or_HOTA=True,not_a_new_classical_theorem=True,goal_status='ACTIVE')
    atomic_json(ROOT/'docs/a100_six_direction_20261122/joint_observation_input_math.json',report)
    print(json.dumps(report))


if __name__ == '__main__':main()
