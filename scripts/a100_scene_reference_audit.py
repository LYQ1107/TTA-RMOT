"""Measure scene/camera nuisance before proposing motion-based adaptation.

Only real images and fixed candidates. Background rigid-scene fitting uses
separate image correspondences for fitting and descriptive validation.
Neither homography nor epipolar residual is an object semantic label.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.a100_six_direction import OUT as SIX

OUT = SIX/'scene_reference_audit_20261125'
PLAN = OUT/'registration.json'
PUBLIC = ROOT/'docs/a100_six_direction_20261122'
SOURCES = ('scripts/a100_scene_reference_audit.py', 'flex_uda/a100/mechanism.py',
           'flex_uda/a100/runtime.py', 'scripts/active_run.py',
           'flex_uda/active/joint_observation_inputs.py')


def prepare():
    import cv2
    import numpy as np
    from flex_uda.a100.mechanism import generation_guard, metadata
    guard = generation_guard(); assert not PLAN.exists()
    cases = []
    for target in ('Dance', 'KITTI-V2', 'SportsMOT'):
        md = metadata(target); video = sorted(md['videos'])[0]
        frames = np.unique(np.asarray(md['videos'][video]['observations'])[:, 0])
        current = frames[frames >= frames[0]+32]
        chosen = np.unique(np.linspace(0, len(current)-1, 12).astype(int))
        pairs = [(int(frames[np.searchsorted(frames, current[i]-32, side='right')-1]),
                  int(current[i])) for i in chosen]
        assert len(pairs) == 12 and all(a < b for a, b in pairs)
        cases.append(dict(target=target, video=video, frame_pairs=pairs))
    plan = dict(status='REGISTERED_IMAGE_ONLY_SCENE_REFERENCE_DIAGNOSIS', cases=cases,
        source_sha256={n: file_sha(ROOT/n) for n in SOURCES}, opencv_version=cv2.__version__,
        selection='First physical video/domain,12uniform observed-time pairs/domain, previous observation at least32raw frames earlier. No FPS claim, teacher answers or GT used.',
        image_max_side=1024, candidate_mask_margin=.05,
        background='Shi-Tomasi max1200,quality.01,minDistance8; exclude provided candidate boxes in BOTH frames.',
        matching='Pyramidal LK21x21,maxLevel3,30iterations/epsilon.01; forward-backward cycle<=1.5pixels.',
        held_correspondences='Every fifth surviving correspondence is excluded from model fitting; dependent descriptive check, not independent statistical validation.',
        models='Homography RANSAC2.5px and fundamental matrix RANSAC2px/confidence.995. Fixed OpenCV RNG0, no neural fits/seeds.',
        candidates='At most20Shi-Tomasi points/track; retain FB-consistent points inside same provided track box at current frame. Track identity and rigid motion not certified.',
        evidence_limits=['A planar homography cannot generally compensate depth-dependent ego motion.',
            'Small epipolar error does not prove stationary; moving objects can satisfy the same epipolar constraint.',
            'Background model can be unidentified under planar/rotation/low-parallax/poor-correspondence conditions.',
            'Bounding-box center displacements are not surface-point correspondences; use actual tracked pixels.',
            'No query truth, automatic semantic No, teacher adaptation or output calibration is produced.'],
        existing_feature_gap='Current12Dtrajectory subtracts median candidate motion, explicitly not image-derived camera motion.',
        no_Qwen_backbone_or_neural_training=True, no_GT_or_RMOT_predictions=True,
        CPU_workers=3, all_data_DEV=True, goal_status='ACTIVE')
    assert not guard
    atomic_json(PLAN, plan); atomic_json(PUBLIC/'scene_reference_audit_registration_20261125.json', plan)
    print(json.dumps(dict(status=plan['status'], frame_pairs=36, GPU_jobs=0, GT_reads=0)), flush=True)


def quantiles(values):
    import numpy as np
    x = np.asarray(values); x = x[np.isfinite(x)]
    return None if not len(x) else dict(zip(('p10', 'median', 'p90'), map(float, np.quantile(x, [.1, .5, .9]))))


def worker(case):
    import cv2
    import numpy as np
    from PIL import Image
    from flex_uda.a100.mechanism import metadata, generation_guard
    from scripts.active_run import video_spec
    guard = generation_guard(); started = time.perf_counter()
    cv2.setNumThreads(1); cv2.setRNGSeed(0)
    plan = json.loads(PLAN.read_text()); target, video = case['target'], case['video']
    spec = video_spec(target, video, metadata(target)['videos'][video])
    observations = {}
    for row in spec['observations']: observations.setdefault(row['frame_id'], {})[row['track_id']] = row['bbox_xywh']
    folder = OUT/target; folder.mkdir(parents=True, exist_ok=True)

    def load(frame):
        path = Path(spec['image_root'])/f"{frame+spec['image_offset']:06d}{spec['extension']}"
        with Image.open(path) as image:
            scale = plan['image_max_side']/max(image.size)
            gray = np.asarray(image.convert('L').resize((round(image.width*scale), round(image.height*scale)), Image.Resampling.BICUBIC)).copy()
        boxes = {track: np.asarray([x, y, x+w, y+h])*scale for track, (x, y, w, h) in observations[frame].items()}
        mask = np.full(gray.shape, 255, dtype='uint8')
        for x0, y0, x1, y1 in boxes.values():
            mx, my = .05*(x1-x0), .05*(y1-y0)
            mask[max(0, int(y0-my)):min(gray.shape[0], int(np.ceil(y1+my))),
                 max(0, int(x0-mx)):min(gray.shape[1], int(np.ceil(x1+mx)))] = 0
        return gray, boxes, mask

    def track(a, b, points):
        if points is None or not len(points): return np.empty((0, 2)), np.empty((0, 2)), np.zeros(0, bool)
        p = np.asarray(points, dtype='float32').reshape(-1, 1, 2)
        kwargs = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        q, status, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None, **kwargs)
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None, **kwargs)
        p, q, back = p[:, 0], q[:, 0], back[:, 0]
        valid = status[:, 0].astype(bool) & back_status[:, 0].astype(bool)
        valid &= np.isfinite(q).all(1) & np.isfinite(back).all(1) & (np.linalg.norm(p-back, axis=1) <= 1.5)
        valid &= (q[:, 0] >= 0) & (q[:, 0] < b.shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < b.shape[0])
        return p, q, valid

    def residual(p, q, H, F):
        if not len(p): return dict(raw_displacement=None, homography=None, epipolar=None)
        x = np.c_[p, np.ones(len(p))]; y = np.c_[q, np.ones(len(q))]
        herror = ferror = None
        if H is not None:
            warped = x@H.T; valid = np.abs(warped[:, 2]) > 1e-8
            herror = quantiles(np.linalg.norm(warped[valid, :2]/warped[valid, 2:] - q[valid], axis=1))
        if F is not None:
            fx, fty = x@F.T, y@F
            error = np.abs(np.sum(y*fx, axis=1))/np.sqrt((fx[:, :2]**2).sum(1)+(fty[:, :2]**2).sum(1)).clip(min=1e-12)
            ferror = quantiles(error)
        return dict(raw_displacement=quantiles(np.linalg.norm(p-q, axis=1)), homography=herror, epipolar=ferror)

    results = []
    for fi, (before, current) in enumerate(case['frame_pairs']):
        a, boxes0, mask0 = load(before); b, boxes1, mask1 = load(current)
        assert a.shape == b.shape
        pts = cv2.goodFeaturesToTrack(a, maxCorners=1200, qualityLevel=.01, minDistance=8, mask=mask0)
        p, q, valid = track(a, b, pts)
        selected = np.flatnonzero(valid)
        if len(selected):
            xy = np.round(q[selected]).astype(int)
            xy[:, 0] = xy[:, 0].clip(0, b.shape[1]-1); xy[:, 1] = xy[:, 1].clip(0, b.shape[0]-1)
            valid[selected] &= mask1[xy[:, 1], xy[:, 0]] > 0
        p, q = p[valid], q[valid]
        held = np.arange(len(p)) % 5 == 0; fit = ~held
        H = F = None; hi = ff = None
        if fit.sum() >= 16:
            cv2.setRNGSeed(0)
            H, hi = cv2.findHomography(p[fit], q[fit], cv2.RANSAC, 2.5, maxIters=2000, confidence=.995)
            cv2.setRNGSeed(0)
            F, ff = cv2.findFundamentalMat(p[fit], q[fit], cv2.FM_RANSAC, 2., .995, 5000)
            if H is not None and (H.shape != (3, 3) or not np.isfinite(H).all()): H = None
            if F is not None and (F.shape != (3, 3) or not np.isfinite(F).all()): F = None
        tracks = []
        for tid in sorted(boxes0.keys() & boxes1.keys()):
            x0, y0, x1, y1 = boxes0[tid]; roi = np.zeros(a.shape, dtype='uint8')
            roi[max(0, int(y0)):min(a.shape[0], int(np.ceil(y1))), max(0, int(x0)):min(a.shape[1], int(np.ceil(x1)))] = 255
            points = cv2.goodFeaturesToTrack(a, maxCorners=20, qualityLevel=.01, minDistance=3, mask=roi)
            op, oq, ok = track(a, b, points)
            bx0, by0, bx1, by1 = boxes1[tid]
            ok &= (oq[:, 0] >= bx0) & (oq[:, 0] <= bx1) & (oq[:, 1] >= by0) & (oq[:, 1] <= by1)
            tracks.append(dict(track=int(tid), matched_surface_points=int(ok.sum()),
                diagnostics=residual(op[ok], oq[ok], H, F)))
        background = residual(p[held], q[held], H, F)
        result = dict(before=before, current=current, frame_gap=current-before,
            background_matches=len(p), fitting_points=int(fit.sum()), held_points=int(held.sum()),
            H_fitted=H is not None, F_fitted=F is not None,
            H_inlier_fraction=None if hi is None else float(hi.mean()), F_inlier_fraction=None if ff is None else float(ff.mean()),
            held_background=background, candidate_tracks=tracks,
            fitted_model_not_certified_identifiable_or_semantic_truth=True)
        results.append(result)
        np.savez(folder/f'pair_{fi:02d}.npz', background_previous=p, background_current=q,
                 held=held, H=np.empty((0, 0)) if H is None else H, F=np.empty((0, 0)) if F is None else F)
        print(json.dumps(dict(status='IMAGE_SCENE_REFERENCE_PAIR_COMPLETE', target=target,
            pair=fi+1, pairs=12, background_matches=len(p), held_background=background)), flush=True)
    assert not guard
    report = dict(target=target, video=video, pairs=results, wall_seconds=time.perf_counter()-started,
                  GT_reads=0, teacher_or_neural_updates=0)
    atomic_json(folder/'completion.json', report)
    return report


def main():
    import numpy as np
    plan = json.loads(PLAN.read_text())
    for n, sha in plan['source_sha256'].items(): assert file_sha(ROOT/n) == sha, n
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=3) as pool: reports = list(pool.map(worker, plan['cases']))
    summary = []
    for report in reports:
        pairs = report['pairs']; metrics = {}
        for name in ('raw_displacement', 'homography', 'epipolar'):
            x = [p['held_background'][name]['median'] for p in pairs if p['held_background'][name] is not None]
            metrics[name] = quantiles(x)
        summary.append(dict(target=report['target'], video=report['video'], pairs=len(pairs),
            H_fitted=sum(p['H_fitted'] for p in pairs), F_fitted=sum(p['F_fitted'] for p in pairs),
            pair_median_held_pixel_diagnostics=metrics))
    result = dict(status='COMPLETE_IMAGE_ONLY_SCENE_REFERENCE_DIAGNOSIS', domains=summary,
        real_image_pairs=36, neural_or_teacher_updates=0, GT_reads=0,
        wall_seconds=time.perf_counter()-started, source_registration_sha256=file_sha(PLAN),
        background_compensation_not_teacher_or_RMOT_accuracy=True,
        raw_points_models_and_track_details_nonGit=True, goal_status='ACTIVE')
    atomic_json(OUT/'completion.json', result); atomic_json(PUBLIC/'scene_reference_audit_completion_20261125.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); args = parser.parse_args()
    if args.prepare: prepare()
    else: main()
