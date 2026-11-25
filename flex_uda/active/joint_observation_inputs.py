"""Common image/trajectory inputs for joint and adequate ordinary learners.

The added quantities are observable predicted-track geometry, never target
annotations. Median subtraction is a nuisance-invariant representation; it
is not a camera-motion estimator or an output-score calibration algorithm.
"""
import json
import numpy as np
from flex_uda.active import observation_learning_inputs as original

EXTRA = 12


def use_verified_local_features():
    from scripts.a100_observation_local_cache import CACHE
    receipt = json.loads((CACHE/'completion.json').read_text())
    assert receipt['status'] == 'PASS_BYTE_IDENTICAL_HISTORICAL_OBSERVATION_LOCAL_CACHE'
    assert receipt['files'] == 592 and receipt['source_read_hash_equals_separate_local_hash_for_every_file']
    original.HISTORY = CACHE
    return receipt


def midranks(values):
    ordered = np.sort(values)
    if len(values) == 1:
        return np.full(1,.5)
    return (np.searchsorted(ordered,values,'left')+np.searchsorted(ordered,values,'right')-1)/(2*(len(values)-1))


def trajectory_features(observations, coordinates, stride):
    obs = np.asarray(observations,dtype=int)
    coords = np.asarray(coordinates,dtype='float64')
    assert coords.shape == (len(obs),4) and stride > 0 and np.isfinite(coords).all()
    result = np.zeros((len(obs),EXTRA),dtype='float64')
    velocity = np.zeros((len(obs),4),dtype='float64')
    previous = np.zeros(len(obs),dtype=bool)
    aligned = np.zeros(len(obs),dtype=bool)
    size = np.log(coords[:,2:].clip(min=1e-6))
    for track in np.unique(obs[:,1]):
        rows = np.flatnonzero(obs[:,1] == track)
        rows = rows[np.argsort(obs[rows,0],kind='stable')]
        times = obs[rows,0]
        for position,row in enumerate(rows):
            p = int(np.searchsorted(times,times[position]-stride,'right'))-1
            if p < 0:
                continue
            before = rows[p]; gap = int(obs[row,0]-obs[before,0])
            assert gap > 0
            velocity[row,:2] = (coords[row,:2]-coords[before,:2])/gap
            velocity[row,2:] = (size[row]-size[before])/gap
            previous[row] = True;aligned[row] = gap == stride
    for frame in np.unique(obs[:,0]):
        rows = np.flatnonzero(obs[:,0] == frame)
        result[rows,0] = midranks(coords[rows,0])
        result[rows,1] = midranks(coords[rows,1])
        result[rows,2:4] = size[rows]-np.median(size[rows],axis=0)
        comparable = rows[aligned[rows]]
        if len(comparable) >= 2:
            result[comparable,4:8] = velocity[comparable]-np.median(velocity[comparable],axis=0)
            result[comparable,11] = 1.
    result[:,8:10] = velocity[:,:2]  # Preserve global image-motion clues too.
    result[:,10] = previous
    assert np.isfinite(result).all()
    return result


class JointObservationVideo(original.ObservationVideo):
    def __init__(self,pair,video):
        super().__init__(pair,video)
        from scripts.active_run import video_spec
        self.clip_stride = video_spec(self.target,video,self.bank)['delta']
        self.trajectory = trajectory_features(self.obs,self.geometry[:,:4],self.clip_stride).astype('float32')
        old = self.raw_bags
        self.raw_bags = np.zeros((*old.shape[:-1],old.shape[-1]+EXTRA),dtype='float32')
        ti,si = np.nonzero(self.present)
        extra = self.trajectory[self.sample_rows[ti,si]]
        self.raw_bags[:,ti,si] = np.concatenate((old[:,ti,si,:-6],
            np.broadcast_to(extra,(old.shape[0],*extra.shape)),old[:,ti,si,-6:]),-1)
        # Geometry's final two coordinates remain normalized time and age.
        self.dimension = original.DIMENSION+EXTRA

    def dense_rows(self,qi,rows):
        old = super().dense_rows(qi,rows)
        return np.c_[old[:,:-6],self.trajectory[rows],old[:,-6:]].astype('float32')
