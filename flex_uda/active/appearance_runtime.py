"""Deployment glue for the fixed teacher-only appearance regularizer.

The current video is transductive; the caller appends its teacher rows only
after emitting every requested method. No annotation or QA cache is read here.
"""
import numpy as np
from flex_uda.active.semantic_colour import appearance_cues,crossfit_predictions,correct_belief,append_history


def empty_appearance_history():
    return {family:dict(x=np.empty(0),y=np.empty(0)) for family in (0,1)}


def regularize_appearance(unary,colours,texts,lengths,answers,tracks,video,history):
    cue,available,families,_=appearance_cues(colours,texts,lengths)
    probability,prior,active,fits=crossfit_predictions(cue,available,families,answers,tracks,video,history)
    result=correct_belief(unary,probability,prior,active)
    inputs=dict(cue=cue,available=available,families=families)
    receipt=dict(active_nodes=int(active.sum()),supported_queries=int((families>=0).sum()),
        current_teacher_rows=int((np.isfinite(answers)&available).sum()),crossfit_models=len(fits),
        current_track_group_excluded=True,past_and_current_teacher_only=True,
        bounded_odds_regularization_not_independent_evidence=True)
    return result,inputs,receipt


def commit_appearance_history(history,inputs,answers):
    append_history(history,inputs['cue'],inputs['available'],inputs['families'],answers)
