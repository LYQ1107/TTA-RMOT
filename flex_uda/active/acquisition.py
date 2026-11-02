"""Expected information gain, fixed expression routing, and diverse keyframes."""
from dataclasses import dataclass
import re
import numpy as np
from scipy.special import expit
from flex_uda.active.belief import evidence_increment

VIEWS = ('CROP', 'CLIP', 'FRAME_BOX')
WORDS = {
    'MOTION': 'moving|move|turning|turn|parking|parked|standing|stand|walking|walk|running|run|stopped|stop|faster|slower|same direction|opposite direction|counter direction|braking|accelerating|jumping|serving',
    'SPATIAL': 'left|right|front|behind|back|near|far|ahead|middle|side|beside|closest',
    'APPEARANCE': 'black|white|red|blue|green|yellow|gray|grey|silver|brown|purple|orange|pink|dark|light|shirt|t-shirt|tshirt|pants|skirt|clothes|wearing|dressed|jersey|uniform',
    'ROLE': 'team|libero|goalkeeper|referee|setter|coach',
}
COLORS = 'black|white|red|blue|green|yellow|gray|grey|silver|brown|purple|orange|pink'


def expression_types(text):
    text = text.lower()
    return [kind for kind, terms in WORDS.items() if any(re.search(r'(?<!\w)'+re.escape(w)+r'(?!\w)', text)
                                                       for w in terms.split('|'))] or ['OTHER']


def route(text):
    found = expression_types(text)
    if 'MOTION' in found: return 'CLIP'
    if 'SPATIAL' in found or 'ROLE' in found: return 'FRAME_BOX'
    return 'CROP'


def exclusive_group(texts, target):
    if target != 'Dance' or len(texts) != 17: return False
    return all(re.search(r'\b('+COLORS+r')\b', text.lower()) and
               not any(kind in expression_types(text) for kind in ('MOTION', 'SPATIAL')) for text in texts)


def entropy(p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return -p*np.log(p)-(1-p)*np.log1p(-p)


def information_gain(p, sensitivity, specificity):
    p = np.clip(p, 1e-12, 1-1e-12)
    s, c = np.clip([sensitivity, specificity], 1e-8, 1-1e-8)
    yes = p*s+(1-p)*(1-c)
    p_yes = p*s/yes; p_no = p*(1-s)/(1-yes)
    return float(max(0., entropy(p)-yes*entropy(p_yes)-(1-yes)*entropy(p_no)))


def keyframe(observations, asked_frames, width, height):
    available = [o for o in observations if o['frame_id'] not in asked_frames]
    if not available: return None
    def priority(o):
        x, y, w, h = o['bbox_xywh']
        away = min(x, y, width-x-w, height-y-h) >= 5
        distance = min((abs(o['frame_id']-t) for t in asked_frames), default=float('inf'))
        return (-distance, -int(away), -w*h, o['frame_id'])
    return min(available, key=priority)


@dataclass(frozen=True)
class Question:
    query: int
    track: int
    view: str
    kind: str
    information_gain: float


def questions(state, texts, lengths, reliability, round_index, *, only_crop=False, use_presence=True):
    p = state.posterior; result = []
    for q, text in enumerate(texts):
        if state.absent[q]: continue
        routed = 'CROP' if only_crop else route(text)
        views = ('CROP',) if only_crop else VIEWS
        eligible = [t for t, length in enumerate(lengths) if length >= 3]
        if not eligible: continue
        best_track = min(eligible, key=lambda t: (-state.z[q, t], t))
        for t in eligible:
            ig = {v: information_gain(p[q, t], *reliability[v]) for v in views}
            view = routed if round_index < 2 else max(views, key=lambda v: ig[v])
            result.append(Question(q, t, view, 'IDENTITY', max(ig.values())))
        uncertain = .1 < state.presence_max[q] < .9
        if use_presence and (state.low[q] or uncertain) and not state.confirmed_present[q]:
            # The second No must concern a different physical track. Pick the
            # highest-z remaining track, rather than reusing the first No.
            available = [t for t in eligible if t not in state.no_tracks[q]]
            if available:
                t = min(available, key=lambda t: (-state.z[q, t], t))
                ig = {v: information_gain(state.presence_max[q], *reliability[v]) for v in views}
                view = routed if round_index < 2 else max(views, key=lambda v: ig[v])
                result.append(Question(q, t, view, 'PRESENCE', max(ig.values())))
    return sorted(result, key=lambda x: (-x.information_gain, x.kind != 'PRESENCE', x.query, x.track, x.view))


def round_budget(total, round_index, rounds=4):
    return total//rounds + (total%rounds if round_index == rounds-1 else 0)
