"""Factor-specific external evidence and object/state representation learning.

This is a new supervised-by-frozen-teacher hypothesis, not an accepted method
or an RSI result. Whole-expression No does not identify which factor failed;
appearance/state targets are separately paid teacher answers. All competitors
receive those same answers. The whole-expression predictor remains flexible:
we do not assume independence or multiply the two factor probabilities.
"""
import hashlib
import json

import torch
from torch import nn
from flex_uda.active.teacher_client import LocalTeacher
from flex_uda.teacher.teacher_cache import request_key

MODES = ('WHOLE', 'APPEARANCE', 'STATE')
PROMPTS = {
    'APPEARANCE': 'Consider the indicated object and this description: "{sentence}". '
        'Assess ONLY object category, identity, clothing, colour, and team appearance. '
        'Ignore position, direction, motion, action, and temporary state. '
        'Does the object satisfy the appearance and identity requirements in the description? '
        'If no such requirement is specified, answer Yes. Answer Yes or No.',
    'STATE': 'Consider the indicated object and this description: "{sentence}". '
        'Assess ONLY position, spatial relations, direction, motion, action, and temporary state. '
        'Ignore clothing, colour, team appearance, and identity. '
        'Does the object satisfy the state requirements in the description? '
        'If no such requirement is specified, answer Yes. Answer Yes or No.',
}
PROMPT_SHA = hashlib.sha256(json.dumps(PROMPTS, sort_keys=True).encode()).hexdigest()


class FactorTeacher(LocalTeacher):
    def prepare(self, request):
        _, images, prompt, info = super().prepare(request)
        mode = request['factor_mode']
        assert mode in MODES
        context = ''
        if mode == 'STATE':
            from flex_uda.teacher.decompose_inputs import images_sha
            key = request['target'], request['video_id']
            clips, clip = self.views.clip(key, request['frame_id'], request['track_id'])
            distinct = []
            for index, frame in enumerate(clip['frames']):
                if not distinct or clip['frames'][distinct[-1]] != frame:
                    distinct.append(index)
            chosen = distinct[-2:] if len(distinct) >= 2 else []
            assert request['view'] == 'FRAME_BOX'
            images = images + [clips[i] for i in chosen]
            info['state_history_frames'] = [clip['frames'][i] for i in chosen]
            info['effective_view'] = 'FRAME_BOX_CLIP' if chosen else 'FRAME_BOX'
            info['input']['visual'] = images_sha(images)
            info['input']['method'] = info['effective_view']
            info['input']['state_history_frames'] = info['state_history_frames']
            context = ('The first image shows the scene and the target inside a red box. '
                       'Any additional crops show that same target at earlier and current moments, in time order. ')
        if mode != 'WHOLE':
            location = ('The object is inside the red box. ' if request['view'] == 'FRAME_BOX'
                        else 'The images depict the indicated object. ')
            prompt = context + location + PROMPTS[mode].format(sentence=request['query_text'])
        info['input'] = dict(info['input'], prompt=prompt, factor_mode=mode,
                             factor_prompt_sha=PROMPT_SHA)
        return request_key(info['input']), images, prompt, info


class FactorStudent(nn.Module):
    """Appearance memory attention, state evidence, and a learned joint readout.

    Appearance excludes the 18 explicit geometry/time coordinates. It uses
    current appearance to retrieve compatible anchors, rather than assuming a
    physical Track ID is always pure. This does not prove semantic invariance
    of the frozen PCD, nor protect against all tracking errors.
    """
    def __init__(self, dimension=1101, width=64):
        super().__init__()
        self.appearance = nn.Sequential(nn.Linear(dimension-18, width), nn.SiLU(),
                                        nn.Linear(width, width), nn.SiLU())
        self.state = nn.Sequential(nn.Linear(2*dimension, width), nn.SiLU(),
                                   nn.Linear(width, width), nn.SiLU())
        self.appearance_head = nn.Linear(width, 1)
        self.state_head = nn.Linear(width, 1)
        self.whole = nn.Sequential(nn.Linear(3*width, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, bags, present, current, coarse):
        memory = self.appearance(bags[..., :-18])
        query = self.appearance(current[..., :-18])
        tokens = torch.cat((memory, query[:, None]), dim=1)
        valid = torch.cat((present, torch.ones_like(present[:, :1])), dim=1)
        similarity = (nn.functional.normalize(tokens, dim=-1)
                      * nn.functional.normalize(query, dim=-1)[:, None]).sum(-1)
        attention = similarity.masked_fill(~valid, -torch.inf).softmax(-1)
        identity = (tokens*attention[..., None]).sum(1)
        state = self.state(torch.cat((current, current-coarse), dim=-1))
        whole = self.whole(torch.cat((identity, state, identity*state), dim=-1))
        return torch.cat((whole, self.appearance_head(identity), self.state_head(state)), dim=-1)


class OrdinaryFactorKD(nn.Module):
    """Ordinary unconstrained multi-task soft/hard KD with ALL the same inputs.

    Flattened anchors, their validity, current features and coarse reference
    are available to every output; no teacher factor answer is withheld.
    """
    def __init__(self, dimension=1101, width=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5*dimension+3, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3))

    def forward(self, bags, present, current, coarse):
        values = torch.cat((bags.flatten(1), present.to(current.dtype), current, coarse), dim=-1)
        return self.net(values)
