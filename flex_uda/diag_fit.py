"""Training-side D-FIT measurements. This module never reads target GT."""
from collections import defaultdict
import math
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha

POS_THRESHOLD = 0.95  # Actual frozen PSR direct_evidence qualification.
METHODS = ('DIAG_FIT_OPT','DIAG_FIT_OPT_POS','DIAG_FIT_CAP')


def teacher_weight(rows,loss):
    probabilities=[r['teacher_prob'] for r in rows if r['valid']]
    pos=sum(p>=POS_THRESHOLD for p in probabilities);neg=len(probabilities)-pos
    setting=loss.get('pos_weight')
    weight=min(neg/max(pos,1),loss.get('pos_weight_cap',10.0)) if setting=='auto' else setting
    return dict(n_pos=pos,n_neg=neg,pos_weight=weight,POS_threshold=POS_THRESHOLD,
        NEG_definition='valid teacher probability below POS threshold',
        source='sealed teacher probabilities only',target_GT_open_attempts=0)


def fit_subset(pseudo,output):
    indices=np.random.RandomState(17).choice(len(pseudo.groups),300,replace=False).tolist()
    record=dict(seed=17,window_indices=indices,window_ids=[pseudo.groups[i]['window_id'] for i in indices],
        population_windows=len(pseudo.groups),selected_windows=300)
    path=output/'fit_subset.json'
    atomic_json(path,record)
    return [pseudo.groups[i] for i in indices],file_sha(path)


def metrics(records):
    rows=[r for r in records if r['valid']]
    def summarize(part):
        if not part:return dict(selected_rate=None,mean_probability=None,n=0)
        # Stable sigmoid for very large student logits.
        p=[math.exp(-np.logaddexp(0.,-r['student_margin'])) for r in part]
        return dict(selected_rate=sum(r['student_margin']>0 for r in part)/len(part),
            mean_probability=float(np.mean(p)),n=len(part))
    positive=summarize([r for r in rows if r['teacher_probability']>=POS_THRESHOLD])
    negative=summarize([r for r in rows if r['teacher_probability']<POS_THRESHOLD])
    groups=defaultdict(list)
    for row in rows:groups[row['video_id'],row['query_id'],row['frame_id']].append(row)
    agreements=[]
    for group in groups.values():
        pos=[r['student_margin'] for r in group if r['teacher_probability']>=POS_THRESHOLD]
        neg=[r['student_margin'] for r in group if r['teacher_probability']<POS_THRESHOLD]
        agreements.extend(a>b for a in pos for b in neg)
    return dict(teacher_POS_selected_rate=positive['selected_rate'],teacher_NEG_selected_rate=negative['selected_rate'],
        mean_student_prob_on_POS=positive['mean_probability'],mean_student_prob_on_NEG=negative['mean_probability'],
        pair_agreement=float(np.mean(agreements)) if agreements else None,pair_agreement_pairs=len(agreements),
        teacher_POS=positive['n'],teacher_NEG=negative['n'],POS_threshold=POS_THRESHOLD,
        target_GT_open_attempts=0,scope='teacher training anchor fit, not GT accuracy')


def target_windows(pseudo,step,k):
    return [pseudo.groups[(step*k+i)%len(pseudo.groups)] for i in range(k)]


def optimizer_groups(model,params,config):
    if 'head_lr' not in config:return params
    heads={id(p) for p in model.head.parameters()}
    return [dict(params=[p for p in params if id(p) not in heads],lr=config['lr']),
            dict(params=[p for p in params if id(p) in heads],lr=config['head_lr'])]
