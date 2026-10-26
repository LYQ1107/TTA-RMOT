import re
import torch
import torch.nn.functional as F


def supervised_ce(all_logits,labels,regular=0.):
    n,l,q,c=all_logits.shape
    targets=labels[:,None,None].expand(n,l,q).reshape(-1)
    return F.cross_entropy(all_logits.float().reshape(-1,c),targets)+regular


def presence_probability(margin):
    # Independent Bernoulli candidates; never softmax over objects.
    return -torch.expm1(F.logsigmoid(-margin.float()).sum())


def presence_loss(margin,state):
    if state=='UNKNOWN':return margin.sum()*0
    if state not in ('SUPPORTED_PRESENT','SUPPORTED_ABSENT'):raise ValueError('unknown presence state')
    p=presence_probability(margin).clamp(1e-6,1-1e-6)
    return F.binary_cross_entropy(p,p.new_tensor(float(state=='SUPPORTED_PRESENT')))


def select_pseudo(teacher,source,student,*,threshold=.9,tolerance=.1):
    p=teacher.detach().float().sigmoid();s=source.detach().float().sigmoid();v=student.detach().float().sigmoid()
    agree=(p-v).abs()<=tolerance
    pos=(p>=threshold)&(s>=threshold)&agree
    neg=(p<=1-threshold)&(s<=1-threshold)&agree
    proposed_negative=int(neg.sum())
    # Do not train hard labels using only the known negative-biased tail.
    if not pos.any():neg=torch.zeros_like(neg)
    state=torch.full_like(p,-1,dtype=torch.long);state[pos]=1;state[neg]=0
    return state,dict(pseudo_positive=int(pos.sum()),pseudo_negative=int(neg.sum()),pseudo_unknown=int((state<0).sum()),
        proposed_negative=proposed_negative,teacher_student_disagreement=float((p-v).abs().mean()),
        reliability_is_precision_proxy_not_GT=True)


def pseudo_bce(student,state):
    parts=[]
    for label in (0,1):
        mask=state==label
        if mask.any():parts.append(F.binary_cross_entropy_with_logits(student[mask].float(),student.new_full((int(mask.sum()),),float(label)).float()))
    return torch.stack(parts).mean() if parts else student.sum()*0


def consistency(student,teacher):
    # Logit consistency avoids zero FP16 derivatives in the known very
    # negative tail. It is not a hard NEG label or evidence of correctness.
    return F.smooth_l1_loss(student.float(),teacher.detach().float())


def semantic_kind(text):
    value=text.lower()
    static=any(w in value for w in ('shirt','pants','skirt','dressed','wear','clothes'))
    dynamic=any(w in value for w in ('dancing','walking','running','turning','moving','sitting'))
    spatial=any(w in value for w in ('left','right','behind','front','between','near'))
    active=sum((static,dynamic,spatial))
    if active>1:return 'MIXED'
    return ('STATIC_APPEARANCE' if static else 'CURRENT_DYNAMIC_STATE' if dynamic else 'SPATIAL_RELATION' if spatial else 'UNKNOWN')


def exclusive_query_partner(text,texts):
    # Only literal whole-outfit single-colour statements are declared
    # incompatible, never all clothing queries or different people.
    pattern=r'dressed all in (black|white|brown|green|purple)\b'
    match=re.search(pattern,text.lower())
    if match:
        for other in sorted(texts):
            candidate=re.search(pattern,other.lower())
            if candidate and candidate.group(1)!=match.group(1):return other
    return None


def exclusive_relation_loss(first,second):
    return F.relu(first.float().sigmoid()+second.float().sigmoid()-1).square().mean()


def equivalent_query(text):
    match=re.fullmatch(r'dancing person dressed all in (black|white|brown|green|purple)',text.lower().strip())
    if match:return 'a person who is dancing and dressed entirely in '+match.group(1)
    match=re.fullmatch(r'dancing person with (.+(?:shirt|tshirt|pants|skirt).*)',text.lower().strip())
    if match:return 'a dancing person wearing '+match.group(1)
    return None  # No assumed interpretation/correction of the original typo.
