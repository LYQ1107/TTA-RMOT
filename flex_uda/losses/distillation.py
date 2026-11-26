"""Independent per-candidate soft BCE; never a softmax over people."""
import torch
import torch.nn.functional as F


def soft_distillation(student_margin, teacher_prob, valid, pos_weight=None):
    if student_margin.shape != teacher_prob.shape or valid.shape != student_margin.shape:
        raise ValueError('student/teacher/mask candidate alignment differs')
    if not torch.isfinite(student_margin).all():
        raise FloatingPointError('nonfinite student margin')
    valid = valid.bool()
    target = teacher_prob.detach().float()
    if not torch.isfinite(target).all() or ((target < 0) | (target > 1)).any():
        raise ValueError('invalid raw teacher probability; do not replace it with NEG')
    if not valid.any():
        return student_margin.float().sum() * 0.0
    pw=None if pos_weight is None else torch.tensor(float(pos_weight),device=student_margin.device)
    return F.binary_cross_entropy_with_logits(student_margin[valid].float(), target[valid],pos_weight=pw)
