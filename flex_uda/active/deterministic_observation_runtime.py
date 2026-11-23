"""Execution settings verified by the teacher-only FULL repeatability probe.

Call in a new worker before initializing CUDA. Existing experiments and
saved outputs retain their original execution configuration.
"""
import os


def configure_before_cuda():
    workspace = ':4096:8'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', workspace)
    if os.environ['CUBLAS_WORKSPACE_CONFIG'] != workspace:
        raise RuntimeError('This execution cohort requires CUBLAS_WORKSPACE_CONFIG=:4096:8')
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('Configure deterministic observation learning before CUDA initialization')
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        CUBLAS_WORKSPACE_CONFIG=workspace, CUDA_initialized_before_configuration=False,
        CUDA_matmul_allow_tf32=False, cudnn_allow_tf32=False,
        torch_version=torch.__version__, cuda_version=torch.version.cuda)
