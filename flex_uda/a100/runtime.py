"""Read-only relocation of the sealed A800 inputs; local A100 outputs."""
from pathlib import Path
import json
import os
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
NAS = Path('/data3/liuyeqiang/nas/A800')
OUTPUT = Path(os.environ.get('TTA_RMOT_A100_RUN_ROOT', ROOT / 'a100_runs/three_stage_20261120')).resolve()
HISTORY = ROOT / 'results/runs/presence_active_20261102'


def relocate(value):
    path = str(value)
    prefix = '/data2/user/FlexHook-TTA/'
    if path.startswith(prefix):
        return ROOT / path[len(prefix):]
    for n in range(4):
        prefix = f'/data{n}/user/'
        if path.startswith(prefix):
            return NAS / f'data{n}/user' / path[len(prefix):]
    return Path(path)


def metadata():
    m = json.loads((ROOT / 'results/runs/multidomain_generality_20261101/sports/metadata.json').read_text())
    for bank in m['videos'].values():
        bank['image_root'] = str(relocate(bank['image_root']))
    return m


def source_head(model_name='Dance-best'):
    e = json.loads((ROOT / 'results/presence_active/redesign_v8p/execution.json').read_text())
    return relocate(e['source_heads'][model_name]['checkpoint'])


def sealed_inputs():
    from flex_tta.runtime.pf_receipts import file_sha
    e = json.loads((ROOT / 'results/presence_active/redesign_v8p/execution.json').read_text())
    rows = json.loads((ROOT / 'results/multidomain_generality/margin_paths.json').read_text())['files']
    r = next(r for r in rows if r['target']=='SportsMOT' and r['model']=='Dance-best' and r['coverage']=='complete')
    p = relocate(r['path'])
    if file_sha(p) != r['sha256']:
        raise ValueError('Changed sealed margin bank')
    return np.load(p, mmap_mode='r'), dict(r, relocated_path=str(p)), relocate(e['stage_A_root'])


def video_spec(video, bank):
    from scripts.active_run import video_spec as original
    return original('SportsMOT', video, bank)


def load_student(output, device, *, verify_weights=True, model_name='Dance-best', target='SportsMOT'):
    import compat.sitecustomize
    from transformers import RobertaTokenizerFast
    from scripts.multidomain_sports_smoke import config
    from scripts.multidomain_kitti_scores import config_for
    from scripts.multidomain_runtime import checkpoint_spec
    from flex_tta.runtime.pf_receipts import file_sha
    begin = time.perf_counter()
    print(json.dumps(dict(event='STUDENT_LOAD_START',device=str(device))),flush=True)
    original_checkpoint, expected = checkpoint_spec(model_name)
    checkpoint = ROOT / 'assets/official_checkpoints' / original_checkpoint.name
    if not checkpoint.is_file():
        # LaMOT was restored as a separate read-only NAS asset, not inside
        # the directory holding the V2/Dance checkpoints. Resolve the same
        # registered official weight; do not write into that NAS directory.
        candidate = relocate(original_checkpoint)
        if model_name == 'LaMOT-best' and not candidate.is_file():
            paths = json.loads((ROOT/'configs/a100_paths.json').read_text())
            candidate = Path(paths['assets']['lamot_checkpoint.pth'])
        checkpoint = candidate
    if verify_weights and file_sha(checkpoint) != expected:
        raise ValueError('Changed official '+model_name+' checkpoint')
    prior_cwd = Path.cwd()
    cfg = config(checkpoint, output) if target == 'SportsMOT' else config_for(checkpoint, output, target)
    cfg.defrost()
    cfg.ROBERTA_PATH = str(ROOT / 'assets/roberta-base')
    # The legacy parser treats a present False argument as enabling these.
    # build_model's RoPE backbone uses its False defaults; make cfg agree.
    cfg.FUSED_WINDOW_PROCESS = False
    cfg.FUSED_LAYERNORM = False
    cfg.freeze()
    from models import build_model
    from data.mydataloader import get_transform
    print(json.dumps(dict(event='STUDENT_VENDOR_IMPORTED',seconds=time.perf_counter()-begin)),flush=True)
    model = build_model(cfg)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['model']
    strict = model.load_state_dict(state, strict=True)
    del state
    model.requires_grad_(False).eval().to(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = RobertaTokenizerFast.from_pretrained(cfg.ROBERTA_PATH, local_files_only=True)
    assert tokenizer.encode('person', add_special_tokens=False) == [5970]
    torch.cuda.synchronize(device)
    print(json.dumps(dict(event='STUDENT_LOAD_DONE',seconds=time.perf_counter()-begin)),flush=True)
    os.chdir(prior_cwd)
    return model, tokenizer, get_transform('test', cfg), dict(
        checkpoint=str(checkpoint), checkpoint_sha256=expected, strict_load=str(strict),
        source_model=model_name, target=target,
        model_load_seconds=time.perf_counter()-begin, precision='FP32',
        checkpoint_hash_verified_in_this_load=verify_weights,
        fused_window_process=cfg.FUSED_WINDOW_PROCESS, fused_layernorm=cfg.FUSED_LAYERNORM)


def teacher(videos, cache_path, device='cuda:0'):
    import yaml
    from flex_uda.active.teacher_client import LocalTeacher, REVISION
    from flex_uda.teacher.yestrack_backend import FrozenYesTrack
    from flex_uda.teacher.decompose_inputs import PairedViews
    from flex_uda.teacher.teacher_cache import TeacherCache
    torch.cuda.set_device(torch.device(device))
    obj = LocalTeacher.__new__(LocalTeacher)
    obj.root = ROOT
    obj.config = yaml.safe_load((ROOT/'configs/teacher/base_qwen2b.yaml').read_text())
    obj.config['model_path'] = str(ROOT/'assets/teacher_Qwen3-VL-2B-Instruct')
    obj.config['yestrack_source'] = str(relocate(obj.config['yestrack_source']))
    assert obj.config['revision'] == REVISION and obj.config['lora_path'] is None
    obj.prompts = yaml.safe_load((ROOT/'configs/teacher/prompts_v1.yaml').read_text())
    obj.backend = FrozenYesTrack(obj.config, device=device)
    obj.views = PairedViews(videos, obj.backend.upstream)
    obj.cache = TeacherCache(cache_path)
    obj.stats = dict(logical_calls=0,new_calls=0,cache_hits=0,forward_seconds=0.,equivalent_forward_seconds=0.,batches=0)
    obj.image_shas = {}
    return obj
