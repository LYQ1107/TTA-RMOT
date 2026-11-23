"""Publishable resumed P0: authenticated assets, strict video smokes and costs."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.multidomain_runtime import RUN,PUBLIC,CHECKPOINTS,require_plan
from scripts.multidomain_remaining_launcher import JOBS,output_folder
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def main():
    gate=require_plan();assets=json.loads((PUBLIC/'sports_assets.json').read_text())
    download=json.loads((PUBLIC/'authenticated_downloads.json').read_text())
    if download['missing_selected_payloads'] or assets['annotation_mismatches'] or assets['github_byte_identical']!=466:
        raise ValueError('Official asset/text byte equivalence gates failed')
    smokes=[]
    for target,model in JOBS:
        p=output_folder(target,model)/'smoke_receipt.json';s=json.loads(p.read_text())
        if s['status']!='SMOKE_PASS' or not s['cost_gate_pass'] or s['estimated_full_scoring_GPU_hours']>6 or s['GT_open_attempts']:
            raise ValueError('Stop: failed strict/loader/GT/cost gate for '+target+' '+model)
        if s['strict_load']!='<All keys matched successfully>':
            raise ValueError('All official state_dict keys must match exactly')
        smokes.append(dict(target=target,model=model,receipt_path=str(p),receipt_sha256=file_sha(p),
            strict_load=s['strict_load'],GT_open_attempts=0,
            estimated_full_scoring_GPU_hours=s['estimated_full_scoring_GPU_hours'],
            smoke_wall_seconds=s['wall_seconds'],physical_gpu=s['physical_gpu'],
            peak_allocated_bytes=s['peak_allocated_bytes']))
    result=dict(status='ALL_REMAINING_P0_GATES_PASS_REQUIRES_PUBLICATION_BEFORE_P1',
        prereg_publication=gate,models={m:dict(path=str(p),sha256=h) for m,(p,h) in CHECKPOINTS.items()}|
            {'LaMOT-best':assets['checkpoint']},
        SportsMOT=assets,smokes=smokes,max_GPUs=2,sharing_authorized=True,
        GT_template_completion='AUTHORIZED_V3_POST_SEAL_418_EXACT_GATE_PENDING',
        official_SportsMOT_subset_HOTA='NO_OFFICIAL_SUBSET_NUMBER_FOUND_NO_NUMERICAL_GATE',
        official_number_search=dict(paper='https://arxiv.org/html/2503.07516v5',
            repository='https://github.com/buptLwz/FlexHook',
            share_root_contains=['datasets','pretrained','SOTA_ckpts','tracker_outputs'],
            combined_LaMOT_only='infer.sh 56.756 / paper 56.77; not SportsMOT alone'),
        no_training=True,no_MLLM=True,GT_open_attempts=0,
        source_self_checks='Existing sealed historical V2 KITTI42.526 and Dance32.134 / complete32.139 unchanged')
    atomic_json(PUBLIC/'remaining_p0.json',result)
    inventory=json.loads((PUBLIC/'inventory.json').read_text())
    inventory.update(status='OFFICIAL_ASSETS_RECOVERED_ALL_REMAINING_P0_SMOKES_PASS',
        remaining_P0=str(PUBLIC/'remaining_p0.json'),completed_combinations=['C1','C2'],
        pending_combinations=['C3','C4','C5','C6'],all_official_assets_downloaded=True,
        source_self_checks_unchanged=True,full_scope_preserved=True)
    inventory['models']['LaMOT-best']=dict(assets['checkpoint'],exists=True,status='OFFICIAL_STRICT_LOAD_PASS')
    inventory['targets']['SportsMOT'].update(videos=45,expressions=466,actual_seqmap_expressions=466,
        full_candidate_observations=assets['full_candidate_observations'],
        full_candidate_expression_pairs=assets['full_candidate_query_rows'],
        official_tracker_coverage=45,official_expression_coverage=466,GT_template_originals=418,
        GT_template_missing_authorized_reconstruction=48,per_video=assets['tracker_records'])
    atomic_json(PUBLIC/'inventory.json',inventory)
    record=json.loads((PUBLIC/'downloads.json').read_text())
    record['asset_gates'].update(LaMOT_strict_load='PASS_ALL_KEYS_MATCHED',
        McByte_45_video_coverage='PASS_45_OFFICIAL_FILES_295779_OBSERVATIONS',
        annotations466='PASS_466_SEQMAP_EXACT_466_GITHUB_BYTE_IDENTICAL',
        LaMOT_Sports_smoke='PASS_ORIGINAL_MIX_LOADER_REAL_COMPLETE_VIDEO',
        C3_C4_C5_C6='ALL_P0_PASS_AWAIT_P0_PUBLICATION_THEN_P1',
        official_GT_templates='418_OFFICIAL_PLUS_48_AUTHORIZED_POST_SEAL_EXACT_RECONSTRUCTION_GATE')
    record['status']='OFFICIAL_ASSETS_READY_AND_ALL_P0_SMOKES_PASSED'
    atomic_json(PUBLIC/'downloads.json',record)
    lines=['# 剩余四组合恢复 P0（全量推理前发布）','',
        '项目：TTA-RMOT。完整C1–C6范围不变，C1/C2不重跑；不训练、不调用MLLM，最多两卡。','',
        f"LaMOT-best SHA256：`{assets['checkpoint']['sha256']}`；大小995768578字节；strict=True全部key一致。",'',
        'SportsMOT：45视频、466表达、26970原始有效帧、295779个McByte观测、2968201个候选–表达对。',
        '466官方表达与GitHub原始文件逐字节一致；只独立导出language，原始targets/labels不进模型。',
        'Stock mix loader读取空对象形式的视频索引（不是伪GT）和纯文本；GT审计0。',
        '原图仅建数字JPG软链接视图，排除macOS ._资源分叉，官方TrackEval的目录计帧与seqLength一致；原图不修改。','',
        '| 源模型 | 目标 | 实机卡 | smoke秒 | 全量估计GPU小时 | GT打开 |',
        '|---|---|---|---|---|---|']
    for s in smokes:
        lines.append(f"|{s['model']}|{s['target']}|{s['physical_gpu']}|{s['smoke_wall_seconds']:.3f}|{s['estimated_full_scoring_GPU_hours']:.6f}|0|")
    lines+=['','所有smoke覆盖预先固定的第一个完整视频，原生getitem/forward、缓存特征与隔离query均通过既有1e-3/无符号翻转门。',
        'Sports使用原mcbyte-sports-infer-tiny.yaml/mix loader/4帧/25token/224×672预处理，静态class=0、不加类别token。',
        'KITTI/Dance按各自原yaml和infer.sh的rope-swin-tiny/roberta，FP32/TF32禁用；源模型仅换官方权重。','',
        '无Sports子集官方HOTA可核对：v5正文和infer.sh只给LaMOT四子集合并值，分享根目录无额外补充材料。',
        '执行方式为单进程冻结encoder缓存、原始masked decoder，逐视频真实forward对齐；不宣称实际运行了DDP main.py自动GT评测入口。','',
        'GT模板原包仅418/466。v3先发布后，预测完整封存后才允许独立读取原始targets与Sports官方框；逐行复现418才接入48，任何失败停止Sports评测。',
        '本阶段未打开GT值，未进行新完整评测，不把smoke通过当科研结论。完整日志和恢复路径见remaining_p0.json。']
    (PUBLIC/'P0_REMAINING.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],smokes=smokes)),flush=True)


if __name__=='__main__':main()
