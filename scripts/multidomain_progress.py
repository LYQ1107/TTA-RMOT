"""Publishable P0-resume facts, retaining the original stopped inventory."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.multidomain_runtime import *


def main():
    gate=require_plan();dest=RUN/'kitti'
    smokes={m:json.loads((dest/m/'smoke_receipt.json').read_text()) for m in CHECKPOINTS}
    if any(r['status']!='SMOKE_PASS' or not r['cost_gate_pass'] or r['GT_open_attempts'] for r in smokes.values()):
        raise ValueError('P0 smoke/cost/GT gate failed')
    for m,r in smokes.items():atomic_json(PUBLIC/('smoke_'+m+'.json'),r)
    old=RUN/'history_p0_stop'
    inventory=json.loads((old/'inventory.json').read_text())
    inventory.update(status='P0_RESUMED_C1_C2_KITTI_AUTHORIZED_SPORTS_LAMOT_ASSET_BLOCKED',
        amendment_publication=gate,SportsMOT_seqmap_authorized_sequences=466,
        downloads_record=str(PUBLIC/'downloads.json'),smokes=smokes,
        independent_work_allowed=['C1','C2','KITTI_complete_tail_scores'],
        SportsMOT_LaMOT_full_scope_preserved=True,original_inventory=str(old/'inventory.json'))
    inventory['SportsMOT_seqmap_mismatch']['user_confirmed_full_466']=True
    atomic_json(PUBLIC/'inventory.json',inventory)
    plan=json.loads((old/'smoke_and_costs.json').read_text())
    plan.update(status='KITTI_TWO_OFFICIAL_CHECKPOINT_SMOKES_PASS',completed_smokes=smokes,
        max_simultaneous_GPUs=2,actual_successful_smoke_GPU_process_hours=sum(r['wall_seconds'] for r in smokes.values())/3600,
        cost_stop_over_6_GPU_hours=True)
    atomic_json(PUBLIC/'smoke_and_costs.json',plan)
    files=json.loads((PUBLIC/'downloads.json').read_text())
    last=files['attempts'][-1]
    lines=['# P0恢复盘点（补充预注册后，P1前）','',
        '补充预注册远端：'+gate['remote_commit']+'。原始PREREG正文未改，完整C1–C6范围恢复；Sports按466条。','',
        '| KITTI目标/官方源权重 | strict | 完整一视频margin数 | 耗时(s) | 全量估算GPU h | GT打开 |',
        '|---|---|---:|---:|---:|---:|']
    for m,r in smokes.items():lines.append(f"| {m} | 全部key一致 | {r['new_margin_computations']} | {r['wall_seconds']:.3f} | {r['estimated_full_scoring_GPU_hours']:.6f} | {r['GT_open_attempts']} |")
    lines+=['','两套均用官方kitti2.yaml+infer.sh的rope-swin-tiny/roberta覆盖、FP32、同一官方RMOT_Dataset test；实际getitem/forward对照无符号翻转，box/grid一致。类别不添加新token、不替换car/ped语义。',
        '缓存只复用同一个冻结checkpoint的编码；两模型缓存独立。源历史分数不被重算覆盖，源V2 smoke重算的历史最大差0.0010823如实记录（历史库原值保留）。',
        '',f"公开下载已有{len(files['files'])}文件（含官方LaMOT原始Sports JSON508份、README、树和压缩包），磁盘下载前余{files['disk_free_bytes_before']}字节。原始JSON数字ID命名与FlexHook表达名seqmap并不直接相等，不能把下载成功宣称为466条封装核验通过。",
        '百度提取码已通过，匿名能列出995768578字节的LaMOT-best和45个McByte预测文件；匿名sharedownload返回errno=2（参数/签名未满足），没有返回下载链接。未将errno=2谎称为已证实的登录错误。公开BaiduPCS-Go文档将登录列为必做；无可用账号凭据，不安装需登录工具/不索取密码，不绕过签名。该路径停止，精确手动文件见downloads.json。',
        'FlexHook代码MIT；单独checkpoint/数据授权未列；LaMOT仓库无独立LICENSE，不假定其数据继承MIT。未执行下载内容代码，未改vendor代码，未重下载Sports图像，未自跑tracker。',
        '', 'C1复用路径和SHA已核验，C2及KITTI补分可按用户授权继续。Sports/LaMOT所有组合仍受资产核验门槛约束；不能把2/6完成当三域普遍性成功。',
        '','首次启动环境选错（vllmedit缺yacs），切回既有dfot且目标配置依赖回归通过。没有改动共享环境。smoke成功后先暂停P1到此记录远端发布；健康0005视频保留，不重跑。',
        '','论文最新版arXiv v5正文仅LaMOT汇总56.77，infer.sh汇总56.756；未找到Sports子集官方数字。仓库无补充PDF，CVF补充页面403；不将不存在的子集数字充作自检门槛。',
        '','论文：https://arxiv.org/html/2503.07516v5；官方代码：https://github.com/buptLwz/FlexHook；工具登录说明：https://github.com/qjfoidnh/BaiduPCS-Go#初级使用教程。']
    (PUBLIC/'P0_RESUMED.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=inventory['status'],estimates={m:r['estimated_full_scoring_GPU_hours'] for m,r in smokes.items()})),flush=True)


if __name__=='__main__':main()
