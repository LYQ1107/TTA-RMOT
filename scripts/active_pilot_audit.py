"""Audit completed real pilots without opening GT or declaring full RMOT success."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import tarfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.active.execution import authorize_active
from flex_uda.active.teacher_client import REVISION
from flex_uda.presence.runtime import RUN,target_metadata
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.presence_redesign_report import table


def main():
    p=ArgumentParser(description=__doc__);p.add_argument('--label',required=True);a=p.parse_args()
    execution,publication=authorize_active();rows=[];files=[];snapshots={}
    for pair in ('C1','C3'):
        root=execution.active / pair / 'FULL/rho_0.3';path=root / 'pilot_receipt.json'
        r=json.loads(path.read_text());snapshots[pair]=r
        meta=target_metadata(r['target']);rounds=[z for v in r['video_receipts'] for z in v['rounds']]
        teacher=r['teacher'];updates=sum(z['slow']['optimizer_updates'] for z in rounds)
        if r['GT_open_attempts'] or teacher['GT_open_attempts'] or teacher['revision']!=REVISION:
            raise ValueError('Pilot GT/revision audit failed')
        for v in r['video_receipts']:
            mask=root / 'videos' / v['video'] / 'mask.npy'
            if file_sha(mask)!=v['mask_sha256']:raise ValueError('Pilot mask changed')
            if sum(z['actual_questions'] for z in v['rounds'])!=v['questions'] or v['questions']>v['budget']:
                raise ValueError('Pilot budget accounting failed')
            expected=[q['expression_id'] for q in meta['episodes'] if q['video_id']==v['video']]
            feature=RUN / 'features' / r['target'] / r['model'] / 'videos' / v['video'] / 'metadata.json'
            local=json.loads(feature.read_text())
            if expected!=[q['expression_id'] for q in local['queries']]:raise ValueError('Prior query-axis alignment failed')
        grads=[z['gradient_norm'] for row in rounds for z in row['slow'].get('records',[]) if z.get('optimizer_updated')]
        if len(grads)!=updates or (grads and (not np.isfinite(grads).all() or max(grads)<=0)):
            raise ValueError('Actual optimizer/gradient audit failed')
        checkpoint=sorted(root.glob('stream_*.pt'))[-1];state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        b=[v for k,v in state['adapter'].items() if '.lora_b.' in k]
        bnorm=float(sum(t.double().square().sum() for t in b).sqrt())
        if not np.isfinite(bnorm) or (updates>0 and bnorm==0):raise ValueError('LoRA B never changed from zero')
        rows.append(dict(pair=pair,videos=len(r['videos']),teacher_calls=sum(v['questions'] for v in r['video_receipts']),
            new_forward_calls=teacher['new_calls'],optimizer_updates=updates,
            gradient_min=min(grads) if grads else None,gradient_max=max(grads) if grads else None,
            LoRA_B_norm=bnorm,teacher_forward_seconds=teacher['forward_seconds'],
            LoRA_seconds=r['LoRA_GPU_seconds'],rescore_seconds=r['rescore_GPU_seconds'],
            final_positive_pairs=sum(v['rounds'][-1]['positive_pairs'] for v in r['video_receipts']),
            confident_never_asked=sum(v['rounds'][-1]['confident_pairs_never_asked'] for v in r['video_receipts']),
            GT_open_attempts=0,full_RMOT_evaluation=False))
        files.extend(dict(path=str(p),sha256=file_sha(p)) for p in [path,checkpoint])
    status='BOTH_REAL_UPDATES_MEASURED' if all(r['optimizer_updates']>0 for r in rows) else 'REAL_PILOT_COMPLETE_SOME_SLOW_UPDATES_NOT_OBSERVED'
    label=a.label;record=execution.public;archive=record / ('pilot_'+label+'_logs.tar.gz')
    log_files=[]
    for pair in ('C1','C3'):
        root=execution.active / pair / 'FULL/rho_0.3'
        log_files += [p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.log')]
    with tarfile.open(archive,'w:gz') as handle:
        for path in sorted(log_files):handle.add(path,arcname=str(path.relative_to(execution.active)),recursive=False)
    result=dict(status=status,rows=rows,pilot_snapshots=snapshots,files=files,
        archive=dict(path=str(archive),sha256=file_sha(archive)),stage_A_publication=publication,
        no_GT_or_new_model_calls_by_auditor=True,full_method_success_not_established=True)
    atomic_json(record / ('pilot_'+label+'.json'),result)
    report=record / 'REPORT.md';text=report.read_text();marker='\n## 真实pilot最新状态'
    if marker in text:text=text[:text.index(marker)]
    text+=marker+'\n\n'+status+'。尚无完整B/C HOTA。\n\n'+table(list(rows[0]),rows)
    text+=('\n高置信不等于正确，未问过即高置信单列；GT未进入本诊断，不据此调整阈值。'
        'C1缺少有效互斥列表样本时应跳过，不能强制伪正例制造梯度；完整矩阵成本门槛仍须真实C1/C3更新计时，未通过不展开108项。'
        '每次pilot快照、日志和stream checkpoint SHA独立保留；后续按视频顺序恢复，不重做健康首视频。\n')
    report.write_text(text);print(json.dumps(dict(status=status,rows=rows)),flush=True)


if __name__=='__main__':main()
