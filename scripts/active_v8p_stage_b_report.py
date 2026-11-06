"""Post-seal conditional B tables; never imported by generation workers."""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.active.execution import authorize_active
from flex_uda.presence.predictions import verify_predictions
from scripts.active_run import PAIRS
from scripts.active_v8_stage0_gt import write_csv
from scripts.active_v8_stage0_report import table


def report(execution,tier):
    from scripts.active_eval import bootstrap
    public=execution.public
    plan=json.loads((public/('registered_jobs_'+tier+'.json')).read_text())
    reused={r['pair']:Path(r['folder']) for r in json.loads((public/'inputs.json').read_text())['reused_v8']}
    def main(pair,rho=.3):
        return reused[pair] if rho==.3 and pair in reused else execution.active/pair/'MINUS_SLOW'/('rho_'+str(rho))
    rows=[];intervals={};artifacts=[]
    for job in plan['jobs']:
        pair,method,rho=job['pair'],job['method'],job['rho']
        path=execution.active/pair/method/('rho_'+str(rho));seal=verify_predictions(path)
        m=json.loads((path/'offline_summary.json').read_text())
        assert m['prediction_seal_sha256']==file_sha(path/'predictions_seal.json')
        teacher=seal['teacher'];base=main(pair,rho)
        b=json.loads((base/'offline_summary.json').read_text())
        row=dict(job,HOTA=m['HOTA_unrounded'],DetA=m['DetA'],AssA=m['AssA'],IDF1=m['IDF1'],IDSW=m['IDSW'],
            prediction_rows=m['prediction_rows'],query_empty_prediction_rows=m['query_empty_prediction_rows'],
            delta_vs_MINUS=m['HOTA_unrounded']-b['HOTA_unrounded'],
            logical_calls=teacher['logical_calls'],new_calls=teacher.get('new_calls',0),cache_hits=teacher.get('cache_hits',0),
            teacher_GPU_hours=teacher.get('forward_seconds',0)/3600,
            teacher_equivalent_GPU_hours=teacher.get('equivalent_forward_seconds',0)/3600,
            LoRA_GPU_hours=seal.get('LoRA_GPU_seconds',0)/3600,
            rescore_GPU_hours=seal.get('rescore_GPU_seconds',0)/3600,wall_seconds=seal['wall_seconds'],GT_open_attempts=0,
            path=str(path),prediction_seal_sha256=file_sha(path/'predictions_seal.json'),reused=False)
        rows.append(row);artifacts.append(dict(job,path=str(path),sha256=row['prediction_seal_sha256'],episodes=seal['episodes']))
        intervals[pair+'__'+method+'__'+str(rho)+'_minus_MINUS']=bootstrap(path,base,PAIRS[pair][1])
        row['CI95_vs_MINUS']=intervals[pair+'__'+method+'__'+str(rho)+'_minus_MINUS']['CI95']
    if tier=='B1':
        for pair in ('C2','C3','C4','C5'):
            path=Path(execution.manifest['prior_v8_active_root'])/pair/'MINUS_SLOW/rho_0.3'
            seal=verify_predictions(path);m=json.loads((path/'offline_summary.json').read_text())
            assert m['prediction_seal_sha256']==file_sha(path/'predictions_seal.json')
            base=json.loads((main(pair)/'offline_summary.json').read_text())
            rows.append(dict(pair=pair,method='MINUS_PRESUPPOSITION',rho=.3,HOTA=m['HOTA_unrounded'],
                DetA=m['DetA'],AssA=m['AssA'],IDF1=m['IDF1'],IDSW=m['IDSW'],
                prediction_rows=m['prediction_rows'],query_empty_prediction_rows=m['query_empty_prediction_rows'],
                delta_vs_MINUS=m['HOTA_unrounded']-base['HOTA_unrounded'],path=str(path),
                prediction_seal_sha256=file_sha(path/'predictions_seal.json'),reused=True,new_calls=0,cache_hits=0,
                logical_calls=seal['teacher']['logical_calls'],teacher_GPU_hours=0,
                original_reused_method_teacher_GPU_hours=seal['teacher'].get('forward_seconds',0)/3600,
                teacher_equivalent_GPU_hours=seal['teacher'].get('equivalent_forward_seconds',0)/3600,
                GT_open_attempts=seal['GT_open_attempts']))
            intervals[pair+'__MINUS_PRESUPPOSITION']=bootstrap(path,main(pair),PAIRS[pair][1])
            rows[-1]['CI95_vs_MINUS']=intervals[pair+'__MINUS_PRESUPPOSITION']['CI95']
    if tier=='B2':
        with (public/'direction_summary.csv').open() as handle:old=list(csv.DictReader(handle))
        for r in old:
            for method,field in [('MINUS_SLOW','v8P'),('YESTRACK_BUDGET_PRIOR','YT_PRIOR'),('UNIFORM_K','UNIFORM_K')]:
                source=main(r['pair']) if method=='MINUS_SLOW' else Path(execution.manifest['prior_v5_active_root'])/r['pair']/method/'rho_0.3'
                inherited=verify_predictions(source)
                sm=json.loads((source/'offline_summary.json').read_text())
                assert sm['prediction_seal_sha256']==file_sha(source/'predictions_seal.json')
                assert sm['HOTA_unrounded']==float(r[field])
                rows.append(dict(pair=r['pair'],method=method,rho=.3,HOTA=float(r[field]),reused=True,
                    DetA=sm['DetA'],AssA=sm['AssA'],IDF1=sm['IDF1'],IDSW=sm['IDSW'],
                    prediction_rows=inherited['prediction_rows'],logical_calls=inherited['teacher']['logical_calls'],
                    new_calls=0,cache_hits=0,teacher_GPU_hours=0,
                    original_reused_method_teacher_GPU_hours=inherited['teacher'].get('forward_seconds',0)/3600,
                    teacher_equivalent_GPU_hours=inherited['teacher'].get('equivalent_forward_seconds',0)/3600,
                    path=str(source),prediction_seal_sha256=file_sha(source/'predictions_seal.json'),
                    budget_note='rho=.3 retained historical controls, as explicitly registered'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for pair,ax in zip(PAIRS,axes.flat):
            for method in ('MINUS_SLOW','YESTRACK_BUDGET_PRIOR','UNIFORM_K'):
                group=sorted([r for r in rows if r['pair']==pair and r['method']==method],key=lambda r:r['rho'])
                ax.plot([r['rho'] for r in group],[r['HOTA'] for r in group],marker='o',label=method)
            ax.set(title=pair,xlabel='Budget rho',ylabel='HOTA');ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(public/'budget_curves.png',dpi=150);plt.close(fig)
    write_csv(public/(tier+'_summary.csv'),rows)
    atomic_json(public/(tier+'_bootstrap.json'),intervals)
    atomic_json(public/(tier+'_prediction_paths.json'),dict(files=artifacts))
    text='## '+tier+'\n\n'+table(rows,['pair','method','rho','HOTA','DetA','AssA','IDF1','delta_vs_MINUS','CI95_vs_MINUS','logical_calls','new_calls','cache_hits','reused'])+'\n\n'
    text+='### 成本\n\n'+table(rows,['pair','method','rho','teacher_GPU_hours','original_reused_method_teacher_GPU_hours','teacher_equivalent_GPU_hours','LoRA_GPU_hours','rescore_GPU_hours','wall_seconds','GT_open_attempts'])+'\n\n'
    text+='复用行的teacher_GPU_hours=0只指本阶段没有新增forward，不是该系统无需GPU；原方法实际成本在original_reused_method_teacher_GPU_hours，缓存等价forward成本另列。\n\n'
    if tier=='B2':text+='![预算曲线](budget_curves.png)\n\n预算 .1/1.0 每视频实际调用精确匹配；.3 按预注册直接复用历史对照。\n\n'
    (public/(tier+'_REPORT.md')).write_text(text)
    base=(public/'REPORT.md').read_text().split('\n<!-- CONDITIONAL_B -->')[0]
    sections=[(public/(t+'_REPORT.md')).read_text() for t in ('B1','B2','B3') if (public/(t+'_REPORT.md')).exists()]
    (public/'REPORT.md').write_text(base+'\n<!-- CONDITIONAL_B -->\n\n'+'\n'.join(sections))
    print(json.dumps(dict(event='V8P_CONDITIONAL_REPORT_CLOSED',tier=tier,rows=len(rows))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tier',choices=['B1','B2','B3'],required=True)
    report(authorize_active()[0],p.parse_args().tier)
