"""Exact official pooling, expression/category splits and video bootstrap for B."""
from argparse import ArgumentParser
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decompose_common import *
from scripts.analyze_pf_video_uncertainty import HOTA,Identity,unpack,sufficient,pooled_hota,paired_clusters


def load_result(path,seal=None):
    done=json.loads((path/'official_completion.json').read_text())
    for f,k in [('pedestrian_detailed.csv','detailed_sha256'),('pedestrian_summary.txt','summary_sha256')]:
        if file_sha(path/'results'/f)!=done[k]:raise ValueError('Changed official CSV')
    with (path/'results/pedestrian_detailed.csv').open() as h:raw=list(csv.DictReader(h))
    rows={r['seq']:r for r in raw if r['seq']!='COMBINED'}
    if seal is None:seal=json.loads((path/'predictions_seal.json').read_text())
    hota=HOTA({'PRINT_CONFIG':False});identity=Identity({'PRINT_CONFIG':False})
    hrows={key:unpack(r,hota) for key,r in rows.items()}
    irows={key:unpack(r,identity) for key,r in rows.items()}
    videos=sorted({e['video'] for e in seal['episodes']})
    stat=sufficient([hota.combine_sequences({k:r for k,r in hrows.items() if k.startswith(v+'+')}) for v in videos])
    exact=float(pooled_hota(stat,np.ones(len(videos))))
    combined=next(r for r in raw if r['seq']=='COMBINED')
    if not np.isclose(exact,100*float(combined['HOTA___AUC']),atol=1e-8,rtol=0):raise ValueError('Official bootstrap pooling mismatch')
    return dict(path=path,done=done,seal=seal,raw=rows,hrows=hrows,irows=irows,stats=stat,videos=videos,HOTA=exact)


def selected_metrics(result,keys,present):
    keys=set(keys)
    if not keys:
        return dict(HOTA=None,DetA=None,AssA=None,IDF1=None,IDSW=None,prediction_rows=0,query_empty_prediction_rows=0,expressions=0)
    hota=HOTA({'PRINT_CONFIG':False});identity=Identity({'PRINT_CONFIG':False})
    h=hota.combine_sequences({k:result['hrows'][k] for k in keys})
    i=identity.combine_sequences({k:result['irows'][k] for k in keys})
    episodes={e['video']+'+'+e['expression']:e for e in result['seal']['episodes']}
    return dict(HOTA=100*float(np.mean(h['HOTA'])),DetA=100*float(np.mean(h['DetA'])),AssA=100*float(np.mean(h['AssA'])),
                IDF1=100*float(i['IDF1']),IDSW=sum(int(float(result['raw'][k]['IDSW'])) for k in keys),
                prediction_rows=sum(episodes[k]['selected_rows'] for k in keys),
                query_empty_prediction_rows=sum(episodes[k]['selected_rows'] for k in keys if not present[k]),expressions=len(keys))


def subset_stats(result,keys):
    hota=HOTA({'PRINT_CONFIG':False});selected={k:r for k,r in result['hrows'].items() if k in keys}
    return sufficient([hota.combine_sequences({k:r for k,r in selected.items() if k.startswith(v+'+')}) for v in result['videos']])


def report(combos):
    require_plan();plan=json.loads((RUN/'probe/plan.json').read_text())
    if file_sha(plan['expression_types']['path'])!=plan['expression_types']['sha256']:raise ValueError('Changed preregistered taxonomy')
    taxonomy=list(csv.DictReader((PUBLIC/'expression_types.csv').open()))
    cats=json.loads((OLD/'source_query_categories.json').read_text())['mapping']
    missing=set(json.loads((OLD_PUBLIC/'sports_template_inventory.json').read_text())['missing_GT'])
    rows=[];splitrows=[];sensitivity=[];sensitivity_splits=[];bootstrap={};rhofree=[];granularity=[];manifests=[];stats_export={}
    for combo,model,target in PAIRS:
        if combo not in combos:continue
        meta=load_target(target)[0]
        oracle=json.loads((RUN/'oracle'/f'{combo}.json').read_text())
        present={q['video']+'+'+q['expression']:q['present'] for q in oracle['episodes']}
        methods={n:load_result(folder(combo,n),verify(combo,n)) for n in (*ORACLES,*CALIBRATORS)}
        methods.update({n:load_result(previous.base(target,model)/n,previous.verify(target,n,model)) for n in ('NATIVE','IN_DOMAIN','R_SxK_I')})
        h={n:r['HOTA'] for n,r in methods.items()};gap=h['IN_DOMAIN']-h['NATIVE']
        required=next((n for n in ORACLES[:4] if h[n]>=h['R_SxK_I']),'需要逐帧数量')
        granularity.append(dict(combo=combo,source=model,target=target,required_granularity=required,
                                R_SxK_I_HOTA=h['R_SxK_I'],native_HOTA=h['NATIVE'],in_domain_HOTA=h['IN_DOMAIN'],gap=gap))
        type_by={r['video']+'+'+r['expression']:r['types'].split('|') for r in taxonomy if r['target']==target}
        for name,result in methods.items():
            allkeys=set(result['hrows'])
            m=selected_metrics(result,allkeys,present)
            rows.append(dict(combo=combo,source=model,target=target,configuration=name,oracle=name in ORACLES,
                             GT_protocol=oracle['GT_protocol'],**m))
            if name in CALIBRATORS:rhofree.append(dict(combo=combo,source=model,target=target,calibrator=name,
                HOTA=h[name],native_HOTA=h['NATIVE'],gap=gap,rho_free=(h[name]-h['NATIVE'])/gap if gap else None))
            scopes=[('expression_type',t) for t in (*WORDS,'OTHER')]
            if target=='KITTI-V2':scopes+=[('target_category',c) for c in ('car','pedestrian')]
            for group,label in scopes:
                keys={k for k in allkeys if label in type_by[k]} if group=='expression_type' else {k for k in allkeys if cats[k]==label}
                splitrows.append(dict(combo=combo,source=model,target=target,configuration=name,split_kind=group,split=label,
                                      **selected_metrics(result,keys,present)))
            if target=='SportsMOT':
                keys=allkeys-missing
                if len(keys)!=418:raise ValueError('Official coverage sensitivity must have exactly418 sequences')
                sensitivity.append(dict(combo=combo,source=model,target=target,configuration=name,
                    GT_protocol='V4_GT_RESTRICTED_TO_OFFICIAL_418_COVERAGE',**selected_metrics(result,keys,present)))
                for label in (*WORDS,'OTHER'):
                    selected={k for k in keys if label in type_by[k]}
                    sensitivity_splits.append(dict(combo=combo,source=model,target=target,configuration=name,
                        expression_type=label,GT_protocol='V4_GT_RESTRICTED_TO_OFFICIAL_418_COVERAGE',
                        **selected_metrics(result,selected,present)))
            manifest=dict(combo=combo,configuration=name,path=str(result['path']),
                seal_path=str(result['path']/'predictions_seal.json'),seal_sha256=file_sha(result['path']/'predictions_seal.json'),
                predictions=len(result['seal']['episodes']),prediction_rows=result['seal']['prediction_rows'],
                detailed_path=str(result['path']/'results/pedestrian_detailed.csv'),detailed_sha256=result['done']['detailed_sha256'],
                official_completion_path=str(result['path']/'official_completion.json'),
                official_completion_sha256=file_sha(result['path']/'official_completion.json'),
                official_completion=result['done'])
            manifests.append(manifest)
            num,den=result['stats'];stats_export[combo+'__'+name]=dict(videos=result['videos'],
                assA_TP_numerator=num.tolist(),TP_FN_FP_denominator=den.tolist())
        if target!='KITTI-V2':
            nv=len(methods['NATIVE']['videos']);draws=np.random.default_rng(0).multinomial(nv,np.ones(nv)/nv,size=2000)
            for name in (*ORACLES,*CALIBRATORS):
                samples=paired_clusters(methods[name]['stats'],methods['NATIVE']['stats'],draws)
                bootstrap[combo+'__'+name+'_minus_NATIVE']=dict(combo=combo,target=target,first=name,second='NATIVE',
                    delta_HOTA=h[name]-h['NATIVE'],CI95=np.percentile(samples,[2.5,97.5]).tolist(),clusters=nv,samples=2000,seed=0,
                    unit='video; all queries/frames kept together',pooling='official TP-weighted HOTA sufficient statistics')
            if target=='SportsMOT':
                keys=set(methods['NATIVE']['hrows'])-missing
                native=subset_stats(methods['NATIVE'],keys)
                for name in (*ORACLES,*CALIBRATORS):
                    stats=subset_stats(methods[name],keys);samples=paired_clusters(stats,native,draws)
                    bootstrap[combo+'__418__'+name+'_minus_NATIVE']=dict(combo=combo,target=target,first=name,second='NATIVE',
                        delta_HOTA=float(paired_clusters(stats,native,np.ones(nv))),CI95=np.percentile(samples,[2.5,97.5]).tolist(),
                        clusters=nv,sequences=418,samples=2000,seed=0,GT_protocol='V4_GT_RESTRICTED_TO_OFFICIAL_418_COVERAGE')
        print(json.dumps(dict(event='DECOMPOSE_B_REPORTED',combo=combo,required_granularity=required)),flush=True)
    for name,values in [('b_all.csv',rows),('granularity.csv',[r for r in rows if r['configuration'] in ORACLES]),
                        ('calibrators.csv',[r for r in rows if r['configuration'] in CALIBRATORS]),('rho_free.csv',rhofree),
                        ('granularity_verdicts.csv',granularity),('b_splits.csv',splitrows),('sports418_b.csv',sensitivity),
                        ('sports418_b_splits.csv',sensitivity_splits)]:
        if values:
            write_csv(PUBLIC/name,values)
    atomic_json(PUBLIC/'b_bootstrap.json',bootstrap)
    atomic_json(PUBLIC/'prediction_paths.json',dict(configurations=manifests))
    with gzip.open(PUBLIC/'b_bootstrap_stats.json.gz','wt') as h:json.dump(stats_export,h,separators=(',',':'))
    index=PUBLIC/'all_prediction_files.csv.gz'
    with gzip.open(index,'wt') as h:
        writer=csv.DictWriter(h,fieldnames=['combo','configuration','video','expression','prediction_path','prediction_sha256','selected_rows'])
        writer.writeheader()
        for m in manifests:
            s=json.loads(Path(m['seal_path']).read_text())
            writer.writerows(dict(combo=m['combo'],configuration=m['configuration'],**e) for e in s['episodes'])
    atomic_json(RUN/'b_report.json',dict(status='ALL_REQUESTED_B_CONFIGURATIONS_REPORTED',combos=combos,rows=len(rows),
        splits=len(splitrows),sports418_protocol='V4_GT_RESTRICTED_TO_OFFICIAL_418_COVERAGE',
        prediction_index_path=str(index),prediction_index_sha256=file_sha(index)))


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--combos',nargs='+',required=True);a=p.parse_args();report(a.combos)
