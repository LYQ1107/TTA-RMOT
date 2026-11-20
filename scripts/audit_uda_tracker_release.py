#!/usr/bin/env python3
"""Inspect official public tracker release metadata, never target annotations.

Uses only the publisher-provided share password through its normal verify API.
Cookies and signed download URLs are not written to the report.
"""
import json
from pathlib import Path
import re
import sys
import time
import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.data.manifests import official_splits


def inspect():
    base='https://pan.baidu.com';short='L-43y9SFDKmgl3dJNRlvNA'
    link=base+'/s/1'+short+'?pwd=d3qj'
    session=requests.Session()
    session.headers.update({'User-Agent':'Mozilla/5.0','Referer':base+'/share/init?surl='+short,'Origin':base})
    initial=session.get(link,timeout=25);initial.raise_for_status()
    verified=session.post(base+'/share/verify',params=dict(surl=short,t=int(time.time()*1000),channel='chunlei',web=1,app_id=250528,clienttype=0),
        data=dict(pwd='d3qj',vcode='',vcode_str=''),timeout=25).json()
    report=dict(official_share=link,verify_errno=verified.get('errno'),target_gt_read=False,
        cookies_or_signed_urls_saved=False,listings=[])
    if verified.get('errno')!=0:return report
    page=session.get(link,timeout=25);page.raise_for_status()
    ids={k.lower():int(v) for k,v in re.findall(r'''["'](shareid|share_id|uk|share_uk)["']\s*:\s*["']?(\d+)''',page.text,re.I)}
    report['public_share_ids']=ids
    shareid=ids.get('shareid',ids.get('share_id'));uk=ids.get('share_uk') or ids.get('uk')
    if shareid is None or uk is None:
        report['status']='PUBLIC_PAGE_IDS_NOT_FOUND';return report
    def listing(path=None):
        params=dict(uk=uk,shareid=shareid,num=100,page=1,order='name',desc=0,showempty=0,web=1,app_id=250528,clienttype=0)
        params.update(dict(root=1) if path is None else dict(dir=path))
        payload=session.get(base+'/share/list',params=params,timeout=25).json()
        result=dict(path=path or '<public root>',errno=payload.get('errno'),items=[{k:r[k] for k in ('server_filename','path','isdir','size') if k in r} for r in payload.get('list',[])])
        report['listings'].append(result);return result
    listing()
    bank=listing('/FlexHook/tracker_outputs/DETR-NeuralSORT-Dance')
    listing('/FlexHook/tracker_outputs')
    listing('/FlexHook/SOTA_ckpts')
    train=set(official_splits()['dance']['train']);val=set(official_splits()['dance']['val'])
    names={Path(r['server_filename']).stem for r in bank['items'] if not r.get('isdir')}
    report.update(status='AUDITED_PUBLIC_RELEASE' if bank['errno']==0 else 'PUBLIC_LISTING_UNAVAILABLE',
        released_train_video_ids=sorted(names&train),released_val_video_ids=sorted(names&val),
        missing_train_video_ids=sorted(train-names))
    return report


if __name__=='__main__':
    result=inspect();atomic_json(ROOT/'results/uda/assets/official_tracker_release.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
