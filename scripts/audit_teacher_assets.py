#!/usr/bin/env python3
"""Audit public frozen teacher assets; never read target labels or save cookies."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.parse
import zipfile
import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha

ASSETS=Path('/data2/user/TTA-RMOT-assets/teacher-p1')
QWEN_REV='89644892e4d85e24eaac8bacfd4f463576704203'
QWEN_WEIGHT_SHA='7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0'


def audit_public_lora(download=False):
    short='nwhcEIQshWk9TnNjhiYMrQ';base='https://pan.baidu.com';link=base+'/s/1'+short
    session=requests.Session()
    session.headers.update({'User-Agent':'Mozilla/5.0','Referer':base+'/share/init?surl='+short,'Origin':base})
    session.get(link,timeout=30).raise_for_status()
    verified=session.post(base+'/share/verify',params=dict(surl=short,t=int(time.time()*1000),channel='chunlei',
        web=1,app_id=250528,clienttype=0),data=dict(pwd='z5kt',vcode='',vcode_str=''),timeout=30).json()
    result=dict(official_share=link,verify_errno=verified.get('errno'),listings=[],
        target_GT_payload_read=False,cookies_signed_urls_or_tokens_saved=False)
    if verified.get('errno')!=0:
        return dict(result,status='PUBLIC_VERIFY_FAILED',message=verified.get('err_msg'))
    page=session.get(link,timeout=30);page.raise_for_status()
    ids={k.lower():int(v) for k,v in re.findall(r'''["'](shareid|share_id|uk|share_uk)["']\s*:\s*["']?(\d+)''',page.text,re.I)}
    shareid=ids.get('shareid',ids.get('share_id'));uk=ids.get('share_uk') or ids.get('uk')
    if shareid is None or uk is None:return dict(result,status='PUBLIC_PAGE_IDS_NOT_FOUND')
    def listing(path=None):
        params=dict(uk=uk,shareid=shareid,num=100,page=1,order='name',desc=0,showempty=0,web=1,app_id=250528,clienttype=0)
        params.update(dict(root=1) if path is None else dict(dir=path))
        response=session.get(base+'/share/list',params=params,timeout=30);response.raise_for_status();payload=response.json()
        items=payload.get('list',[])
        result['listings'].append(dict(path=path or '<root>',errno=payload.get('errno'),items=[
            {k:r[k] for k in ('server_filename','path','isdir','size','md5') if k in r} for r in items]))
        return items
    raw_archives=[]
    root=listing();folders=[r for r in root if r.get('isdir') and 'v2bestweight' in r.get('server_filename','').lower()]
    if not folders:
        for r in root:
            if r.get('isdir') and r.get('server_filename','').lower() in ('yestrack','yestrack-models','models'):
                children=listing(r['path'])
                folders.extend(x for x in children if x.get('isdir') and 'v2bestweight' in x.get('server_filename','').lower())
                raw_archives.extend(x for x in children if x.get('server_filename','').lower()=='yestrack.zip')
    queue=[r['path'] for r in folders];adapter_files=[]
    while queue:
        path=queue.pop(0)
        for row in listing(path):
            if row.get('isdir'):queue.append(row['path'])
            elif row.get('server_filename') in ('adapter_config.json','adapter_model.safetensors','README.md'):
                adapter_files.append({k:row[k] for k in ('server_filename','path','size','md5','fs_id') if k in row})
    archives=[r for item in result['listings'] for r in item['items'] if r.get('server_filename','').lower()=='yestrack.zip']
    result['official_v2_adapter_files']=adapter_files
    result['released_archive']=archives
    result['status']='OFFICIAL_V2_LISTED_DOWNLOAD_REQUIRED' if adapter_files else ('OFFICIAL_RELEASE_ZIP_REQUIRES_SELECTIVE_V2_EXTRACTION' if archives else 'OFFICIAL_V2_NOT_LISTED')
    if download and raw_archives:
        item=raw_archives[0];link_to_file=item.get('dlink')
        if not link_to_file:
            sign=re.search(r'''["']sign["']\s*:\s*["']([^"']+)["']''',page.text)
            timestamp=re.search(r'''["']timestamp["']\s*:\s*["']?(\d+)''',page.text)
            if sign and timestamp:
                response=session.post(base+'/api/sharedownload',params=dict(sign=sign.group(1),timestamp=timestamp.group(1),
                    channel='chunlei',web=1,app_id=250528,clienttype=0),data=dict(encrypt=0,product='share',
                    uk=uk,primaryid=shareid,fid_list=json.dumps([item['fs_id']]),
                    extra=json.dumps(dict(sekey=urllib.parse.unquote(verified.get('randsk',''))))),timeout=30).json()
                result['normal_public_download_errno']=response.get('errno')
                links=response.get('list',[]);link_to_file=links[0].get('dlink') if links else None
            else:result['normal_public_download_status']='NO_ANONYMOUS_DOWNLOAD_SIGNATURE'
        if link_to_file:
            dest=ASSETS/'downloads/yestrack.zip';dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():
                response=session.get(link_to_file,stream=True,timeout=90);response.raise_for_status()
                with dest.with_suffix('.zip.part').open('wb') as handle:
                    for chunk in response.iter_content(1<<20):handle.write(chunk)
                dest.with_suffix('.zip.part').replace(dest)
            if dest.stat().st_size!=int(item['size']) or not zipfile.is_zipfile(dest):
                result['normal_public_download_status']='RESPONSE_IS_NOT_PUBLISHED_ARCHIVE';return result
            extracted=[]
            with zipfile.ZipFile(dest) as archive:
                for member in archive.infolist():
                    parts=Path(member.filename).parts
                    if 'v2bestweight' not in [x.lower() for x in parts] or Path(member.filename).name not in ('adapter_config.json','adapter_model.safetensors','README.md'):continue
                    target=ASSETS/'models/v2bestweight'/Path(member.filename).parent.name/Path(member.filename).name
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with archive.open(member) as source,target.open('wb') as handle:
                        import shutil
                        shutil.copyfileobj(source,handle)
                    extracted.append(dict(path=str(target),sha256=file_sha(target),bytes=target.stat().st_size))
            result.update(status='OFFICIAL_V2_SELECTIVELY_EXTRACTED' if extracted else 'NO_V2_ADAPTER_IN_OFFICIAL_ZIP',
                extracted_v2_files=extracted,downloaded_archive_sha256=file_sha(dest),other_archive_payloads_opened=False)
        else:result['status']='PUBLIC_ZIP_LISTED_BUT_ANONYMOUS_BINARY_DOWNLOAD_UNAVAILABLE'
    return result


def inspect():
    model=ASSETS/'models/Qwen3-VL-2B-Instruct'
    result=dict(protocol='DEVELOPMENT_TEACHER_PROBE',base_model='Qwen/Qwen3-VL-2B-Instruct',
        pinned_revision=QWEN_REV,base_license='Apache-2.0',conditions={'T0':'frozen base, no adapter',
        'T1':'same base, verified official Refer-KITTI-V2 LoRA only'},target_GT_used=False,
        pretraining_overlap='UNKNOWN: complete Qwen pretraining corpus is not publicly auditable; no non-overlap claim',
        model_files={},references={})
    for p in sorted(model.glob('*')):
        if p.is_file() and p.suffix in ('.json','.safetensors','.txt'):
            result['model_files'][p.name]=dict(bytes=p.stat().st_size,sha256=file_sha(p))
    result['T0_ready']=all(n in result['model_files'] for n in ('model.safetensors','config.json','tokenizer.json')) and result['model_files']['model.safetensors']['sha256']==QWEN_WEIGHT_SHA
    result['official_weight_SHA256']=QWEN_WEIGHT_SHA
    for name,path in [('YesTrack',ASSETS/'references/YesTrack'),('DINO_Teacher',ASSETS/'references/DINO_Teacher'),
        ('DKGTrack',ASSETS/'references/DKGTrack'),('WSRVOS',Path('/data2/user/TTA-RMOT-assets/uda-references/WSRVOS')),
        ('SoftMatch',ASSETS/'references/SoftMatch')]:
        if (path/'.git').exists():
            commit=subprocess.check_output(['git','-C',str(path),'log','-1','--format=%H %cI'],text=True).strip()
            result['references'][name]=dict(path=str(path),commit_and_date=commit,
                license_files=[p.name for p in path.glob('LICENSE*')])
    local=list((ASSETS/'models/v2bestweight').rglob('adapter_config.json'))
    result['local_v2_adapters']=[]
    for p in local:
        config=json.loads(p.read_text());weight=p.with_name('adapter_model.safetensors')
        result['local_v2_adapters'].append(dict(path=str(p.parent),config=config,config_sha256=file_sha(p),
            weight_sha256=file_sha(weight) if weight.exists() else None))
    result['T1_ready']=any(x['weight_sha256'] for x in result['local_v2_adapters'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--public-lora',action='store_true')
    p.add_argument('--download-lora',action='store_true');a=p.parse_args()
    result=inspect()
    if a.public_lora or a.download_lora:
        try:result['public_lora']=audit_public_lora(a.download_lora)
        except requests.RequestException as exc:result['public_lora']=dict(status='PUBLIC_NETWORK_ERROR',error=type(exc).__name__)
    dest=ROOT/'results/teacher_probe/assets/teacher_asset_audit.json'
    if dest.exists():
        previous=json.loads(dest.read_text())
        atomic_json(dest.parent/'history'/f'teacher_asset_audit_{time.time_ns()}.json',previous)
        if 'public_lora' not in result and 'public_lora' in previous:result['public_lora']=previous['public_lora']
    atomic_json(dest,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
