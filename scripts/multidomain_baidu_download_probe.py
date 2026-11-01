"""Check the public share's own signed-download endpoint, without account login."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import unquote
import requests
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.multidomain_runtime import PUBLIC,require_plan
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def main():
    require_plan();p=PUBLIC/'downloads.json';record=json.loads(p.read_text());dest=Path(record['destination'])/'flexhook_baidu'
    session=requests.Session();base='https://pan.baidu.com';surl='L-43y9SFDKmgl3dJNRlvNA'
    session.get(base+'/share/init',params={'surl':surl},timeout=30)
    verified=session.post(base+'/share/verify',params={'surl':surl},data={'pwd':'d3qj'},headers={'Referer':base+'/share/init?surl='+surl},timeout=30).json()
    result=dict(source='FlexHook_public_signed_download_probe',account_credentials_used=False,extraction_errno=verified.get('errno'))
    if verified.get('errno')!=0:raise RuntimeError('Public extraction session unavailable')
    uk=1714442791;shareid=48200137171
    tpl=session.get(base+'/share/tplconfig',params={'shareid':shareid,'uk':uk,'fields':'sign,timestamp'},timeout=30).json()
    result['public_signature_errno']=tpl.get('errno');data=tpl.get('data',{})
    if tpl.get('errno')==0 and data.get('sign') and data.get('timestamp'):
        listed=session.get(base+'/share/list',params=dict(uk=uk,shareid=shareid,dir='/FlexHook/SOTA_ckpts',page=1,num=1000,web=1),timeout=30).json()
        file=next(x for x in listed.get('list',[]) if x['server_filename']=='LaMOT-best.pth')
        r=session.post(base+'/api/sharedownload',params={'sign':data['sign'],'timestamp':data['timestamp'],'clienttype':0,'web':1,'app_id':250528,'encrypt':0,'channel':'chunlei'},
            data={'encrypt':0,'product':'share','uk':uk,'primaryid':shareid,'fid_list':json.dumps([file['fs_id']]),'type':'nolimit',
                  'extra':json.dumps({'sekey':unquote(session.cookies.get('BDCLND',''))})},headers={'Referer':base+'/s/1'+surl},timeout=30).json()
        result['signed_download_errno']=r.get('errno');links=r.get('list',[])
        result['link_container_type']=type(links).__name__
        if isinstance(links,str):
            result['link_container_chars']=len(links)
            if not links.lstrip().startswith(('[','{')):
                result['status']='STOP_OPAQUE_LINK_RESPONSE_NO_USABLE_ANONYMOUS_DLINK'
                result['response_keys']=sorted(r)
                result['requires_supported_account_client_not_installed']=True
                record['attempts'].append(result);atomic_json(p,record);print(json.dumps(result),flush=True);return
            links=json.loads(links)
        if isinstance(links,dict):links=links.get('list',[links])
        atomic_json(dest/'last_download_probe_status.json',result)
        if r.get('errno')==0 and links and links[0].get('dlink'):
            path=dest/'SOTA_ckpts/LaMOT-best.pth'
            if path.exists():raise FileExistsError('Preserve existing checkpoint')
            if shutil.disk_usage(dest).free<int(file['size'])+(4<<30):raise RuntimeError('Insufficient disk for official checkpoint')
            path.parent.mkdir(exist_ok=True);partial=path.with_suffix('.pth.partial')
            begin=time.perf_counter()
            with session.get(links[0]['dlink'],stream=True,timeout=(30,90)) as response:
                response.raise_for_status()
                with partial.open('xb') as h:
                    for chunk in response.iter_content(1<<20):h.write(chunk)
            if partial.stat().st_size!=file['size']:raise ValueError('Incomplete checkpoint kept as partial')
            partial.rename(path);result['status']='OFFICIAL_CHECKPOINT_DOWNLOADED_ANONYMOUSLY'
            record['files'].append(dict(url='https://pan.baidu.com/s/1'+surl,share_file_path=file['path'],path=str(path),bytes=path.stat().st_size,
                sha256=file_sha(path),downloaded_at_UTC=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-begin))
        else:result['status']='STOP_SIGNED_DOWNLOAD_REQUIRES_UNAVAILABLE_LOGIN_SESSION'
    else:result['status']='STOP_PUBLIC_SIGNATURE_REQUIRES_UNAVAILABLE_SESSION'
    record['attempts'].append(result);atomic_json(p,record);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
