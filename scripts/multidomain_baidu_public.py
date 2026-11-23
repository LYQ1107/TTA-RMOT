"""Anonymous official share inspection; never search for or use account credentials."""
import json
import re
import sys
from pathlib import Path
from datetime import datetime,timezone
import requests
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.multidomain_runtime import PUBLIC,require_plan
from flex_tta.runtime.pf_receipts import file_sha,atomic_json


def main():
    require_plan();dest=Path('/data2/user/TTA-RMOT-assets/flexhook/lamot_20261101/flexhook_baidu')
    log=json.loads((PUBLIC/'downloads.json').read_text());session=requests.Session()
    base='https://pan.baidu.com';surl='L-43y9SFDKmgl3dJNRlvNA'
    session.get(base+'/share/init',params={'surl':surl},timeout=30)
    response=session.post(base+'/share/verify',params={'surl':surl},data={'pwd':'d3qj','vcode':'','vcode_str':''},
        headers={'Referer':base+'/share/init?surl='+surl},timeout=30)
    response.raise_for_status();verification=response.json()
    result=dict(source='FlexHook_Baidu_anonymous_followup',extraction_errno=verification.get('errno'),
                account_credentials_used=False,account_credentials_requested=False)
    if verification.get('errno')!=0:
        result.update(status='STOP_ANONYMOUS_SHARE_VERIFICATION_FAILED')
    else:
        page=session.get(base+'/s/1'+surl,timeout=30);page.raise_for_status();text=page.text
        p=dest/'verified_share_page.html'
        if not p.exists():
            with p.open('xb') as f:f.write(page.content)
            log['files'].append(dict(url=page.url,path=str(p),bytes=p.stat().st_size,sha256=file_sha(p),
                downloaded_at_UTC=datetime.now(timezone.utc).isoformat(),HTTP_status=page.status_code))
        # Inspect JSON literals only. Never execute script tags from the share.
        markers={}
        for key in ('shareid','share_id','share_uk','uk','timestamp','loginstate','isLogin','errno'):
            patterns=[r'["\']'+key+r'["\']\s*:\s*["\']?(\d+)',r'\b'+key+r'\s*:\s*(\d+)']
            value=next((m.group(1) for pat in patterns if (m:=re.search(pat,text))),None)
            if value is not None:markers[key]=int(value)
        result.update(final_url=page.url,page_sha256=file_sha(p),public_numeric_markers=markers,
            page_mentions_signin=bool(re.search(r'登录后|请登录|登录下载|login required',text,re.I)))
        shareid=markers.get('shareid',markers.get('share_id'));uk=markers.get('share_uk',markers.get('uk'))
        if shareid and uk:
            raw=re.search(r'"file_list"\s*:',text)
            roots=json.JSONDecoder().raw_decode(text[raw.end():].lstrip())[0] if raw else []
            listings=[];files=[]
            queue=[r['path'] for r in roots if r.get('isdir')]
            for directory in queue:
                r=session.get(base+'/share/list',params=dict(uk=uk,shareid=shareid,dir=directory,page=1,num=1000,order='name',desc=0,web=1,app_id=250528),timeout=30)
                data=r.json();entries=data.get('list',[])
                listings.append(dict(directory=directory,errno=data.get('errno'),files=[dict(name=x.get('server_filename'),path=x.get('path'),size=x.get('size'),isdir=x.get('isdir')) for x in entries]))
                for x in entries:
                    path=x['path'];name=x.get('server_filename','')
                    if x.get('isdir') and (name in ('SOTA_ckpts','tracker_outputs','datasets','LaMOT-main','McByte-SportsMOT') or '/McByte-SportsMOT/' in path):queue.append(path)
                    if not x.get('isdir') and (name=='LaMOT-best.pth' or '/McByte-SportsMOT/' in path):files.append(x)
            result.update(public_directory_listings=listings,official_selected_files=len(files))
            if files:
                first=files[0]
                r=session.post(base+'/api/sharedownload',data={'encrypt':0,'product':'share','uk':uk,'primaryid':shareid,'fid_list':json.dumps([first['fs_id']]),'type':'nolimit'},
                    headers={'Referer':base+'/s/1'+surl},timeout=30)
                data=r.json();result['anonymous_download_link_errno']=data.get('errno')
                result['download_links_returned']=bool(data.get('list'))
                result['status']='ANONYMOUS_DOWNLOAD_LINK_AVAILABLE' if data.get('errno')==0 and data.get('list') else 'STOP_ANONYMOUS_DOWNLOAD_REQUIRES_LOGIN_OR_UNAVAILABLE_SIGNED_LINK'
            else:
                result['status']='STOP_PUBLIC_LIST_REQUIRES_UNAVAILABLE_SESSION_OR_NO_OFFICIAL_ASSETS'
        else:
            result['status']='STOP_NO_ANONYMOUS_DOWNLOADABLE_SHARE_METADATA'
    log['attempts'].append(result);atomic_json(PUBLIC/'downloads.json',log)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
