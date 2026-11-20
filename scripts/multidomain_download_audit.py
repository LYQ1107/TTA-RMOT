"""Reconcile actual authenticated downloads without opening referring GT values."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.multidomain_runtime import PUBLIC, RUN, VENDOR, require_plan
from scripts.multidomain_baidu_resume import DEST, SHARE
from flex_tta.runtime.pf_receipts import atomic_json, file_sha


def main():
    require_plan()
    path=PUBLIC/'downloads.json';record=json.loads(path.read_text())
    campaign_path=RUN/'baidu_batch_campaign.json'
    if campaign_path.exists():
        batch=json.loads(campaign_path.read_text())
        known={(x.get('source'),x.get('started_at_UTC')) for x in record['attempts']}
        for item in batch['attempts']:
            key=item.get('source'),item.get('started_at_UTC')
            if key in known or not item.get('completed_at_UTC'):continue
            record['attempts'].append(item);known.add(key)
        bypath={x['path']:x for x in record['files']}
        for item in batch['files']:
            if item['path'] in bypath:
                if bypath[item['path']]['sha256']!=item['sha256']:
                    raise ValueError('Conflicting authenticated download content')
            else:
                record['files'].append(item);bypath[item['path']]=item
    # Record the pinned official browser driver archive separately from research assets.
    tools=DEST.parent.parent/'browser-tools'
    driver=tools/'archives/geckodriver-v0.36.0-linux64.tar.gz'
    binary=tools/'geckodriver-v0.36.0/geckodriver'
    known={x['path'] for x in record['files']}
    for p,member in [(driver,None),(binary,'geckodriver')]:
        if p.is_file() and str(p) not in known:
            entry=dict(path=str(p),bytes=p.stat().st_size,sha256=file_sha(p),
                url='https://github.com/mozilla/geckodriver/releases/download/v0.36.0/geckodriver-v0.36.0-linux64.tar.gz',
                recorded_at_UTC=datetime.now(timezone.utc).isoformat(),
                file_mtime_UTC=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat(),
                role='ISOLATED_BROWSER_TOOL_NOT_RESEARCH_MODEL',license='MPL-2.0')
            if member:entry.update(archive_member=member,archive=str(driver))
            record['files'].append(entry)
    bypath={x['path']:x for x in record['files']}
    for item in bypath.values():
        p=Path(item['path'])
        if not p.is_file() or p.stat().st_size!=item['bytes'] or file_sha(p)!=item['sha256']:
            raise ValueError('Downloaded file missing/changed: '+str(p))
    package=DEST/'FlexHook';data=package/'datasets/LaMOT-main'
    latest=next(x for x in reversed(record['attempts']) if x['source']=='FlexHook_authenticated_asset_download')
    selected=latest['selected_metadata']
    missing=[x['path'] for x in selected if not (DEST/x['path'].lstrip('/')).is_file()]
    official=[x for p,x in bypath.items() if Path(p).is_relative_to(DEST)]
    counts=Counter('McByte-SportsMOT' if '/McByte-SportsMOT/' in p else
        'SportsMOT_annotations' if '/annotations_v1/val/SportsMOT/' in p else
        'SportsMOT_GT_templates' if '/gt_templates/SportsMOT/' in p else 'other'
        for p in (x['path'] for x in official))
    seqs=set((VENDOR/'seqmaps/SportsMOT.txt').read_text().splitlines())
    gtdir=data/'gt_templates/SportsMOT'
    gt={str(p.relative_to(gtdir).parent).replace('/','+',1) for p in gtdir.rglob('gt.txt')}
    snapshot=dict(verified_at_UTC=datetime.now(timezone.utc).isoformat(),
        authenticated_server_session_confirmed=True,raw_credentials_or_signed_links_logged=False,
        official_payload_files=len(official),official_payload_bytes=sum(x['bytes'] for x in official),
        per_group=dict(counts),all_file_records_SHA256_verified=True,missing_selected_payloads=missing,
        source=SHARE,GT_template_episodes=len(gt),GT_template_missing=sorted(seqs-gt),
        GT_template_extra=sorted(gt-seqs),GT_payload_contents_read=False,GT_fabricated=False,
        disk_free_bytes=shutil.disk_usage(DEST).free,training_updates=0,MLLM_calls=0,
        existing_C1_C2_results_changed=False)
    record['authenticated_download_summary']=snapshot
    record['Baidu_client_assessment']['latest_conclusion']=(
        'User-confirmed native Netdisk browser QR login succeeded on the server. '
        'Private account cookies used for real HTTP200 payloads; official batch '
        'archives provide an additional working transfer path. See current '
        'authenticated_download_summary and per-file SHA records, not old anonymous failures.')
    record['Baidu_client_assessment']['credentials_available']=True
    record['asset_gates'].update(
        McByte_45_video_coverage='45_OFFICIAL_FILES_DOWNLOADED_REQUIRES_ROW_AUDIT' if counts['McByte-SportsMOT']==45 else 'MISSING',
        annotations466='466_OFFICIAL_FILES_DOWNLOADED_REQUIRES_TEXT_MAPPING' if counts['SportsMOT_annotations']==466 else 'PARTIAL_DOWNLOAD',
        LaMOT_strict_load='PENDING' if (package/'SOTA_ckpts/LaMOT-best.pth').is_file() else 'MISSING_OFFICIAL_CHECKPOINT',
        official_GT_templates=f'{len(gt)}_OF_466_OFFICIAL_GT_FILES_{len(seqs-gt)}_MISSING',
        C3_C4_C5_C6='STOPPED_PENDING_ASSET_AND_P0_VERIFICATION')
    record['status']='AUTHENTICATED_PAYLOADS_VERIFIED_REMAINING_ASSET_GATES_NOT_YET_PASSED'
    record['updated_UTC']=datetime.now(timezone.utc).isoformat()
    atomic_json(path,record);atomic_json(PUBLIC/'authenticated_downloads.json',snapshot)
    print(json.dumps({k:v for k,v in snapshot.items() if k not in ('GT_template_missing','missing_selected_payloads')},indent=2),flush=True)


if __name__=='__main__':main()
