"""Download official public-share small files; no account, transfer, or code execution."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import unquote, urlparse, parse_qs

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.multidomain_runtime import PUBLIC, require_plan
from flex_tta.runtime.pf_receipts import atomic_json, file_sha

BASE = 'https://pan.baidu.com'
SURL = 'L-43y9SFDKmgl3dJNRlvNA'
SHARE = BASE + '/s/1' + SURL
UK, SHARE_ID = 1714442791, 48200137171
DEST = Path('/data2/user/TTA-RMOT-assets/flexhook/lamot_20261101/flexhook_baidu/public_files')


def public_session():
    session = requests.Session()
    session.get(BASE + '/share/init', params={'surl': SURL}, timeout=30).raise_for_status()
    result = session.post(BASE + '/share/verify', params={'surl': SURL},
        data={'pwd': 'd3qj'}, headers={'Referer': BASE + '/share/init?surl=' + SURL}, timeout=30).json()
    if result.get('errno') != 0:
        raise RuntimeError('Public extraction verification failed: ' + str(result.get('errno')))
    data = session.get(BASE + '/share/tplconfig', params={
        'shareid': SHARE_ID, 'uk': UK, 'fields': 'sign,timestamp'}, timeout=30).json()
    if data.get('errno') != 0:
        raise RuntimeError('Public signature unavailable: ' + str(data.get('errno')))
    return session, data['data']


def listing(session, directory):
    files = []
    for page in range(1, 100):
        data = session.get(BASE + '/share/list', params=dict(uk=UK, shareid=SHARE_ID,
            dir=directory, page=page, num=100, order='name', desc=0, web=1, app_id=250528), timeout=30).json()
        if data.get('errno') != 0:
            raise RuntimeError(f'Public listing {directory}: errno {data.get("errno")}')
        rows = data.get('list', [])
        files.extend(rows)
        if len(rows) < 100:
            return files
    raise RuntimeError('Unexpected public directory pagination')


def download_one(file, cookies, signature):
    begin = time.perf_counter()
    result = dict(share_file_path=file['path'], expected_bytes=file['size'], clienttype=0)
    path = DEST / file['path'].lstrip('/')
    if path.exists():
        result.update(status='EXISTING_FILE_PRESERVED', bytes=path.stat().st_size, sha256=file_sha(path))
        return result, None
    session = requests.Session()
    session.cookies.update(cookies)
    try:
        response = session.post(BASE + '/api/sharedownload', params={
            'sign': signature['sign'], 'timestamp': signature['timestamp'], 'clienttype': 0,
            'web': 1, 'app_id': 250528, 'channel': 'chunlei'}, data={
            'encrypt': 0, 'product': 'share', 'uk': UK, 'primaryid': SHARE_ID,
            'fid_list': json.dumps([file['fs_id']]), 'type': 'nolimit',
            'extra': json.dumps({'sekey': unquote(session.cookies.get('BDCLND', ''))})},
            headers={'Referer': SHARE}, timeout=30)
        response.raise_for_status()
        data = response.json()
        links = data.get('list')
        result.update(errno=data.get('errno'), link_container_type=type(links).__name__)
        if isinstance(links, str):
            result['link_container_chars'] = len(links)
            # Never mistake an opaque/encrypted value for a URL or a login error.
            if links.lstrip().startswith(('[', '{')):
                links = json.loads(links)
        if not isinstance(links, list) or not links or not links[0].get('dlink'):
            result['status'] = 'NO_USABLE_PUBLIC_DLINK'
            return result, None
        link = links[0]['dlink']
        host = urlparse(link).hostname or ''
        if not (host.endswith('.baidu.com') or host.endswith('.baidupcs.com')):
            raise ValueError('Download link is not an official Baidu host')
        if shutil.disk_usage(DEST).free < int(file['size']) + (2 << 30):
            raise RuntimeError('Insufficient space; retain at least 2 GiB')
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + '.partial')
        md5 = hashlib.md5()
        with session.get(link, headers={'Referer': SHARE}, stream=True, timeout=(30, 90)) as response:
            result['HTTP_status'] = response.status_code
            if not response.ok:
                try:
                    err = response.json()
                    result.update(CDN_error_code=err.get('error_code'), CDN_error_msg=err.get('error_msg'))
                except ValueError:
                    pass
            response.raise_for_status()
            with partial.open('xb') as handle:
                for chunk in response.iter_content(1 << 20):
                    handle.write(chunk)
                    md5.update(chunk)
        if partial.stat().st_size != file['size']:
            raise ValueError('Download byte count differs from official directory entry')
        # Share-list "md5" values may be obfuscated (including non-hex letters).
        # Preserve them as opaque metadata, not a purported plaintext file digest.
        partial.rename(path)
        entry = dict(url=SHARE, share_file_path=file['path'], path=str(path),
            bytes=path.stat().st_size, sha256=file_sha(path), md5=md5.hexdigest(),
            downloaded_at_UTC=datetime.now(timezone.utc).isoformat(),
            share_metadata_hash=file.get('md5'), share_metadata_hash_not_plaintext_md5=True,
            seconds=time.perf_counter()-begin, HTTP_status=result['HTTP_status'])
        result.update(status='DOWNLOADED_SIZE_VERIFIED_SHA256_RECORDED', seconds=entry['seconds'])
        return result, entry
    except Exception as exc:
        # Keep any partial bytes for explicit recovery; do not overwrite previous files.
        result.update(status='DOWNLOAD_FAILED', partial_bytes_preserved=path.with_name(path.name + '.partial').exists(), error_type=type(exc).__name__,
            error=str(exc).split(' for url:')[0][:300], seconds=time.perf_counter()-begin)
        return result, None


def diagnose_transfers(session, signature):
    """Bounded real transfer checks; stop at failure, never retry account authentication."""
    selected = [listing(session, '/FlexHook/tracker_outputs/McByte-SportsMOT')[0],
        next(x for x in listing(session, '/FlexHook/SOTA_ckpts') if x['server_filename'] == 'LaMOT-best.pth')]
    outcomes = []
    for file, mode in [(selected[0], 'nolimit'), (selected[0], None), (selected[1], 'nolimit'), (selected[1], 'batch')]:
        item = dict(share_file_path=file['path'], type_parameter=mode, checked_at_UTC=datetime.now(timezone.utc).isoformat())
        form = dict(encrypt=0, product='share', uk=UK, primaryid=SHARE_ID,
            fid_list=json.dumps([file['fs_id']]), extra=json.dumps({'sekey': unquote(session.cookies.get('BDCLND', ''))}))
        if mode:
            form['type'] = mode
        data = session.post(BASE + '/api/sharedownload', params=dict(sign=signature['sign'],
            timestamp=signature['timestamp'], clienttype=0, web=1, app_id=250528,
            channel='chunlei', _=time.time_ns()), data=form,
            headers={'Referer': SHARE, 'Cache-Control': 'no-cache'}, timeout=30).json()
        links = data.get('list')
        item.update(errno=data.get('errno'), server_time=data.get('server_time'),
            link_container_type=type(links).__name__)
        link = data.get('dlink') or (links[0].get('dlink') if isinstance(links, list) and links else None)
        if link:
            params = parse_qs(urlparse(link).query)
            item['link_timing'] = {k: v for k, v in params.items() if 'time' in k or k == 'expires'}
            with session.get(link, headers={'Range': 'bytes=0-255', 'Referer': SHARE,
                'User-Agent': 'pan.baidu.com', 'Cache-Control': 'no-cache'}, stream=True, timeout=(30, 40)) as response:
                item.update(HTTP_status=response.status_code, content_range=response.headers.get('Content-Range'))
                if response.status_code not in (200, 206):
                    try:
                        err = response.json()
                        item.update(CDN_error_code=err.get('error_code'), CDN_error_msg=err.get('error_msg'))
                    except ValueError:
                        item['non_JSON_failure'] = True
                else:
                    item['probe_bytes_received'] = len(next(response.iter_content(256), b''))
        elif isinstance(links, str):
            item['opaque_response_characters'] = len(links)
        outcomes.append(item)
        print(json.dumps(dict(event='BOUNDED_TRANSFER_DIAGNOSIS', **item)), flush=True)
    return dict(source='FlexHook_bounded_fresh_transfer_diagnosis', outcomes=outcomes,
        account_credentials_used=False, account_credentials_requested=False, signed_URLs_modified=False,
        credentials_or_expiring_links_recorded=False, timestamp=time.time(), gpu_used=False)


def summarize(record):
    inventories = [x for x in record['attempts'] if x.get('inventory_only') and x.get('selected_metadata')]
    inventory = inventories[-1]
    entries = [x for x in inventory['selected_metadata'] if '/annotations_v1/val/SportsMOT/' in x['path']]
    raw = Path(record['destination']) / 'lamot_github/extracted/LaMOT-main/annotations_v1/val/SportsMOT'
    matches = [x for x in entries if (raw / Path(x['path']).name).is_file()
        and (raw / Path(x['path']).name).stat().st_size == x['size']]
    files = [p for p in DEST.rglob('*') if p.is_file()]
    campaign = next(x for x in record['attempts'] if x['source']=='FlexHook_public_small_files_continuation' and not x.get('inventory_only'))
    outcomes = campaign['outcomes']
    record['continuation_summary'] = dict(
        completed_at_UTC=datetime.now(timezone.utc).isoformat(),
        disk_free_bytes_after=shutil.disk_usage(DEST).free,
        full_pagination_annotation_files=len(entries),
        official_annotation_names_and_sizes_available_in_prior_GitHub_download=len(matches),
        byte_equivalence='UNVERIFIED_SHARE_HASH_IS_OBFUSCATED_NOT_A_PLAIN_MD5',
        new_public_payload_files=len(files), new_public_payload_bytes=sum(p.stat().st_size for p in files),
        first_campaign=dict(selected_files=campaign['selected_files'],
            HTTP403_failures=sum(x.get('HTTP_status')==403 for x in outcomes),
            opaque_responses=sum(x['status']=='NO_USABLE_PUBLIC_DLINK' for x in outcomes),
            downloaded=campaign['downloaded_files'], partial_files_created=0,
            note='First campaign status said PARTIAL_PRESERVED, but HTTP failed before any local file opened; zero actual partial files.'),
        account_credentials_used=False, account_credentials_requested=False,
        GPU_hours=0, new_inference_runs=0, existing_C1_C2_results_changed=False,
        bounded_diagnosis=[x for x in record['attempts'] if x['source']=='FlexHook_bounded_fresh_transfer_diagnosis'][-1])
    record['Baidu_client_assessment']['latest_conclusion'] = (
        'Public verification/listing/signature work. Small-file links return HTTP403/31360 '
        'expire time out error even immediately after cache-bypassing fresh API requests; '
        'batch checkpoint link returns HTTP403/31064. Large single-file response is opaque. '
        'Not mislabeled as a login error; no supported anonymous transfer found, no account '
        'credentials available or sought. Public listings are not downloaded payloads.')
    record['LaMOT_annotation_coverage'].update(
        FlexHook_public_listing_files=len(entries),
        FlexHook_listed_names_and_sizes_present_locally=len(matches),
        FlexHook_json_equivalence='UNVERIFIED_OBFUSCATED_SHARE_HASH_AND_CDN403',
        runtime_text_projection='NOT_DONE_NO_SPORTS_MODEL_INFERENCE', referring_payloads_read=0)
    record['asset_gates'].update(
        McByte_45_video_coverage='45_OFFICIAL_FILES_LISTED_ZERO_DOWNLOADED_HTTP403',
        annotations466='466_OFFICIAL_NAMES_AND_SIZES_MATCH_PRIOR_GITHUB_FILES_BYTE_EQUIVALENCE_UNVERIFIED',
        LaMOT_strict_load='NOT_RUN_NO_DOWNLOADED_OFFICIAL_CHECKPOINT',
        C3_C4_C5_C6='STOPPED_WAITING_TRANSFERABLE_OFFICIAL_ASSETS')
    record['status'] = 'DOWNLOAD_CONTINUATION_COMPLETED_NO_NEW_ASSETS_CDN_HTTP403'
    record['updated_UTC'] = datetime.now(timezone.utc).isoformat()
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory-only', action='store_true', help='Refresh complete paginated listing without download requests')
    parser.add_argument('--diagnose-transfers', action='store_true', help='Only four bounded real HTTP transfer checks')
    parser.add_argument('--summarize', action='store_true', help='Offline provenance/status summary; no network or referring payloads')
    args = parser.parse_args()
    require_plan()
    record_path = PUBLIC / 'downloads.json'
    record = json.loads(record_path.read_text())
    DEST.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        record = summarize(record)
        atomic_json(record_path, record)
        print(json.dumps(record['continuation_summary']), flush=True)
        return
    session, signature = public_session()
    if args.diagnose_transfers:
        attempt = diagnose_transfers(session, signature)
        record['attempts'].append(attempt)
        atomic_json(record_path, record)
        return
    dirs = ['/FlexHook/tracker_outputs/McByte-SportsMOT',
        '/FlexHook/datasets/LaMOT-main/annotations_v1/val/SportsMOT',
        '/FlexHook/datasets/LaMOT-main/sequences/val',
        '/FlexHook/datasets/LaMOT-main/gt_templates']
    files, listings, errors = [], [], []
    for directory in dirs:
        try:
            rows = listing(session, directory)
            listings.append(dict(directory=directory, entries=[dict(path=x['path'],
                size=x.get('size'), md5=x.get('md5'), isdir=x.get('isdir')) for x in rows]))
            for row in rows:
                name = row['server_filename']
                if row.get('isdir'):
                    # Only SportsMOT ground-truth templates, no other benchmark payloads.
                    if '/gt_templates' in directory and (name == 'val' or name == 'SportsMOT'):
                        dirs.append(row['path'])
                    continue
                if directory.endswith('/sequences/val') and name not in {
                    'SportsMOT_labels.json', 'SportsMOT_RESOLUTION.json', 'TAO_frame2file.json'}:
                    continue
                if '/gt_templates' not in directory or '/SportsMOT' in directory:
                    files.append(row)
        except Exception as exc:
            errors.append(dict(directory=directory, error=str(exc)))
    weights = listing(session, '/FlexHook/SOTA_ckpts')
    files.append(next(x for x in weights if x['server_filename'] == 'LaMOT-best.pth'))
    attempt = dict(source='FlexHook_public_small_files_continuation', started_at_UTC=datetime.now(timezone.utc).isoformat(),
        account_credentials_used=False, account_credentials_requested=False, code_from_downloads_executed=False,
        gpu_used=False, disk_free_bytes_before=shutil.disk_usage(DEST).free,
        selected_files=len(files), selected_metadata=[dict(path=f['path'], size=f['size'], md5=f.get('md5')) for f in files],
        listings=listings, listing_errors=errors, outcomes=[])
    record['attempts'].append(attempt)
    attempt['inventory_only'] = args.inventory_only
    atomic_json(record_path, record)
    if args.inventory_only:
        attempt.update(status='COMPLETE_PAGINATED_PUBLIC_LISTING_NO_PAYLOAD_DOWNLOAD', completed_at_UTC=datetime.now(timezone.utc).isoformat())
        atomic_json(record_path, record)
        print(json.dumps(dict(event='PUBLIC_INVENTORY_DONE', selected=len(files), errors=errors)), flush=True)
        return
    print(json.dumps(dict(event='PUBLIC_DOWNLOAD_START', selected=len(files), errors=errors)), flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download_one, f, session.cookies, signature) for f in files]
        for future in as_completed(futures):
            result, entry = future.result()
            attempt['outcomes'].append(result)
            if entry:
                record['files'].append(entry)
            atomic_json(record_path, record)
            print(json.dumps(dict(event='PUBLIC_DOWNLOAD_FILE', path=result['share_file_path'],
                status=result['status'], error=result.get('CDN_error_msg', result.get('error')),
                done=len(attempt['outcomes']), total=len(files))), flush=True)
    attempt['completed_at_UTC'] = datetime.now(timezone.utc).isoformat()
    attempt['downloaded_files'] = sum(x['status'].startswith('DOWNLOADED_') for x in attempt['outcomes'])
    record['status'] = 'PUBLIC_DOWNLOAD_CONTINUED_SEE_PER_FILE_OUTCOMES'
    atomic_json(record_path, record)
    print(json.dumps(dict(event='PUBLIC_DOWNLOAD_DONE', downloaded=attempt['downloaded_files'], total=len(files))), flush=True)


if __name__ == '__main__':
    main()
