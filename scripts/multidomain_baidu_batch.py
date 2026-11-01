"""Authenticated official batch downloads, including evaluation-only GT templates.

The original browser session stays private. Archives are extracted into new
directories; research scripts from downloaded archives are never executed.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from urllib.parse import unquote, urlparse
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.multidomain_baidu_resume import (BASE, SHARE, UK, SHARE_ID, DEST,
    public_session, listing)
from scripts.multidomain_runtime import PUBLIC, require_plan
from flex_tta.runtime.pf_receipts import atomic_json, file_sha

ASSETS = DEST.parent


def official_link(session, signature, row):
    # A large archive can outlive the share-page signature used for the previous
    # item. Refresh per batch, keeping the authenticated browser cookies.
    fresh = session.get(BASE + '/share/tplconfig', params=dict(shareid=SHARE_ID,
        uk=UK, fields='sign,timestamp', _=time.time_ns()), timeout=30).json()
    if fresh.get('errno') != 0:
        raise RuntimeError('Official batch signature refresh failed: ' + str(fresh.get('errno')))
    signature = fresh['data']
    data = session.post(BASE + '/api/sharedownload', params=dict(
        sign=signature['sign'], timestamp=signature['timestamp'], clienttype=0,
        web=1, app_id=250528, channel='chunlei', _=time.time_ns()),
        data=dict(encrypt=0, product='share', uk=UK, primaryid=SHARE_ID,
            fid_list=json.dumps([row['fs_id']]), type='batch',
            extra=json.dumps({'sekey': unquote(session.cookies.get('BDCLND', ''))})),
        headers={'Referer': SHARE}, timeout=30).json()
    link = data.get('dlink')
    if data.get('errno') != 0 or not link:
        raise RuntimeError('Official batch API returned no link; errno=' + str(data.get('errno')))
    host = urlparse(link).hostname or ''
    if not (host.endswith('.baidu.com') or host.endswith('.baidupcs.com')):
        raise ValueError('Non-official archive host')
    return link


def receive_archive(session, link, path):
    if path.exists():
        return dict(status='EXISTING_ARCHIVE_PRESERVED', bytes=path.stat().st_size,
            sha256=file_sha(path), path=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {'Referer': SHARE}
    if offset:
        headers['Range'] = f'bytes={offset}-'
    with session.get(link, headers=headers, stream=True, timeout=(30, 90)) as response:
        if not response.ok:
            try:
                error = response.json()
                raise RuntimeError(f'Official archive CDN HTTP {response.status_code}; code={error.get("error_code")}')
            except ValueError:
                raise RuntimeError(f'Official archive CDN HTTP {response.status_code}') from None
        if offset and (response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
            raise ValueError('Archive resume offset was not acknowledged; partial preserved')
        received = 0
        with partial.open('ab' if offset else 'xb') as handle:
            for chunk in response.iter_content(1 << 20):
                if shutil.disk_usage(path.parent).free < (2 << 30) + len(chunk):
                    raise RuntimeError('Retain at least 2 GiB free; partial preserved')
                handle.write(chunk)
                received += len(chunk)
        expected = response.headers.get('Content-Length')
        if expected and received != int(expected):
            raise ValueError('Archive transfer Content-Length mismatch')
    if not zipfile.is_zipfile(partial):
        raise ValueError('Official batch response is not a ZIP archive; payload preserved')
    partial.rename(path)
    return dict(status='OFFICIAL_ZIP_DOWNLOADED', bytes=path.stat().st_size,
        sha256=file_sha(path), path=str(path), HTTP_status=response.status_code,
        resumed_bytes=offset, downloaded_at_UTC=datetime.now(timezone.utc).isoformat(), url=SHARE)


def unpack(archive, row):
    directory = Path(tempfile.mkdtemp(prefix=Path(row['path']).name + '-',
        dir=ASSETS / 'batch_extracted'))
    published = []
    with zipfile.ZipFile(archive) as handle:
        members = [x for x in handle.infolist() if not x.is_dir()]
        expected_name = Path(row['path']).name
        for member in members:
            # Official Baidu batch ZIPs can encode the complete share path as
            # "<public-prefix>%3A%2FFlexHook%2F..." rather than normal slashes.
            name = Path(unquote(member.filename))
            if name.is_absolute() or '..' in name.parts:
                raise ValueError('Archive path outside the independent extraction directory')
            if row.get('isdir'):
                parts = name.parts
                if expected_name in parts:
                    relative = Path(*parts[parts.index(expected_name)+1:])
                else:
                    relative = name
                if '/gt_templates/' in row['path']:
                    valid = relative.name in {'gt.txt', 'seqinfo.ini'}
                elif '/annotations_v1/val/SportsMOT' in row['path']:
                    valid = len(relative.parts)==1 and relative.suffix=='.json'
                else:
                    valid = False
                if not valid:
                    raise ValueError('Unexpected official dataset archive member: ' + str(name))
                final = DEST / row['path'].lstrip('/') / relative
            else:
                if name.name != expected_name or len(members) != 1 or member.file_size != row['size']:
                    raise ValueError('Single-file archive differs from the official directory entry')
                relative = Path(name.name)
                final = DEST / row['path'].lstrip('/')
            local = directory / relative
            local.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(member) as source, local.open('xb') as target:
                shutil.copyfileobj(source, target, length=1 << 20)
            sha = file_sha(local)
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.exists():
                if file_sha(final) != sha:
                    raise ValueError('Preserve conflicting existing official asset: ' + str(final))
            else:
                local.rename(final)
            published.append(dict(path=str(final), bytes=final.stat().st_size, sha256=sha,
                url=SHARE, share_file_path='/' + str(final.relative_to(DEST)),
                archive_member=member.filename, archive=str(archive),
                downloaded_at_UTC=datetime.now(timezone.utc).isoformat(),
                evaluated_only='/gt_templates/' in str(final), downloaded_code_executed=False))
    return published


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--cookies-file', type=Path, required=True)
    parser.add_argument('--large-files', action='store_true')
    parser.add_argument('--gt-templates', action='store_true')
    parser.add_argument('--annotations', action='store_true')
    parser.add_argument('--record-path', type=Path, default=PUBLIC / 'downloads.json',
        help='Separate campaign record when the small-file downloader is active; merge after both finish')
    args = parser.parse_args()
    require_plan()
    record_path = args.record_path
    record = json.loads(record_path.read_text()) if record_path.exists() else dict(attempts=[], files=[])
    session, signature = public_session(args.cookies_file)
    selected = []
    if args.large_files:
        for directory, name in [('/FlexHook/SOTA_ckpts', 'LaMOT-best.pth'),
                ('/FlexHook/datasets/LaMOT-main/sequences/val', 'SportsMOT_labels.json'),
                ('/FlexHook/datasets/LaMOT-main/sequences/val', 'SportsMOT_RESOLUTION.json'),
                ('/FlexHook/datasets/LaMOT-main/sequences/val', 'TAO_frame2file.json')]:
            row = next(x for x in listing(session, directory) if x['server_filename'] == name)
            if not (DEST / row['path'].lstrip('/')).exists():
                selected.append(row)
    if args.annotations:
        selected.append(next(x for x in listing(session, '/FlexHook/datasets/LaMOT-main/annotations_v1/val')
            if x['server_filename'] == 'SportsMOT'))
    if args.gt_templates:
        row = next(x for x in listing(session, '/FlexHook/datasets/LaMOT-main/gt_templates')
            if x['server_filename'] == 'SportsMOT')
        selected.append(row)
    (ASSETS / 'batch_extracted').mkdir(exist_ok=True)
    attempt = dict(source='FlexHook_authenticated_official_batch',
        started_at_UTC=datetime.now(timezone.utc).isoformat(), account_credentials_used=True,
        raw_credentials_or_download_links_logged=False, downloaded_code_executed=False,
        GPU_used=False, selected_paths=[x['path'] for x in selected], outcomes=[])
    record['attempts'].append(attempt)
    atomic_json(record_path, record)
    for row in selected:
        start = time.perf_counter()
        outcome = dict(share_file_path=row['path'])
        try:
            if shutil.disk_usage(DEST).free < 2*int(row.get('size', 0)) + (2 << 30):
                raise RuntimeError('Insufficient space for archive plus extraction and 2 GiB reserve')
            link = official_link(session, signature, row)
            suffix = '-annotations' if '/annotations_v1/' in row['path'] else ''
            archive = ASSETS / 'batch_archives' / (Path(row['path']).name + suffix + '.zip')
            info = receive_archive(session, link, archive)
            if not any(x.get('path') == info['path'] for x in record['files']):
                record['files'].append(info)
            entries = unpack(archive, row)
            existing = {x.get('path') for x in record['files']}
            record['files'].extend(x for x in entries if x['path'] not in existing)
            outcome.update(status='ARCHIVE_DOWNLOADED_EXTRACTED_VERIFIED', files=len(entries),
                bytes=sum(x['bytes'] for x in entries), archive_sha256=info['sha256'])
        except Exception as error:
            outcome.update(status='OFFICIAL_BATCH_FAILED', error_type=type(error).__name__,
                error=str(error).split(' for url:')[0][:300])
        outcome['seconds'] = time.perf_counter() - start
        attempt['outcomes'].append(outcome)
        atomic_json(record_path, record)
        print(json.dumps(dict(event='OFFICIAL_BATCH_COMPLETE', **outcome)), flush=True)
    attempt['completed_at_UTC'] = datetime.now(timezone.utc).isoformat()
    attempt['status'] = 'PASS' if all(x['status'] == 'ARCHIVE_DOWNLOADED_EXTRACTED_VERIFIED'
        for x in attempt['outcomes']) else 'PARTIAL_PRESERVED_SEE_FAILURES'
    record['status'] = 'AUTHENTICATED_DOWNLOADS_SEE_ASSET_GATES'
    atomic_json(record_path, record)
    if attempt['status'] != 'PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
