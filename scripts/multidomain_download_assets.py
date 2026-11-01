"""Download public provenance/LaMOT annotations, never execute downloaded code.

The Baidu branch uses only anonymous share access plus the published extraction
code. A login requirement stops that branch, not the authorized KITTI work.
"""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha

PUBLIC = ROOT / 'results/multidomain_generality'
RUN = ROOT / 'results/runs/multidomain_generality_20261101'
DEST = Path('/data2/user/TTA-RMOT-assets/flexhook/lamot_20261101')
SHARE = 'https://pan.baidu.com/s/1L-43y9SFDKmgl3dJNRlvNA'


def main():
    publication = json.loads((RUN/'prereg_amendment_publication.json').read_text())
    assert publication['remote_verified'] and file_sha(PUBLIC/'PREREG.md') == publication['prereg_sha256']
    free = shutil.disk_usage(DEST.parent).free
    if free < 4 << 30:
        raise RuntimeError(f'Insufficient download headroom: {free} bytes; stop without deleting user data')
    DEST.mkdir(exist_ok=True)
    out = PUBLIC/'downloads.json'
    record = json.loads(out.read_text()) if out.exists() else dict(
        status='IN_PROGRESS', destination=str(DEST), disk_free_bytes_before=free,
        amendment_publication=publication, files=[], attempts=[],
        licenses=dict(FlexHook_code='MIT (official LICENSE); separate checkpoint/dataset grant not stated',
                      LaMOT_annotations='No standalone license found in official repository; public research annotations, not assumed MIT',
                      SportsMOT_images='Existing images only; original dataset terms continue to apply'))

    def save():
        atomic_json(out, record)

    def get(url, relative):
        path = DEST/relative
        if path.exists():
            prior = next((r for r in record['files'] if r['path'] == str(path)), None)
            if not prior or file_sha(path) != prior['sha256']:
                raise FileExistsError(f'Preserve pre-existing/unverified download: {path}')
            return path
        started = time.perf_counter()
        r = requests.get(url, timeout=(15, 90))
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as f:
            f.write(r.content)
        item = dict(url=url, final_url=r.url, path=str(path), bytes=path.stat().st_size,
                    sha256=file_sha(path), downloaded_at_UTC=datetime.now(timezone.utc).isoformat(),
                    seconds=time.perf_counter()-started, HTTP_status=r.status_code)
        record['files'].append(item)
        save()
        print(json.dumps(dict(event='ASSET_DOWNLOADED', **item)), flush=True)
        return path

    # Inspect the public share without reading or discovering any account credentials.
    try:
        p = get(SHARE+'?pwd=d3qj', 'flexhook_baidu/share.html')
        text = p.read_text(errors='replace')
        session = requests.Session()
        session.get(SHARE+'?pwd=d3qj', timeout=(15,30))
        verify = session.post('https://pan.baidu.com/share/verify',
            params={'surl':'L-43y9SFDKmgl3dJNRlvNA','t':int(time.time()*1000)},
            data={'pwd':'d3qj','vcode':'','vcode_str':''},
            headers={'Referer':'https://pan.baidu.com/share/init?surl=L-43y9SFDKmgl3dJNRlvNA'}, timeout=(15,30))
        vp = DEST/'flexhook_baidu/share_verify_correct_surl_response.json'
        if not vp.exists():
            with vp.open('xb') as f: f.write(verify.content)
            record['files'].append(dict(url=str(verify.url), path=str(vp), bytes=vp.stat().st_size,
                sha256=file_sha(vp), downloaded_at_UTC=datetime.now(timezone.utc).isoformat(),
                HTTP_status=verify.status_code, public_extraction_code_only=True))
        data = verify.json()
        record['attempts'].append(dict(source='FlexHook_Baidu', public_page_saved=str(p),
            verify_HTTP_status=verify.status_code, verify_errno=data.get('errno'),
            status='STOP_BAIDU_PATH_LOGIN_OR_PUBLIC_API_REQUIRED',
            detail='Anonymous published-code request did not supply downloadable asset links; no login credentials used or requested',
            page_mentions_login=any(k in text for k in ('登录','login','passport'))))
    except Exception as e:
        record['attempts'].append(dict(source='FlexHook_Baidu', status='STOP_BAIDU_PUBLIC_PATH_UNAVAILABLE',
                                       error=f'{type(e).__name__}: {e}'))
    save()
    if '--baidu-only' in sys.argv:
        print(json.dumps(record['attempts'][-1],ensure_ascii=False),flush=True)
        return
    # Download only official repository data. Never import/run this repository.
    try:
        api = get('https://api.github.com/repos/Nathan-Li123/LaMOT/git/trees/main?recursive=1', 'lamot_github/tree.json')
        tree = json.loads(api.read_text())
        paths = {x['path'] for x in tree.get('tree', [])}
        record['licenses']['LaMOT_license_files'] = sorted(p for p in paths if 'license' in Path(p).name.lower())
        archive = get('https://codeload.github.com/Nathan-Li123/LaMOT/zip/refs/heads/main', 'lamot_github/LaMOT-main.zip')
        extract = DEST/'lamot_github/extracted'
        if extract.exists():
            raise FileExistsError('Preserve existing extraction directory; inspect receipt before resuming')
        extract.mkdir()
        with zipfile.ZipFile(archive) as z:
            members = [m for m in z.infolist() if not m.is_dir() and (
                '/annotations_v1/val/SportsMOT/' in m.filename or
                Path(m.filename).name.lower() in ('readme.md','license','license.txt','license.md'))]
            for m in members:
                target = extract/m.filename
                if not target.resolve().is_relative_to(extract.resolve()):
                    raise ValueError('Archive member outside extraction root')
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as f: f.write(z.read(m))
                record['files'].append(dict(url='https://github.com/Nathan-Li123/LaMOT/blob/'+tree['sha']+'/'+str(Path(m.filename).relative_to('LaMOT-main')),
                    archive_path=str(archive), archive_member=m.filename, path=str(target), bytes=target.stat().st_size,
                    sha256=file_sha(target), downloaded_at_UTC=datetime.now(timezone.utc).isoformat()))
        observed = {Path(m.filename).stem for m in members if '/annotations_v1/val/SportsMOT/' in m.filename}
        seq = (ROOT/'vendor/FlexHook/seqmaps/SportsMOT.txt').read_text().splitlines()
        expected = {s.replace('+','_') for s in seq}
        record['LaMOT_annotation_coverage'] = dict(expected=len(expected), observed=len(observed),
            missing=sorted(expected-observed), extra=sorted(observed-expected), referring_payloads_read=0,
            comparison_to_FlexHook='NOT_POSSIBLE_WITHOUT_FLEXHOOK_DATA_PACKAGE')
        record['attempts'].append(dict(source='LaMOT_GitHub', status='DOWNLOADED_OFFICIAL_ANNOTATIONS', revision=tree['sha']))
    except Exception as e:
        record['attempts'].append(dict(source='LaMOT_GitHub', status='FAILED_PRESERVED', error=f'{type(e).__name__}: {e}'))
    record['manual_files_if_Baidu_unavailable'] = [
        'FlexHook_best/SOTA_ckpts/LaMOT-best.pth',
        'FlexHook_best/tracker_outputs/McByte-SportsMOT/ (all 45 SportsMOT val videos)',
        'FlexHook_best/datasets/LaMOT-main/annotations_v1/val/SportsMOT/ (466 JSON files)',
        'FlexHook_best/datasets/LaMOT-main/sequences/val/SportsMOT_labels.json',
        'FlexHook_best/datasets/LaMOT-main/sequences/val/SportsMOT_RESOLUTION.json',
        'FlexHook_best/datasets/LaMOT-main/sequences/val/TAO_frame2file.json (mix-loader auxiliary metadata only)']
    record.update(status='PUBLIC_DOWNLOAD_ATTEMPTS_COMPLETE_KITTI_WORK_MAY_CONTINUE', downloaded_code_executed=False,
                  MLLM_calls=0, foreign_files_overwritten=0)
    save()
    print(json.dumps(dict(status=record['status'],files=len(record['files']),attempts=record['attempts'],
        LaMOT_file_name_coverage={k:(len(v) if isinstance(v,list) else v) for k,v in record.get('LaMOT_annotation_coverage',{}).items()}),ensure_ascii=False),flush=True)


if __name__ == '__main__': main()
