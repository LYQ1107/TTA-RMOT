"""User-authorized Baidu Netdisk login, using only Baidu's official HTTPS endpoints.

QR images and credentials stay outside git in a private per-session directory.
Never print authentication tokens, cookies, or credential-bearing response bodies.
This process does not share the Codex app's browser session or read its profile.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from urllib.parse import parse_qs, urlparse

import cv2
import requests

ASSETS = Path('/data2/user/TTA-RMOT-assets/flexhook/lamot_20261101')
PASSPORT = 'https://passport.baidu.com'


def emit(status, **fields):
    print(json.dumps(dict(status=status, **fields), ensure_ascii=False), flush=True)


def save_private(path, value):
    with path.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    path.chmod(0o600)


def parse_jsonp(text):
    text = text.strip().rstrip(';')
    if text.startswith('(') and text.endswith(')'):
        text = text[1:-1]
    return json.loads(text)


def main():
    os.umask(0o077)
    parent = ASSETS / 'auth_sessions'
    parent.mkdir(mode=0o700, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='baidu-qr-', dir=parent))
    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36'
    state_path = directory / 'status.json'
    state = dict(status='INITIALIZING', provider='Baidu Netdisk',
        created_at_UTC=datetime.now(timezone.utc).isoformat(), credentials_exported_to_git=False)
    save_private(state_path, state)
    try:
        session.get('https://pan.baidu.com/', timeout=25).raise_for_status()
        response = session.get(PASSPORT + '/v2/api/getqrcode', params={
            'lp': 'pc', 'qrloginfrom': 'pc', 'apiver': 'v3', 'tpl': 'netdisk',
            'tt': int(time.time()*1000)}, headers={'Referer': 'https://pan.baidu.com/'}, timeout=25)
        response.raise_for_status()
        info = response.json()
        if info.get('errno') != 0 or not info.get('sign'):
            raise RuntimeError('Official QR request rejected')
        prompt = info.get('prompt') or ''
        if prompt and '百度网盘' not in prompt:
            raise RuntimeError('Official QR consent names an application other than Baidu Netdisk')
        url = info['imgurl']
        if not url.startswith('https://'):
            url = 'https://' + url.removeprefix('//')
        if urlparse(url).hostname != 'passport.baidu.com':
            raise RuntimeError('QR image is not from official Baidu Passport')
        image = session.get(url, timeout=25)
        image.raise_for_status()
        if urlparse(image.url).hostname != 'passport.baidu.com':
            raise RuntimeError('Unexpected QR image redirect')
        path = directory / 'baidu-netdisk-login.png'
        with path.open('xb') as handle:
            handle.write(image.content)
        pixels = cv2.imread(str(path))
        if pixels is None:
            raise RuntimeError('Official response is not an image')
        # Official image has no white quiet zone. Pad the decoder's temporary
        # input only; display/store the original unmodified official PNG.
        decoder_input = cv2.copyMakeBorder(pixels, 20, 20, 20, 20,
            cv2.BORDER_CONSTANT, value=(255, 255, 255))
        payload, _, _ = cv2.QRCodeDetector().detectAndDecode(decoder_input)
        decoded = urlparse(payload)
        query = parse_qs(decoded.query)
        if decoded.scheme != 'https' or decoded.hostname not in {'wappass.baidu.com', 'passport.baidu.com'}:
            raise RuntimeError('Decoded QR is not an official HTTPS Baidu login destination')
        if ('qrlogin' not in decoded.query or query.get('sign') != [info['sign']]
                or query.get('tpl') != ['netdisk']):
            raise RuntimeError('QR does not match this exact login session')
        state.update(status='WAITING_FOR_SCAN', qr_image=str(path),
            decoded_origin=decoded.hostname, decoded_path=decoded.path,
            QR_session_matches=True, original_official_image=True,
            prompt=info.get('prompt'), qr_generated_at_UTC=datetime.now(timezone.utc).isoformat())
        save_private(state_path, state)
        emit('QR_READY', image_path=str(path), status_path=str(state_path),
            official_image_origin='passport.baidu.com', decoded_origin=decoded.hostname,
            QR_session_matches=True, prompt=info.get('prompt'), waiting_seconds=300)
        deadline = time.monotonic() + 300
        last = state['status']
        while time.monotonic() < deadline:
            try:
                check = session.get(PASSPORT + '/channel/unicast', params={
                    'channel_id': info['sign'], 'callback': '', 'tpl': 'netdisk',
                    'apiver': 'v3', 'tt': int(time.time()*1000)}, timeout=(10, 25))
                check.raise_for_status()
                data = parse_jsonp(check.text)
                channel = data.get('channel_v')
                if data.get('errno') != 0 or not channel:
                    time.sleep(2)
                    continue
                channel = json.loads(channel) if isinstance(channel, str) else channel
                if channel.get('status') == 0 and channel.get('v'):
                    login = session.get(PASSPORT + '/v3/login/main/qrbdusslogin', params={
                        'bduss': channel['v'], 'tpl': 'netdisk',
                        'u': 'https://pan.baidu.com/disk/main'}, allow_redirects=False, timeout=25)
                    if login.status_code not in (200, 302, 303):
                        raise RuntimeError('Official login exchange rejected')
                    if not any(c.name == 'BDUSS' and c.value for c in session.cookies):
                        raise RuntimeError('Login exchange did not establish an authenticated session')
                    session.get('https://pan.baidu.com/disk/main', timeout=25).raise_for_status()
                    credential_path = directory / 'session.cookies.json'
                    save_private(credential_path, [dict(name=c.name, value=c.value, domain=c.domain,
                        path=c.path, secure=c.secure, expires=c.expires) for c in session.cookies])
                    state.update(status='LOGIN_CONFIRMED_SESSION_SAVED',
                        completed_at_UTC=datetime.now(timezone.utc).isoformat(),
                        credential_path=str(credential_path), session_reusable=True)
                    save_private(state_path, state)
                    emit('LOGIN_CONFIRMED_SESSION_SAVED', status_path=str(state_path),
                        credential_values_printed=False)
                    return
                state['status'] = 'SCAN_RECEIVED_WAITING_FOR_PHONE_CONFIRMATION'
                save_private(state_path, state)
                if state['status'] != last:
                    emit(state['status'])
                    last = state['status']
            except (requests.RequestException, ValueError):
                # Long-poll timeout is normal; the finite session deadline is unchanged.
                pass
            time.sleep(2)
        state.update(status='QR_WAIT_TIMED_OUT_NO_LOGIN', session_reusable=False)
        save_private(state_path, state)
        emit(state['status'], status_path=str(state_path))
    except Exception as exc:
        state.update(status='LOGIN_SETUP_OR_EXCHANGE_FAILED', error_type=type(exc).__name__)
        if isinstance(exc, RuntimeError):
            state['reason'] = str(exc) # Only fixed, local messages; never HTTP response contents.
        save_private(state_path, state)
        emit(state['status'], error_type=type(exc).__name__, reason=state.get('reason'), status_path=str(state_path))
        sys.exit(1)


if __name__ == '__main__':
    main()
