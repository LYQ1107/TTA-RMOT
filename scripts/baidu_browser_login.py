"""Keep an isolated, real official Netdisk browser alive for user QR login.

This uses the website's own login UI and JavaScript rather than constructing a
separate Passport login flow. Private browser profiles/cookies never enter git.
"""
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import requests
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

ASSETS = Path('/data2/user/TTA-RMOT-assets/flexhook/lamot_20261101')
DRIVER = ASSETS / 'browser-tools/geckodriver-v0.36.0/geckodriver'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def private_json(path, value):
    with path.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    path.chmod(0o600)


def emit(status, **fields):
    print(json.dumps(dict(status=status, **fields), ensure_ascii=False), flush=True)


def main():
    os.umask(0o077)
    parent = ASSETS / 'auth_sessions'
    parent.mkdir(mode=0o700, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='baidu-browser-', dir=parent))
    state_path = directory / 'status.json'
    state = dict(status='STARTING_REAL_BROWSER', created_at_Asia_Shanghai=now(),
        site='https://pan.baidu.com/', private_directory=str(directory),
        credential_values_printed=False, own_isolated_profile=True)
    private_json(state_path, state)
    browser = None
    try:
        options = Options()
        options.binary_location = '/usr/bin/firefox'
        options.add_argument('-headless')
        options.page_load_strategy = 'eager'
        options.set_preference('intl.accept_languages', 'zh-CN,zh,en-US,en')
        options.set_preference('browser.download.folderList', 2)
        options.set_preference('browser.download.dir', str(directory / 'downloads'))
        service = Service(executable_path=str(DRIVER), service_args=['--log', 'error'],
            log_output=str(directory / 'geckodriver.log'))
        browser = webdriver.Firefox(service=service, options=options)
        browser.set_window_size(1440, 1000)
        browser.set_page_load_timeout(45)
        emit('OFFICIAL_BROWSER_STARTED', status_path=str(state_path))
        browser.get('https://pan.baidu.com/')
        deadline = time.monotonic() + 90
        qr = None
        clicked = False
        seen_sources = set()
        while time.monotonic() < deadline:
            if urlparse(browser.current_url).hostname != 'pan.baidu.com':
                raise RuntimeError('Browser left the official Netdisk origin before login')
            for element in browser.find_elements(By.CSS_SELECTOR,
                    'img.tang-pass-qrcode-img, img[src*="passport.baidu.com/v2/api/qrcode"]'):
                parsed_source = urlparse(element.get_attribute('src') or '')
                seen_sources.add((parsed_source.scheme, parsed_source.hostname, parsed_source.path))
                # The login component initially displays its static loading GIF.
                # Wait for the real official image and completed image load.
                if (element.is_displayed() and element.size['width'] >= 100
                        and parsed_source.hostname == 'passport.baidu.com'
                        and parsed_source.path == '/v2/api/qrcode'
                        and browser.execute_script('return arguments[0].complete && arguments[0].naturalWidth >= 100', element)):
                    qr = element
                    break
            if qr:
                break
            if not clicked:
                for button in browser.find_elements(By.CSS_SELECTOR, 'a,button'):
                    if button.is_displayed() and button.text.strip() in ('登录', '立即登录', '去登录'):
                        button.click()
                        clicked = True
                        break
            time.sleep(1)
        if qr is None:
            browser.save_screenshot(str(directory / 'official-page-debug.png'))
            visible = browser.find_element(By.TAG_NAME, 'body').text
            private_json(directory / 'page_debug.json', dict(title=browser.title,
                origin=urlparse(browser.current_url).hostname, visible_text=visible[:4000],
                seen_QR_sources=sorted(seen_sources, key=str),
                image_sources=[e.get_attribute('src') for e in browser.find_elements(By.TAG_NAME, 'img')][:30]))
            raise RuntimeError('Official page did not expose a visible login QR; private diagnostic saved')
        visible = browser.find_element(By.TAG_NAME, 'body').text
        if '百度网盘' not in visible or '扫码' not in visible:
            raise RuntimeError('Official UI does not identify Netdisk QR login')
        source = qr.get_attribute('src')
        source_url = urlparse(source)
        private_json(directory / 'qr_context.json', dict(source=source,
            official_page=browser.current_url, title=browser.title, visible_text=visible[:4000]))
        if source_url.scheme != 'https' or source_url.hostname != 'passport.baidu.com':
            raise RuntimeError('Visible QR image is not served by official Passport HTTPS')
        # CSS scaling in the page can make a screenshot undecodable. Download
        # the exact visible official image URL, without modifying its QR data.
        response = requests.get(source, headers={'Referer': 'https://pan.baidu.com/'},
            timeout=25, allow_redirects=False)
        response.raise_for_status()
        if not response.headers.get('Content-Type', '').startswith('image/'):
            raise RuntimeError('Visible official image URL did not return an image')
        png = response.content
        pixels = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        padded = cv2.copyMakeBorder(pixels, 20, 20, 20, 20,
            cv2.BORDER_CONSTANT, value=(255, 255, 255))
        payload, _, _ = cv2.QRCodeDetector().detectAndDecode(padded)
        decoded = urlparse(payload)
        query = parse_qs(decoded.query, keep_blank_values=True)
        source_query = parse_qs(source_url.query)
        if (decoded.scheme != 'https' or decoded.hostname not in {'wappass.baidu.com', 'passport.baidu.com'}
                or query.get('tpl') != ['netdisk'] or query.get('cmd') != ['login']
                or not source_query.get('sign') or query.get('sign') != source_query.get('sign')):
            raise RuntimeError('Visible QR failed Netdisk product and live-session identity checks')
        path = directory / 'official-netdisk-login.png'
        with path.open('xb') as handle:
            handle.write(png)
        browser.save_screenshot(str(directory / 'official-login-page.png'))
        state.update(status='QR_READY_BROWSER_WAITING', qr_image=str(path),
            QR_generated_at_Asia_Shanghai=now(), title=browser.title,
            actual_page_origin='pan.baidu.com', QR_image_origin=source_url.hostname,
            QR_decoded_origin=decoded.hostname, QR_product='netdisk', live_session_match=True,
            native_site_login_UI=True, original_official_image=True, waiting_seconds=300)
        private_json(state_path, state)
        emit('QR_READY_BROWSER_WAITING', image_path=str(path), status_path=str(state_path),
            generated_at_Asia_Shanghai=state['QR_generated_at_Asia_Shanghai'],
            official_page='https://pan.baidu.com/', product='netdisk', live_session_match=True,
            waiting_seconds=300)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            cookies = [c for c in browser.get_cookies() if c['domain'].lstrip('.') in {
                'baidu.com', 'pan.baidu.com', 'passport.baidu.com'}]
            names = {c['name'] for c in cookies if c.get('value')}
            if 'BDUSS' in names and '/disk/' in urlparse(browser.current_url).path:
                credential_path = directory / 'session.cookies.json'
                private_json(credential_path, cookies)
                state.update(status='BROWSER_LOGIN_CONFIRMED_SESSION_SAVED',
                    completed_at_Asia_Shanghai=now(), credential_path=str(credential_path),
                    authenticated_page_origin=urlparse(browser.current_url).hostname,
                    BDUSS_present=True, STOKEN_present='STOKEN' in names)
                private_json(state_path, state)
                emit(state['status'], status_path=str(state_path), credential_values_printed=False)
                return
            try:
                current = qr.get_attribute('src')
                if current != source:
                    raise RuntimeError('Official page replaced the QR; do not use the previously shown image')
            except RuntimeError:
                raise
            except Exception:
                pass # Login navigation can detach the old QR while cookies settle.
            time.sleep(2)
        state.update(status='QR_WAIT_TIMED_OUT_NO_LOGIN', session_saved=False)
        private_json(state_path, state)
        emit(state['status'], status_path=str(state_path))
    except Exception as exc:
        state.update(status='BROWSER_LOGIN_NOT_READY_OR_FAILED', error_type=type(exc).__name__)
        if isinstance(exc, RuntimeError):
            state['reason'] = str(exc)
        private_json(state_path, state)
        emit(state['status'], reason=state.get('reason'), error_type=type(exc).__name__, status_path=str(state_path))
        sys.exit(1)
    finally:
        if browser is not None:
            browser.quit()


if __name__ == '__main__':
    main()
