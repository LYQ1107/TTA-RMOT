"""Regression checks for actual public-share pagination and rejected transfers."""
import hashlib

from scripts import multidomain_baidu_resume as downloader


def test_public_directory_paginates_at_server_limit_100():
    class Session:
        def get(self, url, params, timeout):
            assert params['num'] == 100
            class Response:
                def json(self):
                    return {'errno': 0, 'list': [{'path': str(i)} for i in range(100 if params['page'] < 3 else 8)]}
            return Response()
    assert len(downloader.listing(Session(), '/official/SportsMOT')) == 208


def test_opaque_response_is_not_a_download_or_login_error(tmp_path, monkeypatch):
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {'errno': 0, 'list': 'opaque-client-data'}
    class Session:
        cookies = {}
        def post(self, *args, **kwargs):
            return Response()
        def get(self, *args, **kwargs):
            raise AssertionError('Opaque data must never be fetched as a URL')
    monkeypatch.setattr(downloader, 'DEST', tmp_path)
    monkeypatch.setattr(downloader.requests, 'Session', Session)
    result, entry = downloader.download_one({'path': '/FlexHook/weight.pth', 'size': 10, 'fs_id': 1}, {}, {'sign': 'test', 'timestamp': 0})
    assert result['status'] == 'NO_USABLE_PUBLIC_DLINK' and entry is None
    assert list(tmp_path.iterdir()) == []


def test_success_requires_real_bytes_and_records_sha256(tmp_path, monkeypatch):
    payload = b'1,1,1,2,3,4,1,-1,-1,-1\n'
    class Response:
        status_code = 200
        ok = True
        def raise_for_status(self):
            pass
        def json(self):
            return {'errno': 0, 'list': [{'dlink': 'https://d.pcs.baidu.com/file/test'}]}
        def iter_content(self, size):
            yield payload
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    class Session:
        cookies = {}
        def post(self, *args, **kwargs):
            return Response()
        def get(self, *args, **kwargs):
            return Response()
    monkeypatch.setattr(downloader, 'DEST', tmp_path)
    monkeypatch.setattr(downloader.requests, 'Session', Session)
    result, entry = downloader.download_one({'path': '/FlexHook/test.txt', 'size': len(payload),
        'md5': 'opaque-share-hash-not-plain-md5', 'fs_id': 1}, {}, {'sign': 'test', 'timestamp': 0})
    assert result['status'] == 'DOWNLOADED_SIZE_VERIFIED_SHA256_RECORDED'
    assert entry['sha256'] == hashlib.sha256(payload).hexdigest()
    assert entry['share_metadata_hash_not_plaintext_md5']
    assert (tmp_path / 'FlexHook/test.txt').read_bytes() == payload
