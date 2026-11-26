"""Finalize public-download status and legal image-only layout; no GT payload reads."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.multidomain_runtime import PUBLIC,VENDOR,require_plan
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def main():
    require_plan();p=PUBLIC/'downloads.json';record=json.loads(p.read_text());dest=Path(record['destination'])
    # These files were obtained from this archive, not from a reconstructed blob URL.
    for r in record['files']:
        if 'archive_member' in r:r['url']='https://codeload.github.com/Nathan-Li123/LaMOT/zip/refs/heads/main'
    listings=[r for a in record['attempts'] for r in a.get('public_directory_listings',[])]
    public_files={f['path']:f for r in listings for f in r['files']}
    wanted=[f for f in public_files if f.endswith('/LaMOT-best.pth') or '/McByte-SportsMOT/' in f]
    record['manual_files_exactly_listed_in_official_share']=[public_files[f] for f in sorted(wanted)]
    record['manual_files_if_Baidu_unavailable']=[
        '/FlexHook/SOTA_ckpts/LaMOT-best.pth',
        '/FlexHook/tracker_outputs/McByte-SportsMOT/ (45 files enumerated in manual_files_exactly_listed_in_official_share)',
        '/FlexHook/datasets/LaMOT-main/annotations_v1/val/SportsMOT/',
        '/FlexHook/datasets/LaMOT-main/sequences/val/SportsMOT_labels.json',
        '/FlexHook/datasets/LaMOT-main/sequences/val/SportsMOT_RESOLUTION.json',
        '/FlexHook/datasets/LaMOT-main/sequences/val/TAO_frame2file.json']
    record['Baidu_client_assessment']=dict(url='https://github.com/qjfoidnh/BaiduPCS-Go#初级使用教程',
        official_usage_requires_account_login=True,client_installed=False,credentials_available=False,
        no_account_credentials_searched_or_requested=True,
        conclusion='Stop Baidu path: public listing works, anonymous download API returned parameter/signature errno2 with no dlink; account-login client not attempted without credentials',
        errno2_not_mislabeled_as_login_error=True)
    record['Baidu_client_assessment']['signed_download_followups']=[r for r in record['attempts'] if r['source']=='FlexHook_public_signed_download_probe']
    record['Baidu_client_assessment']['latest_conclusion']='Public template signature works; signed API errno0 returns an opaque string list, not a dlink, even with encrypt=0 query. Supported client docs require login; no credentials available, no JS/proxy executed, manual official-file download required.'
    archive=dest/'lamot_github/LaMOT-main.zip'
    downloaded=[r for r in record['files'] if '/annotations_v1/val/SportsMOT/' in r.get('archive_member','')]
    expected={s.split('+')[0] for s in (VENDOR/'seqmaps/SportsMOT.txt').read_text().splitlines()}
    observed={Path(r['path']).stem.rsplit('_',1)[0] for r in downloaded}
    record['LaMOT_annotation_coverage'].update(raw_json_files=len(downloaded),raw_videos=len(observed),
        official_seqmap_videos=len(expected),missing_videos=sorted(expected-observed),
        direct_filename_mapping='NOT_EXPECTED_RAW_NUMERIC_QUERY_IDS_VS_FLEXHOOK_TEXT_NAMES',
        runtime_text_projection='NOT_DONE_SPORTS_ASSET_GATE_NOT_PASSED',
        referring_payloads_read=0,FlexHook_json_equivalence='UNVERIFIED_WITHOUT_FLEXHOOK_PACKAGE')
    raw=Path('/data2/user/LocateMOT_FullDataV2_spill/datasets/SportsMOT/val')
    layout=dest/'flexhook_baidu/layout/LaMOT-main';link=layout/'sequences/val/SportsMOT-sequences'
    link.parent.mkdir(parents=True,exist_ok=True)
    if not link.exists():link.symlink_to(raw,target_is_directory=True)
    if link.resolve()!=raw:raise ValueError('Preserve conflicting layout symlink')
    vendorlink=VENDOR/'datasets/LaMOT-main'
    if not vendorlink.exists() and not vendorlink.is_symlink():vendorlink.symlink_to(layout,target_is_directory=True)
    if vendorlink.resolve()!=layout.resolve():raise ValueError('Preserve existing vendor data layout')
    record['image_links']=[dict(path=str(link),target=str(raw),raw_images_redownloaded=False),
        dict(path=str(vendorlink),target=str(layout),metadata_still_missing=True,vendor_code_changed=False)]
    record['asset_gates']=dict(LaMOT_strict_load='NOT_RUN_MISSING_CHECKPOINT',McByte_45_video_coverage='OFFICIAL_LIST_45_SEEN_FILES_NOT_DOWNLOADED',
        annotations466='NOT_VERIFIED_OFFICIAL_RAW508_NUMERIC_FILES_NOT_FLEXHOOK_LAYOUT',
        LaMOT_Sports_smoke='NOT_RUN_ASSETS_BLOCKED',official_subset_number='NOT_FOUND_MAIN_V5_OR_REPO_SUPPLEMENT_UNAVAILABLE',
        C3_C4_C5_C6='STOPPED_WAITING_OFFICIAL_ASSETS',tracker_generated=False)
    record['status']='PUBLIC_LAMOT_ANNOTATIONS_DOWNLOADED_OFFICIAL_FLEXHOOK_BINARY_ASSETS_BLOCKED'
    record['updated_UTC']=datetime.now(timezone.utc).isoformat();atomic_json(p,record)
    print(json.dumps(dict(status=record['status'],downloaded_annotations=len(downloaded),official_files_needing_manual_download=len(wanted),
        raw_videos=len(observed),links=record['image_links'])),flush=True)


if __name__=='__main__':main()
