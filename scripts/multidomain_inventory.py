"""P0 only: no inference, training, GT payloads, teacher or downloads."""
from collections import Counter
import configparser
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_tta.mot_interface.trackers import merge_kitti_class_splits

RUN=ROOT/'results/runs/multidomain_generality_20261101'
PUBLIC=ROOT/'results/multidomain_generality'
VENDOR=ROOT/'vendor/FlexHook'
V2=ROOT/'results/runs/diag_attrib_20261031_v2'
SOURCE=ROOT/'results/runs/xquery_decode_20261101/source'

def guard():
    attempts=[]
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0]))
        if p.name in ('gt.txt','labels.json') or p.name.endswith('_labels.json') or any(k in p.parts for k in ('gt_template','gt_templates','expression','annotations_v1')):
            attempts.append(str(p));raise PermissionError('P0 must not open GT/referring annotation payloads: '+str(p))
    sys.addaudithook(audit);return attempts

def info(path,sha=False):
    p=Path(path);r=dict(path=str(p),exists=p.exists(),is_symlink=p.is_symlink(),resolved=str(p.resolve()))
    if sha and p.is_file():r.update(bytes=p.stat().st_size,sha256=file_sha(p))
    return r

def seqmap(name):
    p=VENDOR/'seqmaps'/(name+'.txt');rows=list(csv.reader(p.read_text().splitlines(),delimiter=' '));parsed=[r[0] for r in rows]
    videos=Counter(r.split('+',1)[0] for r in parsed)
    raw=p.read_bytes()
    return dict(**info(p,True),rows=len(parsed),unique_rows=len(set(parsed)),LF_count=raw.count(b'\n'),ends_with_LF=raw.endswith(b'\n'),videos=len(videos),per_video_expressions=dict(sorted(videos.items())),
        duplicated_rows={s:n for s,n in Counter(parsed).items() if n>1},first_row=parsed[0],parser_skips_header=False)

def historical(folder,expected,episodes):
    folder=Path(folder);done=json.loads((folder/'official_completion.json').read_text())
    summary=folder/'results/pedestrian_summary.txt';detail=folder/'results/pedestrian_detailed.csv'
    if file_sha(summary)!=done['summary_sha256'] or file_sha(detail)!=done['detailed_sha256']:raise ValueError('changed official reference '+str(folder))
    header,values=summary.read_text().splitlines()[:2];metrics=dict(zip(header.split(),map(float,values.split())))
    if metrics['HOTA']!=expected:raise ValueError('STOP: reference HOTA mismatch '+str(folder))
    s=json.loads((folder/'predictions_seal.json').read_text())
    if len(s['episodes'])!=episodes:raise ValueError('reference episode coverage')
    for e in s['episodes']:
        if file_sha(e['prediction_path'])!=e['prediction_sha256']:raise ValueError('reference prediction changed')
    return dict(status='PASS_HASH_VERIFIED_EXISTING_OFFICIAL_RESULT_NO_RERUN',HOTA=metrics['HOTA'],prediction_rows=metrics['Dets'],
        GT_Dets=metrics['GT_Dets'],episodes=episodes,summary=info(summary,True),detailed=info(detail,True),seal=info(folder/'predictions_seal.json',True),
        GT_open_attempts=0,new_inference=False,new_TrackEval_job=False)

def run():
    start=time.perf_counter();RUN.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    pub=json.loads((RUN/'prereg_publication.json').read_text())
    if not pub['remote_verified'] or file_sha(PUBLIC/'PREREG.md')!=pub['prereg_sha256']:raise ValueError('publish fixed protocol first')
    attempts=guard();models={}
    paths={'V2-best':Path('/data3/user/TTA-RMOT-assets/flexhook/SOTA_ckpts/refer-kitti-v2-best.pth'),
        'Dance-best':Path('/data3/user/TTA-RMOT-assets/flexhook/SOTA_ckpts/refer-dance-best.pth'),
        'LaMOT-best':VENDOR/'SOTA_ckpts/LaMOT-best.pth'}
    search_roots=['/data2/user','/data3/user','/home/user']
    found=subprocess.run(['rg','--files','--hidden',*search_roots,'-g','*LaMOT*','-g','*lamot*','-g','!**/.git/**','-g','!**/node_modules/**'],capture_output=True,text=True)
    candidates=[p for p in found.stdout.splitlines() if Path(p).name.lower() in ('lamot-best.pth','lamot-best.pt')]
    if candidates:paths['LaMOT-best']=Path(sorted(candidates)[0])
    for m,p in paths.items():models[m]=dict(**info(p,True),status='PRESENT_OFFICIAL_FILE' if p.is_file() else 'MISSING_STOP')
    expected={'V2-best':'71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613'}
    if models['V2-best'].get('sha256')!=expected['V2-best']:raise ValueError('audited V2 checkpoint changed')
    smap={k:seqmap(s) for k,s in [('KITTI-V2','kitti-2'),('Dance','dance'),('SportsMOT','SportsMOT')]}
    source=json.loads((SOURCE/'metadata.json').read_text());dance=json.loads((V2/'score_bank/bank.json').read_text())
    source_tracks=VENDOR/'tracker_outputs/Temp-NeuralSORT-kitti2';source_rows=[]
    for video,qcount in smap['KITTI-V2']['per_video_expressions'].items():
        files=[source_tracks/video/c/'predict.txt' for c in ('car','pedestrian')];rows=merge_kitti_class_splits(*files)
        images=Path('/data3/user/TTA-RMOT-assets/flexhook/refer-kitti-v2/KITTI/training/image_02')/video
        missing=sum(not (images/f'{r.frame_id-1:06d}.png').is_file() for r in rows)
        source_rows.append(dict(video=video,expressions=qcount,full_observations=len(rows),historical_observations=len(source['videos'][video]['observations']),
            physical_tracks=len({r.track_id for r in rows}),images=len(list(images.glob('*.png'))),missing_candidate_images=missing,tracker_files=[info(p,True) for p in files]))
    dance_tracks=Path('/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs');dance_rows=[]
    for video,qcount in smap['Dance']['per_video_expressions'].items():
        p=dance_tracks/(video+'.txt');rows=[list(map(float,s.split(','))) for s in p.read_text().splitlines() if s.strip()]
        image_root=next(q['image_root'] for q in dance['episodes'] if q['video_id']==video);ir=Path(image_root)
        dance_rows.append(dict(video=video,expressions=qcount,full_observations=len(rows),physical_tracks=len({int(r[1]) for r in rows}),
            images=len(list(ir.glob('*.png'))),missing_candidate_images=sum(not (ir/f'{int(r[0]):06d}.png').is_file() for r in rows),tracker=info(p,True)))
    sports=VENDOR/'datasets/LaMOT-main';raw=Path('/data2/user/LocateMOT_FullDataV2_spill/datasets/SportsMOT/val');raw_rows=[]
    for video in smap['SportsMOT']['per_video_expressions']:
        image_root=raw/video/'img1';indices=sorted(int(p.stem) for p in image_root.glob('*.jpg') if p.stem.isdigit());ini=raw/video/'seqinfo.ini'
        config=configparser.ConfigParser();config.read(ini);expected_frames=config.getint('Sequence','seqLength',fallback=None)
        raw_rows.append(dict(video=video,image_root=str(image_root),images=len(indices),seqinfo=info(ini,True),expected_frames=expected_frames,
            contiguous=indices==list(range(1,len(indices)+1)),complete=bool(indices) and len(indices)==expected_frames))
    required=[sports/'sequences/val/SportsMOT-sequences',sports/'annotations_v1/val/SportsMOT',sports/'sequences/val/SportsMOT_labels.json',
        sports/'sequences/val/SportsMOT_RESOLUTION.json',sports/'sequences/val/TAO_frame2file.json',VENDOR/'tracker_outputs/McByte-SportsMOT']
    targets={}
    for target,rows,trackroot in [('KITTI-V2',source_rows,source_tracks),('Dance',dance_rows,dance_tracks)]:
        targets[target]=dict(seqmap=smap[target],videos=len(rows),expressions=smap[target]['rows'],tracker_root=info(trackroot),
            full_candidate_observations=sum(r['full_observations'] for r in rows),full_candidate_expression_pairs=sum(r['full_observations']*r['expressions'] for r in rows),
            historical_candidate_expression_pairs=source['candidate_query_rows'] if target=='KITTI-V2' else 210393*17,
            per_video=rows)
    targets['SportsMOT']=dict(seqmap=smap['SportsMOT'],videos=45,requested_expressions=465,actual_seqmap_expressions=smap['SportsMOT']['rows'],
        full_candidate_observations=None,full_candidate_expression_pairs=None,required_paths=[info(p) for p in required],
        raw_images_discovered_outside_project=str(raw),raw_images_videos_complete=sum(r['complete'] for r in raw_rows),raw_images_count=sum(r['images'] for r in raw_rows),raw_images_inventory=raw_rows,
        other_project_artifacts_used=False,note='Only original image filenames and seqinfo metadata inspected; no LocateMOT code, checkpoints, predictions or labels used; no links modified.')
    references={'V2-best_KITTI_historical':historical(SOURCE.parent/'source_boundary/evaluation/FROZEN__NATIVE',42.526,862),
        'Dance-best_Dance_historical':historical(V2/'evaluation/HIST_OFFICIAL_DANCE_COMPLETE__OFFICIAL_DANCE_COMPLETE',32.134,425),
        'Dance-best_Dance_complete':historical(V2/'evaluation/OFFICIAL_DANCE_COMPLETE',32.139,425),
        'V2-best_Dance_complete':historical(V2/'evaluation/M_FROZEN__FROZEN',8.218,425)}
    bank=json.loads((V2/'score_bank/bank.json').read_text());margins=[]
    for name in ('FROZEN','OFFICIAL_DANCE_COMPLETE'):
        p=V2/'score_bank'/(name+'.npy');h=bank['cache_files_sha256'][name+'.npy']
        if file_sha(p)!=h:raise ValueError('historical margin bank changed '+name)
        margins.append(dict(role='REUSABLE_OFFICIAL_MARGIN_BANK',model='V2-best' if name=='FROZEN' else 'Dance-best',target='Dance',**info(p,True),candidate_expression_pairs=3588445))
    p=Path(source['source_score_path'])
    if file_sha(p)!=source['source_score_sha256']:raise ValueError('source historical margin changed')
    margins.append(dict(role='HISTORICAL_ONLY_NOT_COMPLETE_TAIL_COVERAGE',model='V2-best',target='KITTI-V2',**info(p,True),candidate_expression_pairs=2355528))
    missing=[dict(kind='CHECKPOINT',name=m,path=r['path']) for m,r in models.items() if not r['exists']]
    missing.extend(dict(kind='SPORTS_REQUIRED_PATH',path=str(p)) for p in required if not p.exists())
    mismatch=smap['SportsMOT']['rows']!=465
    status='STOPPED_P0_MISSING_ASSETS_AND_SEQMAP_MISMATCH' if missing or mismatch else 'ASSETS_PRESENT_REQUIRES_SMOKES_BEFORE_P1'
    receipt=dict(status=status,root=str(ROOT),base_commit='321ec54899a25f836b18bb127da2578fbb5e32ec',prereg_publication=pub,models=models,targets=targets,
        missing_assets=missing,search_roots=search_roots,checkpoint_name_matches=candidates,search_exit_code=found.returncode,search_errors=found.stderr.splitlines(),
        SportsMOT_seqmap_mismatch=dict(expected=465,observed=smap['SportsMOT']['rows'],unique=smap['SportsMOT']['unique_rows'],removed_or_added_rows=0,
            newline_count=smap['SportsMOT']['LF_count'],terminal_newline=smap['SportsMOT']['ends_with_LF'],explanation='wc -l counts465 newline bytes, but there are466 unique valid sequences; last line has no LF',
            parser='vendor/FlexHook/TrackEval/trackeval/datasets/mot_challenge_2d_box.py:168-177; header-skipping code commented out'),
        code_findings=dict(mix_last_observation_omission=True,mix_sampling='range(-sample_frame_num+1,obj_last-sample_frame_num,1), exclusive end omits last observation',
            mix_test_constructor_opens_GT='data/mydataloader_mix.py:270 unconditionally reads SportsMOT_labels.json; must not execute as-is under current GT ban',
            normal_test_loader_preexisting_GT_free_change='data/mydataloader.py:171-177; preserved',
            checkpoint_backbone='kitti/dance yaml VISUAL=resnet34 but infer.sh official best commands override --visual rope-swin-tiny; do not infer incompatibility from raw YAML alone',
            tokenizer_policy='Use unchanged compat/sitecustomize.py canonical vocab.json+merges.txt ByteLevel BPE with RobertaProcessing; no cross-source category/text remapping',
            actual_runtime_token_checks='NOT_RUN_P0_HARD_STOP',SportsMOT_official_paper_subset_number='NOT_SEARCHED_DUE_P0_STOP; infer.sh only establishes mix HOTA56.756, not SportsMOT subset'),
        smoke_status='NOT_RUN_GLOBAL_P0_HARD_STOP',GPU_hours_estimates=None,new_GPU_process_hours=0,model_loads=0,checkpoint_updates=0,teacher_calls=0,GT_open_attempts=len(attempts),
        downloads=0,links_modified=0,new_official_evaluation_jobs=0,P1_started=False,P2_started=False,wall_seconds=time.perf_counter()-start)
    if attempts:raise RuntimeError('P0 attempted GT payload')
    atomic_json(PUBLIC/'inventory.json',receipt);atomic_json(PUBLIC/'reproduction.json',dict(status='EXISTING_REFERENCES_VERIFIED_NO_NEW_EVALUATION',references=references,
        LaMOT_best_SportsMOT=dict(status='NOT_RUN_MISSING_CHECKPOINT_AND_DATA_PACKAGE'),all9_smokes='NOT_RUN',GT_open_attempts=0))
    atomic_json(PUBLIC/'margin_paths.json',dict(status='HISTORICAL_BANKS_HASH_VERIFIED_ONLY',files=margins,new_inference=False))
    with gzip.open(PUBLIC/'reused_prediction_files.csv.gz','wt') as f:
        writer=csv.DictWriter(f,fieldnames=['reference','video','expression','prediction_path','prediction_sha256','selected_rows']);writer.writeheader()
        for reference,r in references.items():
            for e in json.loads(Path(r['seal']['path']).read_text())['episodes']:
                writer.writerow(dict(reference=reference,**{k:e[k] for k in writer.fieldnames if k in e}))
    print(json.dumps(dict(status=status,missing_assets=missing,SportsMOT_actual=smap['SportsMOT']['rows'],raw_SportsMOT_complete_videos=sum(r['complete'] for r in raw_rows),new_GPU_hours=0,GT_open_attempts=len(attempts),targets={t:{k:r.get(k) for k in ('videos','expressions','full_candidate_observations','full_candidate_expression_pairs')} for t,r in targets.items()}),indent=2),flush=True)

if __name__=='__main__':run()
