"""Isolated diagnosis of already sealed predictions; never an adaptation input."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path('/data3/liuyeqiang/FlexHook-TTA')
SUP=ROOT/'a100_runs/visual_classical_20261121/supervisor'
BASE=ROOT/'a100_runs/visual_classical_20261121/six_direction_20261122/fullscope_visual_six_20261126'
DIAG=ROOT/'a100_runs/mechanism_20261121'
DEST=SUP/'current_temporal_revisit_audit_20261127'
OWN=DEST/'registration.json'
OUT=DEST/'result.json'
CANONICAL={'C1':'C1','C2':'C5','C3':'C4','C4':'C4','C5':'C5','C6':'C1'}
METHODS=('FULL_SCOPE_VISUAL_RSI','SOFT_KD288','SOFT_POOLED','UNIFORM_SOFT_KD288','UNIFORM_POOLED')
PLAN_SHA='d159ed56af4d614edd68c991ab752f6341074752c648650cd61d336d5a68ba12'
SEAL_SHA='2a33eaf9cc1178797bd3c577df74534e6925d4870ca2e15be56a97232120440d'
ACTIVE=True


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 assert not p.exists(),p
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:f.write(json.dumps(d,ensure_ascii=False,indent=2)+'\n')


def prepare():
 assert not OWN.exists() and sha(BASE/'registration.json')==PLAN_SHA
 assert sha(BASE/'all4588_sealed_before_GT.json')==SEAL_SHA
 repro=ROOT/'docs/a100_six_direction_20261122/fullscope_visual_six_20261126_reproduction.json'
 assert read(repro)['status']=='PASS_ALL186_HOTA_AND180_EFFECTS'
 plan=read(BASE/'registration.json');assert len(plan['cases'])==148
 paths={Path(__file__),BASE/'registration.json',BASE/'all4588_sealed_before_GT.json',repro,
   ROOT/'scripts/a100_mechanism_offline.py',ROOT/'scripts/a100_fullscope_visual_postseal.py',ROOT/'scripts/a100_observation_evidence_transfer_diagnose.py'}
 units=[]
 for case in plan['cases']:
  pair,video=case['pair'],case['evaluated_video'];folder=BASE/'units'/pair/video
  labels=DIAG/'offline_GT'/CANONICAL[pair]/video/'labels.npy'
  union=labels.parent/'union_ids.npy' if case['target']!='KITTI-V2' else None
  unit=dict(pair=pair,video=video,target=case['target'],observations=case['observations'],queries=case['temporal_queries'],
   labels=str(labels),union_ids=str(union) if union else None,
   active_draws=str(folder/'primary/draws.json'),uniform_draws=str(folder/'uniform_draws.json'),
   masks={name:str(folder/'predictions'/name/'mask.npy') for name in METHODS},
   official_csv={name:str(folder/'predictions'/name/'results/pedestrian_detailed.csv') for name in METHODS})
  paths.update(Path(unit[k]) for k in ('observations','queries','labels','active_draws','uniform_draws'))
  if union:paths.add(union)
  paths.update(map(Path,unit['masks'].values()));paths.update(map(Path,unit['official_csv'].values()))
  units.append(unit)
 for name,expected in plan['source_sha256'].items():
  p=ROOT/name;assert sha(p)==expected;paths.add(p)
 write(OWN,dict(status='REGISTERED_ALL148_CURRENT_MASK_TEMPORAL_REVISIT_AND_OFFICIAL_ABSENCE_DIAGNOSIS',
  source_sha256=sha(__file__),inputs={str(p):sha(p) for p in sorted(paths)},units=units,methods=METHODS,
  hypotheses=['Current framewise learned masks may fluctuate despite oldv8P being track-constant.',
   'Object revisits may reveal positive observations at another time or repeat already available evidence; metadata coverage alone cannot decide utility.',
   'A query without canonical positive candidate rows can still have official GT detections; existence and candidate support must be separated.'],
  definitions=dict(neighbor_scopes=['exact_frame_gap1','legacy_positive_gap_le2','all_consecutive_observations'],
   stability='Canonical membership equality; separately same matched query-union GTID plus membership equality for Dance/Sports only. KITTI legacy labels0/-1 do not encode identity.',
   hard_teacher_probability_threshold=.5,confident_low=.2,confident_high=.8,
   presence='Per-query official GT_Dets, checked against HOTA_TP___50 + HOTA_FN___50 and across all5 frozen methods.',
   revisit='Previously asked the same official candidate TrackID for this query at a different observation; repeated actions remain forbidden.'),
  restrictions=['No fits, new masks, checkpoints, teacher calls, inference, sampling seeds or GPU operations.',
   'No filters or choices from this manual-GT diagnosis are returned to adaptation.',
   'Identity from union GT covers only annotated query-union matches, not every physical object.',
   'Row error floors and temporal counts are not HOTA bounds or additive causal error shares.',
   'Time-varying membership may reflect action, spatial changes, visibility, matching or ID mixing; do not automatically call every change semantic flicker.',
   'Temporal averaging is track-constant by construction. This diagnosis does not establish a causal benefit of any new memory model.',
   'All data remain DEV; no independent test or statistical significance claim.'],goal_achieved=False,time_unix=time.time()))
 print(json.dumps(dict(status='REGISTERED_CURRENT_MASK_DIAGNOSIS',units=148,inputs=len(paths))),flush=True)


def neighbor_counts(bits,labels,union,obs):
 truth=labels>=0;total=Counter();mask_totals={k:Counter() for k in bits};constant=Counter()
 for tid in np.unique(obs[:,1]):
  rows=np.flatnonzero(obs[:,1]==tid);rows=rows[np.argsort(obs[rows,0],kind='stable')]
  p=truth[:,rows];positive=p.sum(1);negative=len(rows)-positive
  constant.update(query_track_groups=len(positive),mixed_membership_query_track_groups=int(((positive>0)&(negative>0)).sum()),
    constant_candidate_ID_canonical_row_error_floor=int(np.minimum(positive,negative).sum()))
  if len(rows)<2:continue
  left,right=rows[:-1],rows[1:];gaps=obs[right,0]-obs[left,0];assert (gaps>0).all()
  stable=truth[:,left]==truth[:,right];same_query_label=labels[:,left]==labels[:,right]
  if union is not None:same_identity=(union[left]==union[right])&(union[left]>=0)
  for name,include in [('exact_frame_gap1',gaps==1),('legacy_positive_gap_le2',gaps<=2),('all_consecutive_observations',np.ones(len(gaps),bool))]:
   keep=include[None,:];total[name+'/query_observation_neighbor_pairs']+=int(keep.sum())*len(truth)
   total[name+'/membership_stable_pairs']+=int((keep&stable).sum())
   total[name+'/membership_changed_pairs']+=int((keep&~stable).sum())
   if union is not None:
    identified=keep&same_identity[None,:]&stable
    exact_identified=identified&same_query_label
    total[name+'/same_union_identity_stable_membership_pairs']+=int(identified.sum())
    total[name+'/same_union_identity_exact_query_label_pairs']+=int(exact_identified.sum())
   for method,mask in bits.items():
    flips=mask[:,left]!=mask[:,right];c=mask_totals[method]
    c[name+'/all_flips']+=int((keep&flips).sum())
    c[name+'/membership_stable_flips']+=int((keep&stable&flips).sum())
    c[name+'/membership_changed_flips']+=int((keep&~stable&flips).sum())
    if union is not None:
     c[name+'/same_union_identity_stable_membership_flips']+=int((identified&flips).sum())
     c[name+'/same_union_identity_exact_query_label_flips']+=int((exact_identified&flips).sum())
 return dict(denominators=dict(total),methods={k:dict(v) for k,v in mask_totals.items()},constant_candidate_bound=dict(constant),identity_available=union is not None)


def revisit_counts(draws,truth,obs):
 seen=set();positive_queries=set();positive_tracks=set();past={};c=Counter();n=len(obs)
 for d in draws:
  q,oi=divmod(int(d['uid']),n);tid=int(obs[oi,1]);key=q,tid;repeat=key in seen;positive=bool(truth[q,oi]);p=float(d['probability'])
  c['events']+=1;c['revisit_events']+=int(repeat);c['canonical_positive_asked_events']+=int(positive)
  if positive and q not in positive_queries:
   c['first_query_positive_discoveries']+=1;c['first_query_positive_discovered_on_revisit']+=int(repeat)
  if positive and key not in positive_tracks:
   c['first_candidate_ID_positive_discoveries']+=1;c['first_candidate_ID_positive_discovered_on_revisit']+=int(repeat)
  if repeat:
   old=past[key];flip=(p>=.5)!=(old[0]>=.5);stable=positive==old[1]
   confident=(p>=.8 and old[0]<=.2) or (p<=.2 and old[0]>=.8)
   c['teacher_hard_flips']+=int(flip);c['confident_opposite_answers']+=int(confident)
   c['teacher_hard_flip_with_canonical_membership_stable']+=int(flip and stable)
   c['teacher_hard_flip_with_canonical_membership_changed']+=int(flip and not stable)
   c['confident_opposite_with_canonical_membership_stable']+=int(confident and stable)
  seen.add(key);past[key]=(p,positive)
  if positive:positive_queries.add(q);positive_tracks.add(key)
 c['canonical_positive_queries']=int(truth.any(1).sum());c['canonical_positive_queries_never_asked_positive']=c['canonical_positive_queries']-len(positive_queries)
 return dict(c)


def main():
 contract=read(OWN);assert sha(__file__)==contract['source_sha256'] and not OUT.exists()
 for p,expected in contract['inputs'].items():assert sha(p)==expected,p
 allowed=set(contract['inputs'])|{str(OWN)};allowed|={os.path.realpath(x) for x in allowed}
 libs=[Path(sys.prefix)/'lib',Path(sys.base_prefix)/'lib',Path(np.__file__).parent.parent]
 libraries=tuple(dict.fromkeys(str(x)+'/' for p in libs for x in (p,p.resolve())))
 outputs=(str(DEST)+'/',str(DEST.resolve())+'/');denied=[]
 def guard(event,args):
  if not ACTIVE or event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=os.path.abspath(os.fsdecode(args[0]));mode=args[1] if len(args)>1 else 'r';flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
  writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT))
  permit=p.startswith(outputs) if writing else p in allowed or p.startswith(libraries+outputs)
  if not permit:denied.append(p);raise PermissionError('Unregistered current-output evaluation IO: '+p)
 sys.addaudithook(guard)
 results=[];aggregate={};start=time.perf_counter()
 for unit in contract['units']:
  obs=np.load(unit['observations'],allow_pickle=False);queries=read(unit['queries'])['queries'];labels=np.load(unit['labels'],allow_pickle=False);truth=labels>=0
  assert labels.shape==(len(queries),len(obs));union=np.load(unit['union_ids'],allow_pickle=False) if unit['union_ids'] else None
  if union is not None:assert union.shape==(len(obs),)
  bits={k:np.load(p,allow_pickle=False) for k,p in unit['masks'].items()}
  assert all(x.dtype==bool and x.shape==truth.shape for x in bits.values())
  gt_counts=None
  for method,p in unit['official_csv'].items():
   with Path(p).open() as f:rows={r['seq']:r for r in csv.DictReader(f) if r['seq']!='COMBINED'}
   assert len(rows)==len(queries)
   counts=[]
   for q in queries:
    row=rows[unit['video']+'+'+q['query_id']];count=float(row['GT_Dets'])
    assert count>=0 and count.is_integer() and count==float(row['HOTA_TP___50'])+float(row['HOTA_FN___50'])
    counts.append(int(count))
   if gt_counts is None:gt_counts=np.asarray(counts,dtype=np.int64)
   else:assert np.array_equal(gt_counts,counts)
  category=np.where(gt_counts==0,0,np.where(truth.any(1),2,1))
  assert not truth[category==0].any(),(unit['pair'],unit['video'],'CanonicalpositivewithoutofficialGT')
  errors={}
  for category_id,name in enumerate(('official_GT_absent','official_GT_present_but_canonical_zero','canonical_positive_support')):
   chosen=category==category_id;c=Counter(queries=int(chosen.sum()),official_GT_observations=int(gt_counts[chosen].sum()),canonical_positive_observations=int(truth[chosen].sum()))
   methods={}
   for method,mask in bits.items():
    methods[method]=dict(selected_candidate_rows=int(mask[chosen].sum()),canonical_FP_rows=int((mask[chosen]&~truth[chosen]).sum()),
      canonical_FN_rows=int((~mask[chosen]&truth[chosen]).sum()),zero_output_queries=int((~mask[chosen].any(1)).sum()))
   errors[name]=dict(counts=dict(c),methods=methods)
  neighbor=neighbor_counts(bits,labels,union,obs)
  active=read(unit['active_draws'])['draws'];uniform=read(unit['uniform_draws'])['draws']
  assert len(active)==len(uniform)==len({d['uid'] for d in active})==len({d['uid'] for d in uniform})
  revisit={k:revisit_counts(d,truth,obs) for k,d in [('active',active),('uniform',uniform)]}
  row=dict(pair=unit['pair'],video=unit['video'],queries=len(queries),observations=len(obs),errors=errors,neighbors=neighbor,revisits=revisit)
  write(DEST/unit['pair']/unit['video']/'diagnosis.json',row);results.append(row)
  print(json.dumps(dict(status='CURRENT_OUTPUT_TEMPORAL_REVISIT_UNIT_DIAGNOSED',pair=unit['pair'],video=unit['video'])),flush=True)
 for pair in sorted({x['pair'] for x in results}):
  group=[x for x in results if x['pair']==pair]
  def combine(get):
   c=Counter()
   for x in group:c.update(get(x))
   return dict(c)
  errors={name:dict(counts=combine(lambda x,name=name:x['errors'][name]['counts']),methods={m:combine(lambda x,name=name,m=m:x['errors'][name]['methods'][m]) for m in METHODS}) for name in group[0]['errors']}
  neighbors=dict(denominators=combine(lambda x:x['neighbors']['denominators']),methods={m:combine(lambda x,m=m:x['neighbors']['methods'][m]) for m in METHODS},constant_candidate_bound=combine(lambda x:x['neighbors']['constant_candidate_bound']),identity_supported_videos=sum(x['neighbors']['identity_available'] for x in group))
  aggregate[pair]=dict(videos=len(group),errors=errors,neighbors=neighbors,revisits={pol:combine(lambda x,pol=pol:x['revisits'][pol]) for pol in ('active','uniform')})
 assert len(results)==148 and sum(x['queries'] for x in results)==3506
 assert sum(x['revisits']['active']['events'] for x in results)==55019
 assert sum(x['revisits']['active'].get('revisit_events',0) for x in results)==9726
 assert sum(x['revisits']['active'].get('teacher_hard_flips',0) for x in results)==1963
 for p,expected in contract['inputs'].items():assert sha(p)==expected,p
 write(OUT,dict(status='COMPLETE_ALL148_CURRENT_TEMPORAL_REVISIT_AND_OFFICIAL_PRESENCE_DIAGNOSIS',aggregate=aggregate,
  registration_sha256=sha(OWN),source_sha256=sha(__file__),units=148,query_instances=3506,source_inputs_unchanged=True,
  no_new_training_inference_teacher_GPU_predictions_or_seeds=True,rejected_IO=denied,wall_seconds=time.perf_counter()-start,
  interpretation=contract['restrictions'],goal_achieved=False))
 print(json.dumps(dict(status='COMPLETE_ALL148_CURRENT_TEMPORAL_REVISIT_AND_OFFICIAL_PRESENCE_DIAGNOSIS',output=str(OUT))),flush=True)


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
 try:prepare() if args.prepare else main()
 finally:ACTIVE=False
