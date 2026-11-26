"""Independent global-pair/group recomputation and official error synthesis."""
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
BASE=ROOT/'a100_runs/visual_classical_20261121/supervisor/current_temporal_revisit_audit_20261127'
PARENT=ROOT/'a100_runs/visual_classical_20261121/six_direction_20261122/fullscope_visual_six_20261126'
OWN=BASE/'independent_verification_registration.json'
OUT=BASE/'independent_verification_and_official_synthesis.json'
ACTIVE=True


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 assert not p.exists(),p
 with p.open('x') as f:f.write(json.dumps(d,ensure_ascii=False,indent=2)+'\n')


def grouped_revisits(draws,truth,obs):
 groups=defaultdict(list);query_positive={};n=len(obs)
 for order,d in enumerate(draws):
  q,i=divmod(int(d['uid']),n);assert int(d['query'])==q
  key=q,int(obs[i,1]);positive=bool(truth[q,i]);groups[key].append((order,float(d['probability']),positive))
  if positive and q not in query_positive:query_positive[q]=(key,order)
 first_indices={key:rows[0][0] for key,rows in groups.items()}
 c=Counter(events=len(draws),revisit_events=sum(len(rows)-1 for rows in groups.values()),
  canonical_positive_asked_events=sum(r[2] for rows in groups.values() for r in rows),
  first_query_positive_discoveries=len(query_positive),
  first_query_positive_discovered_on_revisit=sum(order>first_indices[key] for key,order in query_positive.values()),
  canonical_positive_queries=int(truth.any(1).sum()))
 c['canonical_positive_queries_never_asked_positive']=c['canonical_positive_queries']-len(query_positive)
 for key,rows in groups.items():
  positive=[r for r in rows if r[2]]
  if positive:
   c['first_candidate_ID_positive_discoveries']+=1
   c['first_candidate_ID_positive_discovered_on_revisit']+=int(positive[0][0]>rows[0][0])
  for a,b in zip(rows,rows[1:]):
   flip=(a[1]>=.5)!=(b[1]>=.5);stable=a[2]==b[2]
   confident=(a[1]<=.2 and b[1]>=.8) or (b[1]<=.2 and a[1]>=.8)
   c.update(teacher_hard_flips=int(flip),confident_opposite_answers=int(confident),
    teacher_hard_flip_with_canonical_membership_stable=int(flip and stable),
    teacher_hard_flip_with_canonical_membership_changed=int(flip and not stable),
    confident_opposite_with_canonical_membership_stable=int(confident and stable))
 return c


def main():
 contract=read(BASE/'registration.json');result=read(BASE/'result.json')
 assert result['status']=='COMPLETE_ALL148_CURRENT_TEMPORAL_REVISIT_AND_OFFICIAL_PRESENCE_DIAGNOSIS'
 post_path=PARENT/'postseal_diagnosis.json';post=read(post_path)
 old={(v['pair'],v['video']):v for v in post['videos']}
 paths={Path(__file__),BASE/'registration.json',BASE/'result.json',post_path}
 paths.update(map(Path,contract['inputs']))
 for u in contract['units']:paths.add(BASE/u['pair']/u['video']/'diagnosis.json')
 for p,expected in contract['inputs'].items():assert sha(p)==expected,p
 inputs={str(p):sha(p) for p in sorted(paths)}
 write(OWN,dict(status='REGISTERED_GLOBAL_LEXSORT_AND_GROUPED_EVENT_RECOMPUTATION',inputs=inputs,source_sha256=sha(__file__),
  method='All148: globally lexsorted same-track neighbor pairs and reduceat constant-state bound; grouped event histories for revisits; compare canonical errors to the previously completed postseal study. Add official IoU50 errors by true-GT presence category and fixed pooling correction/damage counts.',
  no_new_predictions_models_teacher_GPU_or_seeds=True,time_unix=time.time()))
 allowed=set(inputs)|{str(OWN),str(OUT)};allowed|={os.path.realpath(p) for p in allowed}
 libs=[Path(sys.prefix)/'lib',Path(sys.base_prefix)/'lib',Path(np.__file__).parent.parent]
 libraries=tuple(dict.fromkeys(str(q)+'/' for p in libs for q in (p,p.resolve())));denied=[]
 def guard(event,args):
  if not ACTIVE or event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=os.path.abspath(os.fsdecode(args[0]));mode=args[1] if len(args)>1 else 'r';flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
  write_mode=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT))
  permitted=p==str(OUT) if write_mode else p in allowed or p.startswith(libraries)
  if not permitted:denied.append(p);raise PermissionError('Unregistered independent temporal verification IO: '+p)
 sys.addaudithook(guard)
 checked=0;extra=[]
 def check(a,b):
  nonlocal checked
  assert int(a)==int(b),(a,b)
  checked+=1
 for u in contract['units']:
  d=read(BASE/u['pair']/u['video']/'diagnosis.json');obs=np.load(u['observations']);labels=np.load(u['labels']);truth=labels>=0
  qdoc=read(u['queries'])['queries'];union=np.load(u['union_ids']) if u['union_ids'] else None
  masks={m:np.load(p) for m,p in u['masks'].items()};nq,n=truth.shape
  order=np.lexsort((obs[:,0],obs[:,1]));starts=np.r_[0,1+np.flatnonzero(np.diff(obs[order,1]))]
  lengths=np.diff(np.r_[starts,n]);positives=np.add.reduceat(truth[:,order].astype(np.int64),starts,axis=1)
  floor=int(np.minimum(positives,lengths[None,:]-positives).sum())
  check(nq*len(starts),d['neighbors']['constant_candidate_bound']['query_track_groups'])
  check(int(((positives>0)&(positives<lengths)).sum()),d['neighbors']['constant_candidate_bound']['mixed_membership_query_track_groups'])
  check(floor,d['neighbors']['constant_candidate_bound']['constant_candidate_ID_canonical_row_error_floor'])
  left,right=order[:-1],order[1:];same_track=obs[left,1]==obs[right,1];gap=obs[right,0]-obs[left,0]
  stable=truth[:,left]==truth[:,right];exact=labels[:,left]==labels[:,right]
  identity=(union[left]==union[right])&(union[left]>=0) if union is not None else None
  for scope,keep in [('exact_frame_gap1',same_track&(gap==1)),('legacy_positive_gap_le2',same_track&(gap>0)&(gap<=2)),('all_consecutive_observations',same_track)]:
   prefix=scope+'/'
   den={prefix+'query_observation_neighbor_pairs':int(keep.sum())*nq,prefix+'membership_stable_pairs':int((stable&keep).sum()),prefix+'membership_changed_pairs':int((~stable&keep).sum())}
   if union is not None:
    identified=stable&keep&identity
    den[prefix+'same_union_identity_stable_membership_pairs']=int(identified.sum());den[prefix+'same_union_identity_exact_query_label_pairs']=int((identified&exact).sum())
   for k,v in den.items():check(v,d['neighbors']['denominators'].get(k,0))
   for m,mask in masks.items():
    flip=mask[:,left]!=mask[:,right];counts={prefix+'all_flips':int((flip&keep).sum()),prefix+'membership_stable_flips':int((flip&keep&stable).sum()),prefix+'membership_changed_flips':int((flip&keep&~stable).sum())}
    if union is not None:counts.update({prefix+'same_union_identity_stable_membership_flips':int((flip&identified).sum()),prefix+'same_union_identity_exact_query_label_flips':int((flip&identified&exact).sum())})
    for k,v in counts.items():check(v,d['neighbors']['methods'][m].get(k,0))
    if 'POOLED' in m:assert not counts[prefix+'all_flips']
  for pol,key in [('active','active_draws'),('uniform','uniform_draws')]:
   c=grouped_revisits(read(u[key])['draws'],truth,obs);expected=d['revisits'][pol]
   for k in c.keys()|expected.keys():check(c.get(k,0),expected.get(k,0))
  csv_rows={}
  for m,path in u['official_csv'].items():
   with Path(path).open() as f:csv_rows[m]={r['seq']:r for r in csv.DictReader(f) if r['seq']!='COMBINED'}
  gt=np.array([int(float(csv_rows['FULL_SCOPE_VISUAL_RSI'][u['video']+'+'+q['query_id']]['GT_Dets'])) for q in qdoc]);cat=np.where(gt==0,0,np.where(truth.any(1),2,1))
  official={};nonconstant={}
  for m,mask in masks.items():
   expected=old[u['pair'],u['video']]['methods'][m]['canonical']
   fp=int((mask&~truth).sum());fn=int((~mask&truth).sum());tp=int((mask&truth).sum())
   check(fp,expected['FP']);check(fn,expected['FN']);check(tp,expected['TP'])
   check(fp,sum(c['methods'][m]['canonical_FP_rows'] for c in d['errors'].values()))
   check(fn,sum(c['methods'][m]['canonical_FN_rows'] for c in d['errors'].values()))
   if 'POOLED' in m:assert fp+fn>=floor
   selected=np.add.reduceat(mask[:,order].astype(np.int64),starts,axis=1);nonconstant[m]=int(((selected>0)&(selected<lengths)).sum())
   official[m]={}
   for cid,name in enumerate(('official_GT_absent','official_GT_present_but_canonical_zero','canonical_positive_support')):
    rows=[csv_rows[m][u['video']+'+'+qdoc[q]['query_id']] for q in np.flatnonzero(cat==cid)]
    official[m][name]={key:int(sum(float(r['HOTA_'+key+'___50']) for r in rows)) for key in ('TP','FP','FN')}
   for key in ('TP','FP','FN'):check(sum(v[key] for v in official[m].values()),old[u['pair'],u['video']]['methods'][m]['official_IoU50'][key])
  changed={}
  for raw,pooled in [('SOFT_KD288','SOFT_POOLED'),('UNIFORM_SOFT_KD288','UNIFORM_POOLED')]:
   a,b=masks[raw],masks[pooled];different=a!=b
   changed[raw]=dict(changed_rows=int(different.sum()),canonical_errors_corrected=int((different&(b==truth)).sum()),canonical_errors_introduced=int((different&(a==truth)).sum()))
   assert sum(changed[raw][k] for k in ('canonical_errors_corrected','canonical_errors_introduced'))==changed[raw]['changed_rows']
  extra.append(dict(pair=u['pair'],video=u['video'],official_IoU50_by_presence=official,nonconstant_query_candidate_ID_groups=nonconstant,pooling_row_changes=changed))
 aggregate={}
 for pair in sorted({r['pair'] for r in extra}):
  rows=[x for x in extra if x['pair']==pair];official={};pooling={}
  for m in masks:
   official[m]={name:{key:sum(x['official_IoU50_by_presence'][m][name][key] for x in rows) for key in ('TP','FP','FN')} for name in ('official_GT_absent','official_GT_present_but_canonical_zero','canonical_positive_support')}
  for m in ('SOFT_KD288','UNIFORM_SOFT_KD288'):pooling[m]={k:sum(x['pooling_row_changes'][m][k] for x in rows) for k in ('changed_rows','canonical_errors_corrected','canonical_errors_introduced')}
  aggregate[pair]=dict(videos=len(rows),official_IoU50_by_presence=official,pooling_row_changes=pooling,nonconstant_query_candidate_ID_groups={m:sum(x['nonconstant_query_candidate_ID_groups'][m] for x in rows) for m in masks})
 for p,expected in inputs.items():assert sha(p)==expected,p
 write(OUT,dict(status='PASS_ALL148_GLOBAL_PAIR_GROUP_REPRODUCTION_AND_OFFICIAL_ERROR_SYNTHESIS',verified_integer_cells=checked,integer_max_difference=0,units=len(extra),aggregate=aggregate,
  registration_sha256=sha(OWN),source_sha256=sha(__file__),no_training_predictions_teacher_GPU_or_new_seeds=True,rejected_IO=denied,
  boundary='OfficialIoU50 category errors are exact count summaries, not additive HOTA attribution; pooling row corrections use canonical membership only. No new memory model has been evaluated.',time_unix=time.time()))
 print(json.dumps(dict(status='PASS_ALL148_GLOBAL_PAIR_GROUP_REPRODUCTION_AND_OFFICIAL_ERROR_SYNTHESIS',units=len(extra),verified_integer_cells=checked,output=str(OUT))),flush=True)


if __name__=='__main__':
 try:main()
 finally:ACTIVE=False
