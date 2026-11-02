"""CPU-only, fixed ORACLE/GT track-ranking diagnostics. No model imports."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scripts.search_feasibility_common import sha, dump, record, checked, json_rows, csv_rows, boundary_search

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'results/track_ranking'
RUN = ROOT/'results/runs/track_ranking_20261102'
SOURCE_COMMIT = '01add8876fb8bb8cb092a22e98c67f29b48addac'
SOURCE_BLOB = '8b3fcfdbe3ff3f11f9eaa92b54acaeeab0d24464'
TARGETS = ('KITTI-V2','SportsMOT','Dance')
MODELS = ('V2-best','Dance-best','LaMOT-best')
IN_DOMAIN = {'KITTI-V2':'V2-best','SportsMOT':'LaMOT-best','Dance':'Dance-best'}
PAIRS = [('C1','V2-best','Dance'),('C2','Dance-best','KITTI-V2'),('C3','V2-best','SportsMOT'),
         ('C4','Dance-best','SportsMOT'),('C5','LaMOT-best','KITTI-V2'),('C6','LaMOT-best','Dance')]
AGGS = ('MEAN','MEDIAN','TOP25','LOGIT_MEAN')
FPS = {'KITTI-V2':10,'Dance':20,'SportsMOT':25}
ROLE = 'ORACLE_GT_ONLY_DIAGNOSTIC_NOT_A_METHOD'


def require_plan():
    p = json.loads((PUBLIC/'prereg_publication.json').read_text())
    if not p['remote_verified'] or sha(PUBLIC/'PREREG.md') != p['prereg_sha256']:
        raise PermissionError('Publish unchanged track-ranking preregistration first')
    return p


def seed_for(target,video,query):
    key = json.dumps([target,video,query],ensure_ascii=False,separators=(',',':'))
    return int.from_bytes(hashlib.sha256(('0|'+key).encode()).digest()[:4],'big')


def frame_segments(frames,length):
    """Indices in temporal order, anchored at track start; gaps are not filled."""
    frames = np.asarray(frames,dtype=np.int64)
    if not len(frames): return []
    if (np.diff(frames)<=0).any(): raise ValueError('One physical track needs unique increasing frame IDs')
    bins = (frames-frames[0])//length
    _,start = np.unique(bins,return_index=True)
    pieces = [a for a in np.split(np.arange(len(frames)),start[1:])]
    tail_span = int(frames[-1]-(frames[0]+bins[-1]*length)+1)
    if len(pieces)>1 and 2*tail_span<length:
        pieces[-2] = np.r_[pieces[-2],pieces[-1]];pieces.pop()
    return pieces


def aggregate(values):
    x = np.asarray(values,dtype=np.float64); n = len(x); k = max(1,(n+3)//4)
    return dict(MEAN=float(x.mean()),MEDIAN=float(np.median(x)),
        TOP25=float(np.partition(x,n-k)[n-k:].mean()),
        LOGIT_MEAN=float(logsumexp(-np.logaddexp(0,-x))-logsumexp(-np.logaddexp(0,x))))


def auc(scores,labels):
    """Same midrank binary AUROC as v2, with fast paths for tiny frames."""
    y = np.asarray(labels,dtype=bool); p = int(y.sum()); n = len(y)-p
    if not p or not n: return None
    from scipy.stats import rankdata
    r = rankdata(np.asarray(scores,dtype=np.float64))
    return float((r[y].sum()-p*(p+1)/2)/(p*n))


def order(scores,ids):
    return np.lexsort((np.asarray(ids),-np.asarray(scores)))


def binary_f1(selected,y):
    y = np.asarray(y,dtype=bool); k = len(selected); p = int(y.sum())
    if not p: return float(not k)
    return 2*int(y[np.asarray(selected,dtype=int)].sum())/(k+p)


def prefix_ceiling(scores,y,ids):
    ranked = order(scores,ids); p = int(np.asarray(y).sum())
    if not p: return 1.,0
    cumulative = np.cumsum(np.asarray(y,dtype=bool)[ranked]); ks = np.arange(1,len(y)+1)
    # Exact rational comparison preserves the smaller-k tie without roundoff.
    best_num,best_den,best_k = 0,1,0
    for k,tp in zip(ks,cumulative):
        num,den = 2*int(tp),int(k)+p
        if num*best_den>best_num*den: best_num,best_den,best_k = num,den,int(k)
    return best_num/best_den,best_k


def flip_labels(y,target,video,query,fnr,fpr):
    y = np.asarray(y,dtype=bool)
    uniform = np.random.RandomState(seed_for(target,video,query)).uniform(size=len(y))
    return np.logical_xor(y,uniform<np.where(y,fnr,fpr))


def bisect_result(scores,y,answers,ids):
    ranked = order(scores,ids)
    boundary,positions,_ = boundary_search(tuple(bool(x) for x in np.asarray(answers)[ranked]),'BISECT_EXACT')
    chosen = ranked[:boundary]
    return dict(questions=len(positions),F1=binary_f1(chosen,y),selected_count=boundary,
        asked_positions=json.dumps(list(positions),separators=(',',':')),
        asked_track_ids=json.dumps([int(ids[ranked[i-1]]) for i in positions],separators=(',',':')),
        answers=json.dumps([bool(answers[ranked[i-1]]) for i in positions],separators=(',',':')))


def file_writer(path,fieldnames):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    stream = path.open('w',newline='');writer = csv.DictWriter(stream,fieldnames=fieldnames,lineterminator='\n')
    writer.writeheader();return stream,writer
