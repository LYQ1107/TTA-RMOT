"""Seal gates, all target decoders and source SA before independent GT use."""
import json
from collections import defaultdict
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.gated_common import *
from flex_uda.data.manifests import read_jsonl

def gate_bank():
    receipt=json.loads((RUN/'teacher/receipt.json').read_text());plan=json.loads((RUN/'teacher_plan.json').read_text())
    if receipt['status']!='ALL_T1_OUTPUTS_SEALED' or receipt['GT_open_attempts']:raise ValueError('finish all GT-free teacher calls first')
    if file_sha(receipt['outputs_path'])!=receipt['outputs_sha256']:raise ValueError('teacher output changed')
    rows=read_jsonl(receipt['outputs_path']);by=defaultdict(dict)
    for r in rows:
        k=r['dataset'],r['video_id'],r['gate_mode'],r['frame_id']
        if (r['track_id'],r['query_id']) in by[k]:raise ValueError('duplicate teacher judgement')
        by[k][r['track_id'],r['query_id']]=r['yes_probability']
    probs=defaultdict(dict);queries={}
    for f in read_jsonl(plan['manifest_path']):
        base=f['dataset'],f['video_id'],f['gate_mode'];k=(*base,f['frame_id']);qs=f['queries'];queries[base]=qs
        expected={(c['track_id'],q['query_id']) for c in f['candidates'] for q in qs}
        if set(by[k])!=expected:raise ValueError('incomplete gate frame '+str(k))
        probs[base][f['frame_id']]=np.array([[by[k][c['track_id'],q['query_id']] for q in qs] for c in f['candidates']])
    result={};gate_rows=[];E_rows=[]
    for k,p in sorted(probs.items()):
        gate,detail=gate_and_exclusivity(p,len(queries[k]));dataset,video,mode=k
        result[k]=dict(present=gate,queries=queries[k],**detail)
        for j,q in enumerate(queries[k]):gate_rows.append(dict(dataset=dataset,video=video,gate_mode=mode,expression=q['query_id'],
            predicted_present=bool(gate[j]),votes=detail['votes'][j],distinct_vote_frames=detail['distinct_frames'][j]))
        if mode=='normal':E_rows.append(dict(dataset=dataset,video=video,positive_candidates=detail['positive_candidates'],
            multiple_candidates=detail['multiple_candidates'],E=detail['E'],zero_denominator=detail['zero_denominator'],
            selected_decoder='EMPTY_NO_TEACHER_EVIDENCE' if detail['E'] is None else 'TG_X0' if detail['E']<=.5 else 'NATIVE'))
    v2.write_csv(PUBLIC/'gates.csv',gate_rows);v2.write_csv(PUBLIC/'exclusivity.csv',E_rows)
    atomic_json(RUN/'gates_seal.json',dict(status='GATES_SEALED_BEFORE_GT',gate_path=str(PUBLIC/'gates.csv'),
        gates_sha256=file_sha(PUBLIC/'gates.csv'),exclusivity_path=str(PUBLIC/'exclusivity.csv'),exclusivity_sha256=file_sha(PUBLIC/'exclusivity.csv'),
        teacher_output_path=receipt['outputs_path'],teacher_output_sha256=receipt['outputs_sha256'],gate_rows=len(gate_rows),GT_open_attempts=0))
    return result

def generate():
    setup();require_plan();attempts=deny_annotations();started=time.perf_counter();meta,arrays,_=inputs();gates=gate_bank()
    lines=v2.candidate_lines(meta);methods=[];gmm=[];statistics={}
    sa_margins=('FROZEN','T2V_D1_s17','T2V_D1_s23','T2V_D1_s42')
    for m in MARGINS:
        array=arrays[m];stats=global_moments(meta,array);statistics[m]=stats
        decoders=list(GENERAL)+(['SA_DECODE'] if m in sa_margins else [])
        if m=='FROZEN':decoders+=['TG_ONLY_FROZEN_X0','TG_ONLY_NATIVE']
        masks={d:np.zeros(len(array),bool) for d in decoders}
        for video,queries,tracks,frames,ids,unique,inverse in cs.video_banks(meta):
            matrix=np.stack([array[q['start']:q['stop']] for q in queries]);normal=gates['Dance',video,'normal'];early=gates['Dance',video,'early']
            if [q['expression_id'] for q in queries]!=[q['query_id'] for q in normal['queries']]:raise ValueError('query order mismatch')
            out,detail=gated_masks(matrix,inverse,normal['present'],early['present'],stats,queries);gmm.append(dict(margin=m,video=video,**detail))
            if m in sa_margins:out['SA_DECODE'],_=sa_mask(matrix,inverse,normal['present'],normal['E'])
            if m=='FROZEN':out.update(TG_ONLY_FROZEN_X0=out['TG_X0'],TG_ONLY_NATIVE=(matrix>0)&normal['present'][:,None])
            for d in decoders:
                for j,q in enumerate(queries):masks[d][q['start']:q['stop']]=out[d][j]
        for d,mask in masks.items():
            name=m+'__'+d;seal=v2.write_selection(name,mask,meta,lines,protocol='FROZEN_T1_TEST_VIDEO_GATED_DIAGNOSTIC' if d.startswith('TG') or d=='SA_DECODE' else 'GT_FREE_FIXED_SCORE_DECODING')
            p=RUN/'masks'/(name+'.npy');p.parent.mkdir(exist_ok=True);np.save(p,mask)
            methods.append(dict(method=name,margin=m,decoder=d,rows=seal['prediction_rows'],mask_path=str(p),mask_sha256=file_sha(p),
                GT_open_attempts=len(attempts),uses_future_frames=d!='X1_C3_CAUSAL',uses_all_test_video_statistics=d in ('G0_GZ_X0','G0_GZ_X1','TG_GZ_X0'),
                gate_mode='early' if d=='TG_EARLY_X1' else 'normal' if d.startswith('TG') or d=='SA_DECODE' else None))
        print(json.dumps(dict(event='GATED_MARGIN_ALL_PREDICTIONS_SEALED',margin=m,sets=len(decoders))),flush=True)
    for m in MARGINS:
        for d in ('X0_C3','X1_C3_CAUSAL'):
            seal=v2.verify_seal(m+'__'+d)
            methods.append(dict(method=m+'__'+d,margin=m,decoder=d,rows=seal['prediction_rows'],mask_path=None,mask_sha256=None,
                GT_open_attempts=0,uses_future_frames=d=='X0_C3',uses_all_test_video_statistics=False,gate_mode=None))
    if len(methods)!=60:raise ValueError('all60 target decoder sets required')
    # Only source geometry/text/scores and teacher outputs; NO source GT.
    source,src_arrays=source_inputs();src_methods=[];output=RUN/'source_boundary/evaluation'
    for m,array in src_arrays.items():
        for d in ('NATIVE','SA_DECODE'):
            entries=[]
            for video,b in sorted(source['videos'].items()):
                qs=sorted([q for q in source['episodes'] if q['video_id']==video],key=lambda q:q['expression_id'])
                matrix=np.stack([array[q['start']:q['stop']] for q in qs]);inverse=np.unique(np.asarray(b['observations'])[:,1],return_inverse=True)[1];gate=gates['KITTI',video,'normal']
                if [q['expression_id'] for q in qs]!=[q['query_id'] for q in gate['queries']]:raise ValueError('source query order mismatch')
                mask=matrix>0 if d=='NATIVE' else sa_mask(matrix,inverse,gate['present'],gate['E'])[0]
                for j,q in enumerate(qs):
                    p=output/(m+'__'+d)/'results'/video/q['expression_id']/'predict.txt';chosen=[b['lines'][i] for i in np.flatnonzero(mask[j])]
                    v2.write_new(p,''.join(line+'\n' for line in chosen));entries.append(dict(video=video,expression=q['expression_id'],prediction_path=str(p),prediction_sha256=file_sha(p),selected_rows=len(chosen)))
            if len(entries)!=862:raise ValueError('source needs862 expressions')
            record=dict(method=m+'__'+d,margin=m,decoder=d,episodes=entries,prediction_rows=sum(e['selected_rows'] for e in entries))
            atomic_json(output/record['method']/'predictions_seal.json',record);src_methods.append(record)
    if attempts:raise RuntimeError('generator opened annotations')
    v2.write_csv(PUBLIC/'gmm_fits.csv',gmm);atomic_json(PUBLIC/'global_query_statistics.json',statistics)
    source_seal=dict(status='ALL4_SOURCE_SETS_SEALED_BEFORE_GT',methods=src_methods,GT_open_attempts=len(attempts))
    atomic_json(RUN/'source_boundary/predictions_seal.json',source_seal)
    atomic_json(RUN/'predictions_seal.json',dict(status='ALL60_DECODER_SETS_SEALED_BEFORE_GT',methods=methods,
        source_seal_sha256=file_sha(RUN/'source_boundary/predictions_seal.json'),gates_seal_sha256=file_sha(RUN/'gates_seal.json'),
        GT_open_attempts=len(attempts),CPU_seconds=time.perf_counter()-started))
    print(json.dumps(dict(event='GATED_ALL_PREDICTIONS_SEALED',target_sets=60,source_sets=4,GT_open_attempts=len(attempts))),flush=True)

if __name__=='__main__':generate()
