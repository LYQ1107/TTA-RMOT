"""Static, GT-free numerical replay of sealed teacher evidence, not RMOT inference."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.presence.runtime import PUBLIC,RUN,target_metadata,sealed_margins,runtime_guard
from flex_uda.active.belief import BeliefState
from flex_uda.active.map_calibration import source_map_fast_step
from scripts.active_run import PAIRS,video_prior
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def main():
    attempts=runtime_guard();rows=[]
    baseline=json.loads((PUBLIC / 'redesign_v5/active/pilot_first_video.json').read_text())
    for pair in ('C1','C3'):
        r=baseline['pilot_snapshots'][pair];model,target=PAIRS[pair];video=r['videos'][0]
        original=RUN / 'redesign_v5/active' / pair / 'FULL/rho_0.3/videos' / video
        meta=target_metadata(target);margin,_=sealed_margins(target,model);bank=meta['videos'][video]
        prior=video_prior(meta,bank,video,model,target,margin)
        state=BeliefState(prior['track_means'],prior['source_expected'],prior['max_presence'],track_lengths=prior['lengths'])
        source_a=state.a.copy();saved=np.load(original / 'belief.npz')
        state.evidence=np.array(saved['evidence']);state.confirmed_present=np.array(saved['confirmed_present'])
        state.absent=np.array(saved['absent'])
        qindex={q['expression_id']:j for j,q in enumerate(prior['queries'])};tindex={t:j for j,t in enumerate(prior['tracks'])}
        questions=json.loads((original / 'questions.json').read_text())
        state.events=[dict(query=qindex[q['query_id']],track=tindex[q['track_id']]) for q in questions]
        evidence=state.evidence.copy()
        result=source_map_fast_step(state,prior['presence_logits'],source_intercepts=source_a)
        if not np.array_equal(evidence,state.evidence):raise ValueError('Teacher evidence changed')
        if not np.isfinite([result['temperature'],result['bias'],result['slope']]).all():raise ValueError('Replay nonfinite')
        rows.append(dict(pair=pair,video=video,result=result,
            original_legacy_slope=float(saved['slope']),original_legacy_temperature=float(saved['temperature']),
            original_legacy_bias=float(saved['bias']),belief_path=str(original / 'belief.npz'),
            belief_sha256=file_sha(original / 'belief.npz'),questions_sha256=file_sha(original / 'questions.json')))
    output=PUBLIC / 'redesign_v5/active_map_v1/cpu_replay.json'
    atomic_json(output,dict(status='FINITE_OR_EXPLICIT_SKIP',rows=rows,GT_open_attempts=len(attempts),
        purpose='fresh source priors plus fixed legacy teacher evidence; not an interactive-policy equivalence or HOTA test',
        new_model_or_teacher_calls=0,new_GPU_hours=0))
    print(json.dumps(dict(path=str(output),statuses=[r['result']['status'] for r in rows])),flush=True)


if __name__=='__main__':main()
