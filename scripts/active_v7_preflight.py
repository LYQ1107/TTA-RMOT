"""CPU replay all first-video sentences of each frozen v7 direction."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.active.execution import authorize_active
from flex_uda.active.v7 import BeliefStateV7
from flex_uda.active.v7_frozen import FrozenFrameInputs
from flex_uda.presence.runtime import runtime_guard,target_metadata,sealed_margins
from scripts.active_run import PAIRS


def main():
    execution,_=authorize_active();opened=runtime_guard();records=[]
    for pair,(model,target) in PAIRS.items():
        meta=target_metadata(target);margins,seal=sealed_margins(target,model)
        video=sorted(meta['videos'])[0];bank=meta['videos'][video]
        queries=[q for q in meta['episodes'] if q['video_id']==video]
        fc=FrozenFrameInputs(execution.stage_a,target,model,video,bank,queries,margins)
        fractions,probability,mask=fc.decode(1.,0.)
        reference=execution.stage_a/target/model/'PRES_TOPK/predictions_seal.json'
        entries={(e['video'],e['expression']):e for e in json.loads(reference.read_text())['episodes']}
        for qi,q in enumerate(queries):
            entry=entries[video,q['expression_id']]
            assert file_sha(entry['prediction_path'])==entry['prediction_sha256']
            assert ''.join(line+'\n' for line,keep in zip(bank['lines'],mask[qi]) if keep)==Path(entry['prediction_path']).read_text()
        state=BeliefStateV7(fc.track_means(),fractions,probability.max(1),track_lengths=[len(fc.by_track[t]) for t in fc.tracks])
        assert (state.pseudo_labels()==-1).all() and ((state.prior>=-2)&(state.prior<=2)).all()
        records.append(dict(pair=pair,video=video,expressions=len(queries),all_PRES_TOPK_rows_exact=True,
            margin_sha256=seal['sha256'],model_or_PCD_loaded=False,decode_CPU_seconds=fc.decode_seconds))
    assert not opened
    atomic_json(execution.public/'frozen_input_preflight.json',dict(status='ALL_SIX_REAL_FIRST_VIDEO_FROZEN_INPUTS_PASSED',
        records=records,GT_open_attempts=0,GPU_or_model_or_teacher_calls=0))
    print(json.dumps(dict(status='ALL_SIX_REAL_FIRST_VIDEO_FROZEN_INPUTS_PASSED',records=records)),flush=True)


if __name__=='__main__':main()
