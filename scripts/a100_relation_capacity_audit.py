"""Unlabelled representational audit of the registered bounded relation field.

Odds changes bounded by exp(cap) cannot flip probabilities below
sigmoid(-cap) or above sigmoid(cap). Count this limit and its overlap with
within-view teacher response variation. No GT correctness is inferred.
"""
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import expit
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import metadata,generation_guard
from flex_uda.active.relational_transport import horizontal_direction,MAX_LOGIT_SHIFT
from scripts.a100_six_direction import SEEDS,PUBLIC
from scripts.a100_relation_transport import CORE,PREP

OUT=PREP.parent/'capacity_audit'


def main():
    guard=generation_guard();start=time.perf_counter();registration=json.loads((PUBLIC/'relation_capacity_audit_registration.json').read_text())
    assert registration['source_sha256']==file_sha(Path(__file__))
    threshold=float(expit(MAX_LOGIT_SHIFT));m=metadata('KITTI-V2');result={}
    for pair in ('C2','C5'):
        seeds={}
        for seed in SEEDS:
            count=defaultdict(int)
            for video,bank in sorted(m['videos'].items()):
                queries=[q for q in m['episodes'] if q['video_id']==video];eligible=np.array([horizontal_direction(q['query_text'])!=0 for q in queries])
                root=CORE/'quality'/pair/f'seed{seed}';assert (root/'completion.json').is_file()
                with np.load(root/'videos'/video/'coarse_scores.npz') as a:tracks=a['tracks'];base=a['DS_FULL']
                protected=(base<1-threshold)|(base>threshold);lookup={int(t):i for i,t in enumerate(tracks)}
                obs=np.asarray(bank['observations']);lengths=np.asarray([(obs[:,1]==track).sum() for track in tracks])
                count['eligible_query_track_cells']+=int(eligible.sum()*len(tracks));count['protected_query_track_cells']+=int(protected[eligible].sum())
                count['eligible_candidate_rows']+=int(eligible.sum()*lengths.sum());count['protected_candidate_rows']+=int((protected[eligible]*lengths).sum())
                rows=json.loads((PREP/pair/video/'teacher_rows.json').read_text())['rows'];grouped=defaultdict(list)
                for row in rows:grouped[tuple(row['group'])].append(row)
                changing=set()
                for values in grouped.values():
                    if len({r['frame'] for r in values})>1 and len({r['probability']>.5 for r in values})>1:
                        changing.add((values[0]['query'],values[0]['track']))
                qi={q['expression_id']:i for i,q in enumerate(queries)}
                count['teacher_changing_query_track_cells']+=len(changing)
                count['protected_teacher_changing_cells']+=sum(bool(protected[qi[q],lookup[int(t)]]) for q,t in changing)
            seeds[str(seed)]=dict(count,protected_candidate_row_fraction=count['protected_candidate_rows']/max(count['eligible_candidate_rows'],1))
        result[pair]=seeds
    assert not guard
    report=dict(status='UNLABELLED_BOUNDED_RELATION_CAPACITY_AUDIT_COMPLETE',pairs=result,
        maximum_logit_shift=MAX_LOGIT_SHIFT,protected_below=1-threshold,protected_above=threshold,
        mathematical_scope='For a strict threshold>0.5, these strict-confidence cells cannot change binary selection under any allowed bounded shift. Protected does not mean wrong; teacher changes may be noise.',
        neural_seeds_do_not_repeat_teacher_events_independently=True,manual_GT_open_attempts=0,new_predictions_or_fits_or_GPU_calls=0,
        current_registered450_protocol_unchanged=True,
        next_hypothesis='If a persistent composite unary is overconfident, a small bounded perturbation cannot express a changing predicate. Consider re-estimating a stable nuisance component with known state offsets, rather than merely increasing a cap after HOTA inspection.',
        all_data_development=True,wall_seconds=time.perf_counter()-start,goal_status='ACTIVE')
    atomic_json(OUT/'completion.json',report)
    (PUBLIC/'relation_capacity_audit.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
