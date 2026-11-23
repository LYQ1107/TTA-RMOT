"""Real image-derived inputs and an independent monotone-solver check, no GT."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.optimize import isotonic_regression
from flex_uda.a100.mechanism import generation_guard,metadata,PAIRS
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.active.semantic_colour import appearance_cues,colour_probabilities,pava_fit,appearance_spec
from scripts.a100_visual_classical_rgb_parity import source_spec


def main():
    guard=generation_guard();checks=[]
    for pair in ('C4','C1'):
        m=metadata(PAIRS[pair][1]);video=sorted(m['videos'])[0];folder=source_spec(pair)[0]/'source'/video
        with np.load(folder/'sparse_inputs.npz') as a:colours,tracks=a['colours'],a['tracks']
        obs=np.asarray(m['videos'][video]['observations']);texts=[e['query_text'] for e in m['episodes'] if e['video_id']==video]
        lengths=np.asarray([(obs[:,1]==t).sum() for t in tracks]);cue,available,families,specs=appearance_cues(colours,texts,lengths)
        probability,_,valid=colour_probabilities(colours)
        assert np.isfinite(cue).all() and np.all((cue>=0)&(cue<=1))
        mass_error=float(np.max(np.abs(probability[valid].sum(1)-1)));assert mass_error<1e-12
        answers=np.full(cue.shape,np.nan)
        for r in json.loads((folder/'answers.json').read_text())['answers']:answers[int(r['local_query']),int(r['local_track'])]=r['yes_probability']
        known=available&np.isfinite(answers);x=cue[known];y=np.clip(answers[known],.02,.98);fitted=pava_fit(x,y)
        assert fitted is not None
        order=np.argsort(x,kind='stable');unique,start,count=np.unique(x[order],return_index=True,return_counts=True)
        means=np.add.reduceat(y[order],start)/count;reference=isotonic_regression(means,weights=count).x
        error=float(np.max(np.abs(fitted[1]-reference)));assert error<1e-12
        assert np.array_equal(unique,fitted[0]) and np.all(np.diff(fitted[1])>=0)
        checks.append(dict(pair=pair,video=video,real_teacher_examples=len(y),maximum_colour_mass_error=mass_error,
            independent_SciPy_PAVA_error=error,all_query_support=sum(appearance_spec(e['query_text']) is not None for e in m['episodes']),
            total_queries=len(m['episodes']),first_video_specs=specs))
    assert not guard
    result=dict(status='PASS_REAL_SEMANTIC_CUES_AND_INDEPENDENT_PAVA',checks=checks,GT_open_attempts=0,
        new_teacher_or_encoder_calls=0,new_neural_head_updates=0,
        semantic_implementation_sha256=file_sha(ROOT/'flex_uda/active/semantic_colour.py'))
    atomic_json(ROOT/'docs/a100_visual_classical_20261121/semantic_colour_generation_sanity.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
