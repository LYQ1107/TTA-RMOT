"""Post-seal queried-anchor coverage and teacher agreement, never adaptation."""
import json
from collections import Counter
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import metadata,PAIRS,OUT as DIAG
from scripts.a100_visual_classical_canonical_pcd_launch import conditions
from scripts.a100_visual_classical_rgb_parity import source_spec
from scripts.a100_visual_classical_prepare import OUT


def rates(counts):
    def fraction(a,b):return a/b if b else None
    return dict(anchor_membership_agreement=fraction(counts['TP']+counts['TN'],counts['asked']),
        teacher_yes_precision=fraction(counts['TP'],counts['TP']+counts['FP']),
        positive_anchor_recall=fraction(counts['TP'],counts['TP']+counts['FN']),
        positive_anchor_rate=fraction(counts['TP']+counts['FN'],counts['asked']))


def shirt_wording(text):
    text=re.sub(r'\b(?:tshirt|t-shirt|shirt)\b','shirt',text.lower())
    return ' '.join(text.replace('dacning','dancing').split())


def main():
    # The already completed sources may be inspected only after every raw and
    # quantity output of the ongoing canonical study is immutable.
    for condition in conditions():
        root=Path(condition['folder'])
        for p in (root/'completion.json',root/'countmatched/completion.json'):
            assert p.is_file(),str(p)
            receipt=json.loads(p.read_text());assert receipt['GT_open_attempts']==0
    pairs={}
    for pair in ('C4','C1'):
        m=metadata(PAIRS[pair][1]);source=source_spec(pair)[0];confusion=Counter();query_counts=Counter();rows=[];types={}
        probabilities={k:[] for k in ('TP','FP','TN','FN')};bins=[Counter() for _ in range(10)]
        fp_context=Counter();fp_pairs=Counter()
        for video,bank in sorted(m['videos'].items()):
            obs=np.asarray(bank['observations']);queries=[e for e in m['episodes'] if e['video_id']==video]
            truth=np.load(DIAG/'offline_GT'/pair/video/'labels.npy')>=0
            assert truth.shape==(len(queries),len(obs))
            folder=source/'source'/video
            requests=json.loads((folder/'questions_before_answers.json').read_text())['requests']
            answers=json.loads((folder/'answers.json').read_text())['answers'];assert len(requests)==len(answers)
            local=[Counter() for _ in queries]
            for request,answer in zip(requests,answers):
                qi,ti=int(answer['local_query']),int(answer['local_track'])
                assert (qi,ti)==(int(request['local_query']),int(request['local_track']))
                ids=np.flatnonzero((obs[:,0]==int(request['frame_id']))&(obs[:,1]==int(request['track_id'])))
                assert len(ids)==1
                positive=bool(truth[qi,ids[0]]);yes=answer['yes_probability']>=.5
                key=('TP' if positive else 'FP') if yes else ('FN' if positive else 'TN')
                confusion.update({key:1,'asked':1});local[qi].update({key:1,'asked':1,'yes':int(yes),'positive_asked_anchor':int(positive)})
                probability=float(answer['yes_probability']);probabilities[key].append(probability)
                bins[min(9,int(probability*10))].update(count=1,positive_membership=int(positive),probability_sum=probability)
                if key=='FP':
                    others=np.flatnonzero(truth[:,ids[0]])
                    if not len(others):fp_context['no_query_positive_membership_at_anchor']+=1
                    elif any(shirt_wording(queries[j]['query_text'])==shirt_wording(queries[qi]['query_text']) for j in others):
                        fp_context['shirt_wording_variant_of_anchor_positive_query']+=1
                    else:fp_context['different_from_anchor_positive_queries']+=1
                    for j in others:fp_pairs[queries[qi]['query_text'],queries[j]['query_text']]+=1
            for qi,(query,record) in enumerate(zip(queries,local)):
                recoverable=bool(truth[qi].any());asked=record['asked'];yes=record['yes'];covered=record['positive_asked_anchor']>0
                query_counts.update(queries=1,recoverable_positive_queries=int(recoverable),
                    no_questions=int(not asked),no_teacher_yes=int(not yes),
                    recoverable_no_questions=int(recoverable and not asked),
                    recoverable_without_positive_asked_anchor=int(recoverable and not covered),
                    recoverable_no_yes=int(recoverable and not yes),
                    recoverable_no_yes_and_no_positive_anchor=int(recoverable and not yes and not covered),
                    recoverable_no_yes_despite_positive_anchor=int(recoverable and not yes and covered),
                    no_positive_candidate_with_teacher_yes=int(not recoverable and yes>0))
                from flex_uda.active.acquisition import expression_types
                kind='+'.join(sorted(expression_types(query['query_text']))) or 'OTHER'
                types.setdefault(kind,Counter()).update(record)
                rows.append(dict(video=video,expression_id=query['expression_id'],query_text=query['query_text'],
                    recoverable_positive_query=recoverable,positive_asked_anchor_coverage=covered,**dict(record)))
        pairs[pair]=dict(anchor_confusion=dict(confusion),anchor_rates=rates(confusion),query_coverage=dict(query_counts),
            query_types={k:dict(counts=dict(v),rates=rates(v)) for k,v in types.items()},queries=rows,
            probability_quantiles={k:dict(count=len(v),quantiles_0_10_25_50_75_90_100=np.quantile(v,[0,.1,.25,.5,.75,.9,1]).tolist() if v else None) for k,v in probabilities.items()},
            fixed_probability_bins=[dict(lower=i/10,upper=(i+1)/10,count=b['count'],
                mean_teacher_probability=b['probability_sum']/b['count'] if b['count'] else None,
                observed_positive_membership=b['positive_membership']/b['count'] if b['count'] else None) for i,b in enumerate(bins)],
            wrong_yes_anchor_context=dict(fp_context),
            common_wrong_yes_and_anchor_positive_query_pairs=[dict(asked=a,anchor_positive_query=b,count=count) for (a,b),count in fp_pairs.most_common(20)])
    result=dict(status='POSTSEAL_FRESH_TEACHER_ANCHOR_DIAGNOSIS',pairs=pairs,
        source_teacher_budgets=dict(C4=1160,C1=1103),new_teacher_or_encoder_calls=0,postseal_GT_only=True,
        no_automatic_parameter_or_acquisition_changes=True,
        teacher_yes_rule='Probability>=.5, matching hard-KD label convention; equality does not mean certain positive.',
        probability_bins_are_diagnosis_not_GT_threshold_selection=True,
        interpretation=['GT agreement means IoU candidate membership at the queried anchor, not pure semantic teacher accuracy.',
            'Queried nodes are a selected subset; rates cannot be extrapolated to all candidates.',
            'Recoverable presence means at least one positive candidate row, not complete GT coverage.',
            'Some language attributes require temporal or scene context unavailable in the queried crop.',
            'All data are development data; this is diagnosis, not a new policy or HOTA effect.'])
    atomic_json(OUT/'analysis/fresh_teacher_anchor_diagnosis.json',result)
    public=dict(result,pairs={k:{name:value for name,value in row.items() if name!='queries'} for k,row in pairs.items()},
        detailed_query_records_local_only=True)
    atomic_json(ROOT/'docs/a100_visual_classical_20261121/fresh_teacher_anchor_diagnosis.json',public)
    print(json.dumps({k:{n:v[n] for n in ('anchor_confusion','anchor_rates','query_coverage')} for k,v in pairs.items()}),flush=True)


if __name__=='__main__':main()
