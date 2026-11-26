"""Conditional remaining-pool propensity audit of paid preanswer forecasts."""
import argparse
import ast
from collections import Counter, defaultdict
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path('/data3/liuyeqiang/FlexHook-TTA')
SUP = ROOT/'a100_runs/visual_classical_20261121/supervisor'
OLD_SOURCE = SUP/'a100_fullscope_paid_forecast_audit_20261127.py'
OLD_REG = SUP/'paid_forecast_audit_registration_20261127.json'
OLD_RESULT = SUP/'fullscope_paid_forecast_audit_20261127.json'
OWN = SUP/'paid_forecast_propensity_registration_20261127.json'
OUT = SUP/'paid_forecast_propensity_result_20261127.json'
OLD_SOURCE_SHA = '2db6f667b753864c9d68411e4408613a878b25627c5db20e258139bd8621554d'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, doc):
    assert not path.exists(), path
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2)+'\n')


def enumeration_check():
    p = [F(1,10), F(1,5), F(7,10)]
    g = [F(1,5), F(-2,5), F(4,5)]
    naive = sum(a*b for a,b in zip(p,g))
    exact = sum(p[i]*g[i]/(3*p[i]) for i in range(3))
    assert exact == sum(g)/3 == F(1,5) and naive == F(1,2)
    second = F(0); second_target = F(0); maximum = max(1/(3*x) for x in p)
    for i in range(3):
        remaining = [j for j in range(3) if j!=i]
        total = sum(j+1 for j in remaining)
        pi = {j:F(1,10)+F(4,5)*F(j+1,total) for j in remaining}
        assert sum(pi.values()) == 1
        conditional = sum(pi[j]*g[j]/(2*pi[j]) for j in remaining)
        assert conditional == sum(g[j] for j in remaining)/2
        second += p[i]*conditional
        second_target += p[i]*sum(g[j] for j in remaining)/2
        maximum = max(maximum, *(1/(2*pi[j]) for j in remaining))
    assert second == second_target and (exact+second)/2 == F(1,8)
    assert maximum <= 5
    return dict(status='PASS_EXACT_RATIONAL_INITIAL_AND_ADAPTIVE_REMAINING_POOL_ENUMERATION',
        first_naive_expectation=float(naive), first_uniform_target=float(exact),
        two_step_HT_expectation=float((exact+second)/2),
        maximum_toy_weight=float(maximum), samples_or_training_seeds=0)


def prepare():
    assert sha(OLD_SOURCE) == OLD_SOURCE_SHA
    original = read(OLD_REG); result = read(OLD_RESULT)
    assert original['source_sha256'] == result['source_sha256'] == OLD_SOURCE_SHA
    assert result['registration_sha256'] == sha(OLD_REG)
    assert result['paid_primary_events'] == 55019 and result['GT_or_HOTA_reads'] == 0
    assert result['preanswer_forecasts_equal_preanswer_selection_records']
    write(OWN, dict(status='REGISTERED_BEFORE_PROPENSITY_ADJUSTED_REAL_VALUE_ANALYSIS',
        source_sha256=sha(__file__), prior_source_sha256=OLD_SOURCE_SHA,
        prior_registration_sha256=sha(OLD_REG), prior_result_sha256=sha(OLD_RESULT),
        mathematical_enumeration=enumeration_check(),
        estimand='At each within-query draw, the uniform mean over then-remaining frame/ID actions, at the model state frozen before that draw/stage.',
        weight='1 / ((query_initial_action_count - already_selected_actions) * logged_selection_probability)',
        support_floor=.2, maximum_weight_bound=5, clipping_or_normalization_for_HT=False,
        reported_normalized_estimate='Self-normalized ratios are supplementary and not conditionally unbiased.',
        shared_random_stream_caveat='All case RNGs reuse seed0. No pooled independent-draw confidence interval, new seeds, or independence assumption across videos is used.',
        exclusions=['Not LURE and not a claim that clipped training weights are unbiased.',
            'Not final-model loss, HOTA, true semantic error, or learning information gain.',
            'Only actually requested primary answers and preanswer forecasts; auxiliary answers excluded.',
            'Conditional unbiasedness requires faithful logged propensities, positive support, and fixed potential teacher responses or stable conditional teacher response means.',
            'The target remaining pool and learned models change with history; this is not one fixed full-pool risk.',
            'One realized fixed-seed diagnostic does not prove that a sampling policy is better.'],
        new_models_GT_HOTA_unrequested_answers_or_sampling_operations=0, time_unix=time.time()))


def summarize(rows):
    a = np.asarray([[z[k] for z in rows] for k in ('y','s','r','v','w')], dtype=np.float64)
    y,s,r,v,w = a
    actual = (y-s)**2; re = (y-r)**2; disagreement = (r-s)**2
    cross = 2*(y-r)*(r-s); proxy = v+disagreement
    assert np.max(np.abs(actual-re-disagreement-cross)) < 2e-14
    assert np.max(np.abs(actual-proxy-(re-v+cross))) < 2e-14
    metrics = dict(student_teacher_MSE=actual,observer_teacher_MSE=re,
        R_residual_proxy=v,observed_R_residual=re,R_S_disagreement_squared=disagreement,
        observed_cross_term=cross,risk_proxy=proxy,actual_minus_proxy=actual-proxy)
    means = {name:dict(raw=float(z.mean()),conditional_remaining_pool_HT=float((w*z).mean()),
                      self_normalized_supplement=float(np.dot(w,z)/w.sum())) for name,z in metrics.items()}
    return dict(events=len(rows),mean_weight=float(w.mean()),maximum_weight=float(w.max()),
        minimum_weight=float(w.min()),effective_weight_count_descriptive_only=float(w.sum()**2/np.dot(w,w)),
        means=means)


def main():
    c = read(OWN)
    assert c['source_sha256'] == sha(__file__)
    assert c['prior_source_sha256'] == sha(OLD_SOURCE) == OLD_SOURCE_SHA
    assert c['prior_registration_sha256'] == sha(OLD_REG)
    assert c['prior_result_sha256'] == sha(OLD_RESULT)
    original = read(OLD_REG)
    paths = set(original['inputs']) | {str(OWN),str(OLD_REG),str(OLD_RESULT),str(Path(__file__))}
    for p, expected in original['inputs'].items():
        assert sha(p) == expected, p
    allowed = {name for p in paths for name in (os.path.abspath(p),os.path.realpath(p))}
    rejected = []
    tree = ast.parse(OLD_SOURCE.read_text())
    prior_main = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    guard = next(n for n in prior_main.body if isinstance(n,ast.FunctionDef) and n.name=='guard')
    ns = dict(os=os,sys=sys,Path=Path,ROOT=ROOT,allowed=allowed,
              output_names={str(OUT),str(OUT.resolve())},rejected=rejected)
    exec(compile(ast.Module(body=[guard],type_ignores=[]),str(OLD_SOURCE),'exec'),ns)
    sys.addaudithook(ns['guard'])
    grouped = defaultdict(list); stage_rows = defaultdict(list); query_rows = defaultdict(list)
    for case in original['units']:
        n = len(np.load(case['observations'],mmap_mode='r',allow_pickle=False))
        queries = read(case['queries'])['queries']; previous = Counter(); seen = set()
        for draw in read(case['draws'])['draws']:
            uid = int(draw['uid']); q,oi = divmod(uid,n)
            assert uid not in seen and 0<=q<len(queries) and draw['query']==q
            seen.add(uid)
            remaining = n-previous[q]; pi = float(draw['proposal_probability'])
            assert remaining>0 and np.isfinite(pi) and .2/remaining-1e-14<=pi<=1
            weight = 1/(remaining*pi)
            assert 0<weight<=5+1e-9
            row = dict(y=float(draw['probability']),s=float(draw['student_probability_BEFORE_answer']),
                r=float(draw['observer_mean_BEFORE_answer']),v=float(draw['observer_residual_proxy_BEFORE_answer']),w=weight)
            grouped[case['pair']].append(row);stage_rows[(case['pair'],draw['stage'])].append(row)
            query_rows[(case['pair'],case['video'],q)].append(row);previous[q]+=1
        assert all(previous[q]==int(query['logical_budget']) for q,query in enumerate(queries))
    assert sum(map(len,grouped.values()))==55019 and len(query_rows)==3506
    for p, expected in original['inputs'].items():
        assert sha(p)==expected,p
    query_macros = defaultdict(list)
    for (pair,video,q),rows in query_rows.items():
        z = summarize(rows)
        query_macros[pair].append({name:values['conditional_remaining_pool_HT'] for name,values in z['means'].items()})
    result = dict(status='COMPLETE_ALL148_CONDITIONAL_REMAINING_POOL_PROPENSITY_FORECAST_AUDIT',
        source_sha256=sha(__file__),registration_sha256=sha(OWN),prior_source_sha256=OLD_SOURCE_SHA,
        video_units=148,events=55019,queries=3506,
        by_setting={pair:summarize(rows) for pair,rows in sorted(grouped.items())},
        by_setting_stage={pair+'/'+str(stage):summarize(rows) for (pair,stage),rows in sorted(stage_rows.items())},
        by_setting_query_macro_HT={pair:{name:float(np.mean([z[name] for z in rows])) for name in rows[0]} for pair,rows in sorted(query_macros.items())},
        all_importance_weights_bounded_by_five_without_clipping=True,
        new_GT_HOTA_model_unrequested_answer_or_sampling_operations=0,rejected_IO=rejected,
        conditional_estimand=c['estimand'],scope=c['exclusions'],shared_random_stream_caveat=c['shared_random_stream_caveat'],
        current_numerical_study_unchanged=True,goal_achieved=False)
    write(OUT,result)
    print(json.dumps(dict(status=result['status'],output=str(OUT),by_setting=result['by_setting'],
        new_GT_HOTA_model_unrequested_answer_or_sampling_operations=0)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args();prepare() if args.prepare else main()
