"""Independently recompute every propensity audit aggregate with math.fsum."""
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path('/data3/liuyeqiang/FlexHook-TTA')
SUP = ROOT/'a100_runs/visual_classical_20261121/supervisor'
OWN = SUP/'paid_forecast_propensity_registration_20261127.json'
RESULT = SUP/'paid_forecast_propensity_result_20261127.json'
OLD_OWN = SUP/'paid_forecast_audit_registration_20261127.json'
OLD_SOURCE = SUP/'a100_fullscope_paid_forecast_audit_20261127.py'
SOURCE = SUP/'paid_forecast_propensity_check_20261127.py'
OUT = SUP/'paid_forecast_propensity_independent_reproduction_20261127.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def main():
    contract, result, original = read(OWN), read(RESULT), read(OLD_OWN)
    assert contract['source_sha256'] == sha(SOURCE) == result['source_sha256']
    assert result['registration_sha256'] == sha(OWN)
    assert contract['prior_registration_sha256'] == sha(OLD_OWN)
    assert sha(OLD_SOURCE) == '2db6f667b753864c9d68411e4408613a878b25627c5db20e258139bd8621554d'
    inputs = dict(original['inputs'])
    for p in (Path(__file__),OWN,RESULT,OLD_OWN,OLD_SOURCE,SOURCE):
        inputs[str(p)] = sha(p)
    for p, expected in inputs.items():
        assert sha(p) == expected
    allowed = {n for p in inputs for n in (os.path.abspath(p),os.path.realpath(p))}
    rejected = []
    tree = ast.parse(OLD_SOURCE.read_text())
    outer = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    fn = next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=='guard')
    ns = dict(os=os,sys=sys,Path=Path,ROOT=ROOT,allowed=allowed,
              output_names={str(OUT),str(OUT.resolve())},rejected=rejected)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(OLD_SOURCE),'exec'),ns)
    sys.addaudithook(ns['guard'])
    groups = defaultdict(list); stages = defaultdict(list); queries_group = defaultdict(list)
    for unit in original['units']:
        doc = read(unit['queries']); n = int(doc['observations'])
        seen = set(); counts = Counter()
        for z in read(unit['draws'])['draws']:
            uid = int(z['uid']); query = uid//n
            assert uid not in seen and z['query'] == query
            seen.add(uid)
            weight = 1./((n-counts[query])*z['proposal_probability']); counts[query]+=1
            assert 0<weight<=5+1e-9
            y,s,r,v = [float(z[k]) for k in ('probability','student_probability_BEFORE_answer',
                'observer_mean_BEFORE_answer','observer_residual_proxy_BEFORE_answer')]
            se = (s-y)*(s-y); re = (r-y)*(r-y); diff = (r-s)*(r-s)
            m = dict(student_teacher_MSE=se,observer_teacher_MSE=re,R_residual_proxy=v,
                observed_R_residual=re,R_S_disagreement_squared=diff,
                observed_cross_term=se-re-diff,risk_proxy=v+diff,actual_minus_proxy=se-v-diff)
            assert abs(m['observed_cross_term']-2*(y-r)*(r-s))<2e-14
            row = (weight,m)
            groups[unit['pair']].append(row);stages[unit['pair']+'/'+str(z['stage'])].append(row)
            queries_group[(unit['pair'],unit['video'],query)].append(row)
        assert all(counts[q]==int(doc['queries'][q]['logical_budget']) for q in range(len(doc['queries'])))
    errors=[]; checks=Counter()
    def compare(a,b,kind):
        error=abs(a-b);errors.append(error);checks[kind]+=1
        assert error<1e-12,(kind,a,b,error)
    def aggregate(rows, saved):
        count=len(rows);weights=[w for w,_ in rows];total=math.fsum(weights)
        for name in rows[0][1]:
            raw=math.fsum(m[name] for _,m in rows)/count
            ht=math.fsum(w*m[name] for w,m in rows)/count
            normalized=math.fsum(w*m[name] for w,m in rows)/total
            for key,value in [('raw',raw),('conditional_remaining_pool_HT',ht),('self_normalized_supplement',normalized)]:
                compare(value,saved['means'][name][key],'metric_means')
        compare(count,saved['events'],'event_counts')
        for key,value in [('mean_weight',total/count),('maximum_weight',max(weights)),('minimum_weight',min(weights)),
                ('effective_weight_count_descriptive_only',total*total/math.fsum(w*w for w in weights))]:
            compare(value,saved[key],'weight_summaries')
    for key,rows in groups.items():aggregate(rows,result['by_setting'][key])
    for key,rows in stages.items():aggregate(rows,result['by_setting_stage'][key])
    macros=defaultdict(list)
    for (pair,video,q),rows in queries_group.items():
        macros[pair].append({name:math.fsum(w*m[name] for w,m in rows)/len(rows) for name in rows[0][1]})
    for pair,rows in macros.items():
        for name in rows[0]:
            compare(math.fsum(r[name] for r in rows)/len(rows),result['by_setting_query_macro_HT'][pair][name],'query_macro_means')
    assert sum(map(len,groups.values()))==55019 and len(queries_group)==3506
    assert checks['metric_means']==720 and checks['query_macro_means']==48
    for p,expected in inputs.items():assert sha(p)==expected
    assert not rejected and not OUT.exists()
    proof=dict(status='PASS_ALL768_MEANS_AND150_COUNTS_WEIGHTS_INDEPENDENT_FSUM_RECOMPUTATION',
        source_sha256=sha(__file__),result_sha256=sha(RESULT),registration_sha256=sha(OWN),
        checks=dict(checks),maximum_absolute_error=max(errors),
        difference_from_primary='Uses query-document action count, scalar Python arithmetic, math.fsum, and cross-term reconstruction instead of NumPy vector means.',
        GT_HOTA_models_unrequested_answers_or_sampling_operations=0,rejected_IO=rejected,
        all_registered_inputs_unchanged=True,goal_achieved=False)
    OUT.write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof))


if __name__=='__main__':
    main()
