"""Read-only teacher-event holdout diagnosis, not a new adaptation method.

All current predictions and teacher attempts are already sealed. Uniform
answers are used only by this separate diagnostic, never by an active model.
This is a same-video transductive DEV footprint, not independent semantic GT.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/data3/liuyeqiang/FlexHook-TTA')
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import numpy as np
import torch
import PIL
from flex_uda.a100.mechanism import generation_guard, KEYS, metadata
from flex_uda.active.joint_observation_inputs import JointObservationVideo, use_verified_local_features
from flex_uda.active.factor_rsi_learning import Inputs
from flex_uda.active.fullscope_visual_runtime import LazyFeatures
from flex_uda.active import evidence_visual_learning as core
from scripts import active_run
from scripts.a100_observation_local_cache import CACHE
from scripts.a100_anchor_observability import OUT as ANCHORS
from scripts.a100_dense_rgb_collect import OUT as RGB
from scripts.a100_dino_fullsix_collect import OUT as DINO, PLAN as DINO_PLAN

SUP = ROOT/'a100_runs/visual_classical_20261121/supervisor'
BASE = ROOT/'a100_runs/visual_classical_20261121/six_direction_20261122/fullscope_visual_six_20261126'
DEST = SUP/'uniform_teacher_holdout_audit_20261127_v3'
OWN = DEST/'registration.json'
OUT = DEST/'result.json'
PLAN_SHA = 'd159ed56af4d614edd68c991ab752f6341074752c648650cd61d336d5a68ba12'
BARRIER_SHA = '2a33eaf9cc1178797bd3c577df74534e6925d4870ca2e15be56a97232120440d'
T0 = '9a13380b4279bc98f4680bd083ff4995fdcfec120cfded459513be90d041a956'
ROLES = ('bootstrap_S', 'bootstrap_R', 'final_S', 'final_R', 'SOFT_KD288', 'SAME_ARCH_KD128')
AUDIT_ACTIVE = True


def runtime_library_roots():
    paths = [ROOT/'.venv/lib', (ROOT/'.venv/lib').resolve(), Path(sys.prefix)/'lib',
             (Path(sys.prefix)/'lib').resolve(), Path(sys.base_prefix)/'lib',
             (Path(sys.base_prefix)/'lib').resolve()]
    paths.extend(Path(module.__file__).resolve().parent.parent for module in (np,torch,PIL))
    return tuple(dict.fromkeys(str(p)+'/' for p in paths))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8*1024*1024):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    assert not path.exists(), path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def text_key(text):
    return ' '.join(text.split()).casefold()


def exclusion_key(uid, obs, queries):
    q, oi = divmod(int(uid), len(obs))
    return text_key(queries[q]['query_text']), int(obs[oi, 0]), int(obs[oi, 1])


def check_contract():
    obs = np.asarray([[10, 7], [20, 8]])
    qs = [{'query_text': 'White shirt'}, {'query_text': ' white  SHIRT '}, {'query_text': 'running'}]
    assert exclusion_key('0', obs, qs) == exclusion_key('2', obs, qs)
    assert exclusion_key('0', obs, qs) != exclusion_key('4', obs, qs)
    assert exclusion_key('0', obs, qs)[1:] == (10, 7)
    return dict(status='PASS_LITERAL_ALIAS_AND_PHYSICAL_AUXILIARY_EXCLUSION_CONTRACT', new_sampling_seeds=0)


def prepare():
    assert not OWN.exists()
    assert sha(BASE/'registration.json') == PLAN_SHA
    assert sha(BASE/'all4588_sealed_before_GT.json') == BARRIER_SHA
    plan = read(BASE/'registration.json'); barrier = read(BASE/'all4588_sealed_before_GT.json')
    assert len(plan['cases']) == 148 and len(barrier['tasks']) == 4588
    assert barrier['GT_open_attempts'] == 0 and barrier['teacher_final_weight_seals'] == 4
    assert len(barrier['all_attempt_teacher_sessions']) == 10
    paths = {Path(__file__), BASE/'registration.json', BASE/'all4588_sealed_before_GT.json', CACHE/'completion.json', DINO_PLAN, ROOT/'configs/a100_paths.json'}
    units = []
    for case in plan['cases']:
        folder = BASE/'units'/case['pair']/case['evaluated_video']
        models = {'bootstrap': str(folder/'primary/bootstrap.pt'), 'final': str(folder/'primary/final.pt'),
                  'SOFT_KD288': str(folder/'ordinary/SOFT_KD288/SOFT_KD.pt'),
                  'SAME_ARCH_KD128': str(folder/'ordinary/SAME_ARCH_KD128/SOFT_KD.pt')}
        unit = dict(pair=case['pair'], target=case['target'], video=case['evaluated_video'], models=models,
                    observations=case['observations'], queries=case['temporal_queries'],
                    auxiliary_keys=str(Path(case['auxiliary_path']).parent/'row_keys.json'),
                    active_draws=str(folder/'primary/draws.json'), uniform_draws=str(folder/'uniform_draws.json'),
                    active_answers=str(folder/'active_answers.jsonl'), uniform_answers=str(folder/'uniform_answers.jsonl'))
        paths.update(map(Path, models.values()))
        paths.update(Path(unit[k]) for k in ('observations','queries','auxiliary_keys','active_draws','uniform_draws','active_answers','uniform_answers'))
        paths.add(folder/'generation_complete.json')
        paths.add(ROOT/'results/runs/multidomain_generality_20261101'/KEYS[case['target']]/'metadata.json')
        bank = metadata(case['target'])['videos'][case['evaluated_video']]
        first_frame = min(int(r[0]) for r in bank['observations'])
        extension = '.jpg' if case['target']=='SportsMOT' else '.png'
        paths.add(Path(bank['image_root'])/f'{first_frame:06d}{extension}')
        assert sha(case['observations']) == case['observations_sha256']
        assert sha(case['temporal_queries']) == case['temporal_queries_sha256']
        units.append(unit)
    for gpu in range(4):
        p = BASE/'supervisor'/f'GPU{gpu}_teacher_final.json'
        assert read(p)['teacher']['backend_unmodified']
        paths.add(p)
    for name, expected in plan['source_sha256'].items():
        p = ROOT/name
        assert sha(p) == expected, name
        paths.add(p)
    inputs = {str(p): sha(p) for p in sorted(paths)}
    write(OWN, dict(status='REGISTERED_BEFORE_UNIFORM_ONLY_TEACHER_VALUES_AND_INFERENCE', source_sha256=sha(__file__),
        parent_registration_sha256=PLAN_SHA, full_seal_sha256=BARRIER_SHA, inputs=inputs, units=units,
        roles=ROLES, CPU_threads=2, inference_batch_size=256, math_contract=check_contract(), runtime_library_roots=runtime_library_roots(),
        exclusion='Exclude every active-training literal-query/frame/track key, coalescing whitespace/case aliases and ignoring view; additionally exclude every physical frame/track used by AUX in this video, regardless of query/view.',
        footprint='Only already requested uniform events not covered by these exclusions. Include all148 units and all3506 queries in support accounting, explicitly recording empty query/unit footprints.',
        measurements='Teacher-probability MSE/soft BCE and query-macro MSE for fixed bootstrap/final S/R and two adequate active-answer KD heads. Bootstrap/final R-risk components and within-query rank correlation on exactly the same retained events.',
        no_target_GT_new_training_teacher_sampling_or_GPU=True, no_RMOT_predictions_or_HOTA=True,
        no_current_algorithm_or_threshold_changes=True, not_an_ablation_or_method_acceptance_screen=True,
        feature_integrity='Use the existing verified read-only local PCD cache and original feature constructors. Fresh hashes cover registered metadata, weights and first-frame resolution images. Geometry reads only the registered first-image header; no new pixel descriptor or teacher forward. Do not claim a new full PCD-tree byte hash.',
        boundaries=['Teacher prediction is not semantic correctness or HOTA; final risk is not a reconstruction of four preanswer stages.',
            'Uniform-only event answers were not supplied to any evaluated active learner; images and other teacher events from the same video were available.',
            'Literal alias and exact auxiliary-observation exclusion do not remove all semantic or temporal dependence.',
            'The uniform footprint is object-balanced rather than frame-uniform; no inverse-propensity or independent-observation confidence interval is claimed.',
            'Bootstrap-to-final change is not an isolated causal RSI effect. Risk diagnostics omit acquisition Jacobian and coverage factors.',
            'No final model, parameter, checkpoint or prediction is selected or changed using these diagnostic answers.'],
        time_unix=time.time(), goal_achieved=False))
    print(json.dumps(dict(status='REGISTERED_UNIFORM_ONLY_EVENT_DIAGNOSTIC', units=len(units), hashed_inputs=len(inputs))), flush=True)


def ranks(x):
    values, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends-counts+ends-1)/2)[inverse]


def summarize(y, p, v, groups):
    assert p.shape == (len(y), len(ROLES)) and v.shape == (len(y), 2)
    out = dict(events=len(y), queries=len(groups), supported_queries=sum(bool(len(g)) for g in groups))
    if not len(y):
        return out
    assert np.isfinite(p).all() and np.isfinite(v).all() and ((p>0)&(p<1)).all()
    out['roles'] = {}
    for j, name in enumerate(ROLES):
        error = (y-p[:,j])**2
        out['roles'][name] = dict(teacher_probability_MSE=float(error.mean()),
            soft_teacher_BCE=float((-y*np.log(p[:,j])-(1-y)*np.log1p(-p[:,j])).mean()),
            supported_query_macro_teacher_MSE=float(np.mean([error[g].mean() for g in groups if len(g)])))
    out['risk'] = {}
    for i, name in enumerate(('bootstrap','final')):
        s, r = p[:,2*i], p[:,2*i+1]
        actual, residual = (y-s)**2, (y-r)**2
        cross = 2*(y-r)*(r-s); proxy = v[:,i]+(r-s)**2
        assert np.max(np.abs(actual-proxy-(residual-v[:,i]+cross))) < 2e-14
        correlations = []
        for g in groups:
            if len(g) >= 3:
                a, b = ranks(proxy[g]), ranks(actual[g])
                if np.std(a)>0 and np.std(b)>0:
                    correlations.append(float(np.corrcoef(a,b)[0,1]))
        out['risk'][name] = dict(proxy_mean=float(proxy.mean()), observed_student_teacher_MSE=float(actual.mean()),
            actual_minus_proxy_mean=float((actual-proxy).mean()), residual_prediction_error_mean=float((residual-v[:,i]).mean()),
            cross_term_mean=float(cross.mean()), within_query_Spearman_supported_queries=len(correlations),
            within_query_Spearman_macro_mean=float(np.mean(correlations)) if correlations else None)
    return out


def main():
    contract = read(OWN)
    assert contract['source_sha256'] == sha(__file__) and not OUT.exists()
    for p, expected in contract['inputs'].items():
        assert sha(p) == expected, p
    torch.set_num_threads(2)
    guard = generation_guard(); opened = set(); rejected = []
    allowed = {os.path.abspath(p) for p in contract['inputs']} | {os.path.abspath(OWN)}
    allowed |= {os.path.realpath(p) for p in allowed}
    roots = [CACHE, ANCHORS, RGB, DINO]
    feature_roots = tuple(str(p.resolve())+'/' for p in roots) + tuple(str(p)+'/' for p in roots)
    library_roots = runtime_library_roots()
    assert contract['runtime_library_roots'] == list(library_roots)
    output_roots = (str(DEST)+'/', str(DEST.resolve())+'/')
    def audit(event,args):
        if not AUDIT_ACTIVE:
            return
        if event in ('socket.connect','socket.getaddrinfo'):
            raise PermissionError('Offline diagnostic forbids network access')
        if event != 'open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        p = os.path.abspath(os.fsdecode(args[0])); mode = args[1] if len(args)>1 else 'r'
        flags = args[2] if len(args)>2 and isinstance(args[2],int) else 0
        writing = (isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT))
        permitted = p.startswith(output_roots) if writing else (p in allowed or p.startswith(feature_roots+library_roots+output_roots) or (p.startswith(str(ROOT)+'/') and p.endswith(('.py','.pyc'))))
        if (Path(p).suffix.lower() in ('.jpg','.png','.jpeg') and p not in allowed) or Path(p).suffix.lower()=='.safetensors' or Path(p).name in ('official_completion.json','predictions_seal.json','quality_results.json'):
            permitted = False
        if not permitted:
            rejected.append(p); raise PermissionError('Unregistered holdout diagnostic IO: '+p)
        if not writing and not p.endswith(('.py','.pyc')) and not p.startswith(library_roots):
            opened.add(p)
    sys.addaudithook(audit)
    use_verified_local_features()
    by_setting = defaultdict(list); unit_rows = []; all_queries = 0; total_uniform = 0; started = time.perf_counter()
    for unit in contract['units']:
        obs = np.load(unit['observations'], allow_pickle=False)
        queries = read(unit['queries'])['queries']; n = len(obs); all_queries += len(queries)
        active = read(unit['active_draws'])['draws']; uniform = read(unit['uniform_draws'])['draws']
        active_uids = {d['uid'] for d in active}; uniform_uids = {d['uid'] for d in uniform}
        assert len(active_uids)==len(active)==len(uniform_uids)==len(uniform)
        train_keys = {exclusion_key(d['uid'],obs,queries) for d in active}
        aux_obs = {(int(r[3]),int(r[2])) for r in read(unit['auxiliary_keys'])['keys'] if r[0]==unit['video']}
        journals = {}
        for name in ('active','uniform'):
            rows = [json.loads(s) for s in Path(unit[name+'_answers']).read_text().splitlines() if s]
            assert len(rows)==len({r['uid'] for r in rows})
            assert all(r['frozen_weight_sha256']==T0 for r in rows)
            journals[name] = {r['uid']:r for r in rows}
        assert set(journals['active'])==active_uids and set(journals['uniform'])==uniform_uids
        for name, draws in (('active',active),('uniform',uniform)):
            assert all(d['probability']==journals[name][d['uid']]['probability'] for d in draws)
        common = active_uids & uniform_uids
        overlap_delta = max((abs(journals['active'][i]['probability']-journals['uniform'][i]['probability']) for i in common),default=0.)
        retained = [d for d in uniform if exclusion_key(d['uid'],obs,queries) not in train_keys and exclusion_key(d['uid'],obs,queries)[1:] not in aux_obs]
        ids = np.asarray([int(d['uid']) for d in retained],dtype=np.int64)
        qi = ids//n; y = np.asarray([d['probability'] for d in retained],dtype=np.float64)
        assert all(int(d['query'])==int(q) for d,q in zip(retained,qi))
        assert ((y>=0)&(y<=1)).all()
        p = np.empty((len(ids),len(ROLES)),dtype=np.float64); v = np.empty((len(ids),2),dtype=np.float64)
        if len(ids):
            context = type('ReadOnlyContext',(),{})()
            context.video = JointObservationVideo(unit['pair'],unit['video']); context.inputs = Inputs(context.video)
            context.n, context.nq = len(context.video.obs), len(context.video.queries)
            assert np.array_equal(context.video.obs,obs) and context.nq==len(queries)
            assert all(q['query_id']==a['expression_id'] and q['query_text']==a['query_text'] for q,a in zip(queries,context.video.queries))
            features = LazyFeatures(context); models = []
            for stage in ('bootstrap','final'):
                saved = torch.load(unit['models'][stage],map_location='cpu',weights_only=False)
                assert saved['registration_sha256']==PLAN_SHA
                for role, outputs in (('solver',1),('observer',2)):
                    state = saved[role]; model = core.make(state['mean'].numpy(),state['std'].numpy(),'cpu',outputs=outputs)
                    model.load_state_dict(state); model.eval(); models.append(model)
            for name, kind, width in (('SOFT_KD288','plain',288),('SAME_ARCH_KD128','visual',128)):
                saved = torch.load(unit['models'][name],map_location='cpu',weights_only=False); state = saved['model']
                assert saved['receipt']['total_steps']>=6000 and saved['receipt']['proper_teacher_excess_CE']<=.005
                model = core.make(state['mean'].numpy(),state['std'].numpy(),'cpu',kind=kind,width=width)
                model.load_state_dict(state); model.eval(); models.append(model)
            with torch.inference_mode():
                for start in range(0,len(ids),256):
                    x = torch.as_tensor(features[ids[start:start+256]],dtype=torch.float32)
                    for j, model in enumerate(models):
                        prediction = model(x).sigmoid().numpy()
                        p[start:start+len(x),j] = prediction[:,0] if prediction.ndim==2 else prediction
                        if j in (1,3):
                            v[start:start+len(x),(j-1)//2] = prediction[:,1]
            del context, features, models, saved
        groups = [np.flatnonzero(qi==q) for q in range(len(queries))]
        path = DEST/unit['pair']/unit['video']/'teacher_event_arrays.npz'
        path.parent.mkdir(parents=True,exist_ok=True); assert not path.exists()
        np.savez_compressed(path,uid=ids,query=qi,y=y,probability=p,residual_proxy=v)
        summary = summarize(y,p,v,groups)
        row = dict(pair=unit['pair'],video=unit['video'],uniform_events=len(uniform),active_uniform_same_uid=len(common),
            shared_uid_teacher_probability_max_abs_difference=overlap_delta,uniform_UID_not_active=len(uniform_uids-active_uids),
            excluded_active_literal_events=sum(exclusion_key(d['uid'],obs,queries) in train_keys for d in uniform),
            excluded_additional_auxiliary_events=len(uniform)-len(retained)-sum(exclusion_key(d['uid'],obs,queries) in train_keys for d in uniform),
            arrays=str(path),arrays_sha256=sha(path),**summary)
        unit_rows.append(row); total_uniform += len(uniform)
        by_setting[unit['pair']].append((y,p,v,qi,len(queries)))
        print(json.dumps(dict(status='UNIFORM_TEACHER_EVENT_HOLDOUT_UNIT_COMPLETE_NO_GT',pair=unit['pair'],video=unit['video'],events=len(ids),queries=len(queries))),flush=True)
    assert len(unit_rows)==148 and all_queries==3506 and total_uniform==55019 and not guard
    combined = {}
    for pair, batches in sorted(by_setting.items()):
        offset = 0; groups = []
        for y,p,v,qi,nq in batches:
            groups.extend(np.flatnonzero(qi==q)+offset for q in range(nq)); offset += len(y)
        combined[pair] = summarize(np.concatenate([b[0] for b in batches]),np.concatenate([b[1] for b in batches]),np.concatenate([b[2] for b in batches]),groups)
    for path, expected in contract['inputs'].items():
        assert sha(path)==expected, path
    write(OUT,dict(status='COMPLETE148_UNIFORM_ONLY_TEACHER_EVENT_HOLDOUT_DIAGNOSIS',registration_sha256=sha(OWN),source_sha256=sha(__file__),
        units=unit_rows,by_setting=combined,all_units=148,all_queries=all_queries,uniform_events=total_uniform,
        retained_teacher_event_labels=sum(r['events'] for r in unit_rows),GT_reads=0,new_training_teacher_calls_sampling_or_GPU=0,
        RMOT_outputs_or_HOTA_reads=0,registered_inputs_unchanged=True,opened_input_paths=sorted(opened),rejected_IO=rejected,
        wall_seconds=time.perf_counter()-started,boundaries=contract['boundaries'],goal_achieved=False))
    print(json.dumps(dict(status='COMPLETE148_UNIFORM_ONLY_TEACHER_EVENT_HOLDOUT_DIAGNOSIS',output=str(OUT),by_setting=combined)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    try:
        prepare() if args.prepare else main()
    finally:
        # The permanent GT hook remains. Only the scoped data allowlist ends
        # with this diagnostic; normal interpreter cleanup is not analysis.
        AUDIT_ACTIVE = False
