"""Fixed list means, whole-video uncertainty and literal preregistered verdicts."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.search_feasibility_common import *

FIELDS = ('m', 'questions_over_log', 'F1', 'count_MAE', 'presence_accuracy', 'precision', 'recall')
SCOPES = ('ALL', 'g=0', 'g>0')


def clean(v):
    if isinstance(v, dict): return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating, float)): return float(v) if np.isfinite(v) else None
    return v


def ci(values):
    values = np.asarray(values); good = values[np.isfinite(values)]
    return dict(CI95=np.percentile(good, [2.5, 97.5]).tolist() if len(good) else None,
                valid_bootstrap_samples=len(good), samples=2000, seed=0, unit='whole video')


def auc_scores(scores, y):
    y = np.asarray(y, dtype=bool); scores = np.asarray(scores, dtype=float)
    p, n = y.sum(), (~y).sum()
    if not p or not n: return None
    return float((rankdata(scores)[y].sum() - p*(p+1)/2) / (p*n))


def auc_histogram(df, weights, videos, unique):
    pos = np.zeros((len(videos), len(unique))); neg = np.zeros_like(pos)
    vid = {v: i for i, v in enumerate(videos)}; col = {s: i for i, s in enumerate(unique)}
    for v, score, label in df[['cluster', 'C_online', 'search_failure']].itertuples(index=False, name=None):
        (pos if label else neg)[vid[v], col[score]] += 1
    p, n = weights @ pos, weights @ neg
    den = p.sum(axis=1) * n.sum(axis=1)
    numerator = (p * (np.cumsum(n, axis=1) - n/2)).sum(axis=1)
    return np.divide(numerator, den, out=np.full(len(weights), np.nan), where=den > 0)


def spearman_samples(df, weights, videos):
    """Exact midrank Spearman of replicated video clusters, with tie weights."""
    d = df.dropna(subset=['C_dense', 'A_GT'])
    if len(d) < 2: return np.full(len(weights), np.nan)
    cs, ais = sorted(d.C_dense.unique()), sorted(d.A_GT.unique())
    cm, am, vm = {x:i for i,x in enumerate(cs)}, {x:i for i,x in enumerate(ais)}, {x:i for i,x in enumerate(videos)}
    vv = np.asarray([vm[x] for x in d.video]); cc = np.asarray([cm[x] for x in d.C_dense]); aa = np.asarray([am[x] for x in d.A_GT])
    ccounts = np.zeros((len(videos),len(cs))); acounts = np.zeros((len(videos),len(ais)))
    np.add.at(ccounts,(vv,cc),1); np.add.at(acounts,(vv,aa),1)
    cf, af = weights@ccounts, weights@acounts
    cr, ar = np.cumsum(cf,axis=1)-cf/2+.5, np.cumsum(af,axis=1)-af/2+.5
    center = (cf.sum(axis=1)+1)/2
    result = np.full(len(weights),np.nan)
    for start in range(0,len(weights),128):
        stop = min(start+128,len(weights)); w = weights[start:stop,vv]
        x = cr[start:stop,cc]-center[start:stop,None]; y = ar[start:stop,aa]-center[start:stop,None]
        den = np.sqrt((w*x*x).sum(axis=1)*(w*y*y).sum(axis=1))
        result[start:stop] = np.divide((w*x*y).sum(axis=1),den,out=np.full(stop-start,np.nan),where=den>0)
    return result


def main():
    require_prereg(); verify_search(); start = time.perf_counter()
    evaluation = json.loads((RUN / 'offline/evaluation.json').read_text())
    checked(evaluation['metrics']); checked(evaluation['conflicts']); checked(evaluation['lists'])
    columns = ['list_id', 'target', 'video', 'model', 'cross_domain', 'verifier', 'method', 'N', 'g', *FIELDS]
    df = pd.read_csv(evaluation['metrics']['path'], usecols=columns, dtype={'video': str})
    conflict = pd.read_csv(evaluation['conflicts']['path'], dtype={'video': str})
    videos = {t: sorted(df.loc[df.target == t, 'video'].unique()) for t in TARGETS}
    draws = {t: np.random.default_rng(0).multinomial(len(v), np.ones(len(v))/len(v), size=2000)
             for t, v in videos.items() if t != 'KITTI-V2'}
    bootstrap = dict(protocol=dict(samples=2000, seed=0, clusters={t: len(v) for t, v in videos.items()},
        resampling='multinomial whole videos, same draws paired across models/verifiers/methods',
        KITTI='point estimates only', combined_H2b_CI='conditional: keep KITTI fixed; resample Dance and Sports independently'),
        summaries=[], comparisons=[], conflicts=[], H2b=[])
    summary = []; indexed = {}; boot_means = {}
    groupcols = ['target', 'model', 'verifier', 'method']
    for key, group in df.groupby(groupcols, sort=True):
        target, model, verifier, method = key
        for scope in SCOPES:
            part = group if scope == 'ALL' else group[group.g == 0] if scope == 'g=0' else group[group.g > 0]
            stats = dict(zip(groupcols, key), scope=scope, cross_domain=int(model != IN_DOMAIN[target]),
                lists=len(part), videos=part.video.nunique())
            for f in FIELDS: stats[f] = float(part[f].mean()) if len(part) else None
            summary.append(stats); indexed[(*key, scope)] = stats
            if target == 'KITTI-V2': continue
            count = part.groupby('video').size().reindex(videos[target], fill_value=0).to_numpy(dtype=float)
            sums = part.groupby('video')[list(FIELDS)].sum().reindex(videos[target], fill_value=0).to_numpy(dtype=float)
            den = draws[target] @ count; numerator = draws[target] @ sums
            means = np.divide(numerator, den[:, None], out=np.full_like(numerator, np.nan), where=den[:, None] > 0)
            boot_means[(*key, scope)] = means
            bootstrap['summaries'].append(dict(zip(groupcols, key), scope=scope, clusters=len(videos[target]),
                metrics={f: ci(means[:, i]) for i, f in enumerate(FIELDS)}))
    csv_rows(PUBLIC / 'summary.csv', summary)
    h1 = []
    for target in TARGETS:
        for model in MODELS:
            for verifier in (*TEACHERS, 'GT_ORACLE'):
                pba = indexed[target, model, verifier, 'PBA_0.2', 'ALL']
                dense = indexed[target, model, verifier, 'DENSE', 'ALL']
                valid = pba['questions_over_log'] <= 2 and pba['F1'] >= .9 * dense['F1']
                row = dict(target=target, model=model, verifier=verifier, cross_domain=int(model != IN_DOMAIN[target]),
                    mean_questions=pba['m'], mean_questions_over_log=pba['questions_over_log'],
                    PBA_F1=pba['F1'], DENSE_F1=dense['F1'],
                    F1_fraction_of_DENSE=pba['F1']/dense['F1'] if dense['F1'] else None,
                    question_gate=pba['questions_over_log'] <= 2, F1_gate=pba['F1'] >= .9*dense['F1'],
                    verdict='二分查找有效' if valid else '二分查找无效')
                h1.append(row)
                if target != 'KITTI-V2':
                    a = boot_means[target, model, verifier, 'PBA_0.2', 'ALL']
                    b = boot_means[target, model, verifier, 'DENSE', 'ALL']
                    bootstrap['comparisons'].append(dict(target=target, model=model, verifier=verifier,
                        comparison='PBA_0.2_F1_minus_0.9_DENSE_F1', delta=pba['F1']-.9*dense['F1'],
                        **ci(a[:, 2]-.9*b[:, 2]), clusters=len(videos[target])))
    csv_rows(PUBLIC / 'H1.csv', h1)
    distributions = []
    for (target, model, verifier), part in conflict.groupby(['target', 'model', 'verifier'], sort=True):
        c = part.C_dense.dropna(); joint = part.dropna(subset=['C_dense', 'A_GT'])
        rho = float(spearmanr(joint.C_dense, joint.A_GT).statistic) if len(joint) >= 2 and joint.C_dense.nunique() > 1 and joint.A_GT.nunique() > 1 else None
        distributions.append(dict(target=target, model=model, verifier=verifier,
            cross_domain=int(model != IN_DOMAIN[target]), lists=len(part), C_dense_defined=len(c),
            C_dense_defined_fraction=len(c)/len(part), C_dense_mean=float(c.mean()) if len(c) else None,
            C_dense_std=float(c.std(ddof=0)) if len(c) else None, C_dense_p50=float(c.median()) if len(c) else None,
            C_dense_p90=float(c.quantile(.9)) if len(c) else None, C_online_mean=float(part.C_online.mean()),
            A_GT_defined=int(part.A_GT.notna().sum()), A_GT_mean=float(part.A_GT.mean()),
            spearman_C_dense_A_GT=rho, spearman_defined_lists=len(joint),
            search_failure_rate=float(part.search_failure.mean()),
            C_online_failure_AUROC=auc_scores(part.C_online, part.search_failure)))
        if target != 'KITTI-V2':
            for field in ('C_dense', 'C_online', 'A_GT'):
                counts = part.groupby('video')[field].count().reindex(videos[target],fill_value=0).to_numpy(dtype=float)
                sums = part.groupby('video')[field].sum().reindex(videos[target],fill_value=0).to_numpy(dtype=float)
                den = draws[target]@counts
                samples = np.divide(draws[target]@sums,den,out=np.full(2000,np.nan),where=den>0)
                bootstrap['conflicts'].append(dict(target=target,model=model,verifier=verifier,metric='mean_'+field,
                    point=float(part[field].mean()),clusters=len(videos[target]),**ci(samples)))
            bootstrap['conflicts'].append(dict(target=target,model=model,verifier=verifier,metric='Spearman_C_dense_A_GT',
                point=rho,clusters=len(videos[target]),**ci(spearman_samples(part,draws[target],videos[target]))))
    csv_rows(PUBLIC / 'conflict_summary.csv', clean(distributions))
    h2a = {}; h2b = {}; h2rows = []
    cross = conflict[conflict.cross_domain == 1].copy()
    cross['cluster'] = cross.target + '/' + cross.video
    for verifier in TEACHERS:
        part = cross[cross.verifier == verifier]; means = {}
        for target in TARGETS:
            d = part[part.target == target]; values = d.C_dense.dropna(); means[target] = float(values.mean()) if len(values) else None
            if target != 'KITTI-V2':
                counts = d.groupby('video').C_dense.count().reindex(videos[target], fill_value=0).to_numpy(dtype=float)
                sums = d.groupby('video').C_dense.sum().reindex(videos[target], fill_value=0).to_numpy(dtype=float)
                den = draws[target] @ counts
                samples = np.divide(draws[target] @ sums, den, out=np.full(2000, np.nan), where=den > 0)
                bootstrap['conflicts'].append(dict(target=target, model='CROSS_POOLED', verifier=verifier, metric='cross_domain_mean_C_dense',
                    point=means[target], clusters=len(videos[target]), **ci(samples)))
        ok = all(x is not None for x in means.values()) and means['Dance'] < means['KITTI-V2'] and means['Dance'] < means['SportsMOT']
        h2a[verifier] = dict(means=means, verdict='冲突能区分排序可信度' if ok else '不能区分', passed=ok)
        h2b[verifier] = {}
        for target in ('ALL', *TARGETS):
            d = part if target == 'ALL' else part[part.target == target]
            auc = auc_scores(d.C_online, d.search_failure)
            row = dict(verifier=verifier, target=target, lists_times_cross_rankers=len(d),
                failures=int(d.search_failure.sum()), failure_rate=float(d.search_failure.mean()), AUROC=auc)
            if target != 'KITTI-V2':
                clusters = sorted(d.cluster.unique()); weights = np.ones((2000, len(clusters)))
                # A joint cross-target statistic needs independent draws, not two reset copies of one RNG.
                joint_rng = np.random.default_rng(0)
                for t in ('SportsMOT', 'Dance'):
                    if target not in ('ALL', t): continue
                    joint_draw = joint_rng.multinomial(len(videos[t]),np.ones(len(videos[t]))/len(videos[t]),size=2000) if target=='ALL' else draws[t]
                    for j, v in enumerate(videos[t]): weights[:, clusters.index(t+'/'+v)] = joint_draw[:, j]
                samples = auc_histogram(d, weights, clusters, sorted(d.C_online.unique()))
                row.update(ci(samples), clusters_resampled=sum(len(videos[t]) for t in ('SportsMOT', 'Dance') if target in ('ALL', t)),
                           KITTI_fixed=target == 'ALL')
                bootstrap['H2b'].append(row.copy())
            h2b[verifier][target] = row; h2rows.append({k: v for k, v in row.items() if k not in ('CI95', 'valid_bootstrap_samples', 'samples', 'seed', 'unit', 'clusters_resampled', 'KITTI_fixed')})
        auc = h2b[verifier]['ALL']['AUROC']
        h2b[verifier]['passed'] = auc is not None and auc >= .65
        h2b[verifier]['verdict'] = ('在线冲突可作为自动开关' if auc >= .65 else '不可作为') if auc is not None else '规则未覆盖：搜索失败只有一类，AUROC未定义'
    csv_rows(PUBLIC / 'H2b.csv', h2rows)
    joint = {}
    for verifier in TEACHERS:
        dance = [r for r in h1 if r['target'] == 'Dance' and r['cross_domain'] and r['verifier'] == verifier]
        good = all(r['verdict'] == '二分查找有效' for r in dance)
        passed = good and h2a[verifier]['passed'] and h2b[verifier]['passed']
        joint[verifier] = dict(Dance_both_H1=good, H2a=h2a[verifier]['passed'], H2b=h2b[verifier]['passed'],
            verdict='框架可行，进入原型实现' if passed else '二分查找的前提不成立' if not good else '部分可行')
    overall = ('框架可行，进入原型实现' if any(r['verdict'] == '框架可行，进入原型实现' for r in joint.values()) else
               '二分查找的前提不成立' if not any(r['Dance_both_H1'] for r in joint.values()) else '部分可行')
    # GT-free structural checks; these do not change any gate or add any algorithm.
    uniform = df[df.method.str.match(r'^PBA_0\.')]
    exhaustion = [dict(target=t, model=m, verifier=v, method=k, runs=len(d),
        all_questions_asked=int((d.m == d.N).sum()), fraction_all_questions=float((d.m == d.N).mean()))
        for (t, m, v, k), d in uniform.groupby(['target', 'model', 'verifier', 'method'])]
    pivot = df.pivot(index=['list_id', 'model', 'verifier'], columns='method', values='F1')
    same = dict(PBA_epsilon_F1_exact_equal=bool((pivot['PBA_0.1'] == pivot['PBA_0.2']).all() and (pivot['PBA_0.2'] == pivot['PBA_0.3']).all()),
                PBA_random_budget_F1_exact_equal=bool((pivot['PBA_0.2'] == pivot['RANDOM_SAME_BUDGET']).all()))
    dump(PUBLIC / 'structural_diagnostics.json', dict(uniform_PBA_exhaustion=exhaustion, **same,
        proof='A neighboring boundary differs at only one answer, so its posterior/MAP ratio is >=epsilon/(1-epsilon). Thus max P(b)<=1-epsilon; finite N>=2 makes it strictly <0.9 for epsilon>=0.1. Uniform PBA cannot stop before exhausting all distinct candidates.'))
    verdict = dict(overall=overall, H1=h1, H2a=h2a, H2b=h2b, per_teacher_joint=joint,
                   no_method_implementation_started=True, teacher_calls=0, model_calls=0, GPU_hours=0)
    dump(PUBLIC / 'verdicts.json', clean(verdict)); dump(PUBLIC / 'bootstrap.json', clean(bootstrap))
    dump(RUN / 'offline/summary_completion.json', dict(status='ALL_FIXED_VARIANTS_SUMMARIZED',
        summary_rows=len(summary), conflict_rows=len(conflict), H1_rows=len(h1), all_run_rows=len(df),
        CPU_wall_seconds=time.perf_counter()-start, teacher_calls=0, model_calls=0, GPU_hours=0))
    print(json.dumps(dict(event='SEARCH_FEASIBILITY_VERDICTS', overall=overall, H2a=h2a,
                         H2b={v: h2b[v]['ALL']['AUROC'] for v in TEACHERS}, structural=same)), flush=True)


if __name__ == '__main__': main()
