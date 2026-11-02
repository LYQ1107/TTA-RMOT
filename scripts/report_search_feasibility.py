"""Publish the complete CPU diagnostic, including all variants and negative results."""
import gzip
import json
from pathlib import Path
import shutil
import sys
import tarfile
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.search_feasibility_common import *


def fmt(x):
    if x is None or isinstance(x, float) and not np.isfinite(x): return '未定义'
    if isinstance(x, bool): return '是' if x else '否'
    if isinstance(x, (float, np.floating)): return f'{x:.6f}'
    return str(x).replace('|', '\\|').replace('\n', ' ')


def table(rows, columns):
    rows = list(rows)
    if not rows: return '无样本；未定义，不补零。\n'
    text = '| ' + ' | '.join(columns) + ' |\n|' + '|'.join('---' for _ in columns) + '|\n'
    return text + ''.join('| ' + ' | '.join(fmt(r.get(k)) for k in columns) + ' |\n' for r in rows)


def main():
    require_prereg(); seal = verify_search()
    validation = json.loads((PUBLIC/'validation.json').read_text())
    if validation['status'] != 'PASS': raise ValueError('Complete scientific checks before reporting')
    inventory = json.loads((PUBLIC/'input_inventory.json').read_text())
    evaluation = json.loads((RUN/'offline/evaluation.json').read_text())
    summ = json.loads((RUN/'offline/summary_completion.json').read_text())
    verdict = json.loads((PUBLIC/'verdicts.json').read_text())
    bootstrap = json.loads((PUBLIC/'bootstrap.json').read_text())
    summary = pd.read_csv(PUBLIC/'summary.csv').replace({np.nan: None}).to_dict('records')
    conflict = pd.read_csv(evaluation['conflicts']['path'], dtype={'video': str})
    distribution = pd.read_csv(PUBLIC/'conflict_summary.csv').replace({np.nan: None}).to_dict('records')
    lists = pd.read_csv(PUBLIC/'lists.csv', dtype={'video': str})
    structural = json.loads((PUBLIC/'structural_diagnostics.json').read_text())
    # Preserve the uncompressed machine conflict table outside git; publish a lossless compressed copy.
    with open(evaluation['conflicts']['path'], 'rb') as source, gzip.open(PUBLIC/'conflict.csv.gz', 'wb') as out:
        shutil.copyfileobj(source, out)
    metrics = Path(evaluation['metrics']['path']); exported_metrics = None
    if metrics.stat().st_size <= 5 * 1024**2:
        exported_metrics = PUBLIC/'search_runs.csv.gz'; shutil.copy2(metrics, exported_metrics)
    cost = dict(teacher_calls=0, model_calls=0, optimizer_updates=0, GPU_hours=0,
        CPU_stage_wall_seconds=dict(prepare=inventory['CPU_wall_seconds'], search=seal['CPU_wall_seconds'],
            GT_labels_oracle_and_evaluation=evaluation['CPU_wall_seconds'], summary_and_bootstrap=summ['CPU_wall_seconds'],
            validation_and_regression=validation['CPU_wall_seconds']),
        hypothetical_questions_not_real_teacher_calls=True,
        note='Only sealed probabilities/margins are replayed. These CPU timings do not estimate live teacher or model latency.')
    cost['CPU_measured_stage_wall_seconds_total'] = sum(cost['CPU_stage_wall_seconds'].values())
    attempts = [json.loads(p.read_text()) for p in sorted((RUN/'launchers').glob('*/status.json'))]
    cost['all_completed_launcher_stages'] = [row for a in attempts for row in a['stages']]
    cost['all_completed_launcher_stage_wall_seconds'] = sum(r['wall_seconds'] for r in cost['all_completed_launcher_stages'])
    cost['cost_coverage'] = 'All completed launcher stages including repeated aggregation/validation; excludes current final report rendering and coding/publication time.'
    dump(PUBLIC/'costs.json', cost)
    blind = []
    for (target, verifier), d in conflict[(conflict.cross_domain == 1) & conflict.verifier.isin(TEACHERS)].groupby(['target', 'verifier']):
        failures = d[d.search_failure == 1]
        blind.append(dict(target=target, verifier=verifier, lists_times_rankers=len(d), failures=len(failures),
            failure_without_observed_conflict=int((failures.C_online == 0).sum()),
            failure_without_observed_conflict_fraction=float((failures.C_online == 0).mean()) if len(failures) else None,
            any_online_conflict_fraction=float((d.C_online > 0).mean()), C_dense_defined_fraction=float(d.C_dense.notna().mean())))
    csv_rows(PUBLIC/'online_blindspots.csv', blind)
    archive = PUBLIC/'code_and_completed_logs.tar.gz'
    include = [*sorted((ROOT/'scripts').glob('*search_feasibility*.py')), ROOT/'tests/test_search_feasibility.py',
               PUBLIC/'PREREG.md', PUBLIC/'IMPLEMENTATION_NOTES.md', PUBLIC/'prereg_publication.json',
               RUN/'runtime/seal.json', RUN/'search/seal.json', RUN/'offline/labels_seal.json',
               RUN/'offline/oracle_seal.json', RUN/'offline/evaluation.json', RUN/'offline/summary_completion.json']
    include += sorted((RUN/'validation').glob('*.log'))
    include += [p for p in sorted((RUN/'launchers').rglob('*.log')) if p.name != 'report_search_feasibility.log']
    with tarfile.open(archive, 'w:gz') as out:
        for p in include: out.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
    # Verifiable separate process provenance, without publishing raw images/checkpoints/large margin banks.
    index = dict(source_commit='48dfb245dd4fe9b58f42dc1e37425a5e4095e8a5', inputs=inventory['inputs'],
        runtime=seal['runtime'], GT_free_predictions=seal['predictions'],
        GT_free_search_seal=record(RUN/'search/seal.json'),
        post_search_GT_labels_seal=record(RUN/'offline/labels_seal.json'),
        ORACLE_seal=record(RUN/'offline/oracle_seal.json'),
        ORACLE_predictions=json.loads((RUN/'offline/oracle_seal.json').read_text())['predictions'],
        complete_search_runs=record(metrics), complete_conflict=record(evaluation['conflicts']['path']),
        git_search_runs=record(exported_metrics) if exported_metrics else None,
        compressed_conflict=record(PUBLIC/'conflict.csv.gz'), code_and_logs=record(archive),
        large_data_in_git=False, GT_feedback_to_search=False)
    h1 = verdict['H1']; body = []
    body += ['# 起草—搜索—校验：固定规则可行性诊断\n',
        '本轮仅 CPU 重放封存探针，不调用教师/模型、不训练、不运行 RMOT 新方法。下面的 F1 是列表候选集合指标，不是完整 RMOT HOTA。所有预注册变体及失败都保留。\n',
        '## 1. 预注册判定\n', f"总判定：**{verdict['overall']}**。\n"]
    matrix = []
    for target in TARGETS:
        for model in MODELS:
            if model == IN_DOMAIN[target]: continue
            row = dict(target=target, ranking=model)
            for verifier in TEACHERS: row[verifier] = next(r['verdict'] for r in h1 if r['target']==target and r['model']==model and r['verifier']==verifier)
            matrix.append(row)
    body.append(table(matrix, ['target', 'ranking', *TEACHERS]))
    body.append('\nH2-a / H2-b 分教师判断，不能把不同教师的长处拼成通过：\n')
    decisions = []
    for v in TEACHERS:
        a, b = verdict['H2a'][v], verdict['H2b'][v]
        decisions.append(dict(verifier=v, Dance_C_dense=a['means']['Dance'], KITTI_C_dense=a['means']['KITTI-V2'],
            Sports_C_dense=a['means']['SportsMOT'], H2a=a['verdict'], pooled_online_AUROC=b['ALL']['AUROC'],
            H2b=b['verdict'], joint=verdict['per_teacher_joint'][v]['verdict']))
    body.append(table(decisions, ['verifier', 'Dance_C_dense', 'KITTI_C_dense', 'Sports_C_dense', 'H2a', 'pooled_online_AUROC', 'H2b', 'joint']))
    body.append('\nH2-b 按原门槛成立，但 C_online 是固定 PBA(0.2) 搜索结束时的数值，而本轮它已问完所有候选；尚未证明它能在少量提问时提前报警或带来实际节省。保留字面判定，不把它升级成已验证的高效在线系统。KITTI 的小列表使 N/ceil(log₂(N+1))=1.720238 即使问遍 N 也能通过 H1 数值门槛；不能把该通过解读为真实查询压缩。\n')
    body += ['\n### 1.1 不依赖 GT 的停止规则问题\n',
        '均匀先验下，相邻的两个分界点只在一个候选的答案上不同。每个候选只允许提问一次，因此任意相邻边界的后验与 MAP 后验之比至少为 ε/(1−ε)。在 ε=0.2 时最大后验至多 0.8（内部边界至多 2/3），不可能达到 0.9；ε=0.3 同样不可能，ε=0.1 在 N≥2、所有后验仍正时严格小于 0.9。因此三种均匀先验 PBA 都必须问完 N 个候选，不是 log₂N 查询。\n',
        f"真实全交叉检查：三种均匀 ε 全部问遍候选；各 ε 的 F1 完全一致={fmt(structural['PBA_epsilon_F1_exact_equal'])}，与等预算随机的 F1 完全一致={fmt(structural['PBA_random_budget_F1_exact_equal'])}，b̂ 也经逐条精确核对相同。固定 ε<0.5 下，全量答案的 MAP 就是最小不一致数的前缀，提问顺序已无法影响最后结果。这个 RANDOM 对照在当前协议下不能证明‘问哪里’更重要。PRESENCE 先验和标准二分仍按原文完整运行，没有替换规则。\n",
        '必须区别：预注册 H1 不通过并不自动证明模型排序不可信；停止规则本身已经破坏查询效率。GT_ORACLE 的 F1 损失另行诊断排序/前缀约束与所定 MAP 目标，不把查询效率失败都归罪教师噪声。\n',
        '### 1.2 H1 数值（全部跨域组合）\n']
    body.append(table([r for r in h1 if r['cross_domain'] and r['verifier'] in TEACHERS],
        ['target','model','verifier','mean_questions','mean_questions_over_log','PBA_F1','DENSE_F1','F1_fraction_of_DENSE','question_gate','F1_gate','verdict']))
    body += ['\n## 2. 全部跨域教师搜索结果\n',
        'ALL 为逐列表均值；g=0/g>0 是当前帧–表达的 GT 分组，不是整条视频表达是否为空。m 是模拟读取封存答案次数，不是本轮真实教师调用次数。m/log 先逐列表除以 ceil(log₂(N+1)) 再平均。空组留未定义，不补零。\n',
        table(inventory['inventory'], ['target','videos','lists','N_mean','N_p50','N_p90','N_min','N_max','excluded_N_lt_2_lists']),
        '\nGT 分组分母：\n']
    denom = [dict(target=t, lists=len(d), g_zero=int((d.g==0).sum()), g_present=int((d.g>0).sum()),
                  full_GT_count=int(d.g.sum()), candidate_matched_GT=int(d.matched_candidates.sum()))
             for t,d in lists.groupby('target')]
    body.append(table(denom, ['target','lists','g_zero','g_present','full_GT_count','candidate_matched_GT']))
    body.append('\nDance 的 753/850 个列表为 g=0。以 V2-best + CROP 为例，PBA(0.2) 的 ALL F1=0.821807 高于 DENSE=0.759393，但 g>0 F1=0.500371 低于 DENSE=0.716329；不能只用空列表占优的整体均值宣称恢复了目标。完美校验者下同一排序的 PBA 整体 F1=0.941816，而 g>0 仅0.490141。完整拆分均在下表。\n')
    columns = ['target','model','verifier','method','scope','lists','m','questions_over_log','F1','count_MAE','presence_accuracy','precision','recall']
    body += ['\n所有 10 个固定方法、三个教师和 g 拆分：\n',
        table([r for r in summary if r['cross_domain'] and r['verifier'] in TEACHERS], columns),
        '\n## 3. 冲突诊断\n',
        'C_dense 只在教师同时答 Yes/No 的列表定义；同质答案列表不当成零冲突纳入均值。C_online 的分母是实际提问数，后到回答若与任一先前回答矛盾只计一次。A_GT 使用真实 margin 与固定匹配候选的列表内 AUROC，单类标签未定义。Spearman 只用两者都有定义的同一列表，不给逐帧独立样本的显著性 p 值。\n',
        table([r for r in distribution if r['cross_domain'] and r['verifier'] in TEACHERS],
            ['target','model','verifier','lists','C_dense_defined','C_dense_defined_fraction','C_dense_mean','C_dense_std','C_dense_p50','C_dense_p90','C_online_mean','A_GT_defined','A_GT_mean','spearman_C_dense_A_GT','spearman_defined_lists']),
        '\n在线冲突预测失败（失败精确按有理 F1 差 > 1/10，不把浮点恰等于 0.1 误算为失败）：\n']
    auc_rows = []
    for v in TEACHERS:
        for t in ('ALL', *TARGETS):
            r = verdict['H2b'][v][t]; auc_rows.append(dict(verifier=v,target=t,rows=r['lists_times_cross_rankers'],
                failures=r['failures'],failure_rate=r['failure_rate'],AUROC=r['AUROC'],CI95=r.get('CI95'),
                valid_bootstrap=r.get('valid_bootstrap_samples')))
    body += [table(auc_rows, ['verifier','target','rows','failures','failure_rate','AUROC','CI95','valid_bootstrap']),
        '\n没有观察到在线矛盾的失败，说明一致但错误的回答/错误前缀也可能悄然失败；下表仅作归因，不新增开关或阈值：\n',
        table(blind, ['target','verifier','failures','failure_without_observed_conflict','failure_without_observed_conflict_fraction','any_online_conflict_fraction','C_dense_defined_fraction']),
        '\n## 4. GT_ORACLE【ORACLE，仅诊断】\n',
        'ORACLE 在独立离线进程中、GT-free 搜索封存之后才创建并另行封存。不参与主关卡。DENSE 选择固定一对一匹配的全部候选；g>0 但固定 MOT 没有可匹配候选时 F1 仍为0。完美校验者的 PBA 若达不到 0.9×DENSE，说明教师噪声之外还有排序/前缀限制；这里的 MAP 优化不一致数，不是事后用 GT 找最大 F1 的前缀。\n',
        table([r for r in h1 if r['cross_domain'] and r['verifier']=='GT_ORACLE'],
            ['target','model','mean_questions','mean_questions_over_log','PBA_F1','DENSE_F1','F1_fraction_of_DENSE','question_gate','F1_gate','verdict']),
        '\n全部 ORACLE 跨域方法和空/非空拆分：\n',
        table([r for r in summary if r['cross_domain'] and r['verifier']=='GT_ORACLE'], columns),
        '\n## 5. 同域参照（不参与跨域判定）\n',
        table([r for r in h1 if not r['cross_domain']], ['target','model','verifier','mean_questions','mean_questions_over_log','PBA_F1','DENSE_F1','F1_fraction_of_DENSE','verdict']),
        '\n所有同域方法与分组：\n', table([r for r in summary if not r['cross_domain']], columns),
        '\n## 6. 视频聚类不确定性\n',
        'Dance 5、SportsMOT 15 个完整视频为重采样单位，2000 次、seed 0；同一视频的全部帧、句子、排序来源和方法一起重采样。点估计仍是逐列表平均。KITTI-V2 4 视频只给点估计。合并 H2-b 的区间只重采样 Dance/Sports、保持 KITTI 固定，是条件区间，不宣称对 KITTI 总体也完成 bootstrap。不存在逐帧独立显著性假设。\n',
        '全部组合、全部 g 拆分和所有 7 个均值字段的 CI 在 [bootstrap.json](bootstrap.json)。没有把区间条件额外加到预注册的点估计判定。空组/单类重采样不补默认值，保留有效重复次数。\n',
        table(bootstrap['comparisons'], ['target','model','verifier','comparison','delta','CI95','clusters','valid_bootstrap_samples']),
        '\n跨域来源 pooled C_dense 的视频级区间：\n',
        table([r for r in bootstrap['conflicts'] if r['metric']=='cross_domain_mean_C_dense'], ['target','verifier','point','CI95','clusters','valid_bootstrap_samples']),
        '\n各排序来源的冲突、A_GT 均值及 Spearman 的视频级区间（含同域与 ORACLE）：\n',
        table([r for r in bootstrap['conflicts'] if r['metric']!='cross_domain_mean_C_dense'], ['target','model','verifier','metric','point','CI95','clusters','valid_bootstrap_samples']),
        '\n## 7. 隔离、成本、失败与交付\n',
        f"原文预注册已在 `{inventory['prereg_publication']['remote_commit']}` 推送后才计算；计算口径说明在 `08aa71de7336079e654c9e9a7f699efc36620ec3` 提前公开。GT-free 搜索实现及阶段记录在 `042023116391c15b4465714da038d69104496557` 推送后才离线评测。基点是用户指定的精确公开提交48dfb245，而非混用另一个项目。\n",
        f"GT-free 准备打开尝试 0；GT-free 搜索打开尝试 0；Landlock kernel read whitelist 已启用，只允许 runtime 列表和代码/环境及自己的输出。{seal['runs']} 条 GT-free 方法输出先封存 SHA `{seal['predictions']['sha256']}`；然后独立进程才读取 GT、生成 {evaluation['runs']-seal['runs']} 条 ORACLE 输出并另行封存，再做指标。负向单元测试故意拒读不算实验进程尝试。GT/历史事后表均没有进入模拟搜索输入。\n",
        'GT 协议复用 v2：mark!=0/class=1，IoU≥0.5，最大有效匹配数优先再最大 IoU，固定候选顺序、不依赖模型分数。Sports 使用已授权统一原始标注 v4，不能退回不完整旧418模板。列表 g 是完整 GT 数；集合 F1 对固定匹配候选集合，二者不混淆。输入教师 p_yes≥0.5，NATIVE margin>0；框、ID、帧号全程不改。\n',
        f"本轮真实教师调用 0、模型调用 0、优化步 0、GPU 小时 0。{validation['tests_passed']} 项新诊断/既有数据口径回归通过，compileall 通过；逐列表 × 3 模型 × 4 校验者 × 10 方法共 {evaluation['runs']} 行完整。DENSE 排序无关和 NATIVE 校验者无关的逐列表指标自检通过。\n",
        table([dict(stage=k,wall_seconds=v) for k,v in cost['CPU_stage_wall_seconds'].items()], ['stage','wall_seconds']),
        f"\n最近一套各阶段记录合计 {cost['CPU_measured_stage_wall_seconds_total']:.3f} 秒；包含首次及补齐不确定性后重复汇总/检查的全部已完成 launcher 阶段墙钟累计 {cost['all_completed_launcher_stage_wall_seconds']:.3f} 秒，逐次成本在 [costs.json](costs.json)。不包含人工编码、等待发布和末次报告渲染。没有重新搜索、标注或评测预测。日志/代码归档见 [code_and_completed_logs.tar.gz](code_and_completed_logs.tar.gz)。模拟提问数不能当成本轮真实教师耗时，也不能据此给出在线加速倍数。\n",
        '失败与修复：科学协议未改，未换 seed、ε、门槛或排序。编码期修正了严格 0.1 失败比较的浮点边界，用精确有理数实现原规则，并通过定向回归；没有根据 GT 结果修改算法。运行阶段若有非零退出，保留对应 launcher 日志并在 failure_records 中记录，不隐藏重跑。均匀 PBA 的停止限制是实质负面发现，不启动另一个修复算法。\n',
        '完整性补充：仅重新汇总已封存离线表，补齐所有冲突/A_GT/Spearman 的视频 bootstrap；用加权并列秩精确复现整视频重复抽样，并经字面重复样本测试。合并 H2-b 的两个目标用同一 seed=0 RNG 的独立后续抽样，KITTI 保持固定；不把重置的同一随机流当独立抽样。初次汇总保存在 offline/initial_aggregation，预测和逐列表指标 SHA 未变，全部 H1/H2 点估计与判定未变。\n',
        '适用边界：列表最多19个候选，不能据此声称大 N 的渐近 log₂N；这里只检验已开发过的稀疏探针，不是新盲测或完整 RMOT 结果。教师概率是一次冻结缓存的确定性阈值回答，ε 是假定的似然参数，未重新采样教师噪声。GT_ORACLE 仍不包含 MOT 不可恢复目标。若关卡失败，应停留在诊断结论，而不是把 loss、单测或某个有利变体包装成框架成功。\n',
        '机器结果：[summary.csv](summary.csv)、[lists.csv](lists.csv)、[conflict.csv.gz](conflict.csv.gz)、[conflict_summary.csv](conflict_summary.csv)、[H1.csv](H1.csv)、[H2b.csv](H2b.csv)、[bootstrap.json](bootstrap.json)、[validation.json](validation.json)、[input_output_paths.json](input_output_paths.json)。逐列表逐方法完整 search_runs.csv.gz 与全部输入、搜索预测及 ORACLE 的原始路径/SHA 见清单；大分数库、图像和 checkpoint 不入 git。\n',
        '完成后停止：不训练、不调用教师、不启动模型推理、不实现或运行新方法。\n']
    if exported_metrics: body.append('完整逐列表机器表也已作为小型压缩件发布：[search_runs.csv.gz](search_runs.csv.gz)。\n')
    (PUBLIC/'REPORT.md').write_text('\n'.join(body))
    failure_records = []
    for path in sorted((RUN/'launchers').glob('*/status.json')):
        status = json.loads(path.read_text())
        failure_records += [r for r in status['stages'] if r['exit_code']]
    index['failure_records'] = failure_records
    if failure_records:
        with (PUBLIC/'REPORT.md').open('a') as out:
            out.write('\n实际运行失败（原日志保留）：\n\n' + table(failure_records, ['stage','exit_code','wall_seconds','log']))
    index['report'] = record(PUBLIC/'REPORT.md')
    index['published_machine_files'] = [record(p) for p in sorted(PUBLIC.iterdir()) if p.is_file() and p.name not in ('REPORT.md','input_output_paths.json')]
    dump(PUBLIC/'input_output_paths.json', index)
    print(json.dumps(dict(event='SEARCH_REPORT_COMPLETE', overall=verdict['overall'], report=str(PUBLIC/'REPORT.md'),
                         report_bytes=(PUBLIC/'REPORT.md').stat().st_size, GPU_hours=0, teacher_calls=0)), flush=True)


if __name__ == '__main__': main()
