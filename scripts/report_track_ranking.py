"""All fixed results and scientific limitations; large unit tables stay external."""
import gzip
import json
from pathlib import Path
import shutil
import sys
import tarfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.track_ranking_common import *


def fmt(x):
    if x is None or isinstance(x,float) and not np.isfinite(x):return '未定义'
    if isinstance(x,(float,np.floating)):return f'{x:.6f}'
    if isinstance(x,(bool,np.bool_)):return '是' if x else '否'
    return str(x).replace('|','\\|').replace('\n',' ')


def table(rows,cols):
    rows=list(rows)
    if not rows:return '无此类样本；未定义，不填零。\n'
    return '| '+' | '.join(cols)+' |\n|'+'|'.join('---' for _ in cols)+'|\n'+''.join(
        '| '+' | '.join(fmt(r.get(c)) for c in cols)+' |\n' for r in rows)


def records(name):
    return pd.read_csv(PUBLIC/name).replace({np.nan:None}).to_dict('records')


def main():
    proof=require_plan();validation=json.loads((PUBLIC/'validation.json').read_text())
    if validation['status']!='PASS':raise ValueError('Do not publish an unvalidated diagnostic')
    done=json.loads((RUN/'summary_completion.json').read_text());v=json.loads((PUBLIC/'verdicts.json').read_text())
    plan=json.loads((RUN/'plan.json').read_text());noise=json.loads((RUN/'noise_rates.json').read_text())
    boot=json.loads((PUBLIC/'bootstrap.json').read_text());q=pd.read_csv(done['machine_files']['sequence_metrics.csv']['path'])
    aucs=records('auroc_summary.csv');prefix=records('prefix_summary.csv');bis=records('bisect_summary.csv')
    variability=records('variability_summary.csv');comparisons=records('comparison_summary.csv')
    presence=records('presence_summary.csv');objects=records('object_counts.csv');byvideo=records('object_counts_by_video.csv')
    exports=[]
    for name,rec in done['machine_files'].items():
        source=Path(rec['path']);compressed=RUN/'compressed'/(name+'.gz');compressed.parent.mkdir(exist_ok=True)
        with source.open('rb') as inp,gzip.open(compressed,'wb') as out:shutil.copyfileobj(inp,out)
        public_path=None
        if compressed.stat().st_size<=5*1024**2:
            public_path=PUBLIC/compressed.name;shutil.copy2(compressed,public_path)
        if source.stat().st_size<=80000:
            raw_public=PUBLIC/name;shutil.copy2(source,raw_public)
        exports.append(dict(name=name,raw=rec,compressed=record(compressed),
            git_compressed=record(public_path) if public_path else None,role=ROLE))
    failures=[json.loads(p.read_text()) for p in sorted((RUN/'launchers').glob('*/*_failed.json'))]
    stages=[r for path in sorted((RUN/'launchers').glob('*/status.json')) for r in json.loads(path.read_text())['stages']]
    launchers=[dict(path=str(p),**json.loads(p.read_text())) for p in sorted((RUN/'launchers').glob('*/completion.json'))]
    cost=dict(role=ROLE,new_teacher_calls=0,new_model_calls=0,GPU_hours=0,optimizer_updates=0,
        CPU_completed_worker_wall_seconds=sum(r['wall_seconds'] for r in stages),
        CPU_failed_worker_wall_seconds=sum(r['wall_seconds'] for r in failures),
        completed_stages=stages,failed_stages=failures,completed_launchers=launchers,
        note='Three target CPU workers run concurrently: accumulated worker wall time is not elapsed end-to-end time or physical CPU-core time. Excludes current report compression/rendering, coding and GitHub publication.')
    dump(PUBLIC/'costs.json',cost)
    combo_variability=[]
    for row in v['matrix']:
        var=next(r for r in variability if r['protocol']=='FULL_MAIN_V4' and r['target']==row['target'] and r['model']==row['model'] and r['scope']=='ALL')
        combo_variability.append(dict(combo=row['combo'],**{k:var[k] for k in ('target','model','q_track_pairs','time_varying_pairs','fraction','annotation')}))
    csv_rows(PUBLIC/'variability_by_combo.csv',combo_variability)
    cohort=[]
    for target,d in q[q.model=='V2-best'].groupby('target'):
        cohort.append(dict(target=target,sequences=len(d),GT_present=int(d.GT_present.sum()),GT_empty=int((d.GT_present==0).sum()),
            Y_positive_sequences=int((d.positive_tracks>0).sum()),GT_present_but_Y_zero=int(((d.GT_present==1)&(d.positive_tracks==0)).sum()),
            common_AUROC_q=int(d.common_AUROC_q.sum())))
    csv_rows(PUBLIC/'cohort.csv',cohort)
    # Include complete small receipts/logs and code, never raw images, weights, or huge score/GT matrices.
    archive=PUBLIC/'code_and_logs.tar.gz'
    include=[*sorted((ROOT/'scripts').glob('*track_ranking*.py')),ROOT/'tests/test_track_ranking.py',
        PUBLIC/'PREREG.md',PUBLIC/'IMPLEMENTATION_NOTES.md',PUBLIC/'prereg_publication.json',RUN/'plan.json',RUN/'noise_rates.json',RUN/'summary_completion.json']
    include+=sorted((RUN/'targets').glob('*/*/receipt.json'))+sorted((RUN/'targets').glob('*/completion.json'))
    include+=sorted((RUN/'validation').glob('*.log'))
    include+=[p for p in sorted((RUN/'launchers').rglob('*.log')) if p.name!='report_track_ranking.log']
    with tarfile.open(archive,'w:gz') as out:
        for path in include:out.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
    body=['# 轨迹级排序可靠性诊断\n',
        '【ORACLE / GT，仅诊断】本轮只读取封存官方完整 margin 与 GT，并重放旧 CROP 答案以估计模拟噪声率。没有新模型/教师调用、GPU 推理或训练。所有聚合使用整条轨迹，包括未来观测；不是因果在线方法，不是完整 RMOT HOTA，也不是可部署的 GT-free 选择器。\n',
        '## 1. 预注册判定\n',f"总判定：**{v['overall']}**。{v['joint_pass_count']}/6 同时通过 T1/T2，{v['no_improvement_count']}/6 被固定规则归为‘轨迹级没有明显改善’。\n",
        table(v['matrix'],['combo','target','model','common_sequences','FRAME_AUROC','TRACK_MEAN_AUROC','AUROC_delta','T1','TRACK_prefix_F1','T2','joint_pass']),
        '\nT1 的点估计严格用三粒度共同有定义 q，FRAME 先逐 q 平均；不把所有帧当独立样本。其他聚合只完整报告，不替换主聚合 MEAN 或改判定。T2 按原文使用全部 q，因此必须同时看下面的有目标/有正轨迹拆分。\n',
        '时变轨迹（严格 0.1 < 匹配比例 < 0.9；分母为全部合格 q–轨迹对，非唯一物理轨迹）：\n',
        table(v['target_time_variability'],['target','q_track_pairs','time_varying_pairs','fraction','annotation']),
        '\n同一目标下的时变标签不依赖模型，按组合重复报告不会得到不同占比。原阈值只对上述 ALL 比例应用，不事后换条件分母。\n',
        table(combo_variability,['combo','target','model','q_track_pairs','time_varying_pairs','fraction','annotation']),
        '\n三个目标均未超过原定20%门槛，未触发“该目标需要片段级而非轨迹级的判断”标注；这不是排除少量时变目标的证据。\n',
        table(cohort,['target','sequences','GT_present','GT_empty','Y_positive_sequences','GT_present_but_Y_zero','common_AUROC_q']),
        '\n### 1.1 需要防止的误读\n',
        'FRAME AUROC、SEGMENT AUROC、TRACK AUROC 的对象和多数标签不同；共同 q 控制了序列取样，但不能使三个对象定义变成同一任务。TRACK Y 会把不足50%匹配的轨迹归负，因此 GT-present 也可能没有正轨迹。前缀 F1 是相对本粒度标签的上限，不是覆盖完整 GT 的 RMOT 召回。\n',
        'Dance 有385/425条空序列；即使所有有目标序列失败，ALL 前缀 F1 仍至少385/425。因此高 ALL T2 分数不能单独证明有目标轨迹可被可靠排序。完美校验者的二分也不等于前缀上限：前者只见搜索路径上的 Y，后者事后扫描所有 k 并使用 GT 选最优 k。\n',
        '## 2. 三种粒度的 AUROC、全部聚合与对象数\n',
        'ALL_DEFINED_FRAMES 是所有有双类帧的原始平均；ALL_DEFINED_Q 是各粒度自身有定义 q 的平均；COMMON_Q 是 FRAME/SEGMENT/TRACK 都有定义的同一批 q。主 T1 只能比较 COMMON_Q，未定义单元不补0.5。RAW/COMMON 的分母均完整列出，原始逐单元 AUROC 在机器表。\n',
        table([r for r in aucs if r['protocol']=='FULL_MAIN_V4'],['target','model','granularity','aggregation','scope','sequences','common_sequences','defined_units','undefined_units','mean_AUROC']),
        '\n物理对象数分布（FRAME 按每个实际 tracker 帧一次，TRACK 按每视频一次；不被表达数量重复加权）：\n',
        table(objects,['target','granularity','physical_units','mean','p50','p90','minimum','maximum','excluded_physical_tracks','excluded_physical_observations']),
        '\n逐视频过滤与分段信息：\n',table(byvideo,['target','video','raw_tracks','retained_tracks','excluded_tracks','raw_observations','retained_observations','excluded_observations','frames','frames_with_zero_retained_candidates','segment_count','fps','segment_length_frames']),
        '\n## 3. FRAME / TRACK 前缀上限\n',
        'FRAME 在 q 内对全部候选帧的最佳前缀 F1 平均，再跨 q 平均；TRACK 每 q 扫描全部 k。无正标签时空前缀 F1=1。GT_PRESENT/EMPTY 使用官方整序列 GT；Y_POSITIVE/ZERO 使用多数轨迹标签，不混用两种存在性。\n',
        table([r for r in prefix if r['protocol']=='FULL_MAIN_V4'],['target','model','scope','granularity','sequences_in_scope','mean_prefix_F1']),
        '\n## 4. 完美与模拟校验者的查询代价 / F1\n',
        '只做固定标准二分 BISECT_EXACT，不再使用上一轮会问全的 PBA。轨迹按 MEAN 降序、ID升序；同一 q 的噪声答案按物理 ID 生成一次，三个模型和二分/全量对照共用，不能按不同排序重新抽噪声。提问数是模拟读答案数，不是本轮真实教师调用次数。\n',
        '旧 decompose CROP 的原始条件错误率（全29146个候选判断，包含 N=1；按完整候选匹配标签，不是 GT-R@1 或视频存在性错误）：\n',
        table(noise['rates'].values(),['target','candidate_pairs','TP','FN','FP','TN','positive_recall','FNR','FPR']),
        '\n这些是单帧 CROP 的实测候选错误率，投射到轨迹仅作为独立 Bernoulli 翻转的假设诊断；未验证轨迹级教师有同样错误率或时间独立性。所有模拟结果一并上报，不挑 seed 或较好的模型。\n',
        table([r for r in bis if r['protocol']=='FULL_MAIN_V4'],['target','model','scope','verifier','sequences_in_scope','mean_tracks','mean_bisect_questions','mean_dense_questions','questions_ratio','mean_bisect_F1','mean_dense_F1','F1_ratio']),
        '\nquestions_ratio = 平均提问数 / 平均轨迹数；F1_ratio = 平均二分F1 / 平均全量F1。它们不是逐 q 比例的平均。不对 T3 新增成功门槛，F1 损失和查询节省必须同时看。完美全量对照逐 q F1=1 已自检。\n',
        '## 5. Dance 视频内存在性可分性\n',
        '每个视频17句，标签取官方过滤后整序列 GT 是否非空；每句分数取合格轨迹聚合分数最大值。每视频 AUROC 后再平均；只有一类的视频未定义，不补0.5。四种既定聚合全列出，MEAN 为主，不改教师/阈值。\n',
        table(presence,['model','aggregation','videos','defined_videos','undefined_videos','mean_AUROC','CI95','valid_bootstrap_samples']),
        '\n逐视频值在 presence_auc.csv / 其无损压缩件，见文件清单。Sports 和 KITTI 没有 GT-empty 序列，本节没有定义为其存在性分类实验。\n',
        '## 6. 视频聚类 bootstrap\n',
        'Dance25、Sports45视频为单位，2000次、seed0，整视频携带全部表达和帧；FRAME/TRACK配对用相同重采样权重。AUROC差值只在共同 q 上计算，前缀差值先逐 q 配对再汇总。KITTI4视频只报告点估计。Sports418保留同一v4内容并沿用原45视频权重，不把帧/表达误当独立样本。点判定不加入未注册的 CI 门槛。\n',
        table(boot['comparisons'],['protocol','target','model','metric','point','CI95','clusters','valid_bootstrap_samples']),
        '\n## 7. SportsMOT 原官方418序列覆盖的敏感性\n',
        '用户已确认原文第1节的章节引用歧义：本节重算 T1/T2/T3，不把仅适用于 Dance 的第5节强套 Sports。主结果仍466条；418敏感性使用相同 v4 GT 内容，仅去掉原模板缺失的48个ID，不使用已知缺58897行的8条旧模板、不补空GT、不删主结果。分数、观测标签和模拟噪声逐 q 复用，不重新匹配/抽样。\n',
        table(v['Sports418'],['combo','model','sequences','common_sequences','FRAME_AUROC','TRACK_MEAN_AUROC','AUROC_delta','T1','TRACK_prefix_F1','T2']),
        '\n418条覆盖内全部既定聚合和自身/共同分母：\n',
        table([r for r in aucs if r['protocol']=='SPORTS418_V4_CONTENT'],['model','granularity','aggregation','scope','sequences','common_sequences','defined_units','undefined_units','mean_AUROC']),
        '\n418条的前缀与分组：\n',
        table([r for r in prefix if r['protocol']=='SPORTS418_V4_CONTENT'],['model','scope','granularity','sequences_in_scope','mean_prefix_F1']),
        '\n418条的 T3：\n',
        table([r for r in bis if r['protocol']=='SPORTS418_V4_CONTENT'],['model','scope','verifier','sequences_in_scope','mean_tracks','mean_bisect_questions','questions_ratio','mean_bisect_F1','mean_dense_F1','F1_ratio']),
        '\n## 8. 成本、失败修复、输入与可恢复路径\n',
        f"仓库 `/data2/user/FlexHook-TTA`，分支 research/track-ranking-20261102，从精确公开提交8e7d6e3创建，不进入LocateMOT。原文预注册先公开 `{proof['remote_commit']}`，用户两项澄清及计算口径先公开 `2bfc9ac160ee8a3ac65c87e0cf1770e4491fce93`，才开始计算。九套完整分数的清单 Git blob `{SOURCE_BLOB}` 与用户指定的01add887完全相同，所有SHA开始前/交付前核对，末尾观测保留。\n",
        '轨迹 n<3 对三粒度统一过滤；先在完整帧候选上匹配，再过滤，不重分配GT；不插值、不改框/ID/帧号。按首观测帧起每20/40/50帧划窗，空窗不补帧；短尾按时间跨度并入前一非空段。TOP25取ceil(n/4)，LOGIT_MEAN用数学等价稳定形式，不裁剪原分数。每个数据行均为 ORACLE/GT 诊断。\n',
        f"新增教师调用0、模型调用0、GPU小时0、优化步0。{validation['videos']}视频、{validation['sequences']}序列、{validation['model_sequence_rows']}个模型–序列组合；原始 FRAME AUROC {validation['frame_AUROC_rows']}行。{validation['tests_passed']}项定向及旧口径回归通过、compileall通过；过滤/分段观测守恒、标签多数规则、AUROC未定义规则、全量完美F1、共享噪声、二分提问界和前缀上限均通过。\n",
        f"CPU已完成worker累计墙钟 {cost['CPU_completed_worker_wall_seconds']:.3f} 秒，失败worker {cost['CPU_failed_worker_wall_seconds']:.3f} 秒。三个目标worker并行，累计worker墙钟不是实际总等待时间或物理CPU核时；各launcher实际墙钟、日志与阶段成本在 costs.json。末次压缩/报告渲染、编码、发布耗时不混作科学计算时长。\n",
        '失败修复记录：首个预检把含历史覆盖项的旧清单总行数误当成完整覆盖项数，在读GT/算指标前停止。最小修复为只筛 coverage=complete；新增针对真实清单的回归后通过九套完整分数核验。失败日志保留，没有改数据、弱化九模型覆盖要求、换种子、调阈值或隐藏失败。\n',
        '验证阶段曾出现混合类型视频ID的pandas提示；读取units表时显式指定video为字符串，保留KITTI前导零，并重跑验证/打包。逐视频匹配、聚合、随机答案和全部统计未重跑或修改。\n',
        table(failures,['stage','exit_code','wall_seconds','log']),
        '\n恢复单位为完整视频：每个 targets/<target>/<video>/receipt.json 保存其文件SHA、GT来源及耗时；metrics阶段可验证后跳过完成视频。summary_completion.json索引全部机器表。没有未完成的视频、训练队列或模型推理任务。\n',
        '复现命令：使用本机既有Python环境 `/data2/user/TTA-RMOT-assets/teacher-p1/env/bin/python`（只借用解释器/NumPy/SciPy，不加载教师），`CUDA_VISIBLE_DEVICES=\'\'`；运行 scripts/run_track_ranking.py 的 --phase metrics、--phase aggregate、--phase report。汇总只读封存输出，不能把GT上限或模拟答案接回正式TTA。\n',
        '文件：input_inventory.json、input_output_paths.json、noise_rates.csv、auroc_summary.csv、prefix_summary.csv、bisect_summary.csv、presence_summary.csv、bootstrap.json、validation.json；完整逐单元原始 CSV 的路径/SHA 和无损压缩件如下。大帧表、GT标签缓存、原始margin/GT/图像/checkpoint留在data2/既有资产目录，不入git。\n']
    body.append(table([dict(file=r['name'],rows=r['raw']['rows'],bytes=r['raw']['bytes'],
        raw_path=r['raw']['path'],sha256=r['raw']['sha256'],git_copy=r['git_compressed']['path'].split('/results/track_ranking/')[-1] if r['git_compressed'] else '大文件仅外部路径') for r in exports],
        ['file','rows','bytes','raw_path','sha256','git_copy']))
    body+=['\n代码、完整精简日志与74视频receipt在 [code_and_logs.tar.gz](code_and_logs.tar.gz)。全部GT输入逐文件SHA随receipt保存；本任务没有创建/覆盖GT、教师缓存或旧实验结果。\n',
        '完成后停止：不调用教师、不训练、不开始方法实现。即使个别聚合/子集指标更好，也不改变已注册的MEAN判定或自行运行新实验。\n']
    (PUBLIC/'REPORT.md').write_text('\n'.join(body))
    paths=dict(role=ROLE,margin_source_commit=SOURCE_COMMIT,base_commit=proof['base_commit'],inputs=plan['inputs'],
        GT_and_output_receipts=done['video_receipts'],noise_GT_sources=record(RUN/'noise_rates.json'),
        machine_files=exports,summary_completion=record(RUN/'summary_completion.json'),code_and_logs=record(archive),
        report=record(PUBLIC/'REPORT.md'),new_teacher_calls=0,new_model_calls=0,GPU_hours=0,
        completed=True,failed_stages=failures,all_processes_complete=True)
    paths['published_files']=[record(p) for p in sorted(PUBLIC.iterdir()) if p.is_file() and p.name not in ('REPORT.md','input_output_paths.json')]
    dump(PUBLIC/'input_output_paths.json',paths)
    print(json.dumps(dict(event='TRACK_REPORT_COMPLETE',overall=v['overall'],report=str(PUBLIC/'REPORT.md'),
        tests_passed=validation['tests_passed'],GPU_hours=0,new_teacher_calls=0)),flush=True)


if __name__=='__main__':main()
