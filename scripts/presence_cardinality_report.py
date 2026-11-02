"""Independent cardinality decision reports; historical reports are never rewritten."""
from argparse import ArgumentParser
import csv
import json
from pathlib import Path
import sys
import tarfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.presence.runtime import RUN,PUBLIC,IN_DOMAIN
from flex_tta.runtime.pf_receipts import file_sha,atomic_json
from scripts.presence_redesign_report import table

OUT=RUN / 'redesign_v4';RECORD=PUBLIC / 'redesign_v4'


def main():
    global OUT,RECORD
    parser=ArgumentParser(description=__doc__);parser.add_argument('--version',choices=['v4','v5'],default='v4')
    args=parser.parse_args();version=args.version
    OUT=RUN / ('redesign_'+version);RECORD=PUBLIC / ('redesign_'+version)
    gate=json.loads((RECORD / 'gate.json').read_text())
    with (RECORD / ('stage_A.csv' if gate['completed_full_A'] else 'same_domain.csv')).open() as h:rows=list(csv.DictReader(h))
    with (PUBLIC / 'redesign_v3/same_domain.csv').open() as h:previous=list(csv.DictReader(h))
    with (PUBLIC / 'redesign_v4/same_domain.csv').open() as h:previous_v4=list(csv.DictReader(h))
    source=[];choices=json.loads((RECORD / 'selection.json').read_text())['models']
    for target,name in IN_DOMAIN.items():
        r=json.loads(Path(choices[name]['receipt_path']).read_text());neg,pos=r['fits']
        source.append(dict(source=target,model=name,alpha_neg=neg['alpha'],alpha_pos=pos['alpha'],
            train_native_NLL=r['train_metrics']['native_NLL'],train_calibrated_NLL=r['train_metrics']['calibrated_NLL'],
            validation_native_NLL=r['validation_metrics']['native_NLL'],validation_calibrated_NLL=r['validation_metrics']['calibrated_NLL'],
            validation_native_Brier=r['validation_metrics']['native_Brier'],validation_calibrated_Brier=r['validation_metrics']['calibrated_Brier'],
            CPU_fit_wall_seconds=r['wall_seconds'],function_evaluations=0 if version=='v5' else r['scalar_solver_evaluations'],
            inherited_v4_function_evaluations=r['scalar_solver_evaluations'] if version=='v5' else 0,parameters=r['parameters'],GPU_hours=0.,
            source_train_rows=r['train_units'],source_validation_rows=r['validation_units'],
            GT_open_attempts=r['target_GT_open_attempts']))
    own=[r for r in rows if r['own_domain']=='True' and r['method']=='PRES_TOPK']
    compare=[]
    for r in own:
        v3=next(p for p in previous if p['target']==r['target'] and p['method']=='PRES_TOPK')
        v4=next(p for p in previous_v4 if p['target']==r['target'] and p['method']=='PRES_TOPK')
        row=dict(source=r['target'],model=r['model'],Native=r['native_HOTA'],v3=v3['HOTA'],v4=v4['HOTA'])
        row[version]=r['HOTA'];row['delta']=r['delta_HOTA'];compare.append(row)
    status='同域门槛失败，未启动B/C' if gate['failed_targets'] else '同域门槛通过；不等于跨域或TTA成功'
    title='v4：源校准多伯努利数量分支' if version=='v4' else 'v5：固定源概率的预期集合IoU决策'
    text='# '+title+'\n\n'+status+'。\n\n'
    text+='## 1. 同域验收\n\n各官方模型在自身源域验证，不是V2→Dance跨域TTA。原Native−1门槛不变，主结果只用源Train固定拟合解，不能改选Native、未校准头、平滑版或其他温度。\n\n'
    text+=table(['source','model','Native','v3','v4']+(['v5'] if version=='v5' else [])+['delta'],compare)
    text+='\n## 2. 从原始文献得到的构造\n\n'
    text+=('[Guo et al.](https://proceedings.mlr.press/v70/guo17a.html)支持低维正温度概率校准的思路；'
        '[Reuter et al.](https://ba-ngu.vo-au.com/vo/RVVD_LMB_TSP14.pdf)给出多伯努利集合建模。'
        '本轮只借用候选概率及数量聚合，不复现该跟踪器、也不把论文的无偏结论套在本方法上。\n\n'
        'v3监督单个排序前缀的能量；v4使用候选级源GT拟合两个正斜率，正/负margin各一参数，无偏置、不读PCD/句子/图像。'
        '按候选独立近似计算Poisson-binomial数量分布，包含全部候选子集；P(存在)=1−prod(1−p_i)，'
        'v4数量解码为0..n后验的MAP，再用未变的源排序top-k。没有Native数量下界或失败回退。'
        '单候选分类/排序保持不保证集合数量保持，本轮HOTA是真实解码输出，不是复制Native。\n\n'
        '两个参数只由完整自身源Train候选BCE的一维凸优化决定，范围[.05,20]、xatol1e-6、maxiter128；'
        '源验证仅诊断，不选参数。selection.json中的best_source_loss兼容字段仅表示该固定解的源验证NLL，'
        '不是按验证选择的最小值。预注册与参数SHA先公开，再预测封存、独立GT评测。\n\n')
    if version=='v5':
        text+=('[Dai & Li, RankSEG, JMLR2023 Lemma13](https://jmlr.org/papers/volume24/22-0712/22-0712.pdf)'
            '启发本轮固定变化：复用v4三个源Train拟合参数及SHA，不重新拟合或改变概率。'
            '对原margin排序前缀S_k，独立伯努利近似下，U(0)=prod(1−p_i)；'
            'k≥1时U(k)=sum_inside(p_i)×E[1/(k+F_outside)]，suffix动态规划精确求分布。'
            '选择U最大的k，并列取最小值；U是集合IoU效用、不是数量概率，不归一化。'
            '不强制Native下界、不回选v4/平滑。源候选分类、候选排序和校准概率均与v4不变，'
            '因此v4→v5差异可归于固定决策规则，不是新语义学习。'
            '论文针对分割，本轮针对固定候选集合；单帧代理并不保证多阈值匹配、轨迹关联和RMOT HOTA。\n\n')
    text+=table(['source','alpha_neg','alpha_pos','train_native_NLL','train_calibrated_NLL','validation_native_NLL','validation_calibrated_NLL'],source)
    text+='\n'+table(['source','validation_native_Brier','validation_calibrated_Brier','source_train_rows','source_validation_rows'],source)
    text+='\nDance源Train NLL下降而源验证NLL变差，已在HOTA前记录且未换参数。Brier变好也不能消除错误高置信概率的尾部风险；源GT候选到tracker候选支持偏移仍存在。\n\n'
    text+='## 3. 全部已计算指标\n\n'+table(['target','model','method','native_HOTA','HOTA','delta_HOTA','DetA','AssA','IDF1','IDSW','prediction_rows','query_empty_prediction_rows','Present_HOTA','GT_open_attempts'],rows)
    behavior=[];probability_audits=[]
    for r in own:
        folder=Path(r['path']).parent;k=np.load(folder / 'raw_k.npy');j=np.load(folder / 'joint_count_probs.npy')
        c=np.load(folder / 'count_probs.npy');pl=np.load(folder / 'presence_logits.npy')
        from scipy.special import expit
        mass_error=float(np.abs(j.sum(1)-1).max())
        presence_error=float(np.abs((1-j[:,0])-expit(pl)).max())
        mean_error=float(np.abs(j@np.arange(j.shape[1])-expit(pl)*(c@np.arange(c.shape[1]))).max())
        if mass_error>2e-6 or presence_error>2e-6 or mean_error>2e-4:raise ValueError('Saved probability outputs are inconsistent')
        probability_audits.append(dict(target=r['target'],max_mass_error=mass_error,max_presence_error=presence_error,max_mean_error=mean_error))
        if version=='v5':
            utility=np.load(folder / 'expected_iou_utilities.npy')
            regret=float((utility.max(1)-utility[np.arange(len(k)),k]).max())
            if not np.isfinite(utility).all() or regret>2e-6:raise ValueError('Stored IoU utility decision is inconsistent')
            probability_audits[-1].update(max_utility_rounding_regret=regret,utility_is_not_probability=True)
        behavior.append({key:r[key] for key in ['target','native_presence_recall','decoded_presence_recall','native_presence_false_positive_rate',
            'decoded_presence_false_positive_rate','native_count_MAE','decoded_count_MAE','changed_count_units_vs_native',
            'added_prediction_rows_vs_native','removed_prediction_rows_vs_native']})
    text+='\n## 4. 召回、数量及局限\n\n'+table(list(behavior[0]),behavior)
    text+=('\n该分支没有生成新视觉语言语义。所有输出仍按原身份排序，HOTA变化只能来自数量改变及后续匹配/关联变化。'
        '大量弱候选的noisy-or可放大存在性，条件独立假设也可能使真实相关目标集合的数量被平均化；'
        '这些是模型假设和待验证解释，不以一次结果断言唯一因果。不根据本表改变任何温度/阈值/主结果。\n')
    from scripts.active_eval import bootstrap
    from scripts.presence_finish_a import references
    intervals={r['target']:bootstrap(r['path'],dict(references(r['target'],r['model']))['NATIVE'],r['target']) for r in own}
    atomic_json(RECORD / 'bootstrap.json',dict(comparisons=intervals,scope='post-hoc descriptive source-domain comparisons; not parameter selection',
        KITTI='four videos, points only',Dance='25 video clusters',SportsMOT='45 video clusters',samples=2000,seed=0))
    text+='\n后验描述性视频聚类bootstrap（不是确认性盲测/参数选择）：\n\n'+table(['target','delta_HOTA','CI95','clusters'],[dict(target=t,**v) for t,v in intervals.items()])
    with (RECORD / 'sports418.csv').open() as h:s418=list(csv.DictReader(h))
    text+='\n## 5. Sports418\n\n主结果全部466序列采用v4 GT，敏感性仅限制原418ID，内容不换回旧冲突模板。\n\n'+table(['model','method','HOTA','DetA','AssA','IDF1','IDSW','Dets'],s418)
    text+='\n## 6. 成本、隔离与恢复\n\n'+table(['source','parameters','function_evaluations','CPU_fit_wall_seconds','GPU_hours','GT_open_attempts'],source)
    cpu=[]
    for p in sorted((OUT / 'stage_A').glob('*/*/decode_receipt.json')):
        r=json.loads(p.read_text());evaluations=[]
        for method in ('PRES_TOPK','PRES_TOPK_SMOOTH'):
            completion=json.loads((p.parent / method / 'official_completion.json').read_text())
            evaluations.append(dict(method=method,CPU_wall_seconds=completion['wall_seconds']))
        cpu.append(dict(target=p.parent.parent.name,model=p.parent.name,decode_CPU_wall_seconds=r['wall_seconds'],evaluations=evaluations))
    atomic_json(RECORD / 'costs.json',dict(fits=source,CPU_decode_evaluation=cpu,
        coordinators=[dict(path=str(p),**json.loads(p.read_text())) for p in sorted((OUT / 'launchers').glob('*/completion.json'))],
        new_GPU_hours=0,new_teacher_calls=0,target_LoRA_updates=0))
    atomic_json(RECORD / 'probability_audit.json',dict(status='PASS',files=probability_audits))
    tests=json.loads((RECORD / 'functional_checks.json').read_text())['tests_passed']
    text+=('\n本阶段新增GPU小时=0、教师调用=0、目标LoRA更新=0；源凸拟合、头解码和官方评测均CPU。'
        '源训练仅允许自身域缓存标签，其他域数值GT打开被拒绝；解码启用runtime_guard并在独立进程封存预测后才评价。'
        '冻结分数与全部MOT候选复用，没有图像/PCD重导；原主报告、v3/v4和所有历史预测不覆写。'
        f'{tests}项定向测试和compileall通过；每行原MOT/完整episode覆盖与进程收尾见validation.json。'
        '动态规划小集合精确枚举通过，保存后概率归一化/存在性/均值核对见probability_audit.json。\n\n')
    if version=='v5':text+='v5新增拟合CPU时间=0、优化更新=0。两个源斜率的原始v4拟合/验证成本保留在v4成本表，不计为本轮新开销，也不把所复用的校准说成无监督或无源监督。\n\n'
    if gate['failed_targets']:
        text+='**本次固定构造未通过：'+', '.join(gate['failed_targets'])+'。未生成六跨域新头成绩，未启动B/C，不把未运行写为0 HOTA。**\n'
    else:text+='同域通过后还须完成九格A和发布；只有真实C1/C3 pilot及40GPU小时关卡之后才可展开B/C，不把源通过当研究成功。\n'
    text+='\n新缓存/预测在data1既有presence_active_20261102/redesign_'+version+'，经原data2 results/runs链接访问。机器路径与SHA在artifacts.json、selection.json；大文件不入Git。独立日志包logs.tar.gz包含阶段回执、全部官方详细表和失败尝试，旧归档不改。\n'
    (RECORD / 'REPORT.md').write_text(text)
    paths=sorted(p for p in OUT.rglob('*') if p.is_file() and (p.suffix in ('.log','.json','.jsonl') or p.name in ('pedestrian_detailed.csv','pedestrian_summary.txt')))
    records=[dict(path=str(p),archive_path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=file_sha(p)) for p in paths]
    archive_path=RECORD / 'logs.tar.gz'
    with tarfile.open(archive_path,'w:gz') as archive:
        for p,r in zip(paths,records):archive.add(p,arcname=r['archive_path'],recursive=False)
    for r in records:
        if file_sha(r['path'])!=r['sha256']:raise RuntimeError('A log owner is still writing')
    atomic_json(RECORD / 'logs.json',dict(path=str(archive_path),bytes=archive_path.stat().st_size,sha256=file_sha(archive_path),files=records))
    atomic_json(RECORD / 'report_receipt.json',dict(status=status,report_sha256=file_sha(RECORD / 'REPORT.md'),
        selection_sha256=file_sha(RECORD / 'selection.json'),gate_sha256=file_sha(RECORD / 'gate.json')))
    print(json.dumps(dict(status=status,report=str(RECORD / 'REPORT.md'))),flush=True)


if __name__=='__main__':main()
