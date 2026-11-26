"""Strong teacher-confidence weighting audit; all54 outputs sealed first."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_confidence_teacher_launch import conditions
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_sparse_observation_analyze import effect
from scripts.a100_visual_classical_sparse_observation_report import describe
from scripts.a100_visual_classical_sparse_statistics_reproduce import resolve_record,share_identical_statistics

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'
WEIGHTED=('KD400','KD_FINAL','KD_MEAN','HARD400','HARD_FINAL')


def main():
    official=json.loads((OUT/'analysis/confidence_teacher_official.json').read_text());assert official['methods']==54
    while not (PUBLIC/'stable_topology_statistical_reproduction.json').is_file():
        state=OUT/'supervisor/stable_topology_state.json'
        if state.is_file() and json.loads(state.read_text())['status'].startswith('FAILED'):
            raise RuntimeError('Stable topology dependency failed; preserve weighted predictions and diagnose it')
        time.sleep(5)
    assert json.loads((PUBLIC/'stable_topology_statistical_reproduction.json').read_text())['status'].startswith('PASS_')
    comparisons={'JOINT_vs_'+name:('JOINT',name) for name in WEIGHTED}
    comparisons.update({'STABLE_JOINT_vs_'+name:('STABLE_JOINT',name) for name in WEIGHTED})
    comparisons.update(KD_MEAN_vs_UNWEIGHTED_KD=('KD_MEAN','UNWEIGHTED_KD'),JOINT_vs_UNWEIGHTED_KD=('JOINT','UNWEIGHTED_KD'),
        STRICT_JOINT_vs_KD_MEAN=('STRICT_JOINT','KD_MEAN'),
        COUNT_OWN_JOINT_vs_KD_MEAN=('JOINT__AT_JOINT_COUNT','KD_MEAN__AT_JOINT_COUNT'),
        COUNT_BASELINE_JOINT_vs_KD_MEAN=('JOINT__AT_KD_MEAN_COUNT','KD_MEAN__AT_KD_MEAN_COUNT'))
    data={};summary={};statistics={};cache={}
    for record in conditions():
        pair,seed=record['pair'],record['seed'];slug=pair+f'_seed{seed}';root=Path(record['folder'])
        raw=json.loads((root/'completion.json').read_text());counts=json.loads((root/'countmatched/completion.json').read_text())
        assert raw['GT_open_attempts']==raw['new_teacher_calls']==raw['new_encoder_calls']==0
        assert raw['source_logit_residual_disabled'] and raw['logical_teacher_calls']==(1160 if pair=='C4' else 1103)
        assert not any(counts['CountSelf_changed_query_files'].values())
        paths={m['method']:root/m['method'] for m in raw['methods']};paths.update({m['method']:root/'countmatched'/m['method'] for m in counts['methods']})
        data[slug]={name:read_stats(path) for name,path in paths.items()}
        for name,row in data[slug].items():statistics[slug+'/'+name]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],numerator=row['parent_stats'][0].tolist(),denominator=row['parent_stats'][1].tolist())
        references=dict(JOINT=('direct_teacher_pcd_parent_statistics.json','JOINT'),UNWEIGHTED_KD=('direct_teacher_pcd_parent_statistics.json','KD_MEAN'),STRICT_JOINT=('predictive_topology_parent_statistics.json','SELECTED_JOINT'),STABLE_JOINT=('stable_topology_parent_statistics.json','SELECTED_JOINT'))
        for name,(file,source) in references.items():
            row=resolve_record(PUBLIC,file,slug+'/'+source,cache);data[slug][name]=dict(parent_videos=row['parents'],HOTA=row['HOTA'],parent_stats=(np.asarray(row['numerator']),np.asarray(row['denominator'])))
            statistics[slug+'/'+name]=dict(reference=dict(file=file,record=slug+'/'+source))
        fits=[r['fits'][label] for r in raw['receipts'] for label in ('SOFT','HARD')]
        summary[slug]=dict(pair=pair,seed=seed,raw_HOTA={name:row['HOTA'] for name,row in data[slug].items() if '__AT_' not in name},
            comparisons={name:effect([data[slug][a]],[data[slug][b]]) for name,(a,b) in comparisons.items()},
            actual_head_fits=raw['actual_head_fits'],max_steps=max(r['steps'] for r in fits),max_weighted_teacher_KL=max(r['teacher_KL'] for r in fits),
            fits_exceeding_weighted_KL_tolerance=sum(not r['teacher_fit_within_0_005'] for r in fits),
            CountSelf_changed_query_files=counts['CountSelf_changed_query_files'],fitting_component_wall_seconds=sum(r['wall_seconds'] for r in fits))
    repeated={}
    for pair in ('C4','C1'):
        keys=[pair+f'_seed{s}' for s in (0,17,42)]
        repeated['CONFIDENCE_'+pair]=dict(condition_keys=keys,neural_seeds=[0,17,42],QA_seed=0,
            mean_HOTA={name:float(np.mean([summary[k]['raw_HOTA'][name] for k in keys])) for name in summary[keys[0]]['raw_HOTA']},
            comparisons={name:effect([data[k][a] for k in keys],[data[k][b] for k in keys]) for name,(a,b) in comparisons.items()})
    result=dict(status='CONFIDENCE_WEIGHTED_ORDINARY_TEACHER_ANALYZED',conditions=summary,repeated=repeated,
        weight_rule='1+3*abs(2*rawteacherp-1),sameasgraphanchorprecision;weightedCE/entropy/KLnormalizedbytotalweight',
        baseline='Native768D directhead,sourcez0;samefullprefixnormalization,2layer64GELU,AdamW,4000min/16000max,softclip.02/.98 andhard;fixed400/final/softmean',
        all54_outputs_sealed_before_new_GT=True,actual_head_fits=sum(r['actual_head_fits'] for r in summary.values()),
        new_teacher_calls=0,new_encoder_calls=0,all_development_data=True,independent_blind_validation=False,
        uncertainty='Fixedsealedpredictionoriginal-parent/shared-neural-seed descriptivebootstrap;no repeated-development/multiplicity correction or pipeline refit.',
        fitting_component_time_not_independent_cold_deployment=True,goal_status='ACTIVE_REQUIRES_ROOT_REVIEW')
    atomic_json(OUT/'analysis/confidence_teacher_analysis.json',result)
    (PUBLIC/'confidence_teacher_analysis.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    compact,_=share_identical_statistics(statistics,'confidence_teacher_parent_statistics.json')
    (PUBLIC/'confidence_teacher_parent_statistics.json').write_text(json.dumps(compact,separators=(',',':'))+'\n')
    lines=['TTA-RMOT：教师置信度加权的普通蒸馏公平性补查','2026-11-22','',
        '本方法图锚点使用教师可靠性权重，必须给普通神经蒸馏同样机会。权重固定1+3|2p-1|，不通过目标GT选择阈值。',
        '其它输入、结构、种子、前缀、初始化、训练和解码不变。软/硬均充分训练，保留400步、最终和软均值。',
        '6条件420小头拟合，30原始+24双向数量共54输出先封存后本研究GT；0新增教师/主干前向。',
        'JOINT是此前固定局部原生组合，STRICT_JOINT是已失败晋级的严格风险选择。STABLE_JOINT是独立固定ONE_SE主规则，不代表已通过本身验收。',
        'STABLE_JOINT对五种加权KD的跨研究比较在本轮任何GT前补充登记，等待两组完整复算后才分析；不根据文件完成先后决定是否报告。','']
    for pair,row in repeated.items():
        lines += [pair,'平均HOTA：'+json.dumps(row['mean_HOTA'])]
        for name,e in row['comparisons'].items():lines.append(name+'：'+describe(e))
    lines += ['逐种子：'+json.dumps(summary,separators=(',',':')),
        '若加权普通KD消除了旧优势，必须如实记录、继续研究，不改名或弱化比较对象。',
        '拟合时间只是复用已有数据的组件时间，不冒充完整冷启动。已有四卡真实时间只配当时未加权KD/固定局部配方。',
        '全部数据已用于开发，不能由此声称独立新域泛化；目标人工GT只在全部预测封存后评估/诊断。Goal不自动关闭。']
    (ROOT/'docs/TTA_RMOT_教师置信度加权蒸馏_公平性补查_20261122.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(event='CONFIDENCE_TEACHER_ANALYSIS_COMPLETE',repeated=repeated,goal_status='ACTIVE')),flush=True)


if __name__=='__main__':main()
