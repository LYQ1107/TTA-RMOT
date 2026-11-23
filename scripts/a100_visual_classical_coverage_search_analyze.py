"""All new search predictions must be sealed before this manual-GT audit."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_coverage_search_launch import conditions
from scripts.a100_visual_classical_search_collect import EVIDENCE
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_sparse_observation_analyze import effect
from scripts.a100_visual_classical_sparse_observation_report import describe
from scripts.a100_visual_classical_sparse_statistics_reproduce import resolve_record,share_identical_statistics,methods
from scripts.a100_visual_classical_weighted_classical_analyze import ORDINARY

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'
COUNT_REFERENCES=('KD_MEAN','GRAPH','CAL','COVER_KD','COVER_JOINT','COMPLETE')


def references(slug):
    result={name:dict(file='confidence_teacher_parent_statistics.json',record=slug+'/'+source) for name,source in
        dict(COVER_KD='KD_MEAN',COVER_KD400='KD400',COVER_KD_FINAL='KD_FINAL',COVER_HARD400='HARD400',COVER_HARD_FINAL='HARD_FINAL').items()}
    result.update({name:dict(file='weighted_classical_parent_statistics.json',record=slug+'/'+source) for name,source in
        dict(COVER_GRAPH='GRAPH',COVER_CAL='CAL',COVER_JOINT='JOINT').items()})
    result.update({name:dict(file='direct_teacher_pcd_parent_statistics.json',record=slug+'/'+name) for name in ORDINARY})
    result['UNWEIGHTED_KD']=dict(file='direct_teacher_pcd_parent_statistics.json',record=slug+'/KD_MEAN')
    return result


def contrasts():
    names=('KD_MEAN','KD400','KD_FINAL','HARD400','HARD_FINAL','GRAPH','CAL','COMPLETE')+tuple(references('unused'))
    result={'JOINT_vs_'+name:('JOINT',name) for name in names}
    result.update(KD_MEAN_vs_COVER_KD=('KD_MEAN','COVER_KD'),GRAPH_vs_KD_MEAN=('GRAPH','KD_MEAN'),CAL_vs_KD_MEAN=('CAL','KD_MEAN'))
    for name in COUNT_REFERENCES:
        result['COUNT_OWN_JOINT_vs_'+name]=('JOINT__AT_JOINT_COUNT',name+'__AT_JOINT_COUNT')
        result['COUNT_BASELINE_JOINT_vs_'+name]=('JOINT__AT_'+name+'_COUNT',name+'__AT_'+name+'_COUNT')
    assert all(tuple(methods(label))==value for label,value in result.items())
    return result


def diagnosis(records):
    from collections import Counter
    from flex_uda.a100.mechanism import OUT as DIAG,metadata,PAIRS
    from scripts.a100_visual_classical_rgb_parity import source_spec
    result={};m=metadata(PAIRS['C4'][1]);source=source_spec('C4')[0]
    for record in records:
        root=Path(record['folder']);seed=record['seed'];names=('KD_MEAN','GRAPH','CAL','JOINT','COMPLETE','COVER_KD','COVER_JOINT')
        refs=dict(COVER_KD=OUT/'confidence_teacher/C4'/f'seed{seed}'/'KD_MEAN',COVER_JOINT=OUT/'weighted_classical/C4'/f'seed{seed}'/'JOINT')
        counts={name:Counter() for name in names};trade={name:Counter() for name in ('KD_MEAN','COVER_KD','COVER_JOINT')};coverage={name:Counter() for name in ('SEARCH','COVER')}
        for video,bank in sorted(m['videos'].items()):
            truth=np.load(DIAG/'offline_GT/C4'/video/'labels.npy')>=0;present=truth.any(1)
            masks={name:np.load((refs[name] if name in refs else root/name)/'videos'/video/'mask.npy') for name in names}
            correct={name:(mask&truth).any(1) for name,mask in masks.items()}
            for name,mask in masks.items():
                counts[name].update(queries=len(mask),recoverable_positive_queries=int(present.sum()),positive_queries_empty=int((present&~mask.any(1)).sum()),
                    positive_queries_without_correct_output=int((present&~correct[name]).sum()),TP=int((mask&truth).sum()),FP=int((mask&~truth).sum()),FN=int((~mask&truth).sum()))
            for name in trade:trade[name].update(rescued=int((present&~correct[name]&correct['JOINT']).sum()),regressed=int((present&correct[name]&~correct['JOINT']).sum()))
            with np.load(source/'source'/video/'sparse_inputs.npz') as a:tracks=a['tracks']
            observations=np.asarray(bank['observations']);membership=np.stack([truth[:,observations[:,1]==t].any(1) for t in tracks],axis=1)
            for name,path in dict(SEARCH=EVIDENCE/'C4/source'/video/'answers.json',COVER=source/'source'/video/'answers.json').items():
                answers=json.loads(path.read_text())['answers'];asked_correct=np.zeros(len(truth),bool);yes_correct=np.zeros(len(truth),bool)
                for a in answers:
                    q,t=a['local_query'],a['local_track'];positive=bool(membership[q,t]);asked_correct[q]|=positive;yes_correct[q]|=positive and a['yes_probability']>=.5
                coverage[name].update(queries=len(truth),positive_queries_unasked_correct=int((present&~asked_correct).sum()),positive_queries_without_correct_Yes=int((present&~yes_correct).sum()),logical_calls=len(answers))
        result['C4_seed'+str(seed)]=dict(candidate_membership={k:dict(v) for k,v in counts.items()},J_retrieval_versus={k:dict(v) for k,v in trade.items()},teacher_candidate_coverage={k:dict(v) for k,v in coverage.items()})
    return dict(conditions=result,postseal_manual_GT_only=True,not_additive_HOTA_attribution=True,coverage_does_not_guarantee_full_GT_track_quality=True)


def main():
    official=json.loads((OUT/'analysis/coverage_search_official.json').read_text());assert official['methods']==84
    data={};summary={};statistics={};cache={};pairs=contrasts()
    for record in conditions():
        seed=record['seed'];slug='C4_seed'+str(seed);root=Path(record['folder']);raw=json.loads((root/'completion.json').read_text());counts=json.loads((root/'countmatched/completion.json').read_text())
        assert raw['GT_open_attempts']==raw['new_teacher_calls']==raw['new_encoder_calls']==0
        assert raw['logical_teacher_calls']==1160 and not any(counts['CountSelf_changed_query_files'].values())
        paths={m['method']:root/m['method'] for m in raw['methods']};paths.update({m['method']:root/'countmatched'/m['method'] for m in counts['methods']})
        data[slug]={name:read_stats(path) for name,path in paths.items()}
        for name,row in data[slug].items():statistics[slug+'/'+name]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],numerator=row['parent_stats'][0].tolist(),denominator=row['parent_stats'][1].tolist())
        for name,ref in references(slug).items():
            row=resolve_record(PUBLIC,ref['file'],ref['record'],cache);data[slug][name]=dict(parent_videos=row['parents'],HOTA=row['HOTA'],parent_stats=(np.asarray(row['numerator']),np.asarray(row['denominator'])))
            statistics[slug+'/'+name]=dict(reference=ref)
        summary[slug]=dict(pair='C4',seed=seed,raw_HOTA={n:r['HOTA'] for n,r in data[slug].items() if '__AT_' not in n},comparisons={name:effect([data[slug][a]],[data[slug][b]]) for name,(a,b) in pairs.items()},CountSelf_changed_query_files=counts['CountSelf_changed_query_files'])
    keys=['C4_seed'+str(s) for s in (0,17,42)]
    repeated=dict(COVERAGE_SEARCH_C4=dict(condition_keys=keys,neural_seeds=[0,17,42],QA_seed=0,
        mean_HOTA={n:float(np.mean([summary[k]['raw_HOTA'][n] for k in keys])) for n in summary[keys[0]]['raw_HOTA']},
        comparisons={name:effect([data[k][a] for k in keys],[data[k][b] for k in keys]) for name,(a,b) in pairs.items()}))
    diagnostic=diagnosis(conditions());evidence=json.loads((EVIDENCE/'C4/completion.json').read_text())
    result=dict(status='COVERAGE_SEARCH_SAME_BUDGET_ANALYZED',conditions=summary,repeated=repeated,query_diagnosis=diagnostic,teacher_evidence=evidence,
        primary_rule='SameCOVERwithinqueryorder;initialqueryfloor;PRESUPPOSEDallocatebyobservedNo-product/(querycalls+1);CHECKLISTsameCOVER',
        all84_new_outputs_sealed_before_manual_GT=True,same1160_total_teacher_budget=True,ordinary_heads_get_exact_same_SEARCH_answers=True,
        C1_CHECKLIST_quality_inherited_exactly_from_shared_weighted_head=True,all_development_data=True,independent_blind_validation=False,
        uncertainty='Fixedsealedoriginal-parent/shared-seeddescriptivebootstrap,nopipelinerefitordevelopment/multiplicitycorrection.',
        quality_evidence_preparation_not_actual_cold_cost=True,goal_status='ACTIVE_REQUIRES_ROOT_SCIENTIFIC_REVIEW')
    atomic_json(OUT/'analysis/coverage_search_analysis.json',result);(PUBLIC/'coverage_search_analysis.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    compact,_=share_identical_statistics(statistics,'coverage_search_parent_statistics.json');(PUBLIC/'coverage_search_parent_statistics.json').write_text(json.dumps(compact,separators=(',',':'))+'\n')
    lines=['TTA-RMOT：同预算、教师反馈驱动的覆盖搜索','2026-11-22','',
        '同原有COVER逐查询候选顺序，先每查询检查一次；PRESUPPOSED剩余预算按已观察No概率乘积/(已用query预算+1)重新分配。CHECKLIST保持原均匀query预算。',
        '这是搜索优先级代理，不是经校准的存在性后验或独立教师证据定理。不得借低置信度声称目标不存在。',
        '总预算C4仍1160，新增缺失问题真实Qwen前向；私有缓存复用是质量证据准备，不是零成本冷启动。普通充分软/硬KD也取得完全相同新回答。',
        '三个神经种子、270头拟合、27原始+57双向数量共84新输出封存后本组GT。','']
    for pair,row in repeated.items():
        lines += [pair,'三种子平均HOTA：'+json.dumps(row['mean_HOTA'])]
        for name,e in row['comparisons'].items():lines.append(name+'：'+describe(e))
    lines += ['封存后覆盖/检索/TPFPFN：'+json.dumps(diagnostic,separators=(',',':')),
        '搜到正确候选不保证正确传播；需要同时检验普通头下选题贡献、同问答经典模块贡献和双向数量效应。',
        'C1按任务契约未改变，复用原同头质量结果不是新增独立验证。所有数据均为开发数据。',
        '如新配方保留，需自己的真实端到端成本，不能拼接旧wall。Goal不自动完成。']
    (ROOT/'docs/TTA_RMOT_同预算覆盖搜索与普通蒸馏_机制实证_20261122.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(event='COVERAGE_SEARCH_ANALYSIS_COMPLETE',repeated=repeated,goal_status='ACTIVE')),flush=True)


if __name__=='__main__':main()
