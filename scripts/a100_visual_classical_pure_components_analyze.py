"""Post-seal separation of raw teacher anchors, edges and cue correspondence."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_pure_components_launch import conditions
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_sparse_observation_analyze import effect
from scripts.a100_visual_classical_sparse_observation_report import describe
from scripts.a100_visual_classical_sparse_statistics_reproduce import resolve_record,share_identical_statistics,methods

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'
PRIMARY='PROTOCOL_JOINT'


def references(slug):
    result={n:dict(file='protocol_calibration_parent_statistics.json',record=slug+'/'+n) for n in ('KD','PROTOCOL_JOINT','PROTOCOL_GRAPH','FINAL_CAL')}
    return result


def contrasts():
    result={PRIMARY+'_vs_'+n:(PRIMARY,n) for n in ('KD','ANCHOR_ONLY','ANCHOR_CAL','SHUFFLED_CAL','SHUFFLED_JOINT','FINAL_CAL','PROTOCOL_GRAPH')}
    result.update(PROTOCOL_GRAPH_vs_ANCHOR_ONLY=('PROTOCOL_GRAPH','ANCHOR_ONLY'),ANCHOR_ONLY_vs_KD=('ANCHOR_ONLY','KD'),
        ANCHOR_CAL_vs_KD=('ANCHOR_CAL','KD'),FINAL_CAL_vs_SHUFFLED_CAL=('FINAL_CAL','SHUFFLED_CAL'))
    for n in ('ANCHOR_CAL','SHUFFLED_JOINT'):
        result['COUNT_OWN_'+PRIMARY+'_vs_'+n]=(PRIMARY+'__AT_'+PRIMARY+'_COUNT',n+'__AT_'+PRIMARY+'_COUNT')
        result['COUNT_BASELINE_'+PRIMARY+'_vs_'+n]=(PRIMARY+'__AT_'+n+'_COUNT',n+'__AT_'+n+'_COUNT')
    assert all(tuple(methods(k))==v for k,v in result.items())
    return result


def main():
    official=json.loads((OUT/'analysis/pure_components_official.json').read_text());assert official['methods']==66
    data={};summary={};statistics={};cache={};pairs=contrasts()
    for record in conditions():
        pair,seed=record['pair'],record['seed'];slug=pair+f'_seed{seed}';root=Path(record['folder'])
        raw=json.loads((root/'completion.json').read_text());counts=json.loads((root/'countmatched/completion.json').read_text())
        assert raw['GT_open_attempts']==raw['new_teacher_calls']==raw['new_encoder_calls']==raw['new_head_fits']==0
        assert not any(counts['CountSelf_changed_query_files'].values())
        paths={m['method']:root/m['method'] for m in raw['methods']};paths.update({m['method']:root/'countmatched'/m['method'] for m in counts['methods']})
        data[slug]={name:read_stats(path) for name,path in paths.items()}
        for name,row in data[slug].items():statistics[slug+'/'+name]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],numerator=row['parent_stats'][0].tolist(),denominator=row['parent_stats'][1].tolist())
        for name,ref in references(slug).items():
            row=resolve_record(PUBLIC,ref['file'],ref['record'],cache)
            data[slug][name]=dict(parent_videos=row['parents'],HOTA=row['HOTA'],parent_stats=(np.asarray(row['numerator']),np.asarray(row['denominator'])))
            statistics[slug+'/'+name]=dict(reference=ref)
        summary[slug]=dict(pair=pair,seed=seed,raw_HOTA={n:r['HOTA'] for n,r in data[slug].items() if '__AT_' not in n},
            comparisons={n:effect([data[slug][a]],[data[slug][b]]) for n,(a,b) in pairs.items()},
            primary_CAL_replay_max_error=max(max(r['primary_and_CAL_replay_max_error'].values()) for r in raw['receipts']),
            CountSelf_changed_query_files=counts['CountSelf_changed_query_files'])
    repeated={}
    for pair in ('C4','C1'):
        keys=[pair+f'_seed{s}' for s in (0,17,42)]
        repeated['PURE_COMPONENTS_'+pair]=dict(condition_keys=keys,neural_seeds=[0,17,42],QA_seed=0,
            mean_HOTA={n:float(np.mean([summary[k]['raw_HOTA'][n] for k in keys])) for n in summary[keys[0]]['raw_HOTA']},
            comparisons={n:effect([data[k][a] for k in keys],[data[k][b] for k in keys]) for n,(a,b) in pairs.items()})
    checks={}
    for pair in ('C4','C1'):
        e=repeated['PURE_COMPONENTS_'+pair]['comparisons']['PROTOCOL_JOINT_vs_ANCHOR_ONLY']
        checks[pair+'_beats_simple_teacher_anchor_overwrite']=all(v>0 for v in e['per_neural_seed_delta']) and e['parent_and_neural_seed_sensitivity_CI95'][0]>0
    checks['C1_pool_edges_beyond_raw_anchors']=repeated['PURE_COMPONENTS_C1']['comparisons']['PROTOCOL_JOINT_vs_ANCHOR_CAL']['parent_and_neural_seed_sensitivity_CI95'][0]>0
    checks['C4_visual_correspondence_beyond_shuffled']=repeated['PURE_COMPONENTS_C4']['comparisons']['PROTOCOL_JOINT_vs_SHUFFLED_JOINT']['parent_and_neural_seed_sensitivity_CI95'][0]>0
    result=dict(status='PURE_ANCHOR_EDGE_VISUAL_CORRESPONDENCE_ANALYZED',conditions=summary,repeated=repeated,attribution_checks=checks,
        current_primary_recipe_unchanged=True,all66_sealed_before_this_study_GT=True,new_teacher_encoder_neural_head_fits=0,
        control_definitions=dict(ANCHOR_ONLY='Same head, known raw teacher anchors and same query support but zero graph edges; .1 source precision and logit clipping unchanged',
            ANCHOR_CAL='ANCHOR_ONLY plus identical original cue PAVA',SHUFFLED_CAL='Same FINAL, fixed withinquery cue permutation and teacher-only crossfit refitting',
            SHUFFLED_JOINT='Same protocol-pooled unary, same cue permutation/refitting; same cap/folds/sample budget'),
        posthoc_mechanism_audit_not_blind_confirmation=True,all_development_data=True,
        no_primary_promotion_or_parameter_change_from_these_controls=True,
        uncertainty='Fixed original-parent/shared-neural-seed descriptive intervals; no full pipeline refit or development/multiplicity correction',
        goal_status='ACTIVE_REQUIRES_ROOT_REVIEW_AND_ACTUAL_COLD')
    atomic_json(OUT/'analysis/pure_components_analysis.json',result)
    (PUBLIC/'pure_components_analysis.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    compact,_=share_identical_statistics(statistics,'pure_components_parent_statistics.json')
    (PUBLIC/'pure_components_parent_statistics.json').write_text(json.dumps(compact,separators=(',',':'))+'\n')
    lines=['TTA-RMOT：教师答案回填、跨候选图边与视觉对应关系的纯净拆解','2026-11-22','',
        '主配方和正在执行的四卡冷流不改动。这是已知主质量结果后的机制补查，不是独立盲测。所有新控制与数量输出固定、66份全封存后本组GT。',
        'ANCHOR_ONLY保留相同已问节点教师回填、源精度、概率裁剪、查询支持范围，只把图边全部置零。ANCHOR_CAL随后执行同一PAVA。不能把图/回填总差全部记作纯图边贡献。',
        'SHUFFLED把每个查询的视觉颜色线索在可用候选中按固定hash排列，再用同样TrackID排除和前缀教师证据重新拟合PAVA，保留原容量、odds上限、回答与预算。',
        '零新增教师/编码器/神经头拟合；有真实CPU图/PAVA计算，回放墙钟不当新物理成本。','']
    for pair,row in repeated.items():
        lines += [pair,'三种子均值：'+json.dumps(row['mean_HOTA'])]
        lines += [name+'：'+describe(e) for name,e in row['comparisons'].items()]
    lines += ['纯净归因检查：'+json.dumps(checks),
        '若不成立，只能保留有证据的模块整体贡献，不能声称该内部成分已被证明；也不能把最高的控制事后提升为新主方法。',
        '数量投影两方向全部披露。数据全DEV，协议与域在现有任务中重合，不是独立协议泛化证明。Goal不自动结项。']
    (ROOT/'docs/TTA_RMOT_教师回填图边与视觉对应_纯净归因_20261122.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(event='PURE_COMPONENTS_ANALYZED',checks=checks,goal_status='ACTIVE')),flush=True)


if __name__=='__main__':main()
