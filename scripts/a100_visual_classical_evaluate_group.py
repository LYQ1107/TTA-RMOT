"""Parallel official evaluation of distinct immutable folders in a new group."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.a100_visual_classical_prepare import OUT
from flex_tta.runtime.pf_receipts import atomic_json


def job(arguments):
    group,pair,method=arguments
    from flex_uda.a100.evaluation_reuse import evaluate
    folder=OUT/group/pair/method
    return dict(pair=pair,method=method,folder=str(folder),evaluation=evaluate(folder,pair))


def main(group,primary,workers):
    begin=time.perf_counter();tables={}
    if group in ('facility_prefix','facility_countmatched'):
        for pair in ('C4','C1'):
            evidence=OUT/'facility_evidence'/pair/'completion.json'
            while not evidence.is_file():time.sleep(15)
            sealed=json.loads(evidence.read_text());assert sealed['GT_open_attempts']==0
            assert sealed['status']=='FACILITY_REAL_FROZEN_EVIDENCE_COMPLETE'
    if group.startswith('strong_field'):
        for pair in ('C4','C1'):
            evidence=OUT/'field_evidence'/pair/'completion.json'
            while not evidence.is_file():time.sleep(15)
            sealed=json.loads(evidence.read_text());assert sealed['GT_open_attempts']==0
            assert sealed['status']=='SEALED_REAL_FIELD_TEACHER_EVIDENCE'
    for pair in ('C4','C1'):
        record=json.loads((OUT/group/pair/'completion.json').read_text())
        assert record['GT_open_attempts']==0
        tables[pair]=[r['method'] for r in record['methods']]
    from flex_uda.a100.evaluation_reuse import bootstrap
    # Exact-equivalence validation is archived in docs. A clean checkout must
    # not depend on a one-off local test artifact: an empty cache simply runs
    # every query through the original vendor matcher and metric functions.
    episode_cache=bootstrap()
    methods=sorted(set(tables['C4'])|set(tables['C1']),key=lambda m:(m!=primary,m))
    tasks=[];results=[];identical={};scheduled={};aliases=[]
    # Evaluation-only reuse: an exact prediction signature may share official
    # sufficient statistics. No score, label, or method parameter is changed.
    for family in ('predictions','convergence','frame_convergence','refinement_v2','semantic_memory_v3',
                   'strict_prefix','exemplar_prefix','continuity_prefix','prefix_countmatched','colour_prefix','colour_countmatched','facility_prefix','facility_countmatched',
                   'peer_prefix','strong_cover_seed0','strong_uniform_seed0',group):
        for pair in ('C4','C1'):
            for path in (OUT/family/pair).glob('*/official_completion.json'):
                seal_path=path.parent/'predictions_seal.json'
                if not seal_path.is_file():continue
                seal=json.loads(seal_path.read_text())
                if seal['GT_open_attempts']:continue
                signature=tuple((q['video'],q['expression'],q['prediction_sha256']) for q in seal['episodes'])
                identical[pair,signature]=(path.parent,json.loads(path.read_text()))
    for method in methods:
        for pair in ('C4','C1'):
            if method not in tables[pair]:continue
            folder=OUT/group/pair/method;seal=json.loads((folder/'predictions_seal.json').read_text())
            signature=tuple((q['video'],q['expression'],q['prediction_sha256']) for q in seal['episodes'])
            reference=identical.get((pair,signature))
            if reference and reference[0]!=folder:
                record=dict(reference[1],identical_prediction_reference=str(reference[0]))
                atomic_json(folder/'official_equivalent.json',record)
                results.append(dict(pair=pair,method=method,folder=str(folder),evaluation=record))
            elif (pair,signature) in scheduled:
                aliases.append((pair,method,folder,scheduled[pair,signature]))
            else:
                tasks.append((group,pair,method));scheduled[pair,signature]=folder
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:results.extend(pool.map(job,tasks))
    # Equivalent folders scheduled together must wait for their canonical run.
    # Reuse requires identical final MOT bytes for every query, not similar HOTA.
    for pair,method,folder,reference in aliases:
        record=dict(json.loads((reference/'official_completion.json').read_text()),identical_prediction_reference=str(reference))
        atomic_json(folder/'official_equivalent.json',record)
        results.append(dict(pair=pair,method=method,folder=str(folder),evaluation=record))
    atomic_json(OUT/group/'official_evaluation.json',dict(status='PASS_ALL_REGISTERED_GROUP_METHODS',group=group,
        results=results,wall_seconds=time.perf_counter()-begin,workers=workers,parallel_disjoint_folders=True,
        exact_official_episode_cache=episode_cache))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--primary',default='RESET_MEMORY_HMM');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();main(a.group,a.primary,a.workers)
