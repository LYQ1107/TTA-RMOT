"""Post-completion validation of the four real frozen v8 pilot videos."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.active.execution import authorize_active
from flex_uda.presence.runtime import runtime_guard
from flex_uda.active.v8 import BeliefStateV8
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def audit():
    execution,_=authorize_active();attempts=runtime_guard()
    smoke=json.loads((execution.public/'gpu_smoke.json').read_text())
    assert smoke['status']=='V8_TWO_DIRECTION_TWO_VIDEO_PILOTS_PASS'
    results=[]
    for item in smoke['records']:
        root=Path(item['path'])
        for video in item['videos']:
            directory=root/'videos'/video
            receipt=json.loads((directory/'receipt.json').read_text())
            questions=json.loads((directory/'questions.json').read_text())
            with np.load(directory/'belief.npz') as b:
                nq=len(b['queries']);assert np.all(b['pseudo_labels'][~b['asked_mask']]==-1)
                assert np.all(b['covered_strata'][b['absent']]>=b['needed_strata'][b['absent']])
                assert not np.any(b['absent'] & b['low_tentative'])
            glances=[a for a in questions if a['kind']=='GLANCE']
            assert len({a['query_id'] for a in glances})==len(glances)
            assert len(glances)==nq
            assert len(questions)==receipt['questions']<=receipt['budget']
            assert len({(a['query_id'],a['track_id'],a['round']) for a in questions})==len(questions)
            assert all(a['kind']!='IDENTITY' or not a['low_tentative_before'] for a in questions)
            assert receipt['slow_GPU_seconds']==receipt['rescore_GPU_seconds']==0
            assert receipt['EB']['after']['converged']
            for r in range(5):
                with np.load(directory/f'presence_round{r}.npz') as b:
                    assert set(b.files)==set(BeliefStateV8.snapshot_fields)
                    assert all(b[k].shape==(nq,) for k in b.files)
            results.append(dict(pair=item['pair'],video=video,queries=nq,GLANCE_count=len(glances),
                questions=len(questions),budget=receipt['budget'],all_queries_glanced=True,
                EM_iterations=receipt['EB']['after']['iterations'],GT_open_attempts=0,
                receipt_sha256=file_sha(directory/'receipt.json'),belief_sha256=file_sha(directory/'belief.npz')))
    assert not attempts
    record=dict(status='PASS',records=results,videos=len(results),snapshot_files=5*len(results),
                GT_open_attempts=0,new_GPU_or_teacher_calls=0,
                teacher_by_pair={x['pair']:x['teacher'] for x in smoke['records']},
                sources='gpu_smoke.json plus closed video receipts; no new RMOT evaluation')
    atomic_json(execution.public/'pilot_audit.json',record)
    worker_root=execution.launchers/'workers_1793958029366294566'
    admission=json.loads((worker_root/'admission.json').read_text())
    started=json.loads((execution.public/'pilot_started.json').read_text())
    durable=Path(started['state_dir'])
    launch=json.loads((durable/'launch.json').read_text());end=json.loads((durable/'completion.json').read_text())
    assert end['exit_code']==0
    summary=dict(status='STAGE1_COMPLETE',videos=4,query_episodes=sum(r['queries'] for r in results),
        GLANCE_coverage=1.,snapshot_files=20,GT_open_attempts=0,
        physical_GPUs=[x['index'] for x in admission['selected']],admission=admission,
        logical_calls=sum(x['teacher']['logical_calls'] for x in smoke['records']),
        new_calls=sum(x['teacher']['new_calls'] for x in smoke['records']),
        cache_hits=sum(x['teacher']['cache_hits'] for x in smoke['records']),
        teacher_forward_seconds=sum(x['teacher']['forward_seconds'] for x in smoke['records']),
        scheduler_wall_seconds=(end['finished_time_ns']-launch['launched_time_ns'])/1e9,
        durable=str(durable),source_phase0_commit='122ad412a922025f44e8798829e5fd26005b32c1',
        formal_RMOT_not_evaluated=True,pilot_cost_in_final_resumed_stream=True)
    atomic_json(execution.public/'stage1_summary.json',summary)
    report=execution.public/'REPORT.md'
    report.write_text('# 阶段1最新状态\n\nC1/C2各2视频冒烟与独立收据审计通过，4视频/459个查询全部完成GLANCE；20个存在性快照齐全。GPU0、4按共享准入运行，未停止他人进程；GT打开0，没有进行RMOT评测。\n\n'
        +f"教师逻辑调用{summary['logical_calls']}，新调用{summary['new_calls']}、缓存{summary['cache_hits']}；实际新forward {summary['teacher_forward_seconds']}秒，调度壁钟{summary['scheduler_wall_seconds']}秒。这些调用会计入正式恢复后的流，不重复加成本。\n\n"
        +'下一步从同一checkpoint的next_video=2继续C1/C2；其余固定阶段2/3任务按11项清单执行。\n\n'+report.read_text())
    print(json.dumps(record),flush=True)


if __name__=='__main__':audit()
