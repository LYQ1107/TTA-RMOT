"""Close the CPU-only v9 offline tier, without new fitting, teacher or GT."""
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.active_v9_inputs import PUBLIC,DATA,PAIRS,write_csv
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def close():
    queue=json.loads((PUBLIC/'queue_completion.json').read_text())
    decision=json.loads((PUBLIC/'selection.json').read_text())
    assert queue['status']=='V9_OFFLINE_COMPLETE' and decision['status']=='STOP_NO_ESTIMATOR_PASSED'
    inventory=json.loads((PUBLIC/'inventory_D3.json').read_text());expected={r['pair']:r for r in inventory['inputs']}
    checked=[];curvature=[];checkpoints=[]
    for method in ('E0','E1','E3'):
        for pair in PAIRS:
            seal_path=DATA/method/pair/'estimates_seal.json';seal=json.loads(seal_path.read_text())
            assert file_sha(seal['path'])==seal['sha256']
            result=json.loads(Path(seal['path']).read_text())
            assert result['GT_open_attempts']==result['GPU_calls']==result['teacher_calls']==0
            assert result['source_seal_sha256']==expected[pair]['seal_sha256']
            assert result['manifest_sha256']==file_sha(PUBLIC/'manifest.json')
            episodes=sum(len(v['queries']) for v in result['records'])
            assert episodes==expected[pair]['episodes']
            for video in result['records']:
                for q in video['queries']:
                    assert q['output_absent']==(q['evidence_absent'] or(q['low_tentative'] and q['recomputed_probability']<.1))
                if method=='E0':
                    assert video['reproduction']['parameter_max_error']<1e-9
                    assert video['reproduction']['probability_max_error']<1e-9
                    assert video['reproduction']['output_absence_exact']
                if method=='E1':
                    fit=video['after'];assert fit['optimizer']['success']
                    curvature.append(dict(pair=pair,video=video['video'],tau=fit['tau'],
                        tau_curvature=fit['laplace']['hessian'][4][4],
                        minimum_hessian_eigenvalue=min(fit['laplace']['eigenvalues']),
                        rho_laplace_std=fit['rho_laplace_std'],
                        optimizer_reported_success=fit['optimizer']['success'],
                        optimizer_message=fit['optimizer']['message']))
            last=sorted((DATA/method/pair).glob('stream_*.pkl'))[-1]
            checkpoints.append(dict(estimator=method,pair=pair,path=str(last),bytes=last.stat().st_size,sha256=file_sha(last)))
            checked.append(dict(estimator=method,pair=pair,episodes=episodes,videos=len(result['records']),
                path=seal['path'],sha256=seal['sha256'],seal_path=str(seal_path),seal_sha256=file_sha(seal_path),GT_open_attempts=0))
    for pair in PAIRS:
        assert json.loads((PUBLIC/f'E2_{pair}.json').read_text())['status']=='NOT_APPLICABLE'
    logs=[];(PUBLIC/'logs').mkdir(exist_ok=True)
    for method in ('E1','E2','E3'):
        source=DATA/(method+'_queue.log');destination=PUBLIC/'logs'/(method+'_queue.txt')
        completion=DATA/(method+'_queue_completion.json')
        assert json.loads(completion.read_text())['exit_code']==0
        shutil.copyfile(source,destination)
        assert file_sha(source)==file_sha(destination)
        logs.append(dict(estimator=method,path=str(destination),original_path=str(source),
            bytes=source.stat().st_size,sha256=file_sha(source),completion_path=str(completion)))
    write_csv(PUBLIC/'curvature_diagnostics.csv',curvature)
    atomic_json(PUBLIC/'artifacts.json',dict(estimates=checked,latest_CPU_checkpoints=checkpoints,logs=logs,
        E0_stdout_note='No separate original stdout file retained; all per-video fit, query decisions and exact reproduction receipts are preserved',
        new_teacher_or_GPU_calls=0))
    result=dict(status='V9_OFFLINE_COMPLETE_STOPPED_NO_ESTIMATOR_PASSED',selected=None,
        E2='NOT_APPLICABLE',completed_direction_replays=len(checked),
        video_replays=sum(r['videos'] for r in checked),query_episode_replays=sum(r['episodes'] for r in checked),
        fixed_dataset_episode_count=3506,GT_open_attempts_estimation=0,GPU_calls=0,teacher_calls=0,
        live_v9_implementation_started=False,MINUS_SLOW_v9_started=False,stage4_started=False,
        stage3_original_queue_not_cancelled=True,all_numerical_results_preserved=True)
    atomic_json(PUBLIC/'validation.json',dict(result,status='PASS_CLOSED_ESTIMATE_AND_B2_SCOPE_AUDIT'))
    atomic_json(PUBLIC/'final_status.json',result)
    text=(PUBLIC/'REPORT.md').read_text().rstrip()+'\n'
    text+='\n## 数值核查与解释边界\n\n'
    text+='所有L-BFGS-B收据都返回success，但这不等于已证明求到全局或局部MAP。尤其E1 C4终值tau=3.1791e-6，光滑负对数后验的tau方向曲率约−15.8638，完整Hessian也非正定。因此该处不能认定已找到局部最优，Laplace标准差留空；不能把它直接解释成只有数据不足。逐视频曲率见curvature_diagnostics.csv。没有在看到结果后更换初值、重启、改tol或重新拟合。其它C1/C2/C3/C6也分别违反固定门槛，所以本轮不选择E1；这不是“随机效应模型永远不可行”的证明。\n\n'
    text+='C2/C5错拒从647/540降到52/33，说明本配方能缓解部分拒绝；但Dance正确标空从347/330降到187/175，Sports仍错拒73/107。不能只展示KITTI的改善，也不跨数据集拼接E0/E1/E3最优格。\n\n'
    text+='固定问题流下，E1取消判定截断只改变C1–C6最终拒绝集合的0/0/3/2/0/1条表达，不能把当前主要失败简单归于±2截断；这项反事实不覆盖重新选题后的反馈。E1在Sports的tau接近0且mu再次接近上限，真实D1异质性没有被当前回答/DS/拟合充分恢复；教师错误率失配与相关回答仍是模型假设限制，不作单因归因。\n\n'
    text+='关闭审计通过：18方向重放、444视频、10518个查询episode（同一固定数据重复分析，不是独立重复实验），E2六项明确NA。预测不是新的RMOT HOTA；GT开发门槛也不是盲测。输入、输出、关闭CPU日志及最新恢复checkpoint的路径/SHA见artifacts.json和validation.json。E0没有另存原始stdout文件，完整逐视频数值收据与复现核验仍保留。\n\n'
    text+='本轮CPU诊断结束并停止新增v9工作；原v8阶段3继续，阶段4仍暂停，等待用户下一条指令。\n'
    (PUBLIC/'REPORT.md').write_text(text)
    print(json.dumps(result),flush=True);return result


if __name__=='__main__':close()
