"""Retain finished original v8 results without overriding its failed audit.

The unchanged original validator can separately audit its nine non-replay
methods. This scoped receipt is explicitly not an eleven-method acceptance.
The other mode archives finished records, including the original traceback.
Neither mode writes into the original v8 result directory or invokes a model.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha

PUBLIC = ROOT / 'results/presence_active/redesign_v8p/source_v8_retained'
MANIFEST = ROOT / 'results/presence_active/redesign_v8/execution.json'


def paths():
    m = json.loads(MANIFEST.read_text())
    return m, Path(m['public_root']), Path(m['active_root'])


def audit_online_subset():
    from scripts.active_v8_validate import validate
    m, original, active = paths()
    plan_path = original / 'registered_jobs.json'
    plan = json.loads(plan_path.read_text())
    assert len(plan['jobs']) == 11
    subset = dict(plan, jobs=[j for j in plan['jobs'] if j['method'] != 'SLOW_REPLAY'],
                  audit_scope='POST_HOC_NINE_NONREPLAY_METHODS_NOT_GLOBAL_ACCEPTANCE',
                  original_plan_path=str(plan_path), original_plan_sha256=file_sha(plan_path))
    assert len(subset['jobs']) == 9
    public = PUBLIC / 'online_subset_audit'
    public.mkdir(parents=True, exist_ok=True)
    atomic_json(public / 'registered_jobs.json', subset)
    result = validate(SimpleNamespace(public=public, active=active))
    # The old validator unconditionally prints a replay flag even on a subset
    # with no replay methods. Do not present that vacuous flag as evidence.
    result.update(audit_scope=subset['audit_scope'], original_global_audit_not_overridden=True,
        observed_replay_jobs=0, fixed_replay_presence_byte_exact='NOT_EVALUATED_NO_REPLAY',
        unchanged_validator_path=str(ROOT / 'scripts/active_v8_validate.py'),
        unchanged_validator_sha256=file_sha(ROOT / 'scripts/active_v8_validate.py'))
    atomic_json(public / 'validation.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)


def archive_failed_close():
    m, original, active = paths()
    plan = json.loads((original / 'registered_jobs.json').read_text())
    start = json.loads((original / 'execution_started.json').read_text())
    durable = Path(start['state_dir'])
    finished = json.loads((durable / 'completion.json').read_text())
    assert finished['exit_code'] == 1  # retained, never rewritten to zero
    diagnostic = json.loads((PUBLIC.parent / 'original_v8_audit_diagnostic.json').read_text())
    assert diagnostic['frozen_replay_strata_provenance_explains_failure']
    online = json.loads((PUBLIC / 'online_subset_audit/validation.json').read_text())
    assert online['status'] == 'PASS' and online['jobs'] == 9
    rows = []
    for job in plan['jobs']:
        folder = active / job['pair'] / job['method'] / 'rho_0.3'
        seal_path = folder / 'predictions_seal.json'
        seal = json.loads(seal_path.read_text())
        summary = json.loads((folder / 'offline_summary.json').read_text())
        assert summary['prediction_seal_sha256'] == file_sha(seal_path)
        assert seal['GT_open_attempts'] == 0
        rows.append(dict(pair=job['pair'], method=job['method'], rho=.3,
            HOTA=summary['HOTA_unrounded'], DetA=summary['DetA'], AssA=summary['AssA'],
            IDF1=summary['IDF1'], prediction_rows=seal['prediction_rows'],
            episodes=seal['episode_count'], path=str(folder),
            prediction_seal_sha256=file_sha(seal_path), original_supervisor_exit_code=1,
            global_audit_status='FAILED_NOT_OVERRIDDEN'))
    assert len(rows) == 11
    with (PUBLIC / 'official_results.csv').open('w') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    from scripts.active_archive import package, reassemble_archive
    from scripts.active_v6_close import verify_packaged_records
    execution = SimpleNamespace(active=active, public=PUBLIC, launchers=Path(m['launcher_root']))
    logs = package(execution, rows)
    verified = verify_packaged_records(PUBLIC, logs)
    for key in ('archive', 'raw_teacher_responses'):
        item = logs[key]
        if item.get('github_transport'):
            reassemble_archive(item['github_transport']['manifest'], Path(item['path'] + '.transport_verified'))
    record = dict(status='ALL_ELEVEN_OFFICIAL_RESULTS_RETAINED_GLOBAL_AUDIT_FAILED',
        supervisor_receipt=dict(path=str(durable / 'completion.json'),
                                sha256=file_sha(durable / 'completion.json'), **finished),
        unchanged_original_results_root=str(original), original_files_modified=False,
        methods=11, episodes=sum(r['episodes'] for r in rows),
        unchanged_original_validator_subset_methods=9, original_global_PASS_not_claimed=True,
        replay_counterfactual_coverage_conflicts=diagnostic['replay_counterfactual_current_coverage_mismatches'],
        replay_source_decision_coverage_valid=diagnostic['all_source_decision_coverage_valid'],
        source_presence_snapshots_byte_exact=diagnostic['all_six_presence_arrays_byte_exact'],
        diagnostic_path=str(PUBLIC.parent / 'original_v8_audit_diagnostic.json'),
        archive_verification=verified['status'], unpackaged_paths=logs['unpackaged_launcher_logs'],
        stage4_started=False, new_models_or_teacher_calls=0, raw_GT_files_read=False)
    atomic_json(PUBLIC / 'retention_status.json', record)
    text = '# 原v8已完成预测的隔离保留\n\n'
    text += '11项GPU工作与官方TrackEval已完成，但原supervisor退出码为1：旧审计对冻结回放使用了LoRA分支的反事实分层。原失败状态、原文件及全部预测不改，不声称11项全局审计通过，也不启动旧阶段4。\n\n'
    text += '原封不动的旧校验器在独立目录复查9项非回放方法通过；不是删除两项回放来宣称全局通过。回放26处当前分层冲突全部记录，350份六存在性数组快照、问题/回答/DS与源MINUS一致，实际源决策的三档覆盖有效；完整证据见../original_v8_audit_diagnostic.json。\n\n'
    text += '| pair | method | HOTA | episodes | global audit |\n| --- | --- | --- | --- | --- |\n'
    text += '\n'.join('| ' + ' | '.join(str(r[k]) for k in ('pair', 'method', 'HOTA', 'episodes', 'global_audit_status')) + ' |' for r in rows)
    text += '\n\n原始关闭traceback包含在日志归档中；logs.json列出全部已关闭日志、原始冻结教师输出与服务器大文件路径。archive_validation.json只证明归档字节及教师revision/BF16概率，不能取代失败的全局实验审计。未打包路径如非空保留在retention_status.json，不冒称已关闭。\n'
    (PUBLIC / 'REPORT.md').write_text(text)
    print(json.dumps(record), flush=True)


def snapshot_public_outputs():
    """Retain generated old tables/figures without adopting their verdicts."""
    import hashlib
    import tarfile
    _, original, _ = paths()
    retention_path = PUBLIC / 'retention_status.json'
    retention = json.loads(retention_path.read_text())
    assert retention['supervisor_receipt']['exit_code'] == 1
    files = sorted(p for p in original.iterdir()
                   if p.is_file() and p.suffix in ('.md', '.json', '.csv', '.png'))
    files += sorted((original / 'offline_diagnostics').glob('*.json'))
    assert (original / 'REPORT.md') in files and (original / 'main.csv') in files
    records = [dict(path=str(p), archive_path=str(p.relative_to(original)),
                    bytes=p.stat().st_size, sha256=file_sha(p)) for p in files]
    archive = PUBLIC / 'original_public_outputs.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=6) as output:
        for p, record in zip(files, records):
            output.add(p, arcname=record['archive_path'], recursive=False)
    with tarfile.open(archive, 'r:gz') as source:
        assert len(source.getmembers()) == len(records)
        for record in records:
            with source.extractfile(record['archive_path']) as entry:
                assert hashlib.sha256(entry.read()).hexdigest() == record['sha256']
    manifest = PUBLIC / 'public_outputs_manifest.json'
    snapshot = dict(status='ORIGINAL_PUBLIC_OUTPUTS_RETAINED_GLOBAL_AUDIT_FAILED',
        original_results_root=str(original), source_files=records,
        archive=dict(path=str(archive), bytes=archive.stat().st_size, sha256=file_sha(archive)),
        original_supervisor_exit_code=1, original_global_PASS_not_claimed=True,
        raw_GT_files_read=False, derived_GT_metrics_retained=True,
        new_model_or_teacher_calls=0, original_files_modified=False,
        user_owned_stage4_and_unrelated_files_excluded=True,
        note='Raw original verdict fields predate the failed global audit; they are retained, not endorsed.')
    atomic_json(manifest, snapshot)
    retention['original_public_outputs_snapshot'] = dict(path=str(manifest),
        sha256=file_sha(manifest), files=len(records), archive=snapshot['archive'])
    atomic_json(retention_path, retention)
    report = PUBLIC / 'REPORT.md'
    marker = '\n## 原生成表图的字节快照\n'
    if marker not in report.read_text():
        report.write_text(report.read_text() + marker + '\n'
            '原v8根目录的MD/JSON/CSV/PNG以及offline_diagnostics中的派生GT诊断，'
            '按原字节保留在original_public_outputs.tar.gz；每个源文件和归档的SHA见public_outputs_manifest.json。'
            '这些文件由原程序在全局审计失败前生成；其中原verdict/PASS字段只按原样保留，'
            '不覆盖supervisor退出1，不构成11项全局验收。未纳入旧stage4或其它未授权文件，旧目录不改。\n')
    print(json.dumps(dict(status=snapshot['status'], files=len(records),
        archive=snapshot['archive'], original_files_modified=False)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('audit-online-subset', 'archive', 'snapshot-public'))
    args = p.parse_args()
    if args.mode == 'audit-online-subset':
        audit_online_subset()
    elif args.mode == 'archive':
        archive_failed_close()
    else:
        snapshot_public_outputs()
