"""Only the revised stage boundary and variable-cardinality prior are changed."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from flex_tta.runtime.pf_receipts import file_sha


def test_published_complete_gate_is_required_without_rewriting_old_gate(tmp_path,monkeypatch):
    from flex_uda.active.execution import authorize_active
    selection=tmp_path / 'selection.json';selection.write_text('{}')
    gate=tmp_path / 'gate.json'
    data=dict(status='PASS_SAME_DOMAIN',completed_full_A=False,failed_targets=[],same_domain=[{}, {}, {}],
        selection_path=str(selection),selection_sha256=file_sha(selection))
    gate.write_text(json.dumps(data))
    manifest=tmp_path / 'runtime.json'
    def seal():
        manifest.write_text(json.dumps(dict(gate_path=str(gate),gate_sha256=file_sha(gate),published_commit='prior-published-commit',
            active_root=str(tmp_path / 'new/active'),stage_A_root=str(tmp_path / 'new/stage_A'),
            public_root=str(tmp_path / 'new/public'),launcher_root=str(tmp_path / 'new/launchers'))))
    seal();monkeypatch.setenv('PRESENCE_EXECUTION_MANIFEST',str(manifest))
    with pytest.raises(PermissionError):authorize_active()
    data['completed_full_A']=True;gate.write_text(json.dumps(data));seal()
    paths,publication=authorize_active()
    assert paths.active==tmp_path / 'new/active' and publication=='prior-published-commit'
    assert json.loads(gate.read_text())==data


def test_video_prior_accepts_counts_above_ten(tmp_path,monkeypatch):
    import scripts.active_run as runner
    import flex_uda.active.execution as execution
    folder=tmp_path / 'features/Dance/V2-best/videos/v';folder.mkdir(parents=True)
    (folder / 'metadata.json').write_text(json.dumps(dict(video_id='v',frames=[1,2],
        queries=[dict(expression_id='q')],width=100,height=100)))
    decoded=tmp_path / 'v5/stage_A/Dance/V2-best';decoded.mkdir(parents=True)
    np.save(decoded / 'presence_logits.npy',np.zeros(2))
    counts=np.zeros((2,13));counts[:,12]=1;np.save(decoded / 'count_probs.npy',counts)
    monkeypatch.setattr(runner,'RUN',tmp_path)
    monkeypatch.setattr(execution,'active_paths',lambda:SimpleNamespace(stage_a=tmp_path / 'v5/stage_A'))
    bank=dict(observations=[(t,i) for t in (1,2) for i in range(12)])
    meta=dict(episodes=[dict(video_id='v',expression_id='q',start=0,stop=24)])
    result=runner.video_prior(meta,bank,'v','V2-best','Dance',np.ones(24))
    assert np.array_equal(result['expected'],[6.])
    assert result['track_means'].shape==(1,12)


def test_pilot_only_manifest_blocks_registered_queue_before_gpu_access(tmp_path,monkeypatch):
    import os,subprocess,sys
    selection=tmp_path / 'selection.json';selection.write_text('{}')
    gate=tmp_path / 'gate.json';gate.write_text(json.dumps(dict(status='PASS_SAME_DOMAIN',completed_full_A=True,
        failed_targets=[],same_domain=[{}, {}, {}],selection_path=str(selection),selection_sha256=file_sha(selection))))
    manifest=tmp_path / 'execution.json';manifest.write_text(json.dumps(dict(pilot_only=True,
        gate_path=str(gate),gate_sha256=file_sha(gate),published_commit='prior-published-commit',
        active_root=str(tmp_path / 'active'),stage_A_root=str(tmp_path / 'stage_A'),
        public_root=str(tmp_path / 'public'),launcher_root=str(tmp_path / 'launchers'))))
    env=dict(os.environ,PRESENCE_EXECUTION_MANIFEST=str(manifest),CUDA_VISIBLE_DEVICES='')
    root=__import__('pathlib').Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'scripts/active_launch.py','--phase','registered'],cwd=root,
        env=env,capture_output=True,text=True)
    assert result.returncode!=0 and 'Only first-video small pilots are authorized' in result.stderr
