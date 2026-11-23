"""GT-free numerical equivalence of the rank-one solve and fixed readouts."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import generation_guard, metadata, PAIRS
from flex_uda.active.acquisition import exclusive_group
from flex_uda.active.topology_selection import graph_matrices, apply_graph
from flex_uda.active.protocol_calibration import global_pool, contract_graph, nonempty_baseline
from flex_uda.active.semantic_colour import appearance_cues, crossfit_predictions, correct_belief, append_history
from scripts.a100_visual_classical_semantic_colour import empty_history
from scripts.a100_visual_classical_strict_prefix import plain
from scripts.a100_visual_classical_protocol_calibration import SOURCES
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_rgb_parity import source_spec


def main():
    guard = generation_guard(); rows = []
    for pair in ('C4', 'C1'):
        _, target = PAIRS[pair]; m = metadata(target); source = source_spec(pair)[0]
        protocol = json.loads((OUT/'preregistration_v1.json').read_text())['protocol'][pair]
        history = empty_history()
        for video, bank in sorted(m['videos'].items()):
            texts = [e['query_text'] for e in m['episodes'] if e['video_id'] == video]
            with np.load(source/'source'/video/'sparse_inputs.npz') as a:
                pcd, colours, tracks = a['pcd'], a['colours'], a['tracks']
            with np.load(OUT/'confidence_teacher'/pair/'seed0/videos'/video/'coarse_scores.npz') as a:
                final, mean = a['KD_FINAL'], a['KD_MEAN']
            obs = np.asarray(bank['observations']); index = {int(t):i for i,t in enumerate(tracks)}
            columns = np.asarray([index[int(t)] for _, t in obs]); lengths = np.asarray([(obs[:,1] == t).sum() for t in tracks])
            cue, available, families, _ = appearance_cues(colours, texts, lengths); teacher = np.full(cue.shape, np.nan)
            for answer in json.loads((source/'source'/video/'answers.json').read_text())['answers']:
                teacher[int(answer['local_query']),int(answer['local_track'])] = answer['yes_probability']
            geometry, local, global_, valid, supported = graph_matrices(pcd, colours, texts)
            dense = apply_graph(final, teacher, geometry, global_, supported)
            rank_one = global_pool(final, teacher, global_, valid, supported)
            error = float(np.max(np.abs(dense-rank_one)))
            exclusive = exclusive_group(texts, target)
            changed = int(np.count_nonzero(plain(dense[:,columns],exclusive) != plain(rank_one[:,columns],exclusive)))
            assert error < 1e-12 and changed == 0, (pair,video,error,changed)
            final_mask = np.load(OUT/'confidence_teacher'/pair/'seed0/KD_FINAL/videos'/video/'mask.npy')
            assert np.array_equal(plain(final[:,columns],exclusive), final_mask)
            assert np.array_equal(contract_graph(final,rank_one,'PRESUPPOSED'), final)
            assert np.array_equal(contract_graph(final,rank_one,'CHECKLIST'), rank_one)
            assert np.array_equal(nonempty_baseline(final,'CHECKLIST'), final)
            probability, base, active, _ = crossfit_predictions(cue, available, families, teacher, tracks, video, history)
            with np.load(OUT/'weighted_classical'/pair/'seed0/videos'/video/'coarse_scores.npz') as a:
                cal_error = float(np.max(np.abs(correct_belief(mean,probability,base,active)-a['CAL'])))
            # The global control was already executed on the same mean head.
            mean_global = global_pool(mean,teacher,global_,valid,supported)
            with np.load(OUT/'optional_transfer'/pair/'seed0/videos'/video/'coarse_scores.npz') as a:
                global_error = float(np.max(np.abs(correct_belief(mean_global,probability,base,active)-a['GLOBAL_JOINT'])))
            assert cal_error == global_error == 0, (pair,video,cal_error,global_error)
            append_history(history,cue,available,families,teacher)
            rows.append(dict(pair=pair,video=video,rank_one_probability_max_error=error,rank_one_decision_changes=changed,
                mean_CAL_replay_error=cal_error,mean_GLOBAL_replay_error=global_error))
    toy = np.asarray([[.2,.3],[0.,0.],[1.,0.]],dtype=np.float32)
    expected = np.asarray([[.2/.44,.3/.44],[.5,.5],[1.,0.]],dtype=np.float32)
    assert np.max(np.abs(nonempty_baseline(toy,'PRESUPPOSED')-expected)) < 1e-7
    assert not guard and len(rows) == 70
    atomic_json(ROOT/'docs/a100_visual_classical_20261121/protocol_calibration_sanity.json',dict(
        status='PASS_PROTOCOL_CALIBRATION_OPERATOR_PARITY',videos=len(rows),neural_seed=0,rows=rows,
        maximum_rank_one_probability_error=max(r['rank_one_probability_max_error'] for r in rows),
        changed_decisions=sum(r['rank_one_decision_changes'] for r in rows),GT_open_attempts=0,
        protocol_switch_uses_only_task_input=True,no_new_teacher_encoder_head_calls=True,
        implementation_sha256={name:file_sha(ROOT/name) for name in SOURCES},goal_status='ACTIVE'))
    print('PASS_PROTOCOL_CALIBRATION_OPERATOR_PARITY',flush=True)


if __name__ == '__main__':
    main()
