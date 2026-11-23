"""Shared converged ordinary head, task-contract pooling, frozen visual PAVA."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, metadata, generation_guard
from flex_uda.active.acquisition import exclusive_group
from flex_uda.active.semantic_colour import appearance_cues, crossfit_predictions, correct_belief, append_history
from flex_uda.active.topology_selection import graph_matrices, apply_graph
from flex_uda.active.protocol_calibration import global_pool, contract_graph, nonempty_baseline
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_rgb_parity import source_spec
from scripts.a100_visual_classical_semantic_colour import empty_history
from scripts.a100_visual_classical_strict_prefix import plain, immediate_predictions

STUDY = OUT/'protocol_calibration'
METHODS = ('PROTOCOL_GRAPH', 'PROTOCOL_JOINT', 'FINAL_CAL', 'FINAL_LOCAL', 'FINAL_GLOBAL', 'BASE_PROTOCOL')
SOURCES = ('flex_uda/active/protocol_calibration.py', 'scripts/a100_visual_classical_protocol_calibration.py',
           'flex_uda/active/topology_selection.py', 'flex_uda/active/semantic_colour.py')


def run(pair, seed):
    guard = generation_guard(); begin = time.perf_counter(); root = STUDY/pair/f'seed{seed}'
    if (root/'completion.json').is_file():
        return
    root.mkdir(parents=True, exist_ok=True)
    implementation = {name: file_sha(ROOT/name) for name in SOURCES}
    model, target = PAIRS[pair]; m = metadata(target); source = source_spec(pair)[0]
    previous = OUT/'confidence_teacher'/pair/f'seed{seed}'
    protocol = json.loads((OUT/'preregistration_v1.json').read_text())['protocol'][pair]
    history = empty_history(); receipts = []; calls = 0
    for vi, (video, bank) in enumerate(sorted(m['videos'].items())):
        stamp = time.perf_counter(); local = root/'videos'/video; local.mkdir(parents=True, exist_ok=True)
        queries = [q for q in m['episodes'] if q['video_id'] == video]; texts = [q['query_text'] for q in queries]
        with np.load(source/'source'/video/'sparse_inputs.npz') as a:
            pcd, colours, prior, tracks = a['pcd'], a['colours'], a['prior'], a['tracks']
        with np.load(previous/'videos'/video/'coarse_scores.npz') as a:
            assert np.array_equal(tracks, a['tracks'])
            unary = a['KD_FINAL']
        obs = np.asarray(bank['observations']); lookup = {int(t): i for i, t in enumerate(tracks)}
        columns = np.asarray([lookup[int(t)] for _, t in obs]); lengths = np.asarray([(obs[:, 1] == t).sum() for t in tracks])
        cue, available, families, _ = appearance_cues(colours, texts, lengths); teacher = np.full(cue.shape, np.nan)
        for answer in json.loads((source/'source'/video/'answers.json').read_text())['answers']:
            teacher[int(answer['local_query']), int(answer['local_track'])] = answer['yes_probability']
        calls += int(np.isfinite(teacher).sum())
        probability, base, active, _ = crossfit_predictions(cue, available, families, teacher, tracks, video, history)
        operation_start = time.perf_counter()
        geometry, local_weights, global_weights, valid, supported = graph_matrices(pcd, colours, texts)
        global_probability = global_pool(unary, teacher, global_weights, valid, supported)
        local_probability = apply_graph(unary, teacher, geometry, local_weights, supported)
        selected = contract_graph(unary, global_probability, protocol)
        values = dict(KD=unary, PROTOCOL_GRAPH=selected,
            PROTOCOL_JOINT=correct_belief(selected, probability, base, active),
            FINAL_CAL=correct_belief(unary, probability, base, active),
            FINAL_LOCAL=correct_belief(local_probability, probability, base, active),
            FINAL_GLOBAL=correct_belief(global_probability, probability, base, active),
            BASE_PROTOCOL=nonempty_baseline(unary, protocol))
        operation_seconds = time.perf_counter()-operation_start
        exclusive = exclusive_group(texts, target)
        mask = plain(unary[:, columns], exclusive)
        errors = dict(KD_FINAL_mask_changed=int(np.count_nonzero(mask != np.load(previous/'KD_FINAL/videos'/video/'mask.npy'))))
        assert not any(errors.values()), (pair, seed, video, errors)
        assert np.array_equal(selected, unary if protocol == 'PRESUPPOSED' else global_probability)
        if (local/'receipt.json').is_file():
            receipt = json.loads((local/'receipt.json').read_text()); assert receipt['implementation'] == implementation
        else:
            for name in METHODS:
                mask = plain(values[name][:, columns], exclusive); folder = root/name/'videos'/video; folder.mkdir(parents=True, exist_ok=True)
                np.save(folder/'mask.npy', mask)
                immediate_predictions(root/name, type('Context', (), dict(bank=bank, queries=queries))(), mask)
            np.savez_compressed(local/'coarse_scores.npz', tracks=tracks, **values)
            receipt = dict(video=video, index=vi, protocol=protocol, implementation=implementation,
                current_teacher_examples=int(np.isfinite(teacher).sum()), frozen_prediction_max_errors=errors,
                operation_CPU_seconds=operation_seconds, same_exact_final_head_as_ordinary=True,
                protocol_not_selected_by_dataset_or_GT=True, GT_open_attempts=len(guard), wall_seconds=time.perf_counter()-stamp)
            atomic_json(local/'receipt.json', receipt)
        receipts.append(receipt); append_history(history, cue, available, families, teacher)
        print(json.dumps(dict(event='PROTOCOL_CALIBRATION_VIDEO', pair=pair, seed=seed, completed=vi+1, total=len(m['videos']))), flush=True)
    methods = []
    for name in METHODS:
        mask = np.zeros(m['candidate_query_rows'], bool)
        for video in sorted(m['videos']):
            local_mask = np.load(root/name/'videos'/video/'mask.npy')
            for q, e in enumerate(e for e in m['episodes'] if e['video_id'] == video):
                mask[e['start']:e['stop']] = local_mask[q]
        seal = seal_mask(root/name, m, mask, target=target, model=model, method='PROTOCOL_CALIBRATION_'+name,
            GT_open_attempts=len(guard), extra=dict(seed=seed, implementation=implementation,
                neural_dimension=768, source_logit_zero=True, shared_head_readout='CONVERGED_FINAL',
                new_head_fits=0, new_teacher_calls=0, new_encoder_calls=0,
                current_video_transductive=True, no_future_teacher=True, quality_replay_not_cold_time=True))
        methods.append(dict(method=name, rows=seal['prediction_rows']))
    assert not guard and calls == (1160 if pair == 'C4' else 1103)
    atomic_json(root/'completion.json', dict(status='SEALED_PROTOCOL_CALIBRATION', pair=pair, seed=seed, methods=methods,
        receipts=receipts, logical_teacher_calls=calls, GT_open_attempts=0, new_teacher_calls=0, new_encoder_calls=0,
        new_head_fits=0, implementation=implementation, wall_seconds=time.perf_counter()-begin, goal_status='ACTIVE'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--pair', choices=['C4', 'C1'], required=True); p.add_argument('--seed', type=int, required=True)
    args = p.parse_args(); run(args.pair, args.seed)
