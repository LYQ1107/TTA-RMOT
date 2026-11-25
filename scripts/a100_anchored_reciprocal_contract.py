"""Exact path-cover, preserved supervision mass and reciprocal-gradient checks."""
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    from flex_tta.runtime.pf_receipts import atomic_json, file_sha
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.active.anchored_reciprocal import anchor_edge_cover, peer_weights, train
    from flex_uda.active.reciprocal_memory import MemoryInputs, models, train as old_train
    from flex_uda.active.joint_observation_inputs import JointObservationVideo, use_verified_local_features
    from scripts.a100_anchored_reciprocal_pilot import OUT, PUBLIC, registration

    guard = generation_guard()
    plan = registration()
    torch.set_num_threads(2)
    path_cases = 0
    for size in range(1, 8):
        obs = np.column_stack((np.arange(size), np.zeros(size, int)))
        video = SimpleNamespace(obs=obs, columns=np.zeros(size, int),
                                positions={(i, 0): i for i in range(size)})
        inputs = SimpleNamespace(video=video, left=np.arange(size-1), right=np.arange(1, size),
                                 visual=np.ones(size-1))
        for marked in itertools.product((False, True), repeat=size):
            paid = [dict(local_query=0, frame_id=i, track_id=0) for i in range(size) if marked[i]]
            chosen, report = anchor_edge_cover(inputs, paid)
            needed = {i for i in range(size) if marked[i] and size > 1}
            best = size
            for mask in itertools.product((False, True), repeat=max(0, size-1)):
                covered = {v for i, use in enumerate(mask) if use for v in (i, i+1)}
                if needed <= covered:
                    best = min(best, sum(mask))
            assert len(chosen) == best, (size, marked, chosen, best)
            assert report['isolated_paid_image_groups'] == (int(marked[0]) if size == 1 else 0)
            path_cases += 1
    kl = torch.tensor([0., 20., 1., 1000.], requires_grad=True)
    group = torch.tensor([0, 0, 1, 1])
    w = peer_weights(kl, group)
    assert not w.requires_grad and bool(((w >= .2) & (w <= 5)).all())
    assert torch.allclose(torch.zeros(2).index_add_(0, group, w), torch.tensor([2., 2.]))
    generator = np.random.default_rng(0)
    arrays = dict(bags=generator.normal(size=(12, 3, 1101)).astype('float32'),
                  present=np.ones((12, 3), bool), current=generator.normal(size=(12, 1101)).astype('float32'),
                  coarse=generator.normal(size=(12, 1101)).astype('float32'),
                  memory=generator.normal(size=(12, 2202)).astype('float32'))
    paid = [dict(row=i, local_query=0, factor_mode=('WHOLE', 'APPEARANCE', 'STATE')[i % 3],
                 yes_probability=(.02, .7, .4, .96)[i % 4]) for i in range(12)]

    class Toy:
        def arrays(self, records):
            return {k: v[[r['row'] for r in records]] for k, v in arrays.items()}

    old, new = models('cpu'), models('cpu')
    edges = ({k: v[:4] for k, v in arrays.items()}, {k: v[4:8] for k, v in arrays.items()},
             np.ones((4, 3), dtype='float32'))
    old_train(old, Toy(), paid, device='cpu', steps=2, mutual=True, edges=edges, weights=np.full((2, 3), .5))
    report = train(new, Toy(), paid, device='cpu', steps=2, edges=edges, weights=np.full((2, 3), .5))
    changes = [max(float((a-b).abs().max()) for a, b in zip(first.parameters(), second.parameters()))
               for first, second in zip(old, new)]
    assert min(changes) > 0 and report['original_query_role_mass_preserved']
    use_verified_local_features()
    real = []
    for case in plan['cases']:
        paid = json.loads(Path(case['paid_answers']).read_text())['paid']
        inputs = MemoryInputs(JointObservationVideo(case['pair'], case['video']))
        chosen, report = anchor_edge_cover(inputs, paid)
        assert report['covered_paid_image_groups']+report['isolated_paid_image_groups'] == report['unique_paid_image_groups']
        real.append(dict(pair=case['pair'], **report))
    assert not guard
    receipt = dict(status='PASS_ANCHORED_PATH_COVER_PEER_MASS_AND_BOTH_RECIPROCAL_GRADIENTS',
                   exact_exhaustive_path_mask_cases=path_cases,
                   both_students_change_against_old_objective=changes,
                   peer_weights_stop_gradient_and_query_role_mass_preserved=True,
                   conditional_weight_bounds=[.2, 5], real_cases=real,
                   no_manual_GT_teacher_calls_or_RMOT_predictions=True,
                   tiny_CPU_synthetic_fits_only=True, not_task_quality_evidence=True,
                   source_sha256=file_sha(Path(__file__)))
    atomic_json(OUT/'contract.json', receipt)
    atomic_json(PUBLIC/'anchored_reciprocal_primary_contract_20261125.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
