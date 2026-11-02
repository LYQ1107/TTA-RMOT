"""CPU-only DS comparison against crowd-kit on one fixed simulated dataset."""
import importlib.metadata
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.presence.runtime import RUN, PUBLIC
from flex_uda.active.reliability import Reliability, WORKERS
from flex_tta.runtime.pf_receipts import atomic_json


def main():
    sys.path.insert(0, str(RUN / 'deps/crowdkit'))
    import pandas as pd
    from crowdkit.aggregation import DawidSkene
    ds = Reliability(); rng = np.random.RandomState(0)
    labels = rng.rand(2000) < .4; rows = []
    accuracies = dict(zip(WORKERS, [.9, .85, .8, .75]))
    for task, label in enumerate(labels):
        for worker in WORKERS:
            answer = int(label if rng.rand() < accuracies[worker] else not label)
            ds.answers[task][worker].append(answer)
            rows.append(dict(task=task, worker=worker, label=answer))
    ours = ds.fit()
    reference = DawidSkene(n_iter=20, tol=0.).fit(pd.DataFrame(rows))
    compare = []
    for worker in WORKERS:
        error = reference.errors_.loc[worker]
        sensitivity, specificity = float(error.loc[1, 1]), float(error.loc[0, 0])
        compare.append(dict(worker=worker, simulated_accuracy=accuracies[worker],
            ours_sensitivity=ours[worker][0], ours_specificity=ours[worker][1],
            crowdkit_sensitivity=sensitivity, crowdkit_specificity=specificity))
    own_labels = np.asarray([ds.posteriors[i] > .5 for i in range(len(labels))])
    reference_labels = reference.labels_.reindex(range(len(labels))).to_numpy()
    result = dict(status='SIMULATED_REFERENCE_COMPARISON', seed=0, tasks=len(labels),
        crowdkit_version=importlib.metadata.version('crowd-kit'),
        ours_accuracy=float((own_labels == labels).mean()), crowdkit_accuracy=float((reference_labels == labels).mean()),
        label_agreement=float((own_labels == reference_labels).mean()), workers=compare,
        expected_difference='Our Beta(8,2) posterior-mean regularization; crowd-kit default unregularized EM. Not expected bit-exact.',
        target_data_used=False, target_GT_used=False, GPU_calls=0)
    atomic_json(PUBLIC / 'ds_reference_comparison.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
