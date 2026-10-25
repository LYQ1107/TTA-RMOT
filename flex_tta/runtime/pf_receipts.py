"""Atomic whole-episode sealing and restart; no mid-episode state fiction."""
from __future__ import annotations

import hashlib
import json
import os
import time
import gzip
from contextlib import contextmanager
from pathlib import Path


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w") as f:
        json.dump(value, f, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def append_json(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps(value)+"\n"); f.flush(); os.fsync(f.fileno())


class EpisodeStore:
    def __init__(self, method_root, video, expression, config, *, resume=False):
        self.method_root = Path(method_root)
        self.root = self.method_root / "results" / video / expression
        self.video, self.expression = video, expression
        self.config = config
        self.fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        self.resume = resume
        self.receipt = self.root / "completion.json"
        self.prediction = self.root / "predict.txt"
        self.margins = self.root / "candidate_margins.jsonl.gz"

    def completed(self):
        if not self.receipt.exists():
            return None
        if not self.resume:
            raise FileExistsError(f"episode already sealed: {self.root}")
        receipt = json.loads(self.receipt.read_text())
        if receipt["config_fingerprint"] != self.fingerprint:
            raise ValueError(f"resume configuration mismatch: {self.root}")
        for key, path in (("prediction_sha256", self.prediction), ("margins_sha256", self.margins)):
            if not path.is_file() or file_sha(path) != receipt[key]:
                raise ValueError(f"sealed episode integrity mismatch: {path}")
        return receipt

    def begin(self):
        self.root.mkdir(parents=True, exist_ok=True)
        # Preserve an interrupted attempt; never treat it as a completed episode.
        leftovers = [p for p in self.root.iterdir() if p.is_file() and p.name != "gt.txt"]
        if leftovers:
            if not self.resume:
                raise FileExistsError(f"partial episode exists; pass --resume: {self.root}")
            old = self.root / "failed_attempts" / f"{time.time_ns()}"
            old.mkdir(parents=True)
            for path in leftovers:
                os.replace(path, old / path.name)
        return self.prediction.with_suffix(".txt.partial"), self.margins.with_suffix(".gz.partial")

    def seal(self, prediction_partial, margins_partial, record):
        os.replace(prediction_partial, self.prediction)
        os.replace(margins_partial, self.margins)
        receipt = dict(record, status="complete", video=self.video, expression_id=self.expression,
                       config=self.config, config_fingerprint=self.fingerprint,
                       prediction_path=str(self.prediction), margins_path=str(self.margins),
                       prediction_sha256=file_sha(self.prediction), margins_sha256=file_sha(self.margins))
        atomic_json(self.receipt, receipt)
        append_json(self.method_root / "completed_episodes.jsonl", receipt)
        return receipt

    def fail(self, error):
        append_json(self.method_root / "failed_episodes.jsonl",
                    dict(video=self.video, expression_id=self.expression, status="failed",
                         time=time.time(), error=repr(error), config_fingerprint=self.fingerprint))

    @contextmanager
    def attempt(self):
        self.partials = self.begin()
        try:
            with self.partials[0].open("w") as prediction, gzip.open(self.partials[1], "wt") as margins:
                yield prediction, margins
                prediction.flush(); os.fsync(prediction.fileno())
        except BaseException as error:
            self.fail(error)
            raise

    def finish(self, record):
        return self.seal(*self.partials, record)
