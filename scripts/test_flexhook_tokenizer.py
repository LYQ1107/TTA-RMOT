#!/usr/bin/env python3
"""Regression test for the local RoBERTa ByteLevel-BPE compatibility hook."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import RobertaTokenizerFast


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, default=Path("vendor/FlexHook/pretrained/roberta-base"))
    args = parser.parse_args()
    tokenizer = RobertaTokenizerFast.from_pretrained(str(args.tokenizer))
    person = tokenizer("person", padding="max_length", truncation=True, max_length=25)
    sentence = tokenizer(
        "dancing person dressed all in white",
        padding="max_length",
        truncation=True,
        max_length=25,
    )
    person_ids = person["input_ids"]
    sentence_ids = sentence["input_ids"]
    # The fixed local RoBERTa BPE encodes person as one content token.  The
    # broken Transformers 5.4 combination emitted one token per character.
    content = [item for item in person_ids if item not in (0, 1, 2)]
    if len(content) != 1:
        raise AssertionError(f"person was not encoded as one BPE token: {person_ids}")
    if len(sentence_ids) != 25:
        raise AssertionError(f"unexpected fixed length: {len(sentence_ids)}")
    print(json.dumps({"tokenizer": str(args.tokenizer.resolve()), "person_ids": person_ids, "sentence_ids": sentence_ids}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
