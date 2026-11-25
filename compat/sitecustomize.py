"""Runtime compatibility for the released FlexHook code.

The released data loader calls ``batch_encode_plus``.  Transformers 5.x
removed that public alias while retaining the equivalent ``__call__`` API.
This module is loaded only when the FlexHook launcher prepends ``compat`` to
PYTHONPATH, so it does not modify the installed environment globally.
"""

try:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "batch_encode_plus"):
        _tokenizer_call = PreTrainedTokenizerBase.__call__

        def _legacy_batch_encode_plus(self, *args, **kwargs):
            # Transformers 4.x truncated sequences when the released code
            # requested a fixed max_length through this legacy entry point.
            # Keep that behavior for FlexHook's text_len=25 batches.
            if kwargs.get("padding") == "max_length" and kwargs.get("max_length") is not None:
                kwargs.setdefault("truncation", True)
            return _tokenizer_call(self, *args, **kwargs)

        PreTrainedTokenizerBase.batch_encode_plus = _legacy_batch_encode_plus
except Exception:
    # Keep Python startup usable for unrelated commands if Transformers is not
    # installed in the selected environment.
    pass


# Transformers 5.x can construct RobertaTokenizerFast from the released
# vocab/merges files with an empty BPE merge table under the current
# tokenizers combination.  That silently turns normal English into
# character-level tokens (for example, ``person`` -> ``p e r s o n``), which
# makes the released FlexHook text-conditioned checkpoints unusable.  Build
# the fast tokenizer directly from the canonical RoBERTa BPE files for local
# model directories.  The fallback preserves normal Hugging Face behavior for
# remote identifiers and unrelated tokenizers.
try:
    import json
    import os
    from pathlib import Path

    from tokenizers import ByteLevelBPETokenizer
    from tokenizers.decoders import ByteLevel
    from tokenizers.processors import RobertaProcessing
    from transformers import RobertaTokenizerFast
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

    _roberta_fast_from_pretrained = RobertaTokenizerFast.from_pretrained

    @classmethod
    def _flexhook_roberta_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        local_path = Path(os.fspath(pretrained_model_name_or_path)).expanduser()
        if local_path.is_dir() and (local_path / "vocab.json").is_file() and (local_path / "merges.txt").is_file():
            add_prefix_space = bool(kwargs.pop("add_prefix_space", False))
            with (local_path / "tokenizer_config.json").open() as handle:
                tokenizer_config = json.load(handle)
            bpe = ByteLevelBPETokenizer(
                str(local_path / "vocab.json"),
                str(local_path / "merges.txt"),
                add_prefix_space=add_prefix_space,
            )
            bpe._tokenizer.post_processor = RobertaProcessing(
                ("</s>", 2),
                ("<s>", 0),
                trim_offsets=True,
                add_prefix_space=add_prefix_space,
            )
            bpe._tokenizer.decoder = ByteLevel()
            return PreTrainedTokenizerFast(
                tokenizer_object=bpe._tokenizer,
                bos_token="<s>",
                eos_token="</s>",
                sep_token="</s>",
                cls_token="<s>",
                unk_token="<unk>",
                pad_token="<pad>",
                mask_token="<mask>",
                model_max_length=tokenizer_config.get("model_max_length", 512),
                **kwargs,
            )
        return _roberta_fast_from_pretrained.__func__(cls, pretrained_model_name_or_path, *args, **kwargs)

    RobertaTokenizerFast.from_pretrained = _flexhook_roberta_from_pretrained
except Exception:
    # The compatibility hook is optional outside the FlexHook environment.
    pass
