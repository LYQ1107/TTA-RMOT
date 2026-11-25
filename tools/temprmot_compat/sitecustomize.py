"""Small runtime shims for the pinned TempRMOT code on the shared runtime.

The current shared Transformers runtime is 5.x, where the legacy
``batch_encode_plus`` alias was removed from fast tokenizers. TempRMOT only
uses that method as a batch tokenizer call, so preserving the alias through
the project-local compatibility path keeps the official model/data protocol
unchanged without editing the external reference checkout.
"""

try:
    from transformers import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "batch_encode_plus"):
        PreTrainedTokenizerBase.batch_encode_plus = PreTrainedTokenizerBase.__call__
except Exception:
    # The shim must not mask the actual import error when Transformers is not
    # part of the selected environment.
    pass
