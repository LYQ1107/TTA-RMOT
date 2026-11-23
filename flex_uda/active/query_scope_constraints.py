"""Syntax/literal-span constraints for shared, fallible query preprocessing.

This controls representation format, not which interpretation is correct.
Every downstream comparator receives the same resulting structures.
"""
import re
from flex_uda.active.query_scope import ENUMS, SUBJECTS


def literal_spans(text):
    tokens = list(re.finditer(r'\w+|[^\w\s]', text))
    return sorted({text[left.start():right.end()]
        for i, left in enumerate(tokens) for right in tokens[i:]})


def schema(text):
    spans = literal_spans(text)
    assert spans
    properties = {key:dict(type='string', enum=sorted(values)) for key, values in ENUMS.items()}
    properties.update(value=dict(type='string', minLength=1), negated=dict(type='boolean'),
        span=dict(type='string', enum=spans))
    term = dict(type='object', properties=properties, required=list(properties), additionalProperties=False)
    top = dict(subject=dict(type='string', enum=sorted(SUBJECTS)),
        subject_span=dict(type='string', enum=spans), logic=dict(type='string', enum=['and','opaque']),
        terms=dict(type='array', items=term, minItems=1, maxItems=12),
        uncertain=dict(type='array', items=dict(type='string')))
    return dict(type='object', properties=top, required=list(top), additionalProperties=False)


def batch_prefix(tokenizer_data, texts):
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
    filters = [build_transformers_prefix_allowed_tokens_fn(tokenizer_data, JsonSchemaParser(schema(text)))
        for text in texts]

    def allowed(batch_id, tokens):
        return filters[int(batch_id)](0, tokens)
    return allowed
