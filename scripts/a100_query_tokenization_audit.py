"""Audit the actual source text boundary using only provided task queries.

The frozen historical PCD exporter uses FrozenFeatureCache._text, whose
tokenizer limit is 25 including special tokens. No model forward, teacher
reply, prediction, or manual target annotation is needed for this audit.
"""
import ast
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.a100_fullscope_visual_six import PLAN, PUBLIC
from scripts.a100_six_direction import OUT as SIX

OUT = SIX / 'query_tokenization_audit_20261126'
SOURCES = (
    'scripts/a100_query_tokenization_audit.py',
    'flex_tta/runtime/frozen_features.py',
    'flex_uda/presence/runtime.py',
    'scripts/presence_export_target.py',
    'flex_uda/active/observation_learning_inputs.py',
    'flex_uda/active/factor_rsi_learning.py',
    'flex_uda/active/fullscope_visual_runtime.py',
)


def main():
    from flex_uda.a100.mechanism import generation_guard
    guard = generation_guard()
    from transformers import RobertaTokenizerFast
    tree = ast.parse((ROOT / 'flex_tta/runtime/frozen_features.py').read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_text')
    call = next(n for n in ast.walk(function) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == 'tokenizer')
    options = {k.arg: ast.literal_eval(k.value) for k in call.keywords if k.arg in (
        'padding', 'truncation', 'max_length')}
    assert options == dict(padding='max_length', truncation=True, max_length=25)
    tokenizer = RobertaTokenizerFast.from_pretrained(ROOT / 'assets/roberta-base', local_files_only=True)
    assert tokenizer.encode('person', add_special_tokens=False) == [5970]
    plan = json.loads(PLAN.read_text())
    assert len(plan['cases']) == 148
    documents = []
    for case in plan['cases']:
        path = Path(case['temporal_queries'])
        assert file_sha(path) == case['temporal_queries_sha256']
        doc = json.loads(path.read_text())
        documents.append((case, doc['queries']))
    texts = sorted({q['query_text'] for _, queries in documents for q in queries})
    full = tokenizer(texts, padding=False, truncation=False, return_offsets_mapping=True)
    limited = tokenizer(texts, **options, return_offsets_mapping=True)
    tokenized = {}
    for i, text in enumerate(texts):
        end = max((int(b) for a, b in limited['offset_mapping'][i]), default=0)
        tokenized[text] = dict(full_tokens_including_special=len(full['input_ids'][i]),
            full_ids=tuple(full['input_ids'][i]),
            limited_ids=tuple(limited['input_ids'][i]),
            limited_mask=tuple(limited['attention_mask'][i]),
            truncated=len(full['input_ids'][i]) > 25,
            retained_character_end=end, retained_prefix=text[:end], omitted_suffix=text[end:])
    rows, cases, aliases = [], [], []
    for case, queries in documents:
        groups = defaultdict(list)
        local = []
        for query in queries:
            text = query['query_text']; encoded = tokenized[text]
            row = dict(pair=case['pair'], target=case['target'], video=case['evaluated_video'],
                query_id=query['query_id'], query_text=text,
                **{k: v for k, v in encoded.items() if k not in ('full_ids', 'limited_ids', 'limited_mask')})
            rows.append(row); local.append(row)
            groups[encoded['limited_ids'], encoded['limited_mask']].append(row)
        local_aliases = []
        for group in groups.values():
            if len({tokenized[r['query_text']]['full_ids'] for r in group}) <= 1:
                continue
            entry = dict(pair=case['pair'], target=case['target'], video=case['evaluated_video'],
                queries=[dict(query_id=r['query_id'], query_text=r['query_text'],
                              omitted_suffix=r['omitted_suffix']) for r in group])
            aliases.append(entry); local_aliases.append(entry)
        cases.append(dict(pair=case['pair'], video=case['evaluated_video'], queries=len(local),
            truncated_queries=sum(r['truncated'] for r in local),
            maximum_full_tokens=max(r['full_tokens_including_special'] for r in local),
            within_video_truncation_alias_groups=len(local_aliases),
            within_video_truncation_aliased_queries=sum(len(a['queries']) for a in local_aliases)))
    by_pair = {}
    for pair in sorted({c['pair'] for c in plan['cases']}):
        selected = [r for r in rows if r['pair'] == pair]
        blocks = [r for r in cases if r['pair'] == pair]
        count = sum(r['truncated'] for r in selected)
        by_pair[pair] = dict(videos=len(blocks), queries=len(selected), truncated_queries=count,
            truncated_fraction=count / len(selected),
            full_token_length_histogram=dict(sorted(Counter(r['full_tokens_including_special'] for r in selected).items())),
            within_video_truncation_alias_groups=sum(r['within_video_truncation_alias_groups'] for r in blocks),
            within_video_truncation_aliased_queries=sum(r['within_video_truncation_aliased_queries'] for r in blocks),
            examples=sorted((r for r in selected if r['truncated']),
                            key=lambda r: (-r['full_tokens_including_special'], r['video'], r['query_id']))[:3])
    assert not guard
    report = dict(status='COMPLETE_SOURCE_QUERY_TOKEN_BOUNDARY_AUDIT_NO_GT',
        source_sha256={name: file_sha(ROOT / name) for name in SOURCES},
        parent_registration_sha256=file_sha(PLAN), tokenizer_options=options,
        tokenizer_person_ids=[5970], unique_task_texts=len(texts),
        unique_texts_truncated=sum(r['truncated'] for r in tokenized.values()),
        video_units=len(cases), query_episodes=len(rows), settings=by_pair,
        actual_historical_source='presence_export_target.export_video -> presence.runtime.cache_for -> FrozenFeatureCache._text',
        current_student_query_channel='Historical query-conditioned PCD for three anchors/current/coarse; geometry, RGB and DINO are query-independent. No separate untruncated token representation is added by the9352D runtime.',
        original_teacher_receives_full_query_text=True,
        truncation_does_not_prove_semantic_error_or_HOTA_loss=True,
        token_alias_does_not_assert_byte_identical_PCD_under_all_batch_numerics=True,
        GT_open_attempts=0, model_or_teacher_forwards=0, fits=0, GPU_calls=0,
        main_numerical_sources_questions_weights_or_thresholds_changed=False,
        all_data_DEV=True, goal_status='ACTIVE')
    atomic_json(OUT / 'all_query_records_private.json', dict(queries=rows, aliases=aliases, videos=cases))
    atomic_json(OUT / 'completion.json', report)
    atomic_json(PUBLIC / 'query_tokenization_audit_20261126.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
