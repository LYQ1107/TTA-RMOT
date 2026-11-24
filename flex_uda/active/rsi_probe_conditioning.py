"""Keep comparable paid evidence for a relative observer-progress score.

The caller has already charged every reply. Ineligible pairs still belong
to its paid training library; only the progress critic filters them. This
metadata rule uses no teacher probability, target label, or future reply.
Equal effective views are necessary for the intended shared-view nuisance
assumption, but do not establish that assumption or semantic correctness.
"""


def comparable_paid_probe(probe):
    assert len(probe) % 2 == 0
    kept = []
    dropped = 0
    for a, b in zip(probe[::2], probe[1::2]):
        assert a['local_query'] == b['local_query']
        assert a['local_track'] != b['local_track']
        assert a['case']['view'] == b['case']['view']
        if a['effective_view'] != b['effective_view']:
            dropped += 1
        else:
            kept.extend((a, b))
    # An identity return makes unchanged conditioning exactly reproducible,
    # including the original numeric path and pair ordering.
    eligible = probe if dropped == 0 else kept
    return eligible, dict(paid_probe_nodes=len(probe),
        original_pairs=len(probe)//2, comparable_pairs=len(eligible)//2,
        effective_view_mismatch_pairs=dropped,
        all_original_teacher_calls_still_charged=True,
        all_paid_replies_retained_for_final_fit=True,
        only_metadata_not_probabilities_used=True,
        conditioning='same query, requested view and effective view; different tracks',
        same_frame_not_required_or_guaranteed=True)
