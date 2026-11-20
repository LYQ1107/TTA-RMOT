"""Conservative POS/NEG/UNKNOWN selection for aligned probe scores."""
from __future__ import annotations

import math

from rmot_tta.evidence.records import CandidateEvidence, EvidenceLabel
from rmot_tta.types import AlignedCandidate, FrameSnapshot


def _weight(alignment_quality: float, p: float, source_type_weight: float) -> float:
    entropy = -(p * math.log(max(p, 1e-6)) + (1.0 - p) * math.log(max(1.0 - p, 1e-6))) / math.log(2.0)
    return max(0.0, min(1.0, alignment_quality * (1.0 - entropy) * source_type_weight))


def select_candidate_evidence(snapshot: FrameSnapshot, aligned: list[AlignedCandidate],
                              tau_pos: float = 0.8, tau_neg: float = 0.2,
                              equiv_tolerance: float = 0.1,
                              counterfactual_margin: float = 0.3) -> list[CandidateEvidence]:
    records: list[CandidateEvidence] = []
    for item in aligned:
        p0, peq, pcf = item.r0, item.req, item.rcf
        if item.status != "OK" or p0 is None or peq is None:
            records.append(CandidateEvidence(item.main_uid, EvidenceLabel.UNKNOWN, None, 0.0, "ALIGNMENT", "q0", item.alignment_quality,
                                              "missing_or_ambiguous_probe_alignment", snapshot.frame_index, int(snapshot.pred_track_id[item.main_index])))
            continue
        if abs(p0 - peq) > equiv_tolerance:
            records.append(CandidateEvidence(item.main_uid, EvidenceLabel.UNKNOWN, None, 0.0, "CONTRADICTORY_PROBE", "q_eq", item.alignment_quality,
                                              "equivalent_query_disagreement", snapshot.frame_index, int(snapshot.pred_track_id[item.main_index])))
            continue
        mean_probability = 0.5 * (p0 + peq)
        if mean_probability >= tau_pos:
            if pcf is not None:
                if pcf <= tau_neg and mean_probability - pcf >= counterfactual_margin:
                    source, source_weight = "CONTRAST_SELF_PROBE", 1.0
                elif pcf >= tau_pos:
                    records.append(CandidateEvidence(item.main_uid, EvidenceLabel.CONTRADICTORY_PROBE, None, 0.0, "CONTRADICTORY_PROBE", "q_cf", item.alignment_quality,
                                                      "original_and_counterfactual_both_high", snapshot.frame_index, int(snapshot.pred_track_id[item.main_index])))
                    continue
                else:
                    records.append(CandidateEvidence(item.main_uid, EvidenceLabel.UNKNOWN, None, 0.0, "UNKNOWN", "q_cf", item.alignment_quality,
                                                      "counterfactual_not_decisive", snapshot.frame_index, int(snapshot.pred_track_id[item.main_index])))
                    continue
            else:
                source, source_weight = "EQ_SELF_PROBE", 0.5
            records.append(CandidateEvidence(item.main_uid, EvidenceLabel.SUPPORTED_POS, mean_probability,
                                              _weight(item.alignment_quality, mean_probability, source_weight), source,
                                              "q0+q_eq", item.alignment_quality, "equivalent_queries_agree", snapshot.frame_index,
                                              int(snapshot.pred_track_id[item.main_index]),
                                              probe_q0=p0, probe_eq=peq, probe_cf=pcf))
            continue
        if mean_probability <= tau_neg and pcf is not None and pcf >= tau_pos:
            records.append(CandidateEvidence(item.main_uid, EvidenceLabel.SUPPORTED_NEG, 0.0,
                                              _weight(item.alignment_quality, 1.0 - mean_probability, 1.0), "CONTRAST_SELF_PROBE", "q_cf",
                                              item.alignment_quality, "counterfactual_high_same_candidate", snapshot.frame_index,
                                              int(snapshot.pred_track_id[item.main_index]),
                                              probe_q0=p0, probe_eq=peq, probe_cf=pcf))
        else:
            records.append(CandidateEvidence(item.main_uid, EvidenceLabel.UNKNOWN, None, 0.0, "UNKNOWN", "q0+q_eq", item.alignment_quality,
                                              "double_low_is_not_a_verified_negative", snapshot.frame_index,
                                              int(snapshot.pred_track_id[item.main_index])))
    return records
