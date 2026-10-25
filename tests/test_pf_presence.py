"""Scientific non-identifiability/causality checks, without target annotations."""
import pytest
import torch

from flex_tta.adaptation.presence_evidence import CausalInteractionAudit, interaction_contrast


def scores(real=(2.,-3.), interaction=(1.,1.)):
    real=torch.tensor(real)
    return dict(real_query=real,real_reference=real-torch.tensor(interaction),
                null_query=torch.zeros_like(real),null_reference=torch.zeros_like(real))


def test_additive_visual_and_query_offsets_cancel():
    q,r,nq,nr=[torch.tensor(x) for x in ((9.,1.),(2.,4.),(6.,3.),(1.,5.))]
    v,v0,bq,br=[torch.tensor(x) for x in ((3.,7.),(-2.,4.),(5.,2.),(-1.,8.))]
    assert torch.allclose(interaction_contrast(q,r,nq,nr),interaction_contrast(q+v+bq,r+v+br,nq+v0+bq,nr+v0+br))


def test_stable_positive_interaction_is_not_a_semantic_positive():
    audit=CausalInteractionAudit()
    for frame in range(1,5):
        result=audit.step(frame,[10,20],**scores())
    assert result["provisional_interaction_support_ids"] == [10,20]
    assert result["past_history_latest_frame"]==3
    assert result["presence_status"]=="UNKNOWN" and not result["support_is_semantic_positive"]
    assert not result["optimizer_update_allowed"] and result["prediction_override"] is None
    assert result["source_selected_ids"]==[10]


def test_empty_and_multiple_outputs_do_not_force_a_single_target():
    audit=CausalInteractionAudit()
    empty=audit.step(1,[],**scores((),()))
    assert empty["source_selected_ids"]==[] and empty["presence_status"]=="UNKNOWN"
    audit.reset()
    multiple=audit.step(1,[11,22],**scores((1.,2.),(1.,1.)))
    assert multiple["source_selected_ids"]==[11,22] and multiple["prediction_override"] is None


def test_history_is_causal_and_resets_between_episodes():
    audit=CausalInteractionAudit()
    for frame in range(1,5):
        audit.step(frame,[10,20],**scores())
    with pytest.raises(ValueError,match="causally"):
        audit.step(3,[10,20],**scores())
    audit.reset()
    result=audit.step(1,[10,20],**scores())
    assert not result["provisional_interaction_support_ids"] and result["past_history_latest_frame"] is None


def test_old_or_nonpositive_interaction_does_not_become_stable_support():
    audit=CausalInteractionAudit()
    for frame in (1,2,3):
        audit.step(frame,[10,20],**scores())
    result=audit.step(4,[10,20],**scores(interaction=(-1.,-1.)))
    assert result["provisional_interaction_support_ids"]==[]
    result=audit.step(10,[10,20],**scores())
    assert result["past_history_latest_frame"] is None and result["provisional_interaction_support_ids"]==[]


def test_zero_interaction_does_not_prove_absence():
    result=CausalInteractionAudit().step(1,[10,20],**scores(interaction=(0.,0.)))
    assert result["presence_status"]=="UNKNOWN" and not result["optimizer_update_allowed"]
    assert result["source_selected_ids"]==[10]  # UNKNOWN is not a negative pseudo-label.
