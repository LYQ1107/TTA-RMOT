# Independent tracklet-language evidence

`flex_tta/evidence/tracklet_language.py` implements a conservative S1/S2
development mechanism that is independent of the source FlexHook score.  It
parses the current Dance clothing query, extracts top/bottom/full-body color
attributes, reads only current and causal four-frame tracker boxes, and
computes robust RGB/HSV color evidence from the corresponding image crops.
Tracklet motion stability is used only as a reliability gate.  A candidate is
labelled `SUPPORTED_POSITIVE`, `SUPPORTED_NEGATIVE`, or `UNKNOWN`; unknown
crops do not contribute to the evidence BCE loss.

The implementation handles compositional queries such as “white Tshirt and
purple pants” by requiring each requested garment attribute to be supported.
It does not use `labels.json`, GT templates, source logits, future frames, or a
new tracker.  The optional runner flag is:

```text
--evidence TRACKLET_LANGUAGE --evidence-weight 0.25
```

This is intentionally an auditable evidence method, not a claimed result
improver.  Its first image-only regression passed on Dance video
`dancetrack0004`; the neural full-suite ablation remains a separately tracked
experiment and must be reported with its actual TrackEval numbers if run.
The PCD runner records positive/negative/unknown evidence counts and the
source-free flags in `run_metadata.json`.
