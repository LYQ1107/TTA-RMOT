#!/usr/bin/env bash
set -euo pipefail
PF_REPO="${PF_REPO:-/data2/user/FlexHook-TTA}"
PF_PYTHON="${PF_PYTHON:-/home/szgx/anaconda3/envs/dfot/bin/python}"
export PYTHONPATH="$PF_REPO/compat:$PF_REPO:/data2/user/TTA-RMOT-assets/flexhook/pydeps:${PYTHONPATH:-}"
if [[ "${PF_LAUNCH_FULL:-NO}" != YES || "${FLEXHOOK_FULL_RUN_AUTHORIZED:-NO}" != YES ]]; then
  echo 'BLOCKED: full-suite launch requires both explicit full-run switches' >&2
  exit 3
fi
if (( $# )); then PF_SUITE_METHODS=("$@"); else
  PF_SUITE_METHODS=(ZERO_UPDATE RELATIVE_ONLY PF_RANK PF_RANK_GUARD PF_IDENTITY PF_HYBRID)
fi
for PF_SUITE_METHOD in "${PF_SUITE_METHODS[@]}"; do
  PF_SUITE_LOWER="${PF_SUITE_METHOD,,}"
  PF_SUITE_LOWER="${PF_SUITE_LOWER#pf_}"
  if [[ "$PF_SUITE_METHOD" == ZERO_UPDATE ]]; then PF_SUITE_LOWER=zero; fi
  PF_SUITE_RUN="$PF_REPO/results/runs/pf_${PF_SUITE_LOWER}_completebank_full_20261025"
  PF_SUITE_MERGED="${PF_SUITE_RUN}_merged"
  if [[ "$PF_SUITE_METHOD" != ZERO_UPDATE && ! -f "$PF_REPO/results/positive_free/full/zero_update/metrics.json" ]]; then
    echo 'ZERO_UPDATE complete-bank baseline must be evaluated before subsequent methods' >&2; exit 4
  fi
  if [[ ! -f "$PF_SUITE_MERGED/$PF_SUITE_METHOD/run_metadata.json" ]]; then
    PF_SUITE_DERIVE=NO
    if [[ "$PF_SUITE_METHOD" == PF_RANK_GUARD ]]; then
      PF_SUITE_RANK="$PF_REPO/results/runs/pf_rank_completebank_full_20261025_merged"
      PF_SUITE_VERIFY="$PF_REPO/results/runs/pf_guard_equivalence_completebank_20261025"
      if [[ ! -d "$PF_SUITE_VERIFY" ]]; then
        PF_SUITE_VERIFY_GPU="$($PF_PYTHON -c 'from flex_tta.runtime.pf_shards import gpu_snapshot,idle_gpus; x=idle_gpus(gpu_snapshot()); assert x, "No idle GPU for guard equivalence"; print(x[0]["index"])')"
        CUDA_VISIBLE_DEVICES="$PF_SUITE_VERIFY_GPU" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
          "$PF_PYTHON" "$PF_REPO/scripts/verify_pf_guard_equivalence.py" --rank-root "$PF_SUITE_RANK" \
          --output-root "$PF_SUITE_VERIFY" || echo 'Guard equivalence failed; preserve evidence and use direct full GPU fallback'
      fi
      if [[ -f "$PF_SUITE_VERIFY/equivalence.json" ]] && \
        "$PF_PYTHON" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "PASS"' "$PF_SUITE_VERIFY/equivalence.json"; then
        "$PF_PYTHON" "$PF_REPO/scripts/derive_pf_rank_guard.py" --rank-root "$PF_SUITE_RANK" --output-root "$PF_SUITE_MERGED"
        PF_SUITE_DERIVE=YES
      fi
    fi
    if [[ "$PF_SUITE_DERIVE" != YES ]]; then
      PF_METHOD="$PF_SUITE_METHOD" PF_RUN_ROOT="$PF_SUITE_RUN" bash "$PF_REPO/scripts/run_pf_4gpu.sh" --launch-full
      "$PF_PYTHON" "$PF_REPO/scripts/merge_flexhook_pcd_shards.py" --shards-root "$PF_SUITE_RUN" \
        --output-root "$PF_SUITE_MERGED" --method "$PF_SUITE_METHOD" --resume
    fi
  fi
  "$PF_PYTHON" "$PF_REPO/scripts/audit_pf_full.py" --output-root "$PF_SUITE_MERGED" --method "$PF_SUITE_METHOD"
  if [[ "$PF_SUITE_METHOD" == ZERO_UPDATE ]]; then
    "$PF_PYTHON" -c 'import json,sys; a=json.load(open(sys.argv[1])); assert a["source_sign_mismatches"] == 0, "Diagnose ZERO_UPDATE/Frozen sign differences before continuing"' \
      "$PF_SUITE_MERGED/ZERO_UPDATE/full_audit.json"
  fi
  "$PF_PYTHON" "$PF_REPO/scripts/evaluate_flexhook_dance_tta.py" --output-root "$PF_SUITE_MERGED" \
    --dataset-root /data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance \
    --vendor-root "$PF_REPO/vendor/FlexHook" --methods "$PF_SUITE_METHOD"
  "$PF_PYTHON" "$PF_REPO/scripts/summarize_pf_full.py" --output-root "$PF_SUITE_MERGED" --method "$PF_SUITE_METHOD"
  "$PF_PYTHON" "$PF_REPO/scripts/analyze_pf_full_offline.py" --output-root "$PF_SUITE_MERGED" --method "$PF_SUITE_METHOD" \
    --baseline-root "$PF_REPO/results/runs/pf_zero_completebank_full_20261025_merged" --i-allow-offline-gt \
    --output "$PF_REPO/results/positive_free/full/${PF_SUITE_METHOD,,}/offline_attribution.json"
  echo "STAGE_COMPLETE $PF_SUITE_METHOD"
done
