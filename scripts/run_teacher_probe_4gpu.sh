#!/usr/bin/env bash
set -euo pipefail
cd /data2/user/FlexHook-TTA
exec /home/szgx/anaconda3/envs/dfot/bin/python scripts/launch_teacher_probe.py "$@"
