#!/bin/bash
# Refresh the full-sweep reports while the sweep runs, and once more when it
# ends. Read-only against the run directories, so it is safe alongside training.
SWEEP_PID="$1"
refresh() {
  nice -n 15 python -c "
from rocklabel.train.ablate_report import render_ablation
render_ablation('training/ablate_vb', 'fullsweep', 'training/results_vb_fullsweep')
" 2>&1 | tail -2
  nice -n 15 python -m rocklabel.train.cli matched --suite fullsweep \
      --cache-dir training/cache_vb_fullsweep \
      --ablate-root training/ablate_vb \
      --out training/results_vb_matched 2>&1 | tail -3
}
while kill -0 "$SWEEP_PID" 2>/dev/null; do
  echo "--- refresh $(date '+%F %T') ---"
  refresh
  sleep 1200
done
echo "=== sweep $SWEEP_PID finished $(date '+%F %T'); final report ==="
refresh
echo "=== ALL DONE ==="
